"""
Vox system-tray icon (Windows).

A small tray presence so a windowless Vox (pythonw / autostart) is visible and
controllable without a console: the icon color + tooltip track live state (idle
/ recording / transcribing / error), and the right-click menu exposes a status
header, a live hotkey picker, a microphone picker (with device rescan for mics
that connect mid-session), a settings-file shortcut, a status/README generator,
a git update-check, and Restart / Quit.

Mirrors hud.py: a real `Tray` when enabled on Windows with pystray+Pillow
available, otherwise a no-op `NullTray`, chosen by `create()`. All the
vox-specific behavior (what a menu item does, what the status says) lives in a
`controller` object passed in by dictation.py; this module only owns the
pystray/Pillow plumbing, so the tray can never break the core dictation loop.

Disable entirely with VOX_TRAY=0.
"""

import os
import threading

IS_WINDOWS = os.name == "nt"

ENABLED = os.environ.get("VOX_TRAY", "1").strip().lower() not in (
    "0", "false", "off", "no",
)

# pystray + Pillow are optional: import lazily and fall back to NullTray if the
# packages (or a display) aren't there, exactly like the HUD tolerates no tk.
try:
    if IS_WINDOWS and ENABLED:
        import pystray
        from PIL import Image, ImageDraw
    _HAVE_DEPS = True
except Exception:  # pragma: no cover - missing optional deps
    _HAVE_DEPS = False


# State -> (disc color, glyph color). The glyph is a small microphone so the
# icon reads as "dictation" at 16px; the disc color carries the live state.
_COLORS = {
    "idle":         ((0x3B, 0x82, 0xF6), (0xFF, 0xFF, 0xFF)),  # blue  - ready
    "recording":    ((0xE5, 0x3E, 0x3E), (0xFF, 0xFF, 0xFF)),  # red   - recording
    "transcribing": ((0xF5, 0xA6, 0x23), (0x20, 0x20, 0x20)),  # amber - working
    "error":        ((0x6B, 0x72, 0x80), (0xFF, 0xFF, 0xFF)),  # gray  - error
}

_STATE_LABEL = {
    "idle": "ready",
    "recording": "recording",
    "transcribing": "transcribing",
    "error": "error",
}


def _make_image(state):
    """Draw a 64x64 RGBA tray icon: a filled disc plus a simple mic glyph."""
    disc, glyph = _COLORS.get(state, _COLORS["idle"])
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill=disc + (255,))
    # Microphone: rounded capsule body, a small stand, and a base line.
    d.rounded_rectangle((26, 16, 38, 38), radius=6, fill=glyph + (255,))
    d.arc((22, 30, 42, 46), start=0, end=180, fill=glyph + (255,), width=3)
    d.line((32, 44, 32, 50), fill=glyph + (255,), width=3)
    d.line((25, 50, 39, 50), fill=glyph + (255,), width=3)
    if state == "error":
        # A red slash so "error" is unmistakable even in grayscale.
        d.line((14, 50, 50, 14), fill=(0xE5, 0x3E, 0x3E, 255), width=5)
    return img


class NullTray:
    """No-op tray: used off-Windows, when disabled, or if pystray is missing.

    Call sites in dictation.py are unconditional, so every method is a safe
    no-op here - the core dictation path never has to know whether a real tray
    is present."""

    def start(self):
        pass

    def recording(self):
        pass

    def busy(self):
        pass

    def idle(self):
        pass

    def error(self):
        pass

    def notify(self, message, title="Vox"):
        pass

    def refresh(self):
        pass

    def stop(self):
        pass


class Tray:
    """Live pystray tray icon driven by a dictation.py controller."""

    def __init__(self, controller, logger=None):
        self._controller = controller
        self._log = logger or (lambda *a, **k: None)
        self._state = "idle"
        self._icon = None
        self._thread = None

    # --- lifecycle ------------------------------------------------------------
    def start(self):
        try:
            images = {s: _make_image(s) for s in _COLORS}
            self._images = images
            self._icon = pystray.Icon(
                "vox",
                icon=images["idle"],
                title=self._title(),
                menu=self._build_menu(),
            )
            self._thread = threading.Thread(
                target=self._icon.run, name="vox-tray", daemon=True
            )
            self._thread.start()
            self._log(">> Tray: started")
        except Exception as e:  # pragma: no cover - defensive
            self._log(f">> Tray: failed to start ({e.__class__.__name__}: {e})")
            self._icon = None

    def stop(self):
        icon = self._icon
        if icon is None:
            return
        try:
            icon.visible = False
        except Exception:
            pass
        try:
            icon.stop()
        except Exception:
            pass

    # --- state (called from dictation threads) --------------------------------
    def recording(self):
        self._set_state("recording")

    def busy(self):
        self._set_state("transcribing")

    def idle(self):
        self._set_state("idle")

    def error(self):
        self._set_state("error")

    def _set_state(self, state):
        self._state = state
        icon = self._icon
        if icon is None:
            return
        try:
            icon.icon = self._images[state]
            icon.title = self._title()
        except Exception as e:  # pragma: no cover - defensive
            self._log(f">> Tray: state update failed ({e.__class__.__name__})")

    def _title(self):
        try:
            hk = self._controller.current_hotkey_label()
        except Exception:
            hk = "?"
        title = f"Vox - {_STATE_LABEL.get(self._state, self._state)}  |  hold {hk}"
        # A dead diagnostic log is invisible by nature (windowless, no stdout),
        # so it rides along in the one piece of UI that is always on screen.
        try:
            kind, _detail = self._controller.log_status()
        except Exception:
            kind = "ok"
        if kind == "failed":
            title += "  |  LOG BROKEN"
        elif kind == "fallback":
            title += "  |  log: fallback file"
        return title

    # --- menu -----------------------------------------------------------------
    def notify(self, message, title="Vox"):
        icon = self._icon
        if icon is None:
            return
        try:
            icon.notify(message, title)
        except Exception as e:  # pragma: no cover - not all shells support it
            self._log(f">> Tray: notify failed ({e.__class__.__name__}: {e})")

    def refresh(self):
        """Re-read dynamic menu text/checkmarks and the tooltip."""
        icon = self._icon
        if icon is None:
            return
        try:
            icon.title = self._title()
            icon.update_menu()
        except Exception:
            pass

    def _build_menu(self):
        c = self._controller
        Item = pystray.MenuItem
        SEP = pystray.Menu.SEPARATOR

        hotkey_items = []
        for key, label in c.hotkey_presets():
            hotkey_items.append(Item(
                label,
                self._guard(lambda i, it, k=key: self._pick_hotkey(k)),
                checked=(lambda it, k=key: c.current_hotkey() == k),
                radio=True,
            ))
        hotkey_items.append(SEP)
        hotkey_items.append(Item(
            "Set custom hotkey (capture)...",
            self._guard(lambda i, it: self._capture()),
        ))

        settings_menu = pystray.Menu(
            Item("Open settings file", self._guard(lambda i, it: c.open_settings())),
        )

        return pystray.Menu(
            Item("Vox dictation", None, enabled=False),
            Item(lambda it: f"  Hotkey:  {c.current_hotkey_label()}", None, enabled=False),
            Item(lambda it: f"  Mic:     {c.current_mic_label()}", None, enabled=False),
            Item(lambda it: f"  Model:   {c.model_line()}", None, enabled=False),
            SEP,
            Item("Hotkey", pystray.Menu(*hotkey_items)),
            Item("Microphone", pystray.Menu(self._mic_items)),
            Item("Settings", settings_menu),
            Item("Status / README", self._guard(lambda i, it: c.show_status())),
            SEP,
            Item("Check for updates", self._guard(lambda i, it: self._check_updates())),
            Item(
                "Update now",
                self._guard(lambda i, it: self._update_now()),
                enabled=(lambda it: c.can_update()),
            ),
            SEP,
            Item("Restart", self._guard(lambda i, it: self._restart())),
            Item("Quit", self._guard(lambda i, it: self._quit())),
        )

    def _mic_items(self):
        """Generate the Microphone submenu fresh on every menu (re)build.

        The device list is live state, not config: mics come and go with
        Bluetooth and USB. pystray re-invokes this generator whenever the menu
        is rebuilt (update_menu / refresh), so after "Rescan devices" the list
        reflects whatever is attached right now."""
        c = self._controller
        Item = pystray.MenuItem
        try:
            options = c.mic_options()
        except Exception as e:  # pragma: no cover - defensive
            self._log(f">> Tray: mic list failed ({e.__class__.__name__}: {e})")
            options = []
        for key, label in options:
            yield Item(
                label,
                self._guard(lambda i, it, k=key: self._pick_mic(k)),
                checked=(lambda it, k=key: c.current_mic() == k),
                radio=True,
            )
        yield pystray.Menu.SEPARATOR
        yield Item(
            "Rescan devices",
            self._guard(lambda i, it: self._rescan_mics()),
        )

    # --- menu handlers --------------------------------------------------------
    def _guard(self, fn):
        """Wrap a menu callback so an exception can never kill the tray thread."""
        def wrapped(icon, item):
            try:
                fn(icon, item)
            except Exception as e:  # pragma: no cover - defensive
                self._log(f">> Tray: menu action failed ({e.__class__.__name__}: {e})")
        return wrapped

    def _pick_hotkey(self, key):
        self._controller.set_hotkey(key)
        self.refresh()

    def _pick_mic(self, key):
        # Runs in a worker: reopening the stream on the new device can take
        # ~1s (WASAPI negotiation) and menu callbacks should return instantly.
        def worker():
            try:
                name = self._controller.set_mic(key)
                self.refresh()
                self.notify(f"Microphone: {name}", "Vox: microphone")
            except Exception as e:  # pragma: no cover - defensive
                self._log(f">> Tray: mic switch failed ({e.__class__.__name__}: {e})")

        threading.Thread(target=worker, name="vox-tray-mic", daemon=True).start()

    def _rescan_mics(self):
        # Worker for the same reason: PortAudio reinit + stream reopen ~1-2s.
        def worker():
            try:
                name = self._controller.rescan_mics()
                self.refresh()
                self.notify(f"Devices rescanned - mic: {name}", "Vox: microphone")
            except Exception as e:  # pragma: no cover - defensive
                self._log(f">> Tray: rescan failed ({e.__class__.__name__}: {e})")

        threading.Thread(target=worker, name="vox-tray-rescan", daemon=True).start()

    def _capture(self):
        # Capture runs in a worker so the menu closes immediately; the user gets
        # a "hold now" nudge, then the detected result (or an Fn-invisible note).
        self.notify(
            "Hold your chord now (e.g. Ctrl+Fn) for about a second, then release.",
            "Vox: capturing hotkey",
        )

        def worker():
            try:
                message, saved = self._controller.capture_custom_hotkey()
            except Exception as e:  # pragma: no cover - defensive
                message, saved = f"Capture failed: {e.__class__.__name__}: {e}", False
            if saved:
                self.refresh()
            self.notify(message, "Vox: hotkey")

        threading.Thread(target=worker, name="vox-tray-capture", daemon=True).start()

    def _check_updates(self):
        title, message = self._controller.check_updates()
        self.refresh()  # "Update now" may have become enabled
        self.notify(message, title)

    def _update_now(self):
        title, message = self._controller.update_now()
        self.refresh()
        self.notify(message, title)

    def _restart(self):
        self.stop()
        self._controller.restart()

    def _quit(self):
        # Remove the icon promptly, then hand off to the controller which stops
        # the listener + audio and exits the process.
        try:
            if self._icon is not None:
                self._icon.visible = False
        except Exception:
            pass
        self._controller.quit()
        self.stop()


def create(controller, logger=None):
    """Return a live Tray on Windows (deps present, not disabled), else NullTray."""
    if IS_WINDOWS and ENABLED and _HAVE_DEPS:
        tray = Tray(controller, logger)
        tray.start()
        return tray
    return NullTray()
