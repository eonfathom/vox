#!/usr/bin/env python3
"""
Vox shadow capture - record Wispr Flow dictations through Vox's own microphone.

shadow.py replays the audio WISPR recorded. That settles "given the same
sound, which engine hears better", but not "does Vox's own recording lose
something": Wispr records through Chromium's audio stack, Vox through
PortAudio (MME at 16 kHz on Windows), with its own device choice and a 0.4 s
pre-roll. This small process closes that gap. While Wispr's push-to-talk
chord is held, it records the same speech through Vox's capture code, and
nothing else:

  - it imports dictation.py from the live Vox checkout and opens the mic with
    Vox's own _ensure_stream() (same device resolution, same stream settings,
    same callback and pre-roll ring), so the recording IS Vox's recording;
  - it never loads a model, never transcribes, never pastes;
  - Wispr's chord is read from Wispr's own settings (the shortcut mapped to
    "ptt"), polled with GetAsyncKeyState (no keyboard hook);
  - each capture is saved as a WAV + JSON sidecar under <shadow dir>/capture;
    `shadow.py ingest` pairs it with the Wispr dictation that started at the
    same moment, and DELETES captures that match nothing after 10 minutes
    (a Ctrl+Alt shortcut that was not a dictation).

The pairs let shadow.py replay each dictation twice - Wispr's audio and
Vox's audio - through the same Vox, which separates microphone effects
from recognition effects.

Windows only. Run under Task Scheduler at logon (see README, "Shadow mode").
`--self-test` records 2 seconds right away and exits.
"""

import argparse
import ctypes
import json
import os
import sys
import threading
import time
import wave

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = sys.platform == "win32"
POLL_SEC = 0.01
HEARTBEAT_SEC = 60
MIN_CAPTURE_SEC = 0.3
DEFAULT_CHORD = (0xA2, 0xA4)  # left Ctrl + left Alt, Wispr's Windows default


def shadow_dir(arg):
    d = arg or os.environ.get("VOX_SHADOW_DIR") or os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "vox", "shadow")
    os.makedirs(d, exist_ok=True)
    return d


def wispr_chord():
    """VK codes of Wispr Flow's push-to-talk shortcut ("162+164" -> 0xA2, 0xA4)."""
    path = os.path.join(os.environ.get("APPDATA", ""), "Wispr Flow", "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        shortcuts = cfg.get("prefs", {}).get("user", {}).get("shortcuts", {})
        for combo, action in shortcuts.items():
            if action == "ptt":
                codes = tuple(int(c) for c in combo.split("+") if c.strip().isdigit())
                if codes:
                    return codes
    except (OSError, ValueError, AttributeError):
        pass
    return DEFAULT_CHORD


class Capturer:
    def __init__(self, sdir, vox_dir):
        self.sdir = sdir
        self.cap_dir = os.path.join(sdir, "capture")
        os.makedirs(self.cap_dir, exist_ok=True)
        self.log_path = os.path.join(sdir, "logs", "capture.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        # Import Vox's capture code from the live checkout, with every side
        # effect switched off before the import runs.
        os.environ.pop("VOX_TRANSCRIPT_DIR", None)
        os.environ.update({
            "VOX_LOG": os.path.join(sdir, "logs", "capture-vox.log"),
            "VOX_SAVE_AUDIO": "0", "VOX_HUD": "0", "VOX_TRAY": "0",
            "VOX_SMARTCASE": "0",
        })
        sys.path = [vox_dir] + [p for p in sys.path
                                if os.path.abspath(p or ".") not in
                                (SCRIPT_DIR, os.path.abspath(vox_dir))]
        import dictation as vx
        self.vx = vx
        self.release_tail = float(getattr(vx, "RELEASE_TAIL_SEC", 0.12))
        self.preroll = float(getattr(vx, "PREROLL_SEC", 0.4))

    def say(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        try:
            print(line, flush=True)
        except Exception:
            pass

    def open_stream(self):
        self.vx._ensure_stream()
        self.say(f"mic open: {getattr(self.vx, '_mic_name', '?')} "
                 f"(Vox's own capture path, {self.preroll:.1f}s pre-roll)")

    def start(self):
        """Begin a capture exactly as Vox's start_recording() seeds its buffer."""
        self.vx._ensure_stream()  # no-op when healthy; reopens after sleep/unplug
        buf = list(self.vx._preroll)
        pre = sum(len(b) for b in buf) / float(self.vx.SAMPLE_RATE)
        self.vx._active_buf = buf
        return buf, time.time() - pre, pre

    def stop(self, buf, t_start, pre):
        if self.release_tail > 0:
            time.sleep(self.release_tail)
        self.vx._active_buf = None
        frames = list(buf)
        if not frames:
            return None
        import numpy as np
        audio = np.concatenate(frames, axis=0).flatten()
        sec = len(audio) / float(self.vx.SAMPLE_RATE)
        if sec - pre < MIN_CAPTURE_SEC:
            return None
        stamp = time.strftime("%Y-%m", time.gmtime(t_start))
        d = os.path.join(self.cap_dir, stamp)
        os.makedirs(d, exist_ok=True)
        name = f"cap-{int(t_start * 1000)}"
        wav_path = os.path.join(d, name + ".wav")
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(wav_path + ".tmp", "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(self.vx.SAMPLE_RATE))
            w.writeframes(pcm.tobytes())
        os.replace(wav_path + ".tmp", wav_path)
        meta = {
            "start_utc": t_start,              # capture start, pre-roll included
            "key_down_utc": t_start + pre,     # when the chord went down
            "end_utc": time.time(),
            "preroll_sec": pre, "audio_sec": sec,
            "device": getattr(self.vx, "_mic_name", "?"),
            "sample_rate": int(self.vx.SAMPLE_RATE),
            "vox_dir": os.path.dirname(os.path.abspath(self.vx.__file__)),
        }
        with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return wav_path, sec

    def heartbeat(self, state):
        try:
            with open(os.path.join(self.sdir, "capture-heartbeat.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"at": time.time(), **state}, f)
        except OSError:
            pass

    def run(self):
        chord = wispr_chord()
        self.say(f"started (pid {os.getpid()}); Wispr push-to-talk chord VKs "
                 + "+".join(f"0x{c:02X}" for c in chord))
        self.open_stream()
        gaks = ctypes.windll.user32.GetAsyncKeyState
        held = False
        cur = None
        captures = 0
        last_beat = 0.0
        while True:
            now = time.time()
            if now - last_beat > HEARTBEAT_SEC:
                self.heartbeat({"pid": os.getpid(), "captures": captures})
                last_beat = now
            down = all(gaks(vk) & 0x8000 for vk in chord)
            if down and not held:
                held = True
                try:
                    cur = self.start()
                except Exception as e:
                    self.say(f"start failed ({e.__class__.__name__}: {e})")
                    cur = None
            elif held and not down:
                held = False
                if cur is not None:
                    try:
                        res = self.stop(*cur)
                        if res:
                            captures += 1
                            self.say(f"captured {res[1]:.1f}s -> "
                                     f"{os.path.relpath(res[0], self.sdir)}")
                    except Exception as e:
                        self.say(f"save failed ({e.__class__.__name__}: {e})")
                    cur = None
            time.sleep(POLL_SEC)

    def self_test(self, seconds=2.0):
        self.open_stream()
        cur = self.start()
        time.sleep(seconds)
        res = self.stop(*cur)
        self.say(f"self-test: {'saved ' + res[0] if res else 'nothing saved'}")
        return res


def single_instance():
    """Hold a named mutex for the process lifetime; False if another holds it."""
    k32 = ctypes.windll.kernel32
    h = k32.CreateMutexW(None, False, "Local\\VoxShadowCapture")
    if k32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        return None
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dir", help="shadow data folder (same as shadow.py --dir)")
    ap.add_argument("--vox-dir", default=os.environ.get(
        "VOX_CAPTURE_VOX_DIR", os.path.join(os.path.expanduser("~"), "dev", "vox")),
        help="the Vox checkout whose capture code to use (default ~/dev/vox)")
    ap.add_argument("--self-test", action="store_true",
                    help="record 2 s now, save it, exit")
    args = ap.parse_args()
    if not IS_WINDOWS:
        sys.exit("shadow_capture.py is Windows-only")
    sdir = shadow_dir(args.dir)
    cap = Capturer(sdir, os.path.abspath(args.vox_dir))
    if args.self_test:
        sys.exit(0 if cap.self_test() else 1)
    guard = single_instance()
    if guard is None:
        cap.say("another capture process is running; exiting")
        return
    try:
        cap.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        cap.say(f"crashed ({e.__class__.__name__}: {e})")
        raise


if __name__ == "__main__":
    main()
