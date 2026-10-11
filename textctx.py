"""
Caret text context via Windows UI Automation (Wispr-style insertion awareness).

`read()` returns (before, after): up to a few dozen characters of the focused
control's text on each side of the caret, or None when unavailable. dictation.py
uses it to decide whether a dictation is starting a sentence or continuing one
(smart_case): pasting into the middle of "the plan is | good" must not open
with a capital letter or glue words together.

Only some apps expose a caret through UIA's TextPattern2 (classic edit fields,
Word, browsers' text boxes mostly do; many Electron apps don't unless
accessibility is switched on) - so None is a normal, common answer and the
caller falls back to its own recent-paste memory. Every failure path returns
None; like hud.py/tray.py this module must never break the dictation loop.

The UIA call runs on a worker thread with a hard deadline: a hung target app
can block UIA indefinitely, and a paste must never wait on that.
"""

import os
import threading

IS_WINDOWS = os.name == "nt"

ENABLED = os.environ.get("VOX_SMARTCASE", "1").strip().lower() not in (
    "0", "false", "off", "no",
)

_UIA_TEXT_PATTERN2_ID = 10024  # UIA_TextPattern2Id
_TEXT_UNIT_CHARACTER = 0       # TextUnit_Character
_ENDPOINT_START = 0            # TextPatternRangeEndpoint_Start
_ENDPOINT_END = 1              # TextPatternRangeEndpoint_End

_lock = threading.Lock()
_uia = None          # cached IUIAutomation instance (created once, first use)
_dead = False        # comtypes missing / UIA unavailable: stop trying


def _get_uia():
    """Create (once) and return the IUIAutomation COM object, or None."""
    global _uia, _dead
    if _dead:
        return None
    if _uia is not None:
        return _uia
    try:
        import comtypes
        import comtypes.client
        try:
            comtypes.CoInitialize()
        except OSError:
            pass  # thread already initialized (possibly with another model)
        comtypes.client.GetModule("UIAutomationCore.dll")
        from comtypes.gen.UIAutomationClient import (
            CUIAutomation, IUIAutomation,
        )
        _uia = comtypes.client.CreateObject(
            CUIAutomation, interface=IUIAutomation)
        return _uia
    except Exception:
        _dead = True  # don't pay the failed-import cost on every dictation
        return None


def _selection_range(pattern):
    """The first NON-degenerate selected range, or None (no selection).

    Matters because a dictation typed over a selection REPLACES it: the
    context that will actually surround the pasted text is what lies outside
    the selection's endpoints. The caret sits at one END of a selection, so
    reading around the caret would count the doomed selected words as
    "context" - e.g. selecting "really good" inside "the plan is really good"
    and dictating a replacement would look sentence-final instead of
    mid-sentence.
    """
    try:
        ranges = pattern.GetSelection()
        if ranges is None or ranges.Length < 1:
            return None
        r = ranges.GetElement(0)
        if r is None:
            return None
        if r.CompareEndpoints(_ENDPOINT_START, r, _ENDPOINT_END) == 0:
            return None  # degenerate = just the caret
        return r
    except Exception:
        return None


def _read_now(max_before, max_after):
    """The actual UIA walk: focused element -> TextPattern2 -> caret range."""
    import comtypes
    try:
        comtypes.CoInitialize()  # worker threads each need COM initialized
    except OSError:
        pass
    uia = _get_uia()
    if uia is None:
        return None
    from comtypes.gen.UIAutomationClient import IUIAutomationTextPattern2
    element = uia.GetFocusedElement()
    if element is None:
        return None
    raw = element.GetCurrentPattern(_UIA_TEXT_PATTERN2_ID)
    if raw is None:
        return None
    pattern = raw.QueryInterface(IUIAutomationTextPattern2)
    # With a live selection, read around the selection's OUTER endpoints
    # (that's what will surround the pasted text after it replaces the
    # selection); otherwise read around the caret.
    sel = _selection_range(pattern)
    if sel is not None:
        before_range = sel.Clone()
        before_range.MoveEndpointByRange(_ENDPOINT_END, sel, _ENDPOINT_START)
        after_range = sel.Clone()
        after_range.MoveEndpointByRange(_ENDPOINT_START, sel, _ENDPOINT_END)
    else:
        active, caret = pattern.GetCaretRange()
        if caret is None:
            return None
        before_range = caret.Clone()
        after_range = caret.Clone()
    before_range.MoveEndpointByUnit(
        _ENDPOINT_START, _TEXT_UNIT_CHARACTER, -max_before)
    before = before_range.GetText(-1) or ""
    after_range.MoveEndpointByUnit(
        _ENDPOINT_END, _TEXT_UNIT_CHARACTER, max_after)
    after = after_range.GetText(-1) or ""
    return before, after


def read(max_before=64, max_after=16, timeout=0.25):
    """(text_before_caret, text_after_caret) for the focused control, or None.

    Serialized (UIA COM objects are not meant for concurrent use) and bounded:
    if the target app doesn't answer within `timeout`, give up - the abandoned
    worker finishes (or hangs) harmlessly on its daemon thread.
    """
    if not (IS_WINDOWS and ENABLED) or _dead:
        return None
    result = []

    def worker():
        try:
            with _lock:
                result.append(_read_now(max_before, max_after))
        except Exception:
            result.append(None)

    t = threading.Thread(target=worker, name="vox-caret", daemon=True)
    t.start()
    t.join(timeout)
    return result[0] if result else None


# --- Whole-element access for learn.py (edit observation) ---------------------
# Called ONLY from learn.Watcher's own daemon thread, never from the dictation
# path, and deliberately without _lock: if a target app hangs this call, it
# stalls learning alone, while read() above keeps its own deadline.
_UIA_TEXT_PATTERN_ID = 10014  # UIA_TextPatternId
_UIA_VALUE_PATTERN_ID = 10002  # UIA_ValuePatternId (plain inputs, fallback)


def _pattern(element, pattern_id):
    """The element's pattern object, or None (comtypes hands back a NULL
    pointer, not None, for an unsupported pattern)."""
    raw = element.GetCurrentPattern(pattern_id)
    return raw if raw else None


def focused_text_element():
    """The focused element if it exposes its text (TextPattern, else
    ValuePattern) and is not a password field, else None."""
    if not (IS_WINDOWS and ENABLED) or _dead:
        return None
    try:
        import comtypes
        try:
            comtypes.CoInitialize()
        except OSError:
            pass
        uia = _get_uia()
        if uia is None:
            return None
        el = uia.GetFocusedElement()
        if not el or el.CurrentIsPassword:
            return None
        if (_pattern(el, _UIA_TEXT_PATTERN_ID) is None
                and _pattern(el, _UIA_VALUE_PATTERN_ID) is None):
            return None
        return el
    except Exception:
        return None


def element_text(element, max_chars=100000):
    """The element's current text (up to max_chars), or None."""
    try:
        from comtypes.gen.UIAutomationClient import (
            IUIAutomationTextPattern, IUIAutomationValuePattern)
        raw = _pattern(element, _UIA_TEXT_PATTERN_ID)
        if raw is not None:
            pattern = raw.QueryInterface(IUIAutomationTextPattern)
            return pattern.DocumentRange.GetText(max_chars)
        raw = _pattern(element, _UIA_VALUE_PATTERN_ID)
        if raw is not None:
            value = raw.QueryInterface(IUIAutomationValuePattern).CurrentValue
            return (value or "")[:max_chars]
        return None
    except Exception:
        return None
