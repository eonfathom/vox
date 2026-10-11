"""Learn vocabulary from the user's own corrections (Wispr Flow style).

Three parts, each fail-safe - any exception turns that step off for the
dictation at hand and never touches the paste itself:

1. OBSERVE (Windows, UI Automation). After a paste, remember the pasted text
   and the focused text element. About 20 s later, or when the next dictation
   starts (whichever comes first, once), re-read that element, find the pasted
   passage in it and diff it word by word. Only CORRECTION-shaped edits count:
   a 1-2 word substitution whose new form is a near-spelling or sound-alike of
   the old one (Zavier -> Xavier, poles -> polls, Mac Kenzie -> McKenzie), never an
   insertion, deletion or rewrite. Each goes to learned-candidates.jsonl in
   Vox's state dir as {ts, heard, corrected, before, after, app} with only 3
   words of context per side - the text box itself is never stored.

2. PROMOTE. A correction seen twice (the same heard -> corrected pair, or the
   same corrected term reached from two different heard forms) becomes an
   entry of dictionary.learned.json: {term, heard: [...], count, first, last}.

3. APPLY (dictation.py via learned_view + fuzzy_correct). Learned vocabulary
   is kept OUT of Whisper's hotwords by default (VOX_LEARNED_HOTWORDS=N adds
   the top N): any change to the hotword prompt re-rolls Whisper's decoding,
   and on the shadow corpus 44 extra names cost 0.6 points of word agreement
   and 10 learned terms 0.18 points. Instead every learned term joins the
   cleanup model's reference spellings, heard forms that are not real words
   become corrections, and a fuzzy pass after the LLM fixes a near-spelling
   of a learned term - only when exactly one term matches and never on a
   real word or a protected one.
"""
import difflib
import json
import os
import queue
import re
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORDLIST_PATH = os.path.join(HERE, "common_words.txt")

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
_JUNK_RE = re.compile(r"[/@\\:]|\.[A-Za-z0-9]|^www|https?|\d")

# Never vocabulary: function words and other tiny words whose edits are
# grammar, not spelling (a -> the, you -> we'll).
FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "so", "if", "of", "to", "too", "two",
    "in", "on", "at", "by", "for", "with", "from", "as", "is", "are", "was",
    "were", "be", "been", "it", "its", "it's", "i", "me", "my", "we", "us",
    "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "there", "this", "that", "these", "those", "no", "not", "do",
    "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "then", "than", "what", "which", "who", "whom", "how", "when", "where",
    "here", "yes", "yeah", "yep", "ok", "okay", "oh", "um", "uh", "just",
    "like", "also", "very", "really", "into", "onto", "up", "out", "off",
}
_INFLECTIONS = ("'s", "’s", "s", "es", "ed", "d", "ing", "ly", "er", "ers")
# Word endings that make a grammatical variant of the same word: two forms
# differing only in these after a shared stem (improve/improving, easy/
# easily, spouse/spousal) are a grammar edit, not a respelling.
_MORPH_TAILS = {"", "e", "s", "es", "ed", "d", "ing", "ly", "ily", "y", "ies",
                "al", "er", "ers", "or", "ors", "ion", "ions", "ment",
                "ness", "ity", "ive", "ation", "ations", "ate", "ating"}

# --- word helpers ---------------------------------------------------------------
_common = None


def common_words():
    """The real-word list (lowercase): common_words.txt, built from speech
    model vocabularies by tools/build_wordlist.py. Empty if missing - which
    makes is_common() say False, so fuzzy_correct then refuses to run."""
    global _common
    if _common is None:
        words = set()
        try:
            with open(WORDLIST_PATH, encoding="utf-8") as f:
                words = {ln.strip() for ln in f
                         if ln.strip() and not ln.startswith("#")}
        except OSError:
            pass
        _common = words
    return _common


def squash(s):
    """Comparison key: lowercase letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", (s or "").replace("’", "'").lower())


def is_common(word):
    """A real word (or a well-known name), including simple inflections."""
    w = (word or "").replace("’", "'").lower().strip("'-")
    if not w:
        return False
    c = common_words()
    if w in c or squash(w) in c:
        return True
    for suf in _INFLECTIONS:
        if w.endswith(suf) and len(w) - len(suf) >= 3 and w[:-len(suf)] in c:
            return True
    return False


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def phonetic_key(word):
    """Crude sound key: Zavier and Xavier both -> 'safar'; Corin and Corrine
    -> 'karan'. Silent onsets dropped, voiced/unvoiced f-v merged, vowel runs
    folded, doubled letters collapsed."""
    s = re.sub(r"[^a-z]", "", (word or "").lower())
    if not s:
        return ""
    for pat, rep in (("^ts", "s"), ("^ps", "s"), ("^pn", "n"), ("^kn", "n"),
                     ("^gn", "n"), ("^wr", "r"), ("^x", "s"), ("ph", "f"),
                     ("ck", "k"), ("c(?=[eiy])", "s"), ("c", "k"), ("q", "k"),
                     ("x", "ks"), ("z", "s"), ("v", "f"),
                     ("(?<=[aeiouy])w", ""), ("(?<=.{3})e$", "")):
        s = re.sub(pat, rep, s)
    s = re.sub(r"[aeiouy]+", "a", s)
    return re.sub(r"(.)\1+", r"\1", s)


def sounds_alike(a, b):
    ka, kb = phonetic_key(a), phonetic_key(b)
    return bool(ka) and ka == kb and len(re.sub("a", "", ka)) >= 2


# --- 1. correction-shaped edits ------------------------------------------------------
def correction_shaped(old, new):
    """Is old -> new a respelling of the same spoken word(s), as opposed to
    a different word, a grammar fix or a rewrite? Both are 1-2 word strings."""
    a, b = squash(old), squash(new)
    if not a or not b or a == b and old.split() == new.split():
        return False                  # identical, or a casing/punctuation edit
    if _JUNK_RE.search(new) or _JUNK_RE.search(old):
        return False                  # numbers, paths, emails, URLs, files
    ow = [w.lower().strip("'.,") for w in old.split()]
    nw = [w.lower().strip("'.,") for w in new.split()]
    if any(w in FUNCTION_WORDS for w in ow + nw):
        return False
    if a == b:                        # "Pod Cast Hub" -> "PodcastHub"
        return len(nw) < len(ow) and not is_common(b)
    if len(b) < 3 or len(a) < 2:
        return False
    both_common = is_common(old) and is_common(new)
    stem = len(os.path.commonprefix([a, b]))
    if stem >= 3 and a[stem:] in _MORPH_TAILS and b[stem:] in _MORPH_TAILS:
        return False                  # improving -> improve, draws -> drawing
    if both_common and (a.endswith(b) or b.endswith(a)):
        return False                  # rethinking -> thinking
    if both_common and min(len(a), len(b)) < 4:
        return False                  # yan/jan, too/two: homophone grammar
    proper = new[:1].isupper()
    if a.startswith(b) or b.startswith(a):
        # Extension/truncation is grammar (human -> humanity, Eon -> Eons)
        # unless a capitalized name gains or loses one letter (Sara -> Sarah).
        if any(a == b + s or b == a + s for s in ("s", "es", "ed", "d", "ing", "ly")):
            return False
        return proper and abs(len(a) - len(b)) == 1
    d = edit_distance(a, b)
    longest = max(len(a), len(b))
    if d <= 2 and min(len(a), len(b)) >= 3:
        return True                   # Karen -> Caryn, Briony -> Bryony
    if d <= 3 and longest >= 7:
        return True                   # Kovalchik -> Kowalczyk
    return sounds_alike(a, b)         # Zavier -> Xavier


def _words(text):
    return [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(text or "")]


def _locate(pk, ck):
    """Index window [s, e) of ck (document keys) best matching pk (pasted)."""
    sm = difflib.SequenceMatcher(None, pk, ck, autojunk=False)
    i, j, n = sm.find_longest_match(0, len(pk), 0, len(ck))
    if n == 0:
        return None
    base_s = j - i
    base_e = base_s + len(pk)
    best = None
    for ds in range(-2, 3):
        for de in range(-2, 3):
            s, e = max(0, base_s + ds), min(len(ck), base_e + de)
            if e <= s:
                continue
            sm = difflib.SequenceMatcher(None, pk, ck[s:e], autojunk=False)
            m = sum(blk.size for blk in sm.get_matching_blocks())
            # Matched words, minus a half-word penalty per length mismatch:
            # a trailing edited word stays INSIDE the window (dropping it
            # would raise a plain match ratio), surrounding text stays out.
            score = m - 0.5 * abs((e - s) - len(pk))
            if best is None or score > best[0] + 1e-9:
                best = (score, s, e, m)
    if best is None:
        return None
    _, s, e, m = best
    return (2.0 * m / (len(pk) + e - s), s, e)


def diff_corrections(pasted, current, protected=(), blocked=(), min_ratio=0.6):
    """Correction-shaped edits the user made to `pasted`, now part of
    `current` (the text box). Returns [{heard, corrected, before, after}].
    Nothing when the passage can't be found or was rewritten wholesale.
    protected: spellings Vox produces on purpose (hotwords, correction
    results) - an edit AWAY from one is never learned (Aethyr -> ether).
    blocked: forms the dictionary deliberately rewrites (correction keys) -
    never learned as the right spelling, or the two would fight."""
    pw, cw = _words(pasted), _words(current)
    if len(pw) < 3 or not cw:
        return []
    pk = [squash(w) for w, _, _ in pw]
    ck = [squash(w) for w, _, _ in cw]
    loc = _locate(pk, ck)
    if loc is None or loc[0] < min_ratio:
        return []
    _, s, e = loc
    win = ck[s:e]
    ops = difflib.SequenceMatcher(None, pk, win, autojunk=False).get_opcodes()
    changed = sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in ops if t != "equal")
    if changed == 0 or changed > max(4, 0.3 * len(pk)):
        return []                     # untouched, or rewritten rather than fixed
    prot = {squash(p) for p in protected}
    block = {squash(b) for b in blocked}
    out = []
    for tag, i1, i2, j1, j2 in ops:
        if tag != "replace" or not (1 <= i2 - i1 <= 2 and 1 <= j2 - j1 <= 2):
            continue
        heard = " ".join(w for w, _, _ in pw[i1:i2])
        corrected = " ".join(w for w, _, _ in cw[s + j1:s + j2])
        if squash(heard) in prot or any(squash(w) in prot for w in heard.split()):
            continue                  # a deliberate spelling (Aethyr) stays
        if squash(corrected) in block:
            continue
        if not correction_shaped(heard, corrected):
            continue
        out.append({
            "heard": heard, "corrected": corrected,
            "before": " ".join(w for w, _, _ in cw[max(0, s + j1 - 3):s + j1]),
            "after": " ".join(w for w, _, _ in cw[s + j2:s + j2 + 3]),
        })
    return out


# --- 2. promotion ------------------------------------------------------------------
def read_candidates(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    c = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(c, dict) and c.get("heard") and c.get("corrected"):
                    out.append(c)
    except OSError:
        pass
    return out


def append_candidates(path, cands):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_learned(path):
    """dictionary.learned.json -> {"terms": [...], "forgotten": [...]}."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    terms = [t for t in data.get("terms") or []
             if isinstance(t, dict) and isinstance(t.get("term"), str)
             and t["term"].strip()]
    for t in terms:
        t["heard"] = [h for h in t.get("heard") or [] if isinstance(h, str)]
        try:
            t["count"] = int(t.get("count") or 0)
        except (TypeError, ValueError):
            t["count"] = 0
    return {"terms": terms,
            "forgotten": [f for f in data.get("forgotten") or []
                          if isinstance(f, str)]}


def save_learned(path, data):
    data = dict(data)
    data["_comment"] = (
        "Vocabulary Vox learned from your own corrections (machine-local, "
        "git-ignored). term = the right spelling; heard = what Whisper wrote "
        "instead; count = times seen. List: python dictation.py --learned; "
        "remove: python dictation.py --forget <term>.")
    data["terms"] = sorted(data.get("terms") or [],
                           key=lambda t: (-t.get("count", 0), t["term"].lower()))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"_comment": data["_comment"], "terms": data["terms"],
                   "forgotten": data.get("forgotten") or []},
                  f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def promote(cands, learned, protected=(), blocked=(), min_seen=2,
            superseded=None):
    """Fold candidates into `learned` (load_learned shape, edited in place).
    Returns the terms that are NEW this time. protected/blocked as in
    diff_corrections (re-checked here: the dictionary may have changed).

    Conflicts resolve by recency: when you correct X -> Y after having taught
    Y -> X (Keycron vs Keychron), the newer direction wins, the older term is
    dropped (named in `superseded`, a list, if given) and never becomes a
    correction loop."""
    prot = {squash(p) for p in protected}
    block = {squash(b) for b in blocked}
    forgotten = {squash(f) for f in learned.get("forgotten") or []}
    groups = {}
    for c in cands:
        key = squash(c["corrected"])
        if (not key or key in block or key in forgotten
                or squash(c["heard"]) in prot):
            continue
        g = groups.setdefault(key, {"term": c["corrected"], "pairs": {},
                                    "heard": {}, "first": c.get("ts"),
                                    "last": c.get("ts"), "n": 0})
        g["term"] = c["corrected"]            # most recent spelling wins
        hk = squash(c["heard"])
        g["pairs"][hk] = g["pairs"].get(hk, 0) + 1
        g["heard"].setdefault(hk, c["heard"])
        g["n"] += 1
        ts = c.get("ts")
        if ts:
            g["first"] = min(g["first"] or ts, ts)
            g["last"] = max(g["last"] or ts, ts)
    ready = {k: g for k, g in groups.items()
             if max(g["pairs"].values()) >= min_seen or len(g["pairs"]) >= 2}
    for k in list(ready):
        g = ready[k]
        for hk in g["pairs"]:
            o = ready.get(hk)         # another ready term g is the fix FOR
            if o is not None and k in o["pairs"] and (o["last"] or "") > (g["last"] or ""):
                ready.pop(k, None)    # ...and it is the newer direction
                break
    by_key = {squash(t["term"]): t for t in learned["terms"]}
    new = []
    for key, g in ready.items():
        for hk in g["pairs"]:
            if hk == key:
                continue              # a compound join (Pod Cast Hub)
            old = by_key.pop(hk, None)
            if old is not None:
                learned["terms"] = [t for t in learned["terms"] if t is not old]
                if superseded is not None:
                    superseded.append((old["term"], g["term"]))
        for t in learned["terms"]:
            if squash(t["term"]) != key:
                t["heard"] = [h for h in t["heard"] if squash(h) != key]
        t = by_key.get(key)
        if t is None:
            t = {"term": g["term"], "heard": [], "count": 0,
                 "first": g["first"], "last": g["last"], "source": "edits"}
            learned["terms"].append(t)
            by_key[key] = t
            new.append(t["term"])
        for h in g["heard"].values():
            if squash(h) not in {squash(x) for x in t["heard"]}:
                t["heard"].append(h)
        t["count"] = max(t["count"], g["n"])
        if g["last"] and (not t.get("last") or g["last"] > t["last"]):
            t["last"] = g["last"]
    return new


def forget(learned, term):
    key = squash(term)
    before = len(learned["terms"])
    learned["terms"] = [t for t in learned["terms"] if squash(t["term"]) != key]
    if key not in {squash(f) for f in learned["forgotten"]}:
        learned["forgotten"].append(term)
    return before - len(learned["terms"])


# --- 3. applying learned vocabulary -------------------------------------------------
def learned_view(learned, hotwords, corrections, protected=(), n_hotwords=0):
    """What the learned terms contribute, given the base dictionary
    (protected: never a correction source; a base correction key is never a
    learned term - both as in diff_corrections):
      hotwords    - the top n_hotwords by count not already hotwords
      corrections - {heard: term} for heard forms that are NOT real words
                    (and not protected / not already a base correction)
      reference   - every learned term not already a hotword (for the
                    cleanup model's reference-spelling list)
      fuzzy       - single-word learned terms the fuzzy pass may produce
    """
    hot = {h.lower() for h in hotwords}
    base_corr = {k.lower() for k in corrections}
    prot = {squash(p) for p in protected}
    terms = sorted(learned.get("terms") or [],
                   key=lambda t: (-t.get("count", 0), t["term"].lower()))
    block = {squash(k) for k in corrections}
    terms = [t for t in terms if squash(t["term"]) not in block]
    view = {"hotwords": [], "corrections": {}, "reference": [], "fuzzy": [],
            "n": len(terms)}
    for t in terms[:max(0, n_hotwords)]:
        if t["term"].lower() not in hot:
            view["hotwords"].append(t["term"])
    for t in terms:
        term = t["term"]
        if term.lower() not in hot:
            view["reference"].append(term)
        for h in t.get("heard") or []:
            hk = squash(h)
            if (hk and hk != squash(term) and hk not in prot
                    and hk not in {squash(x["term"]) for x in terms}
                    and h.lower() not in base_corr
                    and not all(is_common(w) for w in h.split())):
                view["corrections"][h] = term
        if (" " not in term and len(squash(term)) >= 4
                and not _JUNK_RE.search(term)):
            view["fuzzy"].append(term)
    return view


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")


def fuzzy_correct(text, terms, protected=(), max_dist=2):
    """Replace a near-spelling of a learned term with the term.

    A transcript word is replaced only when ALL hold: it is >= 4 letters, not
    a real word (common_words.txt), not protected, not already a term; and
    exactly one learned term is within edit distance max_dist of it (or sounds
    alike and within 3), sharing its first letter or its sound; a capitalized
    term only replaces a capitalized word. Returns (text, [(old, new), ...])."""
    if not text or not terms or not common_words():
        return text, []
    prot = {squash(p) for p in protected}
    tk = {squash(t): t for t in terms}
    changes = []

    def repl(m):
        w = m.group(0)
        core = w.rstrip("'’-")
        k = squash(core)
        if (len(k) < 4 or k in tk or k in prot or is_common(core)
                or core.endswith(("'s", "’s"))):
            return w
        best, best_d, tie = None, 99, False
        for key, term in tk.items():
            if term[:1].isupper() and not core[:1].isupper():
                continue
            if key.startswith(k) or k.startswith(key):
                continue              # inflection or fragment, not a misspelling
            d = edit_distance(k, key)
            alike = sounds_alike(k, key)
            ok = (d <= max_dist and (k[0] == key[0] or alike)
                  and d < max(len(k), len(key)) / 2) or (alike and d <= 3)
            if not ok:
                continue
            if d < best_d:
                best, best_d, tie = term, d, False
            elif d == best_d and term != best:
                tie = True
        if best is None or tie:
            return w
        changes.append((core, best))
        return best + w[len(core):]

    return _TOKEN_RE.sub(repl, text), changes


# --- the observation loop (Windows UI Automation) ------------------------------------
def foreground_app():
    """Executable name of the foreground window's process ('' if unknown)."""
    try:
        import ctypes
        from ctypes import wintypes
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        hwnd = user32.GetForegroundWindow()
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h = kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFO
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(520)
            n = wintypes.DWORD(520)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
                return os.path.splitext(os.path.basename(buf.value))[0]
        finally:
            kernel32.CloseHandle(h)
    except Exception:
        pass
    return ""


class Watcher:
    """Pastes in, candidates out. All UI Automation work happens on one daemon
    thread fed by a small queue, so a hung target app can stall learning but
    never a dictation: the dictation side only ever does a non-blocking put."""

    def __init__(self, textctx, on_candidates, log=None, delay=20.0,
                 protected=()):
        self._textctx = textctx
        self._on_candidates = on_candidates
        self._log = log or (lambda m: None)
        self._delay = delay
        self.protected = set(protected)
        self.blocked = set()
        self._q = queue.Queue(maxsize=8)
        self._thread = None
        self._pending = None           # the paste being watched (worker only)
        self._gen = 0

    def _put(self, job):
        try:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run,
                                                name="vox-learn", daemon=True)
                self._thread.start()
            self._q.put_nowait(job)
        except Exception:
            pass                       # full (worker hung) or no threads: skip

    def note_paste(self, text):
        """Call right after a paste landed: watch that text box."""
        self._gen += 1
        gen = self._gen
        self._put(("capture", gen, text, foreground_app()))
        t = threading.Timer(self._delay, self._put, args=(("check", gen),))
        t.daemon = True
        t.start()

    def dictation_started(self):
        """Call when a new dictation starts: check the last paste now."""
        self._put(("check", self._gen))

    def _run(self):
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
        while True:
            job = self._q.get()
            try:
                if job[0] == "capture":
                    _, gen, text, app = job
                    if self._pending is not None:
                        self._check(self._pending)   # its 'once' comes first
                    el = self._textctx.focused_text_element()
                    self._pending = (None if el is None else
                                     {"gen": gen, "text": text, "el": el,
                                      "app": app})
                elif job[0] == "check":
                    p = self._pending
                    if p is not None and p["gen"] == job[1]:
                        self._check(p)
            except Exception:
                self._pending = None   # silently off for this dictation

    def _check(self, p):
        self._pending = None
        current = self._textctx.element_text(p["el"])
        if not current:
            return
        cands = diff_corrections(p["text"], current, self.protected,
                                 self.blocked)
        if not cands:
            return
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        for c in cands:
            c["ts"] = ts
            c["app"] = p["app"]
        self._on_candidates(cands)
