"""
Report for Vox shadow mode: Wispr Flow vs Vox on the same audio.

Pure analysis + HTML, no model code. write_report() reads the shadow
archive (shadow.sqlite, written by shadow.py) and produces report.html next
to it - a single self-contained page whose <audio> tags point at the
archived WAVs by relative path, so it works straight from disk.

What counts as a difference
---------------------------
Wispr's pasted text is the reference (it is what the user kept using; it is
not ground truth - the report also shows where the user later corrected
Wispr). Both texts are split on whitespace; each chunk is compared by its
letters/digits only (case and punctuation ignored), aligned by word-level
edit distance. "Word difference" = (substituted + missing + extra words) /
Wispr's word count - a WER against Wispr. Punctuation/capitalization
differences on otherwise-matching words are counted separately.

Each run of differing words (a "hunk") gets one category, first match wins:
  name      a capitalized mid-sentence word, acronym, or a term from Wispr's
            dictionary/on-screen context on either side
  number    digits or number words involved
  compound  the same letters, split differently ("every one" / "everyone")
  filler    words only Vox has, all fillers (um, uh, like, you know...)
  repeat    words only Vox has that restate their neighbours (a false start)
  dropped   3+ consecutive Wispr words Vox has nothing for (lost speech?)
  small     only function words (a, the, that, to...) - mostly style
  missing   1-2 Wispr words Vox lacks
  extra     other words only Vox has
  misheard  any other substitution
"""

import collections
import datetime as dt
import html
import json
import os
import re
import sqlite3
import statistics

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")
_EDGE_PUNCT = "\"'“”‘’()[]{}.,!?;:…-–—*_"

FILLER_WORDS = {
    "um", "umm", "uh", "uhh", "erm", "er", "ah", "hmm", "mm", "mhm",
    "like", "so", "yeah", "okay", "ok", "well", "basically", "actually",
    "literally", "just", "right", "anyway",
}
FILLER_PHRASES = {
    "you know", "i mean", "kind of", "sort of", "i guess", "you know what i mean",
    "or something", "and stuff", "i think", "let's see",
}
SMALL_WORDS = {
    "a", "an", "the", "and", "or", "but", "nor", "to", "of", "in", "on", "at",
    "for", "that", "this", "these", "those", "it", "its", "it's", "is", "are",
    "was", "were", "be", "been", "as", "with", "by", "from", "then", "than",
    "i", "you", "we", "they", "he", "she", "me", "my", "your", "our",
    "do", "did", "does", "have", "has", "had", "will", "would", "can",
    "could", "should", "also", "too", "very", "really", "there", "here",
    "if", "when", "what", "which", "who", "not", "no", "all", "some", "any",
    "up", "out", "about", "into", "over", "i'm", "i'll", "i've", "i'd",
    "that's", "there's", "let's", "don't",
}
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "first", "second", "third", "half",
    "percent", "dollars",
}

CATEGORIES = [
    # (key, label, one-line meaning)
    ("name", "Names & special terms",
     "a person, product or project name, acronym, or dictionary term"),
    ("misheard", "Misheard words", "Vox heard a different ordinary word"),
    ("dropped", "Missing phrases",
     "3+ of Wispr's words with nothing from Vox - possible lost speech"),
    ("missing", "Missing words", "1-2 of Wispr's words that Vox left out"),
    ("filler", "Fillers Vox kept", "um, uh, like, you know... that Wispr removed"),
    ("repeat", "Repeats / false starts Vox kept",
     "a restarted phrase that Wispr collapsed"),
    ("extra", "Extra words in Vox", "other words only Vox has"),
    ("small", "Small words", "only a/the/that/to... differ - mostly style"),
    ("number", "Numbers", "digits vs words, or a different number"),
    ("compound", "Spacing & compounds", "same letters, split differently"),
]
CAT_LABEL = {k: label for k, label, _ in CATEGORIES}
# Categories that are almost never a transcription ERROR by Vox - style and
# formatting choices. Everything else is counted as "substantive".
STYLE_CATS = {"small", "compound", "filler", "repeat"}


def key(chunk):
    """Comparison key of a whitespace chunk: letters/digits/inner
    apostrophes, lowercased ("Wispr's," -> "wispr's", "100,000" -> "100000")."""
    return "".join(_WORD_RE.findall(chunk.replace("’", "'"))).lower()


def align(a, b):
    """Word-level Levenshtein alignment of key lists a (reference) and b.

    Returns ops [(op, i, j)] with op in eq/sub/del/ins; del = only in a,
    ins = only in b. Ties prefer the diagonal, so a substitution is reported
    as one swap rather than a deletion plus an insertion.
    """
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1,
                         prev[j - 1] + (ai != b[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("eq" if a[i - 1] == b[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


class Side:
    """One text split into display chunks, with the comparable ones indexed."""

    def __init__(self, text):
        self.chunks = (text or "").split()
        self.keys = [key(c) for c in self.chunks]
        self.idx = [i for i, k in enumerate(self.keys) if k]  # comparable chunks

    def initial(self, ci):
        """Is chunk ci at a sentence start?"""
        if ci == 0:
            return True
        prev = self.chunks[ci - 1].rstrip("\"'”’)")
        return prev.endswith((".", "!", "?", ":"))


def _namey(side, ci, vocab):
    w = side.chunks[ci].strip(_EDGE_PUNCT)
    if not w:
        return False
    k = side.keys[ci]
    if w[0].isupper():
        if w == "I" or w.startswith(("I'", "I’")) or k in ("ok", "okay"):
            return False
        if len(w) >= 2 and w.isupper():
            return True  # acronym
        if not side.initial(ci):
            return True
        return k in vocab
    # lowercase: only an unusual dictionary term (mixed case, digits,
    # symbols - e.g. "m/dev", "flyGM") counts
    return k in vocab and not k.isalpha()


def _is_repeat(hkeys, bkeys, j0, j1):
    """Extra words hkeys at b[j0:j1] that restate the words right before or
    right after them ("to be able to be able to")."""
    n = len(hkeys)
    if not n:
        return False
    after = bkeys[j1:j1 + n]
    before = bkeys[max(0, j0 - n):j0]
    return after == hkeys or before == hkeys


def _has_num(keys):
    return any(any(c.isdigit() for c in k) or k in NUMBER_WORDS for k in keys)


def categorize(ref, hyp, rset, hset, vocab):
    """Category for one hunk: ref chunk indices rset (Wispr), hyp hset (Vox)."""
    rk = [ref.keys[i] for i in rset]
    hk = [hyp.keys[i] for i in hset]
    namey = any(_namey(ref, i, vocab) for i in rset) or any(
        _namey(hyp, i, vocab) for i in hset)
    if not hk:  # words only Wispr has
        if len(rk) >= 3:
            return "dropped"  # lost speech outranks what it contained
        if namey:
            return "name"
        if all(k in SMALL_WORDS for k in rk):
            return "small"
        return "missing"
    if not rk:  # words only Vox has
        joined = " ".join(hk)
        if all(k in FILLER_WORDS for k in hk) or joined in FILLER_PHRASES:
            return "filler"
        if _is_repeat(hk, [hyp.keys[i] for i in hyp.idx],
                      hyp.idx.index(hset[0]), hyp.idx.index(hset[-1]) + 1):
            return "repeat"
        if namey:
            return "name"
        if all(k in SMALL_WORDS for k in hk):
            return "small"
        return "extra"
    if namey:
        return "name"
    if "".join(rk) == "".join(hk):
        return "compound"
    if _has_num(rk) and _has_num(hk):
        return "number"
    if all(k in SMALL_WORDS for k in rk + hk):
        return "small"
    return "misheard"


def compare(ref_text, hyp_text, vocab):
    """Align Wispr (ref) against Vox (hyp) and classify every difference.

    Returns dict: n_ref, sub, dele, ins, wer, punct, hunks
    [{cat, ref, hyp}], and per-chunk classes for rendering.
    """
    ref, hyp = Side(ref_text), Side(hyp_text)
    ops = align([ref.keys[i] for i in ref.idx], [hyp.keys[i] for i in hyp.idx])
    rcls = ["eq"] * len(ref.chunks)
    hcls = ["eq"] * len(hyp.chunks)
    rtitle = [""] * len(ref.chunks)
    htitle = [""] * len(hyp.chunks)
    sub = dele = ins = punct = 0
    hunks = []
    cur_r, cur_h = [], []

    def flush():
        if not (cur_r or cur_h):
            return
        cat = categorize(ref, hyp, cur_r, cur_h, vocab)
        rt = " ".join(ref.chunks[i] for i in cur_r)
        ht = " ".join(hyp.chunks[i] for i in cur_h)
        hunks.append({"cat": cat, "ref": rt, "hyp": ht})
        tip = f"{CAT_LABEL[cat]} - Wispr: {rt or '(nothing)'} / Vox: {ht or '(nothing)'}"
        for i in cur_r:
            rcls[i] = "d-" + cat
            rtitle[i] = tip
        for i in cur_h:
            hcls[i] = "d-" + cat
            htitle[i] = tip
        cur_r.clear()
        cur_h.clear()

    for op, i, j in ops:
        ri = ref.idx[i] if i is not None else None
        hj = hyp.idx[j] if j is not None else None
        if op == "eq":
            flush()
            if ref.chunks[ri] != hyp.chunks[hj]:
                punct += 1
                rcls[ri] = hcls[hj] = "pc"
                rtitle[ri] = htitle[hj] = (
                    f"Punctuation/case - Wispr: {ref.chunks[ri]} / Vox: {hyp.chunks[hj]}")
            continue
        if op == "sub":
            sub += 1
            cur_r.append(ri)
            cur_h.append(hj)
        elif op == "del":
            dele += 1
            cur_r.append(ri)
        else:
            ins += 1
            cur_h.append(hj)
    flush()
    n_ref = len(ref.idx)
    errors = sub + dele + ins
    return {
        "n_ref": n_ref, "sub": sub, "dele": dele, "ins": ins,
        "errors": errors,
        "wer": errors / n_ref if n_ref else (0.0 if not hyp.idx else 1.0),
        "punct": punct, "hunks": hunks,
        "ref_html": _render(ref.chunks, rcls, rtitle),
        "hyp_html": _render(hyp.chunks, hcls, htitle),
    }


def _render(chunks, classes, titles):
    out = []
    for c, cls, t in zip(chunks, classes, titles):
        e = html.escape(c)
        if cls == "eq":
            out.append(e)
        else:
            out.append(f'<span class="{cls}" title="{html.escape(t)}">{e}</span>')
    return " ".join(out) if out else '<span class="empty">(nothing)</span>'


# --- Data loading ------------------------------------------------------------------
def _local_time(ts_utc, tz_offset_min):
    try:
        t = dt.datetime.fromisoformat(ts_utc)
    except (TypeError, ValueError):
        return ts_utc or "", ""
    if tz_offset_min is not None:
        t = t - dt.timedelta(minutes=int(tz_offset_min))
        zone = {420: "PT", 480: "PT", 360: "MT", 300: "CT", 240: "ET",
                0: "UTC"}.get(int(tz_offset_min))
        if zone is None:
            h = -int(tz_offset_min) / 60
            zone = f"UTC{h:+g}"
    else:
        t = t.astimezone()
        zone = t.tzname() or ""
    return t.strftime("%Y-%m-%d %H:%M"), zone


def _vox_hotwords(vox_dir):
    try:
        with open(os.path.join(vox_dir, "dictionary.json"), encoding="utf-8") as f:
            data = json.load(f)
        return [h for h in (data.get("hotwords") or []) if isinstance(h, str)]
    except (OSError, ValueError):
        return []


def _context_lists(context_json):
    try:
        ctx = json.loads(context_json or "{}")
    except ValueError:
        return [], []
    if not isinstance(ctx, dict):
        return [], []
    dic = [x for x in ctx.get("dictionary_context") or [] if isinstance(x, str)]
    screen = [x for x in (ctx.get("ax_context") or []) + (ctx.get("ocr_context") or [])
              if isinstance(x, str)]
    return dic, screen


def _user_corrected(meta_json, pasted):
    """Did the user change Wispr's own words afterwards (not just keep typing)?

    Wispr's word_edits trace is one letter per word: M match, S substituted,
    D deleted, I inserted. Trailing I's are the user typing on; an S or D (or
    an I before the last dictated word) means Wispr's text was corrected."""
    try:
        meta = json.loads(meta_json or "{}")
    except ValueError:
        return False
    trace = (meta or {}).get("word_edits") or ""
    core = trace.rstrip("I")
    return any(c in "SD" for c in core) or "I" in core


def load(shadow_dir, variants, config_key_fn):
    conn = sqlite3.connect(os.path.join(shadow_dir, "shadow.sqlite"))
    conn.row_factory = sqlite3.Row
    dictations = [dict(r) for r in conn.execute(
        "SELECT * FROM dictation ORDER BY ts_utc DESC")]
    runs = {}
    configs = {}
    for v in variants:
        k = config_key_fn(v)
        configs[v["id"]] = dict(conn.execute(
            "SELECT * FROM variant_config WHERE variant=? AND config_key=?",
            (v["id"], k)).fetchone() or {"code_sha": None, "config_key": k})
        for r in conn.execute("SELECT * FROM run WHERE variant=? AND config_key=?",
                              (v["id"], k)):
            runs[(r["dictation_id"], v["id"])] = dict(r)
    conn.close()
    return dictations, runs, configs


# --- Report ------------------------------------------------------------------------
MAX_CARDS = 400


def analyze(dictations, runs, variants):
    """Per-(dictation, variant) comparisons + per-variant aggregates."""
    wispr_dict = []
    for d in dictations:  # newest first: the latest dictionary snapshot wins
        dic, _ = _context_lists(d.get("context_json"))
        if dic:
            wispr_dict = dic
            break
    results = {}
    agg = {}
    for v in variants:
        vid = v["id"]
        hot = _vox_hotwords(v["vox_dir"])
        base_vocab = {key(w) for t in wispr_dict + hot for w in t.split()} | {
            key(t) for t in wispr_dict + hot}
        a = {
            "n": 0, "ref_words": 0, "errors": 0, "identical": 0,
            "identical_punct": 0, "big_gap": 0, "punct": 0, "raw_ref": 0,
            "raw_err": 0, "cats": collections.Counter(),
            "cat_words": collections.Counter(), "cat_dicts": collections.defaultdict(set),
            "cat_examples": collections.defaultdict(list),
            "pairs": collections.Counter(), "pair_cat": {}, "pair_when": {},
            "vox_lat": [], "wispr_lat": [], "outcomes": collections.Counter(),
            "failed": 0, "substantive": 0, "hotwords": hot,
        }
        for d in dictations:
            r = runs.get((d["id"], vid))
            if r is None:
                continue
            ref = d.get("wispr_pasted") or d.get("wispr_formatted") or ""
            hyp = r.get("vox_final") or ""
            if not ref.strip() and not hyp.strip():
                continue  # nobody heard anything (a blip)
            if r.get("error"):
                a["failed"] += 1
            _, screen = _context_lists(d.get("context_json"))
            vocab = base_vocab | {key(w) for t in screen for w in t.split()}
            c = compare(ref, hyp, vocab)
            raw = compare(d.get("wispr_asr") or "", r.get("vox_raw") or "", vocab)
            results[(d["id"], vid)] = {"c": c, "raw": raw, "run": r}
            a["n"] += 1
            a["ref_words"] += c["n_ref"]
            a["errors"] += c["errors"]
            a["raw_ref"] += raw["n_ref"]
            a["raw_err"] += raw["errors"]
            a["punct"] += c["punct"]
            a["identical"] += c["errors"] == 0
            a["identical_punct"] += c["errors"] == 0 and c["punct"] == 0
            a["big_gap"] += c["wer"] > 0.2
            if any(h["cat"] not in STYLE_CATS for h in c["hunks"]):
                a["substantive"] += 1
            when, zone = _local_time(d["ts_utc"], d.get("tz_offset_min"))
            for h in c["hunks"]:
                cat = h["cat"]
                a["cats"][cat] += 1
                a["cat_words"][cat] += max(len(h["ref"].split()), len(h["hyp"].split()))
                a["cat_dicts"][cat].add(d["id"])
                if len(a["cat_examples"][cat]) < 4:
                    a["cat_examples"][cat].append((h["ref"], h["hyp"]))
                if cat in ("name", "misheard", "number", "compound", "dropped",
                           "missing"):
                    pk = (_strip(h["ref"]), _strip(h["hyp"]))
                    a["pairs"][pk] += 1
                    a["pair_cat"][pk] = cat
                    a["pair_when"].setdefault(pk, f"{when} {zone}")
            if r.get("release_sec") is not None:
                a["vox_lat"].append(r["release_sec"])
            if d.get("wispr_latency_ms"):
                a["wispr_lat"].append(d["wispr_latency_ms"] / 1000.0)
            a["outcomes"][r.get("llm_outcome") or "off"] += 1
        a["wispr_dict"] = wispr_dict
        agg[vid] = a
    return results, agg


def _strip(s):
    return " ".join(w.strip(_EDGE_PUNCT) for w in s.split()).strip()


def _pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


def _median(xs):
    return statistics.median(xs) if xs else None


def _fmt_s(x):
    return "–" if x is None else f"{x:.2f}s"


def _summary_html(v, a, cfg, total_dictations, paired=None):
    if not a["n"]:
        return (f'<p class="muted">No replays yet for this variant with its current '
                f'code (config {html.escape(str(cfg.get("config_key")))}).</p>')
    agree = 1 - (a["errors"] / a["ref_words"]) if a["ref_words"] else 1.0
    raw_agree = 1 - (a["raw_err"] / a["raw_ref"]) if a["raw_ref"] else 1.0
    vl, wl = _median(a["vox_lat"]), _median(a["wispr_lat"])
    tiles = [
        (_pct(max(0.0, agree)), "word agreement with Wispr",
         f"{a['errors']} differing words out of {a['ref_words']} Wispr words"),
        (f"{a['identical']}/{a['n']}", "dictations with identical words",
         f"{a['identical_punct']} also identical in punctuation and case"),
        (f"{a['substantive']}", "dictations with a real difference",
         "names, misheard or missing words - not just style"),
        (f"{a['cats'].get('dropped', 0)}", "missing phrases",
         "3+ consecutive words Wispr has and Vox does not"),
        (f"{_fmt_s(vl)} / {_fmt_s(wl)}", "median latency, Vox / Wispr",
         "Vox: release to text on this PC; Wispr: its own end-to-end"),
        (_pct(max(0.0, raw_agree)), "raw recognition agreement",
         "Whisper vs Wispr's ASR, before any cleanup"),
    ]
    out = [_takeaway_html(a, agree, raw_agree)]
    if paired:
        out.append(f'<p class="takeaway">{html.escape(paired)}</p>')
    out.append('<div class="tiles">')
    for big, label, sub in tiles:
        out.append(f'<div class="tile"><div class="big">{html.escape(big)}</div>'
                   f'<div class="lab">{html.escape(label)}</div>'
                   f'<div class="sub">{html.escape(sub)}</div></div>')
    out.append("</div>")
    cover = f"{a['n']} of {total_dictations} archived dictations compared"
    if a["failed"]:
        cover += f"; {a['failed']} replay error(s)"
    oc = ", ".join(f"{k}: {n}" for k, n in a["outcomes"].most_common())
    out.append(f'<p class="muted small">{html.escape(cover)}. Code '
               f'<code>{html.escape(str(cfg.get("code_sha") or "?"))}</code>, '
               f'config <code>{html.escape(str(cfg.get("config_key")))}</code>. '
               f'Cleanup pass: {html.escape(oc)}.</p>')
    # Category table
    out.append('<h3>What kind of differences</h3><table class="cats"><thead><tr>'
               '<th>Kind</th><th class="num">Times</th><th class="num">Dictations</th>'
               '<th>Examples (Wispr → Vox)</th></tr></thead><tbody>')
    for k, label, meaning in CATEGORIES:
        n = a["cats"].get(k, 0)
        if not n:
            continue
        ex = "<br>".join(
            f'<span class="w">{html.escape(_strip(r)) or "∅"}</span> → '
            f'<span class="x">{html.escape(_strip(h)) or "∅"}</span>'
            for r, h in a["cat_examples"][k][:3])
        style = ' <span class="tag">style</span>' if k in STYLE_CATS else ""
        out.append(f'<tr><td><span class="sw d-{k}"></span>{html.escape(label)}{style}'
                   f'<div class="muted small">{html.escape(meaning)}</div></td>'
                   f'<td class="num">{n}</td><td class="num">{len(a["cat_dicts"][k])}</td>'
                   f'<td class="ex">{ex}</td></tr>')
    if a["punct"]:
        out.append(f'<tr><td><span class="sw pc"></span>Punctuation &amp; capitalization'
                   f' <span class="tag">style</span><div class="muted small">same word, '
                   f'different punctuation or case (not counted as a difference)</div></td>'
                   f'<td class="num">{a["punct"]}</td><td class="num">–</td><td></td></tr>')
    out.append("</tbody></table>")
    # Recurring
    rec = [(pk, n) for pk, n in a["pairs"].most_common() if n >= 2]
    if rec:
        out.append('<h3>Recurring differences</h3><table class="rec"><thead><tr>'
                   '<th>Wispr</th><th>Vox</th><th class="num">Times</th><th>Kind</th>'
                   '</tr></thead><tbody>')
        for (r, h), n in rec[:30]:
            out.append(f'<tr><td class="w">{html.escape(r) or "∅"}</td>'
                       f'<td class="x">{html.escape(h) or "∅"}</td><td class="num">{n}</td>'
                       f'<td>{html.escape(CAT_LABEL[a["pair_cat"][(r, h)]])}</td></tr>')
        out.append("</tbody></table>")
    # Dictionary suggestions
    hot_keys = {key(h) for h in a["hotwords"]}
    wd_missing = [t for t in a["wispr_dict"] if key(t) not in hot_keys
                  and "@" not in t and len(t) < 40]
    name_swaps = [(r, h, n) for (r, h), n in a["pairs"].most_common()
                  if a["pair_cat"][(r, h)] == "name" and r and h
                  and len(r.split()) <= 3 and len(h.split()) <= 3]
    if wd_missing or name_swaps:
        out.append('<h3>What Vox could learn</h3>')
        if name_swaps:
            out.append('<p class="small">Names Wispr got and Vox did not - candidates '
                       'for <code>dictionary.json</code> (a hotword biases Whisper '
                       'toward the spelling; a correction rewrites a known mishearing). '
                       'Review before adding: a correction applies everywhere.</p><ul class="sugg">')
            for r, h, n in name_swaps[:25]:
                out.append(f'<li><span class="x">{html.escape(h)}</span> → '
                           f'<span class="w">{html.escape(r)}</span>'
                           f'{f" <span class=muted>×{n}</span>" if n > 1 else ""}</li>')
            out.append("</ul>")
        if wd_missing:
            out.append('<p class="small">In Wispr\'s dictionary but not in Vox\'s '
                       f'hotwords ({len(wd_missing)}):</p><p class="chips">'
                       + " ".join(f'<span class="chip">{html.escape(t)}</span>'
                                  for t in wd_missing) + "</p>")
    return "\n".join(out)


def _paired_note(v, variants, results):
    """For a variant with "compare_to": the same comparison on exactly the
    dictations both variants replayed (e.g. Vox on its own recording vs Vox
    on Wispr's recording of the same speech)."""
    other = v.get("compare_to")
    if not other or other not in {x["id"] for x in variants}:
        return None
    ids = {d for (d, vid) in results if vid == v["id"]} & {
        d for (d, vid) in results if vid == other}
    if not ids:
        return (f"No dictation has been replayed by both this and "
                f"'{other}' yet.")
    def agree(vid):
        ref = err = 0
        for d in ids:
            c = results[(d, vid)]["c"]
            ref += c["n_ref"]
            err += c["errors"]
        return 1 - err / ref if ref else 1.0
    label = next(x.get("label") or x["id"] for x in variants if x["id"] == other)
    return (f"On the same {len(ids)} dictation(s): this variant matches Wispr on "
            f"{_pct(agree(v['id']), 1)} of words; {label} on "
            f"{_pct(agree(other), 1)}.")


def _takeaway_html(a, agree, raw_agree):
    """Two sentences a person can act on, computed from the numbers."""
    diff_words = sum(a["cat_words"].values()) or 1
    style_words = sum(n for k, n in a["cat_words"].items() if k in STYLE_CATS)
    subst = [(k, n) for k, n in a["cat_words"].most_common() if k not in STYLE_CATS]
    top = ", ".join(f"{CAT_LABEL[k].lower()} ({n} words)" for k, n in subst[:3])
    parts = [
        f"On the same audio, Vox's raw recognition matches Wispr's on "
        f"{_pct(max(0.0, raw_agree))} of words, and its final text on "
        f"{_pct(max(0.0, agree))}.",
        f"{_pct(style_words / diff_words)} of the differing words are style "
        f"(fillers, restarts, small words, spacing); the rest is "
        f"{top or 'nothing substantive'}.",
    ]
    if a["punct"]:
        parts.append(f"Separately, {a['punct']} matching words differ only in "
                     "punctuation or capitalization - mostly where one side "
                     "ends a sentence and the other does not.")
    return f'<p class="takeaway">{html.escape(" ".join(parts))}</p>'


def _card_html(d, per_variant, variants):
    when, zone = _local_time(d["ts_utc"], d.get("tz_offset_min"))
    dur = d.get("audio_sec") or d.get("duration") or 0
    wl = (d.get("wispr_latency_ms") or 0) / 1000.0
    attrs = [f'data-ts="{html.escape(d["ts_utc"])}"']
    blocks = []
    for v in variants:
        vid = v["id"]
        res = per_variant.get(vid)
        if res is None:
            attrs.append(f'data-wer-{vid}="-1"')
            blocks.append(f'<div class="vb" data-v="{vid}"><p class="muted small">'
                          'Not replayed with this variant\'s current code yet.</p></div>')
            continue
        c, raw, r = res["c"], res["raw"], res["run"]
        attrs.append(f'data-wer-{vid}="{c["wer"]:.4f}"')
        sev = "ok" if c["errors"] == 0 else ("mid" if c["wer"] <= 0.2 else "bad")
        badge = ("same words" if c["errors"] == 0
                 else f'{_pct(c["wer"])} different')
        lat = f'Vox {_fmt_s(r.get("release_sec"))}'
        cleanup = r.get("llm_outcome") or "off"
        extra = []
        if r.get("prefetch_segments"):
            extra.append(f'{r["prefetch_segments"]} prefetch segment(s)')
        if r.get("echo_retry"):
            extra.append("echo retry")
        if cleanup != "off":
            extra.append(f"cleanup: {cleanup}")
        if r.get("error"):
            extra.append(f'ERROR: {r["error"]}')
        blocks.append(
            f'<div class="vb" data-v="{vid}">'
            f'<div class="meta"><span class="badge {sev}">{html.escape(badge)}</span>'
            f'<span class="muted small">{html.escape(lat)} · Wispr {wl:.2f}s'
            f'{" · " + html.escape(" · ".join(extra)) if extra else ""}</span></div>'
            f'<div class="line ref"><span class="who">Wispr</span><div class="txt">{c["ref_html"]}</div></div>'
            f'<div class="line hyp"><span class="who vox">Vox</span><div class="txt">{c["hyp_html"]}</div></div>'
            f'<details><summary>Raw recognition (before cleanup): '
            f'{"same words" if raw["errors"] == 0 else _pct(raw["wer"]) + " different"}</summary>'
            f'<div class="line ref"><span class="who">Wispr ASR</span><div class="txt">{raw["ref_html"]}</div></div>'
            f'<div class="line hyp"><span class="who vox">Whisper</span><div class="txt">{raw["hyp_html"]}</div></div>'
            f'<pre class="log">{html.escape(r.get("log") or "")}</pre></details>'
            f'</div>')
    edited = ""
    if _user_corrected(d.get("wispr_edit_meta"), d.get("wispr_pasted")):
        e = " ".join((d.get("wispr_edited") or "").split())
        edited = (f'<p class="edited small">You corrected Wispr afterwards: '
                  f'<q>{html.escape(e[:400])}</q></p>')
    audio = html.escape(d.get("audio_path") or "")
    return (
        f'<article class="card" {" ".join(attrs)}>'
        f'<header><strong>{html.escape(when)} {html.escape(zone)}</strong>'
        f'<span class="muted small"> · {html.escape(d.get("app") or "?")} · {dur:.1f}s</span></header>'
        f'<audio controls preload="none" src="{audio}"></audio>'
        + "".join(blocks) + edited + "</article>")


CSS = """
:root{--bg:#f7f7f8;--surface:#fff;--text:#1d1f24;--muted:#667085;--line:#e4e6ea;
--accent:#6b5ce7;--w:#0f7b3f;--wbg:#dcf5e5;--x:#b42318;--xbg:#fde4e1;--pc:#b26b00;
--ok:#0f7b3f;--mid:#b26b00;--bad:#b42318;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16181d;--surface:#1e2128;
--text:#e6e7ea;--muted:#9aa0aa;--line:#2c3038;--accent:#7c6cf0;--w:#4ad07f;--wbg:#133322;
--x:#ff7b72;--xbg:#3a1a19;--pc:#e3a94b;--ok:#4ad07f;--mid:#e3a94b;--bad:#ff7b72;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#16181d;--surface:#1e2128;--text:#e6e7ea;--muted:#9aa0aa;--line:#2c3038;
--accent:#7c6cf0;--w:#4ad07f;--wbg:#133322;--x:#ff7b72;--xbg:#3a1a19;--pc:#e3a94b;--ok:#4ad07f;
--mid:#e3a94b;--bad:#ff7b72;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px}h3{font-size:15px;margin:22px 0 8px}
.muted{color:var(--muted)}.small{font-size:13px}code{font-size:12px}
nav.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2}
nav.tabs button{border:1px solid var(--line);background:var(--surface);color:var(--text);border-radius:8px;
padding:6px 12px;font:inherit;font-size:14px;cursor:pointer}
nav.tabs button[aria-pressed="true"]{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.tile .big{font-size:22px;font-weight:600}.tile .lab{font-size:13px}.tile .sub{font-size:12px;color:var(--muted)}
table{width:100%;border-collapse:collapse;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top;font-size:14px}
th{font-size:12px;color:var(--muted);font-weight:600}td.num,th.num{text-align:right;white-space:nowrap}
.tablewrap{overflow-x:auto}
.w{color:var(--w)}.x{color:var(--x)}td.ex{font-size:13px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;background:var(--x)}
.sw.pc{background:var(--pc)}.sw.d-small,.sw.d-compound,.sw.d-filler,.sw.d-repeat{background:var(--muted)}
.tag{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 4px;color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{font-size:12px;border:1px solid var(--line);border-radius:999px;padding:1px 8px}
ul.sugg{columns:2;font-size:14px}@media (max-width:640px){ul.sugg{columns:1}}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;font-size:14px;margin:8px 0 12px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
.card header{margin-bottom:6px}.card audio{width:100%;height:32px;margin:4px 0 8px}
.meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:6px}
.badge{font-size:12px;font-weight:600;border-radius:999px;padding:1px 8px;border:1px solid currentColor}
.badge.ok{color:var(--ok)}.badge.mid{color:var(--mid)}.badge.bad{color:var(--bad)}
.line{display:grid;grid-template-columns:78px 1fr;gap:8px;margin:4px 0}
.who{font-size:12px;font-weight:600;color:var(--w);padding-top:2px}.who.vox{color:var(--x)}
.txt{overflow-wrap:anywhere}
.line .txt span[class^="d-"]{border-radius:3px;padding:0 2px}
.line.ref .txt span[class^="d-"]{background:var(--wbg);text-decoration:underline;text-underline-offset:3px}
.line.hyp .txt span[class^="d-"]{background:var(--xbg);text-decoration:line-through}
.txt span.d-small,.txt span.d-compound,.txt span.d-filler,.txt span.d-repeat{opacity:.75}
.txt span.pc{text-decoration:underline dotted var(--pc);text-underline-offset:3px}
.empty{color:var(--muted);font-style:italic}
details{margin-top:6px}summary{cursor:pointer;font-size:13px;color:var(--muted)}
pre.log{font-size:11px;white-space:pre-wrap;color:var(--muted);max-height:220px;overflow:auto}
.edited{border-left:3px solid var(--pc);padding-left:8px;margin:8px 0 0}
.vb,section.vs{display:none}
footer{margin-top:40px;font-size:13px;color:var(--muted)}
@media (max-width:640px){nav.tabs{position:static}.line{grid-template-columns:52px 1fr}td,th{padding:6px 7px}}
.takeaway{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:8px;padding:10px 12px;margin:0 0 12px}
.legend span{margin-right:12px}
"""

JS = """
(function(){
  var body=document.body, ids=JSON.parse(body.getAttribute('data-variants'));
  function pick(v){
    body.setAttribute('data-v',v);
    document.querySelectorAll('nav.tabs button').forEach(function(b){b.setAttribute('aria-pressed',b.dataset.v===v?'true':'false');});
    try{localStorage.setItem('vox-shadow-variant',v);}catch(e){}
    apply();
  }
  function apply(){
    var v=body.getAttribute('data-v'), only=document.getElementById('only').checked,
        sort=document.getElementById('sort').value, list=document.getElementById('cards');
    var cards=[].slice.call(list.children);
    cards.forEach(function(c){var w=parseFloat(c.getAttribute('data-wer-'+v));
      c.style.display=(only&&!(w>0))?'none':'';});
    cards.sort(function(a,b){
      if(sort==='gap'){return parseFloat(b.getAttribute('data-wer-'+v))-parseFloat(a.getAttribute('data-wer-'+v));}
      return a.dataset.ts<b.dataset.ts?1:-1;});
    cards.forEach(function(c){list.appendChild(c);});
  }
  document.querySelectorAll('nav.tabs button').forEach(function(b){b.addEventListener('click',function(){pick(b.dataset.v);});});
  document.getElementById('only').addEventListener('change',apply);
  document.getElementById('sort').addEventListener('change',apply);
  var saved=null; try{saved=localStorage.getItem('vox-shadow-variant');}catch(e){}
  pick(ids.indexOf(saved)>=0?saved:body.getAttribute('data-v'));
})();
"""


def write_report(shadow_dir, variants, config_key_fn, path=None):
    dictations, runs, configs = load(shadow_dir, variants, config_key_fn)
    results, agg = analyze(dictations, runs, variants)
    now = dt.datetime.now().astimezone()
    stamp = now.strftime("%Y-%m-%d %H:%M ") + ("PT" if now.utcoffset() in (
        dt.timedelta(hours=-7), dt.timedelta(hours=-8)) else now.strftime("%Z"))
    total = len(dictations)
    audio_min = sum((d.get("audio_sec") or 0) for d in dictations) / 60
    labelled_h = sum((d.get("audio_sec") or 0) for d in dictations
                     if (d.get("wispr_pasted") or d.get("wispr_formatted") or "").strip()) / 3600
    target_h = float(os.environ.get("VOX_SHADOW_TRAIN_HOURS", "5"))
    first = _local_time(dictations[-1]["ts_utc"], dictations[-1].get("tz_offset_min")) if dictations else ("", "")
    vids = [v["id"] for v in variants]
    show_css = "".join(
        f'body[data-v="{v}"] .vb[data-v="{v}"],body[data-v="{v}"] section.vs[data-v="{v}"]'
        '{display:block}' for v in vids)

    out = ["<!DOCTYPE html>", '<html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           "<title>Vox vs Wispr Flow</title>",
           '<meta name="description" content="Wispr Flow dictations replayed through Vox, compared word by word">',
           f'<meta name="generator" content="vox shadow.py">',
           f'<meta name="revised" content="{html.escape(stamp)}">',
           f"<style>{CSS}{show_css}</style></head>",
           f"<body data-variants='{html.escape(json.dumps(vids))}' "
           f'data-v="{html.escape(vids[0] if vids else "")}"><main>',
           "<h1>Vox vs Wispr Flow</h1>",
           f'<p class="muted">{total} Wispr Flow dictations ({audio_min:.1f} min of audio, '
           f'since {html.escape(" ".join(first))}) replayed through Vox on the same audio. '
           f'Wispr\'s pasted text is the reference. Updated {html.escape(stamp)}.</p>',
           f'<p class="muted small">Training data for a Whisper fine-tune on your voice: '
           f'{labelled_h:.2f} h of {target_h:g} h '
           f'({min(1.0, labelled_h / target_h if target_h else 1.0):.0%}); '
           f'<code>shadow.py dataset</code> exports it, <code>train/finetune_whisper.py</code> '
           f'trains it.</p>',
           '<nav class="tabs" aria-label="Vox variant">']
    for v in variants:
        out.append(f'<button type="button" data-v="{html.escape(v["id"])}" aria-pressed="false">'
                   f'{html.escape(v.get("label") or v["id"])}</button>')
    out.append("</nav>")
    out.append("<h2>Summary</h2>")
    for v in variants:
        out.append(f'<section class="vs" data-v="{html.escape(v["id"])}">'
                   '<div class="tablewrap">'
                   + _summary_html(v, agg[v["id"]], configs[v["id"]], total,
                                   _paired_note(v, variants, results))
                   + "</div></section>")
    out.append('<h2>Dictations</h2><div class="controls">'
               '<label><input type="checkbox" id="only" checked> only dictations that differ</label>'
               '<label>Sort <select id="sort"><option value="new">newest first</option>'
               '<option value="gap">biggest difference first</option></select></label>'
               '<span class="legend small"><span class="w"><u>underlined</u>: only in Wispr</span>'
               '<span class="x"><s>struck</s>: only in Vox</span>'
               '<span style="color:var(--pc)">dotted: punctuation/case</span></span></div>'
               '<div id="cards">')
    for d in dictations[:MAX_CARDS]:
        per = {v["id"]: results.get((d["id"], v["id"])) for v in variants}
        if not any(per.values()):
            continue
        out.append(_card_html(d, {k: x for k, x in per.items() if x}, variants))
    out.append("</div>")
    if total > MAX_CARDS:
        out.append(f'<p class="muted small">Showing the newest {MAX_CARDS} of {total}; '
                   'the summary covers all of them.</p>')
    cfg_rows = "".join(
        f'<li><strong>{html.escape(v.get("label") or v["id"])}</strong>: '
        f'<code>{html.escape(v["vox_dir"])}</code> @ '
        f'<code>{html.escape(str(configs[v["id"]].get("code_sha")))}</code>, '
        f'env <code>{html.escape(json.dumps(v.get("env") or {}))}</code></li>'
        for v in variants)
    out.append(
        '<footer><h3>How this works</h3>'
        '<p>Wispr Flow stores each dictation\'s audio in its local history database. '
        '<code>shadow.py</code> copies new ones into this folder (read-only on Wispr\'s side), '
        'plays each through Vox\'s real release path (<code>stop_and_transcribe</code>, with '
        'segment prefetch fed at the live 26 ms block / 0.25 s poll cadence, the echo guard, '
        'the optional cleanup pass and the dictionary) with pasting, the HUD and the '
        'transcript file switched off, and compares what Vox would have pasted with what '
        'Wispr pasted.</p>'
        '<p>Limits: Wispr\'s audio is captured by its own (Chromium) audio path, which may '
        'apply noise suppression and gain control that Vox\'s raw capture does not; Vox\'s '
        '0.4 s pre-roll is absent; caret-aware casing is off (no caret in a replay); Vox '
        'latency is measured on this PC while other GPU work may be running. Wispr is the '
        'reference, not the truth: where you corrected Wispr afterwards the card says so.</p>'
        f'<ul>{cfg_rows}</ul>'
        f'<p>Archive: <code>{html.escape(shadow_dir)}</code></p></footer>')
    out.append(f"</main><script>{JS}</script></body></html>")
    path = path or os.path.join(shadow_dir, "report.html")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    os.replace(tmp, path)
    return path
