"""One-time seed of dictionary.learned.json from Wispr Flow's own learning.

Wispr Flow keeps what it learned from your edits in its local database
(table Dictionary: phrase, frequencyUsed, source = user_edits | manual,
observedSource = the form it had heard). This copies the frequent entries into
Vox's learned vocabulary, so Vox starts where Wispr is instead of from zero.
The database is opened READ-ONLY (immutable); Wispr is never touched.

What is imported: rows with frequencyUsed >= --min-freq (3) from a manual or
user_edits source. What is skipped, and shown: snippets (a phrase with a
replacement), emails, URLs/domains, file names and paths, terms the base
dictionary already has, and observedSource values that are not a
near-spelling of the phrase (Wispr records some unrelated words there).

--history also runs Vox's own edit detector (learn.diff_corrections) over
Wispr's per-dictation edit traces (History: pastedText -> editedText) and
promotes pairs seen twice - the same rule the live observation loop uses.
--before limits that to dictations before a date (e.g. to keep a test set
unseen).

    python tools/import_wispr_dictionary.py [--db backup.sqlite] [--dry-run]
        [--min-freq 3] [--history] [--before 2026-09-16] [--out path]
"""
import argparse
import collections
import glob
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("VOX_LOG", os.devnull)
import learn  # noqa: E402

_EMAIL = re.compile(r"\S+@\S+")
_URLISH = re.compile(r"(https?:|www\.|/|\\|\?|\.[a-z]{2,4}\b)", re.I)


def newest_backup():
    base = os.path.join(os.environ.get("APPDATA", ""), "Wispr Flow")
    found = sorted(glob.glob(os.path.join(base, "backups", "backup-*.sqlite")))
    if found:
        return found[-1]
    live = os.path.join(base, "flow.sqlite")
    return live if os.path.exists(live) else None


def junk_reason(phrase, replacement):
    if replacement:
        return "snippet (has a replacement)"
    if _EMAIL.search(phrase):
        return "email address"
    if _URLISH.search(phrase):
        return "URL, domain, file name or path"
    if len(learn.squash(phrase)) < 2:
        return "too short"
    if any(ch.isdigit() for ch in phrase):
        return "contains digits"
    return None


def canonical_case(conn, phrase):
    """The casing Wispr's pasted/edited text uses most for this phrase."""
    pat = re.compile(r"(?<![\w'])" + re.escape(phrase) + r"(?![\w'])", re.I)
    seen = collections.Counter()
    like = "%" + phrase.replace("%", "") + "%"
    for row in conn.execute(
            "SELECT editedText, pastedText, formattedText FROM History "
            "WHERE formattedText LIKE ? OR editedText LIKE ?", (like, like)):
        txt = next((t for t in row if t), "")
        for m in pat.finditer(txt):
            seen[m.group(0)] += 1
    if not seen:
        return phrase
    best = seen.most_common()
    # A capitalized form that is used at all beats a lowercase majority that
    # only reflects Wispr's own lowercase dictionary spelling.
    cap = [f for f, _ in best if f[:1].isupper() and not f.isupper()]
    return best[0][0] if best[0][0][:1].isupper() or not cap else cap[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=None)
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--before", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import dictation as vx  # the live dictionary rules: protected, blocked
    out_path = a.out or vx.DICT_LEARNED_PATH
    db = a.db or newest_backup()
    if not db:
        sys.exit("No Wispr Flow database found (pass --db).")
    conn = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    print(f"Wispr database (read-only): {db}")
    base_hot, base_corr, protect = vx._load_base_dictionary()
    known = {learn.squash(h) for h in base_hot}

    learned = learn.load_learned(out_path)
    by_key = {learn.squash(t["term"]): t for t in learned["terms"]}
    imported, skipped = [], []
    rows = conn.execute(
        "SELECT phrase, replacement, frequencyUsed, source, observedSource, "
        "createdAt, lastUsed FROM Dictionary WHERE coalesce(isDeleted,0)=0 "
        "ORDER BY frequencyUsed DESC").fetchall()
    for phrase, repl, freq, source, observed, created, last in rows:
        phrase = (phrase or "").strip()
        freq = int(freq or 0)
        if freq < a.min_freq:
            skipped.append((phrase, freq, f"used {freq}x (< {a.min_freq})"))
            continue
        if source not in ("manual", "user_edits"):
            skipped.append((phrase, freq, f"source {source!r}"))
            continue
        why = junk_reason(phrase, repl)
        if why:
            skipped.append((phrase, freq, why))
            continue
        if learn.squash(phrase) in known:
            skipped.append((phrase, freq, "already in dictionary.json/.local"))
            continue
        heard = []
        if observed and observed.strip():
            if learn.correction_shaped(observed.strip(), phrase):
                heard.append(observed.strip())
            else:
                skipped.append((f"{phrase} <- heard {observed!r}", freq,
                                "heard form dropped: not a near-spelling"))
        term = canonical_case(conn, phrase)
        key = learn.squash(term)
        t = by_key.get(key)
        if t is None:
            t = {"term": term, "heard": [], "count": 0,
                 "first": (created or "")[:10], "last": (last or "")[:10],
                 "source": f"wispr:{source}"}
            learned["terms"].append(t)
            by_key[key] = t
        t["heard"] += [h for h in heard if learn.squash(h) not in
                       {learn.squash(x) for x in t["heard"]}]
        t["count"] = max(t["count"], freq)
        imported.append((term, freq, heard, source))

    mined = []
    if a.history:
        q = ("SELECT timestamp, pastedText, editedText, app FROM History "
             "WHERE pastedText IS NOT NULL AND editedText IS NOT NULL "
             "AND editedText != pastedText")
        args = ()
        if a.before:
            q += " AND timestamp < ?"
            args = (a.before,)
        cands = []
        for ts, pasted, edited, app in conn.execute(q, args):
            for c in learn.diff_corrections(pasted, edited, vx.PROTECTED,
                                            vx._BASE_CORRECTION_KEYS):
                c.update(ts=(ts or "")[:19].replace(" ", "T"), app=app or "")
                cands.append(c)
        print(f"\nEdit traces: {len(cands)} correction-shaped edit(s) found"
              + (f" before {a.before}" if a.before else ""))
        for c in cands:
            print(f"  {c['ts'][:10]} {c['app']:<10} {c['heard']!r} -> "
                  f"{c['corrected']!r}")
        mined = learn.promote(cands, learned, vx.PROTECTED,
                              vx._BASE_CORRECTION_KEYS)

    print(f"\nImported {len(imported)} term(s) from the Dictionary table:")
    for term, freq, heard, source in imported:
        print(f"  {term:<22} x{freq:<4} {source:<10} heard: "
              f"{', '.join(heard) or '-'}")
    if a.history:
        print(f"Promoted from edit traces (seen twice): {', '.join(mined) or 'none'}")
    print(f"\nSkipped {len(skipped)}:")
    for phrase, freq, why in skipped:
        print(f"  {phrase!r:<40} x{freq:<4} {why}")
    if a.dry_run:
        print("\n(dry run: nothing written)")
        return
    learn.save_learned(out_path, learned)
    print(f"\nWrote {len(learned['terms'])} learned term(s) -> {out_path}")


if __name__ == "__main__":
    main()
