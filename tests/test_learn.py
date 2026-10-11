"""Checks for learn.py: edit detection, promotion, fuzzy fixing, the import
filter (no model, no GPU, no LLM, no UI Automation).

Run:  .venv/Scripts/python.exe tests/test_learn.py   (or pytest tests/)
"""
import os
import sys
import tempfile

os.environ.setdefault("VOX_LOG", os.devnull)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import learn  # noqa: E402

PROTECTED = {"Aethyr", "Eon", "Rokid", "Claude", "Code"}
BLOCKED = {"ether", "aether"}


def _pairs(pasted, edited):
    return [(c["heard"], c["corrected"])
            for c in learn.diff_corrections(pasted, edited, PROTECTED, BLOCKED)]


def test_correction_shaped_edits_are_found():
    assert _pairs("Can you ask Zavier about the grant today?",
                  "Can you ask Xavier about the grant today?") == [("Zavier", "Xavier")]
    assert _pairs("We ran the poles on the weekend again.",
                  "We ran the polls on the weekend again.") == [("poles", "polls")]
    assert _pairs("I think Mac Kenzie should get the license.",
                  "I think McKenzie should get the license.") == [
        ("Mac Kenzie", "McKenzie")]
    # An edit to the LAST word of the paste (window edge) is still seen.
    assert _pairs("Have another session working on the build and Siobahn.",
                  " Have another session working on the build and Siobhan. ") == [
        ("Siobahn", "Siobhan")]
    assert _pairs("Send it to Briony before the call.",
                  "Send it to Bryony before the call.") == [("Briony", "Bryony")]
    # The passage sits inside a larger text box with other text around it.
    box = ("Earlier notes here.\nThen: I met Karen yesterday at noon and we "
           "talked. More text after.")
    assert _pairs("I met Karen yesterday at noon and we talked.", box.replace(
        "Karen", "Caryn")) == [("Karen", "Caryn")]


def test_non_corrections_are_ignored():
    # different word, rewrite, grammar, insertion, deletion, numbers, paths
    assert _pairs("Please close the window now.", "Please cover the window now.") == []
    assert _pairs("This is about the human condition.",
                  "This is about the humanity condition.") == []
    assert _pairs("I will meet her here at five.", "I will meet her her at five.") == []
    assert _pairs("We should go now and eat.", "We should really go now and eat.") == []
    assert _pairs("We should go now and eat.", "We should go and eat.") == []
    assert _pairs("It costs 200 dollars per seat.", "It costs 300 dollars per seat.") == []
    assert _pairs("Open the dev folder please now.", "Open the m/dev folder please now.") == []
    assert _pairs("The quick brown fox jumps over the lazy dog.",
                  "A completely different sentence was typed in here instead.") == []
    # Editing AWAY from a protected spelling is never learned (Aethyr -> ether).
    assert _pairs("I opened Aethyr to check the page.",
                  "I opened ether to check the page.") == []
    # ...nor TO a form the dictionary deliberately rewrites.
    assert _pairs("I opened Ethyr to check the page.",
                  "I opened ether to check the page.") == []
    # Untouched text: nothing.
    assert _pairs("All good here, nothing changed.", "All good here, nothing changed.") == []


def test_grammar_variants_are_not_respellings():
    for old, new in (("improving", "improve"), ("draws", "drawing"),
                     ("easy", "easily"), ("spouse", "spousal"),
                     ("rethinking", "thinking"), ("Eon", "Eons")):
        assert not learn.correction_shaped(old, new), (old, new)
    for old, new in (("poles", "polls"), ("effecting", "affecting"),
                     ("Zavier", "Xavier"), ("Sara", "Sarah"),
                     ("Kovalchik", "Kowalczyk")):
        assert learn.correction_shaped(old, new), (old, new)
    assert learn.sounds_alike("Zavier", "Xavier")


def test_promotion_after_two_sightings():
    learned = {"terms": [], "forgotten": []}
    c1 = {"heard": "Zavier", "corrected": "Xavier", "ts": "2026-10-10T10:00:00"}
    assert learn.promote([c1], learned) == []
    assert learned["terms"] == []
    c2 = dict(c1, ts="2026-10-10T11:00:00")
    assert learn.promote([c1, c2], learned) == ["Xavier"]
    t = learned["terms"][0]
    assert t["heard"] == ["Zavier"] and t["count"] == 2
    assert t["first"] == "2026-10-10T10:00:00" and t["last"] == "2026-10-10T11:00:00"
    # Promoting again is not "new".
    assert learn.promote([c1, c2], learned) == []
    # Same term from two different heard forms also promotes.
    other = {"terms": [], "forgotten": []}
    assert learn.promote([{"heard": "Zavier", "corrected": "Xavier"},
                          {"heard": "Xaviar", "corrected": "Xavier"}], other) == ["Xavier"]
    assert sorted(other["terms"][0]["heard"]) == ["Xaviar", "Zavier"]
    # A forgotten term is never re-learned; protected heard forms never count.
    learn.forget(other, "xavier")
    assert other["terms"] == []
    assert learn.promote([c1, c2], other) == []
    assert learn.promote([{"heard": "Aethyr", "corrected": "Aether"}] * 2,
                         {"terms": [], "forgotten": []}, protected={"Aethyr"}) == []


def test_newer_correction_direction_wins():
    learned, dropped = {"terms": [], "forgotten": []}, []
    c = [{"heard": "Keychron", "corrected": "Keycron", "ts": "1"},
         {"heard": "Keychron", "corrected": "Keycron", "ts": "2"}]
    assert learn.promote(c, learned, superseded=dropped) == ["Keycron"]
    c += [{"heard": "Keycron", "corrected": "Keychron", "ts": "3"},
          {"heard": "Keycron", "corrected": "Keychron", "ts": "4"}]
    assert learn.promote(c, learned, superseded=dropped) == ["Keychron"]
    assert [(t["term"], t["heard"]) for t in learned["terms"]] == [
        ("Keychron", ["Keycron"])]
    assert dropped == [("Keycron", "Keychron")]
    # Stable afterwards: no flip-flop, and no correction loop in the view.
    assert learn.promote(c, learned) == []
    view = learn.learned_view(learned, [], {}, ())
    assert view["corrections"] == {"Keycron": "Keychron"}


def test_learned_file_roundtrip_and_view():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dictionary.learned.json")
        data = learn.load_learned(path)
        learn.promote([{"heard": "Zavier", "corrected": "Xavier"}] * 3, data)
        learn.promote([{"heard": "Karen", "corrected": "Caryn"}] * 2, data)
        learn.save_learned(path, data)
        back = learn.load_learned(path)
    assert [t["term"] for t in back["terms"]] == ["Xavier", "Caryn"]
    view = learn.learned_view(back, ["Eon"], {"ether": "Aethyr"}, PROTECTED,
                              n_hotwords=1)
    assert view["hotwords"] == ["Xavier"]          # top 1 only
    assert learn.learned_view(back, ["Eon"], {}, PROTECTED)["hotwords"] == []
    assert view["reference"] == ["Xavier", "Caryn"]
    # Zavier is not a real word -> correction; Karen is a real name -> not.
    assert view["corrections"] == {"Zavier": "Xavier"}
    assert view["fuzzy"] == ["Xavier", "Caryn"]


def test_fuzzy_corrector_protects_real_words():
    terms = ["Kowalczyk", "Lyft", "Siobhan", "Robinhood", "Johanna"]
    fix = lambda s: learn.fuzzy_correct(s, terms, PROTECTED)[0]  # noqa: E731
    assert fix("I talked to Kowalczik today.") == "I talked to Kowalczyk today."
    assert fix("Robinhod has the shares.") == "Robinhood has the shares."
    assert fix("Siobahn said hi.") == "Siobhan said hi."
    # Real words stay, however close: lift is English, even next to Lyft.
    assert fix("The lift was slow.") == "The lift was slow."
    assert fix("Is Joanna coming?") == "Is Joanna coming?"
    # A capitalized term never replaces a lowercase word.
    assert fix("the robinhod app") == "the robinhod app"
    # Inflections of a term, and protected spellings, are left alone.
    assert fix("Two Siobhans here.") == "Two Siobhans here."
    assert learn.fuzzy_correct("Aethyr", ["Aethyx"], PROTECTED)[0] == "Aethyr"
    # Ambiguous: two terms equally close -> no change.
    assert learn.fuzzy_correct("Talk to Marto.", ["Marta", "Marco"], ())[0] == \
        "Talk to Marto."


def test_import_filter():
    import import_wispr_dictionary as imp
    assert imp.junk_reason("someone@example.com", None) == "email address"
    assert imp.junk_reason("example.com", None)
    assert imp.junk_reason("OneDrive?team_deck_v11.pptx", None)
    assert imp.junk_reason("m/dev", None)
    assert imp.junk_reason("my email", "someone@example.com")
    assert imp.junk_reason("Robinhood", None) is None
    assert imp.junk_reason("Ada Lovelace", None) is None
    # observedSource must be a near-spelling of the phrase to be kept as heard
    assert learn.correction_shaped("Sara", "Sarah")
    assert learn.correction_shaped("Karen", "Caryn")
    assert not learn.correction_shaped("Leads", "claude")
    assert not learn.correction_shaped("Fireflies", "nectome")


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
            n += 1
    print(f"{n} checks passed")
