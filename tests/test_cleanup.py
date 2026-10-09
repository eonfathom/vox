"""Unit-style checks for the edit-mode cleanup (no model, no GPU, no LLM).

Run:  .venv/Scripts/python.exe tests/test_cleanup.py   (or pytest tests/)
"""
import os
import sys

os.environ.setdefault("VOX_LOG", os.devnull)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dictation as vx  # noqa: E402

RAW = "Okay, now it should not show any Heading 4"


def test_trailing_number_keeps_no_sign():
    # The cleanup model's proposals seen for this dictation; every one must
    # come back as the digit the speaker said.
    for proposed in ("Okay, now it should not show any Heading -4.",
                     "Okay, now it should not show any Heading-4.",
                     "Okay, now it should not show any Heading #4.",
                     "Okay, now it should not show any Heading – 4.",
                     "Okay, now it should not show any Heading 4."):
        got = vx._keep_speaker_words(RAW, vx._rejoin_split_words(RAW, proposed))
        assert got == "Okay, now it should not show any Heading 4.", (proposed, got)


def test_said_signs_and_real_compounds_survive():
    assert vx._keep_speaker_words("It dropped to -4 degrees.",
                                  "It dropped to -4 degrees.") == "It dropped to -4 degrees."
    assert vx._keep_speaker_words("Still only two max touch points",
                                  "Still only two max touchpoints.") == \
        "Still only two max touchpoints."
    assert vx._keep_speaker_words("I want to see it in heading 4 too",
                                  "I want to see it in heading 4, too.") == \
        "I want to see it in heading 4, too."


def test_boundary_edits_pass_the_word_check():
    raw = "I'm running Claude Code in Auto Mode. So it should be able to handle all these things now."
    model = "I'm running Claude Code in Auto Mode, so it should be able to handle all these things now."
    assert vx._keep_speaker_words(raw, model) == model
    raw = "Can you check git now? Because I think I have it up there."
    model = "Can you check git now because I think I have it up there?"
    assert vx._keep_speaker_words(raw, model) == model


def test_edit_prompt_states_boundary_rules_and_shots():
    if vx.LLM_STYLE == "rewrite":
        return
    p = vx._cleanup_system_prompt()
    assert "never a period" in p and "question mark" in p and "Heading 4" in p
    shots = vx._cleanup_fewshot()
    assert len(shots) == 2 * (len(vx._CLEANUP_SHOTS) + len(vx._BOUNDARY_SHOTS))
    # Every boundary pair keeps the speaker's words exactly.
    for user, assistant in vx._BOUNDARY_SHOTS:
        assert vx._keep_speaker_words(user, assistant) == assistant, user


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name)
    print(f"{n} checks passed")
