"""Audio retention, size cap and dictation records (no model, no GPU, no mic).

Run:  .venv/Scripts/python.exe tests/test_keep_audio.py   (or pytest tests/)

Everything writes under a fresh temp directory; the import itself is pointed
at a temp LOCALAPPDATA so no test can touch the real Vox state folder.
"""
import datetime
import json
import os
import shutil
import sys
import tempfile
import wave

import numpy as np

_TMP_HOME = tempfile.mkdtemp(prefix="vox-test-home-")
os.environ["LOCALAPPDATA"] = _TMP_HOME
os.environ["XDG_STATE_HOME"] = _TMP_HOME
os.environ.setdefault("VOX_LOG", os.devnull)
for _k in ("VOX_AUDIO_KEEP", "VOX_AUDIO_CAP_GB", "VOX_AUDIO_RECORDS",
           "VOX_TRANSCRIPT_DIR", "VOX_SAVE_AUDIO"):
    os.environ.pop(_k, None)
# Deterministic, side-effect-free pipeline for the end-to-end check below.
os.environ["VOX_LLM"] = "off"
os.environ["VOX_LEARN"] = "0"
os.environ["VOX_SMARTCASE"] = "0"
os.environ["VOX_PREFETCH_COMPARE"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dictation as vx  # noqa: E402

WHEN = datetime.datetime(2026, 11, 12, 21, 54, 7).astimezone()


class _Tmp:
    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="vox-test-")
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


class _Patch:
    """Set module attributes for one test and restore them afterwards."""
    def __init__(self, **attrs):
        self.attrs = attrs

    def __enter__(self):
        self.old = {k: getattr(vx, k) for k in self.attrs}
        for k, v in self.attrs.items():
            setattr(vx, k, v)

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(vx, k, v)


def _touch(path, nbytes=10):
    with open(path, "wb") as f:
        f.write(b"x" * nbytes)


# --- settings ------------------------------------------------------------------
def test_parse_audio_keep():
    assert vx.parse_audio_keep("") == (20, "")
    assert vx.parse_audio_keep(None) == (20, "")
    assert vx.parse_audio_keep("20") == (20, "")
    assert vx.parse_audio_keep("500") == (500, "")
    assert vx.parse_audio_keep(7) == (7, "")
    for keep_all in ("0", 0, "all", "ALL", "keep all", "unlimited", "-1"):
        assert vx.parse_audio_keep(keep_all) == (None, ""), keep_all
    keep, problem = vx.parse_audio_keep("lots")
    assert keep == 20 and "not understood" in problem


def test_parse_cap_gb():
    assert vx.parse_cap_gb("") == (50.0, "")
    assert vx.parse_cap_gb("120") == (120.0, "")
    assert vx.parse_cap_gb("75 GB") == (75.0, "")
    assert vx.parse_cap_gb("0") == (None, "")
    assert vx.parse_cap_gb("off") == (None, "")
    cap, problem = vx.parse_cap_gb("huge")
    assert cap == 50.0 and problem


def test_env_beats_settings_file_and_off_words():
    with _Patch(SETTINGS={"audio_keep": "all", "transcript_dir": "x"}):
        os.environ.pop("VOX_AUDIO_KEEP", None)
        assert vx._setting("VOX_AUDIO_KEEP", "audio_keep", "") == "all"
        assert vx._setting_source("VOX_AUDIO_KEEP", "audio_keep") == "settings.local.json"
        os.environ["VOX_AUDIO_KEEP"] = "20"  # the kill switch
        try:
            assert vx._setting("VOX_AUDIO_KEEP", "audio_keep", "") == "20"
            assert vx._setting_source("VOX_AUDIO_KEEP", "audio_keep") == "VOX_AUDIO_KEEP"
        finally:
            os.environ.pop("VOX_AUDIO_KEEP", None)
    with _Patch(SETTINGS={}):
        assert vx._setting_source("VOX_AUDIO_KEEP", "audio_keep") == "default"


def test_expand_dir():
    assert vx.expand_dir("") == ""
    assert vx.expand_dir(None) == ""
    assert vx.expand_dir("0") == ""
    assert vx.expand_dir("off") == ""
    os.environ["VOX_TEST_BASE"] = os.path.join(_TMP_HOME, "base")
    try:
        got = vx.expand_dir("%VOX_TEST_BASE%/vox/transcripts" if os.name == "nt"
                            else "$VOX_TEST_BASE/vox/transcripts")
        assert got == os.path.normpath(os.path.join(_TMP_HOME, "base", "vox", "transcripts")), got
    finally:
        os.environ.pop("VOX_TEST_BASE", None)


# --- retention -----------------------------------------------------------------
def test_unique_wav_path_never_overwrites():
    with _Tmp() as d:
        p1 = vx.unique_wav_path(d, WHEN)
        assert os.path.basename(p1) == "rec-20261112-215407.wav"
        _touch(p1)
        p2 = vx.unique_wav_path(d, WHEN)
        assert os.path.basename(p2) == "rec-20261112-215407-1.wav"
        _touch(p2)
        assert os.path.basename(vx.unique_wav_path(d, WHEN)) == "rec-20261112-215407-2.wav"


def test_rotate_keep_all_deletes_nothing():
    with _Tmp() as d:
        for i in range(30):
            _touch(os.path.join(d, f"rec-20261101-0000{i:02d}.wav"))
        assert vx.rotate_audio(d, None) == []
        assert vx.rotate_audio(d, 0) == []
        assert len(os.listdir(d)) == 30


def test_rotate_keep_n_removes_oldest_wavs_only():
    with _Tmp() as d:
        names = [f"rec-20261101-0000{i:02d}.wav" for i in range(5)]
        for n in names:
            _touch(os.path.join(d, n), 100)
        _touch(os.path.join(d, "records-2026-11-01.jsonl"), 50)
        removed = vx.rotate_audio(d, 2)
        assert [n for n, _ in removed] == names[:3]
        assert all(s == 100 for _, s in removed)
        assert sorted(os.listdir(d)) == sorted(names[3:] + ["records-2026-11-01.jsonl"])


def test_audio_dir_usage():
    with _Tmp() as d:
        assert vx.audio_dir_usage(os.path.join(d, "missing")) == (0, 0)
        _touch(os.path.join(d, "rec-1.wav"), 1000)
        _touch(os.path.join(d, "rec-2.wav"), 500)
        _touch(os.path.join(d, "records-2026-11-12.jsonl"), 25)
        os.makedirs(os.path.join(d, "sub"))
        assert vx.audio_dir_usage(d) == (2, 1525)


def test_save_debug_audio_keep_all_and_keep_n():
    audio = np.zeros(1600, dtype=np.float32)
    with _Tmp() as d:
        with _Patch(SAVE_AUDIO=d, AUDIO_KEEP=None):
            for i in range(25):
                assert vx.save_debug_audio(audio, WHEN + datetime.timedelta(seconds=i))
            assert len([f for f in os.listdir(d) if f.endswith(".wav")]) == 25
            # same second twice: both kept
            a = vx.save_debug_audio(audio, WHEN)
            assert a and a.endswith("-1.wav")
        with _Patch(SAVE_AUDIO=d, AUDIO_KEEP=3):
            vx.save_debug_audio(audio, WHEN + datetime.timedelta(seconds=99))
            assert len([f for f in os.listdir(d) if f.endswith(".wav")]) == 3
        with wave.open(os.path.join(d, "rec-20261112-215546.wav")) as w:
            assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, 1600)
    with _Patch(SAVE_AUDIO="0"):
        assert vx.save_debug_audio(audio, WHEN) is None


def test_cap_warns_once_and_deletes_nothing():
    lines = []
    with _Tmp() as d:
        _touch(os.path.join(d, "rec-a.wav"), 600)
        with _Patch(log=lines.append, AUDIO_CAP_GB=1e-6,  # 1000 bytes
                    _audio_bytes=600, _audio_cap_warned=False):
            vx._account_audio(300, d)
            assert not lines
            vx._account_audio(300, d)
            vx._account_audio(300, d)
            warn = [m for m in lines if "WARNING" in m]
            assert len(warn) == 1 and "Nothing is deleted" in warn[0], lines
        assert os.listdir(d) == ["rec-a.wav"]
    assert vx.over_cap(10, None) is False
    assert vx.over_cap(51e9, 50.0) is True


def test_startup_report_measures_and_warns():
    lines = []
    with _Tmp() as d:
        _touch(os.path.join(d, "rec-a.wav"), 2000)
        with _Patch(log=lines.append, SAVE_AUDIO=d, AUDIO_KEEP=None,
                    AUDIO_CAP_GB=1e-6, _audio_bytes=None, _audio_cap_warned=False):
            msg = vx.audio_startup_report()
            assert vx._audio_bytes == 2000
            assert msg and "Nothing is deleted" in msg
            assert any("keep every recording" in m and "1 recordings" in m for m in lines), lines
        with _Patch(log=lines.append, SAVE_AUDIO=d, AUDIO_KEEP=None,
                    AUDIO_CAP_GB=50.0, _audio_bytes=None, _audio_cap_warned=False):
            assert vx.audio_startup_report() == ""


# --- records -------------------------------------------------------------------
def test_build_and_append_record_roundtrip():
    with _Tmp() as d:
        wav = os.path.join(d, "rec-20261112-215407.wav")
        rec = vx.build_record(wav, WHEN, status="pasted", raw="hello wurld",
                              final="Hello world.", speech_sec=1.23456,
                              wav_sec=1.33456, mic="Mic", latency_sec=0.5,
                              trailing_space=True, cleanup={"outcome": "off"},
                              config={"model": "m"})
        assert rec["v"] == 1 and rec["id"] == "rec-20261112-215407"
        assert rec["wav"] == "rec-20261112-215407.wav"  # relative link
        assert rec["t"].startswith("2026-11-12T21:54:07") and rec["t"][-6] in "+-"
        assert rec["speech_sec"] == 1.235 and rec["raw"] == "hello wurld"
        assert "asr_first" not in rec and "prefetch" not in rec
        path = vx.records_path(d, WHEN)
        assert os.path.basename(path) == "records-2026-11-12.jsonl"
        n1 = vx.append_record(path, rec)
        n2 = vx.append_record(path, dict(rec, final="Zwei — ü"))
        assert os.path.getsize(path) == n1 + n2
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == 2 and rows[1]["final"] == "Zwei — ü"


def test_record_dictation_links_wav_and_respects_switches():
    with _Tmp() as d:
        wav = os.path.join(d, "rec-20261112-215407.wav")
        _touch(wav)
        kw = dict(status="pasted", raw="r", final="f", speech_sec=1, wav_sec=1.1)
        assert vx.record_dictation(None, WHEN, **kw) is None  # nothing saved
        with _Patch(AUDIO_RECORDS=False):
            assert vx.record_dictation(wav, WHEN, **kw) is None
        assert os.listdir(d) == ["rec-20261112-215407.wav"]
        path = vx.record_dictation(wav, WHEN, **kw)
        with open(path, encoding="utf-8") as f:
            row = json.loads(f.readline())
        assert os.path.exists(os.path.join(d, row["wav"]))
        cfg = row["config"]
        for key in ("model", "device", "compute", "beam", "language", "llm",
                    "llm_style", "hotwords", "dictionary", "vox"):
            assert key in cfg, key
        # Nothing about the application the text went into.
        flat = json.dumps(row).lower()
        for word in ("app", "window", "process", "exe", "title"):
            assert f'"{word}' not in flat, word


def test_stop_and_transcribe_end_to_end_with_stubs():
    """The release path writes a WAV and a record that point at each other,
    with the raw and pasted text - model, mic and paste stubbed out."""
    pasted, md = [], []
    sr = vx.SAMPLE_RATE
    frames = [np.full((sr // 2, 1), 0.01, dtype=np.float32) for _ in range(3)]
    with _Tmp() as d:
        with _Patch(SAVE_AUDIO=d, AUDIO_KEEP=None, AUDIO_RECORDS=True,
                    audio_frames=frames, _prefetch_session=None,
                    transcribe_audio=lambda *a, **k: "um hello there world",
                    type_text=lambda text, *a, **k: pasted.append(text),
                    record_transcript=md.append,
                    _learn_note_paste=lambda *a, **k: None,
                    RELEASE_TAIL_SEC=0.0, log=lambda *a, **k: None):
            vx.stop_and_transcribe()
            vx.transcribe_audio = lambda *a, **k: ""
            vx.stop_and_transcribe()  # a no-speech dictation is recorded too
        wavs = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        recs = [f for f in os.listdir(d) if f.endswith(".jsonl")]
        assert len(wavs) == 2 and len(recs) == 1, os.listdir(d)
        with open(os.path.join(d, recs[0]), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert [r["status"] for r in rows] == ["pasted", "no_speech"]
        assert sorted(r["wav"] for r in rows) == wavs
        r = rows[0]
        assert r["raw"] == "um hello there world"
        assert r["final"] == pasted[0] and md == [pasted[0].strip()]
        assert abs(r["speech_sec"] - 1.5) < 1e-6
        assert r["latency_sec"] >= 0 and r["cleanup"] == {"outcome": "off"}
        assert rows[1]["final"] == "" and "latency_sec" not in rows[1]


if __name__ == "__main__":
    n = 0
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                n += 1
                print("ok", name)
        print(f"{n} checks passed")
    finally:
        shutil.rmtree(_TMP_HOME, ignore_errors=True)
