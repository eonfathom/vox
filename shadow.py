#!/usr/bin/env python3
"""
Vox shadow mode - replay every Wispr Flow dictation through Vox and compare.

Use Wispr Flow as your daily dictation tool, and let Vox "shadow" it: Wispr
Flow (1.6.8xx and later) keeps the audio of each dictation in its local
history database, right next to what it transcribed and pasted. This tool
takes that exact audio, runs it through Vox's real release path, and records
what Vox WOULD have pasted - so the two can be compared on the same speech,
dictation by dictation, without ever dictating twice.

  ingest  copy each new Wispr dictation (audio + Wispr's texts) out of Wispr's
          database into Vox's own archive. Wispr's files are only ever READ.
  replay  run the archived audio through Vox for every configured variant -
          the same stop_and_transcribe() the hotkey calls, including segment
          prefetch, the echo guard and the optional LLM cleanup - with paste,
          HUD, tray and the daily transcript file stubbed out. Each variant
          runs in its own process, importing dictation.py from its own
          checkout, so "the Vox running on this PC" and "latest Vox" can be
          measured side by side.
  report  write report.html next to the archive: Wispr vs Vox word-level
          diffs, what kind of differences they are, recurring misses, and
          dictionary suggestions (see shadow_report.py).
  run     all three - what the scheduled task calls.

Nothing is pasted, typed or written into the vault. Data lives in
%LOCALAPPDATA%\\vox\\shadow (Linux/macOS: ~/.local/state/vox/shadow), never
in the repo: it is personal speech. Override with VOX_SHADOW_DIR.

Variants are listed in <shadow dir>/variants.json (created with one default
entry on first run - "this checkout with the live environment"):

  [{"id": "latest", "label": "Latest Vox", "vox_dir": "C:/Users/m/dev/vox",
    "env": {"VOX_LLM": "remote", "VOX_LLM_URL": "http://host:11434/v1"}}]

Each variant's results are keyed by a hash of its dictation.py +
dictionary.json + env, so after a Vox change the corpus is replayed again
(newest first, --max per run) and the report always shows the current code.
That makes the growing Wispr corpus a regression suite for Vox itself.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.parse
import wave

IS_WINDOWS = sys.platform == "win32"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE = 16000
# Vox's capture callback delivers ~26ms blocks (416 samples at 16kHz, host
# default blocksize); replay feeds the audio in the same shape so segment
# prefetch sees what it would have seen live.
BLOCK = 416
# The prefetch worker polls every 0.25s while the chord is held.
PUMP_EVERY_SEC = 0.25


# --- Locations ------------------------------------------------------------------
def shadow_dir():
    d = os.environ.get("VOX_SHADOW_DIR")
    if not d:
        if IS_WINDOWS:
            base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        else:
            base = (os.environ.get("XDG_STATE_HOME")
                    or os.path.expanduser("~/.local/state"))
        d = os.path.join(base, "vox", "shadow")
    os.makedirs(d, exist_ok=True)
    return d


def wispr_db_path():
    """Wispr Flow's local history database (VOX_WISPR_DB overrides)."""
    p = os.environ.get("VOX_WISPR_DB")
    if p:
        return p
    if IS_WINDOWS:
        return os.path.join(os.environ.get("APPDATA", ""), "Wispr Flow",
                            "flow.sqlite")
    return os.path.expanduser(
        "~/Library/Application Support/Wispr Flow/flow.sqlite")


def db_path():
    return os.path.join(shadow_dir(), "shadow.sqlite")


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def say(msg):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass  # pythonw: no console
    try:
        with open(os.path.join(shadow_dir(), "shadow.log"), "a",
                  encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# --- Shadow database ------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS dictation (
    id TEXT PRIMARY KEY,            -- Wispr transcriptEntityId
    ts_utc TEXT NOT NULL,           -- ISO-8601 UTC
    tz_offset_min INTEGER,          -- Wispr's timezoneOffsetMinutes (420 = UTC-7)
    app TEXT, url TEXT,
    duration REAL, speech_duration REAL, num_words INTEGER,
    wispr_status TEXT,
    wispr_asr TEXT,                 -- Wispr's raw ASR
    wispr_formatted TEXT,           -- after Wispr's AI formatting
    wispr_pasted TEXT,              -- what Wispr actually inserted
    wispr_edited TEXT,              -- the text box as Wispr later observed it
    wispr_edit_meta TEXT,           -- Wispr's per-word edit trace (JSON)
    wispr_latency_ms REAL,
    wispr_app_version TEXT, wispr_asr_model TEXT, mic TEXT,
    context_json TEXT,              -- Wispr's additionalContext (on-screen names, dictionary)
    audio_path TEXT,                -- relative to the shadow dir
    audio_sec REAL,
    ingested_at TEXT
);
CREATE TABLE IF NOT EXISTS run (
    dictation_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    config_key TEXT NOT NULL,       -- hash of dictation.py + dictionary.json + env
    code_sha TEXT, model TEXT, llm TEXT,
    vox_raw TEXT,                   -- Whisper's transcript before post-processing
    vox_final TEXT,                 -- what Vox would have pasted
    release_sec REAL,               -- release -> text (tail decode + cleanup)
    decode_sec REAL, prefetch_sec REAL, llm_sec REAL,
    prefetch_segments INTEGER, echo_retry INTEGER, llm_outcome TEXT,
    log TEXT, error TEXT, ran_at TEXT,
    PRIMARY KEY (dictation_id, variant, config_key)
);
CREATE TABLE IF NOT EXISTS variant_config (
    variant TEXT NOT NULL, config_key TEXT NOT NULL,
    code_sha TEXT, vox_dir TEXT, env_json TEXT, first_seen TEXT,
    PRIMARY KEY (variant, config_key)
);
"""


def open_db():
    conn = sqlite3.connect(db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# --- Ingest -----------------------------------------------------------------------
_WISPR_COLUMNS = [
    "transcriptEntityId", "timestamp", "timezoneOffsetMinutes", "app", "url",
    "duration", "speechDuration", "numWords", "status", "asrText",
    "formattedText", "pastedText", "editedText", "userEditMetaData",
    "e2eLatency", "appVersion", "transcriptOrigin", "micDevice",
    "additionalContext",
]


def _open_wispr():
    """Read-only connection to Wispr's live database.

    mode=ro never writes Wispr's database; WAL mode lets this reader run
    alongside Wispr's own writes without blocking them. Reads are short
    (ids first, then one audio blob at a time), so a Wispr checkpoint is
    never held up for long. No snapshot copy: copying a 200+ MB file every
    run would cost far more than it protects.
    """
    path = wispr_db_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Wispr Flow database not found at {path}")
    uri = "file:" + urllib.parse.quote(path.replace("\\", "/"), safe="/:") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_wispr_ts(s):
    """'2026-09-24 21:55:07.169 +00:00' -> aware UTC datetime."""
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z",
                "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return dt.datetime.strptime(s.replace("Z", "+0000"), fmt
                                        ).astimezone(dt.timezone.utc)
        except ValueError:
            continue
    return None


def ingest(quiet=False):
    """Copy new Wispr dictations (those that kept audio) into the archive."""
    conn = open_db()
    known = {r[0] for r in conn.execute("SELECT id FROM dictation")}
    src = _open_wispr()
    try:
        # Wispr's schema grows with every release: take only the columns this
        # version actually has.
        have = {r["name"] for r in src.execute("PRAGMA table_info(History)")}
        cols = [c for c in _WISPR_COLUMNS if c in have]
        if "audio" not in have:
            say("ingest: this Wispr Flow version keeps no audio column; "
                "nothing to shadow")
            return 0
        candidates = [
            (r[0], r[1]) for r in src.execute(
                "SELECT transcriptEntityId, timestamp FROM History "
                "WHERE audio IS NOT NULL AND length(audio) > 44 "
                "ORDER BY timestamp")
            if r[0] not in known
        ]
        added = 0
        for tid, _ts in candidates:
            row = src.execute(
                f"SELECT {', '.join(cols)}, audio FROM History "
                "WHERE transcriptEntityId = ?", (tid,)).fetchone()
            if row is None:
                continue
            rec = {c: row[c] for c in cols}
            when = _parse_wispr_ts(rec.get("timestamp"))
            if when is None:
                continue
            rel = os.path.join("audio", when.strftime("%Y-%m"), f"{tid}.wav")
            dest = os.path.join(shadow_dir(), rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            blob = row["audio"]
            tmp = dest + ".tmp"
            with open(tmp, "wb") as f:
                f.write(blob)
            os.replace(tmp, dest)
            try:
                with wave.open(dest, "rb") as w:
                    audio_sec = w.getnframes() / float(w.getframerate())
            except (wave.Error, EOFError, OSError):
                audio_sec = None
            conn.execute(
                "INSERT OR IGNORE INTO dictation VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    tid, when.isoformat(timespec="milliseconds"),
                    rec.get("timezoneOffsetMinutes"), rec.get("app"),
                    rec.get("url"), rec.get("duration"),
                    rec.get("speechDuration"), rec.get("numWords"),
                    rec.get("status"), rec.get("asrText"),
                    rec.get("formattedText"), rec.get("pastedText"),
                    rec.get("editedText"), rec.get("userEditMetaData"),
                    rec.get("e2eLatency"), rec.get("appVersion"),
                    rec.get("transcriptOrigin"), rec.get("micDevice"),
                    rec.get("additionalContext"), rel.replace("\\", "/"),
                    audio_sec, now_iso(),
                ),
            )
            conn.commit()
            added += 1
        _refresh_recent(src, conn, cols)
        if added or not quiet:
            say(f"ingest: {added} new Wispr dictation(s) archived "
                f"({len(known) + added} total)")
        return added
    finally:
        src.close()
        conn.close()


_REFRESH = [  # Wispr column -> archive column, for fields Wispr fills in later
    ("status", "wispr_status"), ("asrText", "wispr_asr"),
    ("formattedText", "wispr_formatted"), ("pastedText", "wispr_pasted"),
    ("editedText", "wispr_edited"), ("userEditMetaData", "wispr_edit_meta"),
    ("e2eLatency", "wispr_latency_ms"),
]


def _refresh_recent(src, conn, cols, days=2):
    """Re-read Wispr's text fields for the last `days` of archived dictations.

    Wispr keeps working on a History row after the audio lands: formatting
    finishes a moment later, and the text box is re-read after the user edits
    it (editedText). A dictation archived mid-way would otherwise keep a stale
    reference forever."""
    pairs = [(w, a) for w, a in _REFRESH if w in cols]
    if not pairs:
        return
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
    recent = [r[0] for r in conn.execute(
        "SELECT id FROM dictation WHERE ts_utc >= ?", (since,))]
    for tid in recent:
        row = src.execute(
            f"SELECT {', '.join(w for w, _ in pairs)} FROM History "
            "WHERE transcriptEntityId = ?", (tid,)).fetchone()
        if row is None:
            continue
        conn.execute(
            f"UPDATE dictation SET {', '.join(f'{a} = ?' for _, a in pairs)} "
            "WHERE id = ?", [row[w] for w, _ in pairs] + [tid])
    conn.commit()


# --- Variants ----------------------------------------------------------------------
def variants_path():
    return os.path.join(shadow_dir(), "variants.json")


def load_variants():
    """variants.json in the shadow dir; created with one default entry."""
    path = variants_path()
    if not os.path.exists(path):
        default = [{
            "id": "vox",
            "label": "Vox (this checkout, live settings)",
            "vox_dir": SCRIPT_DIR,
            "env": {},
        }]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)
            f.write("\n")
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    out = []
    for v in items:
        if not v.get("id") or v.get("disabled"):
            continue
        v.setdefault("label", v["id"])
        v.setdefault("vox_dir", SCRIPT_DIR)
        v.setdefault("env", {})
        out.append(v)
    return out


def _git_sha(vox_dir):
    try:
        r = subprocess.run(["git", "-C", vox_dir, "rev-parse", "--short=7",
                            "HEAD"], capture_output=True, text=True,
                           timeout=10)
        sha = r.stdout.strip()
        if not sha:
            return None
        dirty = subprocess.run(
            ["git", "-C", vox_dir, "status", "--porcelain", "--",
             "dictation.py", "dictionary.json"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return None


def config_key(variant):
    """Identity of what a replay depends on: code, dictionary, environment."""
    h = hashlib.sha1()
    for name in ("dictation.py", "dictionary.json"):
        p = os.path.join(variant["vox_dir"], name)
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(b"<missing>")
    h.update(json.dumps(variant.get("env", {}), sort_keys=True).encode())
    return h.hexdigest()[:12]


def _venv_python(vox_dir):
    """Interpreter for a variant: its own venv if it has one, else ours."""
    for sub in ((".venv", "Scripts", "python.exe"), (".venv", "bin", "python"),
                ("venv", "Scripts", "python.exe"), ("venv", "bin", "python")):
        p = os.path.join(vox_dir, *sub)
        if os.path.exists(p):
            return p
    exe = sys.executable
    if IS_WINDOWS and exe.lower().endswith("pythonw.exe"):
        exe = exe[:-len("pythonw.exe")] + "python.exe"
    return exe


# --- Replay (orchestrator side) --------------------------------------------------
def pending_ids(conn, variant, key, limit=None, ids=None):
    """Dictations with no result yet for this variant's current config,
    newest first (fresh dictations matter most; backfill follows)."""
    q = ("SELECT d.id FROM dictation d WHERE NOT EXISTS ("
         " SELECT 1 FROM run r WHERE r.dictation_id = d.id"
         " AND r.variant = ? AND r.config_key = ?)")
    args = [variant["id"], key]
    if ids:
        q += f" AND d.id IN ({','.join('?' * len(ids))})"
        args += list(ids)
    q += " ORDER BY d.ts_utc DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    return [r[0] for r in conn.execute(q, args)]


def replay(max_per_variant=None, only_variant=None, ids=None):
    variants = load_variants()
    conn = open_db()
    total = 0
    try:
        for v in variants:
            if only_variant and v["id"] != only_variant:
                continue
            key = config_key(v)
            sha = _git_sha(v["vox_dir"])
            conn.execute(
                "INSERT OR IGNORE INTO variant_config VALUES (?,?,?,?,?,?)",
                (v["id"], key, sha, v["vox_dir"],
                 json.dumps(v.get("env", {}), sort_keys=True), now_iso()))
            conn.commit()
            todo = pending_ids(conn, v, key, max_per_variant, ids)
            if not todo:
                continue
            say(f"replay[{v['id']}]: {len(todo)} dictation(s) "
                f"(code {sha}, config {key})")
            job = {"variant": v, "config_key": key, "code_sha": sha,
                   "ids": todo, "shadow_dir": shadow_dir()}
            job_path = os.path.join(shadow_dir(), f"job-{v['id']}.json")
            with open(job_path, "w", encoding="utf-8") as f:
                json.dump(job, f)
            env = _worker_env(v)
            cmd = [_venv_python(v["vox_dir"]), os.path.abspath(__file__),
                   "_worker", job_path]
            flags = 0
            if IS_WINDOWS:
                # No console window; below-normal CPU priority so a replay
                # never competes with a live dictation for the CPU.
                flags = 0x08000000 | 0x00004000  # CREATE_NO_WINDOW | BELOW_NORMAL
            t0 = time.monotonic()
            r = subprocess.run(cmd, env=env, cwd=v["vox_dir"],
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               creationflags=flags)
            for line in (r.stdout or "").splitlines():
                if line.startswith("[shadow]"):
                    say(f"  {line[9:]}")
            if r.returncode != 0:
                tail = "\n".join((r.stderr or "").strip().splitlines()[-15:])
                say(f"replay[{v['id']}]: worker FAILED (exit {r.returncode}):"
                    f"\n{tail}")
            done = conn.execute(
                "SELECT count(*) FROM run WHERE variant=? AND config_key=?",
                (v["id"], key)).fetchone()[0]
            say(f"replay[{v['id']}]: finished in "
                f"{time.monotonic() - t0:.0f}s ({done} replayed with this "
                "config so far)")
            total += len(todo)
    finally:
        conn.close()
    return total


def _worker_env(variant):
    """Environment for a replay worker: the live environment (what the running
    Vox inherits), minus every side effect, plus the variant's overrides."""
    env = dict(os.environ)
    # Never write the daily transcript file (it is the user's record of what
    # was ACTUALLY dictated) - removed outright rather than set empty.
    env.pop("VOX_TRANSCRIPT_DIR", None)
    logs = os.path.join(shadow_dir(), "logs")
    os.makedirs(logs, exist_ok=True)
    env.update({
        "VOX_LOG": os.path.join(logs, f"{variant['id']}.log"),
        "VOX_SAVE_AUDIO": "0",        # the archive already has the audio
        "VOX_PREFETCH_COMPARE": "0",  # no extra background decode
        "VOX_HUD": "0", "VOX_TRAY": "0",
        "VOX_SMARTCASE": "0",         # no caret to read in a replay
        "VOX_RELEASE_TAIL": "0",      # the audio is already complete
        "PYTHONIOENCODING": "utf-8",
    })
    for k, val in (variant.get("env") or {}).items():
        if val is None:
            env.pop(k, None)
        else:
            env[k] = str(val)
    env.pop("VOX_TRANSCRIPT_DIR", None)
    return env


# --- Replay (worker side: runs inside the variant's own code) ---------------------
class _Null:
    """Stands in for the HUD and tray: every method is a silent no-op."""

    def __getattr__(self, name):
        return lambda *a, **k: None


def _read_wav(path):
    import numpy as np
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"unsupported sample width {w.getsampwidth()}")
        rate, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    audio = pcm.astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if rate != SAMPLE_RATE:
        # Linear resample; Wispr records 16 kHz mono, so this is a guard only.
        n = int(round(len(audio) * SAMPLE_RATE / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def _worker(job_path):
    """Replay a batch inside the variant's checkout (cwd = its vox_dir)."""
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    variant, sdir = job["variant"], job["shadow_dir"]
    vox_dir = os.path.abspath(variant["vox_dir"])
    # Import dictation.py (and its hud/tray/textctx siblings) from the
    # variant's checkout, never from wherever this script happens to live.
    sys.path = [vox_dir] + [p for p in sys.path
                            if os.path.abspath(p or ".") not in
                            (SCRIPT_DIR, vox_dir)]
    import numpy as np
    import dictation as vx

    def emit(msg):
        print(f"[shadow] {msg}", flush=True)

    # Side effects off (belt and braces on top of the environment).
    vx.hud = _Null()
    if hasattr(vx, "tray"):
        vx.tray = _Null()
    vx.record_transcript = lambda *a, **k: None
    vx.save_debug_audio = lambda *a, **k: None
    if hasattr(vx, "_note_paste"):
        vx._note_paste = lambda *a, **k: None
    if hasattr(vx, "smart_case"):
        vx.smart_case = lambda text: (text, True)
    vx.RELEASE_TAIL_SEC = 0.0
    vx.PREFETCH_COMPARE = False
    vx.target_window = None

    pasted = {}

    def fake_type_text(text, *a, **k):
        pasted["text"] = text
    vx.type_text = fake_type_text

    lines = []
    orig_log = vx.log

    def cap_log(msg, *a, **k):
        lines.append(str(msg))
        return orig_log(msg, *a, **k)
    vx.log = cap_log

    timers = {"prefetch": 0.0, "release": 0.0, "llm": 0.0}
    phase = {"now": "prefetch"}
    orig_tx = vx.transcribe_audio

    def timed_tx(*a, **k):
        t = time.monotonic()
        try:
            return orig_tx(*a, **k)
        finally:
            timers[phase["now"]] += time.monotonic() - t
    vx.transcribe_audio = timed_tx
    orig_llm = vx.llm_format

    def timed_llm(*a, **k):
        t = time.monotonic()
        try:
            return orig_llm(*a, **k)
        finally:
            timers["llm"] += time.monotonic() - t
    vx.llm_format = timed_llm

    vx.load_model()
    model = f"{vx.MODEL_SIZE} ({vx.DEVICE}, {vx.COMPUTE_TYPE})"
    chain = list(getattr(vx, "LLM_CHAIN", None) or (
        [getattr(vx, "LLM_BACKEND", "off")]
        if getattr(vx, "LLM_BACKEND", "off") != "off" else []))
    llm_desc = "off"
    if chain:
        parts = []
        for i, b in enumerate(chain):
            m = (vx._backend_model(i, b) if hasattr(vx, "_backend_model")
                 else getattr(vx, "LLM_MODEL", "?"))
            host = (vx._backend_host_note(i, b)
                    if hasattr(vx, "_backend_host_note") else "")
            parts.append(f"{b}:{m}{host}")
        llm_desc = " -> ".join(parts)
        _warm_llm_sync(vx, chain, emit)
    emit(f"{variant['id']}: model {model}; cleanup {llm_desc}; "
         f"{len(job['ids'])} to replay")

    conn = sqlite3.connect(os.path.join(sdir, "shadow.sqlite"), timeout=30)
    conn.row_factory = sqlite3.Row
    pump_blocks = max(1, int(round(PUMP_EVERY_SEC * SAMPLE_RATE / BLOCK)))
    done = 0
    for did in job["ids"]:
        row = conn.execute("SELECT audio_path FROM dictation WHERE id=?",
                           (did,)).fetchone()
        if row is None:
            continue
        lines.clear()
        pasted.clear()
        for k in timers:
            timers[k] = 0.0
        err = None
        release_sec = None
        segs = 0
        try:
            audio = _read_wav(os.path.join(sdir, row["audio_path"]))
            blocks = [audio[i:i + BLOCK].reshape(-1, 1)
                      for i in range(0, len(audio), BLOCK)]
            buf = []
            vx.audio_frames = buf
            vx._active_buf = buf
            vx.overflow_count = 0
            vx.recording = True
            session = None
            use_prefetch = (getattr(vx, "PREFETCH", False)
                            and hasattr(vx, "_PrefetchSession")
                            and hasattr(vx, "_prefetch_pump"))
            if use_prefetch:
                vx._prefetch_generation = getattr(
                    vx, "_prefetch_generation", 0) + 1
                session = vx._PrefetchSession(vx._prefetch_generation, buf)
                session.thread = None  # driven inline below, not threaded
            vx._prefetch_session = session
            phase["now"] = "prefetch"
            # Feed the audio as the capture callback would, pumping the
            # prefetch worker's cut-finder at its live poll cadence; then a
            # final catch-up pump, exactly as the worker does on release.
            for i, blk in enumerate(blocks, 1):
                buf.append(blk)
                if session is not None and i % pump_blocks == 0:
                    vx._prefetch_pump(session)
            if session is not None:
                vx._prefetch_pump(session)
                segs = len(session.texts)
            phase["now"] = "release"
            t0 = time.monotonic()
            vx.stop_and_transcribe()
            release_sec = time.monotonic() - t0
        except Exception as e:
            err = f"{e.__class__.__name__}: {e}"
        raw = next((ln[len(">> Raw: "):] for ln in reversed(lines)
                    if ln.startswith(">> Raw: ")), None)
        outcome = _llm_outcome(lines) if chain else "off"
        conn.execute(
            "INSERT OR REPLACE INTO run VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                did, variant["id"], job["config_key"], job["code_sha"],
                model, llm_desc, raw, pasted.get("text", ""),
                release_sec, timers["release"], timers["prefetch"],
                timers["llm"], segs,
                int(any(ln.startswith(">> Echo detected") for ln in lines)),
                outcome, "\n".join(lines[-40:]), err, now_iso(),
            ),
        )
        conn.commit()
        done += 1
        if err:
            emit(f"{did}: ERROR {err}")
    conn.close()
    emit(f"{variant['id']}: replayed {done}")


def _warm_llm_sync(vx, chain, emit):
    """Load each local-style cleanup model before timing anything.

    Live Vox warms its cleanup model at startup and keeps it warm with a
    heartbeat, so replays must not pay a cold load either (a 30B model can
    take 20s+ to page in, which would lose every early race and paste raw).
    """
    if not hasattr(vx, "_llm_format_local"):
        return
    saved = getattr(vx, "LLM_TIMEOUT", 10)
    vx.LLM_TIMEOUT = 180
    try:
        for i, b in enumerate(chain):
            if b not in getattr(vx, "_LOCAL_BACKENDS", ("local",)):
                continue
            try:
                if hasattr(vx, "_backend_url"):
                    url, m = vx._backend_url(i, b), vx._backend_model(i, b)
                    vx._llm_format_local("ok", vx._cleanup_system_prompt(),
                                         m, url)
                else:
                    vx._llm_format_local("ok", vx._cleanup_system_prompt())
            except TypeError:
                vx._llm_format_local("ok", vx._cleanup_system_prompt())
            except Exception as e:
                emit(f"cleanup warm-up failed for {b} "
                     f"({e.__class__.__name__}: {e})")
    finally:
        vx.LLM_TIMEOUT = saved


def _llm_outcome(lines):
    """Summarize what the cleanup pass did from the dictation's log lines."""
    for ln in reversed(lines):
        if ln.startswith(">> LLM cleanup ("):
            return "cleaned by " + ln[len(">> LLM cleanup ("):].split(")")[0]
        if ln.startswith(">> LLM cleanup rejected"):
            return "rejected (guard)"
        if ln.startswith(">> LLM cleanup: no backend answered"):
            return "timed out (raw pasted)"
        if ln.startswith(">> LLM cleanup skipped"):
            return "error (raw pasted)"
    return "not run"


# --- Status + CLI ------------------------------------------------------------------
def status():
    conn = open_db()
    n = conn.execute("SELECT count(*), coalesce(sum(audio_sec),0), "
                     "min(ts_utc), max(ts_utc) FROM dictation").fetchone()
    print(f"archive : {shadow_dir()}")
    print(f"wispr db: {wispr_db_path()}")
    print(f"dictations: {n[0]} ({n[1] / 60:.1f} min of audio), "
          f"{n[2]} .. {n[3]}")
    for v in load_variants():
        key = config_key(v)
        done = conn.execute("SELECT count(*) FROM run WHERE variant=? AND "
                            "config_key=?", (v["id"], key)).fetchone()[0]
        print(f"  {v['id']:<14} {done}/{n[0]} replayed with current config "
              f"{key} ({_git_sha(v['vox_dir'])} @ {v['vox_dir']})")
    conn.close()


def main():
    ap = argparse.ArgumentParser(
        description="Replay Wispr Flow dictations through Vox and compare.")
    ap.add_argument(
        "--dir", help="data folder (default %%LOCALAPPDATA%%\\vox\\shadow; "
        "same as VOX_SHADOW_DIR). Use a folder outside AppData when shells "
        "run inside an MSIX app container (e.g. the Claude desktop app), "
        "whose AppData writes are silently redirected into the package.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="ingest + replay + report")
    p_run.add_argument("--max", type=int, default=20,
                       help="max dictations to replay per variant (default 20)")
    sub.add_parser("ingest", help="archive new Wispr dictations")
    p_rep = sub.add_parser("replay", help="replay archived audio through Vox")
    p_rep.add_argument("--variant")
    p_rep.add_argument("--max", type=int)
    p_rep.add_argument("--ids", nargs="*")
    p_report = sub.add_parser("report", help="write report.html")
    p_report.add_argument("--open", action="store_true")
    sub.add_parser("status", help="archive + replay coverage")
    p_w = sub.add_parser("_worker")
    p_w.add_argument("job")
    args = ap.parse_args()
    if args.dir:
        os.environ["VOX_SHADOW_DIR"] = os.path.abspath(args.dir)

    if args.cmd == "_worker":
        _worker(args.job)
        return
    if args.cmd == "status":
        status()
        return
    if args.cmd in ("run", "ingest"):
        try:
            added = ingest(quiet=(args.cmd == "run"))
        except FileNotFoundError as e:
            say(f"ingest: {e}")
            added = 0
        if args.cmd == "ingest":
            return
    if args.cmd in ("run", "replay"):
        n = replay(max_per_variant=args.max,
                   only_variant=getattr(args, "variant", None),
                   ids=getattr(args, "ids", None))
        if args.cmd == "run":
            # Heartbeat on every run, even an idle one, so "is it still
            # running?" is answered by the data folder, not by guesswork.
            with open(os.path.join(shadow_dir(), "last-run.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"at": now_iso(), "ingested": added,
                           "replayed": n}, f)
        if args.cmd == "run" and not (n or added) and os.path.exists(
                os.path.join(shadow_dir(), "report.html")):
            return  # nothing new: leave the report as it is
    if args.cmd in ("run", "replay", "report"):
        import shadow_report
        path = shadow_report.write_report(shadow_dir(), load_variants(),
                                          config_key)
        say(f"report: {path}")
        if getattr(args, "open", False):
            os.startfile(path) if IS_WINDOWS else subprocess.run(
                ["xdg-open", path])


if __name__ == "__main__":
    main()
