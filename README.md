<p align="center">
  <img src="assets/vox-logo.svg" alt="Vox" width="480">
</p>

# Vox

Push-to-talk dictation for **Linux and Windows**. Hold **Ctrl+Win** to record, release to transcribe and paste the result wherever your cursor is.

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for local, offline speech-to-text with GPU acceleration. Inspired by [Whisper Writer](https://github.com/savbell/whisper-writer) but built as a lightweight, single-file alternative that just works with minimal setup.

## Features

- **Global push-to-talk** — works in any window (evdev on Linux, pynput on Windows)
- **Cross-platform** — one script runs on Linux/X11 and Windows
- **Auto device detection** — CUDA if an NVIDIA GPU is present, otherwise CPU (`int8`) — no config needed
- **Clipboard paste** — pastes instead of simulating keystrokes, for instant output
- **Personal dictionary** — biases Whisper toward your vocabulary and auto-corrects misheard terms (`dictionary.json`)
- **Filler word removal** — strips "um", "uh", "you know", "I mean", etc.
- **Punctuation** — uses an initial prompt to encourage proper punctuation from Whisper
- **Terminal-aware (Linux)** — detects terminal emulators and uses Ctrl+Shift+V automatically
- **Autostart** — run on login (desktop entry on Linux, Startup shortcut on Windows)

## Requirements

**All platforms:** Python 3.10+. An NVIDIA GPU with CUDA is optional — it's auto-detected, and the app falls back to CPU otherwise.

**Linux:**
- X11 (Wayland is not supported — xdotool requires X11)
- System packages: `xdotool`, `xclip`, `xprop`
- User must be in the `input` group (for global hotkey capture via evdev)

**Windows:**
- Windows 10/11 — no special permissions or groups required
- Python packages `pynput` and `pyperclip` (installed automatically)

## Installation (Linux)

```bash
# Install system dependencies
sudo apt install xdotool xclip x11-utils python3-venv

# Add yourself to the input group (required for global hotkeys)
sudo usermod -aG input $USER
# Log out and back in for group change to take effect

# Clone and install
git clone https://github.com/eonfathom/vox.git
cd vox
chmod +x install.sh
./install.sh
```

The installer creates a venv at `~/.local/share/vox/venv/`, installs dependencies, creates a launcher at `~/.local/bin/vox`, and sets up autostart on login.

## Installation (Windows)

```powershell
# Clone
git clone https://github.com/eonfathom/vox.git
cd vox

# Install: creates a venv, installs deps, adds GPU libraries if an NVIDIA GPU is present
.\install-windows.ps1

# ...or install and also run automatically on login:
.\install-windows.ps1 -Autostart
```

If PowerShell blocks the script, allow it for your user once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

The installer auto-detects your GPU. On an NVIDIA machine it installs the CUDA libraries (cuBLAS, cuDNN, cudart) and runs on the GPU; on an Intel-only laptop it installs the CPU build and the app uses `int8` on the CPU automatically — the same code, no config changes.

## Usage

**Linux:**
```bash
vox          # or it starts automatically on login
```

**Windows:**
```powershell
.\run-windows.ps1          # or starts on login if installed with -Autostart
```

Then, on either platform:
1. Hold **Ctrl+Win** — recording starts
2. Speak
3. Release **Ctrl+Win** — audio is transcribed and pasted at your cursor

## Managing Vox

Vox runs windowless, so use the bundled control script from the repo folder:

```powershell
.\vox.ps1 restart   # reload after editing dictionary.json (also: start / stop / status / guard)
```

`stop` frees the model's VRAM if you need the GPU for something else; `start` brings it back. If PowerShell blocks it: `powershell -ExecutionPolicy Bypass -File .\vox.ps1 restart`.

**Self-healing:** `.\vox.ps1 guard` runs a guardian loop that starts Vox and restarts it within ~20 s if it ever dies (with crash-storm backoff and duplicate-instance cleanup, logged to `%LOCALAPPDATA%\vox\guardian.log`). Point your Startup shortcut at `wscript.exe "<repo>\vox-guard.vbs"` (which runs the guardian with zero console flash at login - aiming the shortcut at powershell directly flashes a window even with `-WindowStyle Hidden`) and dictation becomes unkillable across crashes. `stop` stops the guardian too, so a deliberate stop stays stopped.

## Configuration

Configuration is via environment variables.

- Linux: set them in your shell profile or before launching.
- Windows: set them before launching, e.g. `$env:VOX_MODEL = "large-v3"` then `.\run-windows.ps1`.

| Variable | Default | Description |
|---|---|---|
| `VOX_MODEL` | auto | `large-v3` on a CUDA GPU, `base` on CPU. Override with `tiny`/`base`/`small`/`medium`/`large-v3` |
| `VOX_LANG` | `en` | Language code (`en`, `es`, `fr`, …) or empty for auto-detect |
| `VOX_HOTKEY` | `ctrl+win` | Push-to-talk chord; `+`-joined modifiers from `ctrl`, `alt`, `win`, `shift` |
| `VOX_MIC` | system default | Input device by name (as listed in the tray's Microphone menu, e.g. `Headset (Glasses_4743)`). Easier: pick it from the tray — the choice persists per machine in `settings.local.json`, and an absent device falls back to the system default. Use the tray's "Rescan devices" after connecting a Bluetooth/USB mic mid-session |
| `VOX_BEAM` | `5` | Decoding beam width; higher = more accurate, a bit slower. Use `1` on a slow CPU |
| `VOX_RELEASE_TAIL` | `0.2` | Seconds to keep recording after release so trailing words aren't clipped |
| `VOX_PAD` | `0.15` | Seconds of trailing silence padded onto the buffer so Whisper finalizes the last segment |
| `VOX_DEVICE` | auto | Auto-detected: `cuda` if a GPU is present, else `cpu`. Override with `cuda`/`cpu`. |
| `VOX_COMPUTE` | auto | `float16` on GPU, `int8` on CPU. Override with `float16`/`int8`/`float32`. |
| `VOX_DICT` | `dictionary.json` | Path to the personal dictionary file |
| `VOX_STRIP_PHANTOMS` | `1` | Strip hallucinated trailing hotword echoes from the transcript. Set `0` to disable |
| `VOX_RESTARTS` | `1` | Collapse verbatim sentence restarts ("we need the, we need the plan" → "we need the plan") without an LLM. `0` to disable |
| `VOX_SMARTCASE` | `1` | Insertion-aware casing (Windows): reads the text around the caret (UI Automation, app-dependent; falls back to vox's own last paste) so dictating mid-sentence doesn't start with a capital, glue words together, or split the sentence with a period. `0` to disable |
| `VOX_HUD` | `1` | Floating dictation HUD (Windows): a click-through overlay pinned above the taskbar at the bottom-right of the monitor you're dictating into — a red plus, live elapsed timer, and mic level meter while you hold the chord; then a release-to-paste latency clock and a brief `words · tokens` result flash. `0` to disable |
| `VOX_HUD_ANCHOR` | `corner` | `caret` anchors to the text insertion point where the OS exposes it (static fallback elsewhere); `cursor` follows the mouse live |
| `VOX_LOG` | auto | Diagnostic log file (rotating, 512 KB × 2). Default `%LOCALAPPDATA%\vox\vox.log` (Linux: `~/.local/state/vox/vox.log`); set a path to move it or `0` to disable |
| `VOX_TRANSCRIPT_DIR` | unset | Directory (e.g. an Obsidian folder) for a per-day markdown record of everything you dictate (`YYYY-MM-DD vox.md`, one `- **HH:MM** text` bullet per dictation). Unset = off |
| `VOX_LLM` | `off` | Optional LLM cleanup pass. `local` = a local OpenAI-compatible server (Ollama; offline, no API key); `anthropic` = Claude via API (needs `ANTHROPIC_API_KEY`); a comma list (`anthropic,local`) is a preference chain — backends race in parallel and the earliest-listed success wins, so cloud quality degrades to local automatically when offline |
| `VOX_LLM_MODEL` | `qwen3:4b-instruct` (local) / `claude-haiku-4-5` (anthropic) | Model per backend. With a chain, a comma list aligned by position (`claude-haiku-4-5,qwen3:4b-instruct`); unset positions use the per-backend defaults. The local default is the smallest model that reliably rewrites spoken self-corrections (see benchmark note in `dictation.py`); warm ~0.7–1.2s even on CPU |
| `VOX_LLM_URL` | `http://127.0.0.1:11434/v1` | Local server's OpenAI-compatible base URL (Ollama default; works with LM Studio, llama.cpp `--api`, etc.). Use `127.0.0.1`, not `localhost` — on Windows the latter stalls ~2s per request on IPv6 fallback |
| `VOX_LLM_KEEPALIVE` | `30m` | Keep the local model resident between dictations so there's no reload latency (Ollama) |

Example — force the tiny model on CPU:

```bash
# Linux
VOX_MODEL=tiny VOX_DEVICE=cpu vox
```
```powershell
# Windows
$env:VOX_MODEL="tiny"; $env:VOX_DEVICE="cpu"; .\run-windows.ps1
```

### Optional: LLM cleanup (Wispr-style polish)

By default Vox types the raw (offline) transcript. For Wispr-Flow-style polish — grammar, punctuation, removed filler, spoken commands like "new paragraph" / "scratch that", spoken self-corrections ("…push to main or… sorry, submit a pull request" pastes as just the corrected version; "never mind, what I meant is…" keeps only the replacement), and merged mid-thought pauses (Whisper's spurious "…edit a little bit. With Vox…" becomes one sentence again) — enable an LLM pass. Two backends, usable alone or chained:

**Local (recommended — fully offline, no API key, low latency).** Runs against a local [Ollama](https://ollama.com) server on your own GPU:

```powershell
winget install Ollama.Ollama       # once
ollama pull qwen3:4b-instruct      # small, fast, handles self-corrections
setx VOX_LLM local
```

Restart Vox. It POSTs the raw transcript to the local OpenAI-compatible endpoint (`VOX_LLM_URL`, default Ollama at `127.0.0.1:11434/v1`) using only the Python standard library — no extra package to install. A warm pass is well under a second on a GPU and **~0.7–1.2s on a fast CPU** (measured on a Core Ultra 9); the model is pre-loaded at startup and stays resident (`VOX_LLM_KEEPALIVE`) so there's no per-dictation reload. Point `VOX_LLM_URL` at LM Studio or `llama.cpp --api` instead if you prefer. Model choice matters more than size here — pick one that *follows instructions* (returns cleaned text) rather than *chats back*; `qwen3:4b-instruct` is the tested default and the smallest model that handles mid-sentence self-corrections, including either/or retractions ("…push to main or… sorry, submit a pull request") that `qwen2.5:3b-instruct` refuses at any prompt. Avoid `llama3.2:3b` — it tends to reply to the transcript instead of cleaning it.

**Anthropic (cloud).** Highest quality, needs a key and internet:

```powershell
setx ANTHROPIC_API_KEY sk-ant-...
setx VOX_LLM anthropic
```

This uses the official `anthropic` SDK (`pip install anthropic`), fast **Haiku** by default (~$0.001 per dictation).

**Chained (best of both — cloud quality online, local on a plane):**

```powershell
setx ANTHROPIC_API_KEY sk-ant-...
setx VOX_LLM anthropic,local
```

A comma-separated `VOX_LLM` is a preference chain: every dictation fires all backends in parallel under the same latency budget, and the earliest-listed one that succeeds wins. When the cloud is unreachable — plane, outage, missing key — that attempt fails in milliseconds and the local result, already in flight, serves the paste instead. No mode switching, no config change mid-flight.

Every configuration is **fully fail-safe**: no server/key, no internet, a timeout, or any error falls back to the raw transcript, so a dictation is never lost. Your personal `dictionary.json` corrections still run last, so brand terms like Rokid always survive.

**Missing-model fallback.** If the configured local model isn't actually installed on the server (a fresh machine, a pruned Ollama), every cleanup would 404 identically — and under `pythonw` nothing makes that visible, so raw text would paste for weeks. Vox now detects this at warm-up, switches to the best *installed* known-good model (`qwen3:4b-instruct` → `qwen2.5:3b-instruct` → `qwen2.5:1.5b-instruct`), and logs the `ollama pull` command that restores the configured one.

### Model size guide

| Model | Speed | Accuracy | VRAM | Good for |
|---|---|---|---|---|
| `tiny` | Fastest | Good for clear speech | ~1 GB | Weak CPUs |
| `base` | Fast | Good balance | ~1 GB | Default on CPU-only machines |
| `small` | Moderate | Better accuracy | ~2 GB | CPU laptops wanting accuracy |
| `medium` | Slow | High accuracy | ~5 GB | Mid-range GPUs |
| `large-v3` | Slowest | Best accuracy | ~3 GB (fp16) | Default on any CUDA GPU |

On a CPU-only laptop, stick to `base` or `small`. On a GPU with 10 GB+ VRAM, `large-v3` gives the best accuracy at no real speed cost.

## Personal dictionary

Names, jargon, and product names that Whisper mishears go in `dictionary.json`:

```json
{
  "hotwords": ["Eon", "Obsidian", "Claude Code"],
  "corrections": { "clod code": "Claude Code", "obsidion": "Obsidian" }
}
```

- **`hotwords`** bias Whisper *during* recognition so it's more likely to hear your terms correctly.
- **`corrections`** are a whole-word, case-insensitive find/replace applied *after* transcription — a safety net for anything it still gets wrong.

Edit the file and restart the app to reload.

## Shadow mode: compare Vox with Wispr Flow

Keep dictating with [Wispr Flow](https://wisprflow.ai) and let Vox run alongside it. Wispr Flow (1.6.8xx and later) stores each dictation's audio in its local history database. `shadow.py` plays that same audio through Vox and records what Vox would have pasted. You get a side-by-side comparison of the two on identical speech without dictating twice.

```powershell
.\.venv\Scripts\python.exe shadow.py run      # archive new Wispr dictations, replay them, write the report
.\.venv\Scripts\python.exe shadow.py status   # what is archived and how much has been replayed
.\.venv\Scripts\python.exe shadow.py report --open
```

- **Ingest:** copies each new Wispr dictation (its WAV, Wispr's raw ASR, formatted and pasted text, and your later edits) into `%LOCALAPPDATA%\vox\shadow` (`~/.local/state/vox/shadow` elsewhere; override with `--dir` or `VOX_SHADOW_DIR`). Wispr's database is only ever opened read-only. The ingest also re-reads Wispr's text for the last two days, because Wispr updates a dictation after the audio lands (formatting, then your edits).
- **Shells inside an app container:** a shell started by an MSIX-packaged app, such as the Claude desktop app, has its AppData writes silently redirected into that package's own storage. A scheduled task then never sees them. From such a shell, pass `--dir` with a folder outside AppData, and give the scheduled task the same `--dir`.
- **Replay:** runs Vox's real release path, `stop_and_transcribe()`. That includes segment prefetch fed at the live block and poll cadence, the echo guard, the optional cleanup pass and the dictionary. Pasting, the HUD, the tray and the daily transcript file are all stubbed out. Nothing is typed anywhere.
- **Variants:** `variants.json` in the shadow folder lists which Vox to replay: a checkout directory plus environment overrides. You can compare "the Vox running on this PC" with the latest code, or a different cleanup model, on the same audio. Results are keyed by a hash of `dictation.py` + `dictionary.json` + env. After you change Vox, the corpus is replayed again, newest first, `--max` per run. So the growing Wispr corpus doubles as a regression suite.
- **Report:** `report.html` next to the archive. It shows word-level diffs against Wispr's pasted text, sorts each difference into a kind (names, misheard words, missing phrases, fillers, false starts, small words, numbers, spacing), lists recurring misses and dictionary suggestions, and plays each clip. Wispr is the reference, not the truth, so the report also flags where you corrected Wispr afterwards.
- **Schedule (Windows):** a Task Scheduler entry that runs `pythonw shadow.py run` every 15 minutes keeps the archive current. The run exits quickly when nothing is new.
- **Capture test (`shadow_capture.py`, Windows):** replaying Wispr's audio shows which engine hears better given the same sound. It cannot show whether Vox's own recording loses something. This small process records the same speech through Vox's capture code whenever Wispr's push-to-talk chord is held: the same device, the same stream settings, the same 0.4 s pre-roll. It reads the chord from Wispr's own settings and polls the keys, with no keyboard hook. It never loads a model and never pastes. `shadow.py ingest` pairs each recording with the Wispr dictation that started at the same moment. Recordings that match no dictation are deleted after 10 minutes. A variant with `"audio": "vox"` replays Vox's recording, and `"compare_to": "<variant>"` makes the report compare it with the same Vox on Wispr's recording of the same speech. Run it at logon under Task Scheduler, with restart on failure. `--self-test` records 2 seconds and exits.

- **Training data:** every archived clip comes with Wispr's text, so the archive doubles as a fine-tuning set for Whisper on your own voice. `shadow.py dataset --out <dir>` exports it. The labels are Wispr's pasted text, and the newest 15% of clips are held out for testing. `train/finetune_whisper.py` LoRA-tunes Whisper on the rest, reports the held-out word error rate before and after, and converts the result to a CTranslate2 model that Vox loads with `VOX_MODEL=<run>/ct2`. It runs in its own venv (`train/requirements.txt`). A few minutes of audio only overfits, so `shadow.py run` tracks the labelled hours. It logs each full hour and announces 5 h (`VOX_SHADOW_TRAIN_HOURS`) as ready. With a Helm endpoint in `<shadow dir>/notify.json`, the announcement goes there too. Your later edits are not used as labels, because they also change what was said.

Everything the shadow writes is personal speech, and none of it belongs in this repo.

## Troubleshooting

### No text appears (Linux)

Check your default audio input — if you have multiple devices, the wrong one may be selected:
```bash
wpctl status                    # list audio sources
wpctl set-default <source-id>   # set the correct mic
```

### "No keyboards found" (Linux)

Add yourself to the `input` group, then log out and back in:
```bash
sudo usermod -aG input $USER
```

### Nothing happens / CUDA errors (Windows)

- **First run is slow** — it downloads the model (~140 MB for `base`, ~1.5 GB for the GPU default `large-v3`) and warms up CUDA.
- **`cublas64_12.dll ... cannot be loaded`** — the CUDA libraries aren't installed. Run `.\install-windows.ps1` again, or `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12` into the venv. The app adds these to the DLL search path automatically at startup.
- **Pastes into the wrong place** — the text is pasted into whatever window has focus when you release Ctrl+Win. Keep your cursor where you want the text.
- **Windows key opens the Start menu** — releasing the Win key on its own can pop the Start menu. Press Ctrl a moment before Win, or choose another chord: `VOX_HOTKEY=ctrl+alt` (or `ctrl+shift`).
- **No GPU?** That's fine — it runs on CPU automatically. Use `VOX_MODEL=base` or `small` for reasonable speed.

### Slow transcription

- Use a smaller model: `VOX_MODEL=tiny` (or `base`/`small` on CPU).
- Confirm the device: the startup banner prints `Device: cuda` or `Device: cpu`.

## Uninstall

**Linux:** `./uninstall.sh`

**Windows:** delete the repo folder and remove the `Vox` shortcut from your Startup folder (`shell:startup`) if you enabled autostart.

## How it works

1. Keyboard events are captured globally — **evdev** on Linux, **pynput** on Windows (no window focus required)
2. When Ctrl+Win are both held, audio recording starts via **sounddevice**
3. On release, the audio buffer is passed to **faster-whisper** for transcription (CUDA if available, else CPU)
4. The transcript is cleaned: filler words stripped, then personal-dictionary corrections applied
5. The result is copied to the clipboard and pasted — **xdotool** on Linux (Ctrl+V, or Ctrl+Shift+V in terminals), **pynput** on Windows (Ctrl+V)

## Comparison with Whisper Writer

This project was inspired by [Whisper Writer](https://github.com/savbell/whisper-writer). Here's how they compare:

| | Vox | Whisper Writer |
|---|---|---|
| **Architecture** | Single file | ~12 source files, full PyQt5 GUI |
| **Dependencies** | A handful of Python packages | 69 Python packages |
| **Text output** | Clipboard paste (instant) | Keystroke simulation (slow, known bugs with duplicated/missing chars) |
| **Post-processing** | Filler removal, punctuation cleanup, personal dictionary | Trailing space/period removal only |
| **Recording modes** | Hold-to-record | Continuous, VAD, press-to-toggle, hold-to-record |
| **Hotkey** | Ctrl+Win, configurable via `VOX_HOTKEY` (evdev on Linux, pynput on Windows) | Configurable chord (evdev or pynput) |
| **Configuration** | Environment variables + `dictionary.json` | YAML config + settings GUI |
| **STT backend** | faster-whisper (local) | faster-whisper (local) + OpenAI API |
| **Platforms** | Linux/X11 + Windows | Linux, macOS, Windows |
| **GPU** | CUDA with auto-fallback to CPU | CUDA with auto-fallback to CPU |

**Why this project exists:** Vox trades flexibility for simplicity and reliability — clipboard paste is faster and more reliable than keystroke simulation, the single-file design has no complex dependency chains, and filler-word removal plus a personal dictionary produce cleaner output. If you want something that just works on Linux or Windows with minimal setup, this is for you.

## License

MIT
