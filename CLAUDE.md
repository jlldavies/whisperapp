# WhisperApp — Claude Code Instructions

## Project
Local speech transcription with speaker diarization. Cross-platform (Windows + macOS).
See README.md for full feature overview.

## Key files
- `whisperapp/` — main package
- `tests/` — pytest suite (138 passing, 0 failing as of 2026-05-07)
- `COMMS.md` — **cross-machine coordination file** (Windows ↔ Mac via Dropbox)

## Cross-machine debugging
When working across Windows and Mac simultaneously, both Claude sessions communicate
via `COMMS.md` in this directory. It syncs via Dropbox.
- Read it at the start of any session
- Append status updates under the STATUS LOG section
- Format: `[MACHINE -> MACHINE] date: message`

## Running tests
```bash
python -m pytest tests/ -q
```
Expected: 138 passed, 2 skipped, 0 failed

## Running the app
**Windows:**
```bash
python -m whisperapp
```

**Mac (after install):**
```zsh
.venv/bin/python -m whisperapp
```

## Mac install (first time)
```zsh
brew install portaudio
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/pip install -e ".[desktop,dev]"
```

## Platform notes
- Windows: tray icon appears in notification area (bottom-right)
- macOS: icon appears in menu bar (top-right) — requires `rumps` (installed with pystray)
- Apple Silicon: mlx-whisper installs automatically for fast GPU transcription
- CUDA on Windows/Linux: detected automatically, uses float16

## Ports
- Web UI: http://127.0.0.1:7860
- REST API / health: http://127.0.0.1:7861
