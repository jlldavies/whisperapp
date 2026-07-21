# WhisperApp — Claude Code Instructions

Last reviewed: 2026-07-02

## Project
Local speech transcription with speaker diarization. Cross-platform (Windows + macOS).
See README.md for full feature overview.

## Key files
- `whisperapp/` — main package
- `tests/` — pytest suite
- `COMMS.md` — **cross-machine coordination file** (Windows ↔ Mac via Dropbox)
- `mcp.json` — the app's **own REST tool manifest** (transport/url/tools for its API on :7861).
  It is **not** a Claude MCP config — do not register it in `.mcp.json`/desktop config as-is.

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
The whole suite must pass (skips are fine, failures are not). Coverage is the default —
a reduced run is an explicit, logged opt-in; run and record the full suite before declaring done.

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

## MCP servers
No project-scoped Claude MCP servers — the machine-wide ones (`nexus-remote`, `rsvp-reader`)
suffice. Verify with `claude mcp list` (each ✔ Connected). The app itself is driven over its
REST API on :7861 (see `mcp.json` note above), not via a Claude MCP registration.
