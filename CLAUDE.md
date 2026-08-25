# WhisperApp - Claude Code Instructions

Last reviewed: 2026-07-21

Local speech transcription with speaker diarization. Cross-platform (Windows + macOS +
Linux). Ships a web UI, a system-tray app, a CLI and a REST API around a whisperx /
pyannote pipeline, with an Apple Silicon fast path via mlx-whisper. See README.md for the
full feature overview.

## Commands

Run the full app (web UI + system tray):

```bash
# Windows
python -m whisperapp
```

```zsh
# Mac (after install)
.venv/bin/python -m whisperapp
```

Run the headless CLI (talks to a running daemon over REST):

```bash
whisperapp            # entry point: whisperapp.cli:main
```

Run the test suite:

```bash
python -m pytest tests/ -q
```

## Architecture

The `whisperapp` package is the whole application. A worker loads a Whisper backend
(whisperx on CPU/CUDA, mlx-whisper on Apple Silicon) plus pyannote for diarization, and
serves three front ends over one process:

- `server.py` - FastAPI app exposing the web UI and the REST API.
- `ui.py` / `static/` / `templates/` - the browser UI (served on port 7860).
- `tray.py` - the system-tray / menu-bar app (pystray, plus rumps on macOS).
- `cli.py` - the headless CLI that drives a running daemon via REST.

Supporting modules: `worker.py` and `queue.py` (job execution), `streaming.py` (live
transcription), `speakers.py` (diarization), `acoustic.py` / `pauses.py` / `emotion.py`
(volume, pitch, speaking-rate and emotion markers), `document_formats.py` and
`formatters.py` (DOCX / PDF / text export), `ai.py` (optional Claude / OpenAI / Ollama
post-processing), `model_registry.py` and `checkpoints.py` (model selection and caching),
`config.py`, `startup.py`, `setup_env.py`, `updater.py` and `watcher.py`.

## Key Files

Per-file map (incl. `COMMS.md`, `mcp.json` is NOT a Claude MCP config, `whisper_webui/` submodule):
`docs/CLAUDE-REFERENCE.md` §Key Files.

## Conventions

- Coverage is the default: plain `pytest` runs everything (skips are fine, failures are
  not). A reduced run is an explicit, logged opt-in; run and record the full suite before
  declaring done.
- Cross-machine work is coordinated through `COMMS.md`. Read it at the start of a session
  that spans both machines; append status under the STATUS LOG section in the format
  `[MACHINE -> MACHINE] date: message`.
- Personal repo `github.com/jlldavies/whisperapp`, branch `master`; push directly when
  asked. Never edit the `whisper_webui` submodule from this repo.
- Keep platform-specific dependencies behind environment markers in `pyproject.toml`
  (as with `mlx-whisper`, `truststore` and the pyobjc AVFoundation framework).

## Environment

- Python >= 3.10.
- First-time Mac install:

  ```zsh
  brew install portaudio
  /opt/homebrew/bin/python3.13 -m venv .venv
  .venv/bin/pip install -e ".[desktop,dev]"
  ```

- Optional extras (`pip install -e ".[<extra>]"`): `desktop` (tray, mic recording),
  `claude` / `openai` / `ai-all` (AI post-processing SDKs; Ollama needs none), `dev`
  (pytest plus desktop), `build` (pyinstaller).
- Platform notes (Windows truststore, macOS rumps + mlx-whisper, CUDA float16): `docs/CLAUDE-REFERENCE.md` §Environment.
- Ports: web UI on `http://127.0.0.1:7860`, REST API / health on `http://127.0.0.1:7861`.

## Testing

`pytest.ini` sets `asyncio_mode = auto` and, by default, deselects the `integration` and
`browser` markers (`addopts = -m "not integration and not browser"`):

- `integration` - tests that download models or need `HF_TOKEN`.
- `browser` - Playwright tests that require the live app on ports 7860/7861.

Run those explicitly when needed, for example `pytest -m browser` with the app running.
Everything else runs on a plain `pytest tests/ -q`.

## Gotchas

- `mcp.json` is the app's REST tool manifest, not a Claude MCP config. Do not register it
  in `.mcp.json` or a desktop config as-is.
- `whisper_webui/` is a submodule; never edit files inside it from this repo.
- The web UI and REST API bind fixed ports 7860 and 7861; free them before starting.
- macOS tray support needs `rumps`; without the `desktop` extra the tray will not appear.
- Some dependencies are platform-gated by markers, so a lockstep dependency list differs
  per OS. Trust the `pyproject.toml` markers rather than pinning by hand.
