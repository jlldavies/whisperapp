# whisperapp — CLAUDE.md reference (moved 2026-08-25)

Reference material moved VERBATIM out of `CLAUDE.md` on 2026-08-25 so the always-loaded file stays inside its 120-line root budget (models follow ~100–150 instructions across ALL loaded files). `CLAUDE.md` points here. Sections keep their original headings.

## Key Files

- `whisperapp/` - the main application package (modules listed under Architecture).
- `whisperapp/__main__.py` - full-app entry point (`whisperapp-app` gui-script).
- `whisperapp/cli.py` - CLI entry point (`whisperapp` script).
- `tests/` - the pytest suite (one `test_*.py` per module, plus `fixtures/`).
- `pyproject.toml` / `pytest.ini` - packaging, dependencies and test configuration.
- `COMMS.md` - cross-machine coordination file (Windows and Mac via Dropbox).
- `mcp.json` - the app's own REST tool manifest (transport / url / tools for its API on
  port 7861). It is NOT a Claude MCP config; do not register it as one.
- `whisper_webui/` - a vendored git submodule (upstream Whisper-WebUI). Do not modify it.

- Platform notes:
  - Windows: tray icon in the notification area; SSL routes through the Windows
    certificate store (`truststore`) so corporate proxies / TLS inspectors work.
  - macOS: icon in the menu bar; requires `rumps` (installed with pystray). Apple Silicon
    installs `mlx-whisper` automatically for fast on-device transcription.
  - CUDA on Windows/Linux is detected automatically and uses float16.
