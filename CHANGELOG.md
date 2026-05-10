# Changelog

All notable changes to WhisperApp are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [1.1.0] — 2026-05-10

### Added
- **Chain-of-custody transcripts** — SHA-256 hash of source file embedded in all output formats; `{{source_hash}}`, `{{file_size_bytes}}`, `{{legal_transcript}}` template markers
- **Legal HTML template** — line-numbered (`L0001`), Courier font, provenance block, page numbers; ready for formal disclosure
- **Transcript viewer** — click any completed job to open a built-in document reader: line-numbered, searchable, copyable, print-ready
- **`/jobs/{id}/segments` endpoint** — paged segment data including `source_metadata` (SHA-256, file size, path, timestamps)
- **Headless / Docker mode** — `--headless` flag and `WHISPERAPP_HEADLESS=1` env var; REST API on `0.0.0.0`, no tray or browser; official `Dockerfile` and `docker-compose.yml`
- **`[desktop]` optional extra** — pystray, Pillow, pyaudio moved out of core; headless installs skip them automatically
- **Environment variable config overrides** — `HF_TOKEN`, `WHISPERAPP_AI_PROVIDER`, `WHISPERAPP_AI_API_KEY`, `WHISPERAPP_AI_MODEL`, `WHISPERAPP_OUTPUT_PATH` for Docker/CI deployments
- **GitHub Actions CI** — test suite runs on every push (ubuntu-latest, CPU torch); Docker image built on push, pushed to GHCR on tags
- **PyPI publish workflow** — triggered on GitHub Release; builds and uploads via `pypa/gh-action-pypi-publish`
- **Webhooks** — `callback_url` on `POST /transcribe`; HMAC-SHA256 signed (`X-WhisperApp-Signature`); SSRF protection (loopback-only by default, configurable `webhook_allowed_hosts`); `WHISPERAPP_WEBHOOK_ALLOWED_HOSTS` and `WHISPERAPP_WEBHOOK_SECRET` env vars
- **First-run setup flow** — detects existing torch/whisperx/pyannote, picks correct torch variant (CUDA/CPU/Apple Silicon), streams pip install output in browser; HuggingFace token step before opening the app
- **Model download sizes on button** — Download/Install button shows size inline (e.g. "Download · 1.46 GB") for undownloaded models
- **App update banner** — sidebar shows "Update available" link when a newer GitHub release exists
- **API key auth** — optional `api_key` config field; non-loopback requests require `Authorization: Bearer <key>` when set; safe for `--headless` / Docker deployments
- **Version bump script** — `scripts/bump_version.py X.Y.Z` updates all hardcoded version strings in one command
- **macOS DMG installer** — PyInstaller launcher bundle (~50 MB, no torch); wrapped with `create-dmg`; built by GitHub Actions on release tags
- **Windows installer** — PyInstaller + Inno Setup; per-user install, no UAC; built by GitHub Actions on release tags

### Changed
- `sounddevice` replaces `pyaudio` in the `[desktop]` extra — ships with prebuilt portaudio wheels on Windows, no manual DLL install required
- Version string centralised — bump with `python scripts/bump_version.py X.Y.Z`

### Fixed
- Emotion model size displayed as formatted string (e.g. `1.37 GB`) instead of raw `1400 MB`

---

## [1.0.0] — 2026-04-01

### Added
- Initial release: WhisperX transcription, speaker diarization, live streaming, acoustic analysis, emotion detection, DOCX/PDF output with templates, REST API, CLI, system tray, web UI
