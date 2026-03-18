# WhisperApp

Local audio/video transcription with speaker diarization, powered by [WhisperX](https://github.com/m-bain/whisperX). Runs as a background app with a Gradio web UI, REST API, CLI, and system tray icon.

## Features

- **WhisperX transcription** with word-level timestamps and alignment
- **Speaker diarization** via pyannote.audio — identifies who said what
- **Speaker labelling UI** — review and rename speakers before saving
- **Multiple output formats** — TXT, SRT, VTT, JSON, TSV
- **Job queue** — SQLite-backed, submit multiple files and they process in order
- **Checkpoint recovery** — if the app crashes mid-job, it resumes from the last completed stage
- **REST/MCP API** on `127.0.0.1:7861` — local-only, for automation and integrations
- **CLI** — `whisperapp transcribe`, `whisperapp list`, `whisperapp status`
- **System tray** — runs in the background, double-click to open the UI
- **Auto-updater** — upgrades WhisperX and dependencies on startup
- **Cross-platform** — Windows and macOS (system startup registration for both)

## Quick Start

```bash
# Clone
git clone --recurse-submodules https://github.com/jlldavies/whisperapp.git
cd whisperapp

# Set up venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install
pip install -r requirements.txt
pip install -e .

# Run
python -m whisperapp
```

On first run, open http://127.0.0.1:7860 and enter your HuggingFace token in the Settings tab (required for speaker diarization).

## Architecture

```
whisperapp/
├── __main__.py      # Entry point — launches all services
├── config.py        # Config manager (~/.whisperapp/config.json)
├── sanitise.py      # Input validation and sanitisation
├── queue.py         # SQLite job queue
├── checkpoints.py   # Crash-recovery checkpoints
├── worker.py        # WhisperX processing pipeline
├── formatters.py    # Output formatters (txt/srt/vtt/json/tsv)
├── speakers.py      # Speaker snippet extraction and renaming
├── server.py        # FastAPI REST/MCP server (port 7861)
├── ui.py            # Gradio web UI (port 7860)
├── tray.py          # System tray icon
├── cli.py           # CLI commands
├── updater.py       # Auto-updater
└── startup.py       # OS startup registration
```

## Ports

| Service | URL |
|---------|-----|
| Web UI | http://127.0.0.1:7860 |
| REST API | http://127.0.0.1:7861 |
| API Docs | http://127.0.0.1:7861/docs |

## CLI Usage

```bash
# Submit a file
whisperapp transcribe recording.mp3 -m large-v2 --diarize

# Check status
whisperapp status <job-id>

# List jobs
whisperapp list

# Cancel a job
whisperapp cancel <job-id>

# Get transcript
whisperapp get <job-id> -f txt
```

## API Examples

```bash
# Submit transcription
curl -X POST http://127.0.0.1:7861/transcribe \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/audio.mp3", "model": "large-v2", "diarize": true}'

# Check job status
curl http://127.0.0.1:7861/jobs/<job-id>

# List all jobs
curl http://127.0.0.1:7861/jobs

# Get transcript
curl http://127.0.0.1:7861/jobs/<job-id>/transcript?format=txt
```

## Testing

```bash
# Unit tests (fast, no GPU or HF token needed)
pytest tests/ -v --ignore=tests/test_integration.py

# Integration tests (downloads models, needs HF token)
HF_TOKEN=hf_xxx pytest tests/test_integration.py -v -m integration

# All tests
pytest tests/ -v

# Coverage
pytest tests/ --cov=whisperapp --cov-report=term-missing
```

## Requirements

- Python 3.10+
- HuggingFace token (free, for pyannote.audio speaker diarization models)
- ~2GB disk for WhisperX models (downloaded on first use)
