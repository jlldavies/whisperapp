# WhisperApp

Local audio and video transcription with speaker diarization, acoustic analysis, emotion detection, and live microphone streaming. Powered by [WhisperX](https://github.com/m-bain/whisperX). Runs as a persistent background app with a modern web UI, REST API, CLI, and system tray icon.

---

## Screenshots

### Transcribe — submit files, manage the queue, review history
![Transcribe screen](docs/screenshots/transcribe.png)

### Live — real-time mic streaming with VAD and pause markers
![Live screen](docs/screenshots/live.png)

### Speakers — review snippets and label each speaker before saving
![Speakers screen](docs/screenshots/speakers.png)

### Settings — full configuration including model management
![Settings screen](docs/screenshots/settings.png)

> Screenshots live at `docs/screenshots/`. Run the app and capture your own.

---

## Features

### Transcription
- **WhisperX** word-level alignment — precise timestamps on every word
- **10+ model sizes** — `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `turbo`, and more
- **Multiple output formats** — TXT, SRT, VTT, JSON, TSV
- **Batch job queue** — submit multiple files; they process in order with SQLite persistence
- **Queue management** — reorder jobs (▲▼), delete individual items, pause and resume the processing worker
- **Checkpoint recovery** — jobs resume from the last completed stage after a crash or restart

### Speaker Diarization
- **pyannote.audio** — identifies who is speaking and when
- **Speaker labelling UI** — listen to snippets, then assign names before saving
- **Diarized outputs** — all formats include `[Speaker Name]:` prefixes when diarization is run

### Live Transcription
- **Real-time streaming** — faster-whisper + Silero VAD for sub-second latency
- **Device selection** — choose any microphone from the UI
- **Input level meter** — visualise and check levels before recording
- **Auto-worker-pause** — Live recording pauses the transcription worker to free CPU; resumes on stop
- **Save on stop** — transcript saved to your chosen output formats automatically

### Pause Detection
Three configurable tiers injected inline in all text-based output formats:

| Marker | Default trigger | Example |
|--------|----------------|---------|
| `[Pause, N seconds]` | ≥ 3 s | `[Pause, 4 seconds]` |
| `[Long Pause, N seconds]` | ≥ 10 s | `[Long Pause, 15 seconds]` |
| `[Gap, N minutes]` | ≥ 30 s | `[Gap, 2 minutes]` |

Only the deepest tier fires — a 45-second gap is labelled Gap, not also Long Pause.
All thresholds are configurable in Settings → Transcription.

### Acoustic Analysis
Optional librosa-based analysis that annotates each segment with speaking characteristics:

| Marker | Meaning |
|--------|---------|
| `[Loud]` | RMS energy significantly above baseline |
| `[Quiet]` | RMS energy significantly below baseline |
| `[Raised voice]` | High pitch variance (F0 std dev) |
| `[Animated]` | Moderately elevated pitch variance |
| `[Rapid]` | > 180 words per minute |
| `[Slow]` | < 90 words per minute |

All markers are individually toggleable. Thresholds for volume ratio, pitch std dev, and WPM are configurable.
Enable in Settings → AI features → Acoustic analysis.

### Emotion Detection
Optional Speech Emotion Recognition using HuggingFace models:

| Model | Type | Description |
|-------|------|-------------|
| `speechbrain-iemocap` | Classification | Happy / Sad / Angry / Neutral (SpeechBrain) |
| `wavlm-multi` | Classification | 9-class emotion from WavLM |
| `wav2vec2-8emotion` | Classification | 8-class from wav2vec2 fine-tuned on RAVDESS |
| `audeering-dimensional` | Dimensional | Continuous arousal + valence from audeering |

- Run one or multiple models simultaneously
- Classification models: unanimous agreement → label kept; split → highest confidence wins
- Dimensional model adds `[intense]` (arousal > 0.7) and `[negative tone]` (valence < 0.3)
- Optional AI synthesis — if an AI provider is configured, it can combine model outputs into a narrative annotation
- Models are downloaded on demand from HuggingFace; manage them in Settings → Models

### Model Management
Unified model registry for all five model categories:

| Category | What it controls |
|----------|-----------------|
| Transcription | Whisper model variants |
| Streaming | faster-whisper models for Live mode |
| Alignment | phoneme alignment models per language |
| Diarization | pyannote speaker diarization |
| Emotion | SER models (download / delete / status) |

Settings → Models shows disk usage per category, download progress bars, and lets you activate, deactivate, update, or delete any model.

### AI Integration
Optional LLM features (requires an AI provider in Settings → AI features):

| Feature | What it does |
|---------|-------------|
| Identify speakers | Uses transcript + audio snippets to suggest speaker names and roles |
| Meeting notes | Summarises the transcript into structured notes |
| Emotion synthesis | Combines outputs from multiple emotion models into a narrative |

Supported providers: **Claude** (Anthropic), **OpenAI**, **Ollama** (local).

---

## Quick Start

### macOS (Apple Silicon / Intel)

```zsh
git clone --recurse-submodules https://github.com/jlldavies/whisperapp.git
cd whisperapp

brew install portaudio
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m whisperapp
```

Apple Silicon uses `mlx-whisper` automatically for fast on-device transcription.

### Windows

```powershell
git clone --recurse-submodules https://github.com/jlldavies/whisperapp.git
cd whisperapp

python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m whisperapp
```

NVIDIA GPU is detected and used automatically (float16). SSL certificates are routed through the Windows certificate store via `truststore` — this handles corporate proxies and antivirus CA injection automatically.

### First run

1. The app starts in the system tray and opens http://127.0.0.1:7860
2. Open **Settings → API & CLI**
3. Enter your HuggingFace token (required for speaker diarization and some model downloads)
4. Optionally configure an AI provider for speaker identification, meeting notes, and emotion synthesis

---

## Architecture

```
whisperapp/
├── __main__.py          Entry point — launches all services
├── config.py            Config manager (~/.whisperapp/config.json)
├── sanitise.py          Input validation
├── queue.py             SQLite job queue with ordering and status tracking
├── checkpoints.py       Per-job crash-recovery checkpoints
├── worker.py            WhisperX processing pipeline (pause/resume support)
├── formatters.py        Output writers (txt/srt/vtt/json/tsv)
├── pauses.py            Pause detection and inline marker injection
├── acoustic.py          librosa-based acoustic feature analysis
├── emotion.py           Speech emotion recognition pipeline
├── emotion_registry.py  SER model registry and HuggingFace download manager
├── model_registry.py    Unified model catalogue (all 5 categories)
├── metadata.py          Job metadata extraction (duration, codec, etc.)
├── speakers.py          Speaker snippet extraction and renaming
├── streaming.py         Real-time streaming engine (faster-whisper + Silero VAD)
├── ai.py                AI provider abstraction (Claude / OpenAI / Ollama)
├── server.py            FastAPI REST/MCP server (port 7861)
├── tray.py              System tray icon (pystray)
├── cli.py               Click CLI commands
├── updater.py           Auto-updater for WhisperX and pyannote
├── startup.py           OS startup (login item) registration
└── static/
    ├── index.html
    ├── css/
    │   ├── tokens.css   Design tokens (colours, spacing, radii)
    │   └── app.css      Component styles
    └── js/
        ├── app.js       Router and keyboard shortcuts
        ├── shell.js     Sidebar, waveform, top bar components
        ├── transcribe.js Transcribe screen (queue, history, job detail)
        ├── live.js      Live screen (mic streaming, real-time transcript)
        ├── speakers.js  Speaker review screen
        ├── settings.js  Settings screen (all panels including Models)
        ├── api.js       Typed API client
        └── marks.js     SVG icons
```

### Processing pipeline

Each transcription job passes through stages:

```
QUEUED → EXTRACTING_AUDIO → TRANSCRIBING → ALIGNING → DIARIZING
       → SPEAKER_REVIEW (if diarize) → SAVING → DONE
```

Optional enrichment (run before SAVING):
- Pause markers inserted from word-level timestamps
- Acoustic markers added per segment via librosa
- Emotion labels added per segment via SER models

All enrichment is injected before the formatters run, so every output format (txt, srt, vtt) receives the same annotated content.

### Streaming pipeline

```
Microphone → getUserMedia (16 kHz PCM) → base64 chunk → POST /stream/chunk
           → faster-whisper VAD → segment detection → partial / committed text
```

Live mode auto-pauses the transcription worker when recording starts and resumes it when recording stops, preventing CPU contention between real-time and batch processing.

---

## REST API

Full REST API on `127.0.0.1:7861` — local-only. Interactive docs at http://127.0.0.1:7861/docs

### Jobs

```bash
GET    /jobs                   List all jobs
POST   /jobs                   Submit a new transcription job
GET    /jobs/{id}              Get job status and result
DELETE /jobs/{id}              Permanently delete a job
POST   /jobs/{id}/cancel       Cancel a queued or running job
POST   /jobs/{id}/move-up      Move job up in queue
POST   /jobs/{id}/move-down    Move job down in queue
GET    /jobs/{id}/transcript   Get transcript text (?format=txt|srt|vtt)
```

### Worker

```bash
GET    /worker/status          Worker running / paused state
POST   /worker/pause           Pause batch processing
POST   /worker/resume          Resume batch processing
```

### Live streaming

```bash
POST   /stream/start           Start a streaming session → {session_id}
POST   /stream/chunk           Send audio chunk (base64 PCM) → {new_text, partial}
POST   /stream/stop            Stop and save → {text, segments}
POST   /stream/polish          Run full WhisperX alignment + diarize on session audio
```

### Speakers

```bash
GET    /speakers/{id}          Get speaker snippets for review
POST   /speakers/{id}/label    Save speaker name mappings
POST   /speakers/{id}/identify AI-suggest speaker identities
POST   /speakers/{id}/notes    Generate meeting notes
```

### Models

```bash
GET    /models                 List downloaded models
GET    /models/catalogue       Full model catalogue with status
GET    /models/disk-summary    Disk usage by category
POST   /models/{id}/download   Start background model download
DELETE /models/{id}            Delete a model
POST   /models/{id}/activate   Set as active model for its category
POST   /models/{id}/update     Update to latest version
POST   /models/check-updates   Check all models for available updates
POST   /models/add-custom      Register a custom HuggingFace model
```

### Config & system

```bash
GET    /config                 Read current configuration
POST   /config                 Update configuration fields
GET    /info                   App version and hardware acceleration
POST   /storage/reveal         Open ~/.whisperapp in Finder/Explorer
```

### Example: submit and wait

```bash
# Submit
curl -X POST http://127.0.0.1:7861/jobs \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/audio.mp3", "model": "large-v2", "diarize": true}'

# Poll
curl http://127.0.0.1:7861/jobs/<job-id>

# Get transcript
curl "http://127.0.0.1:7861/jobs/<job-id>/transcript?format=srt"
```

---

## CLI

```bash
# Transcription
whisperapp transcribe recording.mp3
whisperapp transcribe recording.mp3 -m large-v2 --diarize --formats srt,vtt,txt

# Queue management
whisperapp list
whisperapp list --status queued
whisperapp status <job-id>
whisperapp cancel <job-id>
whisperapp delete <job-id>
whisperapp move-up <job-id>
whisperapp move-down <job-id>

# Worker control
whisperapp worker-status
whisperapp pause
whisperapp resume

# Transcript access
whisperapp transcript <job-id>
whisperapp transcript <job-id> --format srt

# Speaker tools (requires AI provider)
whisperapp identify-speakers <job-id>
whisperapp meeting-notes <job-id>

# Model management
whisperapp models
whisperapp download <model-id>
whisperapp delete-model <model-id>

# System info
whisperapp info
whisperapp ai-status
```

---

## Configuration reference

All settings live at `~/.whisperapp/config.json` — editable via Settings UI, `POST /config`, or directly.

| Key | Default | Description |
|-----|---------|-------------|
| `hf_token` | `""` | HuggingFace access token |
| `default_model` | `"large-v2"` | Whisper model for new jobs |
| `default_output_path` | `~/Downloads` | Where transcripts are saved |
| `diarize_by_default` | `true` | Enable diarization for new jobs |
| `streaming_model` | `"base"` | Whisper model for Live mode |
| `vad_silence_threshold` | `0.6` | Silero VAD sensitivity |
| `streaming_max_chunk_sec` | `10.0` | Max audio chunk duration for streaming |
| `ai_provider` | `"none"` | `none` / `claude` / `openai` / `ollama` |
| `ai_api_key` | `""` | API key for AI provider |
| `ai_model` | `""` | Model ID (blank = provider default) |
| `ai_base_url` | `""` | Ollama or OpenAI-compatible endpoint URL |
| `auto_update` | `true` | Check for model updates on launch |
| `pause_detection` | `true` | Insert pause markers in output |
| `pause_threshold` | `3` | Seconds before `[Pause]` marker |
| `long_pause_threshold` | `10` | Seconds before `[Long Pause]` marker |
| `gap_threshold` | `30` | Seconds before `[Gap]` marker |
| `acoustic_enabled` | `false` | Enable acoustic feature analysis |
| `acoustic_volume_enabled` | `true` | Volume markers (`[Loud]` / `[Quiet]`) |
| `acoustic_volume_threshold` | `1.8` | RMS ratio vs rolling baseline |
| `acoustic_pitch_enabled` | `true` | Pitch markers (`[Raised voice]` / `[Animated]`) |
| `acoustic_pitch_threshold` | `40.0` | F0 std dev threshold in Hz |
| `acoustic_rate_enabled` | `true` | Rate markers (`[Rapid]` / `[Slow]`) |
| `acoustic_rate_fast_wpm` | `180` | WPM threshold for `[Rapid]` |
| `acoustic_rate_slow_wpm` | `90` | WPM threshold for `[Slow]` |
| `emotion_enabled` | `false` | Enable emotion detection |
| `emotion_model_ids` | `[]` | Which SER model IDs to run |
| `emotion_confidence_threshold` | `0.65` | Min confidence to include an emotion label |
| `emotion_combine_with_ai` | `false` | Use AI provider to synthesise model outputs |

---

## Development

### Running tests

```bash
python -m pytest tests/ -q
```

Expected: 138 passed, 2 skipped, 0 failed.

Integration tests (require a real HuggingFace token and downloaded models):

```bash
HF_TOKEN=hf_... pytest tests/test_integration.py -v -m integration
```

### Test structure

```
tests/
├── test_config.py              Config load/save/defaults
├── test_queue.py               Job queue CRUD and ordering
├── test_worker.py              Worker lifecycle and pause/resume
├── test_streaming.py           Streaming engine (mocked WhisperX)
├── test_formatters.py          Output format correctness
├── test_pauses.py              Pause marker injection
├── test_acoustic.py            Acoustic analysis (mocked librosa)
├── test_emotion_registry.py    SER model registry
├── test_model_registry.py      Unified model catalogue
├── test_speakers.py            Speaker snippet extraction
├── test_server.py              FastAPI endpoint tests
├── test_ai.py                  AI provider abstraction
└── test_integration.py         End-to-end (requires HF_TOKEN, @integration)
```

---

## Hardware acceleration

| Platform | Backend | Notes |
|----------|---------|-------|
| Apple Silicon (M1–M4) | mlx-whisper | Native Metal, installed automatically |
| NVIDIA CUDA | WhisperX float16 | Detected automatically |
| CPU fallback | WhisperX int8 | Works everywhere |

The `GET /info` endpoint reports the active backend.

---

## Keyboard shortcuts

| Action | macOS | Windows / Linux |
|--------|-------|----------------|
| Transcribe | ⌘1 | Ctrl+1 |
| Live | ⌘2 | Ctrl+2 |
| Speakers | ⌘3 | Ctrl+3 |
| Settings | ⌘, | Ctrl+, |

---

## Ports

| Service | URL |
|---------|-----|
| Web UI | http://127.0.0.1:7860 |
| REST API | http://127.0.0.1:7861 |
| Swagger docs | http://127.0.0.1:7861/docs |

---

## License

MIT
