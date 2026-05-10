# WhisperApp — headless API server
#
# Exposes the full REST API on port 7861. No tray icon, no browser UI.
# All transcription, diarization, speaker review, AI features, and model
# management are accessible via the API. See README for endpoint reference.
#
# Build:
#   docker build -t whisperapp .
#
# Run (CPU):
#   docker run -p 7861:7861 \
#     -e HF_TOKEN=hf_... \
#     -v whisperapp-models:/root/.cache \
#     -v whisperapp-data:/data \
#     whisperapp
#
# Run (NVIDIA GPU):
#   docker run --gpus all -p 7861:7861 \
#     -e HF_TOKEN=hf_... \
#     -v whisperapp-models:/root/.cache \
#     -v whisperapp-data:/data \
#     whisperapp

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────
# ffmpeg   — audio decoding for whisperx
# git      — some pip packages clone at install time
# libgomp1 — OpenMP (torch)
# Pango/fonts — weasyprint PDF rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgomp1 \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libfontconfig1 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
# Install CPU-only PyTorch first to avoid pulling the 3 GB CUDA wheel.
# whisperx pins torch >= 2.0, so this satisfies it without CUDA.
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchaudio==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy only the package definition first so dependency layer is cached
# independently of source code changes.
COPY pyproject.toml ./
COPY whisperapp/ ./whisperapp/

# Install whisperapp in headless mode (no pystray, Pillow, pyaudio).
# The [desktop] extra is omitted intentionally.
RUN pip install --no-cache-dir -e "."

# ── Runtime configuration ──────────────────────────────────────────────────
# Models are downloaded to /root/.cache on first use — mount a volume there
# so they persist across container restarts.
# Output files default to /data — mount a host directory there to retrieve
# transcripts without copying from the container.

ENV WHISPERAPP_HEADLESS=1
ENV WHISPERAPP_OUTPUT_PATH=/data
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /data /root/.cache

EXPOSE 7861

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7861/health')"

ENTRYPOINT ["python", "-m", "whisperapp"]
