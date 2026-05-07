from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, field_validator
from typing import Optional, List
import base64
import uuid
import shutil
import numpy as np
from whisperapp.queue import JobQueue, JobStatus
from whisperapp.checkpoints import CheckpointManager
from whisperapp.sanitise import (sanitise_file_path, sanitise_output_path,
                                   sanitise_model, sanitise_job_id)
from whisperapp.streaming import StreamingEngine
from whisperapp.ui import list_audio_devices
from whisperapp.config import _config_dir, Config
from pathlib import Path

ALLOWED_IPS = {"127.0.0.1", "::1", "testclient"}  # testclient for httpx tests

STREAMING_MODELS = {"tiny", "base", "small"}

class SpeakerReviewRequest(BaseModel):
    names: dict[str, str] = {}  # e.g. {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}


class TranscribeRequest(BaseModel):
    file_path: str
    output_path: Optional[str] = None
    model: str = "large-v2"
    diarize: bool = True
    formats: List[str] = ["txt", "srt", "vtt", "json"]

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        allowed = {"tiny", "base", "small", "medium", "large-v2"}
        if v not in allowed:
            raise ValueError(f"model must be one of {allowed}")
        return v


class StreamStartRequest(BaseModel):
    model: str = "base"
    sample_rate: int = 16000

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        if v not in STREAMING_MODELS:
            raise ValueError(f"model must be one of {STREAMING_MODELS}")
        return v


class StreamChunkRequest(BaseModel):
    session_id: str
    audio_b64: str
    sample_rate: int = 16000


class StreamStopRequest(BaseModel):
    session_id: str


class StreamPolishRequest(BaseModel):
    session_id: str
    hf_token: str = ""


class AIIdentifySpeakersRequest(BaseModel):
    job_id: str
    context: str = ""   # e.g. "Weekly standup between Alice, Bob, Carol"


class AIMeetingNotesRequest(BaseModel):
    job_id: str
    context: str = ""


class AILiveSummaryRequest(BaseModel):
    transcript: str
    context: str = ""


def create_app(queue: JobQueue, worker) -> FastAPI:
    app = FastAPI(title="WhisperApp API", version="1.1.0")

    # In-memory streaming sessions (local-only server, so this is fine)
    _sessions: dict = {}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        client_ip = request.client.host if request.client else "127.0.0.1"
        if client_ip not in ALLOWED_IPS:
            return Response("Forbidden — local connections only", status_code=403)
        return await call_next(request)

    def _ai():
        from whisperapp.config import Config
        from whisperapp.ai import make_provider
        cfg = Config()
        return make_provider(cfg.ai_provider, cfg.ai_api_key, cfg.ai_model, cfg.ai_base_url)

    @app.get("/health")
    async def health():
        ai = _ai()
        return {
            "status": "ok",
            "ai_provider": ai.name,
            "ai_available": ai.is_available(),
        }

    @app.get("/info")
    async def info():
        """Runtime information: device selection, acceleration, versions."""
        import sys
        import torch
        from whisperapp.worker import (
            _WHISPER_DEVICE, _DIARIZE_DEVICE, _COMPUTE_TYPE, _has_mlx_whisper
        )
        return {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "whisper_device": _WHISPER_DEVICE,
            "diarize_device": _DIARIZE_DEVICE,
            "compute_type": _COMPUTE_TYPE,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ),
            "mlx_whisper_available": _has_mlx_whisper(),
            "acceleration": (
                "CUDA" if _WHISPER_DEVICE == "cuda"
                else "MLX (Apple Silicon)" if _has_mlx_whisper()
                else "MPS (diarization only)" if _DIARIZE_DEVICE == "mps"
                else "CPU"
            ),
        }

    @app.post("/transcribe")
    async def transcribe(req: TranscribeRequest):
        try:
            file_path = sanitise_file_path(req.file_path)
            output_path = sanitise_output_path(
                req.output_path or str(Path.home() / "Downloads"))
            model = sanitise_model(req.model)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        job_id = queue.create_job(
            str(file_path), str(output_path),
            model, req.diarize, req.formats)
        return {"job_id": job_id}

    @app.get("/jobs")
    async def list_jobs(status: Optional[str] = None, limit: int = 20):
        return queue.list_jobs(status_filter=status, limit=limit)

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        try:
            sanitise_job_id(job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        job = queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        try:
            sanitise_job_id(job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        queue.cancel_job(job_id)
        return {"success": True}

    @app.get("/jobs/{job_id}/transcript")
    async def get_transcript(job_id: str, format: str = "txt"):
        try:
            sanitise_job_id(job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        job = queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != JobStatus.DONE:
            raise HTTPException(status_code=409, detail=f"Job not done: {job['status']}")
        output_dir = Path(job["output_path"])
        stem = Path(job["file_path"]).stem
        file = output_dir / f"{stem}.{format}"
        if not file.exists():
            raise HTTPException(status_code=404, detail=f"Format {format} not found")
        return {"content": file.read_text(encoding="utf-8"),
                "output_path": str(file)}

    # -----------------------------------------------------------------------
    # Speaker review endpoints
    # -----------------------------------------------------------------------

    @app.get("/jobs/{job_id}/speakers")
    async def get_speakers(job_id: str):
        """Get speaker snippets for a job awaiting speaker review."""
        try:
            sanitise_job_id(job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        job = queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != JobStatus.SPEAKER_REVIEW:
            raise HTTPException(status_code=409,
                                detail=f"Job not in speaker_review: {job['status']}")
        cm = CheckpointManager(job["output_path"], job_id)
        result = cm.load("speaker_review")
        from whisperapp.speakers import extract_speaker_snippets
        snippets = extract_speaker_snippets(result)
        return {"speakers": snippets}

    @app.post("/jobs/{job_id}/speakers")
    async def confirm_speakers(job_id: str, req: SpeakerReviewRequest):
        """Confirm speaker names and complete the job."""
        try:
            sanitise_job_id(job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        job = queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != JobStatus.SPEAKER_REVIEW:
            raise HTTPException(status_code=409,
                                detail=f"Job not in speaker_review: {job['status']}")
        cm = CheckpointManager(job["output_path"], job_id)
        result = cm.load("speaker_review")
        from whisperapp.speakers import apply_speaker_names
        renamed = apply_speaker_names(result, req.names)
        worker.complete_with_result(job_id, renamed)
        return {"success": True}

    # -----------------------------------------------------------------------
    # AI endpoints (all return 503 if no provider configured — never hard-fail)
    # -----------------------------------------------------------------------

    @app.get("/ai/status")
    async def ai_status():
        ai = _ai()
        return {"provider": ai.name, "available": ai.is_available()}

    @app.post("/ai/identify-speakers")
    async def ai_identify_speakers(req: AIIdentifySpeakersRequest):
        try:
            sanitise_job_id(req.job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        ai = _ai()
        if not ai.is_available():
            raise HTTPException(
                status_code=503,
                detail="No AI provider configured. Set ai_provider in Settings.")
        job = queue.get_job(req.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] not in (JobStatus.SPEAKER_REVIEW, JobStatus.DONE):
            raise HTTPException(status_code=409,
                                detail=f"Job not ready for speaker review: {job['status']}")
        cm = CheckpointManager(job["output_path"], req.job_id)
        result = cm.load("speaker_review")
        from whisperapp.speakers import extract_speaker_snippets
        snippets = extract_speaker_snippets(result)
        mapping = ai.identify_speakers(snippets, context=req.context)
        return {"mapping": mapping, "provider": ai.name}

    @app.post("/ai/meeting-notes")
    async def ai_meeting_notes(req: AIMeetingNotesRequest):
        try:
            sanitise_job_id(req.job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        ai = _ai()
        if not ai.is_available():
            raise HTTPException(status_code=503,
                                detail="No AI provider configured.")
        job = queue.get_job(req.job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != JobStatus.DONE:
            raise HTTPException(status_code=409,
                                detail=f"Job not done: {job['status']}")
        txt_file = Path(job["output_path"]) / f"{Path(job['file_path']).stem}.txt"
        if not txt_file.exists():
            raise HTTPException(status_code=404,
                                detail="txt transcript not found for this job")
        transcript = txt_file.read_text(encoding="utf-8")
        notes = ai.meeting_notes(transcript, context=req.context)
        if not notes:
            raise HTTPException(status_code=500,
                                detail="AI provider returned empty response")
        # Save alongside the transcript
        notes_file = txt_file.with_suffix(".notes.md")
        notes_file.write_text(notes, encoding="utf-8")
        return {"notes": notes, "saved_to": str(notes_file), "provider": ai.name}

    @app.post("/ai/live-summary")
    async def ai_live_summary(req: AILiveSummaryRequest):
        ai = _ai()
        if not ai.is_available():
            raise HTTPException(status_code=503,
                                detail="No AI provider configured.")
        summary = ai.live_summary(req.transcript)
        return {"summary": summary, "provider": ai.name}

    # -----------------------------------------------------------------------
    # Streaming endpoints
    # -----------------------------------------------------------------------

    @app.post("/stream/start")
    async def stream_start(req: StreamStartRequest):
        session_id = str(uuid.uuid4())
        engine = StreamingEngine(model_size=req.model)
        engine.start()
        _sessions[session_id] = {
            "engine": engine,
            "sample_rate": req.sample_rate,
            "stopped": False,
        }
        return {"session_id": session_id}

    @app.post("/stream/chunk")
    async def stream_chunk(req: StreamChunkRequest):
        session = _sessions.get(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if session["stopped"]:
            raise HTTPException(status_code=409, detail="Session already stopped")

        # Decode base64 audio (expected: float32 PCM bytes)
        try:
            raw = base64.b64decode(req.audio_b64)
            audio = np.frombuffer(raw, dtype=np.float32)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid audio data: {e}")

        new_text = session["engine"].process_chunk(req.sample_rate, audio)
        return {
            "new_text": new_text,
            "transcript": session["engine"].get_transcript(),
        }

    @app.post("/stream/stop")
    async def stream_stop(req: StreamStopRequest):
        session = _sessions.get(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        result = session["engine"].stop()
        session["stopped"] = True
        return {
            "text": result["text"],
            "segments": result["segments"],
        }

    @app.post("/stream/polish")
    async def stream_polish(req: StreamPolishRequest):
        session = _sessions.get(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if not session["stopped"]:
            raise HTTPException(status_code=409, detail="Stop session before polishing")

        polished = session["engine"].polish(req.hf_token)

        # Clean up session after polish
        del _sessions[req.session_id]

        return {
            "segments": polished.get("segments", []),
        }

    # -----------------------------------------------------------------------
    # Config endpoints
    # -----------------------------------------------------------------------

    @app.get("/config")
    async def get_config():
        cfg = Config()
        return {
            "hf_token": cfg.hf_token,
            "default_model": cfg.default_model,
            "default_output_path": cfg.default_output_path,
            "diarize_by_default": cfg.diarize_by_default,
            "streaming_model": cfg.streaming_model,
            "ai_provider": cfg.ai_provider,
            "ai_api_key": cfg.ai_api_key,
            "ai_model": cfg.ai_model,
            "ai_base_url": cfg.ai_base_url,
        }

    @app.post("/config")
    async def update_config(req: Request):
        data = await req.json()
        cfg = Config()  # noqa: uses module-level Config, patchable via whisperapp.server.Config
        allowed = {
            "hf_token", "default_model", "default_output_path",
            "diarize_by_default", "streaming_model",
            "ai_provider", "ai_api_key", "ai_model", "ai_base_url",
        }
        for k, v in data.items():
            if k in allowed:
                setattr(cfg, k, v)
        cfg.save()
        return {"success": True}

    # -----------------------------------------------------------------------
    # Audio devices
    # -----------------------------------------------------------------------

    @app.get("/audio/devices")
    async def get_audio_devices():
        return list_audio_devices()

    # -----------------------------------------------------------------------
    # File upload — saves to temp dir, returns path for /transcribe
    # -----------------------------------------------------------------------

    @app.post("/upload")
    async def upload_file(file: UploadFile):
        uploads_dir = _config_dir() / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename).suffix if file.filename else ".audio"
        tmp_path = uploads_dir / (uuid.uuid4().hex + suffix)
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"path": str(tmp_path)}

    return app
