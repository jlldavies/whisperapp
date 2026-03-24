from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, field_validator
from typing import Optional, List
import base64
import uuid
import numpy as np
from whisperapp.queue import JobQueue, JobStatus
from whisperapp.checkpoints import CheckpointManager
from whisperapp.sanitise import (sanitise_file_path, sanitise_output_path,
                                   sanitise_model, sanitise_job_id)
from whisperapp.streaming import StreamingEngine
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


def create_app(queue: JobQueue, worker) -> FastAPI:
    app = FastAPI(title="WhisperApp MCP Server", version="1.0.0")

    # In-memory streaming sessions (local-only server, so this is fine)
    _sessions: dict = {}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        client_ip = request.client.host if request.client else "127.0.0.1"
        if client_ip not in ALLOWED_IPS:
            return Response("Forbidden — local connections only", status_code=403)
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

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

    return app
