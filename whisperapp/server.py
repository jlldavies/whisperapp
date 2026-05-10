from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional, List
import base64
import json
import sys
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
    callback_url: Optional[str] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        allowed = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}
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

    # Shared ModelManager instance (emotion models)
    from whisperapp.emotion_registry import ModelManager
    _model_manager = ModelManager()

    # Unified model registry (all categories)
    from whisperapp.model_registry import ModelRegistry
    _model_registry = ModelRegistry()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:7860", "http://localhost:7860"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
        if req.callback_url:
            from whisperapp.worker import _webhook_host_allowed
            cfg = Config()
            if not _webhook_host_allowed(req.callback_url, cfg.webhook_allowed_hosts):
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "callback_url host is not allowed. "
                        "Set webhook_allowed_hosts in config to permit external hosts."
                    ),
                )
        job_id = queue.create_job(
            str(file_path), str(output_path),
            model, req.diarize, req.formats,
            callback_url=req.callback_url)
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

    @app.post("/jobs/clear")
    async def clear_completed_jobs():
        count = queue.clear_completed()
        return {"cleared": count}

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        try:
            sanitise_job_id(job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        queue.cancel_job(job_id)
        return {"success": True}

    @app.delete("/jobs/{job_id}")
    async def delete_job(job_id: str):
        """Permanently remove a job from the queue and history."""
        try:
            sanitise_job_id(job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        queue.delete_job(job_id)
        return {"success": True}

    @app.post("/jobs/{job_id}/move-up")
    async def move_job_up(job_id: str):
        """Move a queued job one position earlier in the queue."""
        try:
            sanitise_job_id(job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        queue.move_job_up(job_id)
        return {"success": True}

    @app.post("/jobs/{job_id}/move-down")
    async def move_job_down(job_id: str):
        """Move a queued job one position later in the queue."""
        try:
            sanitise_job_id(job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        queue.move_job_down(job_id)
        return {"success": True}

    @app.get("/worker/status")
    async def worker_status():
        """Return whether the worker is paused. Paused means queued jobs won't start."""
        return {"paused": worker.is_paused}

    @app.post("/worker/pause")
    async def pause_worker():
        """Pause the worker. The current job finishes; no new jobs start until resumed."""
        worker.pause()
        return {"paused": True}

    @app.post("/worker/resume")
    async def resume_worker():
        """Resume the worker after a pause."""
        worker.resume()
        return {"paused": False}

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

    @app.get("/jobs/{job_id}/segments")
    async def get_job_segments(
        job_id: str,
        offset: int = 0,
        limit: int = 200,
        include_words: bool = False,
    ):
        """Return diarized transcript segments page-by-page.

        Works for both `speaker_review` (loads the in-progress checkpoint) and
        `done` (loads the final checkpoint or saved JSON). Each segment is
        `{start, end, speaker, text}`; word-level timestamps are stripped by
        default to keep the response lean — set `include_words=true` to keep
        them. Useful for AI-driven speaker identification that needs full
        transcript context, not just a few snippets per speaker.
        """
        try:
            sanitise_job_id(job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        if limit < 1 or limit > 500:
            raise HTTPException(status_code=422, detail="limit must be 1..500")
        if offset < 0:
            raise HTTPException(status_code=422, detail="offset must be >= 0")
        job = queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] not in (JobStatus.SPEAKER_REVIEW, JobStatus.DONE):
            raise HTTPException(
                status_code=409,
                detail=f"Job has no transcript yet: {job['status']}",
            )
        cm = CheckpointManager(job["output_path"], job_id)
        # Try the speaker_review checkpoint first (most current view of the
        # diarized transcript); fall back to disk-saved JSON for done jobs
        # where the checkpoint may have been cleaned up.
        result = None
        try:
            result = cm.load("speaker_review")
        except Exception:
            result = None
        if result is None and job["status"] == JobStatus.DONE:
            stem = Path(job["file_path"]).stem
            json_path = Path(job["output_path"]) / f"{stem}.json"
            if json_path.exists():
                result = json.loads(json_path.read_text(encoding="utf-8"))
        if result is None:
            raise HTTPException(status_code=404, detail="Transcript data not available")

        all_segs = result.get("segments", []) or []
        total = len(all_segs)
        page = all_segs[offset : offset + limit]
        if not include_words:
            page = [{k: v for k, v in s.items() if k != "words"} for s in page]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "next_offset": offset + len(page) if offset + len(page) < total else None,
            "source_metadata": result.get("source_metadata") or {},
            "segments": page,
        }

    @app.get("/jobs/{job_id}/audio")
    async def get_job_audio(job_id: str):
        """Stream the source audio file for a job so the speaker-review UI
        can play snippets at specific timestamps. Only the file path stored
        on the job itself is allowed — never an arbitrary path from the
        client. Range requests are handled by FileResponse for seeking."""
        try:
            sanitise_job_id(job_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid job_id")
        job = queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        from fastapi.responses import FileResponse
        path = Path(job["file_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="Source audio missing on disk")
        # Common audio/video MIME hints — browsers will fall back gracefully.
        mime_map = {
            ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
            ".mp4": "video/mp4", ".mov": "video/quicktime",
            ".webm": "audio/webm", ".ogg": "audio/ogg", ".flac": "audio/flac",
        }
        media_type = mime_map.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(str(path), media_type=media_type, filename=path.name)

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
        _sessions.pop(req.session_id, None)
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
    # Emotion model endpoints
    # -----------------------------------------------------------------------

    @app.get("/models")
    async def list_models():
        """List all available emotion models with download status."""
        return _model_manager.list_all()

    # ── New unified catalogue endpoints (MUST be before /models/{model_id}) ──

    @app.get("/models/catalogue")
    async def get_models_catalogue():
        """Return the full model catalogue with state for all categories."""
        return _model_registry.list_catalogue()

    @app.get("/models/disk-summary")
    async def get_disk_summary():
        """Return disk usage by model category."""
        return _model_registry.get_disk_summary()

    @app.post("/models/check-updates")
    async def check_model_updates():
        """Force a fresh update check across all capabilities (and, in the
        future, models). Clears in-process caches so the next catalogue
        render has up-to-date `state == "update"` flags."""
        return _model_registry.force_check_updates()

    @app.post("/models/add-custom")
    async def add_custom_model():
        """Stub: custom model import from HuggingFace."""
        return {"status": "not_implemented", "message": "Custom model import coming soon"}

    @app.post("/models/{model_id}/activate")
    async def activate_model(model_id: str):
        """Set a model as active in its category."""
        result = _model_registry.activate_model(model_id)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
        return result

    @app.post("/models/{model_id}/deactivate")
    async def deactivate_model(model_id: str):
        """Remove a model from the active set."""
        result = _model_registry.deactivate_model(model_id)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
        return result

    @app.post("/models/{model_id}/update")
    async def update_model(model_id: str):
        """Trigger a model/capability update."""
        from whisperapp.emotion_registry import _REGISTRY_BY_ID
        if model_id in _REGISTRY_BY_ID:
            _model_manager.download(model_id)  # restart download = update
            return {"success": True, "message": f"Update started for {model_id}"}
        if _model_registry.is_capability(model_id):
            return _model_registry.update_capability(model_id)
        # HuggingFace-backed models — re-pull the latest revision in the
        # background. The catalogue will reflect `downloading` until done,
        # then `installed` (or `active`) once the new sha is saved.
        from whisperapp.model_registry import _hf_repo_for
        if _hf_repo_for(model_id):
            return _model_registry.update_hf_model(model_id)
        return {"status": "not_implemented"}

    # ── Storage ──────────────────────────────────────────────────────────────

    @app.post("/storage/reveal")
    async def reveal_storage():
        """Open the WhisperApp storage directory in Finder/Explorer."""
        import subprocess
        import sys
        path = str(Path.home() / ".whisperapp")
        Path(path).mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            return {"success": False, "error": str(e)}
        return {"success": True, "path": path}

    @app.get("/templates/download-docx")
    async def download_docx_template():
        """Generate and return the editable DOCX template for download."""
        from fastapi.responses import FileResponse
        import tempfile
        from whisperapp.document_formats import generate_docx_template
        tmp = Path(tempfile.mktemp(suffix=".docx"))
        ok = generate_docx_template(tmp)
        if not ok:
            from fastapi import HTTPException
            raise HTTPException(503, "python-docx not installed")
        return FileResponse(
            str(tmp),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename="whisperapp_template.docx",
        )

    @app.get("/templates/download-html")
    async def download_html_template():
        """Return the default HTML template for PDF output."""
        from fastapi.responses import FileResponse
        from whisperapp.document_formats import _PKG_TEMPLATES
        html_path = _PKG_TEMPLATES / "default.html"
        if not html_path.exists():
            from fastapi import HTTPException
            raise HTTPException(404, "Template not found")
        return FileResponse(str(html_path), media_type="text/html",
                            filename="whisperapp_pdf_template.html")

    @app.get("/templates/download-legal-html")
    async def download_legal_html_template():
        """Return the legal HTML template for formal/chain-of-custody PDF output."""
        from fastapi.responses import FileResponse
        from whisperapp.document_formats import _PKG_TEMPLATES
        html_path = _PKG_TEMPLATES / "legal.html"
        if not html_path.exists():
            from fastapi import HTTPException
            raise HTTPException(404, "Template not found")
        return FileResponse(str(html_path), media_type="text/html",
                            filename="whisperapp_legal_template.html")

    # ── Startup registration (launch on login) ──────────────────────────────

    @app.get("/startup/status")
    async def startup_status():
        """Return whether WhisperApp is registered to launch at login on
        this machine. Cross-platform — uses the HKCU Run key on Windows and
        the LaunchAgent plist on macOS."""
        from whisperapp.startup import is_startup_registered
        return {"registered": is_startup_registered(), "platform": sys.platform}

    @app.post("/startup/enable")
    async def startup_enable():
        from whisperapp.startup import register_startup, is_startup_registered
        try:
            register_startup()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"register failed: {e}")
        return {"registered": is_startup_registered()}

    @app.post("/startup/disable")
    async def startup_disable():
        from whisperapp.startup import unregister_startup, is_startup_registered
        try:
            unregister_startup()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"unregister failed: {e}")
        return {"registered": is_startup_registered()}

    @app.get("/models/{model_id}")
    async def get_model(model_id: str):
        """Get status for a single emotion model."""
        from whisperapp.emotion_registry import _REGISTRY_BY_ID
        if model_id not in _REGISTRY_BY_ID:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        status = _model_manager.get_status(model_id)
        entry = dict(_REGISTRY_BY_ID[model_id])
        entry.update(status)
        return entry

    @app.post("/models/{model_id}/download")
    async def download_model(model_id: str):
        """Start downloading / installing a model or capability (returns
        immediately; the install runs in a background thread and progress is
        reflected in /models/catalogue)."""
        from whisperapp.emotion_registry import _REGISTRY_BY_ID
        if model_id in _REGISTRY_BY_ID:
            _model_manager.download(model_id)
            return {"success": True, "message": f"Download started for {model_id}"}
        if _model_registry.is_capability(model_id):
            return _model_registry.install_capability(model_id)
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    @app.delete("/models/{model_id}")
    async def delete_model(model_id: str):
        """Uninstall a downloaded emotion model or pip capability."""
        from whisperapp.emotion_registry import _REGISTRY_BY_ID
        if model_id in _REGISTRY_BY_ID:
            _model_manager.delete(model_id)
            return {"success": True}
        if _model_registry.is_capability(model_id):
            return _model_registry.uninstall_capability(model_id)
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    @app.get("/models/{model_id}/progress")
    async def model_progress(model_id: str):
        """Get download progress for an emotion model."""
        from whisperapp.emotion_registry import _REGISTRY_BY_ID
        if model_id not in _REGISTRY_BY_ID:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        status = _model_manager.get_status(model_id)
        return {
            "progress": status["progress"],
            "downloading": status["downloading"],
            "error": status["error"],
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
            "auto_update": cfg.auto_update,
            "pause_detection": cfg.pause_detection,
            "pause_threshold": cfg.pause_threshold,
            "long_pause_threshold": cfg.long_pause_threshold,
            "gap_threshold": cfg.gap_threshold,
            # Acoustic features
            "acoustic_enabled": cfg.acoustic_enabled,
            "acoustic_volume_enabled": cfg.acoustic_volume_enabled,
            "acoustic_volume_threshold": cfg.acoustic_volume_threshold,
            "acoustic_pitch_enabled": cfg.acoustic_pitch_enabled,
            "acoustic_pitch_threshold": cfg.acoustic_pitch_threshold,
            "acoustic_rate_enabled": cfg.acoustic_rate_enabled,
            "acoustic_rate_fast_wpm": cfg.acoustic_rate_fast_wpm,
            "acoustic_rate_slow_wpm": cfg.acoustic_rate_slow_wpm,
            # Emotion analysis
            "emotion_enabled": cfg.emotion_enabled,
            "emotion_model_ids": cfg.emotion_model_ids,
            "emotion_confidence_threshold": cfg.emotion_confidence_threshold,
            "emotion_combine_with_ai": cfg.emotion_combine_with_ai,
        }

    @app.post("/config")
    async def update_config(req: Request):
        data = await req.json()
        cfg = Config()  # noqa: uses module-level Config, patchable via whisperapp.server.Config
        allowed = {
            "hf_token", "default_model", "default_output_path",
            "diarize_by_default", "streaming_model",
            "ai_provider", "ai_api_key", "ai_model", "ai_base_url",
            "auto_update",
            "pause_detection", "pause_threshold",
            "long_pause_threshold", "gap_threshold",
            # Acoustic features
            "acoustic_enabled", "acoustic_volume_enabled", "acoustic_volume_threshold",
            "acoustic_pitch_enabled", "acoustic_pitch_threshold",
            "acoustic_rate_enabled", "acoustic_rate_fast_wpm", "acoustic_rate_slow_wpm",
            # Emotion analysis
            "emotion_enabled", "emotion_model_ids",
            "emotion_confidence_threshold", "emotion_combine_with_ai",
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
        ALLOWED_UPLOAD_SUFFIXES = {
            '.mp3', '.wav', '.m4a', '.mp4', '.mov', '.flac', '.ogg', '.webm', '.mkv'
        }
        suffix = Path(file.filename).suffix.lower() if file.filename else '.audio'
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            suffix = '.audio'
        tmp_path = uploads_dir / (uuid.uuid4().hex + suffix)
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"path": str(tmp_path)}

    return app
