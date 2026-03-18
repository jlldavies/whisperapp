from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, field_validator
from typing import Optional, List
from whisperapp.queue import JobQueue, JobStatus
from whisperapp.sanitise import (sanitise_file_path, sanitise_output_path,
                                   sanitise_model, sanitise_job_id)
from pathlib import Path

ALLOWED_IPS = {"127.0.0.1", "::1", "testclient"}  # testclient for httpx tests

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

def create_app(queue: JobQueue, worker) -> FastAPI:
    app = FastAPI(title="WhisperApp MCP Server", version="1.0.0")

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

    return app
