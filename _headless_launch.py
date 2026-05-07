"""Headless WhisperApp launcher - runs server only, no tray."""
import time
import threading
import uvicorn
import whisperapp.updater as _u
_u.run_update = lambda: None
from whisperapp.config import Config
from whisperapp.queue import JobQueue
from whisperapp.worker import Worker
from whisperapp.server import create_app

cfg = Config()
queue = JobQueue()
worker = Worker(queue=queue, hf_token=cfg.hf_token)
worker.start()
mcp_app = create_app(queue=queue, worker=worker)
print(f"WhisperApp REST starting on 127.0.0.1:7861 (HF token set: {bool(cfg.hf_token)})", flush=True)
uvicorn.run(mcp_app, host="127.0.0.1", port=7861, log_level="info")
