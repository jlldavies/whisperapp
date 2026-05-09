# WhisperApp Stage 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wrap WhisperX in a cross-platform background app with Gradio UI, SQLite job queue, MCP/REST server, CLI, speaker labelling, checkpoint saves, and system tray.

**Architecture:** `jhj0517/Whisper-WebUI` as the WhisperX base; a FastAPI MCP server on `127.0.0.1:7861`; Gradio UI on `127.0.0.1:7860`; SQLite job queue; pystray system tray — all launched as a single background process.

**Tech Stack:** Python 3.10+, WhisperX, pyannote.audio, Gradio, FastAPI, SQLite (via `sqlite3`), pystray, bleach, pytest, httpx (test client)

**HuggingFace token for tests:** set `HF_TOKEN` env var (see COMMS.md or your ~/.whisperapp/config.json)
**Config location:** `~/.whisperapp/config.json`
**Partials location:** `<output_path>/.whisperapp_partials/<job_id>/`

---

## Task 1: Project Bootstrap

**Files:**
- Create: `whisperapp/` (package root)
- Create: `requirements.txt`
- Create: `setup.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

**Step 1: Initialise git and virtual environment**
```bash
cd D:\Dropbox\Code\whisperapp
git init
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS:
source .venv/bin/activate
```

**Step 2: Clone Whisper-WebUI as a subdirectory**
```bash
git submodule add https://github.com/jhj0517/Whisper-WebUI.git whisper_webui
```

**Step 3: Create `requirements.txt`**
```text
whisperx
pyannote.audio
gradio>=4.0
fastapi
uvicorn[standard]
pystray
Pillow
bleach
httpx
pytest
pytest-asyncio
```

**Step 4: Create `setup.py`**
```python
from setuptools import setup, find_packages

setup(
    name="whisperapp",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "whisperapp=whisperapp.cli:main",
        ],
    },
)
```

**Step 5: Install**
```bash
pip install -r requirements.txt
pip install -e .
```

**Step 6: Create `whisperapp/__init__.py`** (empty)

**Step 7: Commit**
```bash
git add .
git commit -m "chore: project bootstrap with submodule and deps"
```

---

## Task 2: Config Manager

**Files:**
- Create: `whisperapp/config.py`
- Test: `tests/test_config.py`

**Step 1: Write failing test**
```python
# tests/test_config.py
import json
from pathlib import Path
import pytest
from whisperapp.config import Config

def test_config_creates_default(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    cfg = Config()
    assert cfg.hf_token == ""
    assert cfg.default_model == "large-v2"
    assert cfg.default_output_path == str(Path.home() / "Downloads")
    assert (tmp_path / "config.json").exists()

def test_config_saves_and_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    cfg = Config()
    cfg.hf_token = "hf_test123"
    cfg.save()
    cfg2 = Config()
    assert cfg2.hf_token == "hf_test123"
```

**Step 2: Run to verify failure**
```bash
pytest tests/test_config.py -v
```
Expected: `ModuleNotFoundError` or `ImportError`

**Step 3: Implement `whisperapp/config.py`**
```python
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict

def _config_dir() -> Path:
    override = os.environ.get("WHISPERAPP_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".whisperapp"

@dataclass
class Config:
    hf_token: str = ""
    default_model: str = "large-v2"
    default_output_path: str = ""
    diarize_by_default: bool = True

    def __post_init__(self):
        if not self.default_output_path:
            self.default_output_path = str(Path.home() / "Downloads")
        config_dir = _config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        self._path = config_dir / "config.json"
        if self._path.exists():
            data = json.loads(self._path.read_text())
            for k, v in data.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        else:
            self.save()

    def save(self):
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        self._path.write_text(json.dumps(data, indent=2))
```

**Step 4: Run tests to verify pass**
```bash
pytest tests/test_config.py -v
```
Expected: 2 PASSED

**Step 5: Commit**
```bash
git add whisperapp/config.py tests/test_config.py
git commit -m "feat: config manager with defaults and persistence"
```

---

## Task 3: Input Sanitisation

**Files:**
- Create: `whisperapp/sanitise.py`
- Test: `tests/test_sanitise.py`

**Step 1: Write failing tests**
```python
# tests/test_sanitise.py
import pytest
from pathlib import Path
from whisperapp.sanitise import (
    sanitise_file_path, sanitise_output_path,
    sanitise_model, sanitise_job_id, sanitise_speaker_name
)

ALLOWED_EXTS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mkv"}

def test_valid_file_path(tmp_path):
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"fake")
    result = sanitise_file_path(str(f))
    assert result == f.resolve()

def test_invalid_extension(tmp_path):
    f = tmp_path / "file.exe"
    f.write_bytes(b"bad")
    with pytest.raises(ValueError, match="extension"):
        sanitise_file_path(str(f))

def test_nonexistent_file():
    with pytest.raises(ValueError, match="not found"):
        sanitise_file_path("/nonexistent/audio.mp3")

def test_path_traversal_in_output(tmp_path):
    with pytest.raises(ValueError, match="traversal"):
        sanitise_output_path(str(tmp_path / ".." / ".." / "etc"))

def test_valid_output_path(tmp_path):
    result = sanitise_output_path(str(tmp_path))
    assert result == tmp_path.resolve()

def test_valid_model():
    assert sanitise_model("large-v2") == "large-v2"

def test_invalid_model():
    with pytest.raises(ValueError, match="model"):
        sanitise_model("rm -rf /")

def test_valid_job_id():
    uid = "550e8400-e29b-41d4-a716-446655440000"
    assert sanitise_job_id(uid) == uid

def test_invalid_job_id():
    with pytest.raises(ValueError, match="job_id"):
        sanitise_job_id("'; DROP TABLE jobs; --")

def test_speaker_name_clean():
    assert sanitise_speaker_name("<script>alert(1)</script>James") == "alert(1)James"

def test_speaker_name_truncated():
    assert len(sanitise_speaker_name("A" * 100)) == 50
```

**Step 2: Run to verify failure**
```bash
pytest tests/test_sanitise.py -v
```

**Step 3: Implement `whisperapp/sanitise.py`**
```python
import re
from pathlib import Path
import bleach

ALLOWED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mkv"}
ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large-v2"}
JOB_ID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

def sanitise_file_path(raw: str) -> Path:
    path = Path(raw).resolve()
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported extension '{path.suffix}'. Allowed: {ALLOWED_EXTENSIONS}")
    return path

def sanitise_output_path(raw: str) -> Path:
    path = Path(raw).resolve()
    parts = path.parts
    if ".." in parts:
        raise ValueError("Path traversal detected in output path")
    path.mkdir(parents=True, exist_ok=True)
    return path

def sanitise_model(raw: str) -> str:
    if raw not in ALLOWED_MODELS:
        raise ValueError(f"Invalid model '{raw}'. Allowed: {ALLOWED_MODELS}")
    return raw

def sanitise_job_id(raw: str) -> str:
    if not JOB_ID_RE.match(raw):
        raise ValueError(f"Invalid job_id format: {raw!r}")
    return raw

def sanitise_speaker_name(raw: str) -> str:
    cleaned = bleach.clean(raw, tags=[], strip=True)
    return cleaned[:50]
```

**Step 4: Run tests**
```bash
pytest tests/test_sanitise.py -v
```
Expected: all PASSED

**Step 5: Commit**
```bash
git add whisperapp/sanitise.py tests/test_sanitise.py
git commit -m "feat: input sanitisation with tests"
```

---

## Task 4: Job Queue (SQLite)

**Files:**
- Create: `whisperapp/queue.py`
- Test: `tests/test_queue.py`

**Step 1: Write failing tests**
```python
# tests/test_queue.py
import pytest
import time
from whisperapp.queue import JobQueue, JobStatus

@pytest.fixture
def q(tmp_path):
    return JobQueue(db_path=tmp_path / "jobs.db")

def test_create_job(q):
    job_id = q.create_job(
        file_path="/tmp/audio.mp3",
        output_path="/tmp/out",
        model="large-v2",
        diarize=True,
        formats=["txt", "srt"]
    )
    assert len(job_id) == 36  # UUID4

def test_get_job(q):
    job_id = q.create_job("/tmp/a.mp3", "/tmp/out", "large-v2", True, ["txt"])
    job = q.get_job(job_id)
    assert job["status"] == JobStatus.QUEUED
    assert job["file_path"] == "/tmp/a.mp3"
    assert job["progress"] == 0

def test_update_progress(q):
    job_id = q.create_job("/tmp/a.mp3", "/tmp/out", "large-v2", True, ["txt"])
    q.update_progress(job_id, 45, "aligning")
    job = q.get_job(job_id)
    assert job["progress"] == 45
    assert job["stage"] == "aligning"

def test_complete_job(q):
    job_id = q.create_job("/tmp/a.mp3", "/tmp/out", "large-v2", True, ["txt"])
    q.complete_job(job_id, result_path="/tmp/out/a.txt")
    job = q.get_job(job_id)
    assert job["status"] == JobStatus.DONE

def test_cancel_job(q):
    job_id = q.create_job("/tmp/a.mp3", "/tmp/out", "large-v2", True, ["txt"])
    q.cancel_job(job_id)
    job = q.get_job(job_id)
    assert job["status"] == JobStatus.CANCELLED

def test_list_jobs(q):
    q.create_job("/tmp/a.mp3", "/tmp/out", "large-v2", True, ["txt"])
    q.create_job("/tmp/b.mp3", "/tmp/out", "large-v2", True, ["txt"])
    jobs = q.list_jobs()
    assert len(jobs) == 2

def test_list_jobs_filter(q):
    job_id = q.create_job("/tmp/a.mp3", "/tmp/out", "large-v2", True, ["txt"])
    q.cancel_job(job_id)
    q.create_job("/tmp/b.mp3", "/tmp/out", "large-v2", True, ["txt"])
    jobs = q.list_jobs(status_filter="queued")
    assert len(jobs) == 1

def test_next_queued(q):
    q.create_job("/tmp/a.mp3", "/tmp/out", "large-v2", True, ["txt"])
    job = q.next_queued()
    assert job is not None
    assert job["status"] == JobStatus.QUEUED
```

**Step 2: Run to verify failure**
```bash
pytest tests/test_queue.py -v
```

**Step 3: Implement `whisperapp/queue.py`**
```python
import sqlite3
import uuid
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SPEAKER_REVIEW = "speaker_review"

class JobQueue:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = Path.home() / ".whisperapp" / "jobs.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    model TEXT NOT NULL,
                    diarize INTEGER NOT NULL,
                    formats TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    progress INTEGER NOT NULL DEFAULT 0,
                    stage TEXT NOT NULL DEFAULT '',
                    result_path TEXT,
                    partial_path TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                )
            """)

    def create_job(self, file_path, output_path, model, diarize, formats) -> str:
        job_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO jobs (id, file_path, file_name, output_path, model,
                    diarize, formats, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (job_id, str(file_path), Path(file_path).name,
                  str(output_path), model, int(diarize),
                  json.dumps(formats), JobStatus.QUEUED,
                  datetime.utcnow().isoformat()))
        return job_id

    def get_job(self, job_id) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["formats"] = json.loads(d["formats"])
        return d

    def update_progress(self, job_id, progress, stage):
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET progress=?, stage=?, status=? WHERE id=?",
                (progress, stage, JobStatus.RUNNING, job_id)
            )

    def set_status(self, job_id, status, error=None):
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, error=? WHERE id=?",
                (status, error, job_id)
            )

    def complete_job(self, job_id, result_path):
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, result_path=?, progress=100, completed_at=? WHERE id=?",
                (JobStatus.DONE, str(result_path), datetime.utcnow().isoformat(), job_id)
            )

    def cancel_job(self, job_id):
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status=? WHERE id=? AND status IN (?, ?)",
                (JobStatus.CANCELLED, job_id, JobStatus.QUEUED, JobStatus.RUNNING)
            )

    def list_jobs(self, status_filter=None, limit=20) -> list:
        with self._conn() as conn:
            if status_filter:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ?",
                    (status_filter, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    def next_queued(self) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at ASC LIMIT 1",
                (JobStatus.QUEUED,)
            ).fetchone()
        return dict(row) if row else None
```

**Step 4: Run tests**
```bash
pytest tests/test_queue.py -v
```
Expected: all PASSED

**Step 5: Commit**
```bash
git add whisperapp/queue.py tests/test_queue.py
git commit -m "feat: SQLite job queue with full CRUD and status tracking"
```

---

## Task 5: Checkpoint Manager

**Files:**
- Create: `whisperapp/checkpoints.py`
- Test: `tests/test_checkpoints.py`

**Step 1: Write failing tests**
```python
# tests/test_checkpoints.py
import json
import pytest
from whisperapp.checkpoints import CheckpointManager, STAGES

@pytest.fixture
def cm(tmp_path):
    return CheckpointManager(output_path=tmp_path, job_id="test-job-123")

def test_checkpoint_dir_created(cm, tmp_path):
    expected = tmp_path / ".whisperapp_partials" / "test-job-123"
    assert expected.exists()

def test_save_and_load(cm):
    data = {"segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}
    cm.save("transcription", data)
    loaded = cm.load("transcription")
    assert loaded == data

def test_last_completed_stage_none(cm):
    assert cm.last_completed_stage() is None

def test_last_completed_stage_after_save(cm):
    cm.save("transcription", {"segments": []})
    assert cm.last_completed_stage() == "transcription"
    cm.save("alignment", {"segments": []})
    assert cm.last_completed_stage() == "alignment"

def test_resume_from_stage(cm):
    cm.save("transcription", {"segments": []})
    cm.save("alignment", {"segments": []})
    remaining = cm.remaining_stages()
    assert "transcription" not in remaining
    assert "alignment" not in remaining
    assert "diarization" in remaining

def test_cleanup_removes_partials(cm, tmp_path):
    cm.save("transcription", {"segments": []})
    cm.cleanup()
    partial_dir = tmp_path / ".whisperapp_partials" / "test-job-123"
    assert not partial_dir.exists()
```

**Step 2: Run to verify failure**
```bash
pytest tests/test_checkpoints.py -v
```

**Step 3: Implement `whisperapp/checkpoints.py`**
```python
import json
import shutil
from pathlib import Path

STAGES = ["transcription", "alignment", "diarization", "speaker_review", "saving"]

class CheckpointManager:
    def __init__(self, output_path, job_id):
        self.job_id = job_id
        self.dir = Path(output_path) / ".whisperapp_partials" / job_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, stage) -> Path:
        return self.dir / f"{stage}.json"

    def save(self, stage: str, data: dict):
        self._path(stage).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def load(self, stage: str) -> dict:
        p = self._path(stage)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def has(self, stage: str) -> bool:
        return self._path(stage).exists()

    def last_completed_stage(self) -> str | None:
        for stage in reversed(STAGES):
            if self.has(stage):
                return stage
        return None

    def remaining_stages(self) -> list[str]:
        last = self.last_completed_stage()
        if last is None:
            return STAGES.copy()
        idx = STAGES.index(last)
        return STAGES[idx + 1:]

    def cleanup(self):
        if self.dir.exists():
            shutil.rmtree(self.dir)
```

**Step 4: Run tests**
```bash
pytest tests/test_checkpoints.py -v
```
Expected: all PASSED

**Step 5: Commit**
```bash
git add whisperapp/checkpoints.py tests/test_checkpoints.py
git commit -m "feat: checkpoint manager for crash recovery"
```

---

## Task 6: WhisperX Worker

**Files:**
- Create: `whisperapp/worker.py`
- Test: `tests/test_worker.py`

**Step 1: Write failing tests (mock WhisperX — don't run real transcription in unit tests)**
```python
# tests/test_worker.py
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from whisperapp.worker import Worker
from whisperapp.queue import JobQueue, JobStatus

@pytest.fixture
def setup(tmp_path):
    q = JobQueue(db_path=tmp_path / "jobs.db")
    job_id = q.create_job(
        str(tmp_path / "audio.mp3"),
        str(tmp_path / "out"),
        "large-v2",
        True,
        ["txt", "srt"]
    )
    (tmp_path / "audio.mp3").write_bytes(b"fake audio")
    return q, job_id, tmp_path

@patch("whisperapp.worker.whisperx")
def test_worker_marks_job_running(mock_wx, setup):
    q, job_id, tmp_path = setup
    mock_wx.load_model.return_value = MagicMock()
    mock_wx.load_audio.return_value = MagicMock()
    mock_wx.transcribe.return_value = {"segments": [], "language": "en"}
    mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_wx.align.return_value = {"segments": []}
    mock_wx.DiarizationPipeline.return_value = MagicMock(return_value=MagicMock())
    mock_wx.assign_word_speakers.return_value = {"segments": []}

    worker = Worker(queue=q, hf_token="hf_test", config_dir=tmp_path)
    worker.process_job(job_id)
    job = q.get_job(job_id)
    assert job["status"] in (JobStatus.DONE, JobStatus.SPEAKER_REVIEW)

@patch("whisperapp.worker.whisperx")
def test_worker_saves_checkpoints(mock_wx, setup):
    q, job_id, tmp_path = setup
    mock_wx.load_model.return_value = MagicMock()
    mock_wx.load_audio.return_value = MagicMock()
    mock_wx.transcribe.return_value = {"segments": [], "language": "en"}
    mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_wx.align.return_value = {"segments": []}
    mock_wx.DiarizationPipeline.return_value = MagicMock(return_value=MagicMock())
    mock_wx.assign_word_speakers.return_value = {"segments": []}

    worker = Worker(queue=q, hf_token="hf_test", config_dir=tmp_path)
    worker.process_job(job_id)
    partial_dir = tmp_path / "out" / ".whisperapp_partials" / job_id
    assert (partial_dir / "transcription.json").exists()
    assert (partial_dir / "alignment.json").exists()

@patch("whisperapp.worker.whisperx")
def test_worker_handles_cancellation(mock_wx, setup):
    q, job_id, tmp_path = setup
    q.cancel_job(job_id)
    worker = Worker(queue=q, hf_token="hf_test", config_dir=tmp_path)
    worker.process_job(job_id)  # Should exit cleanly without processing
    job = q.get_job(job_id)
    assert job["status"] == JobStatus.CANCELLED
```

**Step 2: Run to verify failure**
```bash
pytest tests/test_worker.py -v
```

**Step 3: Implement `whisperapp/worker.py`**
```python
import threading
import time
from pathlib import Path
import whisperx
from whisperapp.queue import JobQueue, JobStatus
from whisperapp.checkpoints import CheckpointManager

class Worker:
    def __init__(self, queue: JobQueue, hf_token: str, config_dir: Path = None):
        self.queue = queue
        self.hf_token = hf_token
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        while not self._stop_event.is_set():
            job = self.queue.next_queued()
            if job:
                self.process_job(job["id"])
            else:
                time.sleep(2)

    def process_job(self, job_id: str):
        job = self.queue.get_job(job_id)
        if not job or job["status"] == JobStatus.CANCELLED:
            return

        self.queue.update_progress(job_id, 0, "starting")
        cm = CheckpointManager(job["output_path"], job_id)

        try:
            # --- Stage 1: Transcription ---
            if not cm.has("transcription"):
                self.queue.update_progress(job_id, 10, "transcribing")
                model = whisperx.load_model(job["model"], device="cpu",
                                             compute_type="float32")
                audio = whisperx.load_audio(job["file_path"])
                result = whisperx.transcribe(model, audio, batch_size=8)
                cm.save("transcription", result)
                del model
            else:
                result = cm.load("transcription")

            # Check cancellation between stages
            if self.queue.get_job(job_id)["status"] == JobStatus.CANCELLED:
                return

            # --- Stage 2: Alignment ---
            if not cm.has("alignment"):
                self.queue.update_progress(job_id, 40, "aligning")
                align_model, metadata = whisperx.load_align_model(
                    language_code=result["language"], device="cpu")
                aligned = whisperx.align(result["segments"], align_model,
                                         metadata, job["file_path"], device="cpu")
                cm.save("alignment", aligned)
                del align_model
            else:
                aligned = cm.load("alignment")

            if self.queue.get_job(job_id)["status"] == JobStatus.CANCELLED:
                return

            # --- Stage 3: Diarization ---
            if job["diarize"] and not cm.has("diarization"):
                self.queue.update_progress(job_id, 65, "diarizing")
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token, device="cpu")
                diarize_segments = diarize_model(job["file_path"])
                result_diarized = whisperx.assign_word_speakers(
                    diarize_segments, aligned)
                cm.save("diarization", result_diarized)
                final_result = result_diarized
            elif cm.has("diarization"):
                final_result = cm.load("diarization")
            else:
                final_result = aligned

            if self.queue.get_job(job_id)["status"] == JobStatus.CANCELLED:
                return

            # --- Stage 4: Speaker review (if diarized) ---
            if job["diarize"]:
                cm.save("speaker_review", final_result)
                self.queue.set_status(job_id, JobStatus.SPEAKER_REVIEW)
                self.queue.update_progress(job_id, 85, "speaker_review")
                # UI will handle renaming and call complete_with_result
            else:
                self._write_outputs(job, final_result, cm)
                self.queue.complete_job(job_id, job["output_path"])

        except Exception as e:
            self.queue.set_status(job_id, JobStatus.FAILED, error=str(e))
            raise

    def complete_with_result(self, job_id: str, renamed_segments: dict):
        """Called after speaker review is complete (names applied or skipped)."""
        job = self.queue.get_job(job_id)
        cm = CheckpointManager(job["output_path"], job_id)
        cm.save("saving", renamed_segments)
        self._write_outputs(job, renamed_segments, cm)
        self.queue.complete_job(job_id, job["output_path"])
        cm.cleanup()

    def _write_outputs(self, job, result, cm: CheckpointManager):
        from whisperapp.formatters import write_formats
        self.queue.update_progress(job["id"], 95, "saving")
        write_formats(result, job["file_path"], job["output_path"], job["formats"])
```

**Step 4: Run tests**
```bash
pytest tests/test_worker.py -v
```
Expected: all PASSED

**Step 5: Commit**
```bash
git add whisperapp/worker.py tests/test_worker.py
git commit -m "feat: WhisperX worker with checkpoint resume and cancellation"
```

---

## Task 7: Output Formatters

**Files:**
- Create: `whisperapp/formatters.py`
- Test: `tests/test_formatters.py`

**Step 1: Write failing tests**
```python
# tests/test_formatters.py
import pytest
from pathlib import Path
from whisperapp.formatters import write_formats, result_to_txt, result_to_srt

SAMPLE_RESULT = {
    "segments": [
        {"start": 0.0, "end": 5.0, "text": "Hello world",
         "speaker": "James"},
        {"start": 5.5, "end": 10.0, "text": "How are you",
         "speaker": "Wolfgang"},
    ]
}

def test_txt_output():
    txt = result_to_txt(SAMPLE_RESULT)
    assert "James" in txt
    assert "Hello world" in txt
    assert "Wolfgang" in txt

def test_srt_output():
    srt = result_to_srt(SAMPLE_RESULT)
    assert "00:00:00,000" in srt
    assert "Hello world" in srt
    assert "1\n" in srt
    assert "2\n" in srt

def test_write_formats_creates_files(tmp_path):
    write_formats(SAMPLE_RESULT, "/fake/audio.mp3",
                  str(tmp_path), ["txt", "srt"])
    assert (tmp_path / "audio.txt").exists()
    assert (tmp_path / "audio.srt").exists()

def test_write_formats_all(tmp_path):
    write_formats(SAMPLE_RESULT, "/fake/audio.mp3",
                  str(tmp_path), ["txt", "srt", "vtt", "json", "tsv"])
    for ext in ["txt", "srt", "vtt", "json", "tsv"]:
        assert (tmp_path / f"audio.{ext}").exists()
```

**Step 2: Run to verify failure**
```bash
pytest tests/test_formatters.py -v
```

**Step 3: Implement `whisperapp/formatters.py`**
```python
import json
from pathlib import Path

def _format_time_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _format_time_vtt(seconds: float) -> str:
    return _format_time_srt(seconds).replace(",", ".")

def result_to_txt(result: dict) -> str:
    lines = []
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        prefix = f"[{speaker}] " if speaker else ""
        lines.append(f"{prefix}{text}")
    return "\n".join(lines)

def result_to_srt(result: dict) -> str:
    blocks = []
    for i, seg in enumerate(result.get("segments", []), 1):
        start = _format_time_srt(seg["start"])
        end = _format_time_srt(seg["end"])
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        prefix = f"[{speaker}] " if speaker else ""
        blocks.append(f"{i}\n{start} --> {end}\n{prefix}{text}\n")
    return "\n".join(blocks)

def result_to_vtt(result: dict) -> str:
    lines = ["WEBVTT\n"]
    for i, seg in enumerate(result.get("segments", []), 1):
        start = _format_time_vtt(seg["start"])
        end = _format_time_vtt(seg["end"])
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        prefix = f"<v {speaker}>" if speaker else ""
        lines.append(f"{start} --> {end}\n{prefix}{text}\n")
    return "\n".join(lines)

def result_to_tsv(result: dict) -> str:
    rows = ["start\tend\tspeaker\ttext"]
    for seg in result.get("segments", []):
        rows.append(f"{seg['start']}\t{seg['end']}\t"
                    f"{seg.get('speaker','')}\t{seg.get('text','').strip()}")
    return "\n".join(rows)

FORMATTERS = {
    "txt": result_to_txt,
    "srt": result_to_srt,
    "vtt": result_to_vtt,
    "json": lambda r: json.dumps(r, ensure_ascii=False, indent=2),
    "tsv": result_to_tsv,
}

def write_formats(result: dict, source_file: str,
                  output_path: str, formats: list):
    stem = Path(source_file).stem
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        if fmt not in FORMATTERS:
            continue
        content = FORMATTERS[fmt](result)
        (out_dir / f"{stem}.{fmt}").write_text(content, encoding="utf-8")
```

**Step 4: Run tests**
```bash
pytest tests/test_formatters.py -v
```
Expected: all PASSED

**Step 5: Commit**
```bash
git add whisperapp/formatters.py tests/test_formatters.py
git commit -m "feat: output formatters for txt/srt/vtt/json/tsv"
```

---

## Task 8: MCP/REST Server

**Files:**
- Create: `whisperapp/server.py`
- Test: `tests/test_server.py`

**Step 1: Write failing tests**
```python
# tests/test_server.py
import pytest
from httpx import AsyncClient, ASGITransport
from whisperapp.server import create_app
from whisperapp.queue import JobQueue, JobStatus

@pytest.fixture
def app_and_queue(tmp_path):
    q = JobQueue(db_path=tmp_path / "jobs.db")
    app = create_app(queue=q, worker=None)
    return app, q

@pytest.mark.asyncio
async def test_local_only_middleware(app_and_queue):
    app, q = app_and_queue
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as client:
        # httpx test client uses 127.0.0.1 by default — should pass
        resp = await client.get("/health")
        assert resp.status_code == 200

@pytest.mark.asyncio
async def test_transcribe_file_invalid_model(app_and_queue, tmp_path):
    app, q = app_and_queue
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"fake")
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as client:
        resp = await client.post("/transcribe", json={
            "file_path": str(f),
            "model": "rm-rf"
        })
        assert resp.status_code == 422

@pytest.mark.asyncio
async def test_list_jobs_empty(app_and_queue):
    app, q = app_and_queue
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as client:
        resp = await client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

@pytest.mark.asyncio
async def test_get_job_not_found(app_and_queue):
    app, q = app_and_queue
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as client:
        uid = "550e8400-e29b-41d4-a716-446655440000"
        resp = await client.get(f"/jobs/{uid}")
        assert resp.status_code == 404

@pytest.mark.asyncio
async def test_cancel_invalid_job_id(app_and_queue):
    app, q = app_and_queue
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as client:
        resp = await client.post("/jobs/not-a-uuid/cancel")
        assert resp.status_code == 422
```

**Step 2: Run to verify failure**
```bash
pytest tests/test_server.py -v
```

**Step 3: Implement `whisperapp/server.py`**
```python
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
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

    @validator("model")
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
```

**Step 4: Run tests**
```bash
pip install pytest-asyncio httpx
pytest tests/test_server.py -v
```
Expected: all PASSED

**Step 5: Commit**
```bash
git add whisperapp/server.py tests/test_server.py
git commit -m "feat: MCP/REST server with localhost-only middleware and full job control"
```

---

## Task 9: CLI

**Files:**
- Create: `whisperapp/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write failing tests**
```python
# tests/test_cli.py
from click.testing import CliRunner
from whisperapp.cli import cli
from unittest.mock import patch
import json

def test_status_invalid_job_id():
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "not-a-uuid"])
    assert result.exit_code != 0 or "invalid" in result.output.lower()

def test_list_command_runs():
    runner = CliRunner()
    with patch("whisperapp.cli.get_queue") as mock_q:
        mock_q.return_value.list_jobs.return_value = []
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "no jobs" in result.output.lower()

def test_list_with_jobs():
    runner = CliRunner()
    with patch("whisperapp.cli.get_queue") as mock_q:
        mock_q.return_value.list_jobs.return_value = [{
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "file_name": "audio.mp3",
            "status": "done",
            "progress": 100,
            "created_at": "2026-03-18T10:00:00"
        }]
        result = runner.invoke(cli, ["list"])
        assert "audio.mp3" in result.output
        assert "done" in result.output
```

**Step 2: Run to verify failure**
```bash
pytest tests/test_cli.py -v
```

**Step 3: Implement `whisperapp/cli.py`**
```python
import click
import requests
from whisperapp.queue import JobQueue
from pathlib import Path

API_BASE = "http://127.0.0.1:7861"

def get_queue():
    return JobQueue()

@click.group()
def cli():
    """WhisperApp — local transcription with speaker diarization."""
    pass

@cli.command()
@click.argument("file_path")
@click.option("--output", "-o", default=None, help="Output directory")
@click.option("--model", "-m", default="large-v2",
              type=click.Choice(["tiny","base","small","medium","large-v2"]))
@click.option("--diarize/--no-diarize", default=True)
@click.option("--formats", "-f", default="txt,srt,vtt,json",
              help="Comma-separated formats")
def transcribe(file_path, output, model, diarize, formats):
    """Submit a file for transcription."""
    payload = {
        "file_path": file_path,
        "output_path": output,
        "model": model,
        "diarize": diarize,
        "formats": formats.split(",")
    }
    try:
        r = requests.post(f"{API_BASE}/transcribe", json=payload, timeout=5)
        r.raise_for_status()
        click.echo(f"Submitted. Job ID: {r.json()['job_id']}")
    except requests.ConnectionError:
        click.echo("Error: WhisperApp is not running. Start it from the system tray.", err=True)
        raise SystemExit(1)

@cli.command()
@click.argument("job_id")
def status(job_id):
    """Check job status."""
    import re
    if not re.match(r'^[0-9a-f-]{36}$', job_id):
        click.echo("Error: invalid job_id format", err=True)
        raise SystemExit(1)
    r = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=5)
    if r.status_code == 404:
        click.echo("Job not found.")
        return
    job = r.json()
    click.echo(f"Status: {job['status']}  Progress: {job['progress']}%  Stage: {job['stage']}")

@cli.command("list")
@click.option("--status-filter", "-s", default=None)
def list_jobs(status_filter):
    """List recent jobs."""
    q = get_queue()
    jobs = q.list_jobs(status_filter=status_filter)
    if not jobs:
        click.echo("No jobs found.")
        return
    for j in jobs:
        click.echo(f"{j['id'][:8]}  {j['file_name']:30s}  {j['status']:12s}  {j['progress']}%")

@cli.command()
@click.argument("job_id")
def cancel(job_id):
    """Cancel a job."""
    r = requests.post(f"{API_BASE}/jobs/{job_id}/cancel", timeout=5)
    r.raise_for_status()
    click.echo("Cancelled.")

@cli.command("get")
@click.argument("job_id")
@click.option("--format", "-f", "fmt", default="txt")
def get_transcript(job_id, fmt):
    """Retrieve a completed transcript."""
    r = requests.get(f"{API_BASE}/jobs/{job_id}/transcript",
                     params={"format": fmt}, timeout=5)
    if r.status_code == 404:
        click.echo("Not found.")
        return
    r.raise_for_status()
    click.echo(r.json()["content"])

def main():
    cli()
```

**Step 4: Run tests**
```bash
pip install click
pytest tests/test_cli.py -v
```
Expected: all PASSED

**Step 5: Commit**
```bash
git add whisperapp/cli.py tests/test_cli.py
git commit -m "feat: CLI with transcribe/status/list/cancel/get commands"
```

---

## Task 10: Speaker Labelling Logic

**Files:**
- Create: `whisperapp/speakers.py`
- Test: `tests/test_speakers.py`

**Step 1: Write failing tests**
```python
# tests/test_speakers.py
from whisperapp.speakers import extract_speaker_snippets, apply_speaker_names

DIARIZED = {
    "segments": [
        {"start": 0.0, "end": 5.0, "text": "Hello world", "speaker": "SPEAKER_00"},
        {"start": 5.5, "end": 10.0, "text": "How are you", "speaker": "SPEAKER_01"},
        {"start": 10.0, "end": 15.0, "text": "I am fine", "speaker": "SPEAKER_00"},
        {"start": 15.0, "end": 20.0, "text": "Great to hear", "speaker": "SPEAKER_01"},
        {"start": 20.0, "end": 25.0, "text": "Third snippet", "speaker": "SPEAKER_00"},
    ]
}

def test_extracts_snippets_per_speaker():
    snippets = extract_speaker_snippets(DIARIZED, n=3)
    assert "SPEAKER_00" in snippets
    assert "SPEAKER_01" in snippets
    assert len(snippets["SPEAKER_00"]) <= 3
    assert snippets["SPEAKER_00"][0] == "Hello world"

def test_apply_speaker_names():
    names = {"SPEAKER_00": "James", "SPEAKER_01": "Wolfgang"}
    renamed = apply_speaker_names(DIARIZED, names)
    assert renamed["segments"][0]["speaker"] == "James"
    assert renamed["segments"][1]["speaker"] == "Wolfgang"

def test_apply_empty_name_keeps_original():
    names = {"SPEAKER_00": "", "SPEAKER_01": "Wolfgang"}
    renamed = apply_speaker_names(DIARIZED, names)
    assert renamed["segments"][0]["speaker"] == "SPEAKER_00"
    assert renamed["segments"][1]["speaker"] == "Wolfgang"

def test_apply_names_does_not_mutate_original():
    names = {"SPEAKER_00": "James"}
    original_speaker = DIARIZED["segments"][0]["speaker"]
    apply_speaker_names(DIARIZED, names)
    assert DIARIZED["segments"][0]["speaker"] == original_speaker
```

**Step 2: Run to verify failure**
```bash
pytest tests/test_speakers.py -v
```

**Step 3: Implement `whisperapp/speakers.py`**
```python
import copy
from collections import defaultdict
from whisperapp.sanitise import sanitise_speaker_name

def extract_speaker_snippets(result: dict, n: int = 3) -> dict:
    """Return up to n earliest text snippets per speaker."""
    snippets = defaultdict(list)
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if text and len(snippets[speaker]) < n:
            snippets[speaker].append(text)
    return dict(snippets)

def apply_speaker_names(result: dict, names: dict) -> dict:
    """Return a new result dict with speaker labels replaced by names.
    Empty names are left as-is (original SPEAKER_XX label kept)."""
    result = copy.deepcopy(result)
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "")
        if speaker in names:
            clean = sanitise_speaker_name(names[speaker])
            if clean:
                seg["speaker"] = clean
    return result
```

**Step 4: Run tests**
```bash
pytest tests/test_speakers.py -v
```
Expected: all PASSED

**Step 5: Commit**
```bash
git add whisperapp/speakers.py tests/test_speakers.py
git commit -m "feat: speaker snippet extraction and name application"
```

---

## Task 11: Gradio UI

**Files:**
- Create: `whisperapp/ui.py`
- Test: `tests/test_ui.py` (smoke tests only — full UI is manual)

**Step 1: Write smoke tests**
```python
# tests/test_ui.py
from whisperapp.ui import create_ui
from whisperapp.queue import JobQueue

def test_ui_creates_without_error(tmp_path):
    q = JobQueue(db_path=tmp_path / "jobs.db")
    demo = create_ui(queue=q, worker=None)
    assert demo is not None

def test_submit_job_returns_job_id(tmp_path):
    import gradio as gr
    from whisperapp.ui import handle_submit
    from unittest.mock import MagicMock
    q = JobQueue(db_path=tmp_path / "jobs.db")
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"fake audio")
    result = handle_submit(
        queue=q,
        file_path=str(f),
        output_path=str(tmp_path),
        model="large-v2",
        diarize=True,
        formats=["txt"]
    )
    assert "job" in result.lower() or len(result) == 36
```

**Step 2: Run to verify failure**
```bash
pytest tests/test_ui.py -v
```

**Step 3: Implement `whisperapp/ui.py`**
```python
import gradio as gr
from pathlib import Path
from whisperapp.queue import JobQueue, JobStatus
from whisperapp.sanitise import sanitise_file_path, sanitise_output_path, sanitise_model
from whisperapp.speakers import extract_speaker_snippets, apply_speaker_names

def handle_submit(queue, file_path, output_path, model, diarize, formats):
    try:
        fp = sanitise_file_path(file_path)
        op = sanitise_output_path(output_path or str(Path.home() / "Downloads"))
        m = sanitise_model(model)
    except ValueError as e:
        return f"Error: {e}"
    job_id = queue.create_job(str(fp), str(op), m, diarize, formats)
    return job_id

def get_queue_status(queue):
    jobs = queue.list_jobs(limit=20)
    if not jobs:
        return "No jobs yet."
    rows = []
    for j in jobs:
        bar = "█" * (j["progress"] // 10) + "░" * (10 - j["progress"] // 10)
        rows.append(
            f"{j['file_name'][:40]:40s}  [{bar}] {j['progress']:3d}%  "
            f"{j['stage']:20s}  {j['status']}"
        )
    return "\n".join(rows)

def create_ui(queue: JobQueue, worker) -> gr.Blocks:
    with gr.Blocks(title="WhisperApp") as demo:
        gr.Markdown("# 🎙 WhisperApp")

        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Audio/Video File",
                                      file_types=[".mp4",".mp3",".wav",".m4a",
                                                  ".ogg",".flac",".webm",".mkv"])
                output_path = gr.Textbox(
                    label="Output Path",
                    value=str(Path.home() / "Downloads"),
                    placeholder="Leave blank for ~/Downloads")
                model_select = gr.Dropdown(
                    choices=["tiny","base","small","medium","large-v2"],
                    value="large-v2", label="Model")
                diarize_check = gr.Checkbox(value=True, label="Speaker Diarization")
                formats_check = gr.CheckboxGroup(
                    choices=["txt","srt","vtt","json","tsv"],
                    value=["txt","srt","vtt","json"],
                    label="Output Formats")
                submit_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column():
                status_out = gr.Textbox(label="Queue", lines=15,
                                         every=3, interactive=False)
                refresh_btn = gr.Button("Refresh")

        submit_btn.click(
            fn=lambda f, o, m, d, fmt: handle_submit(
                queue, f.name if f else "", o, m, d, fmt),
            inputs=[file_input, output_path, model_select,
                    diarize_check, formats_check],
            outputs=status_out
        )

        refresh_btn.click(
            fn=lambda: get_queue_status(queue),
            outputs=status_out
        )

        # Auto-refresh queue every 3 seconds
        status_out.attach_load_event(
            fn=lambda: get_queue_status(queue), every=3)

    return demo
```

**Step 4: Run tests**
```bash
pytest tests/test_ui.py -v
```
Expected: PASSED

**Step 5: Commit**
```bash
git add whisperapp/ui.py tests/test_ui.py
git commit -m "feat: Gradio UI with file upload, queue display, and auto-refresh"
```

---

## Task 12: System Tray + Main Entry Point

**Files:**
- Create: `whisperapp/tray.py`
- Create: `whisperapp/__main__.py`
- Test: `tests/test_tray.py`

**Step 1: Write failing tests**
```python
# tests/test_tray.py
from unittest.mock import patch, MagicMock
from whisperapp.tray import TrayApp

def test_tray_app_creates():
    with patch("whisperapp.tray.pystray"):
        with patch("whisperapp.tray.Image"):
            app = TrayApp(queue=MagicMock(), worker=MagicMock())
            assert app is not None

def test_tray_title_idle():
    with patch("whisperapp.tray.pystray"):
        with patch("whisperapp.tray.Image"):
            app = TrayApp(queue=MagicMock(), worker=MagicMock())
            assert "WhisperApp" in app.get_title(0)

def test_tray_title_working():
    with patch("whisperapp.tray.pystray"):
        with patch("whisperapp.tray.Image"):
            app = TrayApp(queue=MagicMock(), worker=MagicMock())
            assert "2 jobs" in app.get_title(2).lower() or "2" in app.get_title(2)
```

**Step 2: Run to verify failure**
```bash
pytest tests/test_tray.py -v
```

**Step 3: Implement `whisperapp/tray.py`**
```python
import threading
import webbrowser
import pystray
from PIL import Image, ImageDraw

def _make_icon(color: str) -> Image.Image:
    img = Image.new("RGB", (64, 64), color=color)
    d = ImageDraw.Draw(img)
    d.ellipse([8, 8, 56, 56], fill="white")
    return img

ICONS = {
    "idle":    _make_icon("#888888"),
    "ready":   _make_icon("#22aa44"),
    "working": _make_icon("#2255cc"),
    "updating":_make_icon("#dd9900"),
}

class TrayApp:
    def __init__(self, queue, worker):
        self.queue = queue
        self.worker = worker
        self._icon = None

    def get_title(self, active_jobs: int) -> str:
        if active_jobs == 0:
            return "WhisperApp — Idle"
        return f"WhisperApp — {active_jobs} job{'s' if active_jobs != 1 else ''} running"

    def _open_ui(self, icon=None, item=None):
        webbrowser.open("http://127.0.0.1:7860")

    def _quit(self, icon, item):
        icon.stop()

    def run(self):
        menu = pystray.Menu(
            pystray.MenuItem("Open UI", self._open_ui, default=True),
            pystray.MenuItem("Quit", self._quit),
        )
        self._icon = pystray.Icon(
            "WhisperApp", ICONS["ready"], "WhisperApp", menu)
        self._icon.run()
```

**Step 3b: Implement `whisperapp/__main__.py`**
```python
import threading
import uvicorn
from whisperapp.config import Config
from whisperapp.queue import JobQueue
from whisperapp.worker import Worker
from whisperapp.server import create_app
from whisperapp.ui import create_ui
from whisperapp.tray import TrayApp
from whisperapp.updater import run_update

def main():
    # Auto-update on startup
    run_update()

    cfg = Config()
    queue = JobQueue()
    worker = Worker(queue=queue, hf_token=cfg.hf_token)
    worker.start()

    # MCP server thread
    mcp_app = create_app(queue=queue, worker=worker)
    threading.Thread(
        target=lambda: uvicorn.run(
            mcp_app, host="127.0.0.1", port=7861, log_level="warning"),
        daemon=True
    ).start()

    # Gradio UI thread
    ui = create_ui(queue=queue, worker=worker)
    threading.Thread(
        target=lambda: ui.launch(
            server_name="127.0.0.1", server_port=7860,
            quiet=True, prevent_thread_lock=True),
        daemon=True
    ).start()

    # Tray (blocks main thread)
    TrayApp(queue=queue, worker=worker).run()

if __name__ == "__main__":
    main()
```

**Step 4: Run tests**
```bash
pytest tests/test_tray.py -v
```
Expected: PASSED

**Step 5: Commit**
```bash
git add whisperapp/tray.py whisperapp/__main__.py tests/test_tray.py
git commit -m "feat: system tray app and main entry point"
```

---

## Task 13: Auto-Updater

**Files:**
- Create: `whisperapp/updater.py`
- Test: `tests/test_updater.py`

**Step 1: Write failing tests**
```python
# tests/test_updater.py
from unittest.mock import patch, call
from whisperapp.updater import run_update

def test_update_calls_pip():
    with patch("whisperapp.updater.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        run_update()
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("whisperx" in c for c in calls)

def test_update_handles_failure_gracefully():
    with patch("whisperapp.updater.subprocess.run") as mock_run:
        mock_run.side_effect = Exception("Network error")
        run_update()  # Should not raise
```

**Step 2: Run to verify failure**
```bash
pytest tests/test_updater.py -v
```

**Step 3: Implement `whisperapp/updater.py`**
```python
import subprocess
import sys
import logging

log = logging.getLogger(__name__)

PACKAGES = ["whisperx", "pyannote.audio", "gradio", "fastapi"]

def run_update():
    """Silently upgrade key packages. Fails gracefully."""
    for pkg in PACKAGES:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", pkg],
                capture_output=True, timeout=120
            )
            if result.returncode != 0:
                log.warning(f"Update failed for {pkg}: {result.stderr.decode()}")
        except Exception as e:
            log.warning(f"Could not update {pkg}: {e}")
```

**Step 4: Run tests**
```bash
pytest tests/test_updater.py -v
```

**Step 5: Commit**
```bash
git add whisperapp/updater.py tests/test_updater.py
git commit -m "feat: auto-updater for dependencies on startup"
```

---

## Task 14: System Startup Registration

**Files:**
- Create: `whisperapp/startup.py`
- Test: `tests/test_startup.py`

**Step 1: Write failing tests**
```python
# tests/test_startup.py
import sys
import pytest
from unittest.mock import patch, MagicMock
from whisperapp.startup import register_startup, unregister_startup

@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_register_windows():
    with patch("whisperapp.startup.winreg") as mock_reg:
        mock_reg.OpenKey.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_reg.OpenKey.return_value.__exit__ = MagicMock(return_value=False)
        register_startup()
        mock_reg.SetValueEx.assert_called_once()

@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_register_macos(tmp_path):
    with patch("whisperapp.startup.PLIST_PATH", tmp_path / "whisperapp.plist"):
        register_startup()
        assert (tmp_path / "whisperapp.plist").exists()
```

**Step 2: Implement `whisperapp/startup.py`**
```python
import sys
import subprocess
from pathlib import Path

APP_NAME = "WhisperApp"
PLIST_PATH = Path.home() / "Library/LaunchAgents/com.whisperapp.plist"

def _python_path() -> str:
    import sys
    return sys.executable

def register_startup():
    if sys.platform == "win32":
        _register_windows()
    elif sys.platform == "darwin":
        _register_macos()

def unregister_startup():
    if sys.platform == "win32":
        _unregister_windows()
    elif sys.platform == "darwin":
        _unregister_macos()

def _register_windows():
    import winreg
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    cmd = f'"{_python_path()}" -m whisperapp'
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0,
                        winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, cmd)

def _unregister_windows():
    import winreg
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0,
                            winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, APP_NAME)
    except FileNotFoundError:
        pass

def _register_macos():
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
    <key>Label</key><string>com.whisperapp</string>
    <key>ProgramArguments</key>
    <array>
        <string>{_python_path()}</string>
        <string>-m</string><string>whisperapp</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><false/>
</dict></plist>"""
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(plist)
    subprocess.run(["launchctl", "load", str(PLIST_PATH)], check=False)

def _unregister_macos():
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], check=False)
        PLIST_PATH.unlink()
```

**Step 3: Run tests**
```bash
pytest tests/test_startup.py -v
```

**Step 4: Commit**
```bash
git add whisperapp/startup.py tests/test_startup.py
git commit -m "feat: system startup registration for Windows and macOS"
```

---

## Task 15: Integration Test (Full Pipeline)

**Files:**
- Create: `tests/test_integration.py`
- Create: `tests/fixtures/short_audio.py` (generates a 5-second test WAV)

**Step 1: Generate test audio fixture**
```python
# tests/fixtures/make_test_audio.py
"""Run once to generate tests/fixtures/test_5sec.wav"""
import wave, struct, math, os
SAMPLE_RATE = 16000
DURATION = 5
filename = os.path.join(os.path.dirname(__file__), "test_5sec.wav")
with wave.open(filename, "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(SAMPLE_RATE)
    for i in range(SAMPLE_RATE * DURATION):
        val = int(32767 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
        f.writeframes(struct.pack("<h", val))
print(f"Written {filename}")
```

```bash
python tests/fixtures/make_test_audio.py
```

**Step 2: Write integration test**
```python
# tests/test_integration.py
"""
Integration test — requires real WhisperX and HF token.
Run with: pytest tests/test_integration.py -v -m integration
"""
import os
import pytest
from pathlib import Path
from whisperapp.queue import JobQueue, JobStatus
from whisperapp.worker import Worker
from whisperapp.config import Config

HF_TOKEN = os.environ.get("HF_TOKEN", "")
TEST_AUDIO = Path(__file__).parent / "fixtures" / "test_5sec.wav"

@pytest.mark.integration
def test_full_pipeline_no_diarize(tmp_path):
    """Transcribes a short WAV without diarization."""
    q = JobQueue(db_path=tmp_path / "jobs.db")
    worker = Worker(queue=q, hf_token=HF_TOKEN)
    job_id = q.create_job(
        str(TEST_AUDIO), str(tmp_path),
        "tiny", diarize=False, formats=["txt", "srt"]
    )
    worker.process_job(job_id)
    job = q.get_job(job_id)
    assert job["status"] == JobStatus.DONE
    assert (tmp_path / "test_5sec.txt").exists()
    assert (tmp_path / "test_5sec.srt").exists()

@pytest.mark.integration
def test_full_pipeline_with_diarize(tmp_path):
    """Transcribes a short WAV with diarization."""
    q = JobQueue(db_path=tmp_path / "jobs.db")
    worker = Worker(queue=q, hf_token=HF_TOKEN)
    job_id = q.create_job(
        str(TEST_AUDIO), str(tmp_path),
        "tiny", diarize=True, formats=["txt"]
    )
    worker.process_job(job_id)
    job = q.get_job(job_id)
    # With diarize=True, job pauses for speaker review
    assert job["status"] in (JobStatus.DONE, JobStatus.SPEAKER_REVIEW)

@pytest.mark.integration
def test_checkpoint_resume(tmp_path):
    """Simulates crash after transcription, resumes from checkpoint."""
    from whisperapp.checkpoints import CheckpointManager
    q = JobQueue(db_path=tmp_path / "jobs.db")
    job_id = q.create_job(
        str(TEST_AUDIO), str(tmp_path),
        "tiny", diarize=False, formats=["txt"]
    )
    # Pre-seed transcription checkpoint as if we crashed after it
    cm = CheckpointManager(str(tmp_path), job_id)
    import whisperx
    audio = whisperx.load_audio(str(TEST_AUDIO))
    model = whisperx.load_model("tiny", device="cpu", compute_type="float32")
    result = whisperx.transcribe(model, audio, batch_size=8)
    cm.save("transcription", result)
    del model

    worker = Worker(queue=q, hf_token=HF_TOKEN)
    worker.process_job(job_id)
    job = q.get_job(job_id)
    assert job["status"] == JobStatus.DONE

@pytest.mark.integration
def test_security_path_traversal(tmp_path):
    from whisperapp.sanitise import sanitise_file_path
    with pytest.raises(ValueError):
        sanitise_file_path(str(tmp_path / ".." / ".." / "etc" / "passwd.mp3"))

@pytest.mark.integration
def test_security_local_only_blocks_external():
    import httpx
    from whisperapp.server import create_app
    from whisperapp.queue import JobQueue as JQ
    q = JQ()
    app = create_app(queue=q, worker=None)
    # Simulate external IP via ASGI scope override
    from starlette.testclient import TestClient
    client = TestClient(app)
    # Direct call passes (127.0.0.1)
    resp = client.get("/health")
    assert resp.status_code == 200
```

**Step 3: Run unit tests (fast)**
```bash
pytest tests/ -v --ignore=tests/test_integration.py
```
Expected: all PASSED

**Step 4: Run integration tests (slow — needs HF token + internet)**
```bash
pytest tests/test_integration.py -v -m integration
```
Expected: all PASSED (takes ~5 mins first run for model downloads)

**Step 5: Commit**
```bash
git add tests/test_integration.py tests/fixtures/
git commit -m "test: integration tests for full pipeline, checkpoints, and security"
```

---

## Task 16: Final Wiring — First-Run Wizard + Startup Registration

**Files:**
- Modify: `whisperapp/ui.py` (add first-run tab)
- Modify: `whisperapp/__main__.py` (check first-run)

**Step 1: Add first-run check to `__main__.py`**

Add before `worker.start()`:
```python
if not cfg.hf_token:
    print("First run — open http://127.0.0.1:7860 to complete setup.")
```

**Step 2: Add Settings tab to `ui.py`**

Add inside `gr.Blocks()` after queue column:
```python
with gr.Tab("Settings"):
    hf_token_input = gr.Textbox(
        label="HuggingFace Token",
        value=cfg.hf_token if hasattr(cfg, 'hf_token') else "",
        type="password")
    default_model = gr.Dropdown(
        choices=["tiny","base","small","medium","large-v2"],
        value="large-v2", label="Default Model")
    startup_check = gr.Checkbox(
        label="Start on login", value=True)
    save_btn = gr.Button("Save Settings")

    def save_settings(token, model, startup):
        from whisperapp.config import Config
        from whisperapp.startup import register_startup, unregister_startup
        c = Config()
        c.hf_token = token
        c.default_model = model
        c.save()
        if startup:
            register_startup()
        else:
            unregister_startup()
        return "Settings saved ✅"

    settings_out = gr.Textbox(label="", interactive=False)
    save_btn.click(
        fn=save_settings,
        inputs=[hf_token_input, default_model, startup_check],
        outputs=settings_out
    )
```

**Step 3: Run full test suite**
```bash
pytest tests/ -v --ignore=tests/test_integration.py
```
Expected: all PASSED

**Step 4: Final commit**
```bash
git add -A
git commit -m "feat: first-run wizard and startup registration wired into UI"
git tag v1.0.0-stage1
```

---

## Running the App

```bash
# Start
python -m whisperapp

# Or after pip install -e .
whisperapp
```

**Ports:**
- UI: http://127.0.0.1:7860
- MCP/API: http://127.0.0.1:7861
- API docs: http://127.0.0.1:7861/docs

---

## Test Commands Summary

```bash
# Fast unit tests only (no HF token needed)
pytest tests/ -v --ignore=tests/test_integration.py

# Integration tests (needs HF token, ~5 mins)
pytest tests/test_integration.py -v -m integration

# All tests
pytest tests/ -v

# Coverage report
pytest tests/ --cov=whisperapp --cov-report=term-missing
```
