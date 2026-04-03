import sqlite3
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone
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
                  datetime.now(timezone.utc).isoformat()))
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
                (JobStatus.DONE, str(result_path), datetime.now(timezone.utc).isoformat(), job_id)
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

    def clear_completed(self):
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM jobs WHERE status IN (?, ?, ?)",
                (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)
            )

    def next_queued(self) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at ASC LIMIT 1",
                (JobStatus.QUEUED,)
            ).fetchone()
        return dict(row) if row else None
