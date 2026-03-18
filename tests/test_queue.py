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
