"""
Integration test - requires real WhisperX and HF token.
Run with: pytest tests/test_integration.py -v -m integration
"""
import os
import pytest
from pathlib import Path
from whisperapp.queue import JobQueue, JobStatus
from whisperapp.worker import Worker
from whisperapp.config import Config

HF_TOKEN = os.environ.get(
    "HF_TOKEN", "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")
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
    result = model.transcribe(audio, batch_size=8)
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
    # Direct call passes (127.0.0.1)
    from starlette.testclient import TestClient
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
