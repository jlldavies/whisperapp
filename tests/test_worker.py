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
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": [], "language": "en"}
    mock_wx.load_model.return_value = mock_model
    mock_wx.load_audio.return_value = MagicMock()
    mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_wx.align.return_value = {"segments": []}
    mock_wx.assign_word_speakers.return_value = {"segments": []}

    with patch("whisperapp.worker.DiarizationPipeline") as mock_dp:
        mock_dp.return_value = MagicMock(return_value=MagicMock())
        worker = Worker(queue=q, hf_token="hf_test", config_dir=tmp_path)
        worker.process_job(job_id)
    job = q.get_job(job_id)
    assert job["status"] in (JobStatus.DONE, JobStatus.SPEAKER_REVIEW)

@patch("whisperapp.worker.whisperx")
def test_worker_saves_checkpoints(mock_wx, setup):
    q, job_id, tmp_path = setup
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"segments": [], "language": "en"}
    mock_wx.load_model.return_value = mock_model
    mock_wx.load_audio.return_value = MagicMock()
    mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_wx.align.return_value = {"segments": []}
    mock_wx.assign_word_speakers.return_value = {"segments": []}

    with patch("whisperapp.worker.DiarizationPipeline") as mock_dp:
        mock_dp.return_value = MagicMock(return_value=MagicMock())
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
