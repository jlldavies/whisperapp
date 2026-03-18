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
