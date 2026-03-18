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
