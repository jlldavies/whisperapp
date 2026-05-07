from unittest.mock import MagicMock, patch
from whisperapp.ui import create_ui, handle_submit
from whisperapp.queue import JobQueue


def test_ui_creates_without_error(tmp_path):
    with patch("whisperapp.ui._list_input_devices", return_value={}):
        q = JobQueue(db_path=tmp_path / "jobs.db")
        demo = create_ui(queue=q, worker=None)
        assert demo is not None


def test_submit_job_returns_job_id(tmp_path):
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


def test_live_tab_exists(tmp_path):
    with patch("whisperapp.ui._list_input_devices", return_value={}):
        q = JobQueue(db_path=tmp_path / "jobs.db")
        demo = create_ui(queue=q, worker=None)
        block_types = [type(b).__name__ for b in demo.blocks.values()]
        assert "Tab" in block_types
