from unittest.mock import MagicMock, patch
import numpy as np
from whisperapp.ui import create_ui, handle_submit, on_audio_chunk
from whisperapp.queue import JobQueue

def test_ui_creates_without_error(tmp_path):
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
    q = JobQueue(db_path=tmp_path / "jobs.db")
    demo = create_ui(queue=q, worker=None)
    # Verify the demo has tab components by checking the blocks
    block_types = [type(b).__name__ for b in demo.blocks.values()]
    assert "Tab" in block_types

def test_on_audio_chunk_returns_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    mock_engine = MagicMock()
    mock_engine.process_chunk.return_value = "Hello world"
    mock_engine.get_transcript.return_value = "Hello world"

    state = {"engine": mock_engine, "transcript": ""}
    audio = (16000, np.zeros(4000, dtype=np.float32))

    transcript, new_state = on_audio_chunk(audio, state)
    assert transcript == "Hello world"
    assert new_state["transcript"] == "Hello world"

def test_on_audio_chunk_none_audio():
    state = {"transcript": "existing"}
    transcript, new_state = on_audio_chunk(None, state)
    assert transcript == "existing"
