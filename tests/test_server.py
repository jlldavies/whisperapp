import base64
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
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


# ---------------------------------------------------------------------------
# Streaming endpoint tests
# ---------------------------------------------------------------------------

class FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


@pytest.mark.asyncio
async def test_stream_start_stop_lifecycle(app_and_queue):
    """Test start/chunk/stop lifecycle with mocked StreamingEngine."""
    app, q = app_and_queue

    mock_engine = MagicMock()
    mock_engine.process_chunk.return_value = "Hello"
    mock_engine.get_transcript.return_value = "Hello"
    mock_engine.stop.return_value = {
        "text": "Hello",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
        "raw_audio": np.zeros(16000, dtype=np.float32),
        "sample_rate": 16000,
    }

    with patch("whisperapp.server.StreamingEngine", return_value=mock_engine) as mock_cls:
        # Patch the import inside the endpoint
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver"
        ) as client:
            # Start
            resp = await client.post("/stream/start", json={"model": "base"})
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]
            assert session_id

            # Chunk
            audio = np.zeros(4000, dtype=np.float32)
            audio_b64 = base64.b64encode(audio.tobytes()).decode()
            resp = await client.post("/stream/chunk", json={
                "session_id": session_id,
                "audio_b64": audio_b64,
            })
            assert resp.status_code == 200
            assert resp.json()["transcript"] == "Hello"

            # Stop
            resp = await client.post("/stream/stop", json={
                "session_id": session_id,
            })
            assert resp.status_code == 200
            assert resp.json()["text"] == "Hello"


@pytest.mark.asyncio
async def test_stream_chunk_invalid_session(app_and_queue):
    app, q = app_and_queue
    audio = np.zeros(4000, dtype=np.float32)
    audio_b64 = base64.b64encode(audio.tobytes()).decode()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as client:
        resp = await client.post("/stream/chunk", json={
            "session_id": "nonexistent",
            "audio_b64": audio_b64,
        })
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stream_polish_before_stop(app_and_queue):
    """Polish should fail if session hasn't been stopped."""
    app, q = app_and_queue

    mock_engine = MagicMock()

    with patch("whisperapp.server.StreamingEngine", return_value=mock_engine):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver"
        ) as client:
            resp = await client.post("/stream/start", json={"model": "base"})
            session_id = resp.json()["session_id"]

            resp = await client.post("/stream/polish", json={
                "session_id": session_id,
            })
            assert resp.status_code == 409


@pytest.mark.asyncio
async def test_stream_polish_after_stop(app_and_queue):
    """Session is cleaned up by stop, so polish returns 404."""
    app, q = app_and_queue

    mock_engine = MagicMock()
    mock_engine.stop.return_value = {
        "text": "Test",
        "segments": [{"start": 0, "end": 1, "text": "Test"}],
        "raw_audio": np.zeros(16000, dtype=np.float32),
        "sample_rate": 16000,
    }
    mock_engine.polish.return_value = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "Test", "speaker": "SPEAKER_00"}],
    }

    with patch("whisperapp.server.StreamingEngine", return_value=mock_engine):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver"
        ) as client:
            resp = await client.post("/stream/start", json={"model": "base"})
            session_id = resp.json()["session_id"]

            resp = await client.post("/stream/stop", json={
                "session_id": session_id,
            })
            assert resp.status_code == 200

            # Session is cleaned up on stop, so polish returns 404
            resp = await client.post("/stream/polish", json={
                "session_id": session_id,
                "hf_token": "hf_test",
            })
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Speaker review endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture
def review_app(tmp_path):
    """App with a mock worker and a job in speaker_review status."""
    q = JobQueue(db_path=tmp_path / "jobs.db")
    mock_worker = MagicMock()
    app = create_app(queue=q, worker=mock_worker)

    # Create a job and move it to speaker_review
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"fake")
    job_id = q.create_job(str(audio), str(tmp_path), "large-v2", True, ["txt"])
    q.update_progress(job_id, 85, "speaker_review")
    q.set_status(job_id, JobStatus.SPEAKER_REVIEW)

    # Write a checkpoint file for the speaker_review stage
    from whisperapp.checkpoints import CheckpointManager
    cm = CheckpointManager(str(tmp_path), job_id)
    cm.save("speaker_review", {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello there", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "text": "Hi, how are you", "speaker": "SPEAKER_01"},
        ]
    })

    return app, q, mock_worker, job_id


@pytest.mark.asyncio
async def test_get_speakers(review_app):
    app, q, worker, job_id = review_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.get(f"/jobs/{job_id}/speakers")
        assert resp.status_code == 200
        data = resp.json()
        assert "SPEAKER_00" in data["speakers"]
        assert "SPEAKER_01" in data["speakers"]


@pytest.mark.asyncio
async def test_get_speakers_wrong_status(review_app):
    app, q, worker, job_id = review_app
    q.set_status(job_id, JobStatus.RUNNING)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.get(f"/jobs/{job_id}/speakers")
        assert resp.status_code == 409


@pytest.mark.asyncio
async def test_confirm_speakers_with_names(review_app):
    app, q, worker, job_id = review_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.post(f"/jobs/{job_id}/speakers", json={
            "names": {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        # Verify worker.complete_with_result was called with renamed segments
        worker.complete_with_result.assert_called_once()
        args = worker.complete_with_result.call_args
        result = args[0][1]  # second positional arg
        assert result["segments"][0]["speaker"] == "Alice"
        assert result["segments"][1]["speaker"] == "Bob"


@pytest.mark.asyncio
async def test_confirm_speakers_skip(review_app):
    app, q, worker, job_id = review_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.post(f"/jobs/{job_id}/speakers", json={"names": {}})
        assert resp.status_code == 200
        worker.complete_with_result.assert_called_once()
        args = worker.complete_with_result.call_args
        result = args[0][1]
        # Names should be unchanged
        assert result["segments"][0]["speaker"] == "SPEAKER_00"


@pytest.mark.asyncio
async def test_confirm_speakers_invalid_job_id(app_and_queue):
    app, q = app_and_queue
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.post("/jobs/not-a-uuid/speakers", json={"names": {}})
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_stream_start_invalid_model(app_and_queue):
    app, q = app_and_queue
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver"
    ) as client:
        resp = await client.post("/stream/start", json={"model": "large-v2"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# New API endpoints: /config, /audio/devices, /upload
# ---------------------------------------------------------------------------

@pytest.fixture
def app(tmp_path):
    q = JobQueue(db_path=tmp_path / "jobs.db")
    worker = MagicMock()
    return create_app(queue=q, worker=worker)

@pytest.mark.asyncio
async def test_get_config(app, tmp_path):
    with patch("whisperapp.server.Config") as MockCfg:
        MockCfg.return_value.hf_token = ""
        MockCfg.return_value.default_model = "large-v2"
        MockCfg.return_value.default_output_path = str(tmp_path)
        MockCfg.return_value.diarize_by_default = True
        MockCfg.return_value.streaming_model = "base"
        MockCfg.return_value.ai_provider = "none"
        MockCfg.return_value.ai_api_key = ""
        MockCfg.return_value.ai_model = ""
        MockCfg.return_value.ai_base_url = ""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/config")
    assert r.status_code == 200
    data = r.json()
    assert "default_model" in data

@pytest.mark.asyncio
async def test_post_config(app, tmp_path):
    with patch("whisperapp.server.Config") as MockCfg:
        instance = MagicMock()
        MockCfg.return_value = instance
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/config", json={"default_model": "small"})
    assert r.status_code == 200
    assert r.json()["success"] is True
    instance.save.assert_called_once()

@pytest.mark.asyncio
async def test_get_audio_devices(app):
    with patch("whisperapp.server.list_audio_devices", return_value=[
        {"name": "Built-in Mic", "index": 0, "sample_rate": 44100, "channels": 1}
    ]):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/audio/devices")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

@pytest.mark.asyncio
async def test_upload_file(app, tmp_path):
    fake_audio = b"RIFF" + b"\x00" * 100
    with patch("whisperapp.server._config_dir", return_value=tmp_path):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/upload", files={"file": ("test.wav", fake_audio, "audio/wav")})
    assert r.status_code == 200
    assert "path" in r.json()


@pytest.mark.asyncio
async def test_upload_file_suffix_allowlist(app, tmp_path):
    """Disallowed extension is replaced with .audio; allowed extensions are kept."""
    fake_audio = b"\x00" * 64
    with patch("whisperapp.server._config_dir", return_value=tmp_path):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            # Disallowed extension → .audio
            r = await c.post("/upload", files={"file": ("evil.exe", fake_audio, "application/octet-stream")})
            assert r.status_code == 200
            assert r.json()["path"].endswith(".audio")

            # Allowed extension → preserved
            r = await c.post("/upload", files={"file": ("clip.mp3", fake_audio, "audio/mpeg")})
            assert r.status_code == 200
            assert r.json()["path"].endswith(".mp3")
