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


# ---------------------------------------------------------------------------
# New model catalogue endpoints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_models_catalogue(app, tmp_path):
    """GET /models/catalogue returns all 5 categories."""
    with patch("whisperapp.model_registry.ModelRegistry.list_catalogue") as mock_catalogue:
        mock_catalogue.return_value = {
            "transcription": {"title": "Transcription", "sub": "...", "models": []},
            "streaming":     {"title": "Streaming",     "sub": "...", "models": []},
            "alignment":     {"title": "Alignment",     "sub": "...", "models": []},
            "diarization":   {"title": "Diarization",   "sub": "...", "models": []},
            "emotion":       {"title": "Emotion",       "sub": "...", "models": []},
        }
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/models/catalogue")
    assert r.status_code == 200
    data = r.json()
    assert "transcription" in data
    assert "emotion" in data


@pytest.mark.asyncio
async def test_get_disk_summary(app, tmp_path):
    """GET /models/disk-summary returns per-category usage."""
    with patch("whisperapp.model_registry.ModelRegistry.get_disk_summary") as mock_summary:
        mock_summary.return_value = {
            "transcription": {"bytes": 1572864000, "count": 1, "label": "Transcription"},
            "streaming":     {"bytes": 77594624,   "count": 1, "label": "Streaming"},
            "alignment":     {"bytes": 377487360,  "count": 1, "label": "Alignment"},
            "diarization":   {"bytes": 28311552,   "count": 1, "label": "Diarization"},
            "emotion":       {"bytes": 314572800,  "count": 1, "label": "Emotion"},
        }
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/models/disk-summary")
    assert r.status_code == 200
    data = r.json()
    assert "transcription" in data
    for cat_key, info in data.items():
        assert "bytes" in info
        assert "count" in info
        assert "label" in info


@pytest.mark.asyncio
async def test_post_storage_reveal(app, tmp_path):
    """POST /storage/reveal calls subprocess to open path."""
    import sys
    with patch("subprocess.Popen") as mock_popen:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/storage/reveal")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert ".whisperapp" in data["path"]
    # subprocess.Popen should have been called once
    mock_popen.assert_called_once()


@pytest.mark.asyncio
async def test_post_check_updates(app):
    """POST /models/check-updates clears caches and returns the new shape
    (includes per-capability latest-version lookups)."""
    with patch("whisperapp.model_registry.ModelRegistry.force_check_updates") as mock_force:
        mock_force.return_value = {"checked": True, "capabilities": [{"id": "librosa", "latest": "0.11.0"}]}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/models/check-updates")
    assert r.status_code == 200
    data = r.json()
    assert data["checked"] is True
    assert "capabilities" in data
    mock_force.assert_called_once()


@pytest.mark.asyncio
async def test_activate_model_unknown(app):
    """POST /models/{id}/activate with unknown id returns 404."""
    with patch("whisperapp.model_registry.ModelRegistry.activate_model") as mock_act:
        mock_act.return_value = {"success": False, "error": "Model not found: bad-id"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/models/bad-id/activate")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_deactivate_model(app):
    """POST /models/{id}/deactivate returns success for known model."""
    with patch("whisperapp.model_registry.ModelRegistry.deactivate_model") as mock_deact:
        mock_deact.return_value = {"success": True}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/models/speechbrain-iemocap/deactivate")
    assert r.status_code == 200
    assert r.json()["success"] is True


@pytest.mark.asyncio
async def test_catalogue_before_model_id_routing(app):
    """FastAPI must resolve /models/catalogue before /models/{model_id}."""
    with patch("whisperapp.model_registry.ModelRegistry.list_catalogue") as mock_catalogue:
        mock_catalogue.return_value = {}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/models/catalogue")
    # Should hit catalogue endpoint, not model_id endpoint (which would 404)
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# GET /jobs/{id}/segments — source_metadata and pagination
# ---------------------------------------------------------------------------

@pytest.fixture
def segments_app(tmp_path):
    """App with a job in speaker_review that has source_metadata + segments."""
    from whisperapp.checkpoints import CheckpointManager
    q = JobQueue(db_path=tmp_path / "jobs.db")
    mock_worker = MagicMock()
    app = create_app(queue=q, worker=mock_worker)

    audio = tmp_path / "interview.mp3"
    audio.write_bytes(b"fake audio")
    job_id = q.create_job(str(audio), str(tmp_path), "large-v2", True, ["txt"])
    q.update_progress(job_id, 85, "speaker_review")
    q.set_status(job_id, JobStatus.SPEAKER_REVIEW)

    cm = CheckpointManager(str(tmp_path), job_id)
    cm.save("speaker_review", {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello there", "speaker": "SPEAKER_00"},
            {"start": 2.5, "end": 5.0, "text": "Hi indeed",   "speaker": "SPEAKER_01"},
        ],
        "source_metadata": {
            "source_file": "interview.mp3",
            "source_path": str(audio),
            "source_hash": "abc123deadbeef",
            "file_size_bytes": "10",
            "duration": "00:05",
        },
    })
    return app, q, job_id


@pytest.mark.asyncio
async def test_segments_returns_source_metadata(segments_app):
    app, q, job_id = segments_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get(f"/jobs/{job_id}/segments")
    assert r.status_code == 200
    data = r.json()
    assert "source_metadata" in data
    assert data["source_metadata"]["source_hash"] == "abc123deadbeef"
    assert data["source_metadata"]["source_file"] == "interview.mp3"


@pytest.mark.asyncio
async def test_segments_returns_all_segments(segments_app):
    app, q, job_id = segments_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get(f"/jobs/{job_id}/segments")
    data = r.json()
    assert data["total"] == 2
    assert len(data["segments"]) == 2
    assert data["segments"][0]["text"] == "Hello there"
    assert data["segments"][1]["speaker"] == "SPEAKER_01"


@pytest.mark.asyncio
async def test_segments_pagination(segments_app):
    app, q, job_id = segments_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get(f"/jobs/{job_id}/segments?limit=1&offset=0")
    data = r.json()
    assert len(data["segments"]) == 1
    assert data["segments"][0]["text"] == "Hello there"
    assert data["next_offset"] == 1

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get(f"/jobs/{job_id}/segments?limit=1&offset=1")
    data = r.json()
    assert len(data["segments"]) == 1
    assert data["segments"][0]["text"] == "Hi indeed"
    assert data["next_offset"] is None


@pytest.mark.asyncio
async def test_segments_strips_words_by_default(segments_app):
    """Word-level timestamps should not appear unless include_words=true."""
    from whisperapp.checkpoints import CheckpointManager
    app, q, job_id = segments_app
    # Re-save checkpoint with a words field
    cm = CheckpointManager(str(q.get_job(job_id)["output_path"]), job_id)
    cm.save("speaker_review", {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello",
             "speaker": "SPEAKER_00",
             "words": [{"word": "Hello", "start": 0.0, "end": 0.5}]},
        ],
        "source_metadata": {"source_file": "f.mp3"},
    })
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get(f"/jobs/{job_id}/segments")
    assert "words" not in r.json()["segments"][0]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get(f"/jobs/{job_id}/segments?include_words=true")
    assert "words" in r.json()["segments"][0]


@pytest.mark.asyncio
async def test_segments_wrong_status(app_and_queue, tmp_path):
    app, q = app_and_queue
    audio = tmp_path / "f.mp3"
    audio.write_bytes(b"x")
    job_id = q.create_job(str(audio), str(tmp_path), "large-v2", False, ["txt"])
    # Job is queued — no transcript yet → 409
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get(f"/jobs/{job_id}/segments")
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_segments_invalid_job_id(app_and_queue):
    app, q = app_and_queue
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get("/jobs/not-a-uuid/segments")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_segments_not_found(app_and_queue):
    app, q = app_and_queue
    uid = "550e8400-e29b-41d4-a716-446655440000"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get(f"/jobs/{uid}/segments")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_segments_source_metadata_empty_when_absent(segments_app):
    """source_metadata key is always present; empty dict if not stored."""
    from whisperapp.checkpoints import CheckpointManager
    app, q, job_id = segments_app
    cm = CheckpointManager(str(q.get_job(job_id)["output_path"]), job_id)
    cm.save("speaker_review", {
        "segments": [{"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "S0"}],
        # no source_metadata key
    })
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get(f"/jobs/{job_id}/segments")
    assert r.status_code == 200
    assert r.json()["source_metadata"] == {}


# ---------------------------------------------------------------------------
# Template download endpoints
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_legal_html_template(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get("/templates/download-legal-html")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "{{source_hash}}" in body
    assert "{{legal_block}}" in body


@pytest.mark.asyncio
async def test_download_default_html_template(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get("/templates/download-html")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "{{transcript_block}}" in r.text


@pytest.mark.asyncio
async def test_templates_before_parameterised_routes(app):
    """Both template endpoints must resolve before /jobs/{id} etc. (routing order)."""
    for path in ("/templates/download-html", "/templates/download-legal-html"):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
            r = await c.get(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}"


@pytest.mark.asyncio
async def test_transcribe_stores_callback_url(app_and_queue, tmp_path):
    app, queue = app_and_queue
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"\x00" * 100)
    body = {
        "file_path": str(audio),
        "output_path": str(tmp_path),
        "model": "base",
        "diarize": False,
        "formats": ["txt"],
        "callback_url": "http://example.com/webhook",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.post("/transcribe", json=body)
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    job = queue.get_job(job_id)
    assert job["callback_url"] == "http://example.com/webhook"


@pytest.mark.asyncio
async def test_transcribe_callback_url_optional(app_and_queue, tmp_path):
    app, queue = app_and_queue
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"\x00" * 100)
    body = {
        "file_path": str(audio),
        "output_path": str(tmp_path),
        "model": "base",
        "diarize": False,
        "formats": ["txt"],
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.post("/transcribe", json=body)
    assert r.status_code == 200
    job = queue.get_job(r.json()["job_id"])
    assert job.get("callback_url") is None
