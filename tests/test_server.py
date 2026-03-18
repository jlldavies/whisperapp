import pytest
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
