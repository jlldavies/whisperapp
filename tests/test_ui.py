import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock
from whisperapp.ui import create_ui
from whisperapp.queue import JobQueue


@pytest.fixture
def app(tmp_path):
    q = JobQueue(db_path=tmp_path / "jobs.db")
    return create_ui(queue=q, worker=MagicMock())


@pytest.mark.asyncio
async def test_ui_serves_index(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


@pytest.mark.asyncio
async def test_ui_serves_css(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/css/tokens.css")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_ui_serves_js(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/js/app.js")
    assert r.status_code == 200
