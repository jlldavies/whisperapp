# UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Gradio UI at port 7860 with a custom HTML/CSS/JS frontend that implements the Design/ mockups exactly — custom window chrome, 4 screens (Transcribe, Live, Speakers, Settings), light + dark themes.

**Architecture:** Static HTML/CSS/JS files served by a FastAPI app at port 7860. The frontend calls the existing REST API at port 7861 for all data. No build step — vanilla JS with ES modules.

**Tech Stack:** FastAPI + `StaticFiles`, vanilla JS ES modules, CSS custom properties (OKLCH tokens), Inter Tight + JetBrains Mono from Google Fonts.

---

## File map

### New files
| Path | Purpose |
|------|---------|
| `whisperapp/static/index.html` | App shell: window chrome, sidebar container, screen containers |
| `whisperapp/static/css/tokens.css` | Design tokens (copy + clean from `Design/tokens.css`) |
| `whisperapp/static/css/app.css` | All component + layout styles |
| `whisperapp/static/js/marks.js` | Logo SVG strings as JS exports |
| `whisperapp/static/js/api.js` | `fetch` wrappers for all REST endpoints |
| `whisperapp/static/js/shell.js` | `renderSidebar()`, `renderTopBar()`, `renderWaveform()` |
| `whisperapp/static/js/router.js` | Hash-based routing, screen lifecycle |
| `whisperapp/static/js/transcribe.js` | Transcribe screen |
| `whisperapp/static/js/live.js` | Live recording screen |
| `whisperapp/static/js/speakers.js` | Speaker review screen |
| `whisperapp/static/js/settings.js` | Settings screen |
| `whisperapp/static/js/app.js` | Entry point: init, theme toggle, router bootstrap |

### Modified files
| Path | Change |
|------|--------|
| `whisperapp/ui.py` | Replace Gradio `Blocks` with FastAPI `StaticFiles` mount |
| `whisperapp/__main__.py` | Replace Gradio `.launch()` thread with `uvicorn.run()` |
| `whisperapp/server.py` | Add `GET /config`, `POST /config`, `GET /audio/devices`, `POST /upload` |
| `tests/test_ui.py` | Replace Gradio-specific tests with static server smoke tests |
| `pyproject.toml` | Remove `gradio` dependency |

---

## Task 1: New API endpoints

**Files:**
- Modify: `whisperapp/server.py`
- Modify: `tests/test_server.py`

These endpoints are needed by the frontend before any UI can work.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_server.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch
from whisperapp.server import create_app
from whisperapp.queue import JobQueue

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
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/upload", files={"file": ("test.wav", fake_audio, "audio/wav")})
    assert r.status_code == 200
    assert "path" in r.json()
```

- [ ] **Step 2: Run to confirm failures**

```bash
.venv/bin/python -m pytest tests/test_server.py::test_get_config tests/test_server.py::test_post_config tests/test_server.py::test_get_audio_devices tests/test_server.py::test_upload_file -v
```
Expected: 4 failures (routes don't exist yet).

- [ ] **Step 3: Add helper `list_audio_devices` to `whisperapp/ui.py`**

Add this function (extracted from the existing `_list_input_devices`):

```python
def list_audio_devices() -> list[dict]:
    """Return list of audio input devices. Returns [] if pyaudio unavailable."""
    try:
        import pyaudio
    except ImportError:
        return []
    p = pyaudio.PyAudio()
    devices = []
    seen = set()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and info["name"] not in seen:
            seen.add(info["name"])
            devices.append({
                "name": info["name"],
                "index": i,
                "sample_rate": int(info["defaultSampleRate"]),
                "channels": int(info["maxInputChannels"]),
            })
    p.terminate()
    return devices
```

- [ ] **Step 4: Add endpoints to `whisperapp/server.py`**

Add these imports near the top of `server.py` (after existing imports):

```python
import shutil
import tempfile
from whisperapp.ui import list_audio_devices
```

Add these routes inside `create_app`, after the existing `/jobs/{job_id}/speakers` block:

```python
    # -----------------------------------------------------------------------
    # Config endpoints
    # -----------------------------------------------------------------------

    @app.get("/config")
    async def get_config():
        from whisperapp.config import Config
        cfg = Config()
        return {
            "hf_token": cfg.hf_token,
            "default_model": cfg.default_model,
            "default_output_path": cfg.default_output_path,
            "diarize_by_default": cfg.diarize_by_default,
            "streaming_model": cfg.streaming_model,
            "ai_provider": cfg.ai_provider,
            "ai_api_key": cfg.ai_api_key,
            "ai_model": cfg.ai_model,
            "ai_base_url": cfg.ai_base_url,
        }

    @app.post("/config")
    async def update_config(req: Request):
        from whisperapp.config import Config
        data = await req.json()
        cfg = Config()
        allowed = {
            "hf_token", "default_model", "default_output_path",
            "diarize_by_default", "streaming_model",
            "ai_provider", "ai_api_key", "ai_model", "ai_base_url",
        }
        for k, v in data.items():
            if k in allowed:
                setattr(cfg, k, v)
        cfg.save()
        return {"success": True}

    # -----------------------------------------------------------------------
    # Audio devices
    # -----------------------------------------------------------------------

    @app.get("/audio/devices")
    async def get_audio_devices():
        return list_audio_devices()

    # -----------------------------------------------------------------------
    # File upload — saves to temp dir, returns path for /transcribe
    # -----------------------------------------------------------------------

    @app.post("/upload")
    async def upload_file(file: "UploadFile"):
        from fastapi import UploadFile as _UploadFile
        suffix = Path(file.filename).suffix if file.filename else ".audio"
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix,
            dir=_config_dir() / "uploads"
        )
        (_config_dir() / "uploads").mkdir(parents=True, exist_ok=True)
        tmp_path = Path(tempfile.mktemp(
            suffix=suffix, dir=str(_config_dir() / "uploads")
        ))
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"path": str(tmp_path)}
```

Also add these imports at the top of `server.py`:

```python
from fastapi import UploadFile
from whisperapp.config import _config_dir
```

Fix the upload endpoint — replace the convoluted mktemp with:

```python
    @app.post("/upload")
    async def upload_file(file: UploadFile):
        uploads_dir = _config_dir() / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename).suffix if file.filename else ".audio"
        tmp_path = uploads_dir / (
            __import__("uuid").uuid4().hex + suffix
        )
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"path": str(tmp_path)}
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/test_server.py::test_get_config tests/test_server.py::test_post_config tests/test_server.py::test_get_audio_devices tests/test_server.py::test_upload_file -v
```
Expected: 4 pass.

- [ ] **Step 6: Commit**

```bash
git add whisperapp/server.py whisperapp/ui.py tests/test_server.py
git commit -m "feat: add /config, /audio/devices, /upload endpoints"
```

---

## Task 2: Replace Gradio with static file server

**Files:**
- Modify: `whisperapp/ui.py`
- Modify: `whisperapp/__main__.py`
- Modify: `tests/test_ui.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write new `tests/test_ui.py`**

Replace the entire file:

```python
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
```

- [ ] **Step 2: Run to confirm failures**

```bash
.venv/bin/python -m pytest tests/test_ui.py -v
```
Expected: 3 failures.

- [ ] **Step 3: Create the static directory skeleton**

```bash
mkdir -p whisperapp/static/css whisperapp/static/js
touch whisperapp/static/index.html
touch whisperapp/static/css/tokens.css
touch whisperapp/static/css/app.css
touch whisperapp/static/js/app.js
touch whisperapp/static/js/marks.js
touch whisperapp/static/js/api.js
touch whisperapp/static/js/shell.js
touch whisperapp/static/js/router.js
touch whisperapp/static/js/transcribe.js
touch whisperapp/static/js/live.js
touch whisperapp/static/js/speakers.js
touch whisperapp/static/js/settings.js
```

- [ ] **Step 4: Replace `whisperapp/ui.py`**

Overwrite the entire file:

```python
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

_STATIC_DIR = Path(__file__).parent / "static"


def create_ui(queue, worker) -> FastAPI:
    app = FastAPI()
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
    return app


# Keep for server.py import — moved here from old ui.py
def list_audio_devices() -> list[dict]:
    """Return list of audio input devices. Returns [] if pyaudio unavailable."""
    try:
        import pyaudio
    except ImportError:
        return []
    p = pyaudio.PyAudio()
    devices = []
    seen: set[str] = set()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and info["name"] not in seen:
            seen.add(info["name"])
            devices.append({
                "name": info["name"],
                "index": i,
                "sample_rate": int(info["defaultSampleRate"]),
                "channels": int(info["maxInputChannels"]),
            })
    p.terminate()
    return devices
```

- [ ] **Step 5: Update `whisperapp/__main__.py`**

Replace the Gradio launch thread:

```python
import atexit
import os
import sys
import threading
import webbrowser
import uvicorn
from pathlib import Path
from whisperapp.config import Config, _config_dir
from whisperapp.queue import JobQueue
from whisperapp.worker import Worker
from whisperapp.server import create_app
from whisperapp.ui import create_ui
from whisperapp.tray import TrayApp
from whisperapp.updater import run_update


def _pid_running(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _acquire_instance_lock() -> bool:
    lock_file = _config_dir() / "whisperapp.pid"
    _config_dir().mkdir(parents=True, exist_ok=True)
    if lock_file.exists():
        try:
            pid = int(lock_file.read_text().strip())
            if _pid_running(pid):
                return False
        except ValueError:
            pass
    lock_file.write_text(str(os.getpid()))
    atexit.register(lambda: lock_file.unlink(missing_ok=True))
    return True


def main():
    if not _acquire_instance_lock():
        print("WhisperApp is already running.")
        webbrowser.open("http://127.0.0.1:7860")
        sys.exit(0)

    threading.Thread(target=run_update, daemon=True).start()

    cfg = Config()
    queue = JobQueue()
    worker = Worker(queue=queue, hf_token=cfg.hf_token)
    worker.start()

    # REST API + MCP server
    mcp_app = create_app(queue=queue, worker=worker)
    threading.Thread(
        target=lambda: uvicorn.run(
            mcp_app, host="127.0.0.1", port=7861, log_level="warning"),
        daemon=True
    ).start()

    # Static UI server (replaces Gradio)
    ui_app = create_ui(queue=queue, worker=worker)
    threading.Thread(
        target=lambda: uvicorn.run(
            ui_app, host="127.0.0.1", port=7860, log_level="warning"),
        daemon=True
    ).start()

    if not cfg.hf_token:
        print("First run - open http://127.0.0.1:7860 to complete setup.")

    TrayApp(queue=queue, worker=worker).run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write minimal `index.html` to make tests pass**

```html
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Whisper</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter+Tight:ital,wght@0,400;0,500;0,600;0,700&family=JetBrains+Mono:wght@400;500&display=swap">
  <link rel="stylesheet" href="/css/tokens.css">
  <link rel="stylesheet" href="/css/app.css">
</head>
<body>
  <div id="app"></div>
  <script type="module" src="/js/app.js"></script>
</body>
</html>
```

Write minimal `whisperapp/static/js/app.js` (just enough for the file to exist and parse):

```js
// Entry point — bootstrapped in later tasks
console.log('WhisperApp loading…');
```

Write minimal `whisperapp/static/css/tokens.css` — copy from `Design/tokens.css` verbatim (already complete).

- [ ] **Step 7: Run tests**

```bash
.venv/bin/python -m pytest tests/test_ui.py -v
```
Expected: 3 pass.

- [ ] **Step 8: Remove Gradio from `pyproject.toml`**

In the `dependencies` list, remove the line:
```
"gradio>=4.0",
```

- [ ] **Step 9: Run full suite to confirm nothing else broke**

```bash
.venv/bin/python -m pytest tests/ -q
```
Expected: 138+ passed (the 3 new test_ui tests replace the old 3; net count is the same).

- [ ] **Step 10: Commit**

```bash
git add whisperapp/ui.py whisperapp/__main__.py whisperapp/static/ tests/test_ui.py pyproject.toml
git commit -m "feat: replace Gradio with FastAPI static file server"
```

---

## Task 3: CSS foundation

**Files:**
- Write: `whisperapp/static/css/tokens.css`
- Write: `whisperapp/static/css/app.css`

No Python tests — verify visually in browser after Task 4.

- [ ] **Step 1: Copy tokens**

Copy `Design/tokens.css` to `whisperapp/static/css/tokens.css` exactly as-is. It is already complete and correct.

- [ ] **Step 2: Write `whisperapp/static/css/app.css`**

```css
/* ─── Reset ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, #app { height: 100%; }
body {
  font-family: var(--font-sans);
  font-size: 13px;
  color: var(--ink);
  background: var(--paper);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
  font-feature-settings: "ss01", "cv11";
  letter-spacing: -0.005em;
  overflow: hidden;
}
button { font-family: inherit; cursor: pointer; font-size: inherit; }
select, input { font-family: inherit; font-size: inherit; color: inherit; }
h1, h2, h3 { letter-spacing: -0.022em; font-weight: 600; }
.mono { font-family: var(--font-mono); letter-spacing: 0; }

/* ─── Window chrome ──────────────────────────────────────────────────── */
#window-root {
  display: flex;
  flex-direction: column;
  height: 100vh;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 30px 80px -20px rgba(20,15,10,0.18), 0 0 0 1px rgba(20,15,10,0.08);
  background: var(--paper);
}

#titlebar {
  height: 36px;
  min-height: 36px;
  background: var(--paper-2);
  border-bottom: 1px solid var(--rule);
  display: flex;
  align-items: center;
  padding: 0 12px;
  gap: 12px;
  -webkit-app-region: drag;
  user-select: none;
}
#traffic-lights {
  display: flex;
  gap: 7px;
  -webkit-app-region: no-drag;
}
.tl-dot {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: var(--rule-strong);
  cursor: pointer;
  transition: background 0.1s;
}
.tl-dot:hover { filter: brightness(0.85); }
#window-title-text {
  flex: 1;
  text-align: center;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--ink-3);
}
#theme-toggle {
  -webkit-app-region: no-drag;
  width: 24px;
  height: 24px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--rule);
  background: transparent;
  color: var(--ink-3);
  font-size: 13px;
  display: grid;
  place-items: center;
  line-height: 1;
}
#theme-toggle:hover { background: var(--paper-3); }

/* ─── App shell ──────────────────────────────────────────────────────── */
#app-shell {
  flex: 1;
  display: grid;
  grid-template-columns: 220px 1fr;
  overflow: hidden;
}

/* ─── Sidebar ────────────────────────────────────────────────────────── */
#sidebar {
  background: var(--paper-2);
  border-right: 1px solid var(--rule);
  padding: 18px 14px;
  display: flex;
  flex-direction: column;
  gap: 22px;
  overflow-y: auto;
}
.wa-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 4px 6px;
}
.wa-brand-mark {
  width: 26px; height: 26px;
  display: grid; place-items: center;
  border-radius: 7px;
  background: var(--ink);
  color: var(--paper);
  flex-shrink: 0;
}
.wa-brand-name { font-weight: 600; letter-spacing: -0.02em; font-size: 15px; }
.wa-brand-sub {
  color: var(--ink-3);
  font-size: 11px;
  font-family: var(--font-mono);
  margin-left: auto;
}
.wa-nav { display: flex; flex-direction: column; gap: 1px; }
.wa-nav-section {
  font-family: var(--font-mono);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-4);
  padding: 8px 8px 6px;
}
.wa-nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  color: var(--ink-2);
  font-size: 13.5px;
  cursor: pointer;
  text-decoration: none;
}
.wa-nav-item:hover { background: var(--paper-3); }
.wa-nav-item.active {
  background: var(--paper);
  color: var(--ink);
  box-shadow: var(--shadow-sm), inset 0 0 0 1px var(--rule);
}
.wa-nav-icon { width: 16px; height: 16px; flex-shrink: 0; color: var(--ink-3); }
.wa-nav-item.active .wa-nav-icon { color: var(--ink); }
.wa-nav-kbd {
  margin-left: auto;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--ink-4);
}
.wa-recent-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--ink-4);
  flex-shrink: 0;
}
.wa-status-bar {
  margin-top: auto;
  padding: 10px;
  border-radius: var(--radius);
  background: var(--paper);
  border: 1px solid var(--rule);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-3);
  display: grid;
  gap: 4px;
}
.wa-status-row { display: flex; justify-content: space-between; align-items: center; }
.wa-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--signal-go);
  display: inline-block;
  margin-right: 4px;
}

/* ─── Main area ──────────────────────────────────────────────────────── */
#main-content { display: flex; flex-direction: column; overflow: hidden; }
.screen { display: none; flex-direction: column; flex: 1; overflow: hidden; }
.screen.active { display: flex; }

/* ─── Top bar ────────────────────────────────────────────────────────── */
.wa-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 28px;
  border-bottom: 1px solid var(--rule);
  background: var(--paper);
  flex-shrink: 0;
}
.wa-topbar h1 { font-size: 18px; }
.wa-topbar-sub {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-3);
  margin-top: 2px;
}
.wa-topbar-meta {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-3);
  display: flex;
  gap: 8px;
  align-items: center;
}

/* ─── Scrollable content area ────────────────────────────────────────── */
.wa-content { flex: 1; overflow-y: auto; padding: 28px; }

/* ─── Cards ──────────────────────────────────────────────────────────── */
.wa-card {
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: var(--radius);
  padding: 20px;
}
.wa-card-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 16px;
}
.wa-card-title { font-size: 13px; font-weight: 600; letter-spacing: -0.01em; }
.wa-card-sub {
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--ink-4);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* ─── Form elements ──────────────────────────────────────────────────── */
.wa-label {
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--ink-3);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  display: block;
  margin-bottom: 6px;
}
.wa-input, .wa-select {
  width: 100%;
  height: 36px;
  padding: 0 12px;
  border: 1px solid var(--rule);
  background: var(--paper);
  color: var(--ink);
  border-radius: var(--radius-sm);
  font-family: var(--font-mono);
  font-size: 12.5px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
}
.wa-select {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' fill='none'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%23857a6c' stroke-width='1.4' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;
  padding-right: 28px;
}
.wa-input:focus, .wa-select:focus {
  border-color: var(--ink-3);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
.wa-field-hint { font-size: 11.5px; color: var(--ink-3); margin-top: 6px; line-height: 1.45; }

/* ─── Buttons ────────────────────────────────────────────────────────── */
.wa-btn {
  height: 36px;
  padding: 0 16px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--rule);
  background: var(--paper);
  color: var(--ink);
  font-size: 13px;
  font-weight: 500;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  white-space: nowrap;
}
.wa-btn:hover { background: var(--paper-2); }
.wa-btn-primary { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.wa-btn-primary:hover { background: var(--ink-2); border-color: var(--ink-2); }
.wa-btn-accent { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.wa-btn-danger { color: var(--signal-rec); border-color: color-mix(in oklch, var(--signal-rec) 30%, var(--rule)); }
.wa-btn-danger:hover { background: color-mix(in oklch, var(--signal-rec) 8%, var(--paper)); }

/* ─── Chips ──────────────────────────────────────────────────────────── */
.wa-chip {
  display: inline-flex;
  align-items: center;
  height: 26px;
  padding: 0 10px;
  border: 1px solid var(--rule);
  border-radius: 999px;
  background: var(--paper);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--ink-2);
  cursor: pointer;
  user-select: none;
}
.wa-chip.on { border-color: var(--ink); background: var(--ink); color: var(--paper); }

/* ─── Toggle ─────────────────────────────────────────────────────────── */
.wa-toggle {
  position: relative;
  width: 28px;
  height: 16px;
  border-radius: 999px;
  background: var(--rule-strong);
  flex-shrink: 0;
  transition: background 0.15s;
  cursor: pointer;
}
.wa-toggle.on { background: var(--ink); }
.wa-toggle::after {
  content: "";
  position: absolute;
  top: 2px;
  left: 2px;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--paper);
  transition: left 0.15s;
}
.wa-toggle.on::after { left: 14px; }

/* ─── Drop zone ──────────────────────────────────────────────────────── */
.wa-drop {
  border: 1.5px dashed var(--rule-strong);
  border-radius: var(--radius);
  padding: 40px 24px;
  text-align: center;
  background: var(--paper-2);
  color: var(--ink-3);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  transition: border-color 0.1s, background 0.1s;
}
.wa-drop.dragover { border-color: var(--accent); background: var(--accent-soft); }
.wa-drop-title { font-size: 15px; color: var(--ink); font-weight: 500; }
.wa-drop-sub { font-family: var(--font-mono); font-size: 11px; }

/* ─── Queue item ─────────────────────────────────────────────────────── */
.queue-item {
  padding: 12px;
  border: 1px solid var(--rule);
  border-radius: var(--radius-sm);
  background: var(--paper);
}
.queue-item.running { background: var(--paper-2); }
.queue-item-header { display: flex; align-items: center; gap: 8px; }
.queue-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.queue-dot.running {
  background: var(--accent);
  box-shadow: 0 0 0 4px color-mix(in oklch, var(--accent) 18%, transparent);
  animation: pulse-ring 1.6s ease-in-out infinite;
}
.queue-dot.queued { background: var(--ink-4); }
.queue-dot.done { background: var(--signal-go); }
.queue-dot.error { background: var(--signal-rec); }
@keyframes pulse-ring {
  0%, 100% { box-shadow: 0 0 0 4px color-mix(in oklch, var(--accent) 18%, transparent); }
  50%       { box-shadow: 0 0 0 6px color-mix(in oklch, var(--accent) 10%, transparent); }
}
.queue-item-title {
  flex: 1;
  font-size: 13px;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.queue-badge {
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--ink-4);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.queue-meta {
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--ink-3);
  margin-top: 4px;
  margin-left: 15px;
}
.queue-progress { margin-top: 10px; margin-left: 15px; }
.queue-progress-bar {
  height: 3px;
  background: var(--rule);
  border-radius: 2px;
  overflow: hidden;
}
.queue-progress-fill { height: 100%; background: var(--accent); transition: width 0.3s; }
.queue-progress-meta {
  font-family: var(--font-mono);
  font-size: 10.5px;
  color: var(--ink-3);
  margin-top: 6px;
  display: flex;
  justify-content: space-between;
}

/* ─── Pipeline stage ─────────────────────────────────────────────────── */
.pipeline-stage {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border: 1px solid var(--rule);
  border-radius: var(--radius-sm);
  background: var(--paper);
}
.pipeline-stage.off { background: var(--paper-2); }
.pipeline-stage-idx { color: var(--ink-4); font-size: 11px; font-family: var(--font-mono); }
.pipeline-stage-detail { font-family: var(--font-mono); font-size: 10.5px; color: var(--ink-3); margin-top: 1px; }

/* ─── Waveform ───────────────────────────────────────────────────────── */
.wa-waveform {
  display: flex;
  align-items: center;
  gap: 1.5px;
  width: 100%;
}
.wa-waveform-bar { flex: 1; border-radius: 1.5px; }

/* ─── Live transcript ────────────────────────────────────────────────── */
.transcript-line { display: grid; grid-template-columns: 60px 1fr; gap: 14px; }
.transcript-ts { font-family: var(--font-mono); font-size: 11px; color: var(--ink-4); padding-top: 3px; }
.transcript-partial-word {
  background: color-mix(in oklch, var(--accent) 16%, transparent);
  border-radius: 3px;
  padding: 0 3px;
}
.transcript-typing { color: var(--ink-4); }

/* ─── Speaker row ────────────────────────────────────────────────────── */
.speaker-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 18px;
  border-bottom: 1px solid var(--rule);
  cursor: pointer;
}
.speaker-row:hover { background: var(--paper-3); }
.speaker-row.active { background: var(--paper-2); }
.speaker-avatar {
  width: 28px; height: 28px;
  border-radius: 7px;
  display: grid;
  place-items: center;
  font-size: 12px;
  font-weight: 600;
  color: var(--paper);
  flex-shrink: 0;
}
.speaker-snippet {
  display: flex;
  gap: 12px;
  padding: 14px;
  border: 1px solid var(--rule);
  border-radius: var(--radius-sm);
  background: var(--paper-2);
}
.snippet-play {
  width: 32px; height: 32px;
  border-radius: 50%;
  border: none;
  background: var(--ink);
  color: var(--paper);
  display: grid;
  place-items: center;
  flex-shrink: 0;
  cursor: pointer;
}
.snippet-play:hover { background: var(--ink-2); }

/* ─── Settings ───────────────────────────────────────────────────────── */
.settings-subnav {
  display: flex;
  flex-direction: column;
  gap: 1px;
  position: sticky;
  top: 0;
}
.settings-subnav-item {
  font-size: 13px;
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  color: var(--ink-3);
  cursor: pointer;
}
.settings-subnav-item:hover { background: var(--paper-2); }
.settings-subnav-item.active { color: var(--ink); background: var(--paper-2); font-weight: 500; }
.settings-section { display: flex; flex-direction: column; gap: 14px; }
.settings-section-head {
  display: flex;
  align-items: baseline;
  gap: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--rule);
}
.settings-section-head h2 { font-size: 15px; }
.provider-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.provider-card {
  padding: 14px;
  border: 1px solid var(--rule);
  border-radius: var(--radius-sm);
  background: var(--paper-2);
  cursor: pointer;
}
.provider-card.active {
  border-color: var(--ink);
  background: var(--paper);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
.provider-card-name { font-size: 13px; font-weight: 500; }
.provider-card-sub { font-family: var(--font-mono); font-size: 10.5px; color: var(--ink-3); margin-top: 4px; }
.toggle-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 0;
  cursor: pointer;
}
.toggle-row label { font-size: 13px; cursor: pointer; }
```

- [ ] **Step 3: Commit**

```bash
git add whisperapp/static/css/
git commit -m "feat: add design tokens and component CSS"
```

---

## Task 4: Marks, Shell, and Window Chrome

**Files:**
- Write: `whisperapp/static/js/marks.js`
- Write: `whisperapp/static/js/shell.js`
- Update: `whisperapp/static/index.html`

- [ ] **Step 1: Write `whisperapp/static/js/marks.js`**

```js
export const LOGO_SVG = `<svg width="26" height="26" viewBox="0 0 32 32" fill="none">
  <rect x="1" y="1" width="30" height="30" rx="9" fill="currentColor"/>
  <path d="M8 12 L11 22 L14 14 L16 14 L18 22 L21 14 L24 22"
    stroke="var(--paper)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>`;

export const NAV_ICONS = {
  transcribe: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><path d="M3 3h7l3 3v7a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/><path d="M9 3v3h4"/><path d="M5 9h6M5 11.5h4"/></svg>`,
  live:        `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><rect x="6" y="2" width="4" height="8" rx="2"/><path d="M3 8a5 5 0 0 0 10 0M8 13v1.5"/></svg>`,
  speakers:    `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><circle cx="6" cy="6.5" r="2.2"/><path d="M2.5 13c0-1.9 1.5-3.4 3.5-3.4s3.5 1.5 3.5 3.4"/><circle cx="11.5" cy="5" r="1.6"/><path d="M9.5 12c.4-1.5 1.5-2.5 3-2.5s2.5 1 2.5 2.5"/></svg>`,
  settings:    `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"><circle cx="8" cy="8" r="2"/><path d="M8 1.5v1.5M8 13v1.5M14.5 8H13M3 8H1.5M12.6 3.4l-1 1M4.4 11.6l-1 1M12.6 12.6l-1-1M4.4 4.4l-1-1"/></svg>`,
  upload:      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12M7 8l5-5 5 5M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>`,
  play:        `<svg viewBox="0 0 16 16" fill="currentColor"><path d="M5 3.5v9l8-4.5z"/></svg>`,
  pause:       `<svg viewBox="0 0 16 16" fill="currentColor"><rect x="5" y="3" width="2.5" height="10" rx="0.6"/><rect x="9" y="3" width="2.5" height="10" rx="0.6"/></svg>`,
};
```

- [ ] **Step 2: Write `whisperapp/static/js/shell.js`**

```js
import { LOGO_SVG, NAV_ICONS } from './marks.js';

const NAV_ITEMS = [
  { id: 'transcribe', label: 'Transcribe', kbd: '⌘1' },
  { id: 'live',       label: 'Live',       kbd: '⌘2' },
  { id: 'speakers',   label: 'Speakers',   kbd: '⌘3' },
  { id: 'settings',   label: 'Settings',   kbd: '⌘,' },
];

export function renderSidebar(container, activeId, onNavigate) {
  const recentJobs = JSON.parse(localStorage.getItem('wa-recent') || '[]');

  container.innerHTML = `
    <div class="wa-brand">
      <div class="wa-brand-mark">${LOGO_SVG}</div>
      <div class="wa-brand-name">Whisper</div>
      <div class="wa-brand-sub">v1.1</div>
    </div>

    <nav class="wa-nav">
      <div class="wa-nav-section">Workspace</div>
      ${NAV_ITEMS.map(item => `
        <div class="wa-nav-item${item.id === activeId ? ' active' : ''}" data-nav="${item.id}">
          <span class="wa-nav-icon">${NAV_ICONS[item.id]}</span>
          <span>${item.label}</span>
          <span class="wa-nav-kbd">${item.kbd}</span>
        </div>
      `).join('')}
    </nav>

    ${recentJobs.length ? `
    <div class="wa-nav">
      <div class="wa-nav-section">Recent</div>
      ${recentJobs.slice(0, 3).map(j => `
        <div class="wa-nav-item" style="font-size:12.5px" data-nav-job="${j.id}">
          <span class="wa-recent-dot"></span>
          <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${j.name}</span>
        </div>
      `).join('')}
    </div>` : ''}

    <div class="wa-status-bar" id="status-bar">
      <div class="wa-status-row">
        <span><span class="wa-dot"></span>Daemon</span>
        <span>:7860</span>
      </div>
      <div class="wa-status-row">
        <span><span class="wa-dot"></span>REST API</span>
        <span>:7861</span>
      </div>
      <div class="wa-status-row" id="gpu-row">
        <span style="color:var(--ink-4)">GPU</span>
        <span>—</span>
      </div>
    </div>
  `;

  container.querySelectorAll('[data-nav]').forEach(el => {
    el.addEventListener('click', () => onNavigate(el.dataset.nav));
  });

  // Fetch /info to populate GPU row
  fetch('http://127.0.0.1:7861/info')
    .then(r => r.json())
    .then(info => {
      const row = document.getElementById('gpu-row');
      if (row) row.innerHTML = `
        <span style="color:var(--ink-4)">GPU</span>
        <span>${info.acceleration}</span>
      `;
    })
    .catch(() => {});
}

export function renderTopBar(container, { title, sub = '', right = '' }) {
  container.innerHTML = `
    <div class="wa-topbar">
      <div>
        <h1>${title}</h1>
        ${sub ? `<div class="wa-topbar-sub mono">${sub}</div>` : ''}
      </div>
      <div class="wa-topbar-meta">${right}</div>
    </div>
  `;
}

export function renderWaveform(container, { height = 56, bars = 80, playhead = null }) {
  const heights = Array.from({ length: bars }, (_, i) => {
    const x = i / bars;
    const v = Math.sin(x * 22) * 0.4 + Math.sin(x * 6 + 1.2) * 0.5 + Math.sin(x * 53) * 0.15;
    return Math.max(0.15, Math.min(1, 0.55 + v * 0.45));
  });
  container.style.height = height + 'px';
  container.className = 'wa-waveform';
  container.innerHTML = heights.map((h, i) => {
    const past = playhead != null && i / bars < playhead;
    return `<div class="wa-waveform-bar" style="
      height:${h * 100}%;
      background:${past ? 'var(--accent)' : 'var(--ink-2)'};
      opacity:${past ? 1 : 0.55}
    "></div>`;
  }).join('');
}
```

- [ ] **Step 3: Update `whisperapp/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Whisper</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter+Tight:ital,wght@0,400;0,500;0,600;0,700&family=JetBrains+Mono:wght@400;500&display=swap">
  <link rel="stylesheet" href="/css/tokens.css">
  <link rel="stylesheet" href="/css/app.css">
</head>
<body>
  <div id="window-root">
    <div id="titlebar">
      <div id="traffic-lights">
        <span class="tl-dot close"   title="Close"></span>
        <span class="tl-dot minimize" title="Minimize"></span>
        <span class="tl-dot maximize" title="Maximize"></span>
      </div>
      <div id="window-title-text" class="mono">Whisper</div>
      <button id="theme-toggle" title="Toggle theme">◑</button>
    </div>
    <div id="app-shell">
      <aside id="sidebar"></aside>
      <div id="main-content">
        <div id="screen-transcribe" class="screen"></div>
        <div id="screen-live"       class="screen"></div>
        <div id="screen-speakers"   class="screen"></div>
        <div id="screen-settings"   class="screen"></div>
      </div>
    </div>
  </div>
  <script type="module" src="/js/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Verify in browser**

Start the app: `.venv/bin/python -m whisperapp`  
Open http://127.0.0.1:7860

Expected: Custom window chrome with 3-dot titlebar, sidebar with "Whisper" brand and nav items, empty screen area.

- [ ] **Step 5: Commit**

```bash
git add whisperapp/static/
git commit -m "feat: window chrome, sidebar, and shell JS"
```

---

## Task 5: Router and API module

**Files:**
- Write: `whisperapp/static/js/api.js`
- Write: `whisperapp/static/js/router.js`
- Update: `whisperapp/static/js/app.js`

- [ ] **Step 1: Write `whisperapp/static/js/api.js`**

```js
const API = 'http://127.0.0.1:7861';

async function req(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(API + path, opts);
  if (!r.ok) throw new Error(`${method} ${path} → ${r.status}`);
  return r.json();
}

export const api = {
  // Health
  health: ()           => req('GET', '/health'),
  info:   ()           => req('GET', '/info'),

  // Jobs
  listJobs: (status)   => req('GET', '/jobs' + (status ? `?status=${status}` : '')),
  getJob:   (id)       => req('GET', `/jobs/${id}`),
  cancelJob:(id)       => req('POST', `/jobs/${id}/cancel`),
  getTranscript: (id, fmt) => req('GET', `/jobs/${id}/transcript?format=${fmt}`),

  // Transcribe
  transcribe: (body)   => req('POST', '/transcribe', body),
  upload: async (file) => {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(API + '/upload', { method: 'POST', body: fd });
    if (!r.ok) throw new Error('Upload failed');
    return r.json();
  },

  // Speakers
  getSpeakers:     (id)       => req('GET',  `/jobs/${id}/speakers`),
  confirmSpeakers: (id, names) => req('POST', `/jobs/${id}/speakers`, { names }),

  // Config
  getConfig:    ()     => req('GET',  '/config'),
  updateConfig: (body) => req('POST', '/config', body),

  // Audio
  getAudioDevices: ()  => req('GET', '/audio/devices'),

  // Streaming
  streamStart: (body)  => req('POST', '/stream/start', body),
  streamChunk: (body)  => req('POST', '/stream/chunk', body),
  streamStop:  (body)  => req('POST', '/stream/stop', body),
  streamPolish:(body)  => req('POST', '/stream/polish', body),
};
```

- [ ] **Step 2: Write `whisperapp/static/js/router.js`**

```js
export class Router {
  constructor(screens, onNavigate) {
    this._screens = screens; // { id: HTMLElement }
    this._onNavigate = onNavigate;
    this._current = null;
    window.addEventListener('hashchange', () => this._apply());
  }

  navigate(id) {
    window.location.hash = id;
  }

  start(defaultId) {
    const id = window.location.hash.slice(1) || defaultId;
    this._show(id);
  }

  _apply() {
    const id = window.location.hash.slice(1);
    if (id) this._show(id);
  }

  _show(id) {
    if (!this._screens[id]) return;
    Object.entries(this._screens).forEach(([sid, el]) => {
      el.classList.toggle('active', sid === id);
    });
    this._current = id;
    this._onNavigate(id);
  }

  get current() { return this._current; }
}
```

- [ ] **Step 3: Write full `whisperapp/static/js/app.js`**

```js
import { renderSidebar } from './shell.js';
import { Router } from './router.js';
import { initTranscribe } from './transcribe.js';
import { initLive } from './live.js';
import { initSpeakers } from './speakers.js';
import { initSettings } from './settings.js';

const SCREENS = {
  transcribe: document.getElementById('screen-transcribe'),
  live:       document.getElementById('screen-live'),
  speakers:   document.getElementById('screen-speakers'),
  settings:   document.getElementById('screen-settings'),
};

let _router;

function onNavigate(id) {
  renderSidebar(document.getElementById('sidebar'), id, id => _router.navigate(id));
}

// Theme
const themeBtn = document.getElementById('theme-toggle');
themeBtn.addEventListener('click', () => {
  const html = document.documentElement;
  const next = html.dataset.theme === 'dark' ? 'light' : 'dark';
  html.dataset.theme = next;
  localStorage.setItem('wa-theme', next);
});
const saved = localStorage.getItem('wa-theme');
if (saved) document.documentElement.dataset.theme = saved;

// Init screens (idempotent — called once each)
initTranscribe(SCREENS.transcribe);
initLive(SCREENS.live);
initSpeakers(SCREENS.speakers);
initSettings(SCREENS.settings);

// Sidebar initial render
renderSidebar(document.getElementById('sidebar'), 'transcribe', id => _router.navigate(id));

// Router
_router = new Router(SCREENS, onNavigate);
_router.start('transcribe');
```

- [ ] **Step 4: Add placeholder init functions to each screen module**

`whisperapp/static/js/transcribe.js`:
```js
export function initTranscribe(container) {
  container.innerHTML = `<div style="padding:40px;color:var(--ink-3)">Transcribe — coming soon</div>`;
}
```

Repeat for `live.js`, `speakers.js`, `settings.js` with matching label.

- [ ] **Step 5: Verify in browser**

Open http://127.0.0.1:7860  
Expected: Sidebar nav items are clickable, active item highlights, URL hash changes, screen area swaps.

- [ ] **Step 6: Commit**

```bash
git add whisperapp/static/js/
git commit -m "feat: hash router, API module, app bootstrap"
```

---

## Task 6: Transcribe screen

**Files:**
- Write: `whisperapp/static/js/transcribe.js`

- [ ] **Step 1: Write full `whisperapp/static/js/transcribe.js`**

```js
import { api } from './api.js';
import { NAV_ICONS } from './marks.js';

const MODELS = ['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'];
const FORMATS = ['txt', 'srt', 'vtt', 'json', 'tsv'];
let _pollTimer = null;

export function initTranscribe(container) {
  let selectedFile = null;
  let selectedFormats = new Set(['txt', 'srt', 'vtt', 'json']);

  container.innerHTML = `
    <div class="wa-topbar">
      <div>
        <h1>Transcribe</h1>
        <div class="wa-topbar-sub mono">Drop a file · or paste a path</div>
      </div>
      <div class="wa-topbar-meta mono" id="tx-meta"></div>
    </div>
    <div class="wa-content" style="display:grid;grid-template-columns:1.4fr 1fr;gap:24px;align-content:start">
      <!-- Left -->
      <div style="display:flex;flex-direction:column;gap:20px">
        <div class="wa-drop" id="drop-zone">
          <div style="width:40px;height:40px;color:var(--ink-3)">${NAV_ICONS.upload}</div>
          <div class="wa-drop-title">Drop audio or video</div>
          <div class="wa-drop-sub">.mp3 · .wav · .m4a · .mp4 · .mov · up to 4 GB</div>
          <div id="drop-file-name" style="font-family:var(--font-mono);font-size:11px;color:var(--accent);display:none"></div>
          <div style="display:flex;gap:8px;margin-top:6px">
            <label class="wa-btn" style="cursor:pointer">
              Choose file…
              <input type="file" id="file-picker" accept=".mp3,.wav,.m4a,.mp4,.mov,.flac,.ogg,.webm" style="display:none">
            </label>
            <button class="wa-btn" id="paste-path-btn" style="background:transparent">Paste path</button>
          </div>
        </div>

        <div class="wa-card">
          <div class="wa-card-head">
            <div class="wa-card-title">Pipeline</div>
            <div class="wa-card-sub" id="pipeline-sub">3 steps</div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
            <div>
              <span class="wa-label">Model</span>
              <select class="wa-select" id="model-select">
                ${MODELS.map(m => `<option value="${m}"${m==='large-v2'?' selected':''}>${m}</option>`).join('')}
              </select>
            </div>
            <div>
              <span class="wa-label">Language</span>
              <select class="wa-select" id="lang-select">
                <option value="">Auto-detect</option>
                <option>en</option><option>fr</option><option>de</option>
                <option>es</option><option>it</option><option>ja</option>
                <option>zh</option><option>pt</option>
              </select>
            </div>
            <div style="grid-column:1/-1">
              <span class="wa-label">Output path</span>
              <input class="wa-input" id="output-path" placeholder="~/Downloads/transcripts">
            </div>
          </div>

          <div style="margin-top:20px;display:flex;flex-direction:column;gap:12px">
            <span class="wa-label">Stages</span>
            <div style="display:flex;flex-direction:column;gap:8px">
              <div class="pipeline-stage" id="stage-transcribe">
                <span class="pipeline-stage-idx mono">01</span>
                <div style="flex:1">
                  <div style="font-size:13px;font-weight:500">Transcribe</div>
                  <div class="pipeline-stage-detail mono">WhisperX · word-level timestamps</div>
                </div>
                <span class="wa-toggle on" data-stage="transcribe"></span>
              </div>
              <div class="pipeline-stage" id="stage-align">
                <span class="pipeline-stage-idx mono">02</span>
                <div style="flex:1">
                  <div style="font-size:13px;font-weight:500">Align</div>
                  <div class="pipeline-stage-detail mono">phoneme · forced alignment</div>
                </div>
                <span class="wa-toggle on" data-stage="align"></span>
              </div>
              <div class="pipeline-stage" id="stage-diarize">
                <span class="pipeline-stage-idx mono">03</span>
                <div style="flex:1">
                  <div style="font-size:13px;font-weight:500">Diarize</div>
                  <div class="pipeline-stage-detail mono">pyannote · who-said-what</div>
                </div>
                <span class="wa-toggle on" data-stage="diarize"></span>
              </div>
            </div>
          </div>

          <div style="margin-top:20px">
            <span class="wa-label">Output formats</span>
            <div style="display:flex;gap:6px;flex-wrap:wrap" id="format-chips">
              ${FORMATS.map(f => `
                <span class="wa-chip${selectedFormats.has(f)?' on':''}" data-fmt="${f}">${f}</span>
              `).join('')}
            </div>
          </div>
        </div>
      </div>

      <!-- Right: queue -->
      <div style="display:flex;flex-direction:column;gap:20px">
        <div class="wa-card">
          <div class="wa-card-head">
            <div class="wa-card-title">Queue</div>
            <div class="wa-card-sub" id="queue-sub">—</div>
          </div>
          <div id="queue-list" style="display:flex;flex-direction:column;gap:10px">
            <div style="color:var(--ink-4);font-size:12.5px">No jobs yet.</div>
          </div>
          <div style="display:flex;gap:8px;margin-top:16px;padding-top:16px;border-top:1px solid var(--rule)">
            <button class="wa-btn wa-btn-primary" id="add-btn" style="flex:1">Add to queue</button>
            <button class="wa-btn" id="clear-btn">Clear done</button>
          </div>
        </div>
        <div class="wa-card" style="background:var(--paper-2)">
          <div class="wa-card-head" style="margin-bottom:8px">
            <div class="wa-card-title">Tip</div>
            <div class="wa-card-sub">⌘K</div>
          </div>
          <p style="font-size:12.5px;color:var(--ink-2);line-height:1.5">
            Drop a folder to queue every audio file inside it. Right-click an item to re-run with a different model.
          </p>
        </div>
      </div>
    </div>
  `;

  // Load saved config
  api.getConfig().then(cfg => {
    document.getElementById('output-path').value = cfg.default_output_path || '';
    if (!cfg.diarize_by_default) {
      const tog = container.querySelector('[data-stage="diarize"]');
      if (tog) { tog.classList.remove('on'); container.querySelector('#stage-diarize').classList.add('off'); }
    }
  }).catch(() => {});

  // Wire up drop zone
  const zone = container.querySelector('#drop-zone');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) selectFile(file);
  });

  container.querySelector('#file-picker').addEventListener('change', e => {
    if (e.target.files[0]) selectFile(e.target.files[0]);
  });

  container.querySelector('#paste-path-btn').addEventListener('click', () => {
    const p = prompt('Paste file path:');
    if (p) { selectedFile = { type: 'path', value: p.trim() }; showFileName(p.trim()); }
  });

  function selectFile(file) {
    selectedFile = { type: 'file', value: file };
    showFileName(file.name);
  }

  function showFileName(name) {
    const el = container.querySelector('#drop-file-name');
    el.textContent = name;
    el.style.display = 'block';
  }

  // Format chip toggles
  container.querySelector('#format-chips').addEventListener('click', e => {
    const chip = e.target.closest('[data-fmt]');
    if (!chip) return;
    const fmt = chip.dataset.fmt;
    if (selectedFormats.has(fmt)) selectedFormats.delete(fmt);
    else selectedFormats.add(fmt);
    chip.classList.toggle('on', selectedFormats.has(fmt));
  });

  // Stage toggles
  container.querySelectorAll('[data-stage]').forEach(tog => {
    tog.addEventListener('click', () => {
      const on = tog.classList.toggle('on');
      tog.closest('.pipeline-stage').classList.toggle('off', !on);
    });
  });

  // Add to queue
  container.querySelector('#add-btn').addEventListener('click', async () => {
    if (!selectedFile) { alert('Select a file first.'); return; }
    const btn = container.querySelector('#add-btn');
    btn.disabled = true;
    btn.textContent = 'Adding…';
    try {
      let filePath;
      if (selectedFile.type === 'path') {
        filePath = selectedFile.value;
      } else {
        const up = await api.upload(selectedFile.value);
        filePath = up.path;
      }
      const diarize = container.querySelector('[data-stage="diarize"]').classList.contains('on');
      await api.transcribe({
        file_path: filePath,
        output_path: container.querySelector('#output-path').value || null,
        model: container.querySelector('#model-select').value,
        diarize,
        formats: [...selectedFormats],
      });
      selectedFile = null;
      container.querySelector('#drop-file-name').style.display = 'none';
      refreshQueue();
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Add to queue';
    }
  });

  // Clear done
  container.querySelector('#clear-btn').addEventListener('click', () => refreshQueue());

  // Poll queue
  refreshQueue();
  _pollTimer = setInterval(refreshQueue, 3000);
}

async function refreshQueue() {
  let jobs;
  try { jobs = await api.listJobs(); } catch { return; }
  const list = document.getElementById('queue-list');
  const sub  = document.getElementById('queue-sub');
  if (!list) return;

  const active = jobs.filter(j => j.status === 'running').length;
  sub.textContent = jobs.length ? `${jobs.length} · ${active} active` : '—';

  if (!jobs.length) {
    list.innerHTML = `<div style="color:var(--ink-4);font-size:12.5px">No jobs yet.</div>`;
    return;
  }
  list.innerHTML = jobs.map(j => renderQueueItem(j)).join('');
}

function renderQueueItem(j) {
  const pct = j.progress ?? 0;
  const stage = j.stage || j.status;
  return `
    <div class="queue-item ${j.status}">
      <div class="queue-item-header">
        <span class="queue-dot ${j.status}"></span>
        <span class="queue-item-title">${escHtml(j.file_name || j.file_path)}</span>
        <span class="queue-badge">${j.status}</span>
      </div>
      <div class="queue-meta mono">${j.model}${j.diarize ? ' · diarize' : ''}</div>
      ${j.status === 'running' ? `
        <div class="queue-progress">
          <div class="queue-progress-bar"><div class="queue-progress-fill" style="width:${pct}%"></div></div>
          <div class="queue-progress-meta mono">
            <span>${escHtml(stage)}</span><span>${pct}%</span>
          </div>
        </div>` : ''}
    </div>
  `;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
```

- [ ] **Step 2: Verify in browser**

Open http://127.0.0.1:7860 → Transcribe tab.  
Expected: Drop zone, Pipeline card with model/language selects and toggleable stages, format chips, empty queue. Drag a file onto the drop zone, click "Add to queue" — job appears in queue with status dot. Progress bar appears while running.

- [ ] **Step 3: Commit**

```bash
git add whisperapp/static/js/transcribe.js
git commit -m "feat: Transcribe screen with file drop, pipeline config, queue polling"
```

---

## Task 7: Live screen

**Files:**
- Write: `whisperapp/static/js/live.js`

- [ ] **Step 1: Write `whisperapp/static/js/live.js`**

```js
import { api } from './api.js';
import { renderWaveform } from './shell.js';

let _session = null;
let _mediaStream = null;
let _audioCtx = null;
let _processor = null;
let _elapsed = 0;
let _elapsedTimer = null;
let _transcript = [];
let _partial = '';
let _waveAnimFrame = null;

export function initLive(container) {
  container.innerHTML = `
    <div class="wa-topbar">
      <div>
        <h1>Live</h1>
        <div class="wa-topbar-sub mono" id="live-sub">Select a microphone and start recording</div>
      </div>
      <div class="wa-topbar-meta" id="live-meta"></div>
    </div>
    <div class="wa-content" style="display:flex;flex-direction:column;gap:20px">
      <!-- Waveform hero -->
      <div class="wa-card" style="padding:22px">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
          <div>
            <div class="wa-card-sub">Input level</div>
            <div class="mono" style="font-size:22px;font-weight:500;margin-top:4px;letter-spacing:-0.01em">
              <span id="db-value">—</span>
              <span style="color:var(--ink-4);font-size:13px"> dB</span>
            </div>
          </div>
          <div style="display:flex;gap:8px;align-items:center">
            <select class="wa-select" id="device-select" style="width:260px">
              <option value="">Loading devices…</option>
            </select>
            <button class="wa-btn" id="check-levels-btn">Check levels</button>
          </div>
        </div>
        <div id="waveform-container"></div>
        <div class="mono" style="display:flex;justify-content:space-between;margin-top:8px;font-size:10.5px;color:var(--ink-4)">
          <span>00:00</span><span id="elapsed-mid">—</span><span id="elapsed-total">—</span>
        </div>
      </div>

      <!-- Transcript + side panel -->
      <div style="display:grid;grid-template-columns:1.7fr 1fr;gap:20px">
        <div class="wa-card" style="padding:0;overflow:hidden;display:flex;flex-direction:column;min-height:360px">
          <div style="display:flex;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--rule)">
            <div class="wa-card-title">Live transcript</div>
            <div class="mono" style="font-size:10.5px;color:var(--ink-4)" id="vad-meta">VAD · ready</div>
          </div>
          <div id="transcript-body" style="padding:18px 22px;display:flex;flex-direction:column;gap:14px;font-size:14px;line-height:1.55;color:var(--ink);flex:1;overflow-y:auto">
            <p style="color:var(--ink-4);font-style:italic">Transcript will appear here…</p>
          </div>
        </div>

        <div style="display:flex;flex-direction:column;gap:20px">
          <div class="wa-card">
            <div class="wa-card-head">
              <div class="wa-card-title">Session</div>
              <div class="wa-card-sub" id="session-sub">—</div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:12px">
              <div><div class="wa-label">Latency</div><div class="mono" style="font-size:13px" id="stat-latency">—</div></div>
              <div><div class="wa-label">Words</div><div class="mono" style="font-size:13px" id="stat-words">0</div></div>
              <div><div class="wa-label">Segments</div><div class="mono" style="font-size:13px" id="stat-segs">0</div></div>
              <div><div class="wa-label">Output</div><div class="mono" style="font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" id="stat-output">~/Downloads</div></div>
            </div>
            <div style="margin-top:16px">
              <span class="wa-label">Save formats</span>
              <div style="display:flex;gap:6px;flex-wrap:wrap" id="live-formats">
                ${['txt','srt','vtt','json','tsv'].map((f,i)=>`<span class="wa-chip${i===0?' on':''}" data-fmt="${f}">${f}</span>`).join('')}
              </div>
            </div>
          </div>

          <div class="wa-card">
            <div class="wa-card-head"><div class="wa-card-title">Controls</div></div>
            <div style="display:flex;flex-direction:column;gap:8px">
              <button class="wa-btn wa-btn-primary" id="record-btn" style="height:44px;font-size:14px">
                Start recording
              </button>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <button class="wa-btn" id="pause-btn" disabled>Pause</button>
                <button class="wa-btn" id="clear-btn">Clear</button>
              </div>
              <button class="wa-btn" id="polish-btn" disabled style="height:40px">
                Polish · align + diarize
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Initial waveform render
  renderWaveform(container.querySelector('#waveform-container'), { height: 64, bars: 120 });

  // Load devices
  api.getAudioDevices().then(devices => {
    const sel = container.querySelector('#device-select');
    sel.innerHTML = devices.length
      ? devices.map(d => `<option value="${d.index}">${d.name} (${d.sample_rate}Hz)</option>`).join('')
      : `<option value="">No input devices found</option>`;
    updateSubTitle(sel.options[sel.selectedIndex]?.text || '');
  }).catch(() => {});

  // Format chips
  const liveFmts = new Set(['txt']);
  container.querySelector('#live-formats').addEventListener('click', e => {
    const chip = e.target.closest('[data-fmt]');
    if (!chip) return;
    const f = chip.dataset.fmt;
    if (liveFmts.has(f)) liveFmts.delete(f); else liveFmts.add(f);
    chip.classList.toggle('on', liveFmts.has(f));
  });

  // Record button
  container.querySelector('#record-btn').addEventListener('click', () => {
    if (_session) stopRecording(container);
    else startRecording(container, liveFmts);
  });

  container.querySelector('#clear-btn').addEventListener('click', () => {
    _transcript = []; _partial = '';
    updateTranscriptUI(container);
  });
}

function updateSubTitle(deviceName) {
  const el = document.getElementById('live-sub');
  if (el && deviceName) el.textContent = deviceName;
}

async function startRecording(container, formats) {
  const deviceIndex = parseInt(container.querySelector('#device-select').value || '0');

  const streamRes = await api.streamStart({ model: 'base' });
  _session = streamRes.session_id;
  _elapsed = 0;

  const btn = container.querySelector('#record-btn');
  btn.className = 'wa-btn wa-btn-danger';
  btn.style.height = '44px';
  btn.style.fontSize = '14px';
  btn.innerHTML = `<span style="width:10px;height:10px;border-radius:2px;background:var(--signal-rec)"></span> Stop &amp; save`;

  container.querySelector('#pause-btn').disabled = false;
  container.querySelector('#polish-btn').disabled = true;

  // Update meta with recording indicator
  document.getElementById('live-meta').innerHTML = `
    <span style="display:inline-flex;align-items:center;gap:6px">
      <span style="width:8px;height:8px;border-radius:50%;background:var(--signal-rec);
        box-shadow:0 0 0 4px color-mix(in oklch,var(--signal-rec) 18%,transparent);
        animation:pulse-ring 1.6s ease-in-out infinite"></span>
      <span style="color:var(--signal-rec);font-weight:500">Recording</span>
    </span>
    <span style="color:var(--ink-4)">·</span>
    <span class="mono" id="elapsed-display">00:00:00</span>
  `;

  _elapsedTimer = setInterval(() => {
    _elapsed++;
    const h = String(Math.floor(_elapsed/3600)).padStart(2,'0');
    const m = String(Math.floor((_elapsed%3600)/60)).padStart(2,'0');
    const s = String(_elapsed%60).padStart(2,'0');
    const el = document.getElementById('elapsed-display');
    if (el) el.textContent = `${h}:${m}:${s}`;
  }, 1000);

  // Get mic stream
  try {
    _mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: deviceIndex } });
    _audioCtx = new AudioContext({ sampleRate: 16000 });
    const src = _audioCtx.createMediaStreamSource(_mediaStream);
    _processor = _audioCtx.createScriptProcessor(4096, 1, 1);
    src.connect(_processor);
    _processor.connect(_audioCtx.destination);
    _processor.onaudioprocess = async e => {
      if (!_session) return;
      const pcm = e.inputBuffer.getChannelData(0);
      const b64 = float32ToBase64(pcm);
      try {
        const res = await api.streamChunk({ session_id: _session, audio_b64: b64 });
        if (res.new_text) {
          _transcript.push({ t: formatTime(_elapsed), txt: res.new_text });
          _partial = '';
        }
        if (res.partial) _partial = res.partial;
        updateTranscriptUI(container);
        const words = _transcript.reduce((n, l) => n + l.txt.split(/\s+/).length, 0);
        const segs  = _transcript.length;
        document.getElementById('stat-words').textContent = words;
        document.getElementById('stat-segs').textContent  = segs;
      } catch { /* chunk errors are non-fatal */ }
    };
  } catch (err) {
    alert('Microphone access denied: ' + err.message);
    await stopRecording(container);
  }
}

async function stopRecording(container) {
  clearInterval(_elapsedTimer);
  if (_processor) { _processor.disconnect(); _processor = null; }
  if (_audioCtx)  { await _audioCtx.close(); _audioCtx = null; }
  if (_mediaStream) { _mediaStream.getTracks().forEach(t => t.stop()); _mediaStream = null; }

  const sid = _session;
  _session = null;

  if (sid) {
    try { await api.streamStop({ session_id: sid }); } catch { /* best effort */ }
  }

  const btn = container.querySelector('#record-btn');
  btn.className = 'wa-btn wa-btn-primary';
  btn.innerHTML = 'Start recording';
  container.querySelector('#pause-btn').disabled = true;
  container.querySelector('#polish-btn').disabled = false;
  document.getElementById('live-meta').innerHTML = '';
}

function updateTranscriptUI(container) {
  const body = container.querySelector('#transcript-body');
  if (!body) return;
  const lines = _transcript.map(l => `
    <div class="transcript-line">
      <span class="transcript-ts mono">${l.t}</span>
      <span>${escHtml(l.txt)}</span>
    </div>`).join('');
  const partial = _partial
    ? `<p style="color:var(--ink-3);font-style:italic">
        …<span class="transcript-partial-word">${escHtml(_partial)}</span>
        <span class="transcript-typing"> ·typing·</span>
       </p>`
    : '';
  body.innerHTML = lines + partial || `<p style="color:var(--ink-4);font-style:italic">Transcript will appear here…</p>`;
  body.scrollTop = body.scrollHeight;
}

function float32ToBase64(f32) {
  const buf = new Uint8Array(f32.buffer);
  let bin = '';
  for (let i = 0; i < buf.length; i++) bin += String.fromCharCode(buf[i]);
  return btoa(bin);
}

function formatTime(secs) {
  const h = String(Math.floor(secs/3600)).padStart(2,'0');
  const m = String(Math.floor((secs%3600)/60)).padStart(2,'0');
  const s = String(secs%60).padStart(2,'0');
  return `${h}:${m}:${s}`;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
```

- [ ] **Step 2: Verify in browser**

Navigate to Live tab. Expected: device selector populates, waveform renders, clicking "Start recording" prompts mic permission, recording indicator + elapsed timer appear in topbar meta, transcript populates as you speak, "Stop & save" ends recording.

- [ ] **Step 3: Commit**

```bash
git add whisperapp/static/js/live.js
git commit -m "feat: Live screen with microphone recording and streaming transcript"
```

---

## Task 8: Speakers screen

**Files:**
- Write: `whisperapp/static/js/speakers.js`

- [ ] **Step 1: Write `whisperapp/static/js/speakers.js`**

```js
import { api } from './api.js';

const SPEAKER_COLORS = [
  'oklch(60% 0.13 45)',
  'oklch(58% 0.11 220)',
  'oklch(60% 0.10 145)',
  'oklch(58% 0.09 290)',
];

export function initSpeakers(container) {
  container.innerHTML = `
    <div class="wa-topbar">
      <div>
        <h1>Speakers</h1>
        <div class="wa-topbar-sub mono" id="speakers-sub">Select a completed job with speaker review pending</div>
      </div>
      <div class="wa-topbar-meta mono" id="speakers-meta"></div>
    </div>
    <div class="wa-content" id="speakers-content">
      <div id="speakers-job-picker" style="display:flex;flex-direction:column;gap:12px;max-width:520px">
        <p style="color:var(--ink-3);font-size:13px">Jobs awaiting speaker review:</p>
        <div id="speakers-job-list"><div style="color:var(--ink-4);font-size:12.5px">Loading…</div></div>
      </div>
      <div id="speakers-review" style="display:none;grid-template-columns:320px 1fr;gap:24px"></div>
    </div>
  `;

  loadPendingJobs(container);
}

async function loadPendingJobs(container) {
  try {
    const jobs = await api.listJobs('speaker_review');
    const list = container.querySelector('#speakers-job-list');
    if (!jobs.length) {
      list.innerHTML = `<div style="color:var(--ink-4);font-size:12.5px">No jobs awaiting speaker review.</div>`;
      return;
    }
    list.innerHTML = jobs.map(j => `
      <div class="queue-item" style="cursor:pointer" data-job-id="${j.id}">
        <div class="queue-item-header">
          <span class="queue-dot done"></span>
          <span class="queue-item-title">${escHtml(j.file_name || j.id)}</span>
          <span class="queue-badge mono">${j.id.slice(0,8)}</span>
        </div>
        <div class="queue-meta mono">${j.model}${j.diarize?' · diarize':''}</div>
      </div>
    `).join('');
    list.querySelectorAll('[data-job-id]').forEach(el => {
      el.addEventListener('click', () => openJob(container, el.dataset.jobId));
    });
  } catch { /* server may not be up yet */ }
}

async function openJob(container, jobId) {
  const job = await api.getJob(jobId);
  const data = await api.getSpeakers(jobId);
  const speakers = data.speakers || [];

  container.querySelector('#speakers-sub').textContent =
    `${job.file_name || jobId} · ${speakers.length} speakers detected`;
  container.querySelector('#speakers-meta').textContent =
    `${job.id.slice(0,8)} · ${job.model}`;

  container.querySelector('#speakers-job-picker').style.display = 'none';
  const review = container.querySelector('#speakers-review');
  review.style.display = 'grid';

  const names = Object.fromEntries(speakers.map(s => [s.id, s.name || '']));
  let activeSpeaker = speakers[0]?.id;

  function render() {
    review.innerHTML = `
      <!-- Voices column -->
      <div class="wa-card" style="padding:0;overflow:hidden;display:flex;flex-direction:column">
        <div style="padding:16px 18px;border-bottom:1px solid var(--rule);display:flex;justify-content:space-between;align-items:baseline">
          <div class="wa-card-title">Voices</div>
          <div class="wa-card-sub">${speakers.length}</div>
        </div>
        <div style="flex:1;overflow-y:auto">
          ${speakers.map((s, i) => {
            const color = SPEAKER_COLORS[i % SPEAKER_COLORS.length];
            const name  = names[s.id] || '';
            const initial = name ? name[0].toUpperCase() : '?';
            return `
              <div class="speaker-row${s.id === activeSpeaker ? ' active' : ''}" data-speaker="${s.id}">
                <div class="speaker-avatar" style="background:${color}">${initial}</div>
                <div style="flex:1;min-width:0">
                  <div style="font-size:13.5px;font-weight:500">
                    ${name || `<span style="color:var(--ink-4);font-style:italic;font-weight:400">Unnamed</span>`}
                  </div>
                  <div class="mono" style="font-size:10.5px;color:var(--ink-3);margin-top:1px">
                    ${s.id} · ${s.lines ?? '?'} lines
                  </div>
                </div>
                ${s.id === activeSpeaker ? `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="var(--ink)" stroke-width="1.6" stroke-linecap="round"><path d="M5 8l2 2 4-4"/></svg>` : ''}
              </div>
            `;
          }).join('')}
        </div>
        <div style="padding:14px;border-top:1px solid var(--rule);display:flex;gap:8px">
          <button class="wa-btn" id="skip-btn" style="flex:1">Skip</button>
          <button class="wa-btn wa-btn-primary" id="confirm-btn" style="flex:2">Confirm names</button>
        </div>
      </div>

      <!-- Review pane -->
      <div class="wa-card" style="padding:0;overflow:hidden;display:flex;flex-direction:column">
        ${renderReviewPane(speakers.find(s => s.id === activeSpeaker), names, activeSpeaker)}
      </div>
    `;

    // Wire up speaker row clicks
    review.querySelectorAll('[data-speaker]').forEach(el => {
      el.addEventListener('click', () => {
        activeSpeaker = el.dataset.speaker;
        render();
      });
    });

    // Name editing in review pane
    const nameInput = review.querySelector('#speaker-name-input');
    if (nameInput) {
      nameInput.addEventListener('input', e => {
        names[activeSpeaker] = e.target.value;
      });
    }

    review.querySelector('#confirm-btn').addEventListener('click', async () => {
      await api.confirmSpeakers(jobId, names);
      review.style.display = 'none';
      container.querySelector('#speakers-job-picker').style.display = 'flex';
      loadPendingJobs(container);
    });

    review.querySelector('#skip-btn').addEventListener('click', () => {
      review.style.display = 'none';
      container.querySelector('#speakers-job-picker').style.display = 'flex';
      loadPendingJobs(container);
    });
  }

  render();
}

function renderReviewPane(speaker, names, speakerId) {
  if (!speaker) return `<div style="padding:20px;color:var(--ink-4)">Select a speaker</div>`;
  const name    = names[speakerId] || '';
  const snippets = speaker.snippets || [];
  return `
    <div style="padding:16px 22px;border-bottom:1px solid var(--rule);display:flex;align-items:center;justify-content:space-between">
      <div>
        <div class="wa-card-title">
          <input id="speaker-name-input" value="${escHtml(name)}" placeholder="Enter name…"
            style="border:none;background:transparent;font-size:13px;font-weight:600;color:var(--ink);outline:none;width:160px">
          <span style="font-weight:400;color:var(--ink-3)"> · ${speakerId}</span>
        </div>
        <div class="mono" style="font-size:10.5px;color:var(--ink-3);margin-top:2px">
          ${snippets.length} representative snippet${snippets.length !== 1 ? 's' : ''}
        </div>
      </div>
    </div>
    <div style="padding:20px 22px;display:flex;flex-direction:column;gap:14px;flex:1;overflow-y:auto">
      ${snippets.map(s => `
        <div class="speaker-snippet">
          <button class="snippet-play" title="Play">
            <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor"><path d="M5 3.5v9l8-4.5z"/></svg>
          </button>
          <div style="flex:1">
            <div class="mono" style="font-size:10.5px;color:var(--ink-3);margin-bottom:4px">${escHtml(s.start ?? '')} → ${escHtml(s.end ?? '')}</div>
            <div style="font-size:13.5px;line-height:1.5">${escHtml(s.text ?? '')}</div>
          </div>
        </div>
      `).join('')}
    </div>
    <div style="margin-top:auto;padding:14px 22px;border-top:1px solid var(--rule)">
      <div class="mono" style="font-size:11px;color:var(--ink-3)">
        Tip: paste <span style="color:var(--ink-2)">SPEAKER_00=Alice</span> lines to rename in bulk
      </div>
    </div>
  `;
}

function escHtml(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
```

- [ ] **Step 2: Verify in browser**

Navigate to Speakers tab. Expected: list of jobs in `speaker_review` state (or empty-state message). Click a job → two-column layout with voice rows and snippet pane. Click speaker row → switches review pane. Edit name in-place. Confirm → calls API and returns to job picker.

- [ ] **Step 3: Commit**

```bash
git add whisperapp/static/js/speakers.js
git commit -m "feat: Speakers screen with voice list and snippet review"
```

---

## Task 9: Settings screen

**Files:**
- Write: `whisperapp/static/js/settings.js`

- [ ] **Step 1: Write `whisperapp/static/js/settings.js`**

```js
import { api } from './api.js';

const SECTIONS = ['Transcription', 'AI features', 'Startup', 'API & CLI', 'About'];
const PROVIDERS = [
  { id: 'none',   label: 'None',   sub: 'manual labels only' },
  { id: 'claude', label: 'Claude', sub: 'anthropic · cloud' },
  { id: 'openai', label: 'OpenAI', sub: 'gpt-4o · cloud' },
  { id: 'ollama', label: 'Ollama', sub: 'local · no key needed' },
];

export function initSettings(container) {
  container.innerHTML = `
    <div class="wa-topbar">
      <div>
        <h1>Settings</h1>
        <div class="wa-topbar-sub mono">~/.whisperapp/config.json</div>
      </div>
      <div class="wa-topbar-meta mono" id="settings-meta"></div>
    </div>
    <div class="wa-content" style="display:grid;grid-template-columns:180px 1fr;gap:28px;max-width:980px;align-content:start">
      <div class="settings-subnav" id="settings-subnav">
        ${SECTIONS.map((s, i) => `
          <div class="settings-subnav-item${i===0?' active':''}" data-section="${i}">${s}</div>
        `).join('')}
      </div>
      <div id="settings-body" style="display:flex;flex-direction:column;gap:24px"></div>
    </div>
  `;

  let cfg = {};
  let activeSection = 0;

  async function load() {
    try {
      cfg = await api.getConfig();
      render();
    } catch { /* server may not be ready */ }
  }

  function render() {
    const body = container.querySelector('#settings-body');
    body.innerHTML = '';

    if (activeSection === 0) renderTranscription(body, cfg);
    if (activeSection === 1) renderAI(body, cfg);
    if (activeSection === 2) renderStartup(body, cfg);
    if (activeSection === 3) renderAPICLI(body);
    if (activeSection === 4) renderAbout(body);

    if (activeSection < 3) renderSaveRow(body, cfg);
  }

  container.querySelector('#settings-subnav').addEventListener('click', e => {
    const item = e.target.closest('[data-section]');
    if (!item) return;
    activeSection = parseInt(item.dataset.section);
    container.querySelectorAll('.settings-subnav-item').forEach((el, i) => {
      el.classList.toggle('active', i === activeSection);
    });
    render();
  });

  load();
}

function field(label, inputHtml, hint = '') {
  return `
    <div>
      <span class="wa-label">${label}</span>
      ${inputHtml}
      ${hint ? `<div class="wa-field-hint">${hint}</div>` : ''}
    </div>`;
}

function toggleRow(label, checked, key) {
  return `
    <label class="toggle-row">
      <span class="wa-toggle${checked?' on':''}" data-toggle="${key}"></span>
      <span>${label}</span>
    </label>`;
}

function renderTranscription(body, cfg) {
  body.innerHTML += `
    <section class="settings-section">
      <header class="settings-section-head">
        <h2>Transcription</h2>
        <span class="wa-card-sub mono">WhisperX &amp; diarization</span>
      </header>
      ${field('HuggingFace token',
        `<input class="wa-input" type="password" data-cfg="hf_token" value="${escHtml(cfg.hf_token||'')}">`,
        'Required for pyannote.audio diarization. Stored locally — never sent off-device.'
      )}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        ${field('Default model',
          `<select class="wa-select" data-cfg="default_model">
            ${['tiny','base','small','medium','large-v2','large-v3'].map(m=>
              `<option${cfg.default_model===m?' selected':''}>${m}</option>`).join('')}
          </select>`
        )}
        ${field('Streaming model',
          `<select class="wa-select" data-cfg="streaming_model">
            ${['tiny','base','small'].map(m=>
              `<option${cfg.streaming_model===m?' selected':''}>${m}</option>`).join('')}
          </select>`
        )}
      </div>
      ${field('Default output path',
        `<input class="wa-input" data-cfg="default_output_path" value="${escHtml(cfg.default_output_path||'')}">`
      )}
      ${toggleRow('Diarization on by default', cfg.diarize_by_default, 'diarize_by_default')}
    </section>`;

  wireToggles(body, cfg);
  wireInputs(body, cfg);
}

function renderAI(body, cfg) {
  body.innerHTML += `
    <section class="settings-section">
      <header class="settings-section-head">
        <h2>AI features</h2>
        <span class="wa-card-sub mono">Optional · speaker ID &amp; meeting notes</span>
      </header>
      <p style="font-size:12.5px;color:var(--ink-3);line-height:1.5">
        Enable a provider for automatic speaker identification, meeting notes, and live summaries.
        All transcription works without AI.
      </p>
      <div class="provider-grid" id="provider-grid">
        ${PROVIDERS.map(p => `
          <div class="provider-card${cfg.ai_provider===p.id?' active':''}" data-provider="${p.id}">
            <div class="provider-card-name">${p.label}</div>
            <div class="provider-card-sub mono">${p.sub}</div>
          </div>`).join('')}
      </div>
      ${field('API key',
        `<input class="wa-input" type="password" data-cfg="ai_api_key" value="${escHtml(cfg.ai_api_key||'')}">`
      )}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        ${field('Model', `<input class="wa-input" data-cfg="ai_model" value="${escHtml(cfg.ai_model||'')}" placeholder="(provider default)">`)}
        ${field('Base URL', `<input class="wa-input" data-cfg="ai_base_url" value="${escHtml(cfg.ai_base_url||'')}" placeholder="https://api.anthropic.com">`)}
      </div>
    </section>`;

  wireInputs(body, cfg);

  body.querySelector('#provider-grid').addEventListener('click', e => {
    const card = e.target.closest('[data-provider]');
    if (!card) return;
    cfg.ai_provider = card.dataset.provider;
    body.querySelectorAll('.provider-card').forEach(c => {
      c.classList.toggle('active', c.dataset.provider === cfg.ai_provider);
    });
  });
}

function renderStartup(body, cfg) {
  body.innerHTML += `
    <section class="settings-section">
      <header class="settings-section-head"><h2>Startup</h2></header>
      ${toggleRow('Launch on login', false, 'launch_on_login')}
      ${toggleRow('Auto-update WhisperX on startup', true, 'auto_update')}
    </section>`;
  wireToggles(body, cfg);
}

function renderAPICLI(body) {
  body.innerHTML += `
    <section class="settings-section">
      <header class="settings-section-head">
        <h2>API &amp; CLI</h2>
        <span class="wa-card-sub mono">Local-only · 127.0.0.1</span>
      </header>
      <div style="display:flex;flex-direction:column;gap:8px;font-size:13px;color:var(--ink-2)">
        ${field('REST API', `<input class="wa-input" value="http://127.0.0.1:7861" readonly>`)}
        ${field('API docs', `<a href="http://127.0.0.1:7861/docs" target="_blank" class="wa-btn" style="display:inline-flex;margin-top:4px">Open API docs →</a>`)}
      </div>
      <div style="margin-top:12px">
        <span class="wa-label">CLI usage</span>
        <div class="wa-card" style="background:var(--paper-2);padding:14px;font-family:var(--font-mono);font-size:12px;color:var(--ink-2);line-height:1.7">
          whisperapp transcribe recording.mp3 -m large-v2 --diarize<br>
          whisperapp status &lt;job-id&gt;<br>
          whisperapp list
        </div>
      </div>
    </section>`;
}

function renderAbout(body) {
  body.innerHTML += `
    <section class="settings-section">
      <header class="settings-section-head"><h2>About</h2></header>
      <div style="display:flex;flex-direction:column;gap:8px;font-size:13px;color:var(--ink-2)">
        <div><span style="color:var(--ink-3);font-family:var(--font-mono);font-size:11px">VERSION</span><br>WhisperApp v1.1.0</div>
        <div><span style="color:var(--ink-3);font-family:var(--font-mono);font-size:11px">ENGINE</span><br>WhisperX · pyannote.audio · faster-whisper</div>
        <div><span style="color:var(--ink-3);font-family:var(--font-mono);font-size:11px">CONFIG</span><br><span class="mono">~/.whisperapp/config.json</span></div>
      </div>
    </section>`;
}

function renderSaveRow(body, cfg) {
  body.innerHTML += `
    <div style="display:flex;gap:8px;padding-top:8px;border-top:1px solid var(--rule)">
      <button class="wa-btn wa-btn-primary" id="save-btn">Save settings</button>
      <button class="wa-btn" id="test-ai-btn">Test AI connection</button>
      <button class="wa-btn" id="reset-btn" style="margin-left:auto;color:var(--signal-rec)">Reset to defaults</button>
    </div>`;

  body.querySelector('#save-btn').addEventListener('click', async () => {
    const btn = body.querySelector('#save-btn');
    btn.disabled = true; btn.textContent = 'Saving…';
    try {
      await api.updateConfig(cfg);
      btn.textContent = 'Saved ✓';
      setTimeout(() => { btn.disabled = false; btn.textContent = 'Save settings'; }, 1500);
    } catch {
      btn.textContent = 'Error saving';
      setTimeout(() => { btn.disabled = false; btn.textContent = 'Save settings'; }, 2000);
    }
  });

  body.querySelector('#test-ai-btn').addEventListener('click', async () => {
    try {
      const res = await api.health();
      alert(`AI: ${res.ai_provider} — ${res.ai_available ? 'available' : 'not configured'}`);
    } catch {
      alert('Could not reach REST API at :7861');
    }
  });
}

function wireInputs(body, cfg) {
  body.querySelectorAll('[data-cfg]').forEach(el => {
    el.addEventListener('input', () => { cfg[el.dataset.cfg] = el.value; });
    el.addEventListener('change', () => { cfg[el.dataset.cfg] = el.value; });
  });
}

function wireToggles(body, cfg) {
  body.querySelectorAll('[data-toggle]').forEach(tog => {
    tog.addEventListener('click', () => {
      const on = tog.classList.toggle('on');
      const key = tog.dataset.toggle;
      if (key in cfg) cfg[key] = on;
    });
  });
}

function escHtml(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
```

- [ ] **Step 2: Verify in browser**

Navigate to Settings tab. Expected: sub-nav works, HF token field shows masked value, provider cards are selectable, toggles work. Click "Save settings" → success flash. Click "Test AI connection" → alert with provider status.

- [ ] **Step 3: Commit**

```bash
git add whisperapp/static/js/settings.js
git commit -m "feat: Settings screen with config load/save"
```

---

## Task 10: Full test suite and cleanup

**Files:**
- Modify: `tests/test_ui.py` (expand)
- Run: full suite

- [ ] **Step 1: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -q
```
Expected: all previous tests still pass (138+).

- [ ] **Step 2: Add a smoke test that the static server sends correct Content-Type headers**

Add to `tests/test_ui.py`:

```python
@pytest.mark.asyncio
async def test_ui_js_content_type(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/js/marks.js")
    assert r.status_code == 200
    assert "javascript" in r.headers.get("content-type", "")
```

- [ ] **Step 3: Run again**

```bash
.venv/bin/python -m pytest tests/ -q
```

- [ ] **Step 4: Final commit**

```bash
git add tests/
git commit -m "test: add static server content-type smoke test"
```

---

## Self-review against spec

### Spec coverage checklist

| Requirement | Task |
|------------|------|
| Custom 36px titlebar with 3-dot controls | Task 4 (index.html + app.css) |
| Light + dark themes with `data-theme` toggle | Task 5 (app.js theme toggle) |
| Design tokens (OKLCH) | Task 3 (tokens.css) |
| Inter Tight + JetBrains Mono fonts | Task 3 (index.html) |
| Sidebar with brand, nav, recent, status bar | Task 4 (shell.js) |
| Transcribe: drop zone, pipeline card, queue | Task 6 |
| Live: waveform, transcript, controls | Task 7 |
| Speakers: voice list, snippet review, rename | Task 8 |
| Settings: all sections, provider cards, save | Task 9 |
| REST API: config GET/POST, devices, upload | Task 1 |
| Static file server replaces Gradio | Task 2 |
| Queue polling with animated status dots | Task 6 (3s poll + CSS animation) |
| Speaker avatar colors (4-color palette) | Task 8 |
| Deterministic waveform bars | Task 4 (shell.js renderWaveform) |
| Theme persistence (localStorage) | Task 5 (app.js) |
| Recent jobs in sidebar | Task 4 (shell.js, from localStorage) |
| GPU/acceleration in status bar | Task 4 (shell.js fetches /info) |
