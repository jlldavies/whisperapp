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
from whisperapp.watcher import MeetingWatcher


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

    try:
        watcher = MeetingWatcher(on_trigger=lambda s, n: None)
    except Exception:
        watcher = None

    TrayApp(queue=queue, worker=worker, watcher=watcher).run()


if __name__ == "__main__":
    main()
