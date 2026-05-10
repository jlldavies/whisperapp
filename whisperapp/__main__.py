import atexit
import os
import sys
import threading
import webbrowser

# Windows: route Python's ssl through the Windows certificate store. This
# fixes two problems in one go: (a) Python's bundled OpenSSL points at
# non-existent /Common Files/SSL/ paths so vanilla HTTPS fails, and (b) TLS
# inspectors like Norton/corporate proxies replace cert chains with their own
# roots, which are trusted by Windows but not by certifi. Must run before
# anything imports requests/urllib3/ssl.
if sys.platform == "win32":
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        try:
            import certifi
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
            os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())
        except ImportError:
            pass

import uvicorn
from pathlib import Path
from whisperapp.config import Config, _config_dir
from whisperapp.queue import JobQueue
from whisperapp.worker import Worker
from whisperapp.server import create_app
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


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        prog="whisperapp",
        description="WhisperApp — local speech transcription",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        default=os.environ.get("WHISPERAPP_HEADLESS", "").lower() in ("1", "true", "yes"),
        help="API server only — no tray icon, no browser, binds to 0.0.0.0. "
             "Also enabled by WHISPERAPP_HEADLESS=1.",
    )
    p.add_argument(
        "--host",
        default=None,
        help="Host to bind the API server (default: 127.0.0.1, or 0.0.0.0 in headless mode)",
    )
    p.add_argument(
        "--api-port", type=int, default=7861,
        help="REST API port (default: 7861)",
    )
    p.add_argument(
        "--ui-port", type=int, default=7860,
        help="Web UI port (default: 7860, desktop mode only)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    headless = args.headless
    api_host = args.host or ("0.0.0.0" if headless else "127.0.0.1")

    if not headless and not _acquire_instance_lock():
        print("WhisperApp is already running.")
        webbrowser.open(f"http://127.0.0.1:{args.ui_port}")
        sys.exit(0)

    cfg = Config()

    if getattr(cfg, "auto_update", True):
        threading.Thread(target=run_update, daemon=True).start()

    queue = JobQueue()
    worker = Worker(queue=queue, hf_token=cfg.hf_token)
    worker.start()

    api_app = create_app(queue=queue, worker=worker)

    if headless:
        # ── Headless / Docker / server mode ──────────────────────────────
        # Single blocking uvicorn process — no tray, no UI server, no browser.
        # All features accessible via REST API on api_host:api_port.
        print(f"WhisperApp API listening on http://{api_host}:{args.api_port}")
        print(f"Swagger docs: http://{api_host}:{args.api_port}/docs")
        uvicorn.run(
            api_app,
            host=api_host,
            port=args.api_port,
            log_level="info",
        )
    else:
        # ── Desktop mode — tray icon + UI server + API server ────────────
        # Defer tray/UI imports so the headless path never requires pystray
        # or Pillow (not installed in server/Docker deployments).
        from whisperapp.ui import create_ui
        try:
            from whisperapp.tray import TrayApp
            _tray_available = True
        except ImportError:
            _tray_available = False
            print("pystray not installed — running without tray icon. "
                  "Install with: pip install 'whisperapp[desktop]'")

        threading.Thread(
            target=lambda: uvicorn.run(
                api_app,
                host="127.0.0.1",
                port=args.api_port,
                log_level="warning",
            ),
            daemon=True,
        ).start()

        ui_app = create_ui(queue=queue, worker=worker)
        threading.Thread(
            target=lambda: uvicorn.run(
                ui_app,
                host="127.0.0.1",
                port=args.ui_port,
                log_level="warning",
            ),
            daemon=True,
        ).start()

        if not cfg.hf_token:
            print(f"First run — open http://127.0.0.1:{args.ui_port} to complete setup.")

        try:
            watcher = MeetingWatcher(on_trigger=lambda s, n: None)
        except Exception:
            watcher = None

        if _tray_available:
            TrayApp(queue=queue, worker=worker, watcher=watcher).run()
        else:
            # Keep the process alive without a tray
            import signal
            signal.pause()


if __name__ == "__main__":
    main()
