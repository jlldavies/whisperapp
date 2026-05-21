import atexit
import os
import sys
import threading
import webbrowser

# Windows: pythonw.exe has no console, so sys.stdout/stderr are None. The ML
# import stack (torchcodec/torchaudio) emits warnings during import; writing
# them to dead streams blocks startup and hangs the app before the server
# starts. Give the process a real file-backed stream. Guarded so it is inert
# on macOS and on any console run.
if sys.platform == "win32" and sys.stderr is None:
    _stdio_dir = os.path.join(os.path.expanduser("~"), ".whisperapp")
    os.makedirs(_stdio_dir, exist_ok=True)
    _stdio_log = open(
        os.path.join(_stdio_dir, "pythonw-stdio.log"),
        "a", buffering=1, encoding="utf-8", errors="replace",
    )
    sys.stdout = sys.stderr = _stdio_log

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

import logging
import logging.handlers
import uvicorn
from pathlib import Path
from whisperapp.config import Config, _config_dir


def _configure_logging() -> None:
    """Set up root logger: stderr for INFO+, rotating file for DEBUG+."""
    log_dir = Path.home() / ".whisperapp"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "whisperapp.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Rotating file — keep 5 × 2 MB = 10 MB max
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Console — INFO only (keeps startup output clean)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(ch)

    # Suppress noisy third-party loggers at DEBUG level
    for name in ("uvicorn.access", "httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# Windows: named mutex held for the lifetime of the process.
# Automatically released by the OS on crash/reboot — immune to PID reuse.
_win_mutex = None


def _acquire_instance_lock() -> bool:
    global _win_mutex
    if sys.platform == "win32":
        import ctypes
        _win_mutex = ctypes.windll.kernel32.CreateMutexW(None, True, "WhisperApp_SingleInstance")
        ERROR_ALREADY_EXISTS = 183
        if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            return False
        return True

    # Non-Windows: PID file (no PID-reuse risk in practice on POSIX)
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
    p.add_argument(
        "--skip-setup-check",
        action="store_true",
        default=False,
        help="Skip the ML dependency check (for development use)",
    )
    return p.parse_args()


def _run_setup(ui_port: int) -> None:
    """
    Serve the first-run setup page, block until install completes,
    then exec() the current process to restart cleanly with ML deps loaded.
    """
    from whisperapp.server import create_setup_app

    _done = threading.Event()

    def _on_restart():
        _done.set()

    setup_app = create_setup_app(on_restart=_on_restart)

    server_thread = threading.Thread(
        target=lambda: uvicorn.run(
            setup_app,
            host="127.0.0.1",
            port=ui_port,
            log_level="warning",
        ),
        daemon=True,
    )
    server_thread.start()

    # Give uvicorn a moment to bind
    import time
    time.sleep(1.0)

    url = f"http://127.0.0.1:{ui_port}/setup"
    print(f"WhisperApp — first run setup: {url}")
    webbrowser.open(url)

    _done.wait()

    # Restart the process — imports will now find the freshly installed packages
    print("Restarting WhisperApp...")
    args = [a for a in sys.argv if a != "--skip-setup-check"]
    os.execv(sys.executable, [sys.executable] + args)


def main():
    _configure_logging()
    args = _parse_args()
    headless = args.headless
    api_host = args.host or ("0.0.0.0" if headless else "127.0.0.1")

    # ── ML dependency check ───────────────────────────────────────────────
    # Must happen BEFORE importing worker/server (which pull in torch/whisperx).
    # Uses importlib.util.find_spec — no import, so safe when packages are absent.
    if not args.skip_setup_check and not headless:
        from whisperapp.setup_env import needs_setup
        if needs_setup():
            _run_setup(args.ui_port)
            return  # exec() in _run_setup replaces this process; return is safety

    # ── Normal startup — ML deps confirmed present ────────────────────────
    from whisperapp.queue import JobQueue
    from whisperapp.worker import Worker
    from whisperapp.server import create_app
    from whisperapp.updater import run_update
    from whisperapp.watcher import MeetingWatcher

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
