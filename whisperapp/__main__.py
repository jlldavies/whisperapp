import threading
import uvicorn
from whisperapp.config import Config
from whisperapp.queue import JobQueue
from whisperapp.worker import Worker
from whisperapp.server import create_app
from whisperapp.ui import create_ui
from whisperapp.tray import TrayApp
from whisperapp.updater import run_update

def main():
    # Auto-update in background so it doesn't block startup
    threading.Thread(target=run_update, daemon=True).start()

    cfg = Config()
    queue = JobQueue()
    worker = Worker(queue=queue, hf_token=cfg.hf_token)
    worker.start()

    # MCP server thread
    mcp_app = create_app(queue=queue, worker=worker)
    threading.Thread(
        target=lambda: uvicorn.run(
            mcp_app, host="127.0.0.1", port=7861, log_level="warning"),
        daemon=True
    ).start()

    # Gradio UI thread
    ui = create_ui(queue=queue, worker=worker)
    threading.Thread(
        target=lambda: ui.launch(
            server_name="127.0.0.1", server_port=7860,
            quiet=True, prevent_thread_lock=True),
        daemon=True
    ).start()

    # First-run check
    if not cfg.hf_token:
        print("First run - open http://127.0.0.1:7860 to complete setup.")

    # Tray (blocks main thread)
    TrayApp(queue=queue, worker=worker).run()

if __name__ == "__main__":
    main()
