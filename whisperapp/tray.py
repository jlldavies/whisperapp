import threading
import webbrowser
import pystray
from PIL import Image, ImageDraw

def _make_icon(color: str) -> Image.Image:
    img = Image.new("RGB", (64, 64), color=color)
    d = ImageDraw.Draw(img)
    d.ellipse([8, 8, 56, 56], fill="white")
    return img

ICONS = {
    "idle":    _make_icon("#888888"),
    "ready":   _make_icon("#22aa44"),
    "working": _make_icon("#2255cc"),
    "updating":_make_icon("#dd9900"),
}

class TrayApp:
    def __init__(self, queue, worker):
        self.queue = queue
        self.worker = worker
        self._icon = None

    def get_title(self, active_jobs: int) -> str:
        if active_jobs == 0:
            return "WhisperApp — Idle"
        return f"WhisperApp — {active_jobs} job{'s' if active_jobs != 1 else ''} running"

    def _open_ui(self, icon=None, item=None):
        webbrowser.open("http://127.0.0.1:7860")

    def _quit(self, icon, item):
        icon.stop()

    def run(self):
        menu = pystray.Menu(
            pystray.MenuItem("Open UI", self._open_ui, default=True),
            pystray.MenuItem("Quit", self._quit),
        )
        self._icon = pystray.Icon(
            "WhisperApp", ICONS["ready"], "WhisperApp", menu)
        self._icon.run()
