import sys
import threading
import webbrowser
import pystray
from PIL import Image, ImageDraw
from whisperapp.startup import register_startup, unregister_startup


# ---------------------------------------------------------------------------
# Icon factory
# ---------------------------------------------------------------------------

def _make_icon(bg: str, dot: str = "white") -> Image.Image:
    """64×64 circle icon: coloured ring on bg, white dot in centre."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([2, 2, 62, 62], fill=bg)
    d.ellipse([20, 20, 44, 44], fill=dot)
    return img


ICONS = {
    "idle":    _make_icon("#888888"),
    "ready":   _make_icon("#22aa44"),
    "working": _make_icon("#2255cc"),
    "error":   _make_icon("#cc2222"),
}


# ---------------------------------------------------------------------------
# TrayApp
# ---------------------------------------------------------------------------

class TrayApp:
    def __init__(self, queue, worker):
        self.queue = queue
        self.worker = worker
        self._icon = None
        self._startup_enabled = self._check_startup_registered()

    # ------------------------------------------------------------------
    # Startup registration helpers
    # ------------------------------------------------------------------

    def _check_startup_registered(self) -> bool:
        try:
            if sys.platform == "win32":
                import winreg as _winreg
                key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
                with _winreg.OpenKey(_winreg.HKEY_CURRENT_USER, key_path) as k:
                    _winreg.QueryValueEx(k, "WhisperApp")
                return True
            elif sys.platform == "darwin":
                from pathlib import Path
                plist = Path.home() / "Library/LaunchAgents/com.whisperapp.plist"
                return plist.exists()
        except Exception:
            pass
        return False

    def _toggle_startup(self, icon, item):
        if self._startup_enabled:
            unregister_startup()
            self._startup_enabled = False
        else:
            register_startup()
            self._startup_enabled = True
        # Rebuild menu to reflect new state
        icon.menu = self._build_menu()
        icon.update_menu()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self):
        startup_label = (
            "✓ Start on login" if self._startup_enabled else "Start on login"
        )
        # pystray doesn't render unicode on all platforms — use plain text fallback
        if sys.platform == "win32":
            startup_label = (
                "[ON] Start on login" if self._startup_enabled
                else "[ ]  Start on login"
            )

        active = len([j for j in self.queue.list_jobs(limit=50)
                      if j["status"] == "running"])
        status_label = (
            f"{active} job{'s' if active != 1 else ''} running"
            if active else "Idle"
        )

        return pystray.Menu(
            pystray.MenuItem("WhisperApp", None, enabled=False),
            pystray.MenuItem(status_label, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open UI", self._open_ui, default=True),
            pystray.MenuItem("Open API docs", self._open_api),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(startup_label, self._toggle_startup),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open_ui(self, icon=None, item=None):
        webbrowser.open("http://127.0.0.1:7860")

    def _open_api(self, icon=None, item=None):
        webbrowser.open("http://127.0.0.1:7861/docs")

    def _quit(self, icon, item):
        icon.stop()

    # ------------------------------------------------------------------
    # Status polling
    # ------------------------------------------------------------------

    def get_title(self, active_jobs: int) -> str:
        if active_jobs == 0:
            return "WhisperApp — Idle"
        return f"WhisperApp — {active_jobs} job{'s' if active_jobs != 1 else ''} running"

    def _poll_status(self):
        """Background thread: update icon and tooltip every 5 seconds."""
        import time
        while self._icon and self._icon.visible:
            try:
                jobs = self.queue.list_jobs(limit=50)
                active = len([j for j in jobs if j["status"] == "running"])
                icon_key = "working" if active else "ready"
                self._icon.icon = ICONS[icon_key]
                self._icon.title = self.get_title(active)
                self._icon.menu = self._build_menu()
                self._icon.update_menu()
            except Exception:
                pass
            time.sleep(5)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        self._icon = pystray.Icon(
            "WhisperApp",
            ICONS["ready"],
            self.get_title(0),
            menu=self._build_menu(),
        )

        threading.Thread(target=self._poll_status, daemon=True).start()

        # On macOS, pystray uses the menu bar automatically via rumps/AppKit.
        # On Windows, it appears in the system tray notification area.
        self._icon.run()
