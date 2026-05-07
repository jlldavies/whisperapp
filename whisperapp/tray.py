import subprocess
import sys
import threading
import webbrowser
import pystray
from PIL import Image, ImageDraw
from whisperapp.startup import register_startup, unregister_startup


# ---------------------------------------------------------------------------
# Icon factory
# ---------------------------------------------------------------------------

def _make_icon(bg: str, fg: str = "#faf8f4") -> Image.Image:
    """64×64 brand mark: rounded square with 'w' waveform path."""
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Rounded square — matches SVG: 32×32 viewBox, rx=9, 1px inset → scale ×2
    d.rounded_rectangle([2, 2, 62, 62], radius=18, fill=bg)
    # 'w' waveform path: SVG coords ×2 (32→64 scale)
    # M8,12 L11,22 L14,14 L16,14 L18,22 L21,14 L24,22
    pts = [(16, 24), (22, 44), (28, 28), (32, 28), (36, 44), (42, 28), (48, 44)]
    d.line(pts, fill=fg, width=4, joint="curve")
    return img


ICONS = {
    "idle":    _make_icon("#888888"),
    "ready":   _make_icon("#5b9168"),   # --signal-go
    "working": _make_icon("#c96442"),   # --accent terracotta
    "error":   _make_icon("#c4523f"),   # --signal-rec
}


# ---------------------------------------------------------------------------
# TrayApp
# ---------------------------------------------------------------------------

class TrayApp:
    def __init__(self, queue, worker, watcher=None):
        self.queue = queue
        self.worker = worker
        self._watcher = watcher
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

        items = [
            pystray.MenuItem("WhisperApp", None, enabled=False),
            pystray.MenuItem(status_label, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open UI", self._open_ui, default=True),
            pystray.MenuItem("Open API docs", self._open_api),
            pystray.Menu.SEPARATOR,
        ]

        # Show dismiss only while a meeting/mic trigger is active and not yet dismissed
        if (self._watcher is not None
                and self._watcher.is_active
                and not self._watcher._dismissed):
            items.append(pystray.MenuItem("Dismiss meeting prompt", self._dismiss_watcher))
            items.append(pystray.Menu.SEPARATOR)

        items += [
            pystray.MenuItem(startup_label, self._toggle_startup),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        ]
        return pystray.Menu(*items)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open_ui(self, icon=None, item=None):
        webbrowser.open("http://127.0.0.1:7860")

    def _open_api(self, icon=None, item=None):
        webbrowser.open("http://127.0.0.1:7861/docs")

    def _quit(self, icon, item):
        icon.stop()

    def _notify(self, message: str, title: str = "WhisperApp") -> None:
        """Send a native OS notification. Best-effort — never raises."""
        try:
            if sys.platform == "darwin":
                safe_msg = message.replace('"', '\\"').replace("'", "\\'")
                safe_title = title.replace('"', '\\"').replace("'", "\\'")
                subprocess.run(
                    ["osascript", "-e",
                     f'display notification "{safe_msg}" with title "{safe_title}"'],
                    check=False, timeout=5,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                if self._icon:
                    self._icon.notify(message, title)
        except Exception:
            pass

    def _on_meeting_detected(self, source: str, name: str) -> None:
        """Called by MeetingWatcher when a trigger fires."""
        if source == "app":
            msg = f"{name} started — transcribe this meeting?"
        else:
            msg = "Microphone activated — start transcribing?"
        self._notify(msg)

    def _dismiss_watcher(self, icon=None, item=None):
        if self._watcher is not None:
            self._watcher.dismiss()
        if self._icon:
            self._icon.menu = self._build_menu()
            self._icon.update_menu()

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

        if self._watcher is not None:
            self._watcher.on_trigger = self._on_meeting_detected
            self._watcher.start()

        # On macOS, pystray uses the menu bar automatically via rumps/AppKit.
        # On Windows, it appears in the system tray notification area.
        self._icon.run()
