import os
import subprocess
import sys
import threading
import webbrowser
import pystray
from PIL import Image, ImageDraw
from whisperapp.startup import register_startup, unregister_startup, is_startup_registered


# ---------------------------------------------------------------------------
# Icon factory
# ---------------------------------------------------------------------------

_WAVEFORM_PTS = [(6, 12), (16, 52), (26, 20), (32, 20), (39, 52), (48, 20), (58, 52)]


def _make_icon_template(color: str) -> Image.Image:
    """Bare waveform on transparent background — used on macOS where the
    status item is marked as a template image and auto-inverts for any menu
    bar colour."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.line(_WAVEFORM_PTS, fill=color, width=6, joint="curve")
    return img


def _make_icon_filled(bg: str, fg: str = "#faf8f4") -> Image.Image:
    """Rounded-square brand mark with waveform inside — used on Windows /
    Linux where the system tray cannot auto-invert a template image, so a
    bare stroke disappears against a same-colour taskbar."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([2, 2, 62, 62], radius=18, fill=bg)
    d.line(_WAVEFORM_PTS, fill=fg, width=6, joint="curve")
    return img


if sys.platform == "darwin":
    ICONS = {
        "idle":    _make_icon_template("#ffffff"),
        "ready":   _make_icon_template("#ffffff"),
        "working": _make_icon_template("#c96442"),
        "error":   _make_icon_template("#c4523f"),
    }
else:
    # Windows / Linux: filled rounded square so the icon stays visible on
    # both light and dark taskbars. Background colour signals state.
    ICONS = {
        "idle":    _make_icon_filled("#1f1d1a"),  # --ink (near-black)
        "ready":   _make_icon_filled("#1f1d1a"),
        "working": _make_icon_filled("#c96442"),  # --accent terracotta
        "error":   _make_icon_filled("#c4523f"),  # --signal-rec
    }


# ---------------------------------------------------------------------------
# App-mode browser launcher
# ---------------------------------------------------------------------------

def _open_app_window(url: str) -> None:
    """Open url in a chromium app-mode window (no address bar).

    On macOS uses the `open` command so the browser gets proper focus and
    process group.  On Windows falls back to direct exe launch.
    Falls back to the default browser if no chromium browser is found.
    """
    if sys.platform == "darwin":
        app_names = [
            "Google Chrome",
            "Microsoft Edge",
            "Brave Browser",
        ]
        for app in app_names:
            path = f"/Applications/{app}.app"
            if os.path.exists(path):
                subprocess.Popen([
                    "open", "-n", "-a", app,
                    "--args", f"--app={url}",
                ])
                return
    elif sys.platform == "win32":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        candidates = [
            rf"{pf}\Google\Chrome\Application\chrome.exe",
            rf"{pf86}\Google\Chrome\Application\chrome.exe",
            rf"{pf}\Microsoft\Edge\Application\msedge.exe",
            rf"{pf86}\Microsoft\Edge\Application\msedge.exe",
        ]
        for path in candidates:
            if os.path.exists(path):
                subprocess.Popen([path, f"--app={url}"])
                return

    webbrowser.open(url)


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
        # Delegate to whisperapp.startup so the server endpoint and the tray
        # menu agree on the same source of truth (Windows registry / Mac plist).
        return is_startup_registered()

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
            # `checked=` makes pystray render the platform-native checkmark
            # indicator — a real check glyph on Windows and macOS — instead of
            # us munging the label text with brackets or unicode symbols.
            pystray.MenuItem(
                "Start on login",
                self._toggle_startup,
                checked=lambda _item: self._startup_enabled,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Restart", self._restart),
            pystray.MenuItem("Quit", self._quit),
        ]
        return pystray.Menu(*items)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open_ui(self, icon=None, item=None):
        _open_app_window("http://127.0.0.1:7860")

    def _open_api(self, icon=None, item=None):
        webbrowser.open("http://127.0.0.1:7861/docs")

    def _restart(self, icon, item):
        from whisperapp.config import _config_dir
        # Clear PID lock so the new instance can acquire it immediately
        (_config_dir() / "whisperapp.pid").unlink(missing_ok=True)
        subprocess.Popen([sys.executable, '-m', 'whisperapp'])
        icon.stop()
        os._exit(0)

    def _quit(self, icon, item):
        icon.stop()
        os._exit(0)

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

        # pystray's Windows backend leaves the icon hidden until something
        # explicitly sets `visible = True` — the `setup` callback is pystray's
        # idiomatic place to do that. macOS shows the menu-bar item as soon as
        # `_NSStatusItem` is constructed, so this is a no-op there but harmless.
        def _setup(icon):
            icon.visible = True
        self._icon.run(setup=_setup)
