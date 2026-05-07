# whisperapp/watcher.py
import sys
import threading
import time
from typing import Callable

# Maps lowercase process name fragment → display name shown in notification
TRIGGER_APPS: dict[str, str] = {
    "zoom":     "Zoom",
    "teams":    "Microsoft Teams",
    "webex":    "Webex",
    "discord":  "Discord",
    "loom":     "Loom",
    "slack":    "Slack",
    "meet":     "Google Meet",
    "skype":    "Skype",
    "gotomeeting": "GoToMeeting",
}


class MeetingWatcher:
    """Polls for meeting-app launches and mic-in-use events, fires on_trigger callback."""

    def __init__(self, on_trigger: Callable[[str, str], None],
                 poll_apps_interval: float = 5.0,
                 poll_mic_interval: float = 3.0):
        self._on_trigger = on_trigger
        self._poll_apps = poll_apps_interval
        self._poll_mic = poll_mic_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_apps: set[str] = set()   # display names currently running
        self._mic_notified = False             # True while mic notification is active
        self._dismissed = False                # True after user dismisses; resets on condition change
        self._lock = threading.Lock()

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="MeetingWatcher")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def dismiss(self) -> None:
        """Silence notifications for this session of the current trigger.
        Resets automatically when the triggering condition disappears."""
        with self._lock:
            self._dismissed = True

    @property
    def is_active(self) -> bool:
        """True while any meeting trigger is in effect (used to show dismiss menu item)."""
        with self._lock:
            return bool(self._active_apps) or self._mic_notified

    @property
    def on_trigger(self):
        return self._on_trigger

    @on_trigger.setter
    def on_trigger(self, value):
        self._on_trigger = value

    def _run(self) -> None:
        last_apps_check = 0.0
        last_mic_check = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - last_apps_check >= self._poll_apps:
                self._check_apps()
                last_apps_check = now
            if now - last_mic_check >= self._poll_mic:
                self._check_mic()
                last_mic_check = now
            time.sleep(0.5)

    def _check_apps(self) -> None:
        running = _running_trigger_apps()
        with self._lock:
            new_apps = running - self._active_apps
            gone_apps = self._active_apps - running
            self._active_apps = running
            if gone_apps and not running:
                # All trigger apps closed — reset state so next launch gets a fresh notification
                self._mic_notified = False
                self._dismissed = False
            notify = [(name,) for name in new_apps if not self._dismissed]
        for (name,) in notify:
            self._on_trigger("app", name)

    def _check_mic(self) -> None:
        in_use = _mic_in_use()
        with self._lock:
            if in_use and not self._mic_notified:
                self._mic_notified = True
                should_notify = not self._dismissed
            elif not in_use and self._mic_notified:
                # Mic released — reset so next activation triggers again
                self._mic_notified = False
                self._dismissed = False
                should_notify = False
            else:
                should_notify = False
        if should_notify:
            self._on_trigger("mic", "")


def _running_trigger_apps() -> set[str]:
    """Return display names of TRIGGER_APPS currently running."""
    if sys.platform == "darwin":
        return _running_trigger_apps_mac()
    return _running_trigger_apps_psutil()


def _running_trigger_apps_mac() -> set[str]:
    try:
        from AppKit import NSWorkspace
        running = NSWorkspace.sharedWorkspace().runningApplications()
        result: set[str] = set()
        for app in running:
            name = (app.localizedName() or "").lower()
            bundle = (app.bundleIdentifier() or "").lower()
            for fragment, display in TRIGGER_APPS.items():
                if fragment in name or fragment in bundle:
                    result.add(display)
        return result
    except Exception:
        return _running_trigger_apps_psutil()


def _running_trigger_apps_psutil() -> set[str]:
    try:
        import psutil
        result: set[str] = set()
        for proc in psutil.process_iter(["name"], ad_value=""):
            name = proc.info["name"].lower().replace(".exe", "")
            for fragment, display in TRIGGER_APPS.items():
                if fragment in name:
                    result.add(display)
        return result
    except Exception:
        return set()


def _mic_in_use() -> bool:
    """Return True if any external process is currently using the microphone."""
    if sys.platform == "darwin":
        return _mic_in_use_mac()
    elif sys.platform == "win32":
        return _mic_in_use_win()
    return False


def _mic_in_use_mac() -> bool:
    """True if any audio input device is currently in use by another application."""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
        devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeAudio) or []
        return any(d.isInUseByAnotherApplication() for d in devices)
    except ImportError:
        return False
    except Exception:
        return False


def _mic_in_use_win() -> bool:
    return False  # stub — implemented in Task 4
