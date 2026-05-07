import threading
import time
from unittest.mock import patch, MagicMock
from whisperapp.watcher import MeetingWatcher, TRIGGER_APPS


def test_watcher_creates_and_stops():
    w = MeetingWatcher(on_trigger=lambda source, name: None)
    w.start()
    assert w._thread is not None
    assert w._thread.is_alive()
    w.stop()
    w._thread.join(timeout=2)
    assert not w._thread.is_alive()


def test_trigger_apps_contains_zoom():
    assert any("zoom" in k.lower() for k in TRIGGER_APPS)


def test_trigger_apps_contains_teams():
    assert any("team" in k.lower() for k in TRIGGER_APPS)
