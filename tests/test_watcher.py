import sys
import threading
import time
import pytest
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


def test_running_trigger_apps_returns_set():
    from whisperapp.watcher import _running_trigger_apps
    result = _running_trigger_apps()
    assert isinstance(result, set)


def test_watcher_fires_on_new_app():
    fired = []
    w = MeetingWatcher(on_trigger=lambda s, n: fired.append((s, n)))
    with patch("whisperapp.watcher._running_trigger_apps", return_value={"Zoom"}):
        w._check_apps()
    assert ("app", "Zoom") in fired


def test_watcher_no_duplicate_for_same_app():
    fired = []
    w = MeetingWatcher(on_trigger=lambda s, n: fired.append((s, n)))
    with patch("whisperapp.watcher._running_trigger_apps", return_value={"Zoom"}):
        w._check_apps()
        w._check_apps()   # second call — Zoom still running, should not re-fire
    assert fired.count(("app", "Zoom")) == 1


def test_watcher_fires_again_after_app_closes_and_reopens():
    fired = []
    w = MeetingWatcher(on_trigger=lambda s, n: fired.append((s, n)))
    with patch("whisperapp.watcher._running_trigger_apps", return_value={"Zoom"}):
        w._check_apps()
    with patch("whisperapp.watcher._running_trigger_apps", return_value=set()):
        w._check_apps()   # Zoom closes
    with patch("whisperapp.watcher._running_trigger_apps", return_value={"Zoom"}):
        w._check_apps()   # Zoom opens again
    assert fired.count(("app", "Zoom")) == 2


def test_watcher_dismissed_suppresses_new_app():
    fired = []
    w = MeetingWatcher(on_trigger=lambda s, n: fired.append((s, n)))
    w.dismiss()  # dismiss before any app launches
    with patch("whisperapp.watcher._running_trigger_apps", return_value={"Zoom"}):
        w._check_apps()
    assert fired == []  # dismissed — should not fire


def test_watcher_dismissed_resets_after_app_closes():
    fired = []
    w = MeetingWatcher(on_trigger=lambda s, n: fired.append((s, n)))
    # App opens and fires
    with patch("whisperapp.watcher._running_trigger_apps", return_value={"Zoom"}):
        w._check_apps()
    assert len(fired) == 1
    # User dismisses
    w.dismiss()
    # App closes — dismissed resets
    with patch("whisperapp.watcher._running_trigger_apps", return_value=set()):
        w._check_apps()
    # App reopens — should fire again (dismiss was reset)
    with patch("whisperapp.watcher._running_trigger_apps", return_value={"Zoom"}):
        w._check_apps()
    assert len(fired) == 2


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_mic_in_use_mac_returns_false_when_device_not_active():
    from whisperapp.watcher import _mic_in_use_mac
    mock_device = MagicMock()
    mock_device.isInUseByAnotherApplication.return_value = False
    with patch.dict("sys.modules", {
        "AVFoundation": MagicMock(
            AVCaptureDevice=MagicMock(
                devicesWithMediaType_=MagicMock(return_value=[mock_device])
            ),
            AVMediaTypeAudio="soun"
        )
    }):
        result = _mic_in_use_mac()
    assert result is False


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_mic_in_use_mac_returns_true_when_device_active():
    from whisperapp.watcher import _mic_in_use_mac
    mock_device = MagicMock()
    mock_device.isInUseByAnotherApplication.return_value = True
    with patch.dict("sys.modules", {
        "AVFoundation": MagicMock(
            AVCaptureDevice=MagicMock(
                devicesWithMediaType_=MagicMock(return_value=[mock_device])
            ),
            AVMediaTypeAudio="soun"
        )
    }):
        result = _mic_in_use_mac()
    assert result is True


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_mic_in_use_mac_returns_false_on_import_error():
    from whisperapp.watcher import _mic_in_use_mac
    with patch.dict("sys.modules", {"AVFoundation": None}):
        result = _mic_in_use_mac()
    assert result is False
