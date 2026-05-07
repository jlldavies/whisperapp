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


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_mic_in_use_mac_returns_false_when_devices_is_none():
    from whisperapp.watcher import _mic_in_use_mac
    with patch.dict("sys.modules", {
        "AVFoundation": MagicMock(
            AVCaptureDevice=MagicMock(
                devicesWithMediaType_=MagicMock(return_value=None)
            ),
            AVMediaTypeAudio="soun"
        )
    }):
        result = _mic_in_use_mac()
    assert result is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_mic_in_use_win_false_when_registry_absent():
    from whisperapp.watcher import _mic_in_use_win
    import winreg
    with patch("winreg.OpenKey", side_effect=OSError):
        result = _mic_in_use_win()
    assert result is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_mic_key_active_true_when_stop_is_zero():
    from whisperapp.watcher import _mic_key_active
    import winreg
    mock_key = MagicMock()
    with patch("winreg.QueryValueEx", side_effect=[
        (100, winreg.REG_QWORD),  # LastUsedTimeStart = 100
        (0,   winreg.REG_QWORD),  # LastUsedTimeStop  = 0 → active
    ]):
        result = _mic_key_active(mock_key)
    assert result is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_mic_key_active_false_when_stop_after_start():
    from whisperapp.watcher import _mic_key_active
    import winreg
    mock_key = MagicMock()
    with patch("winreg.QueryValueEx", side_effect=[
        (100, winreg.REG_QWORD),  # LastUsedTimeStart = 100
        (200, winreg.REG_QWORD),  # LastUsedTimeStop  = 200 > start → not active
    ]):
        result = _mic_key_active(mock_key)
    assert result is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_mic_key_active_false_when_start_is_zero():
    from whisperapp.watcher import _mic_key_active
    import winreg
    mock_key = MagicMock()
    with patch("winreg.QueryValueEx", return_value=(0, winreg.REG_QWORD)):
        result = _mic_key_active(mock_key)
    assert result is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_mic_key_active_true_when_no_stop_key():
    from whisperapp.watcher import _mic_key_active
    import winreg
    mock_key = MagicMock()
    # QueryValueEx: first call (LastUsedTimeStart) returns 100, second call (LastUsedTimeStop) raises OSError
    with patch("winreg.QueryValueEx", side_effect=[
        (100, winreg.REG_QWORD),   # LastUsedTimeStart = 100
        OSError("not found"),       # LastUsedTimeStop absent → still active
    ]):
        result = _mic_key_active(mock_key)
    assert result is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_mic_in_use_win_true_via_packaged_app():
    """Packaged Store app (non-NonPackaged) with active mic."""
    from whisperapp.watcher import _mic_in_use_win
    import winreg

    mock_root_ctx = MagicMock()
    mock_app_ctx = MagicMock()

    def open_key(hive, path, *a, **kw):
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=mock_root_ctx if path.endswith("microphone") else mock_app_ctx)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    with patch("winreg.OpenKey", side_effect=open_key), \
         patch("winreg.EnumKey", side_effect=["SomeStoreApp", OSError()]), \
         patch("winreg.QueryValueEx", side_effect=[
             (100, winreg.REG_QWORD),  # LastUsedTimeStart = 100
             (0,   winreg.REG_QWORD),  # LastUsedTimeStop  = 0 → active
         ]):
        result = _mic_in_use_win()
    assert result is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_mic_in_use_win_true_via_nonpackaged_app():
    """NonPackaged exe (e.g. Chrome, Zoom.exe) with active mic."""
    from whisperapp.watcher import _mic_in_use_win
    import winreg

    # EnumKey calls: first level returns "NonPackaged", then OSError to stop.
    # Second level (inside NonPackaged) returns "chrome.exe", then OSError to stop.
    enum_key_calls = iter(["NonPackaged", OSError(), "chrome.exe", OSError()])

    def enum_key_side(key, idx):
        val = next(enum_key_calls)
        if isinstance(val, Exception):
            raise val
        return val

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with patch("winreg.OpenKey", return_value=mock_ctx), \
         patch("winreg.EnumKey", side_effect=enum_key_side), \
         patch("winreg.QueryValueEx", side_effect=[
             (100, winreg.REG_QWORD),  # LastUsedTimeStart = 100
             (0,   winreg.REG_QWORD),  # LastUsedTimeStop  = 0 → active
         ]):
        result = _mic_in_use_win()
    assert result is True
