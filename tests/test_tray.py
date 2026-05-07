import sys
from unittest.mock import patch, MagicMock
from whisperapp.tray import TrayApp


def _make_tray():
    """Helper: construct a TrayApp with all platform side-effects mocked."""
    mock_queue = MagicMock()
    mock_queue.list_jobs.return_value = []
    with patch("whisperapp.tray.pystray"), \
         patch("whisperapp.tray.Image"), \
         patch.object(TrayApp, "_check_startup_registered", return_value=False):
        return TrayApp(queue=mock_queue, worker=MagicMock())


def test_tray_app_creates():
    app = _make_tray()
    assert app is not None


def test_tray_title_idle():
    app = _make_tray()
    assert "WhisperApp" in app.get_title(0)
    assert "Idle" in app.get_title(0)


def test_tray_title_one_job():
    app = _make_tray()
    title = app.get_title(1)
    assert "1 job" in title
    assert "jobs" not in title  # singular


def test_tray_title_many_jobs():
    app = _make_tray()
    title = app.get_title(3)
    assert "3 jobs" in title


def test_startup_not_registered_by_default():
    """Fresh tray should report startup as not registered (no real registry)."""
    app = _make_tray()
    # _check_startup_registered should return False when key absent
    assert app._startup_enabled is False


def test_toggle_startup_registers(tmp_path):
    """Toggling from off→on calls register_startup."""
    app = _make_tray()
    app._startup_enabled = False

    mock_icon = MagicMock()
    mock_icon.visible = False  # stop poll thread

    with patch("whisperapp.tray.register_startup") as mock_reg, \
         patch("whisperapp.tray.unregister_startup"):
        app._toggle_startup(mock_icon, None)
        mock_reg.assert_called_once()
    assert app._startup_enabled is True


def test_toggle_startup_unregisters(tmp_path):
    """Toggling from on→off calls unregister_startup."""
    app = _make_tray()
    app._startup_enabled = True

    mock_icon = MagicMock()
    mock_icon.visible = False

    with patch("whisperapp.tray.register_startup"), \
         patch("whisperapp.tray.unregister_startup") as mock_unreg:
        app._toggle_startup(mock_icon, None)
        mock_unreg.assert_called_once()
    assert app._startup_enabled is False


def test_tray_creates_with_watcher():
    """TrayApp should accept an optional watcher argument."""
    from whisperapp.watcher import MeetingWatcher
    mock_watcher = MagicMock(spec=MeetingWatcher)
    mock_queue = MagicMock()
    mock_queue.list_jobs.return_value = []
    with patch("whisperapp.tray.pystray"), \
         patch("whisperapp.tray.Image"), \
         patch.object(TrayApp, "_check_startup_registered", return_value=False):
        app = TrayApp(queue=mock_queue, worker=MagicMock(), watcher=mock_watcher)
    assert app._watcher is mock_watcher


def test_on_meeting_detected_sends_notification_mac(monkeypatch):
    """macOS: _on_meeting_detected should call _notify which calls subprocess.run."""
    monkeypatch.setattr(sys, "platform", "darwin")
    app = _make_tray()
    with patch("whisperapp.tray.subprocess.run") as mock_run:
        app._on_meeting_detected("app", "Zoom")
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "osascript" in cmd
    assert "Zoom" in " ".join(cmd)


def test_on_meeting_detected_sends_notification_win(monkeypatch):
    """Windows: _on_meeting_detected should call icon.notify."""
    monkeypatch.setattr(sys, "platform", "win32")
    app = _make_tray()
    app._icon = MagicMock()
    app._on_meeting_detected("app", "Zoom")
    app._icon.notify.assert_called_once()


def test_on_meeting_detected_mic_trigger():
    """Mic trigger uses generic 'Microphone activated' message."""
    app = _make_tray()
    with patch.object(app, "_notify") as mock_notify:
        app._on_meeting_detected("mic", "")
    mock_notify.assert_called_once()
    msg = mock_notify.call_args[0][0]
    assert "microphone" in msg.lower() or "Microphone" in msg
