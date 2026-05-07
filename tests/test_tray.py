import sys
from unittest.mock import patch, MagicMock
from whisperapp.tray import TrayApp


def _make_tray():
    """Helper: construct a TrayApp with all platform side-effects mocked."""
    mock_queue = MagicMock()
    mock_queue.list_jobs.return_value = []
    with patch("whisperapp.tray.pystray"), \
         patch("whisperapp.tray.Image"):
        if sys.platform == "win32":
            with patch("whisperapp.tray.winreg",
                       create=True) as mock_reg:
                mock_reg.OpenKey.side_effect = FileNotFoundError
                return TrayApp(queue=mock_queue, worker=MagicMock())
        else:
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
