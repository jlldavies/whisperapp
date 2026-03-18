from unittest.mock import patch, MagicMock
from whisperapp.tray import TrayApp

def test_tray_app_creates():
    with patch("whisperapp.tray.pystray"):
        with patch("whisperapp.tray.Image"):
            app = TrayApp(queue=MagicMock(), worker=MagicMock())
            assert app is not None

def test_tray_title_idle():
    with patch("whisperapp.tray.pystray"):
        with patch("whisperapp.tray.Image"):
            app = TrayApp(queue=MagicMock(), worker=MagicMock())
            assert "WhisperApp" in app.get_title(0)

def test_tray_title_working():
    with patch("whisperapp.tray.pystray"):
        with patch("whisperapp.tray.Image"):
            app = TrayApp(queue=MagicMock(), worker=MagicMock())
            assert "2 jobs" in app.get_title(2).lower() or "2" in app.get_title(2)
