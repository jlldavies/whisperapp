import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from whisperapp.startup import (
    register_startup, unregister_startup, _launch_command,
)


# ---------------------------------------------------------------------------
# _launch_command resolution
# ---------------------------------------------------------------------------

def test_launch_command_frozen():
    """Frozen PyInstaller binary: returns [sys.executable]."""
    with patch("whisperapp.startup.sys") as mock_sys:
        mock_sys.executable = "/app/WhisperApp"
        mock_sys.platform = "darwin"
        # Simulate frozen bundle
        type(mock_sys).frozen = property(lambda s: True)
        cmd = _launch_command()
    assert cmd == ["/app/WhisperApp"]


def test_launch_command_installed_script_windows(tmp_path):
    """Installed whisperapp-app.exe is preferred over python -m whisperapp."""
    script = tmp_path / "whisperapp-app.exe"
    script.touch()
    with patch("whisperapp.startup.sys") as mock_sys:
        mock_sys.executable = str(tmp_path / "python.exe")
        mock_sys.platform = "win32"
        mock_sys.frozen = False
        cmd = _launch_command()
    assert cmd == [str(script)]


def test_launch_command_fallback_windows(tmp_path):
    """Falls back to pythonw.exe -m whisperapp when no script installed."""
    pythonw = tmp_path / "pythonw.exe"
    pythonw.touch()
    with patch("whisperapp.startup.sys") as mock_sys:
        mock_sys.executable = str(tmp_path / "python.exe")
        mock_sys.platform = "win32"
        mock_sys.frozen = False
        cmd = _launch_command()
    assert cmd[0] == str(pythonw)
    assert "-m" in cmd
    assert "whisperapp" in cmd


# ---------------------------------------------------------------------------
# Windows registration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_register_windows():
    with patch("whisperapp.startup.winreg") as mock_reg:
        mock_ctx = MagicMock()
        mock_reg.OpenKey.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_reg.OpenKey.return_value.__exit__ = MagicMock(return_value=False)
        register_startup()
        mock_reg.SetValueEx.assert_called_once()
        # Confirm the value name is "WhisperApp"
        args = mock_reg.SetValueEx.call_args[0]
        assert args[1] == "WhisperApp"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_unregister_windows_missing_key_is_silent():
    with patch("whisperapp.startup.winreg") as mock_reg:
        mock_reg.OpenKey.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_reg.OpenKey.return_value.__exit__ = MagicMock(return_value=False)
        mock_reg.DeleteValue.side_effect = FileNotFoundError
        unregister_startup()  # should not raise


# ---------------------------------------------------------------------------
# macOS registration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_register_macos_creates_plist(tmp_path):
    plist = tmp_path / "whisperapp.plist"
    with patch("whisperapp.startup.PLIST_PATH", plist), \
         patch("whisperapp.startup.subprocess.run"):
        register_startup()
    assert plist.exists()
    content = plist.read_text()
    assert "com.whisperapp" in content
    assert "RunAtLoad" in content


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_unregister_macos_removes_plist(tmp_path):
    plist = tmp_path / "whisperapp.plist"
    plist.write_text("<plist/>")
    with patch("whisperapp.startup.PLIST_PATH", plist), \
         patch("whisperapp.startup.subprocess.run"):
        unregister_startup()
    assert not plist.exists()
