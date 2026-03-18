import sys
import pytest
from unittest.mock import patch, MagicMock
from whisperapp.startup import register_startup, unregister_startup

@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_register_windows():
    with patch("whisperapp.startup.winreg") as mock_reg:
        mock_reg.OpenKey.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_reg.OpenKey.return_value.__exit__ = MagicMock(return_value=False)
        register_startup()
        mock_reg.SetValueEx.assert_called_once()

@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_register_macos(tmp_path):
    with patch("whisperapp.startup.PLIST_PATH", tmp_path / "whisperapp.plist"):
        register_startup()
        assert (tmp_path / "whisperapp.plist").exists()
