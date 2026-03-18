from unittest.mock import patch, call
from whisperapp.updater import run_update

def test_update_calls_pip():
    with patch("whisperapp.updater.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        run_update()
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("whisperx" in c for c in calls)

def test_update_handles_failure_gracefully():
    with patch("whisperapp.updater.subprocess.run") as mock_run:
        mock_run.side_effect = Exception("Network error")
        run_update()  # Should not raise
