from click.testing import CliRunner
from whisperapp.cli import cli
from unittest.mock import patch
import json

def test_status_invalid_job_id():
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "not-a-uuid"])
    assert result.exit_code != 0 or "invalid" in result.output.lower()

def test_list_command_runs():
    runner = CliRunner()
    with patch("whisperapp.cli.get_queue") as mock_q:
        mock_q.return_value.list_jobs.return_value = []
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "no jobs" in result.output.lower()

def test_list_with_jobs():
    runner = CliRunner()
    with patch("whisperapp.cli.get_queue") as mock_q:
        mock_q.return_value.list_jobs.return_value = [{
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "file_name": "audio.mp3",
            "status": "done",
            "progress": 100,
            "created_at": "2026-03-18T10:00:00"
        }]
        result = runner.invoke(cli, ["list"])
        assert "audio.mp3" in result.output
        assert "done" in result.output
