from click.testing import CliRunner
from whisperapp.cli import cli
from unittest.mock import patch, MagicMock
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

def test_identify_speakers_503_no_provider():
    runner = CliRunner()
    mock_resp = MagicMock(status_code=503)
    with patch("whisperapp.cli.requests.post", return_value=mock_resp):
        result = runner.invoke(cli, [
            "identify-speakers", "550e8400-e29b-41d4-a716-446655440000",
            "--no-full-transcript",
        ])
    assert result.exit_code != 0
    assert "no ai provider configured" in result.output.lower()

def test_identify_speakers_502_provider_call_failed():
    runner = CliRunner()
    mock_resp = MagicMock(status_code=502)
    mock_resp.json.return_value = {"detail": "AI provider call failed: rate limit exceeded"}
    with patch("whisperapp.cli.requests.post", return_value=mock_resp):
        result = runner.invoke(cli, [
            "identify-speakers", "550e8400-e29b-41d4-a716-446655440000",
            "--no-full-transcript",
        ])
    assert result.exit_code != 0
    assert "rate limit exceeded" in result.output
    # no raw traceback / unhandled exception surfaced
    assert result.exception is None or isinstance(result.exception, SystemExit)

def test_identify_speakers_502_detail_missing_falls_back():
    runner = CliRunner()
    mock_resp = MagicMock(status_code=502)
    mock_resp.json.side_effect = ValueError("not json")
    mock_resp.text = "Bad Gateway"
    with patch("whisperapp.cli.requests.post", return_value=mock_resp):
        result = runner.invoke(cli, [
            "identify-speakers", "550e8400-e29b-41d4-a716-446655440000",
            "--no-full-transcript",
        ])
    assert result.exit_code != 0
    assert "bad gateway" in result.output.lower()
    assert result.exception is None or isinstance(result.exception, SystemExit)

def test_meeting_notes_503_no_provider():
    runner = CliRunner()
    mock_resp = MagicMock(status_code=503)
    with patch("whisperapp.cli.requests.post", return_value=mock_resp):
        result = runner.invoke(cli, [
            "meeting-notes", "550e8400-e29b-41d4-a716-446655440000",
        ])
    assert result.exit_code != 0
    assert "no ai provider configured" in result.output.lower()

def test_meeting_notes_502_provider_call_failed():
    runner = CliRunner()
    mock_resp = MagicMock(status_code=502)
    mock_resp.json.return_value = {"detail": "AI provider call failed: auth failed"}
    with patch("whisperapp.cli.requests.post", return_value=mock_resp):
        result = runner.invoke(cli, [
            "meeting-notes", "550e8400-e29b-41d4-a716-446655440000",
        ])
    assert result.exit_code != 0
    assert "auth failed" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
