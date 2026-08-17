"""Tests for the AI provider abstraction layer."""
import json
import pytest
from unittest.mock import patch, MagicMock
from whisperapp.ai import (
    NullProvider, ClaudeProvider, OpenAIProvider, OllamaProvider,
    make_provider, _parse_json_mapping, AIProviderError,
)


# ---------------------------------------------------------------------------
# _parse_json_mapping
# ---------------------------------------------------------------------------

def test_parse_clean_json():
    raw = '{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}'
    assert _parse_json_mapping(raw) == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}


def test_parse_json_with_surrounding_prose():
    raw = 'Sure! Here you go:\n{"SPEAKER_00": "Alice"}\nHope that helps.'
    assert _parse_json_mapping(raw) == {"SPEAKER_00": "Alice"}


def test_parse_empty_string():
    assert _parse_json_mapping("") == {}


def test_parse_no_json():
    assert _parse_json_mapping("I cannot identify any speakers.") == {}


def test_parse_invalid_json():
    assert _parse_json_mapping("{not valid json}") == {}


# ---------------------------------------------------------------------------
# NullProvider
# ---------------------------------------------------------------------------

def test_null_provider_not_available():
    p = NullProvider()
    assert p.is_available() is False
    assert p.name == "none"


def test_null_provider_returns_empty():
    p = NullProvider()
    assert p.identify_speakers({"SPEAKER_00": ["Hello"]}) == {}
    assert p.meeting_notes("transcript") == ""
    assert p.live_summary("transcript") == ""


# ---------------------------------------------------------------------------
# make_provider factory
# ---------------------------------------------------------------------------

def test_make_provider_none():
    p = make_provider("none")
    assert isinstance(p, NullProvider)


def test_make_provider_unknown_returns_null():
    p = make_provider("grok")
    assert isinstance(p, NullProvider)


def test_make_provider_claude():
    p = make_provider("claude", api_key="sk-test")
    assert isinstance(p, ClaudeProvider)
    assert p.name == "claude"


def test_make_provider_openai():
    p = make_provider("openai", api_key="sk-test")
    assert isinstance(p, OpenAIProvider)
    assert p.name == "openai"


def test_make_provider_ollama():
    p = make_provider("ollama")
    assert isinstance(p, OllamaProvider)
    assert p.name == "ollama"


def test_make_provider_case_insensitive():
    p = make_provider("CLAUDE", api_key="key")
    assert isinstance(p, ClaudeProvider)


# ---------------------------------------------------------------------------
# ClaudeProvider
# ---------------------------------------------------------------------------

def test_claude_not_available_without_key():
    p = ClaudeProvider(api_key="")
    assert p.is_available() is False


def test_claude_available_with_key():
    p = ClaudeProvider(api_key="sk-ant-test")
    assert p.is_available() is True


def test_claude_identify_speakers():
    p = ClaudeProvider(api_key="sk-test")
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text='{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}')]

    with patch.object(p, "_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        mock_client_fn.return_value = mock_client

        result = p.identify_speakers(
            {"SPEAKER_00": ["Hi everyone"], "SPEAKER_01": ["Thanks for joining"]},
            context="Weekly standup"
        )
    assert result == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}


def test_claude_raises_on_api_error():
    # A failed call must not collapse into an empty-but-successful mapping —
    # that's the exact shape that let a dead API key read as "no speakers
    # found" on real transcripts. It must be observably distinct: raise.
    p = ClaudeProvider(api_key="sk-test")
    with patch.object(p, "_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("rate limit")
        mock_client_fn.return_value = mock_client
        with pytest.raises(AIProviderError, match="rate limit"):
            p.identify_speakers({"SPEAKER_00": ["Hello"]})


def test_claude_raises_on_meeting_notes_api_error():
    p = ClaudeProvider(api_key="sk-test")
    with patch.object(p, "_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("network down")
        mock_client_fn.return_value = mock_client
        with pytest.raises(AIProviderError, match="network down"):
            p.meeting_notes("Alice: Hi.")


def test_claude_raises_on_live_summary_api_error():
    p = ClaudeProvider(api_key="sk-test")
    with patch.object(p, "_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("timeout")
        mock_client_fn.return_value = mock_client
        with pytest.raises(AIProviderError, match="timeout"):
            p.live_summary("Alice: Hi.")


def test_claude_meeting_notes():
    p = ClaudeProvider(api_key="sk-test")
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="## Summary\nQuick meeting.")]

    with patch.object(p, "_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        mock_client_fn.return_value = mock_client
        notes = p.meeting_notes("Alice: Hi. Bob: Hello.")
    assert "Summary" in notes


def test_claude_missing_package_raises():
    p = ClaudeProvider(api_key="sk-test")
    with patch("whisperapp.ai.ClaudeProvider._client", return_value=None):
        with pytest.raises(AIProviderError, match="anthropic"):
            p.identify_speakers({"SPEAKER_00": ["Hi"]})


def test_claude_identify_speakers_empty_snippets_no_call():
    # No snippets = nothing to ask the model — this is a genuine "nothing to
    # do" short-circuit, not a failure, and must stay a plain {} (no call is
    # even attempted).
    p = ClaudeProvider(api_key="sk-test")
    with patch.object(p, "_client") as mock_client_fn:
        result = p.identify_speakers({})
    assert result == {}
    mock_client_fn.assert_not_called()


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------

def test_openai_not_available_without_key():
    p = OpenAIProvider(api_key="")
    assert p.is_available() is False


def test_openai_identify_speakers():
    p = OpenAIProvider(api_key="sk-test")
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(
        content='{"SPEAKER_00": "Carol"}'))]

    with patch.object(p, "_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_client_fn.return_value = mock_client
        result = p.identify_speakers({"SPEAKER_00": ["Good morning"]})
    assert result == {"SPEAKER_00": "Carol"}


def test_openai_custom_base_url():
    p = OpenAIProvider(api_key="key", base_url="http://localhost:8080/v1")
    assert p._base_url == "http://localhost:8080/v1"


def test_openai_raises_on_api_error():
    p = OpenAIProvider(api_key="sk-test")
    with patch.object(p, "_client") as mock_client_fn:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("auth failed")
        mock_client_fn.return_value = mock_client
        with pytest.raises(AIProviderError, match="auth failed"):
            p.identify_speakers({"SPEAKER_00": ["Hello"]})


def test_openai_missing_package_raises():
    p = OpenAIProvider(api_key="sk-test")
    with patch("whisperapp.ai.OpenAIProvider._client", return_value=None):
        with pytest.raises(AIProviderError, match="openai"):
            p.identify_speakers({"SPEAKER_00": ["Hi"]})


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

def test_ollama_available_when_server_up():
    p = OllamaProvider()
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        assert p.is_available() is True


def test_ollama_not_available_when_server_down():
    p = OllamaProvider()
    with patch("requests.get", side_effect=Exception("connection refused")):
        assert p.is_available() is False


def test_ollama_identify_speakers():
    p = OllamaProvider(model="llama3.2")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "message": {"content": '{"SPEAKER_00": "Dave"}'}
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        result = p.identify_speakers({"SPEAKER_00": ["Hello"]})
    assert result == {"SPEAKER_00": "Dave"}


def test_ollama_custom_base_url():
    p = OllamaProvider(base_url="http://192.168.1.10:11434")
    assert p._base_url == "http://192.168.1.10:11434"


def test_ollama_raises_on_connection_error():
    p = OllamaProvider(model="llama3.2")
    with patch("requests.post", side_effect=Exception("connection refused")):
        with pytest.raises(AIProviderError, match="connection refused"):
            p.identify_speakers({"SPEAKER_00": ["Hello"]})


# ---------------------------------------------------------------------------
# Integration with config
# ---------------------------------------------------------------------------

def test_config_ai_fields_persist(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    from whisperapp.config import Config
    c = Config()
    c.ai_provider = "openai"
    c.ai_api_key = "sk-test"
    c.ai_model = "gpt-4o"
    c.save()

    c2 = Config()
    assert c2.ai_provider == "openai"
    assert c2.ai_api_key == "sk-test"
    assert c2.ai_model == "gpt-4o"


def test_make_provider_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    from whisperapp.config import Config
    from whisperapp.ai import make_provider
    c = Config()
    c.ai_provider = "claude"
    c.ai_api_key = "sk-ant-abc"
    c.save()

    c2 = Config()
    p = make_provider(c2.ai_provider, c2.ai_api_key, c2.ai_model, c2.ai_base_url)
    assert isinstance(p, ClaudeProvider)
    assert p.is_available() is True


# ---------------------------------------------------------------------------
# /ai/identify-speakers endpoint — a failed provider call must not read as
# "checked, nothing found". This is the actual defect: on litigation
# transcripts, a 200 with an empty mapping is indistinguishable from "the AI
# ran and confirmed the speakers are correct" unless a failed call is routed
# differently end to end.
# ---------------------------------------------------------------------------

@pytest.fixture
def speaker_review_app(tmp_path):
    from whisperapp.server import create_app
    from whisperapp.queue import JobQueue, JobStatus
    from whisperapp.checkpoints import CheckpointManager

    q = JobQueue(db_path=tmp_path / "jobs.db")
    app = create_app(queue=q, worker=MagicMock())

    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"fake")
    job_id = q.create_job(str(audio), str(tmp_path), "large-v2", True, ["txt"])
    q.update_progress(job_id, 85, "speaker_review")
    q.set_status(job_id, JobStatus.SPEAKER_REVIEW)

    cm = CheckpointManager(str(tmp_path), job_id)
    cm.save("speaker_review", {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello there", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "text": "Hi, how are you", "speaker": "SPEAKER_01"},
        ]
    })
    return app, job_id


async def _post_identify_speakers(app, job_id):
    from httpx import AsyncClient, ASGITransport
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        return await client.post("/ai/identify-speakers", json={"job_id": job_id})


@pytest.mark.asyncio
async def test_identify_speakers_endpoint_failed_call_is_not_200(speaker_review_app):
    app, job_id = speaker_review_app
    failing_provider = MagicMock()
    failing_provider.name = "claude"
    failing_provider.is_available.return_value = True
    failing_provider.identify_speakers.side_effect = AIProviderError("rate limit hit")

    with patch("whisperapp.ai.make_provider", return_value=failing_provider):
        resp = await _post_identify_speakers(app, job_id)

    # A failed call must be a non-200 with a usable message — never a bare
    # 200 with an empty mapping, which would read as "checked, none found".
    assert resp.status_code != 200
    assert resp.status_code == 502
    assert "rate limit hit" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_identify_speakers_endpoint_genuine_empty_result_is_200(speaker_review_app):
    app, job_id = speaker_review_app
    ok_provider = MagicMock()
    ok_provider.name = "claude"
    ok_provider.is_available.return_value = True
    ok_provider.identify_speakers.return_value = {}  # AI ran, found nothing — not a failure

    with patch("whisperapp.ai.make_provider", return_value=ok_provider):
        resp = await _post_identify_speakers(app, job_id)

    assert resp.status_code == 200
    assert resp.json()["mapping"] == {}


@pytest.mark.asyncio
async def test_identify_speakers_endpoint_success_mapping_unregressed(speaker_review_app):
    app, job_id = speaker_review_app
    ok_provider = MagicMock()
    ok_provider.name = "claude"
    ok_provider.is_available.return_value = True
    ok_provider.identify_speakers.return_value = {"SPEAKER_00": "Alice"}

    with patch("whisperapp.ai.make_provider", return_value=ok_provider):
        resp = await _post_identify_speakers(app, job_id)

    assert resp.status_code == 200
    data = resp.json()
    assert data["mapping"] == {"SPEAKER_00": "Alice"}
    assert data["provider"] == "claude"
