"""
AI provider abstraction for WhisperApp.

All AI-enhanced features (speaker identification, meeting notes, live summary)
are optional. If no provider is configured the app works exactly as before —
manual speaker review, no auto-summary.

Supported providers
-------------------
  none    — no AI, all features disabled (default)
  claude  — Anthropic Claude API  (pip install anthropic)
  openai  — OpenAI API or any OpenAI-compatible endpoint  (pip install openai)
  ollama  — Local Ollama server  (no extra deps, uses requests)

Config fields used (from whisperapp.config.Config)
---------------------------------------------------
  ai_provider  : str  — "none" | "claude" | "openai" | "ollama"
  ai_api_key   : str  — API key (not needed for ollama)
  ai_model     : str  — model name; each provider has a sensible default
  ai_base_url  : str  — override endpoint (Ollama, Azure, local OpenAI-compat)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import json
import logging

log = logging.getLogger("whisperapp.ai")


class AIProviderError(RuntimeError):
    """Raised when a provider call fails (auth, network, rate-limit, model error,
    timeout, missing SDK, etc).

    This exists to keep "the AI call failed" from being silently collapsed into
    "the AI ran and had nothing to say" — the two used to both surface as an
    empty string, which is indistinguishable from a legitimate empty result at
    every call site above it. Callers (see server.py) must catch this
    separately from a genuinely empty/successful response.
    """


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SPEAKER_ID_SYSTEM = """You are helping identify speakers in a meeting transcript.
You will be given sample lines spoken by each anonymised speaker label.
Use the context provided (meeting topic, known attendees, etc.) to suggest
a real name or role for each speaker. Return ONLY a JSON object mapping
speaker label to suggested name, e.g.:
{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob (chair)"}
If you cannot identify a speaker, omit them from the output."""

_SPEAKER_ID_USER = """Context: {context}

Speaker samples:
{samples}

Return only JSON."""

_MEETING_NOTES_SYSTEM = """You are a professional meeting note-taker.
Given a transcript, produce structured meeting notes in Markdown with these sections:
## Summary
## Attendees (if identifiable)
## Key Discussion Points
## Decisions Made
## Action Items (with owner if mentioned)
## Next Steps

Be concise. Extract facts from the transcript — do not invent details."""

_MEETING_NOTES_USER = """Transcript:
{transcript}"""

_LIVE_SUMMARY_SYSTEM = """You are summarising an ongoing meeting in real time.
Given the transcript so far, provide a brief rolling summary (3-5 bullet points)
of what has been discussed. Update the summary as new content arrives."""

_LIVE_SUMMARY_USER = """Transcript so far:
{transcript}

Provide a brief bullet-point summary."""


def _format_snippet_samples(snippets: dict) -> str:
    """Render the {label: [snippet, ...]} dict for the LLM prompt.

    Each snippet may be either a plain string (legacy) or a dict with at
    least a `text` field (new shape from extract_speaker_snippets, which
    also carries start/end for the audio player).
    """
    def _text(s):
        if isinstance(s, dict):
            return s.get("text", "")
        return str(s)
    return "\n".join(
        f"{label}:\n" + "\n".join(f"  - {_text(s)}" for s in lines[:5])
        for label, lines in snippets.items()
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AIProvider(ABC):
    """Abstract AI provider.

    A configured provider that actually calls out (Claude/OpenAI/Ollama) raises
    AIProviderError if the call itself fails — auth, network, rate limit, bad
    model, timeout. It returns an empty/passthrough value only for a genuine
    "ran fine, nothing to report" result (e.g. no snippets given, or the model
    legitimately found no speakers to name). NullProvider (no AI configured) is
    the exception: its methods always return empty, by design, since callers
    are expected to gate on is_available() before invoking it.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def is_available(self) -> bool:
        """Return True if the provider is configured and reachable."""
        return False

    def identify_speakers(
        self,
        snippets: dict[str, list[str]],
        context: str = "",
    ) -> dict[str, str]:
        """
        Return a {SPEAKER_XX: "Name"} mapping based on transcript snippets.
        snippets: {label: [line1, line2, ...]}  (first few lines per speaker)
        Returns empty dict if unavailable or unable to identify.
        """
        return {}

    def meeting_notes(self, transcript: str, context: str = "") -> str:
        """Return structured meeting notes as Markdown, or empty string."""
        return ""

    def live_summary(self, transcript: str) -> str:
        """Return a rolling bullet-point summary of transcript so far."""
        return ""


# ---------------------------------------------------------------------------
# NullProvider — graceful no-op when AI is disabled
# ---------------------------------------------------------------------------

class NullProvider(AIProvider):
    @property
    def name(self) -> str:
        return "none"

    def is_available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Claude (Anthropic)
# ---------------------------------------------------------------------------

class ClaudeProvider(AIProvider):
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str, model: str = ""):
        self._api_key = api_key
        self._model = model or self.DEFAULT_MODEL

    @property
    def name(self) -> str:
        return "claude"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _client(self):
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self._api_key)
        except ImportError:
            log.error("anthropic package not installed. Run: pip install anthropic")
            return None

    def _call(self, system: str, user: str) -> str:
        client = self._client()
        if not client:
            raise AIProviderError(
                "anthropic package not installed. Run: pip install anthropic")
        try:
            msg = client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
        except Exception as e:
            log.error("Claude API error: %s", e)
            raise AIProviderError(f"Claude API error: {e}") from e

    def identify_speakers(self, snippets, context=""):
        if not snippets:
            return {}
        samples = _format_snippet_samples(snippets)
        user = _SPEAKER_ID_USER.format(context=context or "Not provided", samples=samples)
        raw = self._call(_SPEAKER_ID_SYSTEM, user)
        return _parse_json_mapping(raw)

    def meeting_notes(self, transcript, context=""):
        ctx = f"Context: {context}\n\n" if context else ""
        user = _MEETING_NOTES_USER.format(transcript=ctx + transcript)
        return self._call(_MEETING_NOTES_SYSTEM, user)

    def live_summary(self, transcript):
        user = _LIVE_SUMMARY_USER.format(transcript=transcript)
        return self._call(_LIVE_SUMMARY_SYSTEM, user)


# ---------------------------------------------------------------------------
# OpenAI (also covers Azure, local OpenAI-compatible endpoints)
# ---------------------------------------------------------------------------

class OpenAIProvider(AIProvider):
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str, model: str = "", base_url: str = ""):
        self._api_key = api_key
        self._model = model or self.DEFAULT_MODEL
        self._base_url = base_url or None

    @property
    def name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _client(self):
        try:
            import openai
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            return openai.OpenAI(**kwargs)
        except ImportError:
            log.error("openai package not installed. Run: pip install openai")
            return None

    def _call(self, system: str, user: str) -> str:
        client = self._client()
        if not client:
            raise AIProviderError(
                "openai package not installed. Run: pip install openai")
        try:
            resp = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            log.error("OpenAI API error: %s", e)
            raise AIProviderError(f"OpenAI API error: {e}") from e

    def identify_speakers(self, snippets, context=""):
        if not snippets:
            return {}
        samples = _format_snippet_samples(snippets)
        user = _SPEAKER_ID_USER.format(context=context or "Not provided", samples=samples)
        raw = self._call(_SPEAKER_ID_SYSTEM, user)
        return _parse_json_mapping(raw)

    def meeting_notes(self, transcript, context=""):
        ctx = f"Context: {context}\n\n" if context else ""
        user = _MEETING_NOTES_USER.format(transcript=ctx + transcript)
        return self._call(_MEETING_NOTES_SYSTEM, user)

    def live_summary(self, transcript):
        user = _LIVE_SUMMARY_USER.format(transcript=transcript)
        return self._call(_LIVE_SUMMARY_SYSTEM, user)


# ---------------------------------------------------------------------------
# Ollama (local, no API key needed)
# ---------------------------------------------------------------------------

class OllamaProvider(AIProvider):
    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, model: str = "", base_url: str = ""):
        self._model = model or self.DEFAULT_MODEL
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        import requests
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def _call(self, system: str, user: str) -> str:
        import requests
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        try:
            r = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            log.error("Ollama error: %s", e)
            raise AIProviderError(f"Ollama error: {e}") from e

    def identify_speakers(self, snippets, context=""):
        if not snippets:
            return {}
        samples = _format_snippet_samples(snippets)
        user = _SPEAKER_ID_USER.format(context=context or "Not provided", samples=samples)
        raw = self._call(_SPEAKER_ID_SYSTEM, user)
        return _parse_json_mapping(raw)

    def meeting_notes(self, transcript, context=""):
        ctx = f"Context: {context}\n\n" if context else ""
        user = _MEETING_NOTES_USER.format(transcript=ctx + transcript)
        return self._call(_MEETING_NOTES_SYSTEM, user)

    def live_summary(self, transcript):
        user = _LIVE_SUMMARY_USER.format(transcript=transcript)
        return self._call(_LIVE_SUMMARY_SYSTEM, user)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_provider(
    provider: str,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
) -> AIProvider:
    """
    Construct the appropriate AIProvider from config values.
    Always returns a valid provider — NullProvider if unknown/unconfigured.
    """
    p = provider.lower().strip()
    if p == "claude":
        return ClaudeProvider(api_key=api_key, model=model)
    if p == "openai":
        return OpenAIProvider(api_key=api_key, model=model, base_url=base_url)
    if p == "ollama":
        return OllamaProvider(model=model, base_url=base_url)
    return NullProvider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_mapping(text: str) -> dict[str, str]:
    """Extract a JSON object from model output, tolerating surrounding prose."""
    if not text:
        return {}
    # Find the first { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        obj = json.loads(text[start:end])
        return {str(k): str(v) for k, v in obj.items() if k and v}
    except json.JSONDecodeError:
        log.warning("Could not parse AI speaker mapping: %s", text[:200])
        return {}
