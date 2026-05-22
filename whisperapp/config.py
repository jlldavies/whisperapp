import json
import os
import secrets
from pathlib import Path
from dataclasses import dataclass, asdict, field

def _config_dir() -> Path:
    override = os.environ.get("WHISPERAPP_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".whisperapp"

@dataclass
class Config:
    hf_token: str = ""
    default_model: str = "large-v2"
    default_output_path: str = ""
    diarize_by_default: bool = True
    streaming_model: str = "base"
    vad_silence_threshold: float = 0.4
    streaming_max_chunk_sec: float = 5.0
    streaming_min_chunk_sec: float = 0.3
    streaming_show_listening: bool = True
    # AI provider settings
    ai_provider: str = "none"        # none | claude | openai | ollama
    ai_api_key: str = ""
    ai_model: str = ""               # blank = use provider default
    ai_base_url: str = ""            # Ollama URL or OpenAI-compatible endpoint
    # Startup behaviour
    auto_update: bool = True         # run the WhisperX/pyannote update check on launch
    # Pause detection settings
    pause_detection: bool = True
    pause_threshold: int = 3         # seconds
    long_pause_threshold: int = 10   # seconds
    gap_threshold: int = 30          # seconds
    # Acoustic features
    acoustic_enabled: bool = False
    acoustic_volume_enabled: bool = True
    acoustic_volume_threshold: float = 1.8
    acoustic_pitch_enabled: bool = True
    acoustic_pitch_threshold: float = 40.0
    acoustic_rate_enabled: bool = True
    acoustic_rate_fast_wpm: int = 180
    acoustic_rate_slow_wpm: int = 90
    # Emotion analysis
    emotion_enabled: bool = False
    emotion_model_ids: list = field(default_factory=list)
    emotion_confidence_threshold: float = 0.65
    emotion_combine_with_ai: bool = False
    # Output templates — blank = use bundled default
    output_template_docx: str = ""   # path to a .docx template file
    output_template_pdf: str = ""    # path to an .html template file
    # Webhook security
    # Empty list = loopback only (127.x, ::1, localhost). Use ["*"] to allow
    # any host (permissionless). Explicit entries are exact hostname/IP matches.
    webhook_allowed_hosts: list = field(default_factory=list)
    webhook_secret: str = ""         # HMAC-SHA256 key; auto-generated on first run
    # API key auth for non-loopback (headless/Docker) deployments.
    # Blank = no auth required (loopback is always unauthenticated regardless).
    api_key: str = ""

    def __post_init__(self):
        if not self.default_output_path:
            # WHISPERAPP_OUTPUT_PATH lets Docker/headless deployments set the
            # output directory via environment variable without touching config.
            env_path = os.environ.get("WHISPERAPP_OUTPUT_PATH", "")
            self.default_output_path = env_path or str(Path.home() / "Downloads")
        config_dir = _config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        self._path = config_dir / "config.json"
        if self._path.exists():
            data = json.loads(self._path.read_text())
            for k, v in data.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        else:
            self.save()
        # Auto-generate webhook secret on first run so signing works immediately.
        if not self.webhook_secret:
            self.webhook_secret = secrets.token_hex(32)
            self.save()
        # Environment variables override config file — useful for Docker/CI
        # where secrets are injected via env rather than written to disk.
        if os.environ.get("HF_TOKEN"):
            self.hf_token = os.environ["HF_TOKEN"]
        if os.environ.get("WHISPERAPP_AI_PROVIDER"):
            self.ai_provider = os.environ["WHISPERAPP_AI_PROVIDER"]
        if os.environ.get("WHISPERAPP_AI_API_KEY"):
            self.ai_api_key = os.environ["WHISPERAPP_AI_API_KEY"]
        if os.environ.get("WHISPERAPP_AI_MODEL"):
            self.ai_model = os.environ["WHISPERAPP_AI_MODEL"]
        if os.environ.get("WHISPERAPP_WEBHOOK_ALLOWED_HOSTS"):
            raw = os.environ["WHISPERAPP_WEBHOOK_ALLOWED_HOSTS"]
            self.webhook_allowed_hosts = [h.strip() for h in raw.split(",") if h.strip()]
        if os.environ.get("WHISPERAPP_WEBHOOK_SECRET"):
            self.webhook_secret = os.environ["WHISPERAPP_WEBHOOK_SECRET"]
        if os.environ.get("WHISPERAPP_API_KEY"):
            self.api_key = os.environ["WHISPERAPP_API_KEY"]

    def save(self):
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        self._path.write_text(json.dumps(data, indent=2))
