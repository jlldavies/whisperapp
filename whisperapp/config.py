import json
import os
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
    vad_silence_threshold: float = 0.6
    streaming_max_chunk_sec: float = 10.0
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

    def __post_init__(self):
        if not self.default_output_path:
            self.default_output_path = str(Path.home() / "Downloads")
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

    def save(self):
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        self._path.write_text(json.dumps(data, indent=2))
