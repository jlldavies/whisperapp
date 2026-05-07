import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict

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
