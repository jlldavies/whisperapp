import re
from pathlib import Path
import bleach

ALLOWED_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mkv"}
ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large-v2"}
JOB_ID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

def sanitise_file_path(raw: str) -> Path:
    path = Path(raw).resolve()
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported extension '{path.suffix}'. Allowed: {ALLOWED_EXTENSIONS}")
    return path

def sanitise_output_path(raw: str) -> Path:
    if ".." in Path(raw).parts:
        raise ValueError("Path traversal detected in output path")
    path = Path(raw).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

def sanitise_model(raw: str) -> str:
    if raw not in ALLOWED_MODELS:
        raise ValueError(f"Invalid model '{raw}'. Allowed: {ALLOWED_MODELS}")
    return raw

def sanitise_job_id(raw: str) -> str:
    if not JOB_ID_RE.match(raw):
        raise ValueError(f"Invalid job_id format: {raw!r}")
    return raw

def sanitise_speaker_name(raw: str) -> str:
    cleaned = bleach.clean(raw, tags=[], strip=True)
    return cleaned[:50]
