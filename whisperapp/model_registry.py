"""Unified model registry for all WhisperApp model categories.

Covers Transcription, Streaming, Alignment, Diarization, and Emotion models —
plus a "Capabilities" pseudo-category for optional Python packages (e.g.
librosa for acoustic analysis) that the Models page lets users install.
For the UI layer — keeps emotion_registry.py for pipeline inference code.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

# ─── Catalogue ───────────────────────────────────────────────────────────────

MODEL_CATALOGUE: dict = {
    "transcription": {
        "title": "Transcription",
        "sub": "WhisperX models — used for the main offline pipeline",
        "models": [
            {"id": "large-v2",        "name": "large-v2",            "size_mb": 1500, "langs": "100 languages", "desc": "Default · best quality on long-form audio."},
            {"id": "large-v3",        "name": "large-v3",            "size_mb": 1500, "langs": "100 languages", "desc": "Newer release — slightly better non-English accuracy."},
            {"id": "distil-large-v3", "name": "distil-large-v3",     "size_mb": 756,  "langs": "English",       "desc": "Distilled — ~6× faster than large-v3, English only."},
            {"id": "medium",          "name": "medium",              "size_mb": 769,  "langs": "100 languages", "desc": "Balanced speed/quality. Good fit for batch jobs."},
            {"id": "small",           "name": "small",               "size_mb": 244,  "langs": "100 languages", "desc": "Light model for low-resource machines."},
            {"id": "tiny",            "name": "tiny",                "size_mb": 39,   "langs": "100 languages", "desc": "Smallest — useful for previews and tests."},
        ],
    },
    "streaming": {
        "title": "Streaming",
        "sub": "faster-whisper variants for the Live tab — small + fast wins here",
        "models": [
            {"id": "fw-base",  "name": "faster-whisper · base",  "size_mb": 74,  "langs": "100 languages", "desc": "Default streaming model — sub-second latency."},
            {"id": "fw-small", "name": "faster-whisper · small", "size_mb": 244, "langs": "100 languages", "desc": "Better accuracy at the cost of ~1.6× latency."},
            {"id": "fw-tiny",  "name": "faster-whisper · tiny",  "size_mb": 39,  "langs": "100 languages", "desc": "Lightest option — useful on battery."},
        ],
    },
    "alignment": {
        "title": "Alignment",
        "sub": "Per-language phoneme models — required for word-level timestamps",
        "language_filterable": True,
        "models": [
            {"id": "align-en", "name": "wav2vec2-large-960h",    "size_mb": 360, "lang": "en", "langs": "English",     "desc": "Forced alignment for English transcripts."},
            {"id": "align-fr", "name": "wav2vec2-large-xlsr-fr", "size_mb": 360, "lang": "fr", "langs": "French",      "desc": "French phoneme alignment."},
            {"id": "align-de", "name": "wav2vec2-large-xlsr-de", "size_mb": 360, "lang": "de", "langs": "German",      "desc": "German phoneme alignment."},
            {"id": "align-es", "name": "wav2vec2-large-xlsr-es", "size_mb": 360, "lang": "es", "langs": "Spanish",     "desc": "Spanish phoneme alignment."},
            {"id": "align-it", "name": "wav2vec2-large-xlsr-it", "size_mb": 360, "lang": "it", "langs": "Italian",     "desc": "Italian phoneme alignment."},
            {"id": "align-pt", "name": "wav2vec2-large-xlsr-pt", "size_mb": 360, "lang": "pt", "langs": "Portuguese",  "desc": "Portuguese phoneme alignment."},
            {"id": "align-nl", "name": "wav2vec2-large-xlsr-nl", "size_mb": 360, "lang": "nl", "langs": "Dutch",       "desc": "Dutch phoneme alignment."},
            {"id": "align-ja", "name": "wav2vec2-large-xlsr-ja", "size_mb": 360, "lang": "ja", "langs": "Japanese",    "desc": "Japanese phoneme alignment."},
            {"id": "align-zh", "name": "wav2vec2-large-xlsr-zh", "size_mb": 360, "lang": "zh", "langs": "Chinese",     "desc": "Mandarin phoneme alignment."},
        ],
    },
    "diarization": {
        "title": "Diarization",
        "sub": "pyannote.audio — speaker turn detection (requires HF token)",
        "models": [
            {"id": "pyannote-3.1", "name": "pyannote · speaker-diarization-3.1", "size_mb": 27,  "langs": "Language-agnostic", "desc": "Current default. Best balance of accuracy and speed."},
            {"id": "pyannote-3.0", "name": "pyannote · speaker-diarization-3.0", "size_mb": 27,  "langs": "Language-agnostic", "desc": "Previous version — kept for reproducibility."},
            {"id": "pyannote-sep", "name": "pyannote · speech-separation-ami",   "size_mb": 250, "langs": "Language-agnostic", "desc": "Adds source separation for overlapping speech."},
        ],
    },
    "emotion": {
        "title": "Emotion",
        "sub": "Optional — adds emotional tone per utterance",
        "models": [
            {"id": "speechbrain-iemocap",  "name": "SpeechBrain IEMOCAP",    "size_mb": 300,  "langs": "English",      "desc": "Fast 4-label classifier — best default.", "recommended": True},
            {"id": "wavlm-multi",          "name": "WavLM Multi-Corpus",     "size_mb": 400,  "langs": "English",      "desc": "Trained on a wider corpus mix."},
            {"id": "wav2vec2-8emotion",    "name": "wav2vec2 · 8 emotions",  "size_mb": 1200, "langs": "English",      "desc": "Wider emotional vocabulary at a quality cost."},
            {"id": "audeering-dimensional","name": "Audeering Dimensional",  "size_mb": 1400, "langs": "Multilingual", "desc": "Continuous arousal/dominance/valence scores."},
        ],
    },
    "capabilities": {
        "title": "Capabilities",
        "sub": "Optional Python packages that unlock features (no LLM required)",
        "models": [
            {
                "id": "librosa",
                "name": "librosa",
                "package": "librosa",
                "size_mb": 80,
                "langs": "—",
                "desc": "Audio DSP — required by Acoustic settings (volume / pitch / "
                        "speaking-rate markers). No AI provider needed.",
                "feature": "Acoustic analysis",
                "recommended": True,
            },
        ],
    },
}


# ─── Capability install state ────────────────────────────────────────────────
# Capabilities are pip packages. We track in-flight install / update / uninstall
# operations in this module-level dict so the catalogue endpoint can report
# `installing` / `updating` state and the UI can disable buttons accordingly.

_capability_state: dict[str, dict] = {}
_capability_lock = threading.Lock()

# In-process cache of PyPI version lookups. The whole point is to avoid
# hammering PyPI on every catalogue render — entries live for 6 hours by
# default, and `force_check_updates()` clears them.
_PYPI_CACHE_TTL = 6 * 3600  # seconds
_pypi_cache: dict[str, dict] = {}  # {pkg: {version: str|None, fetched_at: float}}
_pypi_lock = threading.Lock()

# HuggingFace revision cache.
# `local_sha`  is the commit sha the local hub cache currently points at (cheap
#              filesystem read).
# `remote_sha` is what HF reports as the latest. We only fetch it when the
#              user clicks "Check for updates" — never on a passive catalogue
#              render — so the cache starts empty and only fills on demand.
_hf_revision_cache: dict[str, dict] = {}
_hf_lock = threading.Lock()

# In-flight HF model update operations (by model_id), so the catalogue
# can render the existing `downloading` state while a snapshot_download
# is running and the user can't kick off a second one.
_hf_op_state: dict[str, dict] = {}
_hf_op_lock = threading.Lock()


def _run_hf_update(model_id: str, repo: str) -> None:
    """Background: re-pull the latest revision into the local HF cache."""
    def _runner():
        with _hf_op_lock:
            _hf_op_state[model_id] = {"op": "update", "error": None}
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo, revision="main")
            with _hf_op_lock:
                _hf_op_state[model_id] = {"op": None, "error": None}
            # Refresh cached shas so the catalogue immediately drops the
            # `update` badge after the download completes.
            local = _local_hf_sha(repo)
            with _hf_lock:
                rev = _hf_revision_cache.get(model_id, {})
                rev["local_sha"] = local
                _hf_revision_cache[model_id] = rev
        except Exception as e:
            with _hf_op_lock:
                _hf_op_state[model_id] = {"op": None, "error": str(e)}

    threading.Thread(target=_runner, daemon=True).start()


def _pypi_latest(pkg: str, *, force: bool = False) -> Optional[str]:
    """Return the latest version on PyPI, or None on any failure (offline,
    rate-limited, package not found)."""
    import time as _t
    now = _t.time()
    with _pypi_lock:
        cached = _pypi_cache.get(pkg)
        if not force and cached and now - cached["fetched_at"] < _PYPI_CACHE_TTL:
            return cached["version"]
    version = None
    try:
        import requests as _req
        r = _req.get(f"https://pypi.org/pypi/{pkg}/json", timeout=8)
        r.raise_for_status()
        version = r.json().get("info", {}).get("version")
    except Exception:
        version = None
    with _pypi_lock:
        _pypi_cache[pkg] = {"version": version, "fetched_at": now}
    return version


def _hf_repo_for(model_id: str) -> Optional[str]:
    """Return the HuggingFace repo path for a catalogue model id, or None
    if the catalogue entry doesn't have a real HF repo (some alignment
    entries reference WhisperX-internal names that aren't actual repos)."""
    repo = _HF_REPOS.get(model_id)
    if not repo or "/" not in repo:
        return None
    return repo


def _local_hf_sha(repo: str) -> Optional[str]:
    """Read the active commit sha from the local HuggingFace hub cache.

    `~/.cache/huggingface/hub/models--<owner>--<name>/refs/main` is a tiny
    text file that holds the sha that the snapshot symlinks point at — the
    `huggingface_hub` library updates it on each download. Reading it is the
    canonical way to know which revision is locally installed.
    """
    slug = "models--" + repo.replace("/", "--")
    for cache_dir in _hf_cache_dirs():
        ref = cache_dir / slug / "refs" / "main"
        if ref.exists():
            try:
                return ref.read_text(encoding="utf-8").strip() or None
            except Exception:
                pass
    return None


def _remote_hf_sha(repo: str) -> Optional[str]:
    """Ask HuggingFace for the current main-branch commit sha. Best-effort —
    on any failure (offline, gated repo, rate-limited) returns None and the
    catalogue simply omits update info for that model."""
    try:
        import requests as _req
        r = _req.get(
            f"https://huggingface.co/api/models/{repo}/revision/main",
            timeout=8,
        )
        if r.status_code != 200:
            return None
        return r.json().get("sha")
    except Exception:
        return None


def _has_update(installed: Optional[str], latest: Optional[str]) -> bool:
    """PEP 440 version comparison via packaging (which pip itself uses, so
    always present). Falls back to string-equality on parse failure."""
    if not installed or not latest or installed == latest:
        return False
    try:
        from packaging.version import Version
        return Version(latest) > Version(installed)
    except Exception:
        return False


def _capability_status(pkg: str) -> dict:
    """Return {installed: bool, version: str|None} for a pip package."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return {"installed": True, "version": version(pkg)}
        except PackageNotFoundError:
            return {"installed": False, "version": None}
    except Exception:
        return {"installed": False, "version": None}


def _run_pip(cap_id: str, op: str, args: list[str]) -> None:
    """Run a pip operation in a background thread, tracking state."""
    def _runner():
        with _capability_lock:
            _capability_state[cap_id] = {"op": op, "log": "", "error": None}
        try:
            cp = subprocess.run(
                [sys.executable, "-m", "pip", *args],
                capture_output=True, text=True, timeout=600,
            )
            with _capability_lock:
                st = _capability_state.get(cap_id, {})
                st["log"] = (cp.stdout or "") + (cp.stderr or "")
                st["error"] = None if cp.returncode == 0 else f"pip exit {cp.returncode}"
                st["op"] = None
                _capability_state[cap_id] = st
        except Exception as e:
            with _capability_lock:
                _capability_state[cap_id] = {"op": None, "log": "", "error": str(e)}

    threading.Thread(target=_runner, daemon=True).start()

# HuggingFace repo IDs for non-emotion models — used for both cache
# detection (`_is_hf_cached`) and revision-update checks.
#
# WhisperX downloads from `Systran/faster-whisper-*`, not `openai/whisper-*`,
# so the transcription category maps there. The pyannote-3.x catalogue
# entries are display-only labels — the actual install pulls
# `pyannote/speaker-diarization-community-1`. Alignment models that don't
# have a real HF repo (WhisperX-internal IDs like `WAV2VEC2-LARGE-960H`)
# are intentionally left without a slash so `_hf_repo_for()` skips them.
_HF_REPOS = {
    "large-v2":        "Systran/faster-whisper-large-v2",
    "large-v3":        "Systran/faster-whisper-large-v3",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "medium":          "Systran/faster-whisper-medium",
    "small":           "Systran/faster-whisper-small",
    "tiny":            "Systran/faster-whisper-tiny",
    "fw-base":         "Systran/faster-whisper-base",
    "fw-small":        "Systran/faster-whisper-small",
    "fw-tiny":         "Systran/faster-whisper-tiny",
    "align-en":        "WAV2VEC2-LARGE-960H",
    "align-fr":        "VOXPOPULI-300M-FR",
    "pyannote-3.1":    "pyannote/speaker-diarization-community-1",
    "pyannote-3.0":    "pyannote/speaker-diarization-3.0",
    "pyannote-sep":    "pyannote/speech-separation-ami-1.0",
}


def _hf_cache_dirs() -> list[Path]:
    """Return candidate HuggingFace / WhisperX cache directories."""
    home = Path.home()
    hf_env = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    dirs = []
    if hf_env:
        dirs.append(Path(hf_env))
    dirs += [
        home / ".cache" / "huggingface" / "hub",
        home / ".cache" / "huggingface",
        home / ".cache" / "whisperx",
    ]
    return dirs


def _is_hf_cached(model_id: str) -> bool:
    """Heuristic: check if a model is present in any known HF cache."""
    repo = _HF_REPOS.get(model_id)
    if not repo:
        return False
    # HuggingFace stores models as models--owner--repo or models--name
    slug = "models--" + repo.replace("/", "--")
    for cache_dir in _hf_cache_dirs():
        if (cache_dir / slug).exists():
            return True
        # Some older layouts just use the repo name directly
        if (cache_dir / repo.split("/")[-1]).exists():
            return True
    return False


def _get_active_transcription_model() -> str:
    """Return the active transcription model id from config, or default."""
    try:
        from whisperapp.config import Config
        cfg = Config()
        model = cfg.default_model or "large-v2"
        # Map config model names to our catalogue IDs
        return model
    except Exception:
        return "large-v2"


def _get_active_streaming_model() -> str:
    """Return the active streaming model id from config."""
    try:
        from whisperapp.config import Config
        cfg = Config()
        model = cfg.streaming_model or "base"
        # Map base → fw-base etc.
        _map = {"base": "fw-base", "small": "fw-small", "tiny": "fw-tiny"}
        return _map.get(model, "fw-base")
    except Exception:
        return "fw-base"


def _get_active_diarization_model() -> str:
    """Return the active diarization model id."""
    # Default to pyannote-3.1 unless config specifies otherwise
    return "pyannote-3.1"


class ModelRegistry:
    """Unified registry for all model categories."""

    def list_catalogue(self) -> dict:
        """Return full catalogue with state for each model."""
        from whisperapp.emotion_registry import ModelManager
        emotion_mgr = ModelManager()
        emotion_statuses = {m["id"]: m for m in emotion_mgr.list_all()}

        active_transcription = _get_active_transcription_model()
        active_streaming = _get_active_streaming_model()
        active_diarization = _get_active_diarization_model()

        result = {}
        for cat_key, cat in MODEL_CATALOGUE.items():
            cat_data = {k: v for k, v in cat.items() if k != "models"}
            cat_data["models"] = []

            for m in cat["models"]:
                entry = dict(m)
                entry["size"] = self.format_size(m["size_mb"])

                if cat_key == "emotion":
                    # Delegate to emotion_registry for real state
                    em = emotion_statuses.get(m["id"], {})
                    if em.get("downloading"):
                        entry["state"] = "downloading"
                        entry["progress"] = em.get("progress", 0) / 100.0
                    elif em.get("downloaded"):
                        # Check if active
                        try:
                            from whisperapp.config import Config
                            cfg = Config()
                            ids = cfg.emotion_model_ids or []
                        except Exception:
                            ids = []
                        if m["id"] in ids:
                            entry["state"] = "active"
                            entry["role"] = "Default" if m.get("recommended") else "Active"
                        else:
                            entry["state"] = "installed"
                    else:
                        entry["state"] = "available"

                elif cat_key in ("transcription", "streaming", "alignment", "diarization"):
                    cached = _is_hf_cached(m["id"])
                    # Per-category "active" rules
                    if cat_key == "transcription":
                        is_active = m["id"] == active_transcription
                        active_role = "Default"
                    elif cat_key == "streaming":
                        is_active = m["id"] == active_streaming
                        active_role = "Default"
                    elif cat_key == "alignment":
                        is_active = m["id"] == "align-en"  # English = auto default
                        active_role = "Auto"
                    else:  # diarization
                        is_active = m["id"] == active_diarization
                        active_role = "Default"

                    if is_active:
                        entry["state"] = "active"
                        entry["role"] = active_role
                    elif cached:
                        entry["state"] = "installed"
                    else:
                        entry["state"] = "available"

                    # In-flight `Update` from a previous click? Show the
                    # spinner-style state and skip update detection while it
                    # runs.
                    with _hf_op_lock:
                        in_flight = _hf_op_state.get(m["id"], {}).get("op")
                    if in_flight:
                        entry["state"] = "downloading"
                        entry.pop("role", None)
                    elif entry["state"] in ("installed", "active"):
                        # If we previously fetched the remote sha (Check for
                        # updates was clicked), promote installed/active
                        # entries to `update` when local and remote diverge.
                        repo = _hf_repo_for(m["id"])
                        if repo:
                            with _hf_lock:
                                rev = _hf_revision_cache.get(m["id"], {})
                            local = _local_hf_sha(repo)
                            remote = rev.get("remote_sha")
                            if local and remote and local != remote:
                                entry["state"] = "update"
                                entry["current"] = local[:7]
                                entry["available"] = remote[:7]

                elif cat_key == "capabilities":
                    pkg = m.get("package", m["id"])
                    status = _capability_status(pkg)
                    with _capability_lock:
                        op_state = _capability_state.get(m["id"], {})
                    entry["installed_version"] = status["version"]
                    entry["last_error"] = op_state.get("error")
                    if op_state.get("op") in ("install", "update", "uninstall"):
                        entry["state"] = "downloading"  # reuse the existing badge
                    elif status["installed"]:
                        # Compare to PyPI; promote to `update` state when newer.
                        latest = _pypi_latest(pkg)
                        if latest and _has_update(status["version"], latest):
                            entry["state"] = "update"
                            entry["current"] = status["version"]
                            entry["available"] = latest
                        else:
                            entry["state"] = "installed"
                            entry["role"] = "Installed"
                    else:
                        entry["state"] = "available"

                cat_data["models"].append(entry)
            result[cat_key] = cat_data

        return result

    def get_disk_summary(self) -> dict:
        """Return {category: {bytes: int, count: int, label: str}} for installed models."""
        catalogue = self.list_catalogue()
        summary = {}
        for cat_key, cat in catalogue.items():
            total_bytes = 0
            count = 0
            for m in cat["models"]:
                if m.get("state") in ("active", "installed", "update", "downloading"):
                    total_bytes += m.get("size_mb", 0) * 1024 * 1024
                    count += 1
            summary[cat_key] = {
                "bytes": total_bytes,
                "count": count,
                "label": cat["title"],
            }
        return summary

    def format_size(self, size_mb: int) -> str:
        """Format MB to human-readable string."""
        if size_mb >= 1024:
            return f"{size_mb / 1024:.2f} GB"
        return f"{size_mb} MB"

    def activate_model(self, model_id: str) -> dict:
        """Set a model as active in its category. Persists to config for emotion models."""
        # Find the category
        cat_key = self._find_category(model_id)
        if not cat_key:
            return {"success": False, "error": f"Model not found: {model_id}"}

        if cat_key == "emotion":
            try:
                from whisperapp.config import Config
                cfg = Config()
                ids = list(cfg.emotion_model_ids or [])
                if model_id not in ids:
                    ids.append(model_id)
                cfg.emotion_model_ids = ids
                cfg.save()
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}
        elif cat_key == "transcription":
            try:
                from whisperapp.config import Config
                cfg = Config()
                cfg.default_model = model_id
                cfg.save()
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}
        elif cat_key == "streaming":
            _map = {"fw-base": "base", "fw-small": "small", "fw-tiny": "tiny"}
            model_name = _map.get(model_id, model_id)
            try:
                from whisperapp.config import Config
                cfg = Config()
                cfg.streaming_model = model_name
                cfg.save()
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            # alignment / diarization — not yet persisted
            return {"success": True, "note": "not_persisted"}

    def deactivate_model(self, model_id: str) -> dict:
        """Remove a model from the active set."""
        cat_key = self._find_category(model_id)
        if not cat_key:
            return {"success": False, "error": f"Model not found: {model_id}"}

        if cat_key == "emotion":
            try:
                from whisperapp.config import Config
                cfg = Config()
                ids = [i for i in (cfg.emotion_model_ids or []) if i != model_id]
                cfg.emotion_model_ids = ids
                cfg.save()
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {"success": True, "note": "not_implemented_for_category"}

    def _find_category(self, model_id: str) -> Optional[str]:
        for cat_key, cat in MODEL_CATALOGUE.items():
            for m in cat["models"]:
                if m["id"] == model_id:
                    return cat_key
        return None

    # ─── Capability management ───────────────────────────────────────────────

    def is_capability(self, model_id: str) -> bool:
        return self._find_category(model_id) == "capabilities"

    def _capability_def(self, model_id: str) -> Optional[dict]:
        for m in MODEL_CATALOGUE["capabilities"]["models"]:
            if m["id"] == model_id:
                return m
        return None

    def install_capability(self, model_id: str) -> dict:
        cap = self._capability_def(model_id)
        if not cap:
            return {"success": False, "error": f"Unknown capability: {model_id}"}
        with _capability_lock:
            if _capability_state.get(model_id, {}).get("op"):
                return {"success": False, "error": "Operation already in progress"}
        _run_pip(model_id, "install", ["install", cap["package"]])
        return {"success": True, "message": f"Installing {cap['package']}…"}

    def update_capability(self, model_id: str) -> dict:
        cap = self._capability_def(model_id)
        if not cap:
            return {"success": False, "error": f"Unknown capability: {model_id}"}
        with _capability_lock:
            if _capability_state.get(model_id, {}).get("op"):
                return {"success": False, "error": "Operation already in progress"}
        _run_pip(model_id, "update", ["install", "--upgrade", cap["package"]])
        return {"success": True, "message": f"Updating {cap['package']}…"}

    def update_hf_model(self, model_id: str) -> dict:
        """Trigger a re-download of an HF-backed model in the background."""
        repo = _hf_repo_for(model_id)
        if not repo:
            return {"success": False, "error": f"No HF repo for {model_id}"}
        with _hf_op_lock:
            if _hf_op_state.get(model_id, {}).get("op"):
                return {"success": False, "error": "Update already in progress"}
        _run_hf_update(model_id, repo)
        return {"success": True, "message": f"Updating {model_id}…"}

    def force_check_updates(self) -> dict:
        """Clear update-check caches and re-query so the next catalogue
        render shows fresh `state == "update"` flags.

        Two checks happen here, both deliberately on-demand only — neither
        runs on a passive catalogue render:
          1. PyPI lookup for every capability (3-5 tops, sequential).
          2. HuggingFace revision check for every locally-cached HF model
             (parallel — one HEAD per repo, ~6 in flight).

        Returns a small summary for the UI's 'Check for updates' button.
        """
        # 1. Capabilities → PyPI
        with _pypi_lock:
            _pypi_cache.clear()
        cap_results = []
        for m in MODEL_CATALOGUE.get("capabilities", {}).get("models", []):
            pkg = m.get("package", m["id"])
            cap_results.append({"id": m["id"], "latest": _pypi_latest(pkg, force=True)})

        # 2. HF models → HF API (parallel)
        with _hf_lock:
            _hf_revision_cache.clear()

        targets: list[tuple[str, str]] = []  # (model_id, repo)
        for cat_key in ("transcription", "streaming", "alignment", "diarization"):
            for m in MODEL_CATALOGUE.get(cat_key, {}).get("models", []):
                repo = _hf_repo_for(m["id"])
                if repo and _is_hf_cached(m["id"]):
                    targets.append((m["id"], repo))

        from concurrent.futures import ThreadPoolExecutor
        import time as _t

        def _probe(item):
            mid, repo = item
            return mid, repo, _local_hf_sha(repo), _remote_hf_sha(repo)

        hf_results = []
        if targets:
            with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
                for mid, repo, local, remote in pool.map(_probe, targets):
                    with _hf_lock:
                        _hf_revision_cache[mid] = {
                            "local_sha": local,
                            "remote_sha": remote,
                            "fetched_at": _t.time(),
                        }
                    hf_results.append({
                        "id": mid,
                        "current": local[:7] if local else None,
                        "available": remote[:7] if remote else None,
                        "needs_update": bool(local and remote and local != remote),
                    })

        return {
            "checked": True,
            "capabilities": cap_results,
            "models": hf_results,
            "updates_available": (
                sum(1 for c in cap_results
                    if c["latest"] and _has_update(_capability_status(c["id"])["version"], c["latest"]))
                + sum(1 for h in hf_results if h["needs_update"])
            ),
        }

    def uninstall_capability(self, model_id: str) -> dict:
        cap = self._capability_def(model_id)
        if not cap:
            return {"success": False, "error": f"Unknown capability: {model_id}"}
        with _capability_lock:
            if _capability_state.get(model_id, {}).get("op"):
                return {"success": False, "error": "Operation already in progress"}
        _run_pip(model_id, "uninstall", ["uninstall", "-y", cap["package"]])
        return {"success": True, "message": f"Uninstalling {cap['package']}…"}
