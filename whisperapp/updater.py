import subprocess
import sys
import logging
from whisperapp import __version__

log = logging.getLogger(__name__)

PACKAGES = ["whisperx", "pyannote.audio", "fastapi"]
RELEASES_URL = "https://api.github.com/repos/jlldavies/whisperapp/releases/latest"

_latest_version: str | None = None   # cached result of check_app_update()


def run_update():
    """Silently upgrade key packages. Fails gracefully."""
    for pkg in PACKAGES:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", pkg],
                capture_output=True, timeout=120
            )
            if result.returncode != 0:
                log.warning(f"Update failed for {pkg}: {result.stderr.decode()}")
        except Exception as e:
            log.warning(f"Could not update {pkg}: {e}")
    check_app_update()


def check_app_update() -> str | None:
    """
    Query the GitHub releases API for the latest WhisperApp version.
    Returns the tag name (e.g. "v1.2.0") if newer than the running version,
    None otherwise. Result is cached for the lifetime of the process.
    """
    global _latest_version
    try:
        import requests
        r = requests.get(RELEASES_URL, timeout=8,
                         headers={"Accept": "application/vnd.github+json"})
        if r.status_code != 200:
            return None
        tag = r.json().get("tag_name", "").lstrip("v")
        if _is_newer(tag, __version__):
            _latest_version = tag
            return tag
        _latest_version = None
        return None
    except Exception as exc:
        log.debug("App update check failed: %s", exc)
        return None


def latest_version() -> str | None:
    """Return the cached latest version string, or None if up to date."""
    return _latest_version


def _is_newer(candidate: str, current: str) -> bool:
    try:
        def parts(v):
            return tuple(int(x) for x in v.split("."))
        return parts(candidate) > parts(current)
    except ValueError:
        return False
