import subprocess
import sys
import logging

log = logging.getLogger(__name__)

PACKAGES = ["whisperx", "pyannote.audio", "gradio", "fastapi"]

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
