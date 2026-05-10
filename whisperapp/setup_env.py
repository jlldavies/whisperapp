"""
First-run ML dependency setup.

Checks whether the heavy ML packages (torch, whisperx, pyannote.audio) are
already installed, installs the right variants for this platform if not, and
streams pip output line-by-line so callers can show progress in a UI.
"""
import importlib.util
import platform
import subprocess
import sys
from typing import Iterator

# Packages that must be importable for the app to function.
_REQUIRED = ["torch", "whisperx", "pyannote"]


def needs_setup() -> bool:
    """True if any required ML package is missing. Fast — no imports."""
    return any(importlib.util.find_spec(pkg) is None for pkg in _REQUIRED)


def detect_existing() -> dict:
    """
    Return {package: version} for ML packages already installed.
    Uses pip show so we never import torch here (slow + risky pre-setup).
    """
    found = {}
    for pkg in ("torch", "whisperx", "pyannote.audio"):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            version = "unknown"
            location = ""
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                if line.startswith("Location:"):
                    location = line.split(":", 1)[1].strip()
            found[pkg] = {"version": version, "location": location}
    return found


def platform_info() -> dict:
    """Return platform details used to pick the right torch variant."""
    system = platform.system()
    machine = platform.machine().lower()
    cuda = _has_cuda()
    return {
        "system": system,
        "machine": machine,
        "apple_silicon": system == "Darwin" and machine == "arm64",
        "cuda": cuda,
        "torch_variant": _torch_variant_name(system, machine, cuda),
    }


def _has_cuda() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _torch_variant_name(system: str, machine: str, cuda: bool) -> str:
    if system == "Darwin" and machine == "arm64":
        return "cpu (Apple Silicon — MLX handles transcription)"
    if cuda:
        return "CUDA (GPU acceleration)"
    return "cpu"


def _torch_pip_args(info: dict) -> list[str]:
    """Return pip install args for the correct torch variant."""
    if info["apple_silicon"]:
        return ["torch", "torchaudio"]
    if info["cuda"]:
        return [
            "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118",
        ]
    return [
        "torch", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu",
    ]


# Approximate download sizes shown to the user before install starts.
_SIZE_ESTIMATES = {
    "cpu": "~800 MB",
    "cuda": "~2.5 GB",
    "apple_silicon": "~500 MB",
}


def size_estimate(info: dict) -> str:
    if info["apple_silicon"]:
        return _SIZE_ESTIMATES["apple_silicon"]
    if info["cuda"]:
        return _SIZE_ESTIMATES["cuda"]
    return _SIZE_ESTIMATES["cpu"]


def install_packages(existing: dict | None = None) -> Iterator[str]:
    """
    Install missing ML packages, yielding human-readable progress lines.
    Skips packages already present in `existing`.
    """
    info = platform_info()
    existing = existing or detect_existing()

    yield f"Platform: {info['system']} {info['machine']}\n"
    yield f"Torch variant: {info['torch_variant']}\n"
    yield f"Estimated download: {size_estimate(info)}\n"
    yield "─" * 60 + "\n"

    # ── torch ────────────────────────────────────────────────────────────
    if "torch" in existing:
        v = existing["torch"]["version"]
        yield f"✓ torch {v} already installed — skipping\n"
    else:
        args = _torch_pip_args(info)
        pkg_names = [a for a in args if not a.startswith("--") and "pytorch.org" not in a]
        yield f"Installing {', '.join(pkg_names)}...\n"
        cmd = [sys.executable, "-m", "pip", "install", "--progress-bar", "off"] + args
        ok = yield from _stream_pip(cmd)
        if not ok:
            yield "[error] torch install failed — see output above\n"
            return

    # ── whisperx ─────────────────────────────────────────────────────────
    if "whisperx" in existing:
        v = existing["whisperx"]["version"]
        yield f"✓ whisperx {v} already installed — skipping\n"
    else:
        yield "Installing whisperx...\n"
        cmd = [sys.executable, "-m", "pip", "install", "--progress-bar", "off", "whisperx"]
        ok = yield from _stream_pip(cmd)
        if not ok:
            yield "[error] whisperx install failed — see output above\n"
            return

    # ── pyannote.audio ────────────────────────────────────────────────────
    if "pyannote.audio" in existing:
        v = existing["pyannote.audio"]["version"]
        yield f"✓ pyannote.audio {v} already installed — skipping\n"
    else:
        yield "Installing pyannote.audio (speaker diarization)...\n"
        cmd = [sys.executable, "-m", "pip", "install", "--progress-bar", "off", "pyannote.audio"]
        ok = yield from _stream_pip(cmd)
        if not ok:
            yield "[error] pyannote.audio install failed — see output above\n"
            return

    # ── mlx-whisper (Apple Silicon only) ─────────────────────────────────
    if info["apple_silicon"]:
        yield "Installing mlx-whisper (fast Apple Silicon inference)...\n"
        cmd = [sys.executable, "-m", "pip", "install", "--progress-bar", "off", "mlx-whisper"]
        yield from _stream_pip(cmd)

    yield "─" * 60 + "\n"
    yield "Setup complete. Restarting WhisperApp...\n"


def _stream_pip(cmd: list) -> Iterator[str]:
    """Run a pip command and yield its combined stdout/stderr line by line."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        yield line
    proc.wait()
    return proc.returncode == 0
