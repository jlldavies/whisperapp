from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

_STATIC_DIR = Path(__file__).parent / "static"


def create_ui(queue, worker) -> FastAPI:
    app = FastAPI()
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
    return app


def list_audio_devices() -> list[dict]:
    """Return list of audio input devices. Returns [] if pyaudio unavailable."""
    try:
        import pyaudio
    except ImportError:
        return []
    p = pyaudio.PyAudio()
    devices = []
    seen: set[str] = set()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and info["name"] not in seen:
            seen.add(info["name"])
            devices.append({
                "name": info["name"],
                "index": i,
                "sample_rate": int(info["defaultSampleRate"]),
                "channels": int(info["maxInputChannels"]),
            })
    p.terminate()
    return devices
