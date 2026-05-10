from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

_STATIC_DIR = Path(__file__).parent / "static"


def create_ui(queue, worker) -> FastAPI:
    app = FastAPI()
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
    return app


def list_audio_devices() -> list[dict]:
    """Return list of audio input devices. Returns [] if sounddevice unavailable."""
    try:
        import sounddevice as sd
    except ImportError:
        return []
    devices = []
    seen: set[str] = set()
    for d in sd.query_devices():
        if d["max_input_channels"] > 0 and d["name"] not in seen:
            seen.add(d["name"])
            devices.append({
                "name": d["name"],
                "index": d.get("index", len(devices)),
                "sample_rate": int(d["default_samplerate"]),
                "channels": int(d["max_input_channels"]),
            })
    return devices
