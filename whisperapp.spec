# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for the WhisperApp launcher.
#
# This produces a ~50 MB bundle: Python + the lightweight whisperapp shell
# (UI, API server, queue, config, setup flow). torch, whisperx, pyannote,
# and librosa are intentionally excluded — they are downloaded at runtime
# via the built-in first-run setup screen.
#
# The build workflow installs only the lightweight deps before running
# PyInstaller, so the heavy ML packages are never visible to the analyser.
#
# Build manually (in a clean env — no torch):
#   pip install ".[desktop]" pyinstaller
#   pyinstaller whisperapp.spec
#
# Output (then wrap with DMG / Inno Setup):
#   dist/WhisperApp/          (Windows / Linux directory bundle)
#   dist/WhisperApp.app/      (macOS .app bundle)

import sys
from pathlib import Path

block_cipher = None

# Modules that must never be bundled (they're installed at runtime).
EXCLUDED = [
    "torch", "torchvision", "torchaudio",
    "whisperx", "faster_whisper", "ctranslate2",
    "pyannote", "pyannote.audio",
    "mlx", "mlx_whisper",
    "librosa", "soundfile", "audioread",
    "sklearn", "scipy", "matplotlib",
    "transformers", "huggingface_hub", "tokenizers",
    "numba", "llvmlite",
    "tkinter",
]

a = Analysis(
    ["whisperapp/__main__.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("whisperapp/static",    "whisperapp/static"),
        ("whisperapp/templates", "whisperapp/templates"),
        ("mcp.json",             "."),
    ],
    hiddenimports=[
        # Core app modules not reachable via static analysis
        # (worker/streaming omitted — they import torch at module level)
        "whisperapp.setup_env",
        "whisperapp.config",
        "whisperapp.queue",
        "whisperapp.server",
        "whisperapp.sanitise",
        "whisperapp.checkpoints",
        "whisperapp.updater",
        "whisperapp.cli",
        # uvicorn dynamic imports
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        # Desktop extras
        "pystray",
        "pystray._win32",
        "pystray._darwin",
        "pystray._xorg",
        "PIL",
        "pyaudio",
    ],
    excludes=EXCLUDED,
    hookspath=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WhisperApp",
    debug=False,
    strip=False,
    upx=False,                 # UPX can break some extensions
    console=False,             # no console window
    icon="assets/icon.ico" if Path("assets/icon.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="WhisperApp",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="WhisperApp.app",
        icon="assets/icon.icns" if Path("assets/icon.icns").exists() else None,
        bundle_identifier="com.handleport.whisperapp",
        info_plist={
            "CFBundleName": "WhisperApp",
            "CFBundleDisplayName": "WhisperApp",
            "CFBundleVersion": "1.1.0",
            "CFBundleShortVersionString": "1.1.0",
            "NSHighResolutionCapable": True,
            "LSUIElement": False,
            "NSMicrophoneUsageDescription":
                "WhisperApp uses the microphone for live transcription.",
        },
    )
