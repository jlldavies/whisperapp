# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for WhisperApp.

Build commands:
  Windows:  pyinstaller whisperapp.spec
  macOS:    pyinstaller whisperapp.spec

The spec produces:
  dist/WhisperApp/WhisperApp.exe   (Windows — no console window)
  dist/WhisperApp.app              (macOS — .app bundle)
"""

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['whisperapp/__main__.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # Include the MCP manifest so users can find it next to the binary
        ('mcp.json', '.'),
    ],
    hiddenimports=[
        # whisperapp modules
        'whisperapp.config',
        'whisperapp.queue',
        'whisperapp.worker',
        'whisperapp.server',
        'whisperapp.ui',
        'whisperapp.tray',
        'whisperapp.startup',
        'whisperapp.streaming',
        'whisperapp.cli',
        'whisperapp.formatters',
        'whisperapp.speakers',
        'whisperapp.checkpoints',
        'whisperapp.sanitise',
        'whisperapp.updater',
        # runtime deps that PyInstaller misses via static analysis
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'gradio',
        'pystray',
        'pystray._win32',
        'pystray._darwin',
        'pystray._xorg',
        'PIL',
        'pyaudio',
        'click',
        'requests',
        'httpx',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'IPython'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ------------------------------------------------------------------
# Windows / Linux: single directory bundle
# ------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WhisperApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # no console window — tray app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if Path('assets/icon.ico').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WhisperApp',
)

# ------------------------------------------------------------------
# macOS: .app bundle (only built on macOS)
# ------------------------------------------------------------------
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='WhisperApp.app',
        icon='assets/icon.icns' if Path('assets/icon.icns').exists() else None,
        bundle_identifier='com.whisperapp',
        info_plist={
            'CFBundleName': 'WhisperApp',
            'CFBundleDisplayName': 'WhisperApp',
            'CFBundleVersion': '1.1.0',
            'CFBundleShortVersionString': '1.1.0',
            'NSHighResolutionCapable': True,
            'LSUIElement': True,          # hide from Dock — menu bar only
            'NSMicrophoneUsageDescription':
                'WhisperApp records audio from your microphone for transcription.',
        },
    )
