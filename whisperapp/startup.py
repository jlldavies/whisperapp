"""
Startup registration — registers WhisperApp to launch at login.

Strategy (in priority order):
1. If a PyInstaller frozen executable exists alongside sys.executable, use it.
2. If the `whisperapp-app` script is installed in the same Scripts/bin dir, use it.
3. Fall back to `pythonw.exe -m whisperapp` (development / pip install -e).
"""

import sys
import subprocess
from pathlib import Path

if sys.platform == "win32":
    import winreg

APP_NAME = "WhisperApp"
PLIST_LABEL = "com.whisperapp"
PLIST_PATH = Path.home() / "Library/LaunchAgents/com.whisperapp.plist"


# ---------------------------------------------------------------------------
# Executable resolution
# ---------------------------------------------------------------------------

def _launch_command() -> list[str]:
    """Return the argv list that should be registered at startup."""
    python_dir = Path(sys.executable).parent

    # 1. Frozen PyInstaller bundle
    if getattr(sys, "frozen", False):
        return [sys.executable]

    # 2. Installed gui-script (whisperapp-app)
    if sys.platform == "win32":
        candidate = python_dir / "whisperapp-app.exe"
    else:
        candidate = python_dir / "whisperapp-app"

    if candidate.exists():
        return [str(candidate)]

    # 3. Development fallback — windowless python
    if sys.platform == "win32":
        pythonw = python_dir / "pythonw.exe"
        py = str(pythonw) if pythonw.exists() else sys.executable
        return [py, "-m", "whisperapp"]
    else:
        return [sys.executable, "-m", "whisperapp"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_startup():
    if sys.platform == "win32":
        _register_windows()
    elif sys.platform == "darwin":
        _register_macos()


def unregister_startup():
    if sys.platform == "win32":
        _unregister_windows()
    elif sys.platform == "darwin":
        _unregister_macos()


# ---------------------------------------------------------------------------
# Platform implementations
# ---------------------------------------------------------------------------

def _register_windows():
    cmd_parts = _launch_command()
    # Quote each part and join so the registry value is a valid command line
    cmd = " ".join(f'"{p}"' if " " in p else p for p in cmd_parts)
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0,
                        winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, cmd)


def _unregister_windows():
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0,
                            winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, APP_NAME)
    except FileNotFoundError:
        pass


def _register_macos():
    args = _launch_command()
    arg_elements = "\n        ".join(f"<string>{a}</string>" for a in args)
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        {arg_elements}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>{Path.home()}/.whisperapp/launch.log</string>
    <key>StandardErrorPath</key>
    <string>{Path.home()}/.whisperapp/launch.log</string>
</dict>
</plist>"""
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(plist)
    # Unload first in case a stale entry exists
    subprocess.run(["launchctl", "unload", str(PLIST_PATH)],
                   check=False, capture_output=True)
    subprocess.run(["launchctl", "load", str(PLIST_PATH)], check=False)


def _unregister_macos():
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)],
                       check=False, capture_output=True)
        PLIST_PATH.unlink()
