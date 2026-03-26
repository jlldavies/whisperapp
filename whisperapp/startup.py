import sys
import subprocess
from pathlib import Path

if sys.platform == "win32":
    import winreg

APP_NAME = "WhisperApp"
PLIST_PATH = Path.home() / "Library/LaunchAgents/com.whisperapp.plist"

def _python_path() -> str:
    return sys.executable

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

def _pythonw_path() -> str:
    """Return pythonw.exe path for windowless startup on Windows."""
    p = Path(_python_path())
    pw = p.parent / "pythonw.exe"
    return str(pw) if pw.exists() else str(p)

def _register_windows():
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    cmd = f'"{_pythonw_path()}" -m whisperapp'
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
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
    <key>Label</key><string>com.whisperapp</string>
    <key>ProgramArguments</key>
    <array>
        <string>{_python_path()}</string>
        <string>-m</string><string>whisperapp</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><false/>
</dict></plist>"""
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(plist)
    subprocess.run(["launchctl", "load", str(PLIST_PATH)], check=False)

def _unregister_macos():
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], check=False)
        PLIST_PATH.unlink()
