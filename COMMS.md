# WhisperApp Cross-Machine Debug Comms File
# Synced via Dropbox between Windows (DESKTOP-3OC30OG) and Mac (macbookpro)
# Both Claude sessions read and write here to coordinate.
# -------------------------------------------------------------------

## HOW THIS WORKS
- Windows Claude session: D:\Dropbox\Code\whisperapp (connected to remote Windows box)
- Mac Claude session: ~/Dropbox/Code/whisperapp (open a NEW terminal on Mac, cd there, run `claude`)
- Each session appends status updates below
- Human relays "check COMMS.md and respond" between sessions

---

## [WINDOWS → MAC] Initial instructions for Mac Claude session

**Date:** 2026-05-07
**From:** Windows session

### Your goal
Get WhisperApp installed and running so the tray icon appears in the Mac menu bar.

### Current Mac state
- macOS Apple Silicon (arm64), james@macbookpro
- Working directory: ~/Dropbox/Code/whisperapp
- Python 3.13 at /opt/homebrew/bin/python3.13
- portaudio already installed via brew
- A .venv may or may not exist already

### Steps — run one at a time, log results here after each

**Step 1 — navigate and check venv:**
```zsh
cd ~/Dropbox/Code/whisperapp
ls .venv 2>/dev/null && echo "venv exists" || echo "no venv"
```

**Step 2 — create venv if needed:**
```zsh
/opt/homebrew/bin/python3.13 -m venv .venv
```

**Step 3 — install (takes several minutes):**
```zsh
.venv/bin/pip install -e ".[dev]"
```
Log any errors. Watch for torch, whisperx, mlx-whisper (should auto-install on arm64).

**Step 4 — verify import:**
```zsh
.venv/bin/python -c "from whisperapp.__main__ import main; print('OK')"
```

**Step 5 — launch:**
```zsh
.venv/bin/python -m whisperapp
```
This blocks — menu bar icon should appear. If it crashes, capture the traceback.

### Known things to check
- `rumps` must be installed (pystray needs it on macOS for menu bar)
- `mlx-whisper` installs automatically via platform marker on arm64
- Gradio UI at http://127.0.0.1:7860 once running
- If icon doesn't appear: check `pip list | grep -i rumps`

### After each step append a [MAC -> WINDOWS] entry below.

---

## STATUS LOG

