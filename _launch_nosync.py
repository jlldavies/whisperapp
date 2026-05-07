"""Launch bootstrap that skips the updater hang and runs the app.

Used as the target of pythonw.exe / python.exe so we avoid cross-shell
quoting issues with `python -c`.
"""
import whisperapp.updater as _u
_u.run_update = lambda: None
from whisperapp.__main__ import main
main()
