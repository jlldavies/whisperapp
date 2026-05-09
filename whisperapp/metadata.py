"""Source-file metadata extraction for transcription outputs.

Reads tags + filesystem stats from the original audio file so the rendered
TXT / SRT / VTT / JSON outputs can carry record time, device, duration etc.

Returns a flat `dict[str, str]` of human-friendly key/value pairs. Values are
already stringified — formatters can render them directly without further
processing. Missing fields are simply absent (no None, no empty strings).
"""

from __future__ import annotations
import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


# Tag keys that mutagen exposes for various container formats. The left
# column is what we'd like to surface in the output; the right column is the
# tag key as mutagen reports it for a given format. Where multiple aliases
# exist we try each in order until one yields a value.
#
# Reference:
#   - MP4 (m4a/mp4/mov):  iTunes-style atoms, prefixed with \xa9 byte
#   - ID3 (mp3):          TIT2/TPE1/TDRC/TENC etc.
#   - Vorbis comments:    free-form keys (.ogg, .flac, .opus)
_TAG_ALIASES = {
    "title": (
        "\xa9nam", "TIT2", "title", "TITLE",
    ),
    "artist": (
        "\xa9ART", "TPE1", "artist", "ARTIST",
    ),
    "album": (
        "\xa9alb", "TALB", "album", "ALBUM",
    ),
    "recorded_at": (
        # iTunes / QuickTime
        "\xa9day", "purd",
        # ID3
        "TDRC", "TDRL", "TYER",
        # Vorbis
        "date", "year", "DATE", "YEAR",
    ),
    "device": (
        "\xa9too", "TENC", "encoder", "ENCODER",
        # iTunes-style "make" / "model" atoms (some recorders use these)
        "\xa9mak", "\xa9mod",
        "make", "model", "MAKE", "MODEL",
    ),
    "comment": (
        "\xa9cmt", "COMM", "comment", "COMMENT",
        "description", "DESCRIPTION",
    ),
    "genre": (
        "\xa9gen", "TCON", "genre", "GENRE",
    ),
}


def _flatten_tag_value(value: Any) -> Optional[str]:
    """Mutagen tag values come in many shapes — ID3 frames, lists, MP4
    free-form data, etc. Reduce to a single readable string or None."""
    if value is None:
        return None
    # Lists (most common for both MP4 and Vorbis): take the first element
    if isinstance(value, (list, tuple)) and value:
        return _flatten_tag_value(value[0])
    # ID3 frames have a `.text` attribute that's itself a list
    text = getattr(value, "text", None)
    if text is not None:
        return _flatten_tag_value(text)
    # MP4 atoms can be raw bytes
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip() or None
        except Exception:
            return None
    s = str(value).strip()
    return s or None


def _extract_tags(audio) -> dict[str, str]:
    """Pull our friendly keys from a mutagen audio object's tags."""
    out: dict[str, str] = {}
    tags = getattr(audio, "tags", None)
    if not tags:
        return out
    for friendly, aliases in _TAG_ALIASES.items():
        for key in aliases:
            try:
                raw = tags.get(key) if hasattr(tags, "get") else None
            except Exception:
                raw = None
            if raw is None and key in getattr(tags, "keys", lambda: [])():
                raw = tags[key]
            value = _flatten_tag_value(raw)
            if value:
                out[friendly] = value
                break
    return out


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def extract(file_path: str | Path) -> dict[str, str]:
    """Return a flat dict of source-file metadata. Never raises."""
    path = Path(file_path)
    out: dict[str, str] = {"source_file": path.name}

    # Filesystem stats (always available)
    try:
        st = path.stat()
        out["source_path"] = str(path)
        out["file_size_bytes"] = str(st.st_size)
        out["file_modified"] = _iso(st.st_mtime)
        # Windows ctime is creation; POSIX ctime is metadata-change. Either is
        # a useful fallback for "when was this captured" — label honestly.
        out["file_ctime"] = _iso(st.st_ctime)
    except Exception as e:
        log.debug("stat failed for %s: %s", path, e)

    # Tag-level data via mutagen — best-effort, never blocks
    try:
        from mutagen import File as MutagenFile  # local import keeps cold start fast
        audio = MutagenFile(str(path))
        if audio is not None:
            out.update(_extract_tags(audio))
            info = getattr(audio, "info", None)
            length = getattr(info, "length", None)
            if isinstance(length, (int, float)) and length > 0:
                out["duration"] = _format_duration(length)
                out["duration_seconds"] = f"{length:.2f}"
            for attr in ("channels", "sample_rate", "bitrate"):
                v = getattr(info, attr, None)
                if v:
                    out[attr] = str(v)
    except Exception as e:
        log.debug("mutagen failed for %s: %s", path, e)

    return out


def render_text_block(meta: dict[str, str], comment_prefix: str = "") -> str:
    """Render a metadata block for human-readable formats (txt/srt/vtt).

    `comment_prefix` is prepended to each line so callers can wrap the block
    in format-specific comment syntax (e.g. `NOTE ` for VTT, `# ` for txt).
    """
    if not meta:
        return ""
    pretty = {
        "source_file":      "Source",
        "source_path":      "Path",
        "recorded_at":      "Recorded",
        "file_modified":    "File modified",
        "file_ctime":       "File created",
        "duration":         "Duration",
        "title":            "Title",
        "artist":           "Artist",
        "album":            "Album",
        "device":           "Device / encoder",
        "comment":          "Comment",
        "genre":            "Genre",
        "channels":         "Channels",
        "sample_rate":      "Sample rate (Hz)",
        "bitrate":          "Bitrate (bps)",
        "file_size_bytes":  "File size (bytes)",
        "duration_seconds": "Duration (s)",
    }
    # Render in a stable, readable order; unknown extra keys at the end.
    order = list(pretty.keys())
    keys = [k for k in order if k in meta] + [k for k in meta if k not in pretty]
    lines = [f"{comment_prefix}{pretty.get(k, k)}: {meta[k]}" for k in keys]
    return "\n".join(lines)
