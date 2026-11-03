import json
from pathlib import Path
from typing import Optional

def _format_time_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _format_time_vtt(seconds: float) -> str:
    return _format_time_srt(seconds).replace(",", ".")

def result_to_txt(result: dict) -> str:
    lines = []
    meta = result.get("source_metadata") or {}
    if meta:
        from whisperapp.metadata import render_text_block
        lines.append("# WhisperApp transcript")
        lines.append(render_text_block(meta, comment_prefix="# "))
        lines.append("")  # blank line before transcript
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        prefix = f"[{speaker}] " if speaker else ""
        lines.append(f"{prefix}{text}")
    return "\n".join(lines)

def result_to_srt(result: dict) -> str:
    blocks = []
    meta = result.get("source_metadata") or {}
    if meta:
        # SRT has no comment syntax, but most players ignore content before
        # the first cue index. We emit a single leading "0" pseudo-cue as a
        # NOTE block — clearly outside the timed range so it's harmless.
        from whisperapp.metadata import render_text_block
        blocks.append(
            "0\n00:00:00,000 --> 00:00:00,000\n"
            + render_text_block(meta) + "\n"
        )
    for i, seg in enumerate(result.get("segments", []), 1):
        start = _format_time_srt(seg["start"])
        end = _format_time_srt(seg["end"])
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        prefix = f"[{speaker}] " if speaker else ""
        blocks.append(f"{i}\n{start} --> {end}\n{prefix}{text}\n")
    return "\n".join(blocks)

def result_to_vtt(result: dict) -> str:
    lines = ["WEBVTT\n"]
    meta = result.get("source_metadata") or {}
    if meta:
        # WebVTT's NOTE block is the right place for free-form metadata.
        from whisperapp.metadata import render_text_block
        lines.append("NOTE")
        lines.append(render_text_block(meta))
        lines.append("")
    for i, seg in enumerate(result.get("segments", []), 1):
        start = _format_time_vtt(seg["start"])
        end = _format_time_vtt(seg["end"])
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        prefix = f"<v {speaker}>" if speaker else ""
        lines.append(f"{start} --> {end}\n{prefix}{text}\n")
    return "\n".join(lines)

def result_to_tsv(result: dict) -> str:
    rows = []
    meta = result.get("source_metadata") or {}
    if meta:
        # TSV has no formal comment syntax but a leading `#` line is a common
        # convention that pandas/csv readers skip with `comment='#'`.
        from whisperapp.metadata import render_text_block
        rows.append(render_text_block(meta, comment_prefix="# "))
    rows.append("start\tend\tspeaker\ttext")
    for seg in result.get("segments", []):
        rows.append(f"{seg['start']}\t{seg['end']}\t"
                    f"{seg.get('speaker','')}\t{seg.get('text','').strip()}")
    return "\n".join(rows)

def _format_time_md(seconds: float) -> str:
    return _format_time_srt(seconds).split(",")[0]

def _is_marker(seg: dict) -> bool:
    return bool(seg.get("is_acoustic_marker") or seg.get("is_pause_marker")
                or seg.get("is_emotion_marker"))

def result_to_md(result: dict) -> str:
    """Readable Markdown: consecutive segments from one speaker become a
    single paragraph under a bold speaker + timestamp line. Without speaker
    labels (e.g. a live transcript) each segment is its own timestamped line.
    Acoustic / emotion / pause markers are speaker-less pseudo-segments that
    follow the line they describe, so they go inline in italics rather than
    breaking the paragraph."""
    meta = result.get("source_metadata") or {}
    title = meta.get("source_file", "")
    lines = [f"# Transcript{' — ' + title if title else ''}", ""]
    if meta:
        from whisperapp.metadata import render_text_block
        lines.append(render_text_block(meta, comment_prefix="- "))
        lines.append("")

    current_speaker = None
    paragraph: list[str] = []

    def _flush():
        if paragraph:
            lines.append(" ".join(paragraph))
            lines.append("")
            paragraph.clear()

    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        if _is_marker(seg):
            marker = f"_{text}_"
            if not paragraph or paragraph[-1] != marker:
                paragraph.append(marker)
            continue
        stamp = _format_time_md(seg.get("start", 0.0))
        speaker = seg.get("speaker", "")
        if not speaker:
            _flush()
            current_speaker = None
            lines.append(f"**[{stamp}]** {text}")
            lines.append("")
            continue
        if speaker != current_speaker:
            _flush()
            current_speaker = speaker
            lines.append(f"**{speaker}** [{stamp}]")
            lines.append("")
        paragraph.append(text)
    _flush()
    return "\n".join(lines).rstrip() + "\n"

FORMATTERS = {
    "txt": result_to_txt,
    "srt": result_to_srt,
    "vtt": result_to_vtt,
    "json": lambda r: json.dumps(r, ensure_ascii=False, indent=2),
    "tsv": result_to_tsv,
    "md": result_to_md,
}

def write_formats(result: dict, source_file: str,
                  output_path: str, formats: list,
                  config: Optional[object] = None,
                  stem: Optional[str] = None) -> list:
    """Write *result* in each of *formats*; return the paths written.

    Files are named after *source_file* unless *stem* is given."""
    stem = stem or Path(source_file).stem
    written = []
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Attach source-file metadata so every formatter can include it. Done
    # once on a shallow copy so the original `result` isn't mutated.
    if "source_metadata" not in result:
        try:
            from whisperapp.metadata import extract as _extract_meta
            result = {**result, "source_metadata": _extract_meta(source_file)}
        except Exception:
            pass

    # Formats that show human-readable text — pause markers are inserted here
    HUMAN_FORMATS = {"txt", "srt", "vtt", "md", "docx", "pdf"}

    marked_result = None  # lazily computed

    for fmt in formats:
        out_file = out_dir / f"{stem}.{fmt}"

        if fmt in ("docx", "pdf"):
            # Use marked result (with pause markers) for rich formats too
            if marked_result is None and config is not None:
                from whisperapp.pauses import insert_pause_markers
                marked_result = insert_pause_markers(result, config)
                if "source_metadata" not in marked_result and "source_metadata" in result:
                    marked_result["source_metadata"] = result["source_metadata"]
            rich = marked_result if marked_result is not None else result
            from whisperapp.document_formats import write_docx, write_pdf
            if fmt == "docx":
                write_docx(rich, source_file, out_file, config)
            else:
                write_pdf(rich, source_file, out_file, config)
            written.append(str(out_file))
            continue

        if fmt not in FORMATTERS:
            continue
        if config is not None and fmt in HUMAN_FORMATS:
            if marked_result is None:
                from whisperapp.pauses import insert_pause_markers
                marked_result = insert_pause_markers(result, config)
                if "source_metadata" not in marked_result and "source_metadata" in result:
                    marked_result["source_metadata"] = result["source_metadata"]
            content = FORMATTERS[fmt](marked_result)
        else:
            content = FORMATTERS[fmt](result)
        out_file.write_text(content, encoding="utf-8")
        written.append(str(out_file))
    return written
