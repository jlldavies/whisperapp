import json
from pathlib import Path

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
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        prefix = f"[{speaker}] " if speaker else ""
        lines.append(f"{prefix}{text}")
    return "\n".join(lines)

def result_to_srt(result: dict) -> str:
    blocks = []
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
    for i, seg in enumerate(result.get("segments", []), 1):
        start = _format_time_vtt(seg["start"])
        end = _format_time_vtt(seg["end"])
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        prefix = f"<v {speaker}>" if speaker else ""
        lines.append(f"{start} --> {end}\n{prefix}{text}\n")
    return "\n".join(lines)

def result_to_tsv(result: dict) -> str:
    rows = ["start\tend\tspeaker\ttext"]
    for seg in result.get("segments", []):
        rows.append(f"{seg['start']}\t{seg['end']}\t"
                    f"{seg.get('speaker','')}\t{seg.get('text','').strip()}")
    return "\n".join(rows)

FORMATTERS = {
    "txt": result_to_txt,
    "srt": result_to_srt,
    "vtt": result_to_vtt,
    "json": lambda r: json.dumps(r, ensure_ascii=False, indent=2),
    "tsv": result_to_tsv,
}

def write_formats(result: dict, source_file: str,
                  output_path: str, formats: list):
    stem = Path(source_file).stem
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        if fmt not in FORMATTERS:
            continue
        content = FORMATTERS[fmt](result)
        (out_dir / f"{stem}.{fmt}").write_text(content, encoding="utf-8")
