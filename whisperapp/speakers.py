import copy
from collections import defaultdict
from whisperapp.sanitise import sanitise_speaker_name

def extract_speaker_snippets(result: dict, n: int = 3) -> dict:
    """Return up to n earliest snippets per speaker, each with text + timing.

    Each snippet is `{"text": str, "start": float, "end": float}` so the UI
    can play just the matching slice of the source audio file.
    """
    snippets = defaultdict(list)
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if text and len(snippets[speaker]) < n:
            snippets[speaker].append({
                "text": text,
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", 0.0) or 0.0),
            })
    return dict(snippets)

def apply_speaker_names(result: dict, names: dict) -> dict:
    """Return a new result dict with speaker labels replaced by names.
    Empty names are left as-is (original SPEAKER_XX label kept)."""
    result = copy.deepcopy(result)
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "")
        if speaker in names:
            clean = sanitise_speaker_name(names[speaker])
            if clean:
                seg["speaker"] = clean
    return result
