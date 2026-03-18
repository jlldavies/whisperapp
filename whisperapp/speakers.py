import copy
from collections import defaultdict
from whisperapp.sanitise import sanitise_speaker_name

def extract_speaker_snippets(result: dict, n: int = 3) -> dict:
    """Return up to n earliest text snippets per speaker."""
    snippets = defaultdict(list)
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if text and len(snippets[speaker]) < n:
            snippets[speaker].append(text)
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
