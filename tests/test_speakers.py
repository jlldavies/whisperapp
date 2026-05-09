from whisperapp.speakers import extract_speaker_snippets, apply_speaker_names

DIARIZED = {
    "segments": [
        {"start": 0.0, "end": 5.0, "text": "Hello world", "speaker": "SPEAKER_00"},
        {"start": 5.5, "end": 10.0, "text": "How are you", "speaker": "SPEAKER_01"},
        {"start": 10.0, "end": 15.0, "text": "I am fine", "speaker": "SPEAKER_00"},
        {"start": 15.0, "end": 20.0, "text": "Great to hear", "speaker": "SPEAKER_01"},
        {"start": 20.0, "end": 25.0, "text": "Third snippet", "speaker": "SPEAKER_00"},
    ]
}

def test_extracts_snippets_per_speaker():
    snippets = extract_speaker_snippets(DIARIZED, n=3)
    assert "SPEAKER_00" in snippets
    assert "SPEAKER_01" in snippets
    assert len(snippets["SPEAKER_00"]) <= 3
    first = snippets["SPEAKER_00"][0]
    assert first["text"] == "Hello world"
    assert first["start"] == 0.0
    assert first["end"] == 5.0

def test_apply_speaker_names():
    names = {"SPEAKER_00": "James", "SPEAKER_01": "Wolfgang"}
    renamed = apply_speaker_names(DIARIZED, names)
    assert renamed["segments"][0]["speaker"] == "James"
    assert renamed["segments"][1]["speaker"] == "Wolfgang"

def test_apply_empty_name_keeps_original():
    names = {"SPEAKER_00": "", "SPEAKER_01": "Wolfgang"}
    renamed = apply_speaker_names(DIARIZED, names)
    assert renamed["segments"][0]["speaker"] == "SPEAKER_00"
    assert renamed["segments"][1]["speaker"] == "Wolfgang"

def test_apply_names_does_not_mutate_original():
    names = {"SPEAKER_00": "James"}
    original_speaker = DIARIZED["segments"][0]["speaker"]
    apply_speaker_names(DIARIZED, names)
    assert DIARIZED["segments"][0]["speaker"] == original_speaker
