import pytest
from pathlib import Path
from whisperapp.formatters import write_formats, result_to_txt, result_to_srt

SAMPLE_RESULT = {
    "segments": [
        {"start": 0.0, "end": 5.0, "text": "Hello world",
         "speaker": "James"},
        {"start": 5.5, "end": 10.0, "text": "How are you",
         "speaker": "Wolfgang"},
    ]
}

def test_txt_output():
    txt = result_to_txt(SAMPLE_RESULT)
    assert "James" in txt
    assert "Hello world" in txt
    assert "Wolfgang" in txt

def test_srt_output():
    srt = result_to_srt(SAMPLE_RESULT)
    assert "00:00:00,000" in srt
    assert "Hello world" in srt
    assert "1\n" in srt
    assert "2\n" in srt

def test_write_formats_creates_files(tmp_path):
    write_formats(SAMPLE_RESULT, "/fake/audio.mp3",
                  str(tmp_path), ["txt", "srt"])
    assert (tmp_path / "audio.txt").exists()
    assert (tmp_path / "audio.srt").exists()

def test_write_formats_all(tmp_path):
    write_formats(SAMPLE_RESULT, "/fake/audio.mp3",
                  str(tmp_path), ["txt", "srt", "vtt", "json", "tsv"])
    for ext in ["txt", "srt", "vtt", "json", "tsv"]:
        assert (tmp_path / f"audio.{ext}").exists()
