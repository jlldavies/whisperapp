import pytest
from pathlib import Path
from whisperapp.formatters import write_formats, result_to_txt, result_to_srt, result_to_md

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
                  str(tmp_path), ["txt", "srt", "vtt", "json", "tsv", "md"])
    for ext in ["txt", "srt", "vtt", "json", "tsv", "md"]:
        assert (tmp_path / f"audio.{ext}").exists()

def test_md_groups_consecutive_segments_by_speaker():
    result = {"segments": [
        {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "James"},
        {"start": 2.0, "end": 4.0, "text": "there.", "speaker": "James"},
        {"start": 65.0, "end": 67.0, "text": "Hi James.", "speaker": "Zoe"},
    ]}
    md = result_to_md(result)
    assert md.startswith("# Transcript")
    assert "**James** [00:00:00]\n\nHello there." in md
    assert "**Zoe** [00:01:05]\n\nHi James." in md
    assert md.count("**James**") == 1

def test_md_without_speakers_timestamps_each_segment():
    result = {"segments": [
        {"start": 3.0, "end": 5.0, "text": "First line"},
        {"start": 3725.0, "end": 3727.0, "text": "Much later"},
        {"start": 3728.0, "end": 3729.0, "text": "  "},
    ]}
    md = result_to_md(result)
    assert "**[00:00:03]** First line" in md
    assert "**[01:02:05]** Much later" in md
    assert md.count("**[") == 2

def test_md_puts_markers_inline_without_breaking_paragraphs():
    result = {"segments": [
        {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "Zoe"},
        {"start": 0.0, "end": 2.0, "text": "[Animated]", "is_acoustic_marker": True},
        {"start": 0.0, "end": 2.0, "text": "[Animated]", "is_acoustic_marker": True},
        {"start": 2.0, "end": 4.0, "text": "James.", "speaker": "Zoe"},
        {"start": 4.0, "end": 4.0, "text": "[pause 5s]", "is_pause_marker": True},
        {"start": 9.0, "end": 10.0, "text": "Hi.", "speaker": "James"},
    ]}
    md = result_to_md(result)
    assert "**Zoe** [00:00:00]\n\nHello _[Animated]_ James. _[pause 5s]_" in md
    assert md.count("**Zoe**") == 1
    assert "**[" not in md

def test_md_includes_metadata_as_list():
    result = {**SAMPLE_RESULT, "source_metadata": {"source_file": "live.wav"}}
    md = result_to_md(result)
    assert md.startswith("# Transcript — live.wav")
    assert "- " in md.split("**James**")[0]

def test_write_formats_returns_paths_and_honours_stem(tmp_path):
    paths = write_formats(SAMPLE_RESULT, "/fake/live_1.wav",
                          str(tmp_path), ["txt", "md"], stem="live_1_polished")
    assert [Path(p).name for p in paths] == ["live_1_polished.txt", "live_1_polished.md"]
    assert all(Path(p).exists() for p in paths)
