"""Tests for DOCX and PDF output formatters."""
import pytest
from pathlib import Path
from whisperapp.document_formats import build_context, substitute, generate_docx_template


SAMPLE_RESULT = {
    "language": "en",
    "model": "large-v2",
    "segments": [
        {"start": 0.0, "end": 3.2, "text": " Hello world.", "speaker": "SPEAKER_00"},
        {"start": 3.5, "end": 3.5, "text": "[Pause, 4 seconds]", "is_pause_marker": True},
        {"start": 7.8, "end": 12.1, "text": " How are you today?", "speaker": "SPEAKER_01"},
    ],
    "source_metadata": {
        "source_file": "interview.mp3",
        "duration": "00:12",
    },
}


def test_build_context_basic():
    ctx = build_context(SAMPLE_RESULT, "/tmp/interview.mp3")
    assert ctx["filename"] == "interview.mp3"
    assert ctx["model"] == "large-v2"
    assert ctx["language"] == "en"
    assert ctx["duration"] == "00:12"
    assert "Hello world." in ctx["transcript"]
    assert "[Pause, 4 seconds]" in ctx["transcript"]
    assert "SPEAKER_00" in ctx["speakers"]
    assert "SPEAKER_01" in ctx["speakers"]


def test_build_context_diarized():
    ctx = build_context(SAMPLE_RESULT, "/tmp/interview.mp3")
    assert "[SPEAKER_00]: Hello world." in ctx["diarized_transcript"]
    assert "[SPEAKER_01]: How are you today?" in ctx["diarized_transcript"]
    assert "[Pause, 4 seconds]" in ctx["diarized_transcript"]


def test_build_context_word_count():
    ctx = build_context(SAMPLE_RESULT, "/tmp/interview.mp3")
    # "Hello world." = 2 words, "How are you today?" = 4 words
    assert ctx["word_count"] == "6"


def test_build_context_segments_timestamped():
    ctx = build_context(SAMPLE_RESULT, "/tmp/interview.mp3")
    assert "→" in ctx["segments"]
    assert "SPEAKER_00" in ctx["segments"]


def test_substitute_basic():
    assert substitute("Hello {{name}}!", {"name": "World"}) == "Hello World!"


def test_substitute_unknown_marker_preserved():
    result = substitute("{{known}} and {{unknown}}", {"known": "yes"})
    assert result == "yes and {{unknown}}"


def test_substitute_multiple():
    ctx = {"a": "1", "b": "2"}
    assert substitute("{{a}}-{{b}}", ctx) == "1-2"


def test_write_docx(tmp_path):
    pytest.importorskip("docx", reason="python-docx not installed")
    from whisperapp.document_formats import write_docx
    out = tmp_path / "output.docx"
    write_docx(SAMPLE_RESULT, "/tmp/interview.mp3", out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_generate_docx_template(tmp_path):
    pytest.importorskip("docx", reason="python-docx not installed")
    dest = tmp_path / "template.docx"
    ok = generate_docx_template(dest)
    assert ok
    assert dest.exists()
    assert dest.stat().st_size > 1000


def test_docx_with_template(tmp_path):
    pytest.importorskip("docx", reason="python-docx not installed")
    from docx import Document
    from whisperapp.document_formats import write_docx

    # Create a minimal template with markers
    tmpl = tmp_path / "tmpl.docx"
    doc = Document()
    doc.add_paragraph("File: {{filename}}")
    doc.add_paragraph("Date: {{date}}")
    doc.add_paragraph("{{diarized_transcript}}")
    doc.save(str(tmpl))

    class FakeCfg:
        output_template_docx = str(tmpl)
        output_template_pdf = ""

    out = tmp_path / "output.docx"
    write_docx(SAMPLE_RESULT, "/tmp/interview.mp3", out, FakeCfg())
    assert out.exists()

    doc2 = Document(str(out))
    full_text = "\n".join(p.text for p in doc2.paragraphs)
    assert "interview.mp3" in full_text
    assert "SPEAKER_00" in full_text


def test_write_docx_formats_integration(tmp_path):
    """write_formats dispatches to write_docx correctly."""
    pytest.importorskip("docx", reason="python-docx not installed")
    from whisperapp.formatters import write_formats

    write_formats(SAMPLE_RESULT, "/tmp/interview.mp3", str(tmp_path), ["docx"])
    assert (tmp_path / "interview.docx").exists()
