"""Tests for metadata extraction including SHA-256 hashing."""
import hashlib
import pytest
from pathlib import Path
from whisperapp.metadata import extract, _sha256, _format_duration, render_text_block


# ── _sha256 ────────────────────────────────────────────────────────────────

def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "sample.bin"
    f.write_bytes(b"hello world 12345")
    result = _sha256(f)
    expected = hashlib.sha256(b"hello world 12345").hexdigest()
    assert result == expected


def test_sha256_empty_file(tmp_path):
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    result = _sha256(f)
    assert result == hashlib.sha256(b"").hexdigest()


def test_sha256_multi_chunk(tmp_path):
    """File larger than one 64KB read chunk."""
    data = b"x" * (65536 * 3 + 17)
    f = tmp_path / "large.bin"
    f.write_bytes(data)
    result = _sha256(f)
    assert result == hashlib.sha256(data).hexdigest()


def test_sha256_missing_file_returns_none():
    result = _sha256(Path("/nonexistent/path/file.mp3"))
    assert result is None


def test_sha256_is_64_hex_chars(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"anything")
    result = _sha256(f)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


# ── extract() integration ──────────────────────────────────────────────────

def test_extract_includes_source_hash(tmp_path):
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"fake audio bytes")
    meta = extract(str(f))
    assert "source_hash" in meta
    assert meta["source_hash"] == hashlib.sha256(b"fake audio bytes").hexdigest()


def test_extract_source_hash_is_deterministic(tmp_path):
    f = tmp_path / "file.mp3"
    f.write_bytes(b"consistent content")
    assert extract(str(f))["source_hash"] == extract(str(f))["source_hash"]


def test_extract_includes_filesystem_fields(tmp_path):
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"data")
    meta = extract(str(f))
    assert meta["source_file"] == "audio.mp3"
    assert "source_path" in meta
    assert "file_size_bytes" in meta
    assert meta["file_size_bytes"] == "4"


def test_extract_missing_file_no_raise():
    meta = extract("/nonexistent/file.mp3")
    assert meta["source_file"] == "file.mp3"
    assert "source_hash" not in meta   # file unreadable → hash absent


# ── render_text_block ──────────────────────────────────────────────────────

def test_render_text_block_includes_sha256():
    meta = {"source_file": "rec.mp3", "source_hash": "abc123"}
    block = render_text_block(meta)
    assert "SHA-256" in block
    assert "abc123" in block


def test_render_text_block_sha256_before_recorded():
    """source_hash should appear near the top of the output, before recorded_at."""
    meta = {
        "source_file": "rec.mp3",
        "source_hash": "deadbeef",
        "recorded_at": "2026-01-01",
    }
    block = render_text_block(meta)
    hash_pos = block.index("deadbeef")
    rec_pos  = block.index("2026-01-01")
    assert hash_pos < rec_pos


def test_render_text_block_empty():
    assert render_text_block({}) == ""


# ── _format_duration ──────────────────────────────────────────────────────

def test_format_duration_seconds_only():
    assert _format_duration(45) == "00:45"


def test_format_duration_minutes():
    assert _format_duration(125) == "02:05"


def test_format_duration_hours():
    assert _format_duration(3661) == "01:01:01"


def test_format_duration_zero():
    assert _format_duration(0) == "00:00"
