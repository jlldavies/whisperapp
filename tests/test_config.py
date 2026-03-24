import json
from pathlib import Path
import pytest
from whisperapp.config import Config

def test_config_creates_default(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    cfg = Config()
    assert cfg.hf_token == ""
    assert cfg.default_model == "large-v2"
    assert cfg.default_output_path == str(Path.home() / "Downloads")
    assert (tmp_path / "config.json").exists()

def test_config_saves_and_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    cfg = Config()
    cfg.hf_token = "hf_test123"
    cfg.save()
    cfg2 = Config()
    assert cfg2.hf_token == "hf_test123"

def test_streaming_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    cfg = Config()
    assert cfg.streaming_model == "base"
    assert cfg.vad_silence_threshold == 0.6
    assert cfg.streaming_max_chunk_sec == 10.0

def test_streaming_config_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    cfg = Config()
    cfg.streaming_model = "tiny"
    cfg.vad_silence_threshold = 0.8
    cfg.streaming_max_chunk_sec = 15.0
    cfg.save()

    cfg2 = Config()
    assert cfg2.streaming_model == "tiny"
    assert cfg2.vad_silence_threshold == 0.8
    assert cfg2.streaming_max_chunk_sec == 15.0
