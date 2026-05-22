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
    assert cfg.vad_silence_threshold == 0.4
    assert cfg.streaming_max_chunk_sec == 5.0
    assert cfg.streaming_min_chunk_sec == 0.3
    assert cfg.streaming_show_listening is True

def test_pause_detection_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    cfg = Config()
    assert cfg.pause_detection is True
    assert cfg.pause_threshold == 3
    assert cfg.long_pause_threshold == 10
    assert cfg.gap_threshold == 30

def test_acoustic_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    cfg = Config()
    assert cfg.acoustic_enabled is False
    assert cfg.acoustic_volume_enabled is True
    assert cfg.acoustic_volume_threshold == 1.8
    assert cfg.acoustic_pitch_enabled is True
    assert cfg.acoustic_pitch_threshold == 40.0
    assert cfg.acoustic_rate_enabled is True
    assert cfg.acoustic_rate_fast_wpm == 180
    assert cfg.acoustic_rate_slow_wpm == 90


def test_emotion_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    cfg = Config()
    assert cfg.emotion_enabled is False
    assert cfg.emotion_model_ids == []
    assert cfg.emotion_confidence_threshold == 0.65
    assert cfg.emotion_combine_with_ai is False


def test_emotion_model_ids_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    cfg = Config()
    cfg.emotion_model_ids = ["speechbrain-iemocap", "wavlm-multi"]
    cfg.save()
    cfg2 = Config()
    assert cfg2.emotion_model_ids == ["speechbrain-iemocap", "wavlm-multi"]


def test_acoustic_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    cfg = Config()
    cfg.acoustic_enabled = True
    cfg.acoustic_volume_threshold = 2.5
    cfg.acoustic_rate_fast_wpm = 200
    cfg.save()
    cfg2 = Config()
    assert cfg2.acoustic_enabled is True
    assert cfg2.acoustic_volume_threshold == 2.5
    assert cfg2.acoustic_rate_fast_wpm == 200


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
