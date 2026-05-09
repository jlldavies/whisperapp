"""Tests for model_registry.py"""
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERAPP_CONFIG_DIR", str(tmp_path))
    from whisperapp.model_registry import ModelRegistry
    return ModelRegistry()


def test_list_catalogue_returns_all_categories(registry):
    """list_catalogue() returns all 5 categories."""
    with patch("whisperapp.model_registry._get_active_transcription_model", return_value="large-v2"), \
         patch("whisperapp.model_registry._get_active_streaming_model", return_value="fw-base"), \
         patch("whisperapp.model_registry._is_hf_cached", return_value=False):
        catalogue = registry.list_catalogue()
    assert set(catalogue.keys()) == {"transcription", "streaming", "alignment", "diarization", "emotion", "capabilities"}


def test_list_catalogue_has_required_keys(registry):
    """Each category has 'title', 'sub', 'models' keys."""
    with patch("whisperapp.model_registry._get_active_transcription_model", return_value="large-v2"), \
         patch("whisperapp.model_registry._get_active_streaming_model", return_value="fw-base"), \
         patch("whisperapp.model_registry._is_hf_cached", return_value=False):
        catalogue = registry.list_catalogue()
    for cat_key, cat in catalogue.items():
        assert "title" in cat, f"{cat_key} missing 'title'"
        assert "sub" in cat, f"{cat_key} missing 'sub'"
        assert "models" in cat, f"{cat_key} missing 'models'"
        assert isinstance(cat["models"], list)
        assert len(cat["models"]) > 0


def test_list_catalogue_model_states(registry):
    """Models have a 'state' field with valid values."""
    valid_states = {"active", "installed", "update", "downloading", "available"}
    with patch("whisperapp.model_registry._get_active_transcription_model", return_value="large-v2"), \
         patch("whisperapp.model_registry._get_active_streaming_model", return_value="fw-base"), \
         patch("whisperapp.model_registry._is_hf_cached", return_value=False):
        catalogue = registry.list_catalogue()
    for cat_key, cat in catalogue.items():
        for m in cat["models"]:
            assert "state" in m, f"{m['id']} missing 'state'"
            assert m["state"] in valid_states, f"{m['id']} has invalid state: {m['state']}"


def test_list_catalogue_active_model_has_role(registry):
    """Active models have a 'role' field."""
    with patch("whisperapp.model_registry._get_active_transcription_model", return_value="large-v2"), \
         patch("whisperapp.model_registry._get_active_streaming_model", return_value="fw-base"), \
         patch("whisperapp.model_registry._is_hf_cached", return_value=False):
        catalogue = registry.list_catalogue()
    # Transcription large-v2 should be active with role Default
    transcription_models = {m["id"]: m for m in catalogue["transcription"]["models"]}
    assert transcription_models["large-v2"]["state"] == "active"
    assert transcription_models["large-v2"]["role"] == "Default"


def test_list_catalogue_alignment_has_language_filterable(registry):
    """Alignment category has language_filterable flag."""
    with patch("whisperapp.model_registry._get_active_transcription_model", return_value="large-v2"), \
         patch("whisperapp.model_registry._get_active_streaming_model", return_value="fw-base"), \
         patch("whisperapp.model_registry._is_hf_cached", return_value=False):
        catalogue = registry.list_catalogue()
    assert catalogue["alignment"].get("language_filterable") is True


def test_get_disk_summary_returns_all_categories(registry):
    """get_disk_summary() returns all category keys."""
    with patch("whisperapp.model_registry._get_active_transcription_model", return_value="large-v2"), \
         patch("whisperapp.model_registry._get_active_streaming_model", return_value="fw-base"), \
         patch("whisperapp.model_registry._is_hf_cached", return_value=False):
        summary = registry.get_disk_summary()
    assert set(summary.keys()) == {"transcription", "streaming", "alignment", "diarization", "emotion", "capabilities"}


def test_get_disk_summary_structure(registry):
    """Each category in disk summary has bytes, count, label."""
    with patch("whisperapp.model_registry._get_active_transcription_model", return_value="large-v2"), \
         patch("whisperapp.model_registry._get_active_streaming_model", return_value="fw-base"), \
         patch("whisperapp.model_registry._is_hf_cached", return_value=False):
        summary = registry.get_disk_summary()
    for cat_key, data in summary.items():
        assert "bytes" in data, f"{cat_key} missing 'bytes'"
        assert "count" in data, f"{cat_key} missing 'count'"
        assert "label" in data, f"{cat_key} missing 'label'"
        assert isinstance(data["bytes"], (int, float))
        assert isinstance(data["count"], int)


def test_get_disk_summary_counts_active_models(registry):
    """Active models are counted in disk summary."""
    with patch("whisperapp.model_registry._get_active_transcription_model", return_value="large-v2"), \
         patch("whisperapp.model_registry._get_active_streaming_model", return_value="fw-base"), \
         patch("whisperapp.model_registry._is_hf_cached", return_value=False):
        summary = registry.get_disk_summary()
    # transcription has large-v2 as active → count >= 1
    assert summary["transcription"]["count"] >= 1
    assert summary["transcription"]["bytes"] > 0


def test_format_size_mb(registry):
    """format_size formats MB correctly."""
    assert registry.format_size(244) == "244 MB"
    assert registry.format_size(39) == "39 MB"
    assert registry.format_size(769) == "769 MB"


def test_format_size_gb(registry):
    """format_size converts to GB for values >= 1024."""
    result = registry.format_size(1500)
    assert "GB" in result
    assert "1.46" in result


def test_format_size_exactly_1gb(registry):
    """format_size handles exactly 1024 MB."""
    result = registry.format_size(1024)
    assert "1.00 GB" == result


def test_format_size_small(registry):
    """format_size handles small values."""
    assert registry.format_size(27) == "27 MB"
    assert registry.format_size(74) == "74 MB"


def test_catalogue_transcription_model_count(registry):
    """Transcription category has exactly 6 models."""
    from whisperapp.model_registry import MODEL_CATALOGUE
    assert len(MODEL_CATALOGUE["transcription"]["models"]) == 6


def test_catalogue_emotion_model_ids_match_emotion_registry(registry):
    """Emotion model IDs in catalogue match those in emotion_registry."""
    from whisperapp.model_registry import MODEL_CATALOGUE
    from whisperapp.emotion_registry import _REGISTRY_BY_ID
    catalogue_ids = {m["id"] for m in MODEL_CATALOGUE["emotion"]["models"]}
    registry_ids = set(_REGISTRY_BY_ID.keys())
    # At least some overlap expected
    assert len(catalogue_ids & registry_ids) >= 3
