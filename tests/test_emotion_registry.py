"""Tests for emotion model registry and ModelManager (whisperapp.emotion_registry)."""
import pytest
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from whisperapp.emotion_registry import (
    EMOTION_MODELS,
    ModelManager,
    _REGISTRY_BY_ID,
)


# ---------------------------------------------------------------------------
# Registry constants
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_has_four_models(self):
        assert len(EMOTION_MODELS) == 4

    def test_all_models_have_required_keys(self):
        required = {"id", "name", "hf_repo", "size_mb", "labels", "type", "description", "recommended"}
        for m in EMOTION_MODELS:
            assert required.issubset(m.keys()), f"Model {m.get('id')} missing keys"

    def test_registry_by_id_matches(self):
        for m in EMOTION_MODELS:
            assert m["id"] in _REGISTRY_BY_ID
            assert _REGISTRY_BY_ID[m["id"]] is m

    def test_speechbrain_iemocap_is_recommended(self):
        m = _REGISTRY_BY_ID["speechbrain-iemocap"]
        assert m["recommended"] is True

    def test_other_models_not_recommended(self):
        for m in EMOTION_MODELS:
            if m["id"] != "speechbrain-iemocap":
                assert m["recommended"] is False

    def test_audeering_is_dimensional(self):
        assert _REGISTRY_BY_ID["audeering-dimensional"]["type"] == "dimensional"

    def test_classification_models(self):
        for mid in ("speechbrain-iemocap", "wavlm-multi", "wav2vec2-8emotion"):
            assert _REGISTRY_BY_ID[mid]["type"] == "classification"


# ---------------------------------------------------------------------------
# ModelManager — list_all / get_status
# ---------------------------------------------------------------------------

class TestModelManagerStatus:
    def test_list_all_returns_four_entries(self, tmp_path):
        mgr = ModelManager(models_dir=tmp_path)
        entries = mgr.list_all()
        assert len(entries) == 4

    def test_list_all_includes_status_keys(self, tmp_path):
        mgr = ModelManager(models_dir=tmp_path)
        for entry in mgr.list_all():
            assert "downloaded" in entry
            assert "downloading" in entry
            assert "progress" in entry
            assert "error" in entry

    def test_get_status_not_downloaded(self, tmp_path):
        mgr = ModelManager(models_dir=tmp_path)
        status = mgr.get_status("speechbrain-iemocap")
        assert status["downloaded"] is False
        assert status["downloading"] is False
        assert status["progress"] == 0

    def test_get_status_downloaded_when_dir_exists(self, tmp_path):
        model_dir = tmp_path / "speechbrain-iemocap"
        model_dir.mkdir()
        mgr = ModelManager(models_dir=tmp_path)
        status = mgr.get_status("speechbrain-iemocap")
        assert status["downloaded"] is True

    def test_get_status_unknown_model(self, tmp_path):
        mgr = ModelManager(models_dir=tmp_path)
        status = mgr.get_status("nonexistent-model")
        assert status["downloaded"] is False
        assert status["error"] is not None

    def test_list_all_shows_downloaded_when_dir_exists(self, tmp_path):
        model_dir = tmp_path / "wavlm-multi"
        model_dir.mkdir()
        mgr = ModelManager(models_dir=tmp_path)
        entries = mgr.list_all()
        wavlm = next(e for e in entries if e["id"] == "wavlm-multi")
        assert wavlm["downloaded"] is True

        others = [e for e in entries if e["id"] != "wavlm-multi"]
        for e in others:
            assert e["downloaded"] is False


# ---------------------------------------------------------------------------
# ModelManager — delete
# ---------------------------------------------------------------------------

class TestModelManagerDelete:
    def test_delete_is_noop_if_not_downloaded(self, tmp_path):
        mgr = ModelManager(models_dir=tmp_path)
        # Should not raise
        mgr.delete("speechbrain-iemocap")
        assert not (tmp_path / "speechbrain-iemocap").exists()

    def test_delete_removes_directory(self, tmp_path):
        model_dir = tmp_path / "speechbrain-iemocap"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"fake data")
        mgr = ModelManager(models_dir=tmp_path)
        mgr.delete("speechbrain-iemocap")
        assert not model_dir.exists()

    def test_delete_updates_status(self, tmp_path):
        model_dir = tmp_path / "speechbrain-iemocap"
        model_dir.mkdir()
        mgr = ModelManager(models_dir=tmp_path)
        assert mgr.get_status("speechbrain-iemocap")["downloaded"] is True
        mgr.delete("speechbrain-iemocap")
        assert mgr.get_status("speechbrain-iemocap")["downloaded"] is False

    def test_delete_raises_for_unknown_model(self, tmp_path):
        mgr = ModelManager(models_dir=tmp_path)
        with pytest.raises(ValueError):
            mgr.delete("unknown-model-xyz")


# ---------------------------------------------------------------------------
# ModelManager — download (mocked HuggingFace)
# ---------------------------------------------------------------------------

class TestModelManagerDownload:
    def test_download_starts_background_thread(self, tmp_path):
        """download() immediately sets downloading=True in state."""
        mgr = ModelManager(models_dir=tmp_path)

        proceed = threading.Event()

        def fake_thread(model_id):
            proceed.wait(timeout=5)
            # Simulate completion
            (tmp_path / model_id).mkdir(parents=True, exist_ok=True)
            with mgr._lock:
                mgr._download_state[model_id] = {
                    "downloading": False, "progress": 100, "error": None
                }

        mgr._download_thread = fake_thread
        mgr.download("speechbrain-iemocap")

        # download() sets state synchronously before starting the thread
        status = mgr.get_status("speechbrain-iemocap")
        assert status["downloading"] is True

        proceed.set()
        time.sleep(0.2)
        status2 = mgr.get_status("speechbrain-iemocap")
        assert status2["downloaded"] is True
        assert status2["progress"] == 100

    def test_download_sets_downloading_state(self, tmp_path):
        mgr = ModelManager(models_dir=tmp_path)

        ready = threading.Event()
        started = threading.Event()

        def fake_download(repo_id, local_dir, local_dir_use_symlinks=True):
            started.set()
            ready.wait(timeout=5)
            Path(local_dir).mkdir(parents=True, exist_ok=True)

        import whisperapp.emotion_registry as reg
        with patch.object(reg, "snapshot_download", fake_download, create=True):
            # Monkey-patch the module's snapshot_download reference
            original_fn = None
            try:
                import huggingface_hub
                original_fn = huggingface_hub.snapshot_download
                reg_thread = None

                # Patch inside the _download_thread
                original_thread = mgr._download_thread

                def patched_thread(model_id):
                    hf_repo = _REGISTRY_BY_ID[model_id]["hf_repo"]
                    model_dir = mgr._models_dir / model_id
                    try:
                        fake_download(hf_repo, str(model_dir))
                        with mgr._lock:
                            mgr._download_state[model_id] = {
                                "downloading": False, "progress": 100, "error": None
                            }
                    except Exception as exc:
                        import shutil
                        shutil.rmtree(str(model_dir), ignore_errors=True)
                        with mgr._lock:
                            mgr._download_state[model_id] = {
                                "downloading": False, "progress": 0, "error": str(exc)
                            }

                mgr._download_thread = patched_thread
                mgr.download("speechbrain-iemocap")

                started.wait(timeout=3)
                status = mgr.get_status("speechbrain-iemocap")
                # While download is running, should be marked downloading
                assert status["downloading"] is True

                ready.set()
                time.sleep(0.2)  # Let thread finish
                status2 = mgr.get_status("speechbrain-iemocap")
                assert status2["progress"] == 100

            finally:
                ready.set()  # Unblock in case of error

    def test_download_noop_if_already_downloaded(self, tmp_path):
        model_dir = tmp_path / "speechbrain-iemocap"
        model_dir.mkdir()
        mgr = ModelManager(models_dir=tmp_path)

        call_count = {"n": 0}

        def fake_download(*args, **kwargs):
            call_count["n"] += 1

        import whisperapp.emotion_registry as reg
        original_thread = mgr._download_thread
        mgr._download_thread = lambda mid: fake_download(mid)

        mgr.download("speechbrain-iemocap")
        time.sleep(0.1)
        assert call_count["n"] == 0  # Should not start download

    def test_download_raises_for_unknown_model(self, tmp_path):
        mgr = ModelManager(models_dir=tmp_path)
        with pytest.raises(ValueError):
            mgr.download("nonexistent-model")

    def test_progress_tracking(self, tmp_path):
        """Progress dict is updated correctly during download."""
        mgr = ModelManager(models_dir=tmp_path)

        with mgr._lock:
            mgr._download_state["speechbrain-iemocap"] = {
                "downloading": True, "progress": 45, "error": None
            }

        status = mgr.get_status("speechbrain-iemocap")
        assert status["downloading"] is True
        assert status["progress"] == 45

    def test_download_error_updates_state(self, tmp_path):
        """If download fails, state reflects the error."""
        mgr = ModelManager(models_dir=tmp_path)

        def fake_failing_thread(model_id):
            with mgr._lock:
                mgr._download_state[model_id] = {
                    "downloading": False, "progress": 0, "error": "Network error"
                }

        mgr._download_thread = fake_failing_thread
        mgr.download("speechbrain-iemocap")
        time.sleep(0.1)

        status = mgr.get_status("speechbrain-iemocap")
        assert status["downloaded"] is False
        assert status["error"] == "Network error"
