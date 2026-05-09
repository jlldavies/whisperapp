"""Emotion model registry and ModelManager for downloading/running inference.

ModelManager handles:
- Listing available models with download status
- Background downloading via huggingface_hub.snapshot_download
- Deleting downloaded models
- Loading models into memory (cached)
- Running inference per-segment
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

EMOTION_MODELS = [
    {
        "id": "speechbrain-iemocap",
        "name": "SpeechBrain IEMOCAP",
        "hf_repo": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        "size_mb": 300,
        "labels": ["angry", "happy", "neutral", "sad"],
        "type": "classification",
        "description": "Fast 4-class model. Best for clear emotional speech. Good starting point.",
        "recommended": True,
    },
    {
        "id": "wavlm-multi",
        "name": "WavLM Multi-Corpus",
        "hf_repo": "3loi/SER-Odyssey-Baseline-WavLM-Multi-Corpus-Emotion-Recognition",
        "size_mb": 400,
        "labels": ["angry", "happy", "neutral", "sad"],
        "type": "classification",
        "description": "4-class model trained across multiple datasets. More robust than IEMOCAP alone.",
        "recommended": False,
    },
    {
        "id": "wav2vec2-8emotion",
        "name": "wav2vec2 8 Emotions",
        "hf_repo": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "size_mb": 1200,
        "labels": ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"],
        "type": "classification",
        "description": "8-class model. Broader range but noisier. Use when you need disgust/fearful/surprised.",
        "recommended": False,
    },
    {
        "id": "audeering-dimensional",
        "name": "Audeering Dimensional",
        "hf_repo": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        "size_mb": 1400,
        "labels": ["arousal", "dominance", "valence"],
        "type": "dimensional",
        "description": "Outputs intensity scores (0-1), not labels. Arousal=energy, Valence=positive/negative. Good for intensity detection.",
        "recommended": False,
    },
]

_REGISTRY_BY_ID: dict[str, dict] = {m["id"]: m for m in EMOTION_MODELS}


class ModelManager:
    """Manage emotion model download, status, and inference."""

    def __init__(self, models_dir: Path | None = None):
        if models_dir is None:
            from pathlib import Path as _Path
            models_dir = _Path.home() / ".whisperapp" / "emotion_models"
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        # {model_id: {"progress": 0-100, "downloading": bool, "error": str|None}}
        self._download_state: dict[str, dict] = {}
        # {model_id: loaded_pipeline}
        self._loaded: dict[str, object] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self, model_id: str) -> dict:
        """Return status dict for a single model."""
        if model_id not in _REGISTRY_BY_ID:
            return {"id": model_id, "downloaded": False, "downloading": False,
                    "progress": 0, "error": "Unknown model ID"}

        with self._lock:
            state = self._download_state.get(model_id, {})

        downloaded = (self._models_dir / model_id).exists()
        return {
            "id": model_id,
            "downloaded": downloaded,
            "downloading": state.get("downloading", False),
            "progress": state.get("progress", 100 if downloaded else 0),
            "error": state.get("error", None),
        }

    def list_all(self) -> list[dict]:
        """Return registry entries merged with status for each model."""
        result = []
        for m in EMOTION_MODELS:
            entry = dict(m)
            entry.update(self.get_status(m["id"]))
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self, model_id: str) -> None:
        """Start a background download for model_id. Returns immediately."""
        if model_id not in _REGISTRY_BY_ID:
            raise ValueError(f"Unknown model ID: {model_id}")

        with self._lock:
            state = self._download_state.get(model_id, {})
            if state.get("downloading"):
                return  # already in progress

        model_dir = self._models_dir / model_id
        if model_dir.exists():
            return  # already downloaded

        with self._lock:
            self._download_state[model_id] = {
                "downloading": True, "progress": 0, "error": None
            }

        thread = threading.Thread(
            target=self._download_thread,
            args=(model_id,),
            daemon=True,
            name=f"emotion-dl-{model_id}",
        )
        thread.start()

    def _download_thread(self, model_id: str) -> None:
        hf_repo = _REGISTRY_BY_ID[model_id]["hf_repo"]
        model_dir = self._models_dir / model_id

        try:
            from huggingface_hub import snapshot_download

            def _progress_callback(progress: dict):
                # huggingface_hub >= 0.14 provides tqdm-style progress info
                # The callback is called per-file, not overall — we estimate
                # based on files completed.
                pass

            # snapshot_download downloads to a cache dir, then we symlink/move
            # Use local_dir to store directly in our models dir
            snapshot_download(
                repo_id=hf_repo,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )

            with self._lock:
                self._download_state[model_id] = {
                    "downloading": False, "progress": 100, "error": None
                }
            logger.info("Downloaded emotion model: %s", model_id)

        except Exception as exc:
            logger.error("Failed to download %s: %s", model_id, exc)
            # Clean up partial download
            if model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=True)
            with self._lock:
                self._download_state[model_id] = {
                    "downloading": False,
                    "progress": 0,
                    "error": str(exc),
                }

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, model_id: str) -> None:
        """Remove model directory from disk. No-op if not downloaded."""
        if model_id not in _REGISTRY_BY_ID:
            raise ValueError(f"Unknown model ID: {model_id}")

        model_dir = self._models_dir / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)

        # Remove from loaded cache
        with self._lock:
            self._loaded.pop(model_id, None)
            self._download_state.pop(model_id, None)

    # ------------------------------------------------------------------
    # Load & infer
    # ------------------------------------------------------------------

    def load(self, model_id: str) -> object:
        """Load model into memory using transformers pipeline. Cached."""
        with self._lock:
            if model_id in self._loaded:
                return self._loaded[model_id]

        model_info = _REGISTRY_BY_ID.get(model_id)
        if not model_info:
            raise ValueError(f"Unknown model ID: {model_id}")

        model_dir = self._models_dir / model_id
        if not model_dir.exists():
            raise RuntimeError(f"Model {model_id} is not downloaded")

        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise RuntimeError(
                "transformers is not installed. Run: pip install transformers"
            )

        model_type = model_info["type"]

        if model_type == "dimensional":
            # Audeering model uses a custom forward pass — load with AutoModel
            try:
                import numpy as np
                from transformers import AutoConfig, AutoProcessor, Wav2Vec2Processor
                pipe = _DimensionalPipeline(str(model_dir))
            except Exception as exc:
                raise RuntimeError(f"Failed to load dimensional model: {exc}") from exc
        else:
            try:
                pipe = hf_pipeline(
                    "audio-classification",
                    model=str(model_dir),
                    device=-1,  # CPU
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to load model {model_id}: {exc}") from exc

        with self._lock:
            self._loaded[model_id] = pipe
        return pipe

    def infer(self, model_id: str, audio_path: str, segments: list[dict]) -> list[dict]:
        """Run inference on each segment.

        Returns list of:
          Classification: {"segment_idx": int, "label": str, "confidence": float}
          Dimensional: {"segment_idx": int, "arousal": float, "dominance": float, "valence": float}
        """
        model_info = _REGISTRY_BY_ID.get(model_id)
        if not model_info:
            raise ValueError(f"Unknown model ID: {model_id}")

        try:
            import librosa
            import numpy as np
        except ImportError:
            raise RuntimeError("librosa is not installed. Run: pip install librosa")

        pipe = self.load(model_id)
        model_type = model_info["type"]

        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio for inference: {exc}") from exc

        results: list[dict] = []

        for idx, seg in enumerate(segments):
            if (seg.get("is_acoustic_marker") or seg.get("is_pause_marker")
                    or seg.get("is_emotion_marker")):
                continue

            start = seg.get("start", 0)
            end = seg.get("end", 0)
            if end <= start:
                continue

            try:
                f1 = int(start * sr)
                f2 = int(end * sr)
                window = audio[f1:f2]
                if len(window) < sr // 2:  # skip segments < 500ms
                    continue

                if model_type == "dimensional":
                    scores = pipe.predict(window, sr)
                    results.append({
                        "segment_idx": idx,
                        "arousal": float(scores[0]),
                        "dominance": float(scores[1]),
                        "valence": float(scores[2]),
                    })
                else:
                    out = pipe(window.tolist(), sampling_rate=sr)
                    if isinstance(out, list) and out:
                        # transformers pipeline returns list of {"label": ..., "score": ...}
                        best = max(out, key=lambda x: x["score"])
                        results.append({
                            "segment_idx": idx,
                            "label": best["label"].lower(),
                            "confidence": float(best["score"]),
                        })

            except Exception as exc:
                logger.debug("Inference failed for segment %d: %s", idx, exc)
                continue

        return results


class _DimensionalPipeline:
    """Minimal wrapper for the Audeering dimensional model."""

    def __init__(self, model_dir: str):
        try:
            import numpy as np
            import torch
            from transformers import AutoConfig, Wav2Vec2Processor, AutoModelForAudioClassification
            self._processor = Wav2Vec2Processor.from_pretrained(model_dir)
            self._model = AutoModelForAudioClassification.from_pretrained(model_dir)
            self._model.eval()
            self._torch = torch
            self._np = np
        except Exception as exc:
            raise RuntimeError(f"Failed to load Audeering model: {exc}") from exc

    def predict(self, audio: "np.ndarray", sr: int) -> tuple[float, float, float]:
        """Return (arousal, dominance, valence) in 0-1 range."""
        import numpy as np
        inputs = self._processor(
            audio, sampling_rate=sr, return_tensors="pt", padding=True
        )
        with self._torch.no_grad():
            out = self._model(**inputs)
        # logits shape: (1, 3) for arousal/dominance/valence
        logits = out.logits.squeeze().numpy()
        # Sigmoid to get 0-1 scores
        scores = 1.0 / (1.0 + np.exp(-logits))
        if scores.ndim == 0:
            scores = np.array([float(scores)] * 3)
        return float(scores[0]), float(scores[1]), float(scores[2])
