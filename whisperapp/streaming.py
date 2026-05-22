"""Real-time streaming transcription engine using faster-whisper + Silero VAD."""

import io
import logging
import threading
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Silero VAD helpers
# ---------------------------------------------------------------------------

_vad_model = None
_vad_lock = threading.Lock()


def _load_vad():
    """Load Silero VAD model (cached singleton)."""
    global _vad_model
    if _vad_model is None:
        with _vad_lock:
            if _vad_model is None:
                import torch
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                _vad_model = model
    return _vad_model


def _speech_prob(audio_16k: np.ndarray) -> float:
    """Return max speech probability across 512-sample windows at 16 kHz."""
    import torch
    vad = _load_vad()
    if audio_16k.shape[0] < 512:
        return 0.0
    max_prob = 0.0
    # Silero VAD requires exactly 512 samples per call at 16 kHz
    for i in range(0, len(audio_16k) - 511, 512):
        window = torch.from_numpy(audio_16k[i:i + 512]).float()
        prob = float(vad(window, 16000).detach())
        if prob > max_prob:
            max_prob = prob
    return max_prob


# ---------------------------------------------------------------------------
# AudioBuffer
# ---------------------------------------------------------------------------

class AudioBuffer:
    """Accumulates PCM chunks, uses Silero VAD to detect utterance boundaries.

    Yields complete utterance buffers when silence is detected after speech.
    """

    def __init__(
        self,
        min_chunk_sec: float = 1.0,
        max_chunk_sec: float = 10.0,
        silence_threshold_sec: float = 0.6,
        target_sr: int = 16000,
    ):
        self.min_chunk_sec = min_chunk_sec
        self.max_chunk_sec = max_chunk_sec
        self.silence_threshold_sec = silence_threshold_sec
        self.target_sr = target_sr

        self._buf: list[np.ndarray] = []
        self._buf_samples = 0
        self._silence_samples = 0
        self._has_speech = False

    def add_chunk(self, sr: int, audio: np.ndarray) -> None:
        """Append audio chunk. *audio* is float32, mono or first-channel."""
        if audio.ndim > 1:
            audio = audio[:, 0] if audio.shape[1] > 1 else audio.ravel()

        # Resample to 16 kHz if needed
        if sr != self.target_sr:
            import soxr
            audio = soxr.resample(audio.astype(np.float32), sr, self.target_sr)

        audio = audio.astype(np.float32)

        self._buf.append(audio)
        self._buf_samples += len(audio)

        # Run VAD on the latest chunk (512-sample windows for Silero)
        prob = _speech_prob(audio) if len(audio) >= 512 else 0.0

        if prob > 0.5:
            self._has_speech = True
            self._silence_samples = 0
        else:
            self._silence_samples += len(audio)

    def get_utterance(self) -> Optional[np.ndarray]:
        """Return buffered audio when an utterance boundary is detected, else None."""
        duration = self._buf_samples / self.target_sr
        silence_dur = self._silence_samples / self.target_sr

        # Force flush if max buffer reached — even if VAD never detected speech,
        # so audio doesn't accumulate unboundedly and crash on stop.
        if duration >= self.max_chunk_sec:
            logger.debug("AudioBuffer force-flush at %.2fs (max_chunk_sec=%.1f)", duration, self.max_chunk_sec)
            return self._flush()

        # Silence after speech → utterance complete
        if (
            self._has_speech
            and silence_dur >= self.silence_threshold_sec
            and duration >= self.min_chunk_sec
        ):
            logger.debug("AudioBuffer utterance boundary: %.2fs audio, %.2fs silence", duration, silence_dur)
            return self._flush()

        return None

    def get_all(self) -> Optional[np.ndarray]:
        """Return all buffered audio regardless of VAD state."""
        if not self._buf:
            return None
        return self._flush()

    def _flush(self) -> np.ndarray:
        out = np.concatenate(self._buf)
        self._buf.clear()
        self._buf_samples = 0
        self._silence_samples = 0
        self._has_speech = False
        return out

    @property
    def duration(self) -> float:
        return self._buf_samples / self.target_sr


# ---------------------------------------------------------------------------
# TranscriptAccumulator
# ---------------------------------------------------------------------------

class TranscriptAccumulator:
    """Maintains a running transcript, deduplicates overlapping segments."""

    def __init__(self):
        self._segments: list[dict] = []
        self._offset: float = 0.0

    def add_segment(self, start: float, end: float, text: str) -> None:
        text = text.strip()
        if not text:
            return

        abs_start = self._offset + start
        abs_end = self._offset + end

        # Deduplicate only exact repeats (case-insensitive). The VAD buffer
        # yields non-overlapping utterances, so anything else is real content.
        if self._segments:
            last = self._segments[-1]
            if last["text"].strip().lower() == text.lower():
                return

        self._segments.append({
            "start": round(abs_start, 3),
            "end": round(abs_end, 3),
            "text": text,
        })

    def advance_offset(self, seconds: float) -> None:
        """Advance the time offset by the duration of the last processed chunk."""
        self._offset += seconds

    def get_full_text(self) -> str:
        return " ".join(s["text"] for s in self._segments)

    def get_segments(self) -> list[dict]:
        return list(self._segments)

    def clear(self) -> None:
        self._segments.clear()
        self._offset = 0.0


def _overlap_ratio(a: str, b: str) -> float:
    """Simple word-overlap ratio between two strings."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


# ---------------------------------------------------------------------------
# StreamingEngine
# ---------------------------------------------------------------------------

class StreamingEngine:
    """Orchestrates live streaming transcription.

    Uses faster-whisper for chunk-based inference and Silero VAD for
    speech boundary detection.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        min_chunk_sec: float = 1.0,
        max_chunk_sec: float = 10.0,
        silence_threshold_sec: float = 0.6,
    ):
        self.model_size = model_size
        if device == "auto":
            from whisperapp.worker import _WHISPER_DEVICE, _COMPUTE_TYPE as _CT
            self.device = _WHISPER_DEVICE
            _auto_compute = _CT
        else:
            self.device = device
            _auto_compute = "float16" if device == "cuda" else "int8"
        self.compute_type = compute_type if compute_type != "auto" else _auto_compute

        self._model = None
        self._buffer = AudioBuffer(
            min_chunk_sec=min_chunk_sec,
            max_chunk_sec=max_chunk_sec,
            silence_threshold_sec=silence_threshold_sec,
        )
        self._accumulator = TranscriptAccumulator()
        self._all_audio: list[np.ndarray] = []
        self._running = False

    def start(self) -> None:
        """Load the faster-whisper model (lazy, first call only)."""
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        self._running = True
        logger.info("StreamingEngine started (model=%s)", self.model_size)

    def process_chunk(self, sr: int, audio: np.ndarray) -> Optional[str]:
        """Feed an audio chunk. Returns new transcript text if an utterance completed."""
        if not self._running:
            self.start()

        self._buffer.add_chunk(sr, audio)

        # Store raw audio for optional Polish step
        if audio.ndim > 1:
            audio_mono = audio[:, 0] if audio.shape[1] > 1 else audio.ravel()
        else:
            audio_mono = audio
        self._all_audio.append(audio_mono.astype(np.float32))

        utterance = self._buffer.get_utterance()
        if utterance is None:
            return None

        return self._transcribe_utterance(utterance)

    def _transcribe_utterance(self, audio_16k: np.ndarray) -> str:
        """Transcribe a single utterance buffer with faster-whisper."""
        duration = len(audio_16k) / 16000
        logger.debug("Transcribing utterance %.2fs", duration)
        segments, _info = self._model.transcribe(
            audio_16k,
            beam_size=1,
            language="en",  # lock to English to prevent hallucination
            vad_filter=False,  # we handle VAD ourselves
        )

        new_text_parts = []
        for seg in segments:
            self._accumulator.add_segment(seg.start, seg.end, seg.text)
            new_text_parts.append(seg.text.strip())

        # Advance offset by utterance duration
        self._accumulator.advance_offset(duration)

        text = " ".join(new_text_parts) if new_text_parts else None
        if text:
            logger.debug("Utterance transcribed: %r", text[:120])
        return text

    def stop(self) -> dict:
        """Stop streaming, flush remaining audio, return final transcript + raw audio."""
        self._running = False
        logger.info("StreamingEngine stopping — flushing remaining buffer")

        # Flush any remaining buffered audio
        remaining = self._buffer.get_all()
        if remaining is not None and len(remaining) >= 8000:  # at least 0.5s
            self._transcribe_utterance(remaining)

        # Concatenate all raw audio for Polish step
        raw_audio = (
            np.concatenate(self._all_audio)
            if self._all_audio
            else np.array([], dtype=np.float32)
        )

        text = self._accumulator.get_full_text()
        segs = self._accumulator.get_segments()
        logger.info("StreamingEngine stopped: %d segments, %d words, %.1fs audio",
                    len(segs), len(text.split()) if text else 0, len(raw_audio) / 16000)
        return {
            "text": text,
            "segments": segs,
            "raw_audio": raw_audio,
            "sample_rate": 16000,
        }

    def get_transcript(self) -> str:
        """Return the current full transcript."""
        return self._accumulator.get_full_text()

    def reset(self) -> None:
        """Clear all state for a new session."""
        self._accumulator.clear()
        self._buffer = AudioBuffer(
            min_chunk_sec=self._buffer.min_chunk_sec,
            max_chunk_sec=self._buffer.max_chunk_sec,
            silence_threshold_sec=self._buffer.silence_threshold_sec,
        )
        self._all_audio.clear()
        self._running = False

    def polish(self, hf_token: str, on_progress=None) -> dict:
        """Run WhisperX alignment + diarization on accumulated audio.

        This is a post-hoc step that produces word-level timestamps and
        speaker labels — not possible in real-time.

        on_progress(stage, detail) is called at each stage if provided.
        """
        def _progress(stage, detail=""):
            if on_progress:
                on_progress(stage, detail)

        if not self._all_audio:
            return {"segments": [], "text": ""}

        raw_audio = np.concatenate(self._all_audio)
        duration_sec = len(raw_audio) / 16000
        logger.info("StreamingEngine.polish: %.1fs audio on %s", duration_sec, self.device)
        _progress("preparing", f"{duration_sec:.0f}s of audio on {self.device}")

        import whisperx
        from whisperapp.worker import _WHISPER_DEVICE, _DIARIZE_DEVICE, _COMPUTE_TYPE

        # Stage 1: Transcribe
        _progress("transcribing", "loading WhisperX model...")
        model = whisperx.load_model(
            self.model_size, device=_WHISPER_DEVICE, compute_type=_COMPUTE_TYPE
        )
        _progress("transcribing", "running transcription...")
        result = model.transcribe(raw_audio, batch_size=8, language="en")
        del model
        _progress("transcribing", "done")

        # Stage 2: Align
        language = result.get("language", "en")
        _progress("aligning", f"loading alignment model (lang={language})...")
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=_WHISPER_DEVICE
        )
        _progress("aligning", "running alignment...")
        aligned = whisperx.align(
            result["segments"], align_model, metadata, raw_audio, device=_WHISPER_DEVICE
        )
        del align_model
        _progress("aligning", "done")

        # Stage 3: Diarize (optional, requires HF token)
        if hf_token:
            try:
                from whisperx.diarize import DiarizationPipeline
                import tempfile
                import soundfile as sf

                _progress("diarizing", "loading diarization pipeline...")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, raw_audio, 16000)
                    tmp_path = f.name

                diarize_pipeline = DiarizationPipeline(
                    token=hf_token, device=_DIARIZE_DEVICE
                )
                _progress("diarizing", "assigning speakers...")
                diarize_segments = diarize_pipeline(tmp_path)
                result_diarized = whisperx.assign_word_speakers(
                    diarize_segments, aligned
                )

                import os
                os.unlink(tmp_path)

                _progress("complete", "")
                return result_diarized
            except Exception as e:
                logger.warning("Diarization failed, returning aligned result: %s", e)
                _progress("complete", f"diarization failed: {e} (returning aligned)")
                return aligned
        else:
            _progress("complete", "")
            return aligned
