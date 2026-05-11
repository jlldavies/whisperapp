import gc
import logging
import threading
import time
from pathlib import Path
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from whisperapp.queue import JobQueue, JobStatus
from whisperapp.checkpoints import CheckpointManager

_log = logging.getLogger(__name__)


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _webhook_host_allowed(url: str, allowed_hosts: list) -> bool:
    """
    Guard against SSRF.

    - Empty allowed_hosts list → loopback only.
    - ["*"] → any host (operator opts in to permissionless mode).
    - Otherwise, exact match against the list entries.
    """
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    if not host:
        return False
    if allowed_hosts == ["*"]:
        return True
    if not allowed_hosts:
        return host in _LOOPBACK_HOSTS
    return host in allowed_hosts or host in _LOOPBACK_HOSTS


def _sign_payload(body: bytes, secret: str) -> str:
    import hmac as _hmac
    import hashlib
    return "sha256=" + _hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _fire_webhook(job: dict) -> None:
    url = job.get("callback_url")
    if not url:
        return

    from whisperapp.config import Config as _Config
    cfg = _Config()

    if not _webhook_host_allowed(url, cfg.webhook_allowed_hosts):
        _log.warning(
            "Webhook to %s blocked — host not in webhook_allowed_hosts. "
            "Set webhook_allowed_hosts=[\"*\"] in config to allow external hosts.",
            url,
        )
        return

    import json as _json
    import requests
    payload = {
        "event": "job.completed" if job["status"] == JobStatus.DONE else "job.failed",
        "job_id": job["id"],
        "status": job["status"],
        "file_name": job.get("file_name"),
        "file_path": job.get("file_path"),
        "output_path": job.get("output_path"),
        "formats": job.get("formats"),
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
    }
    body = _json.dumps(payload, separators=(",", ":")).encode()
    headers = {
        "Content-Type": "application/json",
        "X-WhisperApp-Signature": _sign_payload(body, cfg.webhook_secret),
    }
    try:
        requests.post(url, data=body, headers=headers, timeout=10)
    except Exception as exc:
        _log.warning("Webhook POST to %s failed: %s", url, exc)

def _detect_device() -> tuple[str, str, str]:
    """
    Return (whisper_device, diarize_device, compute_type).

    Whisper uses faster-whisper / ctranslate2, which only supports CUDA and CPU.
    Pyannote (diarization) is pure PyTorch and supports MPS on Apple Silicon.

    Compute types:
      CUDA  → float16  (fast, memory-efficient on GPU)
      CPU   → int8     (2-4x faster than float32 on modern CPUs including Apple Silicon)
    """
    import sys

    if torch.cuda.is_available():
        return "cuda", "cuda", "float16"

    if sys.platform == "darwin" and torch.backends.mps.is_available():
        # ctranslate2 cannot use MPS — Whisper runs on CPU with int8.
        # Pyannote CAN use MPS — use it for diarization.
        return "cpu", "mps", "int8"

    return "cpu", "cpu", "int8"


_WHISPER_DEVICE, _DIARIZE_DEVICE, _COMPUTE_TYPE = _detect_device()
# Legacy alias kept for any code referencing _DEVICE
_DEVICE = _WHISPER_DEVICE


def _has_mlx_whisper() -> bool:
    """True if mlx-whisper is installed (Apple Silicon fast path)."""
    try:
        import mlx_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _cleanup_memory():
    """Force release of GPU/CPU memory between jobs.

    Without this, sequential model loads on Windows + MKL fragment the heap
    (mkl_malloc OOM) and CUDA contexts hold cached allocations that block the
    next model load. Call from process_job's finally block and from
    complete_with_result.
    """
    # Two passes — Python sometimes needs a second collect to reach cyclically
    # referenced tensors that the first collect makes unreachable.
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    # MPS (Apple Silicon) — clear Metal buffer pool
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Stage definitions — each stage owns a slice of the 0-100 progress bar.
# ---------------------------------------------------------------------------
STAGES = {
    "loading_model":  {"label": "1/5 Loading model",      "start":  0, "end":  5},
    "loading_audio":  {"label": "1/5 Loading audio",       "start":  5, "end":  8},
    "transcribing":   {"label": "2/5 Transcribing",        "start":  8, "end": 40},
    "aligning":       {"label": "3/5 Aligning words",      "start": 40, "end": 60},
    "diarizing":      {"label": "4/5 Identifying speakers", "start": 60, "end": 85},
    "speaker_review": {"label": "5/5 Speaker review",      "start": 85, "end": 88},
    "acoustics":      {"label": "5/5 Acoustic analysis",   "start": 88, "end": 90},
    "emotions":       {"label": "5/5 Emotion analysis",    "start": 90, "end": 95},
    "saving":         {"label": "5/5 Saving files",        "start": 95, "end": 100},
}


def _stage_progress(stage_name: str, fraction: float) -> int:
    """Map a 0.0-1.0 fraction within a stage to the overall 0-100 progress."""
    s = STAGES[stage_name]
    return int(s["start"] + fraction * (s["end"] - s["start"]))


class _Heartbeat:
    """Background thread that updates the DB stage text with elapsed time
    so the UI always shows movement, even during long blocking calls."""

    def __init__(self, queue: JobQueue, job_id: str, stage_name: str,
                 interval: float = 5.0):
        self.queue = queue
        self.job_id = job_id
        self.stage_name = stage_name
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._start_time = time.time()

    def start(self):
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self._stop.wait(self.interval)
            if self._stop.is_set():
                break
            elapsed = int(time.time() - self._start_time)
            label = STAGES[self.stage_name]["label"]
            # Read current progress from DB so we don't overwrite segment-level updates
            job = self.queue.get_job(self.job_id)
            current_progress = job["progress"] if job else STAGES[self.stage_name]["start"]
            self.queue.update_progress(
                self.job_id, current_progress,
                f"{label} ({elapsed}s elapsed)")

    def update(self, stage_name: str, progress: int = None, detail: str = ""):
        """Switch to a new stage (resets timer)."""
        self.stage_name = stage_name
        self._start_time = time.time()
        p = progress if progress is not None else STAGES[stage_name]["start"]
        self.queue.update_progress(self.job_id, p, detail or STAGES[stage_name]["label"])

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def _patched_transcribe(original_transcribe, queue, job_id, heartbeat):
    """Wrap whisperx transcribe to report per-segment progress to the DB."""
    import types

    def wrapper(self, audio, batch_size=None, num_workers=0,
                language=None, task=None, chunk_size=30,
                print_progress=False, combined_progress=False, verbose=False):
        import numpy as np
        from whisperx.audio import SAMPLE_RATE, load_audio
        from whisperx.vads import Vad, Pyannote
        from dataclasses import replace
        from whisperx.asr import find_numeral_symbol_tokens, Tokenizer, logger

        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                yield {'inputs': audio[f1:f2]}

        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks

        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments, chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )

        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task, language=language,
            )
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = Tokenizer(
                    self.model.hf_tokenizer,
                    self.model.model.is_multilingual,
                    task=task, language=language,
                )

        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            logger.info("Suppressing numeral and symbol tokens")
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = replace(self.options, suppress_tokens=new_suppressed_tokens)

        from typing import List
        from whisperx.schema import SingleSegment, TranscriptionResult

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)

        for idx, out in enumerate(self.__call__(data(audio, vad_segments),
                                                 batch_size=batch_size,
                                                 num_workers=num_workers)):
            # Update progress on every segment (heartbeat handles throttling)
            frac = (idx + 1) / total_segments if total_segments else 1
            overall = _stage_progress("transcribing", frac)
            stage_label = STAGES["transcribing"]["label"]
            detail = f"{stage_label} — segment {idx+1}/{total_segments}"
            queue.update_progress(job_id, overall, detail)
            # Reset heartbeat timer so it shows elapsed within current segment
            heartbeat._start_time = time.time()
            heartbeat.stage_name = "transcribing"

            text = out['text']
            avg_logprob = out['avg_logprob']
            if batch_size in [0, 1, None]:
                text = text[0]
                avg_logprob = avg_logprob[0]

            segments.append(
                {"text": text, "start": round(vad_segments[idx]['start'], 3),
                 "end": round(vad_segments[idx]['end'], 3),
                 "avg_logprob": avg_logprob})

        # Restore suppressed tokens if needed
        if self.suppress_numerals:
            self.options = replace(self.options, suppress_tokens=previous_suppress_tokens)

        return {"segments": segments, "language": language}

    return types.MethodType(wrapper, original_transcribe.__self__)


def _tracked_align(align_fn, segments, align_model, metadata,
                   audio_path, device, queue, job_id, heartbeat):
    """Wrap whisperx.align with print_progress=True and capture stdout
    to update the DB with per-segment alignment progress."""
    import io, sys, re

    total = len(segments)
    progress_re = re.compile(r"Progress:\s+([\d.]+)%")

    class _ProgressCapture(io.TextIOBase):
        """Intercept stdout writes looking for progress lines."""
        def __init__(self, original):
            self._original = original

        def write(self, s):
            m = progress_re.search(s)
            if m:
                pct = float(m.group(1))
                frac = pct / 100.0
                overall = _stage_progress("aligning", frac)
                seg_num = int(frac * total)
                label = STAGES["aligning"]["label"]
                queue.update_progress(
                    job_id, overall,
                    f"{label} — segment {seg_num}/{total}")
                heartbeat._start_time = time.time()
            # Still print to real stdout
            return self._original.write(s)

        def flush(self):
            self._original.flush()

    old_stdout = sys.stdout
    sys.stdout = _ProgressCapture(old_stdout)
    try:
        aligned = align_fn(
            segments, align_model, metadata,
            audio_path, device=device, print_progress=True)
    finally:
        sys.stdout = old_stdout
    return aligned


def _install_diarize_hook(diarize_pipeline, queue, job_id, heartbeat):
    """Hook into pyannote's DiarizationPipeline to report progress.
    Pyannote pipelines use a hook system — we install one that updates
    the DB as each internal step completes."""
    try:
        from pyannote.audio.pipelines.utils import oracle
        # pyannote pipelines accept a 'hook' callback in __call__
        # The hook receives (step_name, step_artefact, file) on each step.
        # We wrap the pipeline's __call__ to inject our hook.
        original_call = diarize_pipeline.__class__.__call__

        _steps_seen = []

        def _hooked_call(self, file, **kwargs):
            # pyannote passes hook= to internal steps
            def progress_hook(step_name, step_artefact, file=None):
                _steps_seen.append(step_name)
                n = len(_steps_seen)
                # Estimate ~4 major steps in diarization pipeline
                frac = min(n / 4.0, 0.85)
                overall = _stage_progress("diarizing", frac)
                label = STAGES["diarizing"]["label"]
                queue.update_progress(
                    job_id, overall,
                    f"{label} — {step_name}")
                heartbeat._start_time = time.time()

            kwargs.setdefault("hook", progress_hook)
            return original_call(self, file, **kwargs)

        import types as _types
        diarize_pipeline.__call__ = _types.MethodType(_hooked_call, diarize_pipeline)
    except Exception:
        # If hook injection fails, fall back to heartbeat-only progress
        pass


class Worker:
    def __init__(self, queue: JobQueue, hf_token: str, config_dir: Path = None):
        self.queue = queue
        self.hf_token = hf_token
        self._thread = None
        self._stop_event = threading.Event()
        self._paused = False

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    def _loop(self):
        while not self._stop_event.is_set():
            if not self._paused:
                job = self.queue.next_queued()
                if job:
                    try:
                        self.process_job(job["id"])
                    except Exception:
                        pass  # job already marked failed; continue loop
                    continue
            time.sleep(2)

    def process_job(self, job_id: str):
        job = self.queue.get_job(job_id)
        if not job or job["status"] == JobStatus.CANCELLED:
            return

        cm = CheckpointManager(job["output_path"], job_id)
        hb = _Heartbeat(self.queue, job_id, "loading_model", interval=5.0).start()

        # Local handles for explicit cleanup in finally — declared up-front so the
        # finally block can safely `del` them whether or not the try-block reached
        # the corresponding stage.
        model = None
        audio = None
        align_model = None
        metadata = None
        diarize_model = None
        diarize_segments = None
        result = None
        aligned = None
        result_diarized = None
        final_result = None

        try:
            # --- Stage 1a: Load model ---
            if not cm.has("transcription"):
                hb.update("loading_model")

                # --- Stage 1b: Load audio ---
                hb.update("loading_audio")
                audio = whisperx.load_audio(job["file_path"])

                # --- Stage 2: Transcribe ---
                hb.update("transcribing")

                if _has_mlx_whisper() and _WHISPER_DEVICE == "cpu":
                    # Apple Silicon fast path: mlx-whisper uses the Metal GPU
                    # directly via Apple's MLX framework — much faster than
                    # ctranslate2 on CPU for the same model size.
                    import mlx_whisper
                    _MLX_MODEL_MAP = {
                        "tiny": "mlx-community/whisper-tiny-mlx",
                        "base": "mlx-community/whisper-base-mlx",
                        "small": "mlx-community/whisper-small-mlx",
                        "medium": "mlx-community/whisper-medium-mlx",
                        "large-v2": "mlx-community/whisper-large-v2-mlx",
                    }
                    mlx_model = _MLX_MODEL_MAP.get(job["model"],
                                                   "mlx-community/whisper-large-v2-mlx")
                    raw = mlx_whisper.transcribe(
                        job["file_path"],
                        path_or_hf_repo=mlx_model,
                        word_timestamps=True,
                    )
                    # Normalise to whisperx segment format
                    result = {
                        "segments": [
                            {"text": seg["text"],
                             "start": round(seg["start"], 3),
                             "end": round(seg["end"], 3)}
                            for seg in raw.get("segments", [])
                        ],
                        "language": raw.get("language", "en"),
                    }
                else:
                    model = whisperx.load_model(job["model"],
                                                device=_WHISPER_DEVICE,
                                                compute_type=_COMPUTE_TYPE)
                    model.transcribe = _patched_transcribe(
                        model.transcribe, self.queue, job_id, hb)
                    result = model.transcribe(audio, batch_size=8, language="en")
                    del model; model = None

                cm.save("transcription", result)
                # Release audio buffer; can be 500MB+ for a long recording.
                del audio; audio = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                result = cm.load("transcription")

            if self.queue.get_job(job_id)["status"] == JobStatus.CANCELLED:
                return

            # --- Stage 3: Alignment (with per-segment progress) ---
            if not cm.has("alignment"):
                total_segs = len(result["segments"])
                hb.update("aligning",
                           detail=f"{STAGES['aligning']['label']} — 0/{total_segs} segments")
                align_model, metadata = whisperx.load_align_model(
                    language_code=result["language"], device=_WHISPER_DEVICE)

                # Capture print_progress output to update DB per-segment
                aligned = _tracked_align(
                    whisperx.align, result["segments"], align_model,
                    metadata, job["file_path"], _WHISPER_DEVICE,
                    self.queue, job_id, hb)

                cm.save("alignment", aligned)
                # Release alignment model + metadata; ~300 MB combined.
                del align_model; align_model = None
                del metadata; metadata = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                aligned = cm.load("alignment")

            if self.queue.get_job(job_id)["status"] == JobStatus.CANCELLED:
                return

            # --- Stage 4: Diarization (with progress via pyannote hooks) ---
            if job["diarize"] and not cm.has("diarization"):
                hb.update("diarizing")
                diarize_model = DiarizationPipeline(
                    token=self.hf_token, device=_DIARIZE_DEVICE)

                # Hook into pyannote's pipeline progress
                _install_diarize_hook(
                    diarize_model, self.queue, job_id, hb)
                # Pass audio as a pre-loaded waveform dict so pyannote never
                # needs to decode the file itself — avoids torchcodec /
                # FFmpeg-shared-library dependency on Windows.
                _audio_np = whisperx.load_audio(job["file_path"])
                _waveform = torch.from_numpy(_audio_np).unsqueeze(0)  # (1, time)
                diarize_segments = diarize_model(
                    {"waveform": _waveform, "sample_rate": whisperx.audio.SAMPLE_RATE}
                )
                del _audio_np, _waveform

                self.queue.update_progress(
                    job_id, _stage_progress("diarizing", 0.9),
                    f"{STAGES['diarizing']['label']} — assigning speakers")
                result_diarized = whisperx.assign_word_speakers(
                    diarize_segments, aligned)
                cm.save("diarization", result_diarized)
                final_result = result_diarized
                # Release diarization pipeline + segments; ~500 MB+
                del diarize_model; diarize_model = None
                del diarize_segments; diarize_segments = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif cm.has("diarization"):
                final_result = cm.load("diarization")
            else:
                final_result = aligned

            if self.queue.get_job(job_id)["status"] == JobStatus.CANCELLED:
                return

            # --- Stage 5a: Acoustic analysis (fast, no model needed) ---
            from whisperapp.config import Config as _Config
            cfg = _Config()
            if cfg.acoustic_enabled:
                hb.update("acoustics")
                try:
                    from whisperapp.acoustic import analyse_acoustics
                    final_result = analyse_acoustics(job["file_path"], final_result, cfg)
                except Exception as _e:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "Acoustic analysis failed: %s", _e)

            # --- Stage 5b: Emotion analysis (slow, only if models downloaded) ---
            if cfg.emotion_enabled and cfg.emotion_model_ids:
                hb.update("emotions")
                try:
                    from whisperapp.emotion import analyse_emotions
                    from whisperapp.emotion_registry import ModelManager
                    _emotion_manager = ModelManager()
                    final_result = analyse_emotions(
                        job["file_path"], final_result, cfg, _emotion_manager)
                except Exception as _e:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "Emotion analysis failed: %s", _e)

            # --- Stage 5c: Speaker review or save ---
            if job["diarize"]:
                cm.save("speaker_review", final_result)
                hb.update("speaker_review")
                self.queue.set_status(job_id, JobStatus.SPEAKER_REVIEW)
            else:
                self._write_outputs(job, final_result, cm)
                self.queue.complete_job(job_id, job["output_path"])
                threading.Thread(
                    target=_fire_webhook,
                    args=(self.queue.get_job(job_id),),
                    daemon=True,
                ).start()

        except Exception as e:
            self.queue.set_status(job_id, JobStatus.FAILED, error=str(e))
            threading.Thread(
                target=_fire_webhook,
                args=(self.queue.get_job(job_id),),
                daemon=True,
            ).start()
            raise
        finally:
            hb.stop()
            # Best-effort release of any locals still alive (handles partial
            # failure paths). Each `del` is wrapped because some may already be
            # None or never assigned.
            for name in ('model', 'audio', 'align_model', 'metadata',
                         'diarize_model', 'diarize_segments',
                         'result', 'aligned', 'result_diarized', 'final_result'):
                try:
                    del locals()[name]
                except Exception:
                    pass
            _cleanup_memory()

    def complete_with_result(self, job_id: str, renamed_segments: dict):
        """Called after speaker review is complete (names applied or skipped)."""
        try:
            job = self.queue.get_job(job_id)
            cm = CheckpointManager(job["output_path"], job_id)
            final_result = renamed_segments

            # Apply acoustic and emotion analysis before writing outputs
            from whisperapp.config import Config as _Config
            cfg = _Config()
            if cfg.acoustic_enabled:
                try:
                    from whisperapp.acoustic import analyse_acoustics
                    final_result = analyse_acoustics(job["file_path"], final_result, cfg)
                except Exception as _e:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "Acoustic analysis failed in complete_with_result: %s", _e)
            if cfg.emotion_enabled and cfg.emotion_model_ids:
                try:
                    from whisperapp.emotion import analyse_emotions
                    from whisperapp.emotion_registry import ModelManager
                    _emotion_manager = ModelManager()
                    final_result = analyse_emotions(
                        job["file_path"], final_result, cfg, _emotion_manager)
                except Exception as _e:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "Emotion analysis failed in complete_with_result: %s", _e)

            cm.save("saving", final_result)
            self._write_outputs(job, final_result, cm)
            self.queue.complete_job(job_id, job["output_path"])
            threading.Thread(
                target=_fire_webhook,
                args=(self.queue.get_job(job_id),),
                daemon=True,
            ).start()
            cm.cleanup()
        finally:
            _cleanup_memory()

    def _write_outputs(self, job, result, cm: CheckpointManager):
        from whisperapp.formatters import write_formats
        from whisperapp.config import Config
        self.queue.update_progress(
            job["id"], STAGES["saving"]["start"],
            STAGES["saving"]["label"])
        cfg = Config()
        write_formats(result, job["file_path"], job["output_path"], job["formats"], cfg)
