import threading
import time
from pathlib import Path
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from whisperapp.queue import JobQueue, JobStatus
from whisperapp.checkpoints import CheckpointManager

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_COMPUTE_TYPE = "float16" if _DEVICE == "cuda" else "float32"

# ---------------------------------------------------------------------------
# Stage definitions — each stage owns a slice of the 0-100 progress bar.
# ---------------------------------------------------------------------------
STAGES = {
    "loading_model":  {"label": "1/5 Loading model",      "start":  0, "end":  5},
    "loading_audio":  {"label": "1/5 Loading audio",       "start":  5, "end":  8},
    "transcribing":   {"label": "2/5 Transcribing",        "start":  8, "end": 40},
    "aligning":       {"label": "3/5 Aligning words",      "start": 40, "end": 60},
    "diarizing":      {"label": "4/5 Identifying speakers", "start": 60, "end": 85},
    "speaker_review": {"label": "5/5 Speaker review",      "start": 85, "end": 90},
    "saving":         {"label": "5/5 Saving files",        "start": 90, "end": 100},
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

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        while not self._stop_event.is_set():
            job = self.queue.next_queued()
            if job:
                self.process_job(job["id"])
            else:
                time.sleep(2)

    def process_job(self, job_id: str):
        job = self.queue.get_job(job_id)
        if not job or job["status"] == JobStatus.CANCELLED:
            return

        cm = CheckpointManager(job["output_path"], job_id)
        hb = _Heartbeat(self.queue, job_id, "loading_model", interval=5.0).start()

        try:
            # --- Stage 1a: Load model ---
            if not cm.has("transcription"):
                hb.update("loading_model")
                model = whisperx.load_model(job["model"], device=_DEVICE,
                                             compute_type=_COMPUTE_TYPE)

                # --- Stage 1b: Load audio ---
                hb.update("loading_audio")
                audio = whisperx.load_audio(job["file_path"])

                # --- Stage 2: Transcribe (with live progress) ---
                hb.update("transcribing")
                model.transcribe = _patched_transcribe(
                    model.transcribe, self.queue, job_id, hb)
                result = model.transcribe(audio, batch_size=8)

                cm.save("transcription", result)
                del model
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
                    language_code=result["language"], device=_DEVICE)

                # Capture print_progress output to update DB per-segment
                aligned = _tracked_align(
                    whisperx.align, result["segments"], align_model,
                    metadata, job["file_path"], _DEVICE,
                    self.queue, job_id, hb)

                cm.save("alignment", aligned)
                del align_model
            else:
                aligned = cm.load("alignment")

            if self.queue.get_job(job_id)["status"] == JobStatus.CANCELLED:
                return

            # --- Stage 4: Diarization (with progress via pyannote hooks) ---
            if job["diarize"] and not cm.has("diarization"):
                hb.update("diarizing")
                diarize_model = DiarizationPipeline(
                    token=self.hf_token, device=_DEVICE)

                # Hook into pyannote's pipeline progress
                _install_diarize_hook(
                    diarize_model, self.queue, job_id, hb)
                diarize_segments = diarize_model(job["file_path"])

                self.queue.update_progress(
                    job_id, _stage_progress("diarizing", 0.9),
                    f"{STAGES['diarizing']['label']} — assigning speakers")
                result_diarized = whisperx.assign_word_speakers(
                    diarize_segments, aligned)
                cm.save("diarization", result_diarized)
                final_result = result_diarized
            elif cm.has("diarization"):
                final_result = cm.load("diarization")
            else:
                final_result = aligned

            if self.queue.get_job(job_id)["status"] == JobStatus.CANCELLED:
                return

            # --- Stage 5: Speaker review or save ---
            if job["diarize"]:
                cm.save("speaker_review", final_result)
                hb.update("speaker_review")
                self.queue.set_status(job_id, JobStatus.SPEAKER_REVIEW)
            else:
                self._write_outputs(job, final_result, cm)
                self.queue.complete_job(job_id, job["output_path"])

        except Exception as e:
            self.queue.set_status(job_id, JobStatus.FAILED, error=str(e))
            raise
        finally:
            hb.stop()

    def complete_with_result(self, job_id: str, renamed_segments: dict):
        """Called after speaker review is complete (names applied or skipped)."""
        job = self.queue.get_job(job_id)
        cm = CheckpointManager(job["output_path"], job_id)
        cm.save("saving", renamed_segments)
        self._write_outputs(job, renamed_segments, cm)
        self.queue.complete_job(job_id, job["output_path"])
        cm.cleanup()

    def _write_outputs(self, job, result, cm: CheckpointManager):
        from whisperapp.formatters import write_formats
        self.queue.update_progress(
            job["id"], STAGES["saving"]["start"],
            STAGES["saving"]["label"])
        write_formats(result, job["file_path"], job["output_path"], job["formats"])
