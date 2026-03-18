import threading
import time
from pathlib import Path
import whisperx
from whisperx.diarize import DiarizationPipeline
from whisperapp.queue import JobQueue, JobStatus
from whisperapp.checkpoints import CheckpointManager

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

        self.queue.update_progress(job_id, 0, "starting")
        cm = CheckpointManager(job["output_path"], job_id)

        try:
            # --- Stage 1: Transcription ---
            if not cm.has("transcription"):
                self.queue.update_progress(job_id, 10, "transcribing")
                model = whisperx.load_model(job["model"], device="cpu",
                                             compute_type="float32")
                audio = whisperx.load_audio(job["file_path"])
                result = model.transcribe(audio, batch_size=8)
                cm.save("transcription", result)
                del model
            else:
                result = cm.load("transcription")

            # Check cancellation between stages
            if self.queue.get_job(job_id)["status"] == JobStatus.CANCELLED:
                return

            # --- Stage 2: Alignment ---
            if not cm.has("alignment"):
                self.queue.update_progress(job_id, 40, "aligning")
                align_model, metadata = whisperx.load_align_model(
                    language_code=result["language"], device="cpu")
                aligned = whisperx.align(result["segments"], align_model,
                                         metadata, job["file_path"], device="cpu")
                cm.save("alignment", aligned)
                del align_model
            else:
                aligned = cm.load("alignment")

            if self.queue.get_job(job_id)["status"] == JobStatus.CANCELLED:
                return

            # --- Stage 3: Diarization ---
            if job["diarize"] and not cm.has("diarization"):
                self.queue.update_progress(job_id, 65, "diarizing")
                diarize_model = DiarizationPipeline(
                    token=self.hf_token, device="cpu")
                diarize_segments = diarize_model(job["file_path"])
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

            # --- Stage 4: Speaker review (if diarized) ---
            if job["diarize"]:
                cm.save("speaker_review", final_result)
                self.queue.update_progress(job_id, 85, "speaker_review")
                self.queue.set_status(job_id, JobStatus.SPEAKER_REVIEW)
                # UI will handle renaming and call complete_with_result
            else:
                self._write_outputs(job, final_result, cm)
                self.queue.complete_job(job_id, job["output_path"])

        except Exception as e:
            self.queue.set_status(job_id, JobStatus.FAILED, error=str(e))
            raise

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
        self.queue.update_progress(job["id"], 95, "saving")
        write_formats(result, job["file_path"], job["output_path"], job["formats"])
