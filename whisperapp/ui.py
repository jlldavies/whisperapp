import gradio as gr
from pathlib import Path
from whisperapp.queue import JobQueue, JobStatus
from whisperapp.sanitise import sanitise_file_path, sanitise_output_path, sanitise_model

def handle_submit(queue, file_path, output_path, model, diarize, formats):
    try:
        fp = sanitise_file_path(file_path)
        op = sanitise_output_path(output_path or str(Path.home() / "Downloads"))
        m = sanitise_model(model)
    except ValueError as e:
        return f"Error: {e}"
    job_id = queue.create_job(str(fp), str(op), m, diarize, formats)
    return job_id

def get_queue_status(queue):
    jobs = queue.list_jobs(limit=20)
    if not jobs:
        return "No jobs yet."
    rows = []
    for j in jobs:
        bar = "\u2588" * (j["progress"] // 10) + "\u2591" * (10 - j["progress"] // 10)
        rows.append(
            f"{j['file_name'][:40]:40s}  [{bar}] {j['progress']:3d}%  "
            f"{j.get('stage', ''):20s}  {j['status']}"
        )
    return "\n".join(rows)


def _list_input_devices():
    """Return dict of {display_name: device_index} for audio input devices."""
    import pyaudio
    p = pyaudio.PyAudio()
    devices = {}
    seen = set()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            name = info["name"]
            # Deduplicate by name, prefer higher sample rate
            if name not in seen:
                seen.add(name)
                sr = int(info["defaultSampleRate"])
                label = f"{name} ({sr}Hz)"
                devices[label] = (i, sr, info["maxInputChannels"])
    p.terminate()
    return devices


# Global capture state shared between UI callbacks and capture thread
_capture_lock = None
_capture_thread = None
_capture_running = False
_capture_engine = None
_capture_transcript = ""


def _capture_loop(device_index, sample_rate, channels, model_size, max_chunk_sec, vad_threshold):
    """Background thread: capture audio from PyAudio device and feed to StreamingEngine."""
    import pyaudio
    import numpy as np
    import logging
    global _capture_running, _capture_engine, _capture_transcript

    log = logging.getLogger("whisperapp.live")
    from whisperapp.streaming import StreamingEngine

    engine = StreamingEngine(
        model_size=model_size,
        max_chunk_sec=max_chunk_sec,
        silence_threshold_sec=vad_threshold,
    )
    engine.start()
    _capture_engine = engine

    p = pyaudio.PyAudio()
    chunk_samples = int(sample_rate * 0.5)  # 500ms chunks

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_samples,
        )
        log.warning("Capture started: device=%d sr=%d", device_index, sample_rate)

        while _capture_running:
            data = stream.read(chunk_samples, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            log.warning("CAPTURE: samples=%d min=%.4f max=%.4f", len(audio), audio.min(), audio.max())

            new_text = engine.process_chunk(sample_rate, audio)
            if new_text:
                # Each utterance on a new line for visual speaker separation
                if _capture_transcript:
                    _capture_transcript += "\n" + new_text.strip()
                else:
                    _capture_transcript = new_text.strip()

        stream.stop_stream()
        stream.close()
    except Exception as e:
        log.error("Capture error: %s", e)
    finally:
        p.terminate()
        log.warning("Capture stopped")




def create_ui(queue: JobQueue, worker) -> gr.Blocks:
    with gr.Blocks(title="WhisperApp") as demo:
        gr.Markdown("# WhisperApp")

        with gr.Tab("Transcribe"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Audio/Video File",
                                          file_types=[".mp4",".mp3",".wav",".m4a",
                                                      ".ogg",".flac",".webm",".mkv"])
                    output_path = gr.Textbox(
                        label="Output Path",
                        value=str(Path.home() / "Downloads"),
                        placeholder="Leave blank for ~/Downloads")
                    model_select = gr.Dropdown(
                        choices=["tiny","base","small","medium","large-v2"],
                        value="large-v2", label="Model")
                    diarize_check = gr.Checkbox(value=True, label="Speaker Diarization")
                    formats_check = gr.CheckboxGroup(
                        choices=["txt","srt","vtt","json","tsv"],
                        value=["txt","srt","vtt","json"],
                        label="Output Formats")
                    submit_btn = gr.Button("Transcribe", variant="primary")

                with gr.Column():
                    status_out = gr.Textbox(label="Queue", lines=15,
                                             interactive=False)
                    refresh_btn = gr.Button("Refresh")

            submit_btn.click(
                fn=lambda f, o, m, d, fmt: handle_submit(
                    queue, f.name if f else "", o, m, d, fmt),
                inputs=[file_input, output_path, model_select,
                        diarize_check, formats_check],
                outputs=status_out
            )

            refresh_btn.click(
                fn=lambda: get_queue_status(queue),
                outputs=status_out
            )

        with gr.Tab("Speaker Review"):
            review_status = gr.Textbox(label="Status", interactive=False)
            review_job_id = gr.Textbox(label="Job ID", interactive=False)
            speaker_inputs = gr.State(value={})  # {speaker_label: gr component}

            # Container for dynamic speaker name fields
            speaker_names_json = gr.JSON(label="Speaker Snippets", visible=True)
            speaker_name_entries = gr.Textbox(
                label="Speaker Names (one per line: SPEAKER_00=Name)",
                lines=8, placeholder="SPEAKER_00=Alice\nSPECKER_01=Bob")

            with gr.Row():
                skip_btn = gr.Button("Skip (use default labels)")
                confirm_btn = gr.Button("Confirm Names", variant="primary")
                refresh_review_btn = gr.Button("Check for Reviews")

            def check_for_reviews(q):
                jobs = q.list_jobs(status_filter="speaker_review", limit=1)
                if not jobs:
                    return "No jobs awaiting speaker review.", "", {}, ""
                job = jobs[0]
                from whisperapp.checkpoints import CheckpointManager as CM
                from whisperapp.speakers import extract_speaker_snippets as ess
                cm = CM(job["output_path"], job["id"])
                result = cm.load("speaker_review")
                snippets = ess(result)
                prefill = "\n".join(f"{spk}=" for spk in sorted(snippets))
                return (f"Review needed: {job['file_name']}", job["id"],
                        snippets, prefill)

            def do_confirm(w, q, job_id, name_text):
                if not job_id:
                    return "No job selected."
                names = {}
                for line in name_text.strip().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if k and v:
                            names[k] = v
                from whisperapp.checkpoints import CheckpointManager as CM
                from whisperapp.speakers import apply_speaker_names as asn
                job = q.get_job(job_id)
                cm = CM(job["output_path"], job_id)
                result = cm.load("speaker_review")
                renamed = asn(result, names)
                w.complete_with_result(job_id, renamed)
                return f"Job {job_id[:8]} completed with renamed speakers."

            def do_skip(w, q, job_id):
                if not job_id:
                    return "No job selected."
                from whisperapp.checkpoints import CheckpointManager as CM
                job = q.get_job(job_id)
                cm = CM(job["output_path"], job_id)
                result = cm.load("speaker_review")
                w.complete_with_result(job_id, result)
                return f"Job {job_id[:8]} completed with default speaker labels."

            refresh_review_btn.click(
                fn=lambda: check_for_reviews(queue),
                outputs=[review_status, review_job_id,
                         speaker_names_json, speaker_name_entries]
            )
            confirm_btn.click(
                fn=lambda jid, names: do_confirm(worker, queue, jid, names),
                inputs=[review_job_id, speaker_name_entries],
                outputs=review_status
            )
            skip_btn.click(
                fn=lambda jid: do_skip(worker, queue, jid),
                inputs=[review_job_id],
                outputs=review_status
            )

        with gr.Tab("Live"):
            # Build device list
            devices = _list_input_devices()
            device_names = list(devices.keys())
            # Try to default to Wave Link Stream
            default_device = next((n for n in device_names if "Wave Link Stream" in n), device_names[0] if device_names else "")

            with gr.Row():
                device_select = gr.Dropdown(
                    choices=device_names,
                    value=default_device,
                    label="Audio Input Device",
                    scale=3,
                )
                recording_indicator = gr.Textbox(
                    value="",
                    label="Status",
                    interactive=False,
                    scale=1,
                )

            live_transcript = gr.Textbox(
                label="Live Transcript",
                lines=20,
                interactive=False,
            )

            with gr.Row():
                live_model = gr.Dropdown(
                    choices=["tiny", "base", "small"],
                    value="base",
                    label="Streaming Model",
                )
                live_output_path = gr.Textbox(
                    label="Output Path",
                    value=str(Path.home() / "Downloads"),
                )
                live_formats = gr.CheckboxGroup(
                    choices=["txt", "srt", "vtt", "json", "tsv"],
                    value=["txt"],
                    label="Save Formats",
                )

            with gr.Row():
                start_btn = gr.Button("Start Recording", variant="primary")
                stop_save_btn = gr.Button("Stop & Save", variant="stop")
                clear_btn = gr.Button("Clear")
                polish_btn = gr.Button("Polish (Align + Diarize)")

            live_status = gr.Textbox(label="Info", interactive=False)

            # Poll timer to update transcript from capture thread
            poll_timer = gr.Timer(value=1.0, active=False)

            def start_capture(device_name, model):
                import threading
                global _capture_thread, _capture_running, _capture_transcript
                if device_name not in devices:
                    return "Device not found.", "", gr.Timer(active=False)
                dev_idx, dev_sr, dev_ch = devices[device_name]

                from whisperapp.config import Config
                cfg = Config()

                _capture_running = True
                _capture_transcript = ""
                _capture_thread = threading.Thread(
                    target=_capture_loop,
                    args=(dev_idx, dev_sr, dev_ch, model,
                          cfg.streaming_max_chunk_sec, cfg.vad_silence_threshold),
                    daemon=True,
                )
                _capture_thread.start()
                return f"Recording from: {device_name}", "RECORDING", gr.Timer(active=True)

            def poll_transcript():
                import time
                elapsed = ""
                if _capture_running:
                    elapsed = "RECORDING"
                return _capture_transcript, elapsed

            def clear_transcript():
                global _capture_transcript, _capture_engine
                _capture_transcript = ""
                if _capture_engine:
                    _capture_engine.reset()
                return "", ""

            def stop_and_save(output_path, formats):
                global _capture_running, _capture_engine, _capture_transcript
                _capture_running = False
                if _capture_thread:
                    _capture_thread.join(timeout=3)

                if _capture_engine is None:
                    return "No active session.", "", gr.Timer(active=False)

                result = _capture_engine.stop()
                text = result["text"]

                if not text.strip():
                    return "No speech detected.", "", gr.Timer(active=False)

                from whisperapp.formatters import write_formats
                from whisperapp.sanitise import sanitise_output_path as sop
                try:
                    out_dir = sop(output_path or str(Path.home() / "Downloads"))
                except ValueError as e:
                    return f"Error: {e}", "", gr.Timer(active=False)

                fmt_result = {"segments": result["segments"]}
                fmt_list = formats if formats else ["txt"]
                write_formats(fmt_result, "live_recording", str(out_dir), fmt_list)
                return f"Saved to {out_dir}", "", gr.Timer(active=False)

            def do_polish():
                global _capture_engine
                if _capture_engine is None:
                    return "No active session."
                from whisperapp.config import Config
                cfg = Config()
                if not cfg.hf_token:
                    return "HuggingFace token required (set in Settings tab)."
                polished = _capture_engine.polish(cfg.hf_token)
                lines = []
                for seg in polished.get("segments", []):
                    speaker = seg.get("speaker", "")
                    text = seg.get("text", "").strip()
                    prefix = f"[{speaker}] " if speaker else ""
                    lines.append(f"{prefix}{text}")
                return "\n".join(lines) if lines else "Polish produced no output."

            start_btn.click(
                fn=start_capture,
                inputs=[device_select, live_model],
                outputs=[live_status, recording_indicator, poll_timer],
            )

            poll_timer.tick(
                fn=poll_transcript,
                outputs=[live_transcript, recording_indicator],
            )

            stop_save_btn.click(
                fn=stop_and_save,
                inputs=[live_output_path, live_formats],
                outputs=[live_status, recording_indicator, poll_timer],
            )

            clear_btn.click(
                fn=clear_transcript,
                outputs=[live_transcript, live_status],
            )

            polish_btn.click(
                fn=do_polish,
                outputs=[live_transcript],
            )

        with gr.Tab("Settings"):
            from whisperapp.config import Config
            cfg = Config()
            hf_token_input = gr.Textbox(
                label="HuggingFace Token",
                value=cfg.hf_token,
                type="password")
            default_model = gr.Dropdown(
                choices=["tiny","base","small","medium","large-v2"],
                value=cfg.default_model, label="Default Model")
            startup_check = gr.Checkbox(
                label="Start on login", value=True)
            save_btn = gr.Button("Save Settings")

            def save_settings(token, model, startup):
                from whisperapp.config import Config as Cfg
                from whisperapp.startup import register_startup, unregister_startup
                c = Cfg()
                c.hf_token = token
                c.default_model = model
                c.save()
                if startup:
                    register_startup()
                else:
                    unregister_startup()
                return "Settings saved"

            settings_out = gr.Textbox(label="", interactive=False)
            save_btn.click(
                fn=save_settings,
                inputs=[hf_token_input, default_model, startup_check],
                outputs=settings_out
            )

    return demo
