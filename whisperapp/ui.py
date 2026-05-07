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
        pct = j["progress"]
        filled = pct // 5  # 20-char bar for finer granularity
        bar = "\u2588" * filled + "\u2591" * (20 - filled)
        stage = j.get("stage", "") or j["status"]
        rows.append(
            f"{j['file_name'][:40]:40s}\n"
            f"  [{bar}] {pct:3d}%  {stage}"
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


def list_audio_devices() -> list[dict]:
    """Return list of audio input devices. Returns [] if pyaudio unavailable."""
    try:
        import pyaudio
    except ImportError:
        return []
    p = pyaudio.PyAudio()
    devices = []
    seen = set()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and info["name"] not in seen:
            seen.add(info["name"])
            devices.append({
                "name": info["name"],
                "index": i,
                "sample_rate": int(info["defaultSampleRate"]),
                "channels": int(info["maxInputChannels"]),
            })
    p.terminate()
    return devices


def _monitor_all_devices(duration_sec=2.0):
    """Sample every input device and return a markdown table of RMS signal levels."""
    import pyaudio
    import numpy as np

    devices = _list_input_devices()
    if not devices:
        return "No input devices found."

    p = pyaudio.PyAudio()
    rows = []
    for label, (dev_idx, sr, _channels) in devices.items():
        try:
            chunk = int(sr * duration_sec)
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sr,
                input=True,
                input_device_index=dev_idx,
                frames_per_buffer=chunk,
            )
            data = stream.read(chunk, exception_on_overflow=False)
            stream.stop_stream()
            stream.close()
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio ** 2)))
            peak = float(np.abs(audio).max())
            bar = "#" * min(int(rms * 400), 30)
            if rms > 0.005:
                signal = "** SIGNAL **"
            elif rms > 0.0005:
                signal = "low / noise"
            else:
                signal = "silent"
            rows.append(f"| {label} | {rms:.4f} | {peak:.4f} | {bar:<30} | {signal} |")
        except Exception as e:
            rows.append(f"| {label} | ERROR | — | — | {e} |")

    p.terminate()
    header = "| Device | RMS | Peak | Level | Signal? |\n|--------|-----|------|-------|---------|"
    return header + "\n" + "\n".join(rows)


# Global capture state shared between UI callbacks and capture thread
_capture_lock = None
_capture_thread = None
_capture_running = False
_capture_engine = None
_capture_transcript = ""
_capture_session_file = None  # Path to the live session .txt file being appended


def _capture_loop(device_index, sample_rate, channels, model_size, max_chunk_sec, vad_threshold):
    """Background thread: capture audio from PyAudio device and feed to StreamingEngine."""
    import pyaudio
    import numpy as np
    import logging
    global _capture_running, _capture_engine, _capture_transcript, _capture_session_file

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
                utterance = new_text.strip()
                if _capture_transcript:
                    _capture_transcript += "\n" + utterance
                else:
                    _capture_transcript = utterance
                # Persist immediately so nothing is lost on crash/shutdown
                if _capture_session_file:
                    try:
                        with open(_capture_session_file, "a", encoding="utf-8") as fh:
                            fh.write(utterance + "\n")
                    except Exception:
                        pass

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
                    status_out = gr.Textbox(label="Queue", lines=10,
                                             interactive=False)
                    with gr.Row():
                        refresh_btn = gr.Button("Refresh")
                        clear_queue_btn = gr.Button("Clear Queue", variant="stop")
                    with gr.Row():
                        meeting_notes_job_id = gr.Textbox(
                            label="Job ID for meeting notes",
                            placeholder="Paste a completed job ID",
                            scale=3,
                        )
                        meeting_notes_btn = gr.Button("Generate Meeting Notes (AI)", scale=1)
                    meeting_notes_out = gr.Textbox(
                        label="Meeting Notes", lines=10, interactive=False)

                    # Inline speaker review panel (appears when a job needs review)
                    review_panel = gr.Group(visible=False)
                    with review_panel:
                        gr.Markdown("### Speaker Review")
                        inline_review_status = gr.Textbox(
                            label="", interactive=False, lines=1)
                        inline_review_job_id = gr.Textbox(
                            visible=False, interactive=False)
                        inline_snippets = gr.JSON(
                            label="Speaker Snippets", visible=True)
                        inline_names = gr.Textbox(
                            label="Speaker Names (one per line: SPEAKER_00=Name)",
                            lines=4,
                            placeholder="SPEAKER_00=Alice\nSPEAKER_01=Bob")
                        with gr.Row():
                            inline_ai_btn = gr.Button("Auto-identify (AI)")
                            inline_skip_btn = gr.Button("Skip (use default labels)")
                            inline_confirm_btn = gr.Button(
                                "Confirm Names", variant="primary")

            def _poll_queue_and_review(q):
                """Return queue status + show/hide review panel."""
                queue_text = get_queue_status(q)
                # Check for jobs awaiting speaker review
                review_jobs = q.list_jobs(
                    status_filter="speaker_review", limit=1)
                if review_jobs:
                    job = review_jobs[0]
                    from whisperapp.checkpoints import CheckpointManager as CM
                    from whisperapp.speakers import extract_speaker_snippets as ess
                    cm = CM(job["output_path"], job["id"])
                    try:
                        result = cm.load("speaker_review")
                        snippets = ess(result)
                    except Exception:
                        snippets = {}
                    prefill = "\n".join(
                        f"{spk}=" for spk in sorted(snippets))
                    return (queue_text,
                            gr.Group(visible=True),
                            f"Review needed: {job['file_name']}",
                            job["id"], snippets, prefill)
                return (queue_text,
                        gr.Group(visible=False),
                        "", "", {}, "")

            def _inline_confirm(w, q, job_id, name_text):
                if not job_id:
                    return get_queue_status(q), gr.Group(visible=False), "No job selected.", "", {}, ""
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
                return (get_queue_status(q), gr.Group(visible=False),
                        f"Done — saved with renamed speakers.", "", {}, "")

            def _inline_skip(w, q, job_id):
                if not job_id:
                    return get_queue_status(q), gr.Group(visible=False), "No job selected.", "", {}, ""
                from whisperapp.checkpoints import CheckpointManager as CM
                job = q.get_job(job_id)
                cm = CM(job["output_path"], job_id)
                result = cm.load("speaker_review")
                w.complete_with_result(job_id, result)
                return (get_queue_status(q), gr.Group(visible=False),
                        f"Done — saved with default labels.", "", {}, "")

            _review_outputs = [status_out, review_panel,
                               inline_review_status, inline_review_job_id,
                               inline_snippets, inline_names]

            # Auto-poll queue status every 5 seconds
            queue_timer = gr.Timer(value=5.0, active=True)
            queue_timer.tick(
                fn=lambda: _poll_queue_and_review(queue),
                outputs=_review_outputs
            )

            def _inline_ai_identify(job_id):
                """Use AI provider to suggest speaker names and pre-fill the text box."""
                if not job_id:
                    return "No job selected.", ""
                from whisperapp.config import Config
                from whisperapp.ai import make_provider
                from whisperapp.checkpoints import CheckpointManager as CM
                from whisperapp.speakers import extract_speaker_snippets as ess
                cfg = Config()
                ai = make_provider(cfg.ai_provider, cfg.ai_api_key,
                                   cfg.ai_model, cfg.ai_base_url)
                if not ai.is_available():
                    return "No AI provider configured — set one in Settings.", ""
                job = queue.get_job(job_id)
                if not job:
                    return "Job not found.", ""
                cm = CM(job["output_path"], job_id)
                result = cm.load("speaker_review")
                snippets = ess(result)
                mapping = ai.identify_speakers(snippets)
                if not mapping:
                    return "AI could not identify speakers.", ""
                prefill = "\n".join(
                    f"{label}={name}" for label, name in sorted(mapping.items()))
                return f"AI suggestions ({ai.name}):", prefill

            inline_ai_btn.click(
                fn=lambda jid: _inline_ai_identify(jid),
                inputs=[inline_review_job_id],
                outputs=[inline_review_status, inline_names],
            )
            inline_confirm_btn.click(
                fn=lambda jid, names: _inline_confirm(
                    worker, queue, jid, names),
                inputs=[inline_review_job_id, inline_names],
                outputs=_review_outputs
            )
            inline_skip_btn.click(
                fn=lambda jid: _inline_skip(worker, queue, jid),
                inputs=[inline_review_job_id],
                outputs=_review_outputs
            )

            submit_btn.click(
                fn=lambda f, o, m, d, fmt: handle_submit(
                    queue, f.name if f else "", o, m, d, fmt),
                inputs=[file_input, output_path, model_select,
                        diarize_check, formats_check],
                outputs=status_out
            )

            refresh_btn.click(
                fn=lambda: _poll_queue_and_review(queue),
                outputs=_review_outputs
            )

            def clear_all_jobs(q):
                """Cancel running jobs and remove all from queue."""
                jobs = q.list_jobs(limit=100)
                for j in jobs:
                    if j["status"] in ("queued", "running"):
                        q.cancel_job(j["id"])
                q.clear_completed()
                return _poll_queue_and_review(q)

            clear_queue_btn.click(
                fn=lambda: clear_all_jobs(queue),
                outputs=_review_outputs
            )

            def generate_meeting_notes(job_id):
                if not job_id.strip():
                    return "Enter a job ID."
                from whisperapp.config import Config
                from whisperapp.ai import make_provider
                from whisperapp.sanitise import sanitise_job_id
                try:
                    sanitise_job_id(job_id.strip())
                except ValueError:
                    return "Invalid job ID format."
                cfg = Config()
                ai = make_provider(cfg.ai_provider, cfg.ai_api_key,
                                   cfg.ai_model, cfg.ai_base_url)
                if not ai.is_available():
                    return "No AI provider configured — set one in the Settings tab."
                job = queue.get_job(job_id.strip())
                if not job:
                    return "Job not found."
                if job["status"] != "done":
                    return f"Job is not done yet (status: {job['status']})."
                from pathlib import Path as _Path
                txt_file = _Path(job["output_path"]) / f"{_Path(job['file_path']).stem}.txt"
                if not txt_file.exists():
                    return "No .txt transcript found for this job."
                transcript = txt_file.read_text(encoding="utf-8")
                notes = ai.meeting_notes(transcript)
                if not notes:
                    return "AI returned an empty response."
                notes_file = txt_file.with_suffix(".notes.md")
                notes_file.write_text(notes, encoding="utf-8")
                return f"Saved to {notes_file}\n\n{notes}"

            meeting_notes_btn.click(
                fn=generate_meeting_notes,
                inputs=[meeting_notes_job_id],
                outputs=meeting_notes_out,
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

            with gr.Row():
                check_levels_btn = gr.Button("Check Levels (2s sample all devices)")
                with gr.Column():
                    gr.Markdown(
                        "_Samples every input device for 2 seconds. "
                        "Look for **SIGNAL** on the device carrying your audio. "
                        "Wave Link exposes Stream Mix, Local Mix, and per-app channels — "
                        "pick whichever shows signal._"
                    )
            levels_out = gr.Markdown(value="")

            check_levels_btn.click(
                fn=_monitor_all_devices,
                inputs=[],
                outputs=levels_out,
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
                from datetime import datetime
                global _capture_thread, _capture_running, _capture_transcript, _capture_session_file
                if device_name not in devices:
                    return "Device not found.", "", gr.Timer(active=False)
                dev_idx, dev_sr, dev_ch = devices[device_name]

                from whisperapp.config import Config
                cfg = Config()

                # Create a timestamped session file; appended to on every utterance
                session_stem = f"live_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
                session_dir = Path(cfg.default_output_path) if cfg.default_output_path else Path.home() / "Downloads"
                session_dir.mkdir(parents=True, exist_ok=True)
                _capture_session_file = str(session_dir / f"{session_stem}_live.txt")
                try:
                    with open(_capture_session_file, "w", encoding="utf-8") as fh:
                        fh.write(f"# WhisperApp live session — {datetime.now().isoformat()}\n")
                        fh.write(f"# Device: {device_name}\n\n")
                except Exception:
                    _capture_session_file = None

                _capture_running = True
                _capture_transcript = ""
                _capture_thread = threading.Thread(
                    target=_capture_loop,
                    args=(dev_idx, dev_sr, dev_ch, model,
                          cfg.streaming_max_chunk_sec, cfg.vad_silence_threshold),
                    daemon=True,
                )
                _capture_thread.start()
                file_note = f"\nSaving to: {_capture_session_file}" if _capture_session_file else ""
                return f"Recording from: {device_name}{file_note}", "RECORDING", gr.Timer(active=True)

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
                from datetime import datetime
                stem = f"live_recording_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
                write_formats(fmt_result, stem, str(out_dir), fmt_list)
                return f"Saved {stem} to {out_dir}", "", gr.Timer(active=False)

            def do_polish(current_transcript):
                global _capture_engine
                if _capture_engine is None:
                    yield current_transcript, "No active session."
                    return
                from whisperapp.config import Config
                cfg = Config()
                if not cfg.hf_token:
                    yield current_transcript, "HuggingFace token required (set in Settings tab)."
                    return

                import threading
                progress_state = {"stage": "", "detail": ""}
                def on_progress(stage, detail):
                    progress_state["stage"] = stage
                    progress_state["detail"] = detail

                result_holder = {"result": None, "error": None}
                def run_polish():
                    try:
                        result_holder["result"] = _capture_engine.polish(cfg.hf_token, on_progress=on_progress)
                    except Exception as e:
                        result_holder["error"] = str(e)

                t = threading.Thread(target=run_polish, daemon=True)
                t.start()

                import time
                stages = {"preparing": "1/4", "transcribing": "2/4", "aligning": "3/4", "diarizing": "4/4", "complete": ""}
                while t.is_alive():
                    stage = progress_state["stage"]
                    detail = progress_state["detail"]
                    step = stages.get(stage, "")
                    status = f"Polish [{step}] {stage}: {detail}" if stage else "Polish: starting..."
                    yield current_transcript, status
                    time.sleep(0.5)

                if result_holder["error"]:
                    yield current_transcript, f"Polish failed: {result_holder['error']}"
                    return

                polished = result_holder["result"]
                lines = []
                for seg in polished.get("segments", []):
                    speaker = seg.get("speaker", "")
                    text = seg.get("text", "").strip()
                    prefix = f"[{speaker}] " if speaker else ""
                    lines.append(f"{prefix}{text}")
                if lines:
                    yield "\n".join(lines), "Polish complete."
                else:
                    yield current_transcript, "Polish produced no output."

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
                inputs=[live_transcript],
                outputs=[live_transcript, live_status],
            )

        with gr.Tab("Settings"):
            from whisperapp.config import Config
            cfg = Config()

            gr.Markdown("### Transcription")
            hf_token_input = gr.Textbox(
                label="HuggingFace Token",
                value=cfg.hf_token,
                type="password")
            default_model = gr.Dropdown(
                choices=["tiny","base","small","medium","large-v2"],
                value=cfg.default_model, label="Default Model")
            startup_check = gr.Checkbox(
                label="Start on login", value=True)

            gr.Markdown("### AI Features (optional)")
            gr.Markdown(
                "Enable an AI provider to get automatic speaker identification, "
                "meeting notes, and live summaries. All features work without AI — "
                "speaker labels can always be set manually."
            )
            ai_provider_select = gr.Dropdown(
                choices=["none", "claude", "openai", "ollama"],
                value=cfg.ai_provider,
                label="AI Provider",
            )
            ai_api_key_input = gr.Textbox(
                label="API Key (Claude / OpenAI)",
                value=cfg.ai_api_key,
                type="password",
                placeholder="Not needed for Ollama",
            )
            ai_model_input = gr.Textbox(
                label="Model (leave blank for provider default)",
                value=cfg.ai_model,
                placeholder="e.g. gpt-4o, claude-haiku-4-5-20251001, llama3.2",
            )
            ai_base_url_input = gr.Textbox(
                label="Base URL (Ollama or OpenAI-compatible endpoint)",
                value=cfg.ai_base_url,
                placeholder="http://localhost:11434",
            )
            with gr.Row():
                save_btn = gr.Button("Save Settings", variant="primary")
                test_ai_btn = gr.Button("Test AI Connection")

            settings_out = gr.Textbox(label="", interactive=False)

            def save_settings(token, model, startup, ai_prov, ai_key, ai_mod, ai_url):
                from whisperapp.config import Config as Cfg
                from whisperapp.startup import register_startup, unregister_startup
                c = Cfg()
                c.hf_token = token
                c.default_model = model
                c.ai_provider = ai_prov
                c.ai_api_key = ai_key
                c.ai_model = ai_mod
                c.ai_base_url = ai_url
                c.save()
                if startup:
                    register_startup()
                else:
                    unregister_startup()
                return "Settings saved"

            def test_ai_connection(ai_prov, ai_key, ai_mod, ai_url):
                from whisperapp.ai import make_provider
                provider = make_provider(ai_prov, ai_key, ai_mod, ai_url)
                if provider.name == "none":
                    return "No AI provider selected."
                if provider.is_available():
                    return f"{provider.name} — connection OK"
                return f"{provider.name} — not reachable (check key/URL/server)"

            save_btn.click(
                fn=save_settings,
                inputs=[hf_token_input, default_model, startup_check,
                        ai_provider_select, ai_api_key_input,
                        ai_model_input, ai_base_url_input],
                outputs=settings_out,
            )
            test_ai_btn.click(
                fn=test_ai_connection,
                inputs=[ai_provider_select, ai_api_key_input,
                        ai_model_input, ai_base_url_input],
                outputs=settings_out,
            )

    return demo
