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


def on_audio_chunk(audio_tuple, state):
    """Handle a streaming audio chunk from the microphone.

    *audio_tuple* is (sample_rate, np.ndarray) from Gradio streaming audio.
    *state* is a dict carrying the StreamingEngine across calls.
    """
    if audio_tuple is None:
        return state.get("transcript", ""), state

    sr, audio_array = audio_tuple

    # Lazy-init engine on first chunk
    if "engine" not in state:
        from whisperapp.config import Config
        cfg = Config()
        from whisperapp.streaming import StreamingEngine
        state["engine"] = StreamingEngine(
            model_size=cfg.streaming_model,
            max_chunk_sec=cfg.streaming_max_chunk_sec,
            silence_threshold_sec=cfg.vad_silence_threshold,
        )
        state["engine"].start()
        state["transcript"] = ""

    import numpy as np
    audio_array = audio_array.astype(np.float32)
    # Normalise int16 range to [-1, 1] if needed
    if audio_array.max() > 1.0 or audio_array.min() < -1.0:
        audio_array = audio_array / 32768.0

    new_text = state["engine"].process_chunk(sr, audio_array)
    if new_text:
        state["transcript"] = state["engine"].get_transcript()

    return state.get("transcript", ""), state


def on_stop_save(state, output_path, formats):
    """Stop streaming and save the transcript."""
    engine = state.get("engine")
    if engine is None:
        return "No active session.", state

    result = engine.stop()
    text = result["text"]

    if not text.strip():
        return "No speech detected.", state

    # Save using formatters
    from whisperapp.formatters import write_formats
    from whisperapp.sanitise import sanitise_output_path as sop
    try:
        out_dir = sop(output_path or str(Path.home() / "Downloads"))
    except ValueError as e:
        return f"Error: {e}", state

    # Build a WhisperX-compatible result dict for formatters
    fmt_result = {"segments": result["segments"]}
    fmt_list = formats if formats else ["txt"]
    write_formats(fmt_result, "live_recording", str(out_dir), fmt_list)

    # Store result for potential Polish step
    state["last_result"] = result
    state["output_path"] = str(out_dir)
    state["formats"] = fmt_list

    return f"Saved to {out_dir}", state


def on_polish(state):
    """Run WhisperX alignment + diarization on the recorded audio."""
    engine = state.get("engine")
    if engine is None:
        return "No active session.", state

    from whisperapp.config import Config
    cfg = Config()
    if not cfg.hf_token:
        return "HuggingFace token required for Polish (set in Settings tab).", state

    polished = engine.polish(cfg.hf_token)

    # Update saved files if we have output info
    if "output_path" in state:
        from whisperapp.formatters import write_formats
        fmt_list = state.get("formats", ["txt"])
        write_formats(polished, "live_recording", state["output_path"], fmt_list)

    # Build display text from polished segments
    lines = []
    for seg in polished.get("segments", []):
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        prefix = f"[{speaker}] " if speaker else ""
        lines.append(f"{prefix}{text}")

    transcript = "\n".join(lines) if lines else "Polish produced no output."
    state["transcript"] = transcript
    return transcript, state


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
            live_state = gr.State(value={})

            mic_input = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="Microphone",
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
                stop_save_btn = gr.Button("Stop & Save", variant="primary")
                polish_btn = gr.Button("Polish (Align + Diarize)")

            live_status = gr.Textbox(label="Status", interactive=False)

            mic_input.stream(
                fn=on_audio_chunk,
                inputs=[mic_input, live_state],
                outputs=[live_transcript, live_state],
            )

            stop_save_btn.click(
                fn=on_stop_save,
                inputs=[live_state, live_output_path, live_formats],
                outputs=[live_status, live_state],
            )

            polish_btn.click(
                fn=on_polish,
                inputs=[live_state],
                outputs=[live_transcript, live_state],
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
