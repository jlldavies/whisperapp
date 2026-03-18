import gradio as gr
from pathlib import Path
from whisperapp.queue import JobQueue, JobStatus
from whisperapp.sanitise import sanitise_file_path, sanitise_output_path, sanitise_model
from whisperapp.speakers import extract_speaker_snippets, apply_speaker_names

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

def create_ui(queue: JobQueue, worker) -> gr.Blocks:
    with gr.Blocks(title="WhisperApp") as demo:
        gr.Markdown("# WhisperApp")

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

    return demo
