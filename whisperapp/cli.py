import click
import requests
from whisperapp.queue import JobQueue
from pathlib import Path

API_BASE = "http://127.0.0.1:7861"

def get_queue():
    return JobQueue()

def _error_detail(r):
    """Best-effort extraction of the server's error detail, for reporting
    a failed AI call without throwing a second error while doing it."""
    try:
        detail = r.json().get("detail")
        if detail:
            return detail
    except (ValueError, AttributeError):
        pass
    return r.text or f"HTTP {r.status_code}"

@click.group()
def cli():
    """WhisperApp - local transcription with speaker diarization."""
    pass

@cli.command()
@click.argument("file_path")
@click.option("--output", "-o", default=None, help="Output directory")
@click.option("--model", "-m", default="large-v2",
              type=click.Choice(["tiny","base","small","medium","large-v2"]))
@click.option("--diarize/--no-diarize", default=True)
@click.option("--formats", "-f", default="txt,srt,vtt,json",
              help="Comma-separated formats")
def transcribe(file_path, output, model, diarize, formats):
    """Submit a file for transcription."""
    payload = {
        "file_path": file_path,
        "output_path": output,
        "model": model,
        "diarize": diarize,
        "formats": formats.split(",")
    }
    try:
        r = requests.post(f"{API_BASE}/transcribe", json=payload, timeout=5)
        r.raise_for_status()
        click.echo(f"Submitted. Job ID: {r.json()['job_id']}")
    except requests.ConnectionError:
        click.echo("Error: WhisperApp is not running. Start it from the system tray.", err=True)
        raise SystemExit(1)

@cli.command()
@click.argument("job_id")
def status(job_id):
    """Check job status."""
    import re
    if not re.match(r'^[0-9a-f-]{36}$', job_id):
        click.echo("Error: invalid job_id format", err=True)
        raise SystemExit(1)
    r = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=5)
    if r.status_code == 404:
        click.echo("Job not found.")
        return
    job = r.json()
    click.echo(f"Status: {job['status']}  Progress: {job['progress']}%  Stage: {job['stage']}")

@cli.command("list")
@click.option("--status-filter", "-s", default=None)
def list_jobs(status_filter):
    """List recent jobs."""
    q = get_queue()
    jobs = q.list_jobs(status_filter=status_filter)
    if not jobs:
        click.echo("No jobs found.")
        return
    for j in jobs:
        click.echo(f"{j['id'][:8]}  {j['file_name']:30s}  {j['status']:12s}  {j['progress']}%")

@cli.command()
@click.argument("job_id")
def cancel(job_id):
    """Cancel a job (marks it cancelled; job stays in history)."""
    r = requests.post(f"{API_BASE}/jobs/{job_id}/cancel", timeout=5)
    r.raise_for_status()
    click.echo("Cancelled.")

@cli.command()
@click.argument("job_id")
def delete(job_id):
    """Delete a job from the queue entirely."""
    r = requests.delete(f"{API_BASE}/jobs/{job_id}", timeout=5)
    if r.status_code == 404:
        click.echo("Job not found.")
        return
    r.raise_for_status()
    click.echo("Deleted.")

@cli.command("move-up")
@click.argument("job_id")
def move_up(job_id):
    """Move a queued job up one position."""
    r = requests.post(f"{API_BASE}/jobs/{job_id}/move-up", timeout=5)
    r.raise_for_status()
    click.echo("Moved up.")

@cli.command("move-down")
@click.argument("job_id")
def move_down(job_id):
    """Move a queued job down one position."""
    r = requests.post(f"{API_BASE}/jobs/{job_id}/move-down", timeout=5)
    r.raise_for_status()
    click.echo("Moved down.")

@cli.command("pause")
def pause_worker():
    """Pause the worker (finishes current job, holds the next one)."""
    r = requests.post(f"{API_BASE}/worker/pause", timeout=5)
    r.raise_for_status()
    click.echo("Worker paused. Use 'resume' to continue.")

@cli.command("resume")
def resume_worker():
    """Resume the worker after a pause."""
    r = requests.post(f"{API_BASE}/worker/resume", timeout=5)
    r.raise_for_status()
    click.echo("Worker resumed.")

@cli.command("worker-status")
def worker_status():
    """Show whether the worker is paused or running."""
    r = requests.get(f"{API_BASE}/worker/status", timeout=5)
    r.raise_for_status()
    state = "paused" if r.json()["paused"] else "running"
    click.echo(f"Worker: {state}")

@cli.command("get")
@click.argument("job_id")
@click.option("--format", "-f", "fmt", default="txt")
def get_transcript(job_id, fmt):
    """Retrieve a completed transcript."""
    r = requests.get(f"{API_BASE}/jobs/{job_id}/transcript",
                     params={"format": fmt}, timeout=5)
    if r.status_code == 404:
        click.echo("Not found.")
        return
    r.raise_for_status()
    click.echo(r.json()["content"])

@cli.command("info")
def info():
    """Show device and acceleration information."""
    try:
        r = requests.get(f"{API_BASE}/info", timeout=5)
        r.raise_for_status()
        d = r.json()
        click.echo(f"Platform:        {d['platform']}")
        click.echo(f"Acceleration:    {d['acceleration']}")
        click.echo(f"Whisper device:  {d['whisper_device']}  ({d['compute_type']})")
        click.echo(f"Diarize device:  {d['diarize_device']}")
        click.echo(f"CUDA:            {d['cuda_available']}")
        click.echo(f"MPS (Apple):     {d['mps_available']}")
        click.echo(f"MLX Whisper:     {d['mlx_whisper_available']}")
    except requests.ConnectionError:
        # Fall back to direct detection when daemon isn't running
        import sys, torch
        from whisperapp.worker import (
            _WHISPER_DEVICE, _DIARIZE_DEVICE, _COMPUTE_TYPE, _has_mlx_whisper
        )
        click.echo(f"Platform:        {sys.platform}  [daemon not running]")
        click.echo(f"Whisper device:  {_WHISPER_DEVICE}  ({_COMPUTE_TYPE})")
        click.echo(f"Diarize device:  {_DIARIZE_DEVICE}")
        click.echo(f"CUDA:            {torch.cuda.is_available()}")
        click.echo(f"MPS (Apple):     {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
        click.echo(f"MLX Whisper:     {_has_mlx_whisper()}")


@cli.command("ai-status")
def ai_status():
    """Show configured AI provider and whether it is reachable."""
    try:
        r = requests.get(f"{API_BASE}/ai/status", timeout=5)
        r.raise_for_status()
        d = r.json()
        available = "OK" if d["available"] else "not reachable"
        click.echo(f"Provider: {d['provider']}  ({available})")
    except requests.ConnectionError:
        # Fall back to reading config directly when daemon isn't running
        from whisperapp.ai import make_provider
        from whisperapp.config import Config
        cfg = Config()
        ai = make_provider(cfg.ai_provider, cfg.ai_api_key, cfg.ai_model, cfg.ai_base_url)
        available = "OK" if ai.is_available() else "not reachable"
        click.echo(f"Provider: {ai.name}  ({available})  [daemon not running]")


def _fetch_segments(job_id, page_size=200):
    """Yield diarized segments page-by-page from the daemon. Empty on errors."""
    offset = 0
    while True:
        try:
            r = requests.get(
                f"{API_BASE}/jobs/{job_id}/segments",
                params={"offset": offset, "limit": page_size},
                timeout=15,
            )
        except requests.ConnectionError:
            return
        if r.status_code != 200:
            return
        d = r.json()
        for seg in d.get("segments", []):
            yield seg
        if d.get("next_offset") is None:
            return
        offset = d["next_offset"]


def _format_ts(seconds):
    seconds = float(seconds or 0)
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


@cli.command("transcript")
@click.argument("job_id")
@click.option("--page", default=1, help="Page number, 1-indexed.")
@click.option("--page-size", default=100, help="Segments per page (max 500).")
@click.option("--all", "show_all", is_flag=True, help="Print every page, no pagination.")
def transcript_cmd(job_id, page, page_size, show_all):
    """Print the diarized transcript for a job (paged).

    Works for jobs in `speaker_review` or `done` state. Use `--all` to dump
    everything in one go (handy for piping to grep/less or feeding to a
    long-context AI tool); without it you get one page at a time.
    """
    try:
        if show_all:
            shown = 0
            for seg in _fetch_segments(job_id, page_size=page_size):
                spk = seg.get("speaker", "")
                prefix = f"[{spk}] " if spk else ""
                click.echo(f"{_format_ts(seg.get('start'))}–{_format_ts(seg.get('end'))} "
                           f"{prefix}{seg.get('text', '').strip()}")
                shown += 1
            if shown == 0:
                click.echo("No segments returned (job not in speaker_review or done?).", err=True)
                raise SystemExit(1)
            return
        offset = max(0, (page - 1) * page_size)
        r = requests.get(
            f"{API_BASE}/jobs/{job_id}/segments",
            params={"offset": offset, "limit": page_size},
            timeout=15,
        )
    except requests.ConnectionError:
        click.echo("Error: WhisperApp is not running.", err=True)
        raise SystemExit(1)
    if r.status_code == 409:
        click.echo(f"Job not in speaker_review or done state.", err=True)
        raise SystemExit(1)
    r.raise_for_status()
    d = r.json()
    total = d.get("total", 0)
    pages = (total + page_size - 1) // page_size if total else 0
    click.echo(f"# job {job_id}  page {page}/{pages}  ({total} segments total)")
    for seg in d.get("segments", []):
        spk = seg.get("speaker", "")
        prefix = f"[{spk}] " if spk else ""
        click.echo(f"{_format_ts(seg.get('start'))}–{_format_ts(seg.get('end'))} "
                   f"{prefix}{seg.get('text', '').strip()}")
    if d.get("next_offset") is not None:
        next_page = page + 1
        click.echo(f"\n# more — re-run with --page={next_page} or use --all")


@cli.command("identify-speakers")
@click.argument("job_id")
@click.option("--context", "-c", default="",
              help="Meeting context to help the AI (e.g. 'Weekly standup: Alice, Bob, Carol').")
@click.option("--full-transcript/--no-full-transcript", default=True,
              help="Include the full diarized transcript in the AI context. "
                   "On by default — gives the LLM real conversational context. "
                   "Disable for fast-and-cheap mode.")
@click.option("--with-snippets", is_flag=True, default=False,
              help="Also fetch /speakers and add the per-speaker representative "
                   "snippets (with audio timestamps) to the AI context.")
@click.option("--max-chars", default=120_000,
              help="Cap the transcript context size (rough character count) so we "
                   "don't blow the model's context window on very long meetings.")
def identify_speakers(job_id, context, full_transcript, with_snippets, max_chars):
    """Use the configured AI provider to suggest speaker names for a job.

    By default this fetches the full diarized transcript and feeds it to the
    AI as context — far better for accurate identification than a few short
    snippets per speaker. Pass --with-snippets to also include the per-speaker
    snippet block (with audio timestamps), which the AI can use to verify
    voices via the `/jobs/<id>/audio` stream.
    """
    parts = []
    if context:
        parts.append(f"Meeting context: {context}")

    if full_transcript:
        click.echo("Fetching transcript…", err=True)
        lines = []
        used = 0
        truncated = False
        for seg in _fetch_segments(job_id):
            spk = seg.get("speaker", "")
            prefix = f"[{spk}] " if spk else ""
            line = (f"{_format_ts(seg.get('start'))}–{_format_ts(seg.get('end'))} "
                    f"{prefix}{seg.get('text', '').strip()}")
            if used + len(line) > max_chars:
                truncated = True
                break
            lines.append(line)
            used += len(line) + 1
        if lines:
            parts.append("Full diarized transcript:\n" + "\n".join(lines)
                         + ("\n…(truncated to fit context window)" if truncated else ""))
        else:
            click.echo("No transcript segments fetched — is the job in speaker_review or done?", err=True)

    if with_snippets:
        click.echo("Fetching speaker snippets…", err=True)
        try:
            r = requests.get(f"{API_BASE}/jobs/{job_id}/speakers", timeout=15)
            r.raise_for_status()
            speakers = r.json().get("speakers", {}) or {}
        except requests.RequestException:
            speakers = {}
        if speakers:
            block = ["Speaker snippets (with audio timestamps):"]
            for label, items in sorted(speakers.items()):
                block.append(f"  {label}:")
                for s in items:
                    if isinstance(s, dict):
                        block.append(f"    [{_format_ts(s.get('start'))}-{_format_ts(s.get('end'))}] "
                                     f"{s.get('text', '').strip()}")
                    else:
                        block.append(f"    {s}")
            parts.append("\n".join(block))

    rich_context = "\n\n".join(parts) if parts else context

    try:
        r = requests.post(f"{API_BASE}/ai/identify-speakers",
                          json={"job_id": job_id, "context": rich_context},
                          timeout=120)
    except requests.ConnectionError:
        click.echo("Error: WhisperApp is not running.", err=True)
        raise SystemExit(1)
    if r.status_code == 503:
        click.echo("No AI provider configured. Set ai_provider in Settings.", err=True)
        raise SystemExit(1)
    if r.status_code == 502:
        click.echo(f"AI provider call failed: {_error_detail(r)}", err=True)
        raise SystemExit(1)
    r.raise_for_status()
    d = r.json()
    if not d["mapping"]:
        click.echo("AI could not identify any speakers.")
        return
    click.echo(f"Suggestions from {d['provider']} (context: {len(rich_context):,} chars):")
    for label, name in sorted(d["mapping"].items()):
        click.echo(f"  {label} → {name}")


@cli.command("meeting-notes")
@click.argument("job_id")
@click.option("--context", "-c", default="", help="Meeting context")
def meeting_notes(job_id, context):
    """Generate meeting notes from a completed transcript using AI."""
    try:
        r = requests.post(f"{API_BASE}/ai/meeting-notes",
                          json={"job_id": job_id, "context": context}, timeout=120)
    except requests.ConnectionError:
        click.echo("Error: WhisperApp is not running.", err=True)
        raise SystemExit(1)
    if r.status_code == 503:
        click.echo("No AI provider configured. Set ai_provider in Settings.", err=True)
        raise SystemExit(1)
    if r.status_code == 502:
        click.echo(f"AI provider call failed: {_error_detail(r)}", err=True)
        raise SystemExit(1)
    r.raise_for_status()
    d = r.json()
    click.echo(f"Saved to: {d['saved_to']}")
    click.echo()
    click.echo(d["notes"])


@cli.command("models")
def list_models():
    """List all available emotion models with downloaded/available status."""
    try:
        r = requests.get(f"{API_BASE}/models", timeout=5)
        r.raise_for_status()
        models = r.json()
    except requests.ConnectionError:
        # Fall back to direct listing when daemon isn't running
        from whisperapp.emotion_registry import ModelManager
        models = ModelManager().list_all()

    if not models:
        click.echo("No models available.")
        return
    for m in models:
        downloaded = "downloaded" if m.get("downloaded") else "not downloaded"
        downloading = " (downloading...)" if m.get("downloading") else ""
        recommended = " [recommended]" if m.get("recommended") else ""
        click.echo(
            f"{m['id']:30s}  {m['name']:35s}  {downloaded}{downloading}{recommended}"
        )
        click.echo(f"  {m.get('description', '')}")


@cli.command("download")
@click.argument("model_id")
def download_model(model_id):
    """Download an emotion model by ID (e.g. speechbrain-iemocap)."""
    try:
        r = requests.post(f"{API_BASE}/models/{model_id}/download", timeout=5)
        if r.status_code == 404:
            click.echo(f"Unknown model: {model_id}", err=True)
            raise SystemExit(1)
        r.raise_for_status()
        click.echo(f"Download started for {model_id}. Check progress with: whisperapp models")
    except requests.ConnectionError:
        # Fall back to direct download
        from whisperapp.emotion_registry import ModelManager, _REGISTRY_BY_ID
        if model_id not in _REGISTRY_BY_ID:
            click.echo(f"Unknown model: {model_id}", err=True)
            raise SystemExit(1)
        click.echo(f"Downloading {model_id} directly (daemon not running)...")
        mgr = ModelManager()
        mgr.download(model_id)
        # Wait for download since we're in CLI mode
        import time
        while True:
            status = mgr.get_status(model_id)
            if not status["downloading"]:
                break
            click.echo(f"  Progress: {status['progress']}%")
            time.sleep(3)
        if mgr.get_status(model_id)["downloaded"]:
            click.echo(f"Downloaded {model_id} successfully.")
        else:
            err = mgr.get_status(model_id).get("error", "Unknown error")
            click.echo(f"Download failed: {err}", err=True)
            raise SystemExit(1)


@cli.command("delete-model")
@click.argument("model_id")
def delete_model(model_id):
    """Delete a downloaded emotion model."""
    try:
        r = requests.delete(f"{API_BASE}/models/{model_id}", timeout=5)
        if r.status_code == 404:
            click.echo(f"Unknown model: {model_id}", err=True)
            raise SystemExit(1)
        r.raise_for_status()
        click.echo(f"Deleted {model_id}.")
    except requests.ConnectionError:
        from whisperapp.emotion_registry import ModelManager, _REGISTRY_BY_ID
        if model_id not in _REGISTRY_BY_ID:
            click.echo(f"Unknown model: {model_id}", err=True)
            raise SystemExit(1)
        ModelManager().delete(model_id)
        click.echo(f"Deleted {model_id}.")


def main():
    cli()
