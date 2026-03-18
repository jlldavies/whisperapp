import click
import requests
from whisperapp.queue import JobQueue
from pathlib import Path

API_BASE = "http://127.0.0.1:7861"

def get_queue():
    return JobQueue()

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
    """Cancel a job."""
    r = requests.post(f"{API_BASE}/jobs/{job_id}/cancel", timeout=5)
    r.raise_for_status()
    click.echo("Cancelled.")

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

def main():
    cli()
