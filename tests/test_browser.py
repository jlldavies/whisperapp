"""Browser tests for WhisperApp UI — driven by Playwright against the live app.

These tests require the WhisperApp server to be running on the default ports:
  Web UI  : http://127.0.0.1:7860
  REST API: http://127.0.0.1:7861

They seed fake "done" jobs directly into the live database and clean up after
themselves, so no real transcription is needed.

Run with:
    pytest tests/test_browser.py -v -m browser
Or to watch the browser:
    pytest tests/test_browser.py -v -m browser --headed
"""

import json
import sqlite3
import uuid
import datetime
import pytest
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────

UI_URL  = "http://127.0.0.1:7860"
API_URL = "http://127.0.0.1:7861"
DB_PATH = Path.home() / ".whisperapp" / "jobs.db"

# Fixture transcript data with SHA-256 hash + source metadata
FAKE_TRANSCRIPT = {
    "language": "en",
    "model": "large-v2",
    "source_metadata": {
        "source_file":    "browser_test_audio.mp3",
        "source_path":    "/tmp/browser_test_audio.mp3",
        "source_hash":    "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "file_size_bytes": "1048576",
        "duration":       "00:07",
        "file_modified":  "2026-05-10T09:00:00",
    },
    "segments": [
        {
            "start": 0.0, "end": 3.5,
            "text": " Good morning everyone.",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 3.5, "end": 3.5,
            "text": "[Pause, 3 seconds]",
            "is_pause_marker": True,
        },
        {
            "start": 6.5, "end": 10.0,
            "text": " Thanks for joining the call.",
            "speaker": "SPEAKER_01",
        },
        {
            "start": 10.5, "end": 14.2,
            "text": " Let us get started with the agenda.",
            "speaker": "SPEAKER_00",
        },
    ],
}


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def done_job(tmp_path_factory):
    """Insert a fake done job into the live DB and write its checkpoint+JSON.
    Yields the job_id. Cleans up in teardown."""
    if not DB_PATH.exists():
        pytest.skip("Live WhisperApp database not found — is the app running?")

    tmp = tmp_path_factory.mktemp("browser_job")
    job_id  = str(uuid.uuid4())
    stem    = "browser_test_audio"
    audio_f = tmp / f"{stem}.mp3"
    audio_f.write_bytes(b"\x00" * 64)           # fake audio bytes
    out_dir = tmp

    # Write the checkpoint that the /segments endpoint will serve
    from whisperapp.checkpoints import CheckpointManager
    cm = CheckpointManager(str(out_dir), job_id)
    cm.save("speaker_review", FAKE_TRANSCRIPT)

    # Write the JSON output (fallback for done jobs)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(FAKE_TRANSCRIPT, indent=2), encoding="utf-8"
    )

    # Insert the job row into the live SQLite database
    now = datetime.datetime.now().isoformat()
    con = sqlite3.connect(str(DB_PATH))
    try:
        con.execute(
            """
            INSERT INTO jobs
              (id, file_path, file_name, output_path, model, diarize, formats,
               status, progress, stage, result_path, partial_path, error,
               created_at, started_at, completed_at, order_idx)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                job_id,
                str(audio_f),
                f"{stem}.mp3",
                str(out_dir),
                "large-v2",
                1,
                json.dumps(["txt", "json"]),
                "done",
                100,
                "done",
                str(out_dir / f"{stem}.json"),
                "",
                None,
                now, now, now,
                0,
            ),
        )
        con.commit()
    finally:
        con.close()

    yield job_id, stem

    # Teardown — remove the fake job from the live DB
    con = sqlite3.connect(str(DB_PATH))
    try:
        con.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        con.commit()
    finally:
        con.close()


@pytest.fixture(scope="module")
def browser_ctx(playwright):
    """Single Chromium browser shared across all tests in this module."""
    browser = playwright.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={"width": 1280, "height": 800})
    yield ctx
    browser.close()


# ── Helpers ────────────────────────────────────────────────────────────────

def nav_to(page, hash_name):
    """Navigate to a named section by clicking the sidebar link."""
    page.goto(f"{UI_URL}#{hash_name}")
    page.wait_for_timeout(600)


def open_viewer(page, job_id):
    """Navigate to Transcribe, reload job list, then click the done job."""
    nav_to(page, "transcribe")
    page.wait_for_selector(".history-item-done", timeout=5000)
    page.click(".history-item-done")
    page.wait_for_selector("#tx-viewer", timeout=5000)


# ── Mark all tests in this file as `browser` ──────────────────────────────

pytestmark = pytest.mark.browser


# ── Settings page ─────────────────────────────────────────────────────────

class TestSettingsPage:

    def test_template_section_visible(self, page):
        nav_to(page, "settings")
        page.click("text=Storage")
        page.wait_for_selector("text=Output templates", timeout=4000)
        assert page.is_visible("text=Output templates")

    def test_docx_download_button_present(self, page):
        nav_to(page, "settings")
        page.click("text=Storage")
        page.wait_for_selector("text=Download DOCX template", timeout=4000)
        btn = page.locator("text=Download DOCX template")
        assert btn.count() == 1

    def test_html_download_button_present(self, page):
        nav_to(page, "settings")
        page.click("text=Storage")
        page.wait_for_selector("text=Download HTML template", timeout=4000)
        assert page.locator("text=Download HTML template").count() == 1

    def test_legal_template_download_button_present(self, page):
        nav_to(page, "settings")
        page.click("text=Storage")
        page.wait_for_selector("text=Download Legal template", timeout=4000)
        assert page.locator("text=Download Legal template").count() == 1

    def test_source_hash_marker_visible(self, page):
        nav_to(page, "settings")
        page.click("text=Storage")
        page.wait_for_selector("text={{source_hash}}", timeout=4000)
        assert page.is_visible("text={{source_hash}}")

    def test_file_size_bytes_marker_visible(self, page):
        nav_to(page, "settings")
        page.click("text=Storage")
        page.wait_for_selector("text={{file_size_bytes}}", timeout=4000)
        assert page.is_visible("text={{file_size_bytes}}")

    def test_legal_transcript_marker_visible(self, page):
        nav_to(page, "settings")
        page.click("text=Storage")
        page.wait_for_selector("text={{legal_transcript}}", timeout=4000)
        assert page.is_visible("text={{legal_transcript}}")

    def test_all_fifteen_markers_present(self, page):
        nav_to(page, "settings")
        page.click("text=Storage")
        page.wait_for_selector("text=Available markers", timeout=4000)
        # Count marker spans — there should be 15 now
        markers = page.locator(".wa-label:has-text('Available markers') ~ div .mono")
        assert markers.count() >= 15


# ── Transcribe page ────────────────────────────────────────────────────────

class TestTranscribePage:

    def test_docx_format_chip_visible(self, page):
        nav_to(page, "transcribe")
        page.wait_for_selector("[data-fmt='docx']", timeout=4000)
        assert page.is_visible("[data-fmt='docx']")

    def test_pdf_format_chip_visible(self, page):
        nav_to(page, "transcribe")
        page.wait_for_selector("[data-fmt='pdf']", timeout=4000)
        assert page.is_visible("[data-fmt='pdf']")

    def test_done_job_shows_in_history(self, page, done_job):
        nav_to(page, "transcribe")
        page.wait_for_selector(".history-item-done", timeout=5000)
        assert page.locator(".history-item-done").count() >= 1

    def test_done_job_has_checkmark(self, page, done_job):
        nav_to(page, "transcribe")
        page.wait_for_selector(".history-item-done", timeout=5000)
        # The done item shows a ✓ icon
        item_text = page.locator(".history-item-done").first.inner_text()
        assert "✓" in item_text

    def test_done_job_has_filename(self, page, done_job):
        _, stem = done_job
        nav_to(page, "transcribe")
        page.wait_for_selector(".history-item-done", timeout=5000)
        assert page.locator(f"text={stem}.mp3").count() >= 1


# ── Transcript Viewer Overlay ──────────────────────────────────────────────

class TestTranscriptViewer:

    def test_viewer_opens_on_click(self, page, done_job):
        open_viewer(page, done_job[0])
        assert page.is_visible("#tx-viewer")

    def test_viewer_shows_toolbar(self, page, done_job):
        open_viewer(page, done_job[0])
        assert page.is_visible("#txv-toolbar")
        assert page.is_visible("#txv-close")
        assert page.is_visible("#txv-search")
        assert page.is_visible("#txv-copy")
        assert page.is_visible("#txv-print")

    def test_viewer_shows_provenance_block(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector("#txv-provenance", timeout=5000)
        assert page.is_visible("#txv-provenance")
        assert page.is_visible(".txv-doc-title")

    def test_viewer_provenance_title(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector(".txv-doc-title", timeout=5000)
        title = page.locator(".txv-doc-title").inner_text()
        assert "TRANSCRIPT" in title.upper()

    def test_viewer_shows_sha256_hash(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector("#txv-provenance", timeout=5000)
        # The hash prefix should appear in the provenance block
        assert page.locator("text=a1b2c3d4").count() >= 1

    def test_viewer_shows_source_filename(self, page, done_job):
        _, stem = done_job
        open_viewer(page, done_job[0])
        page.wait_for_selector("#txv-provenance", timeout=5000)
        assert page.locator(f"text={stem}.mp3").count() >= 1

    def test_viewer_shows_transcript_lines(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector("#txv-lines", timeout=5000)
        lines = page.locator(".txv-line")
        assert lines.count() >= 4   # 3 speech + 1 pause

    def test_viewer_shows_line_numbers(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector(".txv-ln", timeout=5000)
        assert page.locator("text=L0001").count() >= 1

    def test_viewer_shows_timestamps(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector(".txv-ts", timeout=5000)
        # First segment starts at 0.0 → "00:00"
        assert page.locator(".txv-ts").first.inner_text() in ("00:00", "00:00:00")

    def test_viewer_shows_speaker_labels(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector(".txv-sp", timeout=5000)
        speakers = page.locator(".txv-sp")
        speaker_texts = [speakers.nth(i).inner_text() for i in range(speakers.count())]
        assert any("SPEAKER_00" in t for t in speaker_texts)

    def test_viewer_shows_pause_marker(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector(".txv-pause", timeout=5000)
        assert page.locator(".txv-pause").count() >= 1
        assert "Pause" in page.locator(".txv-pause").first.inner_text()

    def test_viewer_shows_transcript_text(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector("#txv-lines", timeout=5000)
        assert page.locator("text=Good morning everyone").count() >= 1
        assert page.locator("text=Thanks for joining the call").count() >= 1

    def test_search_filters_lines(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector("#txv-lines", timeout=5000)
        total_before = page.locator(".txv-line:not([hidden])").count()

        page.fill("#txv-search", "morning")
        page.wait_for_timeout(200)

        visible = page.locator(".txv-line:not([hidden])").count()
        assert visible < total_before
        assert visible >= 1

    def test_search_highlights_matching_line(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector("#txv-lines", timeout=5000)
        page.fill("#txv-search", "morning")
        page.wait_for_timeout(200)
        assert page.locator(".txv-match").count() >= 1

    def test_search_clear_restores_all_lines(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector("#txv-lines", timeout=5000)
        total = page.locator(".txv-line").count()

        page.fill("#txv-search", "morning")
        page.wait_for_timeout(200)
        page.fill("#txv-search", "")
        page.wait_for_timeout(200)

        visible = page.locator(".txv-line:not([hidden])").count()
        assert visible == total

    def test_copy_button_feedback(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector("#txv-copy", timeout=5000)
        page.click("#txv-copy")
        # Button should briefly show "Copied!"
        page.wait_for_selector("#txv-copy:has-text('Copied!')", timeout=2000)
        assert page.locator("#txv-copy:has-text('Copied!')").count() == 1

    def test_copy_button_reverts_to_copy(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector("#txv-copy", timeout=5000)
        page.click("#txv-copy")
        page.wait_for_selector("#txv-copy:has-text('Copied!')", timeout=2000)
        # After 1.5s the button reverts to "Copy"
        page.wait_for_selector("#txv-copy:has-text('Copy')", timeout=3000)

    def test_close_with_back_button(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector("#txv-close", timeout=5000)
        page.click("#txv-close")
        page.wait_for_timeout(300)
        assert not page.is_visible("#tx-viewer")

    def test_close_with_escape_key(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector("#tx-viewer", timeout=5000)
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        assert not page.is_visible("#tx-viewer")

    def test_viewer_shows_file_size(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector("#txv-provenance", timeout=5000)
        # 1048576 bytes → "1,048,576 bytes" or plain "1048576"
        prov = page.locator("#txv-provenance").inner_text()
        assert "1048576" in prov or "1,048,576" in prov

    def test_viewer_footer_visible(self, page, done_job):
        open_viewer(page, done_job[0])
        page.wait_for_selector(".txv-footer", timeout=5000)
        footer = page.locator(".txv-footer").inner_text()
        # Footer shows source path + hash snippet
        assert "browser_test_audio" in footer or "a1b2c3" in footer


# ── Conftest injection — each test gets a fresh page ──────────────────────
# pytest-playwright provides `page` fixture automatically when
# browser_ctx is not used. We override here to ensure a clean page per test.

@pytest.fixture(autouse=True)
def page(browser_ctx):
    pg = browser_ctx.new_page()
    yield pg
    pg.close()
