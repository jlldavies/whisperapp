"""Pause detection: insert silence markers into transcription results."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whisperapp.config import Config


def _format_duration(seconds: int) -> str:
    """Format a duration in seconds to a human-readable string.

    Under 60s  -> "N seconds"
    60s+       -> "N minutes" or "N minutes M seconds" (seconds omitted if 0)
    """
    if seconds < 60:
        return f"{seconds} seconds"
    minutes, remainder = divmod(seconds, 60)
    if remainder == 0:
        return f"{minutes} minutes"
    return f"{minutes} minutes {remainder} seconds"


def _marker_text(gap_seconds: int, pause_threshold: int,
                 long_pause_threshold: int, gap_threshold: int) -> str | None:
    """Return the marker text for a gap of the given duration, or None if below threshold.

    Only the deepest matching tier fires.
    """
    duration_str = _format_duration(gap_seconds)
    if gap_seconds >= gap_threshold:
        return f"[Gap, {duration_str}]"
    if gap_seconds >= long_pause_threshold:
        return f"[Long Pause, {duration_str}]"
    if gap_seconds >= pause_threshold:
        return f"[Pause, {duration_str}]"
    return None


def _make_marker(gap_start: float, gap_end: float, text: str) -> dict:
    return {"text": text, "start": gap_start, "end": gap_end, "is_pause_marker": True}


def insert_pause_markers(result: dict, config: "Config") -> dict:
    """Return a copy of result with pause marker pseudo-segments inserted.

    Markers are only added to the segments list, not to word-level data.
    JSON and TSV callers should use the original result; txt/srt/vtt callers
    should use this function's output.

    The input result is never mutated.
    """
    if not config.pause_detection:
        return result

    segments = result.get("segments", [])
    if not segments:
        return result

    pt = config.pause_threshold
    lpt = config.long_pause_threshold
    gt = config.gap_threshold

    def _check_gap(prev_end: float, curr_start: float) -> dict | None:
        """Return a marker dict if the gap is above threshold, else None."""
        gap = curr_start - prev_end
        if gap <= 0:
            return None
        gap_seconds = round(gap)
        text = _marker_text(gap_seconds, pt, lpt, gt)
        if text is None:
            return None
        return _make_marker(prev_end, curr_start, text)

    # Build the output segment list by iterating through a flat ordered
    # sequence of "time anchors". Each segment contributes:
    #   - its words (if any), which may contain intra-segment gaps
    #   - or just its (start, end) boundary

    # Strategy: process everything as a flat list of (start, end, seg_ref)
    # entries, detect all gaps, then reassemble into output segments with
    # markers interleaved.

    # We produce a flat list of "slots":
    #   - "seg" slot: emits the segment as-is into output
    #   - "marker" slot: emits a pause marker pseudo-segment
    #
    # For each segment that has words, we look for intra-segment gaps between
    # consecutive words, and also for the inter-segment gap between the last
    # word of segment i and the first word of segment i+1.

    new_segments: list[dict] = []

    # Precompute first/last word boundaries per segment
    def _first_start(seg: dict) -> float:
        words = seg.get("words")
        if words:
            return words[0]["start"]
        return seg["start"]

    def _last_end(seg: dict) -> float:
        words = seg.get("words")
        if words:
            return words[-1]["end"]
        return seg["end"]

    for seg_i, seg in enumerate(segments):
        words = seg.get("words")

        if words and len(words) > 1:
            # Check for intra-segment gaps between consecutive words.
            # We emit the segment first, then any intra gaps as markers.
            # But since markers need to appear "between" content in the output
            # we need to handle this carefully.
            #
            # We collect intra-segment word-gap markers to insert BEFORE
            # appending this segment — actually we insert them AFTER the
            # segment (they are detected inside this segment's duration).
            # The output order is: [seg, marker?, next_seg, ...]
            # For intra-segment, we just append after the segment.
            new_segments.append(seg)
            for w_i in range(1, len(words)):
                marker = _check_gap(words[w_i - 1]["end"], words[w_i]["start"])
                if marker is not None:
                    new_segments.append(marker)
        else:
            new_segments.append(seg)

        # Check for gap between this segment and the next
        if seg_i + 1 < len(segments):
            next_seg = segments[seg_i + 1]
            this_end = _last_end(seg)
            next_start = _first_start(next_seg)
            marker = _check_gap(this_end, next_start)
            if marker is not None:
                new_segments.append(marker)

    new_result = dict(result)
    new_result["segments"] = new_segments
    return new_result
