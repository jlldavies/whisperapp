"""Tests for pause detection (whisperapp.pauses)."""
import pytest
from whisperapp.pauses import insert_pause_markers, _format_duration, _marker_text


# ---------------------------------------------------------------------------
# Helper: minimal Config-like object
# ---------------------------------------------------------------------------

class MockConfig:
    def __init__(self,
                 pause_detection=True,
                 pause_threshold=3,
                 long_pause_threshold=10,
                 gap_threshold=30):
        self.pause_detection = pause_detection
        self.pause_threshold = pause_threshold
        self.long_pause_threshold = long_pause_threshold
        self.gap_threshold = gap_threshold


DEFAULT_CFG = MockConfig()


def _result(*segments):
    """Build a minimal result dict from (start, end, text) tuples."""
    return {
        "segments": [
            {"start": s, "end": e, "text": t}
            for s, e, t in segments
        ]
    }


def _result_with_words(*segments_with_words):
    """Build a result dict with word-level timestamps.

    Each element: (seg_start, seg_end, seg_text, [(w_start, w_end, word), ...])
    """
    segs = []
    for seg_start, seg_end, seg_text, words in segments_with_words:
        segs.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "words": [{"start": ws, "end": we, "word": w} for ws, we, w in words],
        })
    return {"segments": segs}


# ---------------------------------------------------------------------------
# _format_duration
# ---------------------------------------------------------------------------

class TestFormatDuration:
    def test_seconds_only(self):
        assert _format_duration(5) == "5 seconds"

    def test_one_second(self):
        assert _format_duration(1) == "1 seconds"

    def test_59_seconds(self):
        assert _format_duration(59) == "59 seconds"

    def test_exactly_one_minute(self):
        assert _format_duration(60) == "1 minutes"

    def test_two_minutes(self):
        assert _format_duration(120) == "2 minutes"

    def test_minutes_and_seconds(self):
        assert _format_duration(65) == "1 minutes 5 seconds"

    def test_minutes_and_seconds_larger(self):
        assert _format_duration(105) == "1 minutes 45 seconds"

    def test_three_minutes(self):
        assert _format_duration(180) == "3 minutes"

    def test_three_minutes_ten_seconds(self):
        assert _format_duration(190) == "3 minutes 10 seconds"


# ---------------------------------------------------------------------------
# _marker_text
# ---------------------------------------------------------------------------

class TestMarkerText:
    def test_below_threshold_returns_none(self):
        assert _marker_text(2, 3, 10, 30) is None

    def test_exactly_at_pause_threshold(self):
        result = _marker_text(3, 3, 10, 30)
        assert result == "[Pause, 3 seconds]"

    def test_pause_tier(self):
        assert _marker_text(5, 3, 10, 30) == "[Pause, 5 seconds]"

    def test_exactly_at_long_pause_threshold(self):
        result = _marker_text(10, 3, 10, 30)
        assert result == "[Long Pause, 10 seconds]"

    def test_long_pause_tier(self):
        assert _marker_text(20, 3, 10, 30) == "[Long Pause, 20 seconds]"

    def test_exactly_at_gap_threshold(self):
        result = _marker_text(30, 3, 10, 30)
        assert result == "[Gap, 30 seconds]"

    def test_gap_tier(self):
        assert _marker_text(45, 3, 10, 30) == "[Gap, 45 seconds]"

    def test_gap_with_minutes(self):
        assert _marker_text(180, 3, 10, 30) == "[Gap, 3 minutes]"

    def test_gap_with_minutes_and_seconds(self):
        assert _marker_text(105, 3, 10, 30) == "[Gap, 1 minutes 45 seconds]"


# ---------------------------------------------------------------------------
# insert_pause_markers — disabled
# ---------------------------------------------------------------------------

class TestPauseDetectionDisabled:
    def test_returns_same_result_when_disabled(self):
        cfg = MockConfig(pause_detection=False)
        result = _result((0, 5, "Hello"), (20, 25, "World"))
        out = insert_pause_markers(result, cfg)
        assert out is result

    def test_no_markers_when_disabled(self):
        cfg = MockConfig(pause_detection=False)
        result = _result((0, 5, "Hello"), (100, 105, "World"))
        out = insert_pause_markers(result, cfg)
        assert len(out["segments"]) == 2


# ---------------------------------------------------------------------------
# insert_pause_markers — no markers below threshold
# ---------------------------------------------------------------------------

class TestNoMarkersWhenBelowThreshold:
    def test_small_gap_no_marker(self):
        result = _result((0, 5, "Hello"), (7, 10, "World"))
        out = insert_pause_markers(result, DEFAULT_CFG)
        assert len(out["segments"]) == 2
        assert not any(s.get("is_pause_marker") for s in out["segments"])

    def test_zero_gap_no_marker(self):
        result = _result((0, 5, "Hello"), (5, 10, "World"))
        out = insert_pause_markers(result, DEFAULT_CFG)
        assert len(out["segments"]) == 2

    def test_no_segments_returns_unchanged(self):
        result = {"segments": []}
        out = insert_pause_markers(result, DEFAULT_CFG)
        assert out["segments"] == []


# ---------------------------------------------------------------------------
# insert_pause_markers — correct tier fires
# ---------------------------------------------------------------------------

class TestCorrectTierFires:
    def test_pause_tier(self):
        # 5-second gap -> Pause
        result = _result((0, 5, "Hello"), (10, 15, "World"))
        out = insert_pause_markers(result, DEFAULT_CFG)
        markers = [s for s in out["segments"] if s.get("is_pause_marker")]
        assert len(markers) == 1
        assert markers[0]["text"] == "[Pause, 5 seconds]"

    def test_long_pause_tier(self):
        # 20-second gap -> Long Pause
        result = _result((0, 5, "Hello"), (25, 30, "World"))
        out = insert_pause_markers(result, DEFAULT_CFG)
        markers = [s for s in out["segments"] if s.get("is_pause_marker")]
        assert len(markers) == 1
        assert markers[0]["text"] == "[Long Pause, 20 seconds]"

    def test_gap_tier(self):
        # 45-second gap -> Gap (not Long Pause)
        result = _result((0, 5, "Hello"), (50, 55, "World"))
        out = insert_pause_markers(result, DEFAULT_CFG)
        markers = [s for s in out["segments"] if s.get("is_pause_marker")]
        assert len(markers) == 1
        assert markers[0]["text"] == "[Gap, 45 seconds]"

    def test_only_deepest_tier_fires_45s(self):
        # 45s -> Gap only, not also Long Pause or Pause
        result = _result((0, 5, "Hello"), (50, 55, "World"))
        out = insert_pause_markers(result, DEFAULT_CFG)
        markers = [s for s in out["segments"] if s.get("is_pause_marker")]
        assert len(markers) == 1
        assert "Gap" in markers[0]["text"]
        assert "Long Pause" not in markers[0]["text"]

    def test_only_deepest_tier_fires_20s(self):
        # 20s -> Long Pause only, not also Pause
        result = _result((0, 5, "Hello"), (25, 30, "World"))
        out = insert_pause_markers(result, DEFAULT_CFG)
        markers = [s for s in out["segments"] if s.get("is_pause_marker")]
        assert len(markers) == 1
        assert "Long Pause" in markers[0]["text"]
        assert "Pause," not in markers[0]["text"].replace("Long Pause", "")


# ---------------------------------------------------------------------------
# insert_pause_markers — markers are inserted in correct positions
# ---------------------------------------------------------------------------

class TestMarkerInsertion:
    def test_marker_between_correct_segments(self):
        result = _result(
            (0, 5, "First"),
            (10, 15, "Second"),   # 5s gap before this
            (16, 20, "Third"),    # 1s gap (no marker)
        )
        out = insert_pause_markers(result, DEFAULT_CFG)
        texts = [s["text"] for s in out["segments"]]
        # Marker should be between First and Second
        first_idx = texts.index("First")
        second_idx = texts.index("Second")
        assert second_idx - first_idx == 2
        assert out["segments"][first_idx + 1].get("is_pause_marker")

    def test_multiple_markers(self):
        result = _result(
            (0, 5, "A"),
            (10, 15, "B"),   # 5s gap
            (30, 35, "C"),   # 15s gap
        )
        out = insert_pause_markers(result, DEFAULT_CFG)
        markers = [s for s in out["segments"] if s.get("is_pause_marker")]
        assert len(markers) == 2
        assert markers[0]["text"] == "[Pause, 5 seconds]"
        assert markers[1]["text"] == "[Long Pause, 15 seconds]"

    def test_marker_timing(self):
        # Marker start/end should match gap boundaries
        result = _result((0, 5.0, "Hello"), (10.0, 15, "World"))
        out = insert_pause_markers(result, DEFAULT_CFG)
        marker = next(s for s in out["segments"] if s.get("is_pause_marker"))
        assert marker["start"] == 5.0
        assert marker["end"] == 10.0

    def test_input_not_mutated(self):
        result = _result((0, 5, "Hello"), (10, 15, "World"))
        original_segs = len(result["segments"])
        insert_pause_markers(result, DEFAULT_CFG)
        assert len(result["segments"]) == original_segs


# ---------------------------------------------------------------------------
# insert_pause_markers — word-level timestamps
# ---------------------------------------------------------------------------

class TestWordLevelTimestamps:
    def test_uses_word_timestamps_when_available(self):
        # Segment spans 0-20 but words only go 0-5, then 15-20
        # The gap is within the segment — should be detected
        result = _result_with_words(
            (0, 20, "Hello World", [(0.0, 2.5, "Hello"), (15.0, 17.5, "World")]),
        )
        out = insert_pause_markers(result, DEFAULT_CFG)
        # The gap is 15.0 - 2.5 = 12.5s -> Long Pause
        markers = [s for s in out["segments"] if s.get("is_pause_marker")]
        assert len(markers) == 1
        assert "Long Pause" in markers[0]["text"]

    def test_word_gap_below_threshold_no_marker(self):
        result = _result_with_words(
            (0, 10, "Hello World", [(0.0, 2.5, "Hello"), (4.0, 6.0, "World")]),
        )
        out = insert_pause_markers(result, DEFAULT_CFG)
        markers = [s for s in out["segments"] if s.get("is_pause_marker")]
        assert len(markers) == 0

    def test_cross_segment_word_gap(self):
        # Two segments, each with words, large gap between last word of seg1
        # and first word of seg2
        result = _result_with_words(
            (0, 5, "Hello", [(0.0, 2.0, "Hello")]),
            (50, 55, "World", [(50.0, 52.0, "World")]),
        )
        out = insert_pause_markers(result, DEFAULT_CFG)
        markers = [s for s in out["segments"] if s.get("is_pause_marker")]
        assert len(markers) == 1
        # Gap: 50 - 2 = 48s -> Gap tier
        assert "Gap" in markers[0]["text"]


# ---------------------------------------------------------------------------
# insert_pause_markers — duration formatting in output
# ---------------------------------------------------------------------------

class TestDurationFormatInOutput:
    def test_seconds_format(self):
        result = _result((0, 5, "A"), (10, 15, "B"))
        out = insert_pause_markers(result, DEFAULT_CFG)
        marker = next(s for s in out["segments"] if s.get("is_pause_marker"))
        assert "5 seconds" in marker["text"]

    def test_minutes_format(self):
        result = _result((0, 5, "A"), (125, 130, "B"))  # 120s gap
        out = insert_pause_markers(result, DEFAULT_CFG)
        marker = next(s for s in out["segments"] if s.get("is_pause_marker"))
        assert "2 minutes" in marker["text"]
        assert "seconds" not in marker["text"]

    def test_mixed_format(self):
        result = _result((0, 5, "A"), (110, 115, "B"))  # 105s gap
        out = insert_pause_markers(result, DEFAULT_CFG)
        marker = next(s for s in out["segments"] if s.get("is_pause_marker"))
        assert "1 minutes 45 seconds" in marker["text"]


# ---------------------------------------------------------------------------
# insert_pause_markers — result is a new dict (pure function)
# ---------------------------------------------------------------------------

class TestPurity:
    def test_returns_new_dict(self):
        result = _result((0, 5, "Hello"), (10, 15, "World"))
        out = insert_pause_markers(result, DEFAULT_CFG)
        assert out is not result

    def test_preserves_other_keys(self):
        result = _result((0, 5, "Hello"), (10, 15, "World"))
        result["text"] = "Hello World"
        result["language"] = "en"
        out = insert_pause_markers(result, DEFAULT_CFG)
        assert out["text"] == "Hello World"
        assert out["language"] == "en"

    def test_no_markers_returns_copy_not_same(self):
        # Even when no markers, returns a new dict
        result = _result((0, 5, "Hello"), (6, 10, "World"))
        out = insert_pause_markers(result, DEFAULT_CFG)
        # segments list may be same reference if no gaps, but outer dict is new
        assert out is not result
