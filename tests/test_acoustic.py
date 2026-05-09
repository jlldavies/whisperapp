"""Tests for acoustic feature detection (whisperapp.acoustic)."""
import pytest
from whisperapp.acoustic import analyse_acoustics, _words_per_minute


# ---------------------------------------------------------------------------
# Helper: minimal Config-like object
# ---------------------------------------------------------------------------

class MockConfig:
    def __init__(self,
                 acoustic_enabled=True,
                 acoustic_volume_enabled=True,
                 acoustic_volume_threshold=1.8,
                 acoustic_pitch_enabled=True,
                 acoustic_pitch_threshold=40.0,
                 acoustic_rate_enabled=True,
                 acoustic_rate_fast_wpm=180,
                 acoustic_rate_slow_wpm=90):
        self.acoustic_enabled = acoustic_enabled
        self.acoustic_volume_enabled = acoustic_volume_enabled
        self.acoustic_volume_threshold = acoustic_volume_threshold
        self.acoustic_pitch_enabled = acoustic_pitch_enabled
        self.acoustic_pitch_threshold = acoustic_pitch_threshold
        self.acoustic_rate_enabled = acoustic_rate_enabled
        self.acoustic_rate_fast_wpm = acoustic_rate_fast_wpm
        self.acoustic_rate_slow_wpm = acoustic_rate_slow_wpm


def _make_result(*segments):
    """Build a minimal result dict from list of segment dicts."""
    return {"segments": list(segments), "language": "en"}


def _seg(start, end, text, words=None):
    s = {"start": start, "end": end, "text": text}
    if words:
        s["words"] = words
    return s


def _word(start, end, word):
    return {"start": start, "end": end, "word": word}


# ---------------------------------------------------------------------------
# Returns unchanged result when disabled
# ---------------------------------------------------------------------------

class TestDisabled:
    def test_returns_same_object_when_disabled(self):
        cfg = MockConfig(acoustic_enabled=False)
        result = _make_result(_seg(0, 5, "Hello"))
        out = analyse_acoustics("/fake/audio.wav", result, cfg)
        assert out is result

    def test_no_markers_when_disabled(self):
        cfg = MockConfig(acoustic_enabled=False)
        result = _make_result(_seg(0, 5, "Hello"), _seg(5, 10, "World"))
        out = analyse_acoustics("/fake/audio.wav", result, cfg)
        assert not any(s.get("is_acoustic_marker") for s in out["segments"])

    def test_empty_segments_returns_unchanged(self):
        cfg = MockConfig(acoustic_enabled=True)
        result = {"segments": []}
        out = analyse_acoustics("/fake/audio.wav", result, cfg)
        assert out["segments"] == []


# ---------------------------------------------------------------------------
# Pure function (doesn't mutate input)
# ---------------------------------------------------------------------------

class TestPurity:
    def test_returns_new_dict(self):
        cfg = MockConfig(acoustic_enabled=True, acoustic_volume_enabled=False,
                         acoustic_pitch_enabled=False, acoustic_rate_enabled=False)
        result = _make_result(_seg(0, 5, "Hello"))
        out = analyse_acoustics("/fake/audio.wav", result, cfg)
        assert out is not result

    def test_input_segments_not_mutated(self):
        cfg = MockConfig(acoustic_enabled=True, acoustic_volume_enabled=False,
                         acoustic_pitch_enabled=False, acoustic_rate_enabled=False)
        result = _make_result(_seg(0, 5, "Hello"), _seg(5, 10, "World"))
        orig_len = len(result["segments"])
        analyse_acoustics("/fake/audio.wav", result, cfg)
        assert len(result["segments"]) == orig_len

    def test_preserves_other_keys(self):
        cfg = MockConfig(acoustic_enabled=True, acoustic_volume_enabled=False,
                         acoustic_pitch_enabled=False, acoustic_rate_enabled=False)
        result = _make_result(_seg(0, 5, "Hello"))
        result["text"] = "Hello"
        result["language"] = "en"
        out = analyse_acoustics("/fake/audio.wav", result, cfg)
        assert out["text"] == "Hello"
        assert out["language"] == "en"


# ---------------------------------------------------------------------------
# Speaking rate markers (no audio needed — pure timestamp logic)
# ---------------------------------------------------------------------------

class TestSpeakingRate:
    def _rate_cfg(self, fast=180, slow=90):
        return MockConfig(
            acoustic_enabled=True,
            acoustic_volume_enabled=False,
            acoustic_pitch_enabled=False,
            acoustic_rate_enabled=True,
            acoustic_rate_fast_wpm=fast,
            acoustic_rate_slow_wpm=slow,
        )

    def _seg_with_words(self, start, word_count, seconds):
        """Build a segment with word_count words spread over `seconds`."""
        words = []
        for i in range(word_count):
            ws = start + (i / word_count) * seconds
            we = ws + (seconds / word_count) * 0.8
            words.append(_word(ws, we, f"word{i}"))
        end = start + seconds
        return _seg(start, end, " ".join(f"word{i}" for i in range(word_count)), words)

    def test_rapid_marker_above_threshold(self):
        # 12 words in 3 seconds = 240 WPM > 180
        seg = self._seg_with_words(0, 12, 3.0)
        result = _make_result(seg)
        out = analyse_acoustics("/fake/audio.wav", result, self._rate_cfg())
        markers = [s for s in out["segments"] if s.get("is_acoustic_marker")]
        assert any(s["text"] == "[Rapid]" for s in markers)

    def test_slow_marker_below_threshold(self):
        # 6 words in 10 seconds = 36 WPM < 90
        seg = self._seg_with_words(0, 6, 10.0)
        result = _make_result(seg)
        out = analyse_acoustics("/fake/audio.wav", result, self._rate_cfg())
        markers = [s for s in out["segments"] if s.get("is_acoustic_marker")]
        assert any(s["text"] == "[Slow]" for s in markers)

    def test_no_marker_for_normal_rate(self):
        # 10 words in 4 seconds = 150 WPM — between 90 and 180
        seg = self._seg_with_words(0, 10, 4.0)
        result = _make_result(seg)
        out = analyse_acoustics("/fake/audio.wav", result, self._rate_cfg())
        markers = [s for s in out["segments"] if s.get("is_acoustic_marker")]
        assert not markers

    def test_no_marker_for_short_segment(self):
        # 5 words — at the min_words boundary (> 5 needed, 5 is not enough)
        seg = self._seg_with_words(0, 5, 1.0)  # 300 WPM but only 5 words
        result = _make_result(seg)
        out = analyse_acoustics("/fake/audio.wav", result, self._rate_cfg())
        markers = [s for s in out["segments"] if s.get("is_acoustic_marker")]
        assert not markers

    def test_no_marker_for_very_short_segment_4_words(self):
        # Only 4 words — definitely below min_words (5)
        seg = self._seg_with_words(0, 4, 0.5)  # 480 WPM but too few words
        result = _make_result(seg)
        out = analyse_acoustics("/fake/audio.wav", result, self._rate_cfg())
        markers = [s for s in out["segments"] if s.get("is_acoustic_marker")]
        assert not markers

    def test_no_marker_without_word_timestamps(self):
        # Segment without word-level data — rate cannot be computed
        seg = _seg(0, 10, "Hello world")
        result = _make_result(seg)
        out = analyse_acoustics("/fake/audio.wav", result, self._rate_cfg())
        markers = [s for s in out["segments"] if s.get("is_acoustic_marker")]
        assert not markers

    def test_is_acoustic_marker_key_present(self):
        # 12 words in 3 seconds = 240 WPM > 180
        seg = self._seg_with_words(0, 12, 3.0)
        result = _make_result(seg)
        out = analyse_acoustics("/fake/audio.wav", result, self._rate_cfg())
        markers = [s for s in out["segments"] if s.get("is_acoustic_marker")]
        assert len(markers) > 0
        for m in markers:
            assert m.get("is_acoustic_marker") is True
            assert "start" in m
            assert "end" in m
            assert "text" in m

    def test_rate_disabled_no_markers(self):
        seg = self._seg_with_words(0, 12, 3.0)
        result = _make_result(seg)
        cfg = MockConfig(acoustic_enabled=True, acoustic_volume_enabled=False,
                         acoustic_pitch_enabled=False, acoustic_rate_enabled=False)
        out = analyse_acoustics("/fake/audio.wav", result, cfg)
        markers = [s for s in out["segments"] if s.get("is_acoustic_marker")]
        assert not markers

    def test_markers_not_on_existing_markers(self):
        """Acoustic markers should not be processed as segments for further marking."""
        rate_seg = self._seg_with_words(0, 12, 3.0)
        existing_marker = {
            "start": 0, "end": 1, "text": "[Pause]", "is_pause_marker": True
        }
        result = _make_result(existing_marker, rate_seg)
        out = analyse_acoustics("/fake/audio.wav", result, self._rate_cfg())
        # The pause marker should pass through unchanged
        pause_markers = [s for s in out["segments"] if s.get("is_pause_marker")]
        assert len(pause_markers) == 1


# ---------------------------------------------------------------------------
# _words_per_minute helper
# ---------------------------------------------------------------------------

class TestWordsPerMinute:
    def test_returns_none_for_single_word(self):
        words = [_word(0, 1, "Hello")]
        assert _words_per_minute(words) is None

    def test_returns_none_for_zero_duration(self):
        words = [_word(0, 0, "Hello"), _word(0, 0, "World")]
        assert _words_per_minute(words) is None

    def test_correct_wpm(self):
        # 6 words, first starts at 0.0, last ends at 2.0 → duration 2.0s → 180 WPM
        # _words_per_minute uses words[-1]["end"] - words[0]["start"]
        words = [
            _word(0.0, 0.3, "w0"),
            _word(0.3, 0.6, "w1"),
            _word(0.6, 0.9, "w2"),
            _word(0.9, 1.2, "w3"),
            _word(1.2, 1.5, "w4"),
            _word(1.5, 2.0, "w5"),  # last word ends at 2.0
        ]
        # duration = 2.0 - 0.0 = 2.0s, wpm = 6/2.0*60 = 180
        wpm = _words_per_minute(words)
        assert wpm is not None
        assert abs(wpm - 180) < 1
