"""Acoustic feature detection: volume, pitch, and speaking rate markers.

Inserts pseudo-segments (is_acoustic_marker=True) into transcription results
for txt/srt/vtt output only. Pure function — never mutates the input dict.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whisperapp.config import Config

logger = logging.getLogger(__name__)


def _make_marker(start: float, end: float, text: str) -> dict:
    return {"text": text, "start": start, "end": end, "is_acoustic_marker": True}


def _words_per_minute(words: list[dict]) -> float | None:
    """Return WPM for a list of word dicts with start/end times, or None if insufficient data."""
    if len(words) < 2:
        return None
    duration = words[-1]["end"] - words[0]["start"]
    if duration <= 0:
        return None
    return len(words) / duration * 60.0


def _analyse_volume(audio, sr: int, segments: list[dict], config: "Config") -> list[dict]:
    """Return a list of volume marker dicts for qualifying segments."""
    try:
        import librosa
        import numpy as np
    except ImportError:
        logger.warning("librosa not available — skipping volume analysis")
        return []

    markers: list[dict] = []
    threshold = config.acoustic_volume_threshold
    recent_rms: list[float] = []  # rolling baseline (last 10 real segments)

    for seg in segments:
        if seg.get("is_acoustic_marker") or seg.get("is_pause_marker") or seg.get("is_emotion_marker"):
            continue

        start = seg.get("start", 0)
        end = seg.get("end", 0)
        if end <= start:
            continue

        try:
            f1 = int(start * sr)
            f2 = int(end * sr)
            window = audio[f1:f2]
            if len(window) == 0:
                continue
            rms = float(np.sqrt(np.mean(window ** 2)))
        except Exception as exc:
            logger.debug("Volume analysis failed for segment [%s-%s]: %s", start, end, exc)
            continue

        if recent_rms:
            baseline = float(np.median(recent_rms[-10:]))
            if baseline > 0:
                if rms > baseline * threshold:
                    markers.append(_make_marker(start, end, "[Loud]"))
                elif rms < baseline / threshold:
                    markers.append(_make_marker(start, end, "[Quiet]"))

        recent_rms.append(rms)

    return markers


def _analyse_pitch(audio, sr: int, segments: list[dict], config: "Config") -> list[dict]:
    """Return a list of pitch marker dicts for qualifying segments."""
    try:
        import librosa
        import numpy as np
    except ImportError:
        logger.warning("librosa not available — skipping pitch analysis")
        return []

    markers: list[dict] = []
    pitch_threshold = config.acoustic_pitch_threshold

    for seg in segments:
        if seg.get("is_acoustic_marker") or seg.get("is_pause_marker") or seg.get("is_emotion_marker"):
            continue

        start = seg.get("start", 0)
        end = seg.get("end", 0)
        if end <= start:
            continue

        try:
            f1 = int(start * sr)
            f2 = int(end * sr)
            window = audio[f1:f2]
            if len(window) < sr // 4:  # need at least 250ms
                continue

            # Extract F0 using librosa.yin
            f0 = librosa.yin(window, fmin=60, fmax=400, sr=sr)
            # Filter out unvoiced frames (yin returns fmin for unvoiced)
            voiced = f0[f0 > 80]
            if len(voiced) < 5:
                continue

            rms = float(np.sqrt(np.mean(window ** 2)))
            f0_mean = float(np.mean(voiced))
            f0_std = float(np.std(voiced))

            # High pitch + high energy → [Raised voice]
            # Use percentile to define "high" — above 300Hz mean is elevated
            if f0_mean > 300 and rms > 0.01:
                markers.append(_make_marker(start, end, "[Raised voice]"))
            elif f0_std > pitch_threshold:
                # Sustained high variation without necessarily high mean → [Animated]
                markers.append(_make_marker(start, end, "[Animated]"))

        except Exception as exc:
            logger.debug("Pitch analysis failed for segment [%s-%s]: %s", start, end, exc)
            continue

    return markers


def _analyse_rate(segments: list[dict], config: "Config") -> list[dict]:
    """Return a list of speaking rate marker dicts for qualifying segments."""
    markers: list[dict] = []
    fast_wpm = config.acoustic_rate_fast_wpm
    slow_wpm = config.acoustic_rate_slow_wpm
    min_words = 5

    for seg in segments:
        if seg.get("is_acoustic_marker") or seg.get("is_pause_marker") or seg.get("is_emotion_marker"):
            continue

        words = seg.get("words", [])
        if len(words) <= min_words:
            continue

        try:
            wpm = _words_per_minute(words)
            if wpm is None:
                continue
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            if wpm > fast_wpm:
                markers.append(_make_marker(start, end, "[Rapid]"))
            elif wpm < slow_wpm:
                markers.append(_make_marker(start, end, "[Slow]"))
        except Exception as exc:
            logger.debug("Rate analysis failed for segment: %s", exc)
            continue

    return markers


def analyse_acoustics(audio_path: str, result: dict, config: "Config") -> dict:
    """Return a copy of result with acoustic markers inserted as pseudo-segments.

    Markers are inserted at the position of the segment they describe (after it).
    The input result is never mutated.

    Only volume/pitch detectors load librosa audio — rate detection is timestamp-only.
    If librosa fails to load audio, rate detection still runs.
    """
    if not config.acoustic_enabled:
        return result

    segments = result.get("segments", [])
    if not segments:
        return result

    # Collect all markers keyed by segment index
    volume_markers: dict[int, list[dict]] = {}
    pitch_markers: dict[int, list[dict]] = {}
    rate_markers: dict[int, list[dict]] = {}

    audio = None
    sr = None

    needs_audio = config.acoustic_volume_enabled or config.acoustic_pitch_enabled
    if needs_audio:
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as exc:
            logger.warning("Failed to load audio for acoustic analysis: %s", exc)
            needs_audio = False

    # Build index map: segment start -> index (for inserting markers after correct segment)
    real_segments = [
        (i, seg) for i, seg in enumerate(segments)
        if not seg.get("is_acoustic_marker")
        and not seg.get("is_pause_marker")
        and not seg.get("is_emotion_marker")
    ]

    if needs_audio and audio is not None:
        if config.acoustic_volume_enabled:
            vol_list = _analyse_volume(audio, sr, segments, config)
            # Map each marker back to the nearest real segment
            for m in vol_list:
                idx = _find_segment_index(real_segments, m["start"])
                if idx is not None:
                    volume_markers.setdefault(idx, []).append(m)

        if config.acoustic_pitch_enabled:
            pitch_list = _analyse_pitch(audio, sr, segments, config)
            for m in pitch_list:
                idx = _find_segment_index(real_segments, m["start"])
                if idx is not None:
                    pitch_markers.setdefault(idx, []).append(m)

    if config.acoustic_rate_enabled:
        rate_list = _analyse_rate(segments, config)
        for m in rate_list:
            idx = _find_segment_index(real_segments, m["start"])
            if idx is not None:
                rate_markers.setdefault(idx, []).append(m)

    # Rebuild segments with markers inserted after each qualifying segment
    new_segments: list[dict] = []
    for i, seg in enumerate(segments):
        new_segments.append(seg)
        # Only insert after real (non-marker) segments
        if (seg.get("is_acoustic_marker") or seg.get("is_pause_marker")
                or seg.get("is_emotion_marker")):
            continue
        # Find the real_segments index for this position
        real_idx = None
        for ri, (orig_i, _) in enumerate(real_segments):
            if orig_i == i:
                real_idx = ri
                break
        if real_idx is not None:
            new_segments.extend(volume_markers.get(real_idx, []))
            new_segments.extend(pitch_markers.get(real_idx, []))
            new_segments.extend(rate_markers.get(real_idx, []))

    new_result = dict(result)
    new_result["segments"] = new_segments
    return new_result


def _find_segment_index(real_segments: list[tuple[int, dict]], start: float) -> int | None:
    """Find the index in real_segments whose start matches the given start time."""
    for ri, (_, seg) in enumerate(real_segments):
        if seg.get("start") == start:
            return ri
    return None
