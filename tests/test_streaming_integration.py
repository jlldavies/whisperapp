"""Integration tests for streaming transcription (requires model downloads)."""

import numpy as np
import pytest


def _make_audio_with_pauses(duration_sec=3.0, sr=16000, freq=440.0):
    """Synthesize audio: 1s tone, 0.5s silence, 1s tone, 0.5s silence."""
    samples = []
    t_tone1 = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    samples.append(0.5 * np.sin(2 * np.pi * freq * t_tone1).astype(np.float32))
    samples.append(np.zeros(int(sr * 0.5), dtype=np.float32))
    t_tone2 = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    samples.append(0.5 * np.sin(2 * np.pi * freq * t_tone2).astype(np.float32))
    samples.append(np.zeros(int(sr * 0.5), dtype=np.float32))
    return np.concatenate(samples)


@pytest.mark.integration
def test_streaming_engine_with_synthesized_audio():
    """Feed synthesized audio chunks to StreamingEngine, verify it produces output."""
    from whisperapp.streaming import StreamingEngine

    engine = StreamingEngine(model_size="tiny", max_chunk_sec=2.0, silence_threshold_sec=0.4)
    engine.start()

    audio = _make_audio_with_pauses()
    chunk_size = 4000  # 0.25s chunks

    transcripts = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        result = engine.process_chunk(16000, chunk)
        if result:
            transcripts.append(result)

    final = engine.stop()

    # The model may or may not produce text for sine waves, but
    # the pipeline should run without errors and return a dict
    assert "text" in final
    assert "segments" in final
    assert "raw_audio" in final
    assert isinstance(final["raw_audio"], np.ndarray)
    assert len(final["raw_audio"]) > 0


@pytest.mark.integration
def test_polish_produces_aligned_segments():
    """Test that Polish step runs alignment on accumulated audio."""
    from whisperapp.streaming import StreamingEngine

    engine = StreamingEngine(model_size="tiny")
    engine.start()

    # Feed some audio
    audio = _make_audio_with_pauses()
    chunk_size = 4000
    for i in range(0, len(audio), chunk_size):
        engine.process_chunk(16000, audio[i:i + chunk_size])

    engine.stop()

    # Polish without HF token (alignment only, no diarization)
    polished = engine.polish(hf_token="")
    assert "segments" in polished
    # Segments should be a list (may be empty for sine waves)
    assert isinstance(polished["segments"], list)
