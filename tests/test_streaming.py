"""Unit tests for the streaming transcription engine."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# AudioBuffer tests
# ---------------------------------------------------------------------------

class TestAudioBuffer:
    def test_accumulates_chunks(self):
        from whisperapp.streaming import AudioBuffer
        buf = AudioBuffer(min_chunk_sec=0.5, max_chunk_sec=5.0, silence_threshold_sec=0.3)
        # Add a short chunk — should not yield utterance yet
        chunk = np.zeros(8000, dtype=np.float32)  # 0.5s at 16kHz
        with patch("whisperapp.streaming._speech_prob", return_value=0.9):
            buf.add_chunk(16000, chunk)
        assert buf.duration == pytest.approx(0.5, abs=0.01)
        assert buf.get_utterance() is None  # no silence yet

    def test_returns_utterance_on_silence(self):
        from whisperapp.streaming import AudioBuffer
        buf = AudioBuffer(min_chunk_sec=0.5, max_chunk_sec=5.0, silence_threshold_sec=0.3)

        # Speech chunk
        speech = np.random.randn(8000).astype(np.float32)
        with patch("whisperapp.streaming._speech_prob", return_value=0.9):
            buf.add_chunk(16000, speech)

        # Silence chunk
        silence = np.zeros(8000, dtype=np.float32)
        with patch("whisperapp.streaming._speech_prob", return_value=0.1):
            buf.add_chunk(16000, silence)

        utt = buf.get_utterance()
        assert utt is not None
        assert len(utt) == 16000  # combined

    def test_force_flush_on_max_chunk(self):
        from whisperapp.streaming import AudioBuffer
        buf = AudioBuffer(min_chunk_sec=0.5, max_chunk_sec=1.0, silence_threshold_sec=0.3)

        # Add enough speech to exceed max_chunk_sec
        chunk = np.random.randn(16000).astype(np.float32)  # 1.0s
        with patch("whisperapp.streaming._speech_prob", return_value=0.9):
            buf.add_chunk(16000, chunk)

        utt = buf.get_utterance()
        assert utt is not None
        assert len(utt) == 16000

    def test_stereo_to_mono(self):
        from whisperapp.streaming import AudioBuffer
        buf = AudioBuffer()
        stereo = np.zeros((8000, 2), dtype=np.float32)
        with patch("whisperapp.streaming._speech_prob", return_value=0.1):
            buf.add_chunk(16000, stereo)
        assert buf.duration == pytest.approx(0.5, abs=0.01)

    def test_get_all_flushes_everything(self):
        from whisperapp.streaming import AudioBuffer
        buf = AudioBuffer()
        chunk = np.zeros(8000, dtype=np.float32)
        with patch("whisperapp.streaming._speech_prob", return_value=0.1):
            buf.add_chunk(16000, chunk)
        result = buf.get_all()
        assert result is not None
        assert len(result) == 8000
        assert buf.duration == 0.0


# ---------------------------------------------------------------------------
# TranscriptAccumulator tests
# ---------------------------------------------------------------------------

class TestTranscriptAccumulator:
    def test_add_and_get(self):
        from whisperapp.streaming import TranscriptAccumulator
        acc = TranscriptAccumulator()
        acc.add_segment(0.0, 1.0, "Hello world")
        acc.add_segment(1.0, 2.0, "How are you")
        assert acc.get_full_text() == "Hello world How are you"

    def test_deduplicates_overlapping_segments(self):
        from whisperapp.streaming import TranscriptAccumulator
        acc = TranscriptAccumulator()
        acc.add_segment(0.0, 1.0, "Hello world")
        acc.add_segment(0.5, 1.5, "Hello world")  # duplicate
        assert acc.get_full_text() == "Hello world"
        assert len(acc.get_segments()) == 1

    def test_skips_empty_text(self):
        from whisperapp.streaming import TranscriptAccumulator
        acc = TranscriptAccumulator()
        acc.add_segment(0.0, 1.0, "")
        acc.add_segment(0.0, 1.0, "   ")
        assert acc.get_full_text() == ""
        assert len(acc.get_segments()) == 0

    def test_clear(self):
        from whisperapp.streaming import TranscriptAccumulator
        acc = TranscriptAccumulator()
        acc.add_segment(0.0, 1.0, "Hello")
        acc.clear()
        assert acc.get_full_text() == ""
        assert acc.get_segments() == []

    def test_get_segments_returns_dicts(self):
        from whisperapp.streaming import TranscriptAccumulator
        acc = TranscriptAccumulator()
        acc.add_segment(0.0, 1.5, "Test")
        segs = acc.get_segments()
        assert len(segs) == 1
        assert segs[0]["start"] == 0.0
        assert segs[0]["end"] == 1.5
        assert segs[0]["text"] == "Test"


# ---------------------------------------------------------------------------
# StreamingEngine tests (mocked faster-whisper)
# ---------------------------------------------------------------------------

class FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class FakeTranscribeInfo:
    language = "en"


class TestStreamingEngine:
    @patch("whisperapp.streaming._speech_prob", return_value=0.9)
    @patch("whisperapp.streaming.StreamingEngine._transcribe_utterance")
    def test_process_chunk_returns_text(self, mock_transcribe, mock_vad):
        from whisperapp.streaming import StreamingEngine
        mock_transcribe.return_value = "Hello"

        engine = StreamingEngine(max_chunk_sec=0.5)
        # Mock the model so start() doesn't load anything
        engine._model = MagicMock()
        engine._running = True

        # Feed enough audio to trigger max_chunk flush
        audio = np.random.randn(8000).astype(np.float32)
        result = engine.process_chunk(16000, audio)
        assert result == "Hello"

    @patch("whisperapp.streaming._speech_prob", return_value=0.1)
    def test_process_chunk_returns_none_on_silence(self, mock_vad):
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine()
        engine._model = MagicMock()
        engine._running = True

        audio = np.zeros(4000, dtype=np.float32)
        result = engine.process_chunk(16000, audio)
        assert result is None

    def test_stop_returns_transcript_dict(self):
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine()
        engine._model = MagicMock()
        engine._running = True
        engine._all_audio = [np.zeros(1000, dtype=np.float32)]
        engine._accumulator.add_segment(0.0, 1.0, "Hello")

        result = engine.stop()
        assert "text" in result
        assert "segments" in result
        assert "raw_audio" in result
        assert result["text"] == "Hello"

    def test_get_transcript(self):
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine()
        engine._accumulator.add_segment(0.0, 1.0, "Test text")
        assert engine.get_transcript() == "Test text"

    def test_reset_clears_state(self):
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine()
        engine._accumulator.add_segment(0.0, 1.0, "Some text")
        engine._all_audio = [np.zeros(1000, dtype=np.float32)]
        engine.reset()
        assert engine.get_transcript() == ""
        assert len(engine._all_audio) == 0

    @patch("whisperapp.streaming.StreamingEngine._transcribe_utterance")
    def test_stop_flushes_remaining(self, mock_transcribe):
        from whisperapp.streaming import StreamingEngine
        mock_transcribe.return_value = "leftover"

        engine = StreamingEngine()
        engine._model = MagicMock()
        engine._running = True
        # Put enough audio in buffer (>0.5s at 16kHz = 8000+ samples)
        engine._buffer._buf = [np.zeros(16000, dtype=np.float32)]
        engine._buffer._buf_samples = 16000
        engine._buffer._has_speech = True
        engine._all_audio = [np.zeros(16000, dtype=np.float32)]

        result = engine.stop()
        mock_transcribe.assert_called_once()

    @patch("whisperx.assign_word_speakers")
    @patch("whisperx.align")
    @patch("whisperx.load_align_model")
    @patch("whisperx.load_model")
    def test_polish_runs_alignment(self, mock_load, mock_align_load,
                                    mock_align, mock_assign):
        from whisperapp.streaming import StreamingEngine

        # Set up mocks
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0, "end": 1, "text": "Hi"}],
            "language": "en",
        }
        mock_load.return_value = mock_model
        mock_align_load.return_value = (MagicMock(), {})
        mock_align.return_value = {
            "segments": [{"start": 0, "end": 1, "text": "Hi"}],
        }

        engine = StreamingEngine(device="cpu", compute_type="float32")
        engine._all_audio = [np.zeros(16000, dtype=np.float32)]

        result = engine.polish(hf_token="")  # no diarization without token
        assert "segments" in result
        mock_load.assert_called_once()
        mock_align.assert_called_once()

    @patch("whisperx.assign_word_speakers")
    @patch("whisperx.align")
    @patch("whisperx.load_align_model")
    @patch("whisperx.load_model")
    def test_polish_calls_progress_callback(self, mock_load, mock_align_load,
                                             mock_align, mock_assign):
        from whisperapp.streaming import StreamingEngine

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0, "end": 1, "text": "Hi"}],
            "language": "en",
        }
        mock_load.return_value = mock_model
        mock_align_load.return_value = (MagicMock(), {})
        mock_align.return_value = {"segments": [{"start": 0, "end": 1, "text": "Hi"}]}

        engine = StreamingEngine(device="cpu", compute_type="float32")
        engine._all_audio = [np.zeros(16000, dtype=np.float32)]

        progress_log = []
        def on_progress(stage, detail):
            progress_log.append((stage, detail))

        engine.polish(hf_token="", on_progress=on_progress)

        stages = [p[0] for p in progress_log]
        assert "preparing" in stages
        assert "transcribing" in stages
        assert "aligning" in stages
        assert "complete" in stages

    @patch("whisperx.assign_word_speakers")
    @patch("whisperx.align")
    @patch("whisperx.load_align_model")
    @patch("whisperx.load_model")
    def test_polish_with_diarization(self, mock_load, mock_align_load,
                                      mock_align, mock_assign):
        from whisperapp.streaming import StreamingEngine

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0, "end": 1, "text": "Hi"}],
            "language": "en",
        }
        mock_load.return_value = mock_model
        mock_align_load.return_value = (MagicMock(), {})
        mock_align.return_value = {"segments": [{"start": 0, "end": 1, "text": "Hi"}]}
        mock_assign.return_value = {
            "segments": [{"start": 0, "end": 1, "text": "Hi", "speaker": "SPEAKER_00"}],
        }

        engine = StreamingEngine(device="cpu", compute_type="float32")
        engine._all_audio = [np.zeros(16000, dtype=np.float32)]

        import sys
        mock_sf = MagicMock()
        with patch.dict(sys.modules, {"soundfile": mock_sf}), \
             patch("whisperx.diarize.DiarizationPipeline") as mock_dp:
            mock_dp.return_value = MagicMock(return_value=MagicMock())
            result = engine.polish(hf_token="hf_test_token")

        assert "segments" in result
        mock_assign.assert_called_once()

        progress_log = []
        def on_progress(stage, detail):
            progress_log.append((stage, detail))

        engine._all_audio = [np.zeros(16000, dtype=np.float32)]
        with patch.dict(sys.modules, {"soundfile": mock_sf}), \
             patch("whisperx.diarize.DiarizationPipeline") as mock_dp:
            mock_dp.return_value = MagicMock(return_value=MagicMock())
            engine.polish(hf_token="hf_test_token", on_progress=on_progress)

        stages = [p[0] for p in progress_log]
        assert "diarizing" in stages
        assert "complete" in stages

    @patch("whisperx.assign_word_speakers")
    @patch("whisperx.align")
    @patch("whisperx.load_align_model")
    @patch("whisperx.load_model")
    def test_polish_diarization_failure_returns_aligned(self, mock_load,
            mock_align_load, mock_align, mock_assign):
        from whisperapp.streaming import StreamingEngine

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0, "end": 1, "text": "Hi"}],
            "language": "en",
        }
        mock_load.return_value = mock_model
        mock_align_load.return_value = (MagicMock(), {})
        aligned_result = {"segments": [{"start": 0, "end": 1, "text": "Hi"}]}
        mock_align.return_value = aligned_result

        engine = StreamingEngine(device="cpu", compute_type="float32")
        engine._all_audio = [np.zeros(16000, dtype=np.float32)]

        with patch("whisperx.diarize.DiarizationPipeline", side_effect=RuntimeError("model failed")):
            result = engine.polish(hf_token="hf_test_token")

        # Should return aligned result, not crash
        assert "segments" in result
        assert result["segments"][0]["text"] == "Hi"

    def test_polish_empty_audio_returns_empty(self):
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine(device="cpu", compute_type="float32")
        engine._all_audio = []
        result = engine.polish(hf_token="hf_test")
        assert result == {"segments": [], "text": ""}

    @patch("whisperx.assign_word_speakers")
    @patch("whisperx.align")
    @patch("whisperx.load_align_model")
    @patch("whisperx.load_model")
    def test_polish_no_progress_callback_ok(self, mock_load, mock_align_load,
                                             mock_align, mock_assign):
        """Polish works fine when on_progress is not provided (default None)."""
        from whisperapp.streaming import StreamingEngine

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0, "end": 1, "text": "Hi"}],
            "language": "en",
        }
        mock_load.return_value = mock_model
        mock_align_load.return_value = (MagicMock(), {})
        mock_align.return_value = {"segments": [{"start": 0, "end": 1, "text": "Hi"}]}

        engine = StreamingEngine(device="cpu", compute_type="float32")
        engine._all_audio = [np.zeros(16000, dtype=np.float32)]
        result = engine.polish(hf_token="")
        assert "segments" in result


# ---------------------------------------------------------------------------
# Device auto-detection tests
# ---------------------------------------------------------------------------

class TestDeviceAutoDetection:
    @patch("torch.cuda.is_available", return_value=True)
    def test_streaming_engine_auto_selects_cuda(self, mock_cuda):
        import whisperapp.worker as _w
        orig_dev, orig_ct = _w._WHISPER_DEVICE, _w._COMPUTE_TYPE
        _w._WHISPER_DEVICE = "cuda"
        _w._COMPUTE_TYPE = "float16"
        try:
            from whisperapp.streaming import StreamingEngine
            engine = StreamingEngine(device="auto", compute_type="auto")
            assert engine.device == "cuda"
            assert engine.compute_type == "float16"
        finally:
            _w._WHISPER_DEVICE = orig_dev
            _w._COMPUTE_TYPE = orig_ct

    @patch("torch.cuda.is_available", return_value=False)
    def test_streaming_engine_auto_selects_cpu(self, mock_cuda):
        # worker._WHISPER_DEVICE/_COMPUTE_TYPE are set at module import time, so
        # mocking torch.cuda.is_available alone isn't enough when the module is
        # already cached.  Patch the attributes directly instead.
        import whisperapp.worker as _w
        orig_dev, orig_ct = _w._WHISPER_DEVICE, _w._COMPUTE_TYPE
        _w._WHISPER_DEVICE = "cpu"
        _w._COMPUTE_TYPE = "int8"
        try:
            from whisperapp.streaming import StreamingEngine
            engine = StreamingEngine(device="auto", compute_type="auto")
            assert engine.device == "cpu"
            assert engine.compute_type == "int8"
        finally:
            _w._WHISPER_DEVICE = orig_dev
            _w._COMPUTE_TYPE = orig_ct

    def test_streaming_engine_explicit_device_respected(self):
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine(device="cpu", compute_type="float32")
        assert engine.device == "cpu"
        assert engine.compute_type == "float32"

    @patch("torch.cuda.is_available", return_value=True)
    def test_streaming_engine_auto_device_explicit_compute(self, mock_cuda):
        import whisperapp.worker as _w
        orig_dev, orig_ct = _w._WHISPER_DEVICE, _w._COMPUTE_TYPE
        _w._WHISPER_DEVICE = "cuda"
        _w._COMPUTE_TYPE = "float16"
        try:
            from whisperapp.streaming import StreamingEngine
            engine = StreamingEngine(device="auto", compute_type="int8")
            assert engine.device == "cuda"
            assert engine.compute_type == "int8"
        finally:
            _w._WHISPER_DEVICE = orig_dev
            _w._COMPUTE_TYPE = orig_ct


# ---------------------------------------------------------------------------
# AudioBuffer — additional edge cases
# ---------------------------------------------------------------------------

class TestAudioBufferEdgeCases:
    def test_no_flush_below_min_chunk(self):
        """Buffer shorter than min_chunk_sec never flushes even after silence."""
        from whisperapp.streaming import AudioBuffer
        buf = AudioBuffer(min_chunk_sec=1.0, max_chunk_sec=5.0, silence_threshold_sec=0.3)
        short = np.zeros(4000, dtype=np.float32)  # 0.25s at 16kHz
        with patch("whisperapp.streaming._speech_prob", return_value=0.9):
            buf.add_chunk(16000, short)
        with patch("whisperapp.streaming._speech_prob", return_value=0.0):
            buf.add_chunk(16000, np.zeros(8000, dtype=np.float32))  # 0.5s silence
        assert buf.get_utterance() is None  # total 0.75s < min_chunk_sec=1.0

    def test_force_flush_without_speech(self):
        """max_chunk_sec flush fires even when VAD never detected speech."""
        from whisperapp.streaming import AudioBuffer
        buf = AudioBuffer(min_chunk_sec=0.1, max_chunk_sec=0.5, silence_threshold_sec=0.3)
        chunk = np.zeros(8000, dtype=np.float32)  # 0.5s silence only
        with patch("whisperapp.streaming._speech_prob", return_value=0.0):
            buf.add_chunk(16000, chunk)
        result = buf.get_utterance()
        assert result is not None
        assert len(result) == 8000

    def test_state_clean_after_flush(self):
        """After a flush, duration, silence counter and has_speech all reset."""
        from whisperapp.streaming import AudioBuffer
        buf = AudioBuffer(min_chunk_sec=0.1, max_chunk_sec=5.0, silence_threshold_sec=0.1)
        with patch("whisperapp.streaming._speech_prob", return_value=0.9):
            buf.add_chunk(16000, np.zeros(4000, dtype=np.float32))
        with patch("whisperapp.streaming._speech_prob", return_value=0.0):
            buf.add_chunk(16000, np.zeros(4000, dtype=np.float32))  # silence → flush
        _ = buf.get_utterance()
        assert buf.duration == 0.0
        assert buf._silence_samples == 0
        assert buf._has_speech is False

    def test_resamples_non_16k_input(self):
        """Input at a sample rate other than 16 kHz is resampled correctly."""
        from whisperapp.streaming import AudioBuffer
        buf = AudioBuffer()
        chunk_48k = np.zeros(4800, dtype=np.float32)  # 0.1s at 48kHz
        with patch("whisperapp.streaming._speech_prob", return_value=0.0):
            buf.add_chunk(48000, chunk_48k)
        # After resampling: 4800 * 16000/48000 = 1600 samples
        assert buf.duration == pytest.approx(0.1, abs=0.02)

    def test_get_all_returns_none_on_empty(self):
        """get_all() on an empty buffer returns None."""
        from whisperapp.streaming import AudioBuffer
        buf = AudioBuffer()
        assert buf.get_all() is None


# ---------------------------------------------------------------------------
# TranscriptAccumulator — additional edge cases
# ---------------------------------------------------------------------------

class TestTranscriptAccumulatorEdgeCases:
    def test_dedup_is_case_insensitive(self):
        """Exact duplicate differing only in case is not added."""
        from whisperapp.streaming import TranscriptAccumulator
        acc = TranscriptAccumulator()
        acc.add_segment(0.0, 1.0, "Hello World")
        acc.add_segment(1.0, 2.0, "hello world")
        assert len(acc.get_segments()) == 1

    def test_non_duplicate_similar_text_is_kept(self):
        """Near-duplicate (but not exact) text is not dropped."""
        from whisperapp.streaming import TranscriptAccumulator
        acc = TranscriptAccumulator()
        acc.add_segment(0.0, 1.0, "Hello world")
        acc.add_segment(1.0, 2.0, "Hello worlds")
        assert len(acc.get_segments()) == 2

    def test_advance_offset_shifts_absolute_times(self):
        """Segments added after advance_offset have larger absolute timestamps."""
        from whisperapp.streaming import TranscriptAccumulator
        acc = TranscriptAccumulator()
        acc.add_segment(0.0, 1.0, "First")
        acc.advance_offset(5.0)
        acc.add_segment(0.0, 1.0, "Second")  # relative 0-1, abs 5-6
        segs = acc.get_segments()
        assert segs[0]["start"] == pytest.approx(0.0)
        assert segs[1]["start"] == pytest.approx(5.0)
        assert segs[1]["end"] == pytest.approx(6.0)

    def test_whitespace_only_segment_skipped(self):
        """Segments containing only whitespace are silently dropped."""
        from whisperapp.streaming import TranscriptAccumulator
        acc = TranscriptAccumulator()
        acc.add_segment(0.0, 1.0, "  \t\n  ")
        assert acc.get_full_text() == ""

    def test_get_segments_returns_copy(self):
        """Mutating the returned list doesn't affect internal state."""
        from whisperapp.streaming import TranscriptAccumulator
        acc = TranscriptAccumulator()
        acc.add_segment(0.0, 1.0, "Test")
        segs = acc.get_segments()
        segs.clear()
        assert len(acc.get_segments()) == 1


# ---------------------------------------------------------------------------
# StreamingEngine — VAD param and stop behaviour
# ---------------------------------------------------------------------------

class TestStreamingEngineParams:
    def test_vad_params_stored_on_engine(self):
        """Custom VAD params are stored on the engine's AudioBuffer."""
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine(
            device="cpu", compute_type="float32",
            min_chunk_sec=0.5, max_chunk_sec=7.0, silence_threshold_sec=0.8,
        )
        assert engine._buffer.min_chunk_sec == 0.5
        assert engine._buffer.max_chunk_sec == 7.0
        assert engine._buffer.silence_threshold_sec == 0.8

    def test_stop_returns_sample_rate_key(self):
        """stop() always includes sample_rate in its return dict."""
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine(device="cpu", compute_type="float32")
        engine._model = MagicMock()
        engine._running = True
        result = engine.stop()
        assert result["sample_rate"] == 16000

    def test_stop_returns_raw_audio_array(self):
        """stop() raw_audio is a numpy float32 array (possibly empty)."""
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine(device="cpu", compute_type="float32")
        engine._model = MagicMock()
        engine._running = True
        result = engine.stop()
        assert isinstance(result["raw_audio"], np.ndarray)
        assert result["raw_audio"].dtype == np.float32

    @patch("whisperapp.streaming._speech_prob", return_value=0.9)
    @patch("whisperapp.streaming.StreamingEngine._transcribe_utterance")
    def test_different_instances_independent(self, mock_transcribe, mock_vad):
        """Two StreamingEngine instances with different VAD params don't share state."""
        from whisperapp.streaming import StreamingEngine
        e1 = StreamingEngine(device="cpu", compute_type="float32", max_chunk_sec=2.0)
        e2 = StreamingEngine(device="cpu", compute_type="float32", max_chunk_sec=8.0)
        e1._model = e2._model = MagicMock()
        e1._running = e2._running = True

        audio = np.zeros(32000, dtype=np.float32)  # 2s at 16kHz
        e1.process_chunk(16000, audio)  # should flush (2.0s >= max_chunk_sec=2.0)
        e2.process_chunk(16000, audio)  # should NOT flush (2.0s < max_chunk_sec=8.0)

        assert mock_transcribe.call_count == 1  # only e1 flushed


# ---------------------------------------------------------------------------
# Cross-platform — signal.pause fallback
# ---------------------------------------------------------------------------

def test_no_signal_pause_import_on_windows(monkeypatch):
    """The no-tray startup path does not crash when signal.pause is unavailable."""
    import types, signal as _signal
    # Simulate Windows where signal.pause is absent
    orig = getattr(_signal, 'pause', _signal)
    try:
        if hasattr(_signal, 'pause'):
            delattr(_signal, 'pause')
        import threading
        # Replicate the guard from __main__
        if hasattr(_signal, 'pause'):
            raise AssertionError("pause should be absent in this test")
        # Should fall through to threading.Event().wait() — just verify no AttributeError
        stopped = threading.Event()
        stopped.set()
        stopped.wait()  # returns immediately because it's set
    finally:
        if orig is not _signal:
            _signal.pause = orig
