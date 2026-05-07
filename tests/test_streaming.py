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

        with patch("whisperx.diarize.DiarizationPipeline") as mock_dp, \
             patch("soundfile.write"):
            mock_dp.return_value = MagicMock(return_value=MagicMock())
            result = engine.polish(hf_token="hf_test_token")

        assert "segments" in result
        mock_assign.assert_called_once()

        progress_log = []
        def on_progress(stage, detail):
            progress_log.append((stage, detail))

        engine._all_audio = [np.zeros(16000, dtype=np.float32)]
        with patch("whisperx.diarize.DiarizationPipeline") as mock_dp, \
             patch("soundfile.write"):
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
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine(device="auto", compute_type="auto")
        assert engine.device == "cuda"
        assert engine.compute_type == "float16"

    @patch("torch.cuda.is_available", return_value=False)
    def test_streaming_engine_auto_selects_cpu(self, mock_cuda):
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine(device="auto", compute_type="auto")
        assert engine.device == "cpu"
        # CPU auto-detection now uses int8 (faster than float32 on modern CPUs)
        assert engine.compute_type == "int8"

    def test_streaming_engine_explicit_device_respected(self):
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine(device="cpu", compute_type="float32")
        assert engine.device == "cpu"
        assert engine.compute_type == "float32"

    @patch("torch.cuda.is_available", return_value=True)
    def test_streaming_engine_auto_device_explicit_compute(self, mock_cuda):
        from whisperapp.streaming import StreamingEngine
        engine = StreamingEngine(device="auto", compute_type="int8")
        assert engine.device == "cuda"
        assert engine.compute_type == "int8"
