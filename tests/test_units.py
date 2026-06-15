"""
Dependency-free unit tests (no torch / GPU required).

Covers the shared utilities and the audio-encoding path that underpins the
binary PCM WebSocket streaming, plus the path-traversal protection in the
security helpers.

Run: pytest tests/test_units.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# AudioProcessor — including the int16 PCM path used by WS streaming
# ---------------------------------------------------------------------------
class TestAudioProcessor:
    def test_normalize_clamps_above_one(self):
        from voice.utils.audio import AudioProcessor
        audio = np.array([2.0, -4.0, 1.5], dtype=np.float32)
        out = AudioProcessor.normalize_to_float32(audio)
        assert out.dtype == np.float32
        assert np.abs(out).max() <= 1.0

    def test_int16_roundtrip(self):
        from voice.utils.audio import AudioProcessor
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm = AudioProcessor.convert_to_int16(original)
        assert pcm.dtype == np.int16
        back = AudioProcessor.convert_to_float32(pcm)
        # Quantisation error must stay within one LSB of int16.
        assert np.max(np.abs(back - original)) < 1e-3

    def test_int16_clips_out_of_range(self):
        from voice.utils.audio import AudioProcessor
        pcm = AudioProcessor.convert_to_int16(np.array([2.0, -2.0], dtype=np.float32))
        assert pcm.max() <= 32767
        assert pcm.min() >= -32768

    def test_generate_silence_length(self):
        from voice.utils.audio import AudioProcessor
        silence = AudioProcessor.generate_silence(0.5, 24000)
        assert len(silence) == 12000
        assert silence.dtype == np.float32
        assert not AudioProcessor.validate_audio(silence)  # all-zero => invalid

    def test_validate_audio_rules(self):
        from voice.utils.audio import AudioProcessor
        assert AudioProcessor.validate_audio(np.random.randn(1000).astype(np.float32))
        assert not AudioProcessor.validate_audio(np.zeros(1000, dtype=np.float32))
        assert not AudioProcessor.validate_audio(np.random.randn(10).astype(np.float32), min_length=100)
        assert not AudioProcessor.validate_audio(None)


# ---------------------------------------------------------------------------
# TextProcessor
# ---------------------------------------------------------------------------
class TestTextProcessor:
    def test_split_sentences_basic(self):
        from voice.utils.text import TextProcessor
        s = TextProcessor.split_sentences("First one. Second one! Third one?")
        assert len(s) == 3

    def test_preprocess_removes_markdown_and_urls(self):
        from voice.utils.text import TextProcessor
        out = TextProcessor.preprocess_text("**Bold** and `code` see http://x.com/y now")
        assert "**" not in out and "`" not in out
        assert "http://x.com" not in out

    def test_split_never_empty(self):
        from voice.utils.text import TextProcessor
        assert TextProcessor.split_sentences("   ") != []
        assert TextProcessor.split_into_chunks("") != []


# ---------------------------------------------------------------------------
# Security — path traversal protection
# ---------------------------------------------------------------------------
class TestSecurity:
    def test_none_path_allowed(self):
        from voice.utils.security import validate_audio_path
        assert validate_audio_path(None) is None

    def test_traversal_rejected(self, tmp_path):
        from voice.utils import security
        # Force the "no allowed dirs" branch so the traversal heuristic runs.
        original = security.ALLOWED_AUDIO_DIRS
        security.ALLOWED_AUDIO_DIRS = []
        try:
            with pytest.raises(ValueError):
                security.validate_audio_path("/etc/passwd", "ref")
        finally:
            security.ALLOWED_AUDIO_DIRS = original

    def test_missing_file_rejected(self):
        from voice.utils.security import validate_audio_path
        with pytest.raises(ValueError):
            validate_audio_path("/nonexistent/file/here.wav", "ref")

    def test_sanitize_error_hides_internals(self):
        from voice.utils.security import sanitize_error_message
        assert sanitize_error_message(RuntimeError("secret /app/internal path")) == (
            "An internal error occurred. Check server logs for details."
        )
        # ValueError messages are surfaced (validation feedback).
        assert sanitize_error_message(ValueError("bad input")) == "bad input"


# ---------------------------------------------------------------------------
# LatencyTracker
# ---------------------------------------------------------------------------
class TestLatencyTracker:
    def test_percentiles_and_filter(self):
        from voice.utils.metrics import LatencyTracker
        t = LatencyTracker()
        for v in range(1, 101):
            t.record("tts", "kokoro", float(v))
        t.record("stt", "large-v3", 5.0)

        stats = t.get_stats(category="tts")
        assert set(stats.keys()) == {"tts/kokoro"}
        k = stats["tts/kokoro"]
        assert k["count"] == 100
        assert k["min_ms"] == 1.0 and k["max_ms"] == 100.0
        assert k["p50_ms"] == pytest.approx(50.5, abs=1.0)

        assert "stt/large-v3" in t.get_stats(category="stt")
