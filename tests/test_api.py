"""
Basic API tests for Voice Backend
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from voice.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint returns service information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["service"] == "Unified Voice Service API"
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self):
        """Test detailed health check endpoint"""
        response = client.get("/api/voice/health")
        assert response.status_code in [200, 503]  # 503 if service not initialized

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "healthy"
            assert "device" in data
            assert "models" in data
            assert "resources" in data


class TestTTSEndpoints:
    """Test TTS-related endpoints"""

    def test_get_providers(self):
        """Test getting available TTS providers"""
        response = client.get("/api/voice/tts/providers")
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "providers" in data
            assert "current" in data
            assert "available_voices" in data

    def test_get_voices(self):
        """Test getting available voices"""
        response = client.get("/api/voice/tts/voices")
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "voices" in data

    def test_tts_request_validation(self):
        """Test TTS request validation"""
        # Test with invalid text (empty)
        response = client.post("/api/voice/tts", json={"text": "", "language": "en"})
        assert response.status_code == 422  # Validation error

        # Test with invalid language code
        response = client.post(
            "/api/voice/tts", json={"text": "Hello", "language": "invalid"}
        )
        assert response.status_code == 422

        # Test with text too long
        response = client.post(
            "/api/voice/tts", json={"text": "a" * 10000, "language": "en"}
        )
        assert response.status_code == 422

        # Test with invalid speed
        response = client.post(
            "/api/voice/tts", json={"text": "Hello", "language": "en", "speed": 5.0}
        )
        assert response.status_code == 422

    def test_tts_provider_validation(self):
        """Test TTS provider validation"""
        response = client.post(
            "/api/voice/tts",
            json={"text": "Hello", "language": "en", "provider": "invalid-provider"},
        )
        assert response.status_code == 422


class TestUtilities:
    """Test shared utilities"""

    def test_audio_processor_normalize(self):
        """Test audio normalization"""
        from voice.utils.audio import AudioProcessor
        import numpy as np

        # Test with values > 1
        audio = np.array([2.0, -2.0, 1.5], dtype=np.float32)
        normalized = AudioProcessor.normalize_to_float32(audio)
        assert np.abs(normalized).max() <= 1.0
        assert normalized.dtype == np.float32

    def test_audio_processor_resample(self):
        """Test audio resampling"""
        from voice.utils.audio import AudioProcessor
        import numpy as np

        # Test same sample rate (no-op)
        audio = np.random.randn(1000).astype(np.float32)
        resampled = AudioProcessor.resample_audio(audio, 24000, 24000)
        assert len(resampled) == len(audio)

    def test_audio_processor_validate(self):
        """Test audio validation"""
        from voice.utils.audio import AudioProcessor
        import numpy as np

        # Valid audio
        audio = np.random.randn(1000).astype(np.float32)
        assert AudioProcessor.validate_audio(audio) is True

        # Too short
        audio_short = np.random.randn(10).astype(np.float32)
        assert AudioProcessor.validate_audio(audio_short, min_length=100) is False

        # All zeros
        audio_silent = np.zeros(1000, dtype=np.float32)
        assert AudioProcessor.validate_audio(audio_silent) is False

    def test_text_processor_preprocess(self):
        """Test text preprocessing"""
        from voice.utils.text import TextProcessor

        text = "**Bold** text with  extra   spaces and http://example.com"
        cleaned = TextProcessor.preprocess_text(text)

        assert "**" not in cleaned  # Markdown removed
        assert "  " not in cleaned  # Extra spaces removed
        assert "http://example.com" not in cleaned  # URL removed

    def test_text_processor_split_sentences(self):
        """Test sentence splitting"""
        from voice.utils.text import TextProcessor

        text = "First sentence. Second sentence! Third sentence?"
        sentences = TextProcessor.split_sentences(text)

        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]

    def test_device_config(self):
        """Test device configuration"""
        from voice.utils.device import get_device_config, check_gpu_availability

        device, quantization = get_device_config()
        assert device in ["cuda", "mps", "cpu"]
        assert isinstance(quantization, bool)

        gpu_available = check_gpu_availability()
        assert isinstance(gpu_available, bool)


class TestMiddleware:
    """Test middleware functionality"""

    def test_request_tracking(self):
        """Test that requests get tracking headers"""
        response = client.get("/")

        # Check for tracking headers
        assert "X-Request-ID" in response.headers or response.status_code == 200
        assert "X-Process-Time" in response.headers or response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
