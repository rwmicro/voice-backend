"""
Shared audio processing utilities
Eliminates code duplication across TTS providers
"""

import numpy as np
from typing import Optional


class AudioProcessor:
    """Centralized audio processing utilities"""

    @staticmethod
    def normalize_to_float32(audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to float32 format in range [-1, 1]

        Args:
            audio: Input audio array

        Returns:
            Normalized float32 audio array
        """
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize to [-1, 1] range if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        return audio

    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate

        Args:
            audio: Input audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio

        try:
            import librosa

            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            from .logger import get_logger

            logger = get_logger(__name__)
            logger.warning(
                f"librosa not installed, cannot resample from {orig_sr}Hz to {target_sr}Hz. "
                "Returning original audio. Install with: pip install librosa"
            )
            return audio

    @staticmethod
    def validate_audio(audio: np.ndarray, min_length: int = 100) -> bool:
        """
        Validate audio array

        Args:
            audio: Audio array to validate
            min_length: Minimum required length in samples

        Returns:
            True if audio is valid, False otherwise
        """
        if audio is None:
            return False

        if not isinstance(audio, np.ndarray):
            return False

        if len(audio) < min_length:
            return False

        if np.all(audio == 0):
            return False

        return True

    @staticmethod
    def ensure_mono(audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mono if stereo

        Args:
            audio: Input audio array

        Returns:
            Mono audio array
        """
        if len(audio.shape) > 1:
            # Convert stereo to mono by averaging channels
            return audio.mean(axis=1)
        return audio

    @staticmethod
    def convert_to_int16(audio: np.ndarray) -> np.ndarray:
        """
        Convert float32 audio to int16

        Args:
            audio: Float32 audio in range [-1, 1]

        Returns:
            Int16 audio array
        """
        # Clip to [-1, 1] range
        audio = np.clip(audio, -1.0, 1.0)

        # Convert to int16
        return (audio * 32767).astype(np.int16)

    @staticmethod
    def convert_to_float32(audio: np.ndarray) -> np.ndarray:
        """
        Convert int16 audio to float32

        Args:
            audio: Int16 audio array

        Returns:
            Float32 audio in range [-1, 1]
        """
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32)

    @staticmethod
    def generate_silence(duration_seconds: float, sample_rate: int) -> np.ndarray:
        """
        Generate silence

        Args:
            duration_seconds: Duration in seconds
            sample_rate: Sample rate in Hz

        Returns:
            Silence audio array
        """
        num_samples = int(duration_seconds * sample_rate)
        return np.zeros(num_samples, dtype=np.float32)
