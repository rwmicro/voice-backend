"""
Base TTS Provider Interface
All TTS providers must implement this interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, List, Optional, Dict, Any
import numpy as np


@dataclass
class TTSConfig:
    """Configuration for TTS providers"""
    device: str = "cuda"  # cuda, cpu, mps
    use_quantization: bool = False
    sample_rate: int = 24000
    torch_dtype: Any = None  # torch.float16, torch.float32, etc.


@dataclass
class Voice:
    """Voice metadata"""
    id: str
    name: str
    language: str
    gender: str  # male, female, neutral
    description: str = ""
    accent: Optional[str] = None
    quality_grade: Optional[str] = None  # A, B, C, D


class TTSProvider(ABC):
    """Abstract base class for TTS providers"""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.is_initialized = False

    @abstractmethod
    async def initialize(self):
        """Initialize the TTS model(s)"""
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str = "en",
        **kwargs
    ) -> Generator[np.ndarray, None, None]:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            voice_id: Voice ID to use
            language: Language code
            **kwargs: Provider-specific parameters

        Yields:
            Audio chunks as numpy arrays (float32, sample_rate from config)
        """
        pass

    @abstractmethod
    def list_voices(self, language: Optional[str] = None) -> List[Voice]:
        """
        List available voices

        Args:
            language: Filter by language (optional)

        Returns:
            List of available voices
        """
        pass

    @abstractmethod
    def get_default_voice(self, language: str) -> str:
        """
        Get default voice ID for a language

        Args:
            language: Language code

        Returns:
            Default voice ID
        """
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup resources"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Provider description"""
        pass

    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """List of supported language codes"""
        pass

    def supports_language(self, language: str) -> bool:
        """Check if language is supported"""
        return language in self.supported_languages

    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": self.name,
            "description": self.description,
            "supported_languages": self.supported_languages,
            "voices": [
                {
                    "id": v.id,
                    "name": v.name,
                    "language": v.language,
                    "gender": v.gender,
                    "accent": v.accent,
                    "quality_grade": v.quality_grade,
                }
                for v in self.list_voices()
            ],
            "config": {
                "device": self.config.device,
                "sample_rate": self.config.sample_rate,
                "use_quantization": self.config.use_quantization,
            },
            "initialized": self.is_initialized,
        }