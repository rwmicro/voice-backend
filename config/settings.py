"""
Voice Backend Configuration
Lightweight configuration for TTS/STT services
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Voice service settings"""

    # ===========================
    # Server Configuration
    # ===========================
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    LOG_LEVEL: str = "INFO"

    # ===========================
    # TTS Configuration
    # ===========================
    TTS_DEFAULT_PROVIDER: str = "kokoro"  # kokoro, chatterbox, f5-tts
    TTS_DEVICE: str = "cuda"  # cuda, cpu, auto
    TTS_USE_FP16: bool = True

    # Provider-specific settings
    KOKORO_MODEL: str = "hexgrad/Kokoro-82M"
    CHATTERBOX_MODEL: str = "resemble-ai/chatterbox"
    F5_TTS_MODEL: str = "f5-tts"

    # ===========================
    # STT Configuration
    # ===========================
    STT_MODEL: str = "openai/whisper-small"
    STT_DEVICE: str = "cuda"
    STT_USE_FP16: bool = True
    STT_LANGUAGE: Optional[str] = None  # None for auto-detect

    # ===========================
    # GPU Management
    # ===========================
    ENABLE_GPU_QUEUE: bool = False  # Set to true for GPU coordination
    GPU_MAX_VRAM_MB: int = 6000
    GPU_IDLE_TIMEOUT_SECONDS: int = 300

    # Ollama coordination (if ENABLE_GPU_QUEUE is true)
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # ===========================
    # Audio Configuration
    # ===========================
    SAMPLE_RATE: int = 24000
    CHUNK_SIZE: int = 1024
    AUDIO_FORMAT: str = "wav"  # wav, mp3, ogg

    # ===========================
    # Data Paths
    # ===========================
    DATA_DIR: str = "./data"
    VOICE_CACHE_DIR: str = "./data/voice_cache"
    AUDIO_PROMPTS_DIR: str = "./voice/audio_prompts"

    # ===========================
    # Logs
    # ===========================
    LOG_DIR: str = "./logs"
    LOG_FILE: str = "./logs/voice-backend.log"

    # ===========================
    # Performance
    # ===========================
    MAX_WORKERS: int = 4

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()


def ensure_directories():
    """Create necessary directories"""
    directories = [
        settings.DATA_DIR,
        settings.VOICE_CACHE_DIR,
        settings.LOG_DIR,
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
