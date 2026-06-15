"""
Voice Backend Configuration
Lightweight configuration for TTS/STT services
"""
from pydantic_settings import BaseSettings
from pydantic import field_validator
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
    TTS_DEFAULT_PROVIDER: str = "kokoro"  # kokoro, chatterbox, f5-tts, qwen3-tts, orpheus, dia
    TTS_DEVICE: str = "cuda"  # cuda, cpu, auto
    TTS_USE_FP16: bool = True

    # Provider-specific settings
    KOKORO_MODEL: str = "hexgrad/Kokoro-82M"
    CHATTERBOX_MODEL: str = "resemble-ai/chatterbox"
    F5_TTS_MODEL: str = "f5-tts"
    QWEN3_TTS_MODEL_VARIANT: str = "1.7B"  # "0.6B" or "1.7B"
    ORPHEUS_MODEL: str = "canopylabs/orpheus-3b-0.1-ft"
    DIA_MODEL: str = "nari-labs/Dia-1.6B"

    # ===========================
    # STT Configuration
    # ===========================
    STT_MODEL: str = "openai/whisper-small"      # Legacy transformers model
    STT_BACKEND: str = "faster-whisper"           # "faster-whisper" or "transformers"
    STT_PROFILE: str = "default"                  # "fast" | "default" | "accurate"
    STT_DEVICE: str = "cuda"
    STT_USE_FP16: bool = True
    STT_USE_QUANTIZATION: bool = True
    STT_LANGUAGE: Optional[str] = None           # None for auto-detect
    STT_WORD_TIMESTAMPS: bool = False             # Enable word-level timestamps by default
    STT_VAD_FILTER: bool = True                   # Voice Activity Detection filtering

    # ===========================
    # Language Detection
    # ===========================
    LANGUAGE_DETECTOR: str = "lingua"             # "lingua" or "langdetect"
    LANGUAGE_CONFIDENCE_THRESHOLD: float = 0.7

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
    # Security & API
    # ===========================
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:3001"  # Comma-separated
    MAX_UPLOAD_SIZE_MB: int = 25           # Max audio file upload size
    RATE_LIMIT_REQUESTS: int = 60         # Max requests per minute per IP
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    REQUEST_TIMEOUT_SECONDS: int = 120    # Max request processing time

    # ===========================
    # Performance
    # ===========================
    MAX_WORKERS: int = 4

    @field_validator("STT_PROFILE")
    @classmethod
    def validate_stt_profile(cls, v):
        allowed = ["fast", "default", "accurate"]
        if v not in allowed:
            raise ValueError(f"STT_PROFILE must be one of: {allowed}")
        return v

    @field_validator("TTS_DEFAULT_PROVIDER")
    @classmethod
    def validate_tts_provider(cls, v):
        allowed = ["kokoro", "chatterbox", "f5-tts", "qwen3-tts", "orpheus", "dia"]
        if v not in allowed:
            raise ValueError(f"TTS_DEFAULT_PROVIDER must be one of: {allowed}")
        return v

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
