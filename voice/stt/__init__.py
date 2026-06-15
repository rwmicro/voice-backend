"""
STT (Speech-to-Text) backends for rag-voice-backend
"""

from .faster_whisper_stt import FasterWhisperSTT

__all__ = ["FasterWhisperSTT"]
