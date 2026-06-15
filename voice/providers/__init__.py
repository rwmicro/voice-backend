"""
TTS Provider System
Allows switching between different TTS engines
"""

from .base import TTSProvider, TTSConfig
from .kokoro_provider import KokoroProvider
from .chatterbox_provider import ChatterboxProvider
from .f5_tts_provider import F5TTSProvider
from .qwen3_tts_provider import Qwen3TTSProvider
from .orpheus_provider import OrpheusProvider
from .dia_provider import DiaProvider

# Registry of available TTS providers
TTS_PROVIDERS = {
    'kokoro': KokoroProvider,
    'chatterbox': ChatterboxProvider,
    'f5-tts': F5TTSProvider,
    'qwen3-tts': Qwen3TTSProvider,
    'orpheus': OrpheusProvider,
    'dia': DiaProvider,
}

# Provider metadata for documentation
TTS_PROVIDER_INFO = {
    'kokoro': {
        'description': 'Kokoro 82M: Ultra-fast, CPU-capable, 8 languages, 48 voices',
        'features': ['fast', 'multilingual', 'cpu-friendly'],
        'license': 'Apache 2.0',
    },
    'chatterbox': {
        'description': 'ResembleAI Chatterbox: High-quality, 24 languages, voice cloning',
        'features': ['voice-cloning', 'emotion', 'multilingual'],
        'license': 'MIT',
    },
    'f5-tts': {
        'description': 'F5-TTS: Zero-shot voice cloning, flow-matching, high quality',
        'features': ['zero-shot-cloning', 'high-quality', 'multilingual'],
        'license': 'MIT',
    },
    'qwen3-tts': {
        'description': 'Qwen3-TTS: 97ms streaming, voice cloning in 3s, VoiceDesign, 10 languages',
        'features': ['streaming', 'voice-cloning', 'voice-design', 'multilingual'],
        'license': 'Apache 2.0',
    },
    'orpheus': {
        'description': 'Orpheus 3B: Emotion tags, zero-shot cloning, 100ms latency',
        'features': ['emotion-tags', 'zero-shot-cloning', 'streaming'],
        'license': 'Apache 2.0',
    },
    'dia': {
        'description': 'Dia 1.6B: Nonverbal sounds, multi-speaker dialogue, conversational',
        'features': ['nonverbal-sounds', 'multi-speaker', 'dialogue'],
        'license': 'Apache 2.0',
    },
}


def get_tts_provider(provider_name: str, config: TTSConfig, **kwargs) -> TTSProvider:
    """Factory function to get TTS provider by name

    Args:
        provider_name: Name of the TTS provider
        config: TTS configuration
        **kwargs: Additional provider-specific arguments
                  - model_variant: For F5-TTS ('F5TTS_v1_Base' or 'F5TTS_v1_Large')
                                   For Qwen3-TTS ('0.6B' or '1.7B')
    """
    if provider_name not in TTS_PROVIDERS:
        raise ValueError(f"Unknown TTS provider: {provider_name}. Available: {list(TTS_PROVIDERS.keys())}")

    provider_class = TTS_PROVIDERS[provider_name]

    # F5-TTS supports model_variant parameter
    if provider_name == 'f5-tts' and 'model_variant' in kwargs:
        return provider_class(config, model_variant=kwargs['model_variant'])

    # Qwen3-TTS supports model_variant parameter
    if provider_name == 'qwen3-tts' and 'model_variant' in kwargs:
        return provider_class(config, model_variant=kwargs['model_variant'])

    return provider_class(config)


def list_available_providers():
    """List all available TTS providers"""
    return list(TTS_PROVIDERS.keys())


def get_provider_info(provider_name: str = None) -> dict:
    """Get metadata about one or all providers"""
    if provider_name:
        return TTS_PROVIDER_INFO.get(provider_name, {})
    return TTS_PROVIDER_INFO


__all__ = [
    'TTSProvider',
    'TTSConfig',
    'KokoroProvider',
    'ChatterboxProvider',
    'F5TTSProvider',
    'Qwen3TTSProvider',
    'OrpheusProvider',
    'DiaProvider',
    'get_tts_provider',
    'list_available_providers',
    'get_provider_info',
    'TTS_PROVIDERS',
    'TTS_PROVIDER_INFO',
]
