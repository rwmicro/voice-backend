"""
TTS Provider System
Allows switching between different TTS engines
"""

from .base import TTSProvider, TTSConfig
from .kokoro_provider import KokoroProvider
from .chatterbox_provider import ChatterboxProvider
from .f5_tts_provider import F5TTSProvider

# Registry of available TTS providers
TTS_PROVIDERS = {
    'kokoro': KokoroProvider,
    'chatterbox': ChatterboxProvider,
    'f5-tts': F5TTSProvider,
}

def get_tts_provider(provider_name: str, config: TTSConfig, **kwargs) -> TTSProvider:
    """Factory function to get TTS provider by name

    Args:
        provider_name: Name of the TTS provider
        config: TTS configuration
        **kwargs: Additional provider-specific arguments
                  - model_variant: For F5-TTS, specify 'F5TTS_v1_Base' or 'F5TTS_v1_Large'
    """
    if provider_name not in TTS_PROVIDERS:
        raise ValueError(f"Unknown TTS provider: {provider_name}. Available: {list(TTS_PROVIDERS.keys())}")

    provider_class = TTS_PROVIDERS[provider_name]

    # F5-TTS supports model_variant parameter
    if provider_name == 'f5-tts' and 'model_variant' in kwargs:
        return provider_class(config, model_variant=kwargs['model_variant'])

    return provider_class(config)

def list_available_providers():
    """List all available TTS providers"""
    return list(TTS_PROVIDERS.keys())

__all__ = [
    'TTSProvider',
    'TTSConfig',
    'KokoroProvider',
    'ChatterboxProvider',
    'F5TTSProvider',
    'get_tts_provider',
    'list_available_providers',
    'TTS_PROVIDERS'
]
