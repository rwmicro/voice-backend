"""
Shared utilities for voice backend
"""

from .audio import AudioProcessor
from .text import TextProcessor
from .device import get_device_config

# Logger imports with fallback
try:
    from .logger import setup_logger, get_logger

    __all__ = [
        "AudioProcessor",
        "TextProcessor",
        "get_device_config",
        "setup_logger",
        "get_logger",
    ]
except ImportError:
    # Fallback if loguru not installed
    __all__ = [
        "AudioProcessor",
        "TextProcessor",
        "get_device_config",
    ]
