"""
F5-TTS Provider
Using SWivid/F5-TTS from GitHub
https://github.com/SWivid/F5-TTS

F5-TTS is a non-autoregressive, zero-shot TTS system using flow matching
"""

import torch
import numpy as np
from typing import Generator, List, Optional
from .base import TTSProvider, TTSConfig, Voice
from voice.utils.audio import AudioProcessor
from voice.utils.text import TextProcessor

try:
    from voice.utils.logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


# F5-TTS voices (configurable through reference audio)
F5_TTS_VOICES = {
    'default_en': Voice(
        id='default_en',
        name='F5 English Default',
        language='en',
        gender='neutral',
        quality_grade='A+',
        description='High-quality zero-shot English TTS'
    ),
    'default_zh': Voice(
        id='default_zh',
        name='F5 Chinese Default',
        language='zh',
        gender='neutral',
        quality_grade='A+',
        description='High-quality zero-shot Chinese TTS'
    ),
    'default_ja': Voice(
        id='default_ja',
        name='F5 Japanese Default',
        language='ja',
        gender='neutral',
        quality_grade='A',
        description='High-quality zero-shot Japanese TTS'
    ),
}


class F5TTSProvider(TTSProvider):
    """F5-TTS Provider using flow matching for zero-shot TTS"""

    def __init__(self, config: TTSConfig, model_variant: str = "F5TTS_v1_Base"):
        super().__init__(config)
        self.model = None
        self.vocoder = None
        self.model_name = 'F5-TTS'
        self.model_variant = model_variant  # 'F5TTS_v1_Base' or 'F5TTS_v1_Large'
        self.f5_tts_available = False

        # Default reference audio (built-in samples from F5-TTS)
        # These are automatically available when F5-TTS is installed
        self.default_references = {
            'en': {
                'ref_text': 'Some call me nature, others call me mother nature.',
                'ref_file': None  # Will use F5-TTS built-in sample
            },
            'zh': {
                'ref_text': '对，这就是我，万人敬仰的太乙真人。',
                'ref_file': None  # Will use F5-TTS built-in sample
            }
        }

    async def initialize(self):
        """Initialize F5-TTS model"""
        if self.is_initialized:
            return

        logger.info("Initializing F5-TTS model...")
        logger.info(f"Device: {self.config.device}")

        try:
            # Try to import F5-TTS
            # Note: This requires the F5-TTS package to be installed
            # Installation: pip install f5-tts OR git clone + pip install -e .
            try:
                from f5_tts.api import F5TTS
                self.f5_tts_available = True
            except ImportError:
                logger.error("F5-TTS not installed")
                logger.error("Install with: pip install f5-tts")
                logger.error("Or: git clone https://github.com/SWivid/F5-TTS && cd F5-TTS && pip install -e .")
                raise ImportError("F5-TTS not installed")

            # Initialize F5-TTS model
            # model parameter options: 'F5TTS_v1_Base' (default) or 'F5TTS_v1_Large'
            logger.info(f"Using model variant: {self.model_variant}")
            self.model = F5TTS(
                model=self.model_variant,
                device=self.config.device,
            )

            # Load model weights
            # Note: F5-TTS will download weights automatically on first run
            logger.info("Loading model weights (this may take a while on first run)...")

            # Set up default reference audio paths
            import os
            import sys

            # Try multiple methods to find F5-TTS installation directory
            try:
                # Method 1: Use importlib to find the package location
                import importlib.util
                spec = importlib.util.find_spec('f5_tts')
                if spec and spec.submodule_search_locations:
                    f5_tts_dir = spec.submodule_search_locations[0]
                else:
                    # Method 2: Search in site-packages
                    for path in sys.path:
                        candidate = os.path.join(path, 'f5_tts')
                        if os.path.isdir(candidate):
                            f5_tts_dir = candidate
                            break
                    else:
                        raise ImportError("Could not locate f5_tts directory")

                en_ref = os.path.join(f5_tts_dir, 'infer', 'examples', 'basic', 'basic_ref_en.wav')
                zh_ref = os.path.join(f5_tts_dir, 'infer', 'examples', 'basic', 'basic_ref_zh.wav')

                if os.path.exists(en_ref):
                    self.default_references['en']['ref_file'] = en_ref
                    logger.info(f"Found English reference: {en_ref}")
                else:
                    logger.warning(f"English reference not found at: {en_ref}")

                if os.path.exists(zh_ref):
                    self.default_references['zh']['ref_file'] = zh_ref
                    logger.info(f"Found Chinese reference: {zh_ref}")
                else:
                    logger.warning(f"Chinese reference not found at: {zh_ref}")

                # Verify at least one reference was found
                if not self.default_references['en']['ref_file'] and not self.default_references['zh']['ref_file']:
                    logger.warning("No default reference audio found")
                    logger.warning("Voice cloning will require custom reference audio")

            except Exception as ref_error:
                logger.warning(f"Could not locate reference audio files: {ref_error}")
                logger.warning("Voice cloning will require custom reference audio")

            self.is_initialized = True
            logger.info("F5-TTS model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str = "en",
        **kwargs
    ) -> List[np.ndarray]:
        """
        Synthesize speech using F5-TTS

        F5-TTS supports zero-shot voice cloning, so you can provide:
        - ref_audio: path to reference audio file
        - ref_text: transcript of reference audio
        """
        if not self.is_initialized:
            await self.initialize()

        # Extract optional parameters
        ref_audio = kwargs.get('ref_audio', None)
        ref_text = kwargs.get('ref_text', None)
        speed = kwargs.get('speed', 1.0)

        # Split into chunks for better streaming using shared utility
        chunks = TextProcessor.split_into_chunks(text, max_length=200)

        audio_chunks = []
        for chunk in chunks:
            if not chunk.strip():
                continue

            try:
                # Generate audio
                # F5-TTS ALWAYS requires reference audio and text
                # If not provided, use default references
                if not ref_audio or not ref_text:
                    # Determine language for default reference
                    lang_key = 'en'  # Default to English
                    if language in ['zh', 'zh-cn', 'zh-tw']:
                        lang_key = 'zh'

                    # Get default reference for this language
                    default_ref = self.default_references.get(lang_key, self.default_references['en'])
                    ref_audio = default_ref['ref_file']
                    ref_text = default_ref['ref_text']

                    if not ref_audio:
                        logger.error(f"No reference audio available for {lang_key}")
                        logger.error("Please provide ref_audio and ref_text parameters")
                        continue

                    logger.debug(f"Using default {lang_key} reference")

                # Zero-shot voice cloning with correct parameter names
                audio, sr = self.model.infer(
                    ref_file=ref_audio,      # Correct: ref_file not ref_audio
                    ref_text=ref_text,       # Correct: ref_text
                    gen_text=chunk,          # Correct: gen_text not text
                    speed=speed,
                )

                # Convert to numpy if needed
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()

                # Normalize to float32 [-1, 1] using shared utility
                audio = AudioProcessor.normalize_to_float32(audio)

                # Resample if necessary using shared utility
                if sr != self.config.sample_rate:
                    audio = AudioProcessor.resample_audio(audio, sr, self.config.sample_rate)

                audio_chunks.append(audio)

            except Exception as e:
                logger.error(f"Error synthesizing chunk: {e}", exc_info=True)
                continue

        return audio_chunks

    async def synthesize_streaming(
        self,
        text: str,
        voice_id: str,
        language: str = "en",
        **kwargs
    ):
        """Synthesize speech using F5-TTS with streaming (yields chunks as generated)"""
        audio_chunks = await self.synthesize(text, voice_id, language, **kwargs)
        for chunk in audio_chunks:
            yield chunk

    def list_voices(self, language: Optional[str] = None) -> List[Voice]:
        """List available F5-TTS voices"""
        voices = list(F5_TTS_VOICES.values())
        if language:
            voices = [v for v in voices if v.language == language]
        return voices

    def get_default_voice(self, language: str) -> str:
        """Get default voice for language"""
        voice_map = {
            'en': 'default_en',
            'zh': 'default_zh',
            'zh-cn': 'default_zh',
            'zh-tw': 'default_zh',
            'ja': 'default_ja',
        }
        return voice_map.get(language, 'default_en')

    async def cleanup(self):
        """Cleanup F5-TTS resources"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.vocoder is not None:
            del self.vocoder
            self.vocoder = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_initialized = False

    @property
    def name(self) -> str:
        return "F5-TTS"

    @property
    def description(self) -> str:
        return "F5-TTS: Zero-shot voice cloning with flow matching (multilingual, high quality)"

    @property
    def supported_languages(self) -> List[str]:
        return ['en', 'zh', 'zh-cn', 'zh-tw', 'ja', 'es', 'fr', 'de', 'ko']

    def supports_voice_cloning(self) -> bool:
        """F5-TTS supports zero-shot voice cloning"""
        return True

    def clone_voice(self, ref_audio_path: str, ref_text: str):
        """
        Set reference audio for voice cloning

        Args:
            ref_audio_path: Path to reference audio file (3-10 seconds recommended)
            ref_text: Transcript of the reference audio

        Returns:
            Dict with ref_audio and ref_text to pass to synthesize()
        """
        return {
            'ref_audio': ref_audio_path,
            'ref_text': ref_text,
        }
