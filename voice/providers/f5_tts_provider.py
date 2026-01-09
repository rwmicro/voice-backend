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

        print(f"[F5-TTS] Initializing model...")
        print(f"[F5-TTS] Device: {self.config.device}")

        try:
            # Try to import F5-TTS
            # Note: This requires the F5-TTS package to be installed
            # Installation: pip install f5-tts OR git clone + pip install -e .
            try:
                from f5_tts.api import F5TTS
                self.f5_tts_available = True
            except ImportError:
                print("[F5-TTS] F5-TTS package not found. Trying alternative import...")
                # Alternative: if installed from git
                try:
                    import sys
                    sys.path.append('./F5-TTS')  # Adjust path as needed
                    from f5_tts.api import F5TTS
                    self.f5_tts_available = True
                except ImportError:
                    print("[F5-TTS] ✗ F5-TTS not installed")
                    print("[F5-TTS] Install with: pip install f5-tts")
                    print("[F5-TTS] Or: git clone https://github.com/SWivid/F5-TTS && cd F5-TTS && pip install -e .")
                    raise ImportError("F5-TTS not installed")

            # Initialize F5-TTS model
            # model parameter options: 'F5TTS_v1_Base' (default) or 'F5TTS_v1_Large'
            print(f"[F5-TTS] Using model variant: {self.model_variant}")
            self.model = F5TTS(
                model=self.model_variant,
                device=self.config.device,
            )

            # Load model weights
            # Note: F5-TTS will download weights automatically on first run
            print("[F5-TTS] Loading model weights (this may take a while on first run)...")

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
                    print(f"[F5-TTS] ✓ Found English reference: {en_ref}")
                else:
                    print(f"[F5-TTS] ✗ English reference not found at: {en_ref}")

                if os.path.exists(zh_ref):
                    self.default_references['zh']['ref_file'] = zh_ref
                    print(f"[F5-TTS] ✓ Found Chinese reference: {zh_ref}")
                else:
                    print(f"[F5-TTS] ✗ Chinese reference not found at: {zh_ref}")

                # Verify at least one reference was found
                if not self.default_references['en']['ref_file'] and not self.default_references['zh']['ref_file']:
                    print(f"[F5-TTS] ⚠️  Warning: No default reference audio found")
                    print(f"[F5-TTS] Voice cloning will require custom reference audio")

            except Exception as ref_error:
                print(f"[F5-TTS] ⚠️  Warning: Could not locate reference audio files: {ref_error}")
                print(f"[F5-TTS] Voice cloning will require custom reference audio")

            self.is_initialized = True
            print(f"[F5-TTS] ✓ Model loaded successfully")

        except Exception as e:
            print(f"[F5-TTS] ✗ Error loading model: {e}")
            raise

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str = "en",
        **kwargs
    ) -> Generator[np.ndarray, None, None]:
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

        # Split into chunks for better streaming
        chunks = self._split_text_chunks(text, max_length=200)

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
                        print(f"[F5-TTS] Error: No reference audio available for {lang_key}")
                        print(f"[F5-TTS] Please provide ref_audio and ref_text parameters")
                        continue

                    print(f"[F5-TTS] Using default {lang_key} reference")

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

                # Normalize to float32 [-1, 1]
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                if np.abs(audio).max() > 1.0:
                    audio = audio / np.abs(audio).max()

                # Resample if necessary
                if sr != self.config.sample_rate:
                    audio = self._resample(audio, sr, self.config.sample_rate)

                audio_chunks.append(audio)

            except Exception as e:
                print(f"[F5-TTS] Error synthesizing chunk: {e}")
                continue

        return audio_chunks

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

    def _split_text_chunks(self, text: str, max_length: int = 200) -> List[str]:
        """
        Split text into chunks of reasonable length
        F5-TTS works best with chunks of 100-200 characters
        """
        # First split by sentences
        sentences = []
        for delimiter in ['. ', '! ', '? ', '\n']:
            text = text.replace(delimiter, f'{delimiter}|')

        sentence_list = [s.strip() for s in text.split('|') if s.strip()]

        # Then combine sentences into chunks
        chunks = []
        current_chunk = ""

        for sentence in sentence_list:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio

        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            print("[F5-TTS] Warning: librosa not installed, skipping resampling")
            return audio

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
