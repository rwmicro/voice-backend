"""
Kokoro TTS Provider
Using hexgrad/Kokoro-82M model with kokoro package
"""

import torch
import numpy as np
from typing import Generator, List, Optional, Dict
from .base import TTSProvider, TTSConfig, Voice


# Voice mappings for Kokoro
KOKORO_VOICES = {
    'af_bella': Voice(
        id='af_bella',
        name='Bella (American)',
        language='en',
        gender='female',
        accent='american',
        quality_grade='A-',
        description='American English female voice'
    ),
    'bf_emma': Voice(
        id='bf_emma',
        name='Emma (British)',
        language='en',
        gender='female',
        accent='british',
        quality_grade='B-',
        description='British English female voice'
    ),
    'jf_alpha': Voice(
        id='jf_alpha',
        name='Alpha (Japanese)',
        language='ja',
        gender='female',
        quality_grade='C+',
        description='Japanese female voice'
    ),
    'zf_xiaobei': Voice(
        id='zf_xiaobei',
        name='Xiaobei (Chinese)',
        language='zh',
        gender='female',
        quality_grade='D',
        description='Chinese female voice'
    ),
    'ef_dora': Voice(
        id='ef_dora',
        name='Dora (Spanish)',
        language='es',
        gender='female',
        quality_grade='C',
        description='Spanish female voice'
    ),
    'ff_siwis': Voice(
        id='ff_siwis',
        name='Siwis (French)',
        language='fr',
        gender='female',
        quality_grade='B-',
        description='French female voice'
    ),
    'hf_alpha': Voice(
        id='hf_alpha',
        name='Alpha (Hindi)',
        language='hi',
        gender='female',
        quality_grade='C',
        description='Hindi female voice'
    ),
    'if_sara': Voice(
        id='if_sara',
        name='Sara (Italian)',
        language='it',
        gender='female',
        quality_grade='C',
        description='Italian female voice'
    ),
    'pf_dora': Voice(
        id='pf_dora',
        name='Dora (Portuguese)',
        language='pt',
        gender='female',
        quality_grade='C',
        description='Brazilian Portuguese female voice'
    ),
}

# Language to default voice mapping
LANG_TO_VOICE = {
    'en': 'bf_emma',
    'ja': 'jf_alpha',
    'zh': 'zf_xiaobei',
    'zh-cn': 'zf_xiaobei',
    'zh-tw': 'zf_xiaobei',
    'es': 'ef_dora',
    'fr': 'ff_siwis',
    'hi': 'hf_alpha',
    'it': 'if_sara',
    'pt': 'pf_dora',
    'pt-br': 'pf_dora',
}


class KokoroProvider(TTSProvider):
    """Kokoro TTS Provider"""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.pipelines: Dict[str, any] = {}  # Language-specific pipelines
        self.model_id = 'hexgrad/Kokoro-82M'

    async def initialize(self):
        """Initialize Kokoro model"""
        if self.is_initialized:
            return

        print(f"[Kokoro] Initializing Kokoro TTS...")
        print(f"[Kokoro] Device: {self.config.device}")

        try:
            # Import Kokoro's KPipeline
            from kokoro import KPipeline

            # Initialize default pipeline (British English)
            self.pipelines['b'] = KPipeline(lang_code='b', device=self.config.device)

            self.is_initialized = True
            print(f"[Kokoro] ✓ Kokoro TTS loaded successfully")
            print(f"[Kokoro] Pipelines will be loaded on-demand for other languages")

        except ImportError as e:
            print(f"[Kokoro] ✗ Error: kokoro package not found")
            print(f"[Kokoro] Install with: pip install kokoro-onnx")
            print(f"[Kokoro] Or from source: https://huggingface.co/hexgrad/Kokoro-82M")
            raise ImportError("kokoro package required. Install: pip install kokoro-onnx")
        except Exception as e:
            print(f"[Kokoro] ✗ Error initializing: {e}")
            raise

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str = "en",
        **kwargs
    ) -> Generator[np.ndarray, None, None]:
        """Synthesize speech using Kokoro"""
        if not self.is_initialized:
            await self.initialize()

        # Get language code
        lang_code = self._get_lang_code(language)

        # Get or create pipeline for this language
        pipeline = self._get_pipeline(lang_code)

        if pipeline is None:
            print(f"[Kokoro] No pipeline available for language: {language}")
            return []

        # Split text into sentences for better quality
        sentences = self._split_sentences(text)

        audio_chunks = []
        speed = kwargs.get('speed', 1.2)

        for sentence in sentences:
            if not sentence.strip():
                continue

            try:
                # Generate with Kokoro pipeline
                # pipeline() returns generator of (phonemes, tokens, audio)
                for _, (_, _, audio) in enumerate(pipeline(
                    sentence,
                    voice=voice_id,
                    speed=speed
                )):
                    # Convert to numpy if needed
                    if isinstance(audio, torch.Tensor):
                        audio_np = audio.cpu().numpy()
                    else:
                        audio_np = audio

                    # Normalize to float32 [-1, 1]
                    if audio_np.dtype != np.float32:
                        audio_np = audio_np.astype(np.float32)

                    if np.abs(audio_np).max() > 1.0:
                        audio_np = audio_np / np.abs(audio_np).max()

                    audio_chunks.append(audio_np)

            except Exception as e:
                print(f"[Kokoro] Error synthesizing sentence: {e}")
                continue

        return audio_chunks

    def list_voices(self, language: Optional[str] = None) -> List[Voice]:
        """List available Kokoro voices"""
        voices = list(KOKORO_VOICES.values())
        if language:
            voices = [v for v in voices if v.language == language]
        return voices

    def get_default_voice(self, language: str) -> str:
        """Get default voice for language"""
        return LANG_TO_VOICE.get(language, 'bf_emma')

    async def cleanup(self):
        """Cleanup Kokoro resources"""
        for lang_code, pipeline in self.pipelines.items():
            if pipeline is not None:
                del pipeline

        self.pipelines.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_initialized = False

    @property
    def name(self) -> str:
        return "Kokoro"

    @property
    def description(self) -> str:
        return "Kokoro-82M: Fast, multilingual TTS model with multiple voices"

    @property
    def supported_languages(self) -> List[str]:
        return ['en', 'ja', 'zh', 'zh-cn', 'zh-tw', 'es', 'fr', 'hi', 'it', 'pt', 'pt-br']

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = []
        for delimiter in ['.', '!', '?']:
            text = text.replace(delimiter, f'{delimiter}|')

        for sentence in text.split('|'):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)

        return sentences if sentences else [text]

    def _get_pipeline(self, lang_code: str):
        """Get or create pipeline for language"""
        if lang_code not in self.pipelines:
            try:
                from kokoro import KPipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.pipelines[lang_code] = KPipeline(
                    lang_code=lang_code,
                    device=self.config.device
                )
                print(f"[Kokoro] ✓ Loaded pipeline for: {lang_code}")
            except Exception as e:
                print(f"[Kokoro] Failed to load pipeline for {lang_code}: {e}")
                # Fallback to British English
                return self.pipelines.get('b')

        return self.pipelines[lang_code]

    def _get_lang_code(self, language: str) -> str:
        """Map language to Kokoro lang code"""
        # Kokoro uses single letter codes
        lang_map = {
            'en': 'b',  # Default to British
            'ja': 'j',
            'zh': 'z',
            'zh-cn': 'z',
            'zh-tw': 'z',
            'es': 'e',
            'fr': 'f',
            'hi': 'h',
            'it': 'i',
            'pt': 'p',
            'pt-br': 'p',
        }
        return lang_map.get(language, 'b')
