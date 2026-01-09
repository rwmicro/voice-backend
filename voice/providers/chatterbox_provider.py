"""
Chatterbox TTS Provider
Using chatterbox-tts package from ResembleAI
https://github.com/resemble-ai/chatterbox
"""

import torch
import numpy as np
import os
from pathlib import Path
from typing import Generator, List, Optional
from .base import TTSProvider, TTSConfig, Voice
from dotenv import load_dotenv

# Import shared utilities
from voice.utils.audio import AudioProcessor
from voice.utils.text import TextProcessor

try:
    from voice.utils.logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

load_dotenv()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


# Chatterbox voices (based on Resemble AI's model)
# Note: Chatterbox uses single default voice with optional voice cloning
CHATTERBOX_VOICES = {
    "default": Voice(
        id="default",
        name="Chatterbox Default",
        language="multi",
        gender="neutral",
        quality_grade="A",
        description="High-quality multilingual TTS voice (24 languages including Indonesian)",
    ),
}

# Language ID mapping for Chatterbox
LANGUAGE_IDS = {
    "ar": "ar",
    "da": "da",
    "de": "de",
    "el": "el",
    "en": "en",
    "es": "es",
    "fi": "fi",
    "fr": "fr",
    "he": "he",
    "hi": "hi",
    "id": "ms",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "ms": "ms",
    "nl": "nl",
    "no": "no",
    "pl": "pl",
    "pt": "pt",
    "ru": "ru",
    "sv": "sv",
    "sw": "sw",
    "tr": "tr",
    "zh": "zh",
}

# Audio prompts for each language (official Chatterbox samples)
# These provide native voice references for each language
AUDIO_PROMPTS_DIR = PROJECT_ROOT / "audio_prompts" / "multilingual"


def get_audio_prompt_path(language: str) -> Optional[str]:
    """Get the audio prompt file path for a given language"""
    # Indonesian uses Malay prompt
    lang_code = "ms" if language == "id" else language

    prompt_file = AUDIO_PROMPTS_DIR / f"{lang_code}_prompt.flac"

    if prompt_file.exists():
        return str(prompt_file)

    # Fallback to English if language prompt not found
    english_prompt = AUDIO_PROMPTS_DIR / "en_prompt.flac"
    if english_prompt.exists():
        return str(english_prompt)

    return None


class ChatterboxProvider(TTSProvider):
    """Chatterbox TTS Provider using chatterbox-tts package"""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.model = None
        self.multilingual_model = None
        self.sample_rate_output = 24000  # Chatterbox output sample rate

        # Performance optimization flags - respect user configuration
        self.use_cuda = config.device == "cuda"
        use_fp16_config = os.getenv("TTS_USE_FP16", "true").lower() == "true"
        self.use_half_precision = (
            self.use_cuda and use_fp16_config
        )  # Use FP16 only if GPU + enabled

    async def initialize(self):
        """Initialize Chatterbox model"""
        if self.is_initialized:
            return

        logger.info("Initializing Chatterbox TTS...")
        logger.info(f"Device: {self.config.device}")

        try:
            # Import Chatterbox TTS
            from chatterbox.tts import ChatterboxTTS
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            # Performance optimizations
            logger.info(f"Device: {self.config.device}")
            if self.use_cuda:
                precision = "FP16" if self.use_half_precision else "FP32"
                logger.info(f"🚀 GPU acceleration enabled ({precision})")
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
            else:
                logger.info("💻 CPU mode (slower)")

            # Load English model
            logger.info("Loading English model...")
            self.model = ChatterboxTTS.from_pretrained(device=self.config.device)

            # Note: Chatterbox models don't support .half() method
            # FP16 optimization is handled internally by the model
            if self.use_half_precision:
                logger.info(
                    "ℹ️ FP16 optimization requested (handled internally by Chatterbox)"
                )

            # Load multilingual model
            logger.info("Loading multilingual model...")
            self.multilingual_model = ChatterboxMultilingualTTS.from_pretrained(
                device=self.config.device
            )

            if self.use_half_precision:
                logger.info(
                    "ℹ️ FP16 optimization requested (handled internally by Chatterbox)"
                )

            # Get sample rate from model
            self.sample_rate_output = self.model.sr

            self.is_initialized = True
            logger.info("✓ Chatterbox TTS loaded successfully")
            logger.info(f"Sample rate: {self.sample_rate_output} Hz")
            if self.use_cuda:
                logger.info("⚡ Expected speedup: 2-3x faster than CPU")

        except ImportError as e:
            logger.error("✗ Error: chatterbox-tts package not found")
            logger.error("Install with: pip install chatterbox-tts")
            logger.error(
                "Or from source: git clone https://github.com/resemble-ai/chatterbox.git && cd chatterbox && pip install -e ."
            )
            raise ImportError(
                "chatterbox-tts package required. Install: pip install chatterbox-tts"
            )
        except Exception as e:
            logger.error(f"✗ Error initializing: {e}", exc_info=True)
            raise

    async def synthesize(
        self, text: str, voice_id: str, language: str = "en", **kwargs
    ) -> List[np.ndarray]:
        """Synthesize speech using Chatterbox - generates all audio chunks

        This method generates all sentences and returns them as a list.
        For true streaming, use synthesize_streaming() instead.
        """
        audio_chunks = []
        async for chunk in self.synthesize_streaming(
            text, voice_id, language, **kwargs
        ):
            audio_chunks.append(chunk)

        # Ensure we have at least some audio
        if not audio_chunks:
            logger.warning(f"⚠️ No audio chunks generated for text: '{text[:100]}...'")
            logger.warning("This may be due to token repetition or early EOS forcing")
            # Return silence as fallback to avoid errors using shared utility
            silence = AudioProcessor.generate_silence(0.5, self.config.sample_rate)
            audio_chunks.append(silence)

        return audio_chunks

    async def synthesize_streaming(
        self, text: str, voice_id: str, language: str = "en", **kwargs
    ):
        """Synthesize speech using Chatterbox with true streaming (yields chunks as generated)

        This is a generator that yields audio chunks as they are created, enabling
        real-time streaming with minimal latency.
        """
        if not self.is_initialized:
            await self.initialize()

        # Split into sentences for better quality using shared utility
        sentences = TextProcessor.split_sentences(text, max_length=150)

        logger.info(f"Streaming {len(sentences)} sentence(s) for language '{language}'")

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            logger.info(
                f"[Stream {i + 1}/{len(sentences)}] Generating: {sentence[:60]}{'...' if len(sentence) > 60 else ''}"
            )

            try:
                # Official Chatterbox parameters (matching official demo code)
                # Only using the 3 parameters recommended by ResembleAI
                generation_params = {
                    "exaggeration": kwargs.get(
                        "exaggeration", 0.5
                    ),  # Speech expressiveness (default: 0.5, range: 0.25-2.0)
                    "temperature": kwargs.get(
                        "temperature", 0.8
                    ),  # Generation randomness (default: 0.8, range: 0.05-5.0)
                    "cfg_weight": kwargs.get(
                        "cfg_weight", 0.5
                    ),  # Classifier-free guidance weight (default: 0.5, range: 0.2-1.0)
                }

                # Add audio prompt for voice cloning (official prompts or custom)
                audio_prompt = kwargs.get("audio_prompt_path")

                # If no custom prompt provided, use official language-specific prompt
                if not audio_prompt:
                    audio_prompt = get_audio_prompt_path(language)
                    if audio_prompt:
                        print(
                            f"[Chatterbox] Using official audio prompt for '{language}'"
                        )

                if audio_prompt:
                    generation_params["audio_prompt_path"] = audio_prompt
                else:
                    print(f"[Chatterbox] No audio prompt - using default voice")

                # Use inference mode for faster generation (disables gradient computation)
                with torch.inference_mode():
                    # Choose model based on language
                    if language == "en":
                        # Use English model with optimized parameters
                        wav = self.model.generate(sentence, **generation_params)
                    else:
                        # Use multilingual model with language ID and optimized parameters
                        lang_id = LANGUAGE_IDS.get(language, "en")

                        # Log if Indonesian is using Malay voice
                        if language == "id":
                            print(
                                f"[Chatterbox] Using Malay (ms) voice for Indonesian text"
                            )

                        wav = self.multilingual_model.generate(
                            sentence, language_id=lang_id, **generation_params
                        )

                # Convert tensor to numpy
                if isinstance(wav, torch.Tensor):
                    audio_np = wav.squeeze().cpu().numpy()
                else:
                    audio_np = wav

                # Normalize to float32 [-1, 1]
                if audio_np.dtype != np.float32:
                    audio_np = audio_np.astype(np.float32)

                if np.abs(audio_np).max() > 1.0:
                    audio_np = audio_np / np.abs(audio_np).max()

                # Resample if necessary
                if self.config.sample_rate != self.sample_rate_output:
                    audio_np = self._resample(
                        audio_np, self.sample_rate_output, self.config.sample_rate
                    )

                # Yield chunk immediately for streaming
                if len(audio_np) > 0:
                    yield audio_np
                else:
                    print(
                        f"[Chatterbox] Warning: Empty audio generated for sentence: {sentence[:50]}..."
                    )

            except Exception as e:
                print(
                    f"[Chatterbox] Error synthesizing sentence '{sentence[:50]}...': {e}"
                )
                import traceback

                traceback.print_exc()
                continue

    def list_voices(self, language: Optional[str] = None) -> List[Voice]:
        """List available Chatterbox voices"""
        voices = list(CHATTERBOX_VOICES.values())
        # Chatterbox voices work across all 23 supported languages
        return voices

    def get_default_voice(self, language: str) -> str:
        """Get default voice for language"""
        return "default"

    async def cleanup(self):
        """Cleanup Chatterbox resources"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.multilingual_model is not None:
            del self.multilingual_model
            self.multilingual_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_initialized = False

    @property
    def name(self) -> str:
        return "Chatterbox"

    @property
    def description(self) -> str:
        return "ResembleAI Chatterbox: High-quality neural TTS supporting 24 languages (includes Indonesian via Malay voice)"

    @property
    def supported_languages(self) -> List[str]:
        return [
            "ar",
            "da",
            "de",
            "el",
            "en",
            "es",
            "fi",
            "fr",
            "he",
            "hi",
            "id",
            "it",
            "ja",
            "ko",
            "ms",
            "nl",
            "no",
            "pl",
            "pt",
            "ru",
            "sv",
            "sw",
            "tr",
            "zh",
        ]

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for better TTS"""
        # Remove extra whitespace
        text = " ".join(text.split())

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(""", "'").replace(""", "'")

        # Remove markdown formatting
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)  # Italic
        text = re.sub(r"`(.+?)`", r"\1", text)  # Code
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # Links

        # Remove URLs
        text = re.sub(r"http[s]?://\S+", "", text)

        # Fix common abbreviations to prevent awkward splits
        text = text.replace("e.g.", "for example")
        text = text.replace("i.e.", "that is")
        text = text.replace("etc.", "etcetera")
        text = text.replace("Dr.", "Doctor")
        text = text.replace("Mr.", "Mister")
        text = text.replace("Mrs.", "Misses")
        text = text.replace("Ms.", "Miss")

        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling"""
        # Preprocess text first
        text = self._preprocess_text(text)

        # Split on sentence boundaries
        # Use regex to handle periods followed by space and capital letter
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        # Further split very long sentences (> 150 chars) at commas or semicolons
        final_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If sentence is too long, split at natural breaks
            if len(sentence) > 150:
                # Split at commas, semicolons, or conjunctions
                parts = re.split(r"(?<=,)\s+|(?<=;)\s+|\s+(?:and|but|or)\s+", sentence)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 10:  # Avoid tiny fragments
                        final_sentences.append(part)
            else:
                final_sentences.append(sentence)

        return final_sentences if final_sentences else [text]

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio

        try:
            import librosa

            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            print("[Chatterbox] Warning: librosa not installed, skipping resampling")
            return audio
