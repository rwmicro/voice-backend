"""
Orpheus TTS Provider
Canopy AI Orpheus-3B: Highly expressive English TTS with emotion tags and voice cloning
https://huggingface.co/canopylabs/orpheus-3b-0.1-ft

Features:
- 3.78B params, Llama-3B backbone
- Emotion tags: <happy>, <sad>, <angry>, <whispering>, <laughing>, <sighing>,
  <crying>, <surprised>
- Zero-shot voice cloning
- Streaming ~100ms latency
- English only
- Apache 2.0 license
"""

import torch
import numpy as np
from typing import AsyncGenerator, Dict, List, Optional
from .base import TTSProvider, TTSConfig, Voice

from voice.utils.audio import AudioProcessor
from voice.utils.text import TextProcessor

try:
    from voice.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------

try:
    from orpheus_tts import OrpheusModel
    ORPHEUS_PACKAGE = True
    logger.info("orpheus_tts package found — using official OrpheusModel")
except ImportError:
    ORPHEUS_PACKAGE = False
    logger.debug(
        "orpheus_tts package not found — will use transformers directly. "
        "Install with: pip install orpheus-tts"
    )

try:
    from snac import SNAC as SNACDecoder
    SNAC_AVAILABLE = True
except ImportError:
    SNAC_AVAILABLE = False
    logger.warning(
        "SNAC codec not found — Orpheus audio decoding will be unavailable. "
        "Install with: pip install snac"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "canopylabs/orpheus-3b-0.1-ft"

NATIVE_SAMPLE_RATE = 24000

# Range of special audio codec token IDs emitted by the Orpheus model
AUDIO_TOKEN_START = 151672
AUDIO_TOKEN_END = 160000

# Supported emotion tags (can be embedded directly in text)
VALID_EMOTIONS = {
    "happy", "sad", "angry", "whispering", "laughing",
    "sighing", "crying", "surprised",
}

# ---------------------------------------------------------------------------
# Voice catalogue
# ---------------------------------------------------------------------------

ORPHEUS_VOICES: Dict[str, Voice] = {
    "tara": Voice(
        id="tara",
        name="Tara",
        language="en",
        gender="female",
        quality_grade="A",
        description="Warm and expressive female voice (default)",
    ),
    "leo": Voice(
        id="leo",
        name="Leo",
        language="en",
        gender="male",
        quality_grade="A",
        description="Clear and confident male voice",
    ),
    "leah": Voice(
        id="leah",
        name="Leah",
        language="en",
        gender="female",
        quality_grade="A",
        description="Soft and articulate female voice",
    ),
    "dan": Voice(
        id="dan",
        name="Dan",
        language="en",
        gender="male",
        quality_grade="A",
        description="Deep and authoritative male voice",
    ),
    "mia": Voice(
        id="mia",
        name="Mia",
        language="en",
        gender="female",
        quality_grade="A",
        description="Bright and energetic female voice",
    ),
    "zac": Voice(
        id="zac",
        name="Zac",
        language="en",
        gender="male",
        quality_grade="A",
        description="Casual and friendly male voice",
    ),
    "julia": Voice(
        id="julia",
        name="Julia",
        language="en",
        gender="female",
        quality_grade="A",
        description="Elegant and polished female voice",
    ),
    "sky": Voice(
        id="sky",
        name="Sky",
        language="en",
        gender="male",
        quality_grade="A",
        description="Light and versatile male voice",
    ),
}

DEFAULT_VOICE_ID = "tara"


class OrpheusProvider(TTSProvider):
    """
    Orpheus TTS provider using either the official orpheus_tts package or
    transformers + SNAC as a fallback.
    """

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.sample_rate_output = NATIVE_SAMPLE_RATE

        # Populated during initialize()
        self._orpheus_model = None   # official package model
        self._model = None           # transformers LM
        self._tokenizer = None       # transformers tokenizer
        self._snac = None            # SNAC vocoder

    # ------------------------------------------------------------------
    # TTSProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "orpheus"

    @property
    def description(self) -> str:
        return (
            "Canopy AI Orpheus-3B: Highly expressive English TTS with emotion tags, "
            "zero-shot voice cloning, and ~100ms streaming latency"
        )

    @property
    def supported_languages(self) -> List[str]:
        return ["en"]

    async def initialize(self):
        """Load the Orpheus model and SNAC vocoder."""
        if self.is_initialized:
            return

        logger.info(f"Initialising Orpheus TTS on {self.config.device} ...")

        torch_dtype = (
            self.config.torch_dtype
            if self.config.torch_dtype is not None
            else (torch.float16 if self.config.device == "cuda" else torch.float32)
        )

        try:
            if ORPHEUS_PACKAGE:
                self._orpheus_model = OrpheusModel.from_pretrained(
                    MODEL_ID,
                    device=self.config.device,
                    torch_dtype=torch_dtype,
                )
                logger.info("Orpheus model loaded via official orpheus_tts package")
            else:
                from transformers import AutoTokenizer, AutoModelForCausalLM

                logger.info(f"Loading tokenizer from {MODEL_ID} ...")
                self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

                logger.info(f"Loading Llama-based LM from {MODEL_ID} ...")
                self._model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch_dtype,
                    device_map=self.config.device if self.config.device != "cpu" else None,
                )
                if self.config.device == "cpu":
                    self._model = self._model.to("cpu")
                self._model.eval()
                logger.info("Orpheus LM loaded via transformers")

            # Load SNAC vocoder for audio token decoding
            if SNAC_AVAILABLE:
                logger.info("Loading SNAC audio codec ...")
                self._snac = SNACDecoder.from_pretrained("hubertsiuzdak/snac_24khz")
                self._snac = self._snac.to(self.config.device)
                self._snac.eval()
                logger.info("SNAC vocoder ready")
            else:
                logger.warning(
                    "SNAC not available — audio decoding will be skipped. "
                    "Install with: pip install snac"
                )

            self.is_initialized = True
            logger.info(
                f"Orpheus TTS initialised — "
                f"native sample rate: {self.sample_rate_output} Hz"
            )

        except OSError as exc:
            raise ImportError(
                f"Could not load Orpheus model '{MODEL_ID}'. "
                "The model weights may not be downloaded yet. "
                f"Run: huggingface-cli download {MODEL_ID}\n"
                f"Original error: {exc}"
            ) from exc
        except ImportError:
            raise
        except Exception as exc:
            logger.error(f"Error initialising Orpheus TTS: {exc}", exc_info=True)
            raise

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str = "en",
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Synthesize speech and return all chunks as a list.

        kwargs:
            emotion (str): Emotion to prepend as a tag, e.g. "happy".
                Alternatively embed tags directly in text: "Hello! <happy>"
            ref_audio_path (str): Path to reference WAV for zero-shot voice cloning.
        """
        chunks: List[np.ndarray] = []
        async for chunk in self.synthesize_streaming(text, voice_id, language, **kwargs):
            chunks.append(chunk)

        if not chunks:
            logger.warning(f"No audio generated for text: '{text[:80]}...'")
            chunks.append(AudioProcessor.generate_silence(0.5, self.config.sample_rate))

        return chunks

    async def synthesize_streaming(
        self,
        text: str,
        voice_id: str,
        language: str = "en",
        **kwargs,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator that yields audio chunks sentence-by-sentence.
        """
        if not self.is_initialized:
            await self.initialize()

        if language != "en":
            logger.warning(
                f"Orpheus supports English only — ignoring requested language '{language}'"
            )

        emotion: Optional[str] = kwargs.get("emotion")
        ref_audio_path: Optional[str] = kwargs.get("ref_audio_path")

        voice_id = voice_id if voice_id in ORPHEUS_VOICES else DEFAULT_VOICE_ID

        sentences = TextProcessor.split_sentences(text, max_length=200)
        logger.info(
            f"Orpheus streaming {len(sentences)} sentence(s), voice='{voice_id}'"
        )

        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # Prepend emotion tag if requested via kwarg and not already in text
            prepared = self._prepare_text(sentence, emotion)

            logger.debug(
                f"[{idx + 1}/{len(sentences)}] voice={voice_id} "
                f"text={prepared[:60]}{'...' if len(prepared) > 60 else ''}"
            )

            try:
                audio_np = await self._generate_sentence(
                    prepared, voice_id, ref_audio_path
                )

                if audio_np is not None and AudioProcessor.validate_audio(audio_np):
                    if self.sample_rate_output != self.config.sample_rate:
                        audio_np = AudioProcessor.resample_audio(
                            audio_np,
                            self.sample_rate_output,
                            self.config.sample_rate,
                        )
                    yield audio_np
                else:
                    logger.warning(f"Empty audio for sentence: '{sentence[:50]}'")

            except Exception as exc:
                logger.error(
                    f"Error synthesising sentence '{sentence[:50]}': {exc}",
                    exc_info=True,
                )
                continue

    def _prepare_text(self, text: str, emotion: Optional[str]) -> str:
        """
        Optionally prefix the text with an emotion tag.

        Orpheus handles emotion tags embedded directly in the text string,
        e.g. "That's great news! <happy>". If the caller passes an emotion via
        kwarg we append it at the start so the model can condition on it
        throughout the utterance.
        """
        if emotion and emotion.lower() in VALID_EMOTIONS:
            tag = f"<{emotion.lower()}>"
            if tag not in text:
                text = f"{tag} {text}"
        return text

    async def _generate_sentence(
        self,
        text: str,
        voice_id: str,
        ref_audio_path: Optional[str],
    ) -> Optional[np.ndarray]:
        """Run inference for a single sentence."""
        if ORPHEUS_PACKAGE and self._orpheus_model is not None:
            return await self._generate_via_package(text, voice_id, ref_audio_path)
        return await self._generate_via_transformers(text, voice_id, ref_audio_path)

    async def _generate_via_package(
        self,
        text: str,
        voice_id: str,
        ref_audio_path: Optional[str],
    ) -> Optional[np.ndarray]:
        """Use the official orpheus_tts package."""
        call_kwargs: dict = {"voice": voice_id}
        if ref_audio_path:
            call_kwargs["ref_audio_path"] = ref_audio_path

        with torch.inference_mode():
            result = self._orpheus_model.generate(text, **call_kwargs)

        if isinstance(result, dict):
            audio = result.get("audio") or result.get("waveform") or next(iter(result.values()))
        else:
            audio = result

        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().cpu().numpy()

        return AudioProcessor.normalize_to_float32(np.asarray(audio, dtype=np.float32))

    async def _generate_via_transformers(
        self,
        text: str,
        voice_id: str,
        ref_audio_path: Optional[str],
    ) -> Optional[np.ndarray]:
        """
        Fallback: tokenise with the Orpheus prompt template, run the LM, then
        decode audio tokens via SNAC.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Transformers model not loaded — call initialize() first")

        if not SNAC_AVAILABLE or self._snac is None:
            logger.error(
                "SNAC is required to decode Orpheus audio tokens. "
                "Install with: pip install snac"
            )
            return None

        # Build the Orpheus prompt
        prompt = self._build_prompt(text, voice_id)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Extract only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

        # Filter audio codec tokens
        audio_token_ids = [
            t.item() for t in new_ids
            if AUDIO_TOKEN_START <= t.item() < AUDIO_TOKEN_END
        ]

        if not audio_token_ids:
            logger.warning("No audio tokens in Orpheus output")
            return None

        audio_np = self._decode_snac_tokens(audio_token_ids)
        if audio_np is None:
            return None

        return AudioProcessor.normalize_to_float32(audio_np)

    def _build_prompt(self, text: str, voice_id: str) -> str:
        """
        Build the Orpheus inference prompt.

        Format (based on the official Orpheus inference spec):
            <|audio|><|start_header_id|>user<|end_header_id|>
            Convert text to speech: {text}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            <custom_token_3><|{voice_id}|><custom_token_4>
        """
        return (
            "<|audio|><|start_header_id|>user<|end_header_id|>\n"
            f"Convert text to speech: {text}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            f"<custom_token_3><|{voice_id}|><custom_token_4>"
        )

    def _decode_snac_tokens(
        self, audio_token_ids: List[int]
    ) -> Optional[np.ndarray]:
        """
        Decode a flat list of SNAC codec token IDs into a waveform.

        SNAC uses a multi-level residual vector quantiser; the codec tokens
        produced by Orpheus are interleaved across 3 codebook levels.  We
        de-interleave them and call snac.decode().
        """
        try:
            # Remap token IDs to codec indices (subtract base offset)
            codes = [t - AUDIO_TOKEN_START for t in audio_token_ids]

            # SNAC for 24kHz uses 3 codebook levels with a 7:4:1 interleave
            # pattern.  Collect them level by level.
            c0, c1, c2 = [], [], []
            i = 0
            while i + 6 < len(codes):
                c0.append(codes[i])
                c1.append(codes[i + 1])
                c1.append(codes[i + 2])
                c2.append(codes[i + 3])
                c2.append(codes[i + 4])
                c2.append(codes[i + 5])
                c2.append(codes[i + 6])
                i += 7

            if not c0:
                return None

            device = next(self._snac.parameters()).device
            codes_tensor = [
                torch.tensor(c0, dtype=torch.long, device=device).unsqueeze(0),
                torch.tensor(c1, dtype=torch.long, device=device).unsqueeze(0),
                torch.tensor(c2, dtype=torch.long, device=device).unsqueeze(0),
            ]

            with torch.inference_mode():
                audio_tensor = self._snac.decode(codes_tensor)

            audio_np = audio_tensor.squeeze().cpu().numpy().astype(np.float32)
            return audio_np

        except Exception as exc:
            logger.error(f"SNAC decode error: {exc}", exc_info=True)
            return None

    def list_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Return all 8 preset Orpheus voices (all English)."""
        voices = list(ORPHEUS_VOICES.values())
        if language is not None and language != "en":
            return []
        return voices

    def get_default_voice(self, language: str) -> str:
        """Return default voice ID (English only)."""
        if language != "en":
            logger.warning(
                f"Orpheus only supports English — ignoring language '{language}'"
            )
        return DEFAULT_VOICE_ID

    async def cleanup(self):
        """Release model resources and free GPU memory."""
        self._orpheus_model = None
        self._model = None
        self._tokenizer = None
        self._snac = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_initialized = False
        logger.info("Orpheus TTS resources released")
