"""
Dia TTS Provider
Nari Labs Dia-1.6B: Codec language model with native nonverbal sounds and multi-speaker dialogue
https://huggingface.co/nari-labs/Dia-1.6B

Features:
- 1.6B params, codec language model
- Native nonverbal sounds: (laughs), (sighs), (breathes), (clears throat), (pauses)
- Multi-speaker dialogue with [S1] and [S2] speaker tags
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
# Optional dependency check
# ---------------------------------------------------------------------------

try:
    from dia.model import Dia as DiaModel
    DIA_PACKAGE = True
    logger.info("dia package found — using official Dia model")
except ImportError:
    DiaModel = None
    DIA_PACKAGE = False
    logger.debug(
        "dia package not found — will use transformers directly. "
        "Install with: pip install dia-tts"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "nari-labs/Dia-1.6B"

NATIVE_SAMPLE_RATE = 44100  # Dia outputs at 44.1 kHz
TARGET_SAMPLE_RATE = 24000  # We normalise down to 24 kHz to match config default

# Recognised nonverbal tokens (can be embedded in text by the caller)
NONVERBAL_TOKENS = {"(laughs)", "(sighs)", "(breathes)", "(clears throat)", "(pauses)"}

# Speaker tag format used by Dia
SPEAKER_TAG = {
    "S1": "[S1]",
    "S2": "[S2]",
}

# ---------------------------------------------------------------------------
# Voice catalogue
# ---------------------------------------------------------------------------

DIA_VOICES: Dict[str, Voice] = {
    "S1": Voice(
        id="S1",
        name="Speaker 1",
        language="en",
        gender="neutral",
        quality_grade="A",
        description="First speaker role in Dia dialogue (default)",
    ),
    "S2": Voice(
        id="S2",
        name="Speaker 2",
        language="en",
        gender="neutral",
        quality_grade="A",
        description="Second speaker role in Dia dialogue",
    ),
}

DEFAULT_VOICE_ID = "S1"


class DiaProvider(TTSProvider):
    """
    Nari Labs Dia-1.6B TTS provider.

    Supports single-speaker synthesis, multi-speaker dialogue via [S1]/[S2] tags,
    and native nonverbal sound tokens such as (laughs) and (sighs).
    """

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.sample_rate_output = NATIVE_SAMPLE_RATE

        # Populated during initialize()
        self._dia_model = None    # official dia package model instance
        self._model = None        # transformers fallback model
        self._processor = None    # transformers processor / tokenizer

    # ------------------------------------------------------------------
    # TTSProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "dia"

    @property
    def description(self) -> str:
        return (
            "Nari Labs Dia-1.6B: Codec language model with native nonverbal sounds "
            "and multi-speaker dialogue"
        )

    @property
    def supported_languages(self) -> List[str]:
        return ["en"]

    async def initialize(self):
        """Load the Dia model onto the configured device."""
        if self.is_initialized:
            return

        logger.info(f"Initialising Dia-1.6B on {self.config.device} ...")

        torch_dtype = (
            self.config.torch_dtype
            if self.config.torch_dtype is not None
            else (torch.float16 if self.config.device == "cuda" else torch.float32)
        )

        try:
            if DIA_PACKAGE:
                self._dia_model = DiaModel.from_pretrained(
                    MODEL_ID,
                    compute_dtype=torch_dtype,
                )
                self._dia_model = self._dia_model.to(self.config.device)
                self._dia_model.eval()
                logger.info("Dia model loaded via official dia-tts package")
            else:
                # Attempt transformers-based loading
                try:
                    from transformers import AutoProcessor, AutoModel
                    logger.info(f"Loading Dia processor from {MODEL_ID} ...")
                    self._processor = AutoProcessor.from_pretrained(MODEL_ID)
                    logger.info(f"Loading Dia model from {MODEL_ID} ...")
                    self._model = AutoModel.from_pretrained(
                        MODEL_ID,
                        torch_dtype=torch_dtype,
                        device_map=self.config.device if self.config.device != "cpu" else None,
                    )
                    if self.config.device == "cpu":
                        self._model = self._model.to("cpu")
                    self._model.eval()
                    logger.info("Dia model loaded via transformers (AutoModel)")
                except Exception as inner_exc:
                    raise ImportError(
                        f"Could not load Dia model '{MODEL_ID}' via transformers. "
                        "Install the official package for best results: pip install dia-tts\n"
                        f"Original error: {inner_exc}"
                    ) from inner_exc

            self.is_initialized = True
            logger.info(
                f"Dia-1.6B initialised — "
                f"native sample rate: {self.sample_rate_output} Hz "
                f"(will resample to {self.config.sample_rate} Hz)"
            )

        except OSError as exc:
            raise ImportError(
                f"Could not load Dia model '{MODEL_ID}'. "
                "The model weights may not be downloaded yet. "
                f"Run: huggingface-cli download {MODEL_ID}\n"
                f"Original error: {exc}"
            ) from exc
        except ImportError:
            raise
        except Exception as exc:
            logger.error(f"Error initialising Dia TTS: {exc}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        language: str = "en",
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Synthesize speech and return all audio chunks as a list.

        kwargs:
            speaker_audio_s1 (str): Path to reference audio for Speaker 1 voice cloning.
            speaker_audio_s2 (str): Path to reference audio for Speaker 2 voice cloning.
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
        Async generator that yields audio chunks for the given text.

        For multi-speaker text (containing [S1] / [S2] tags) the entire text is
        passed to Dia as-is.  For single-speaker text we wrap it with the
        appropriate speaker tag based on voice_id.
        """
        if not self.is_initialized:
            await self.initialize()

        if language != "en":
            logger.warning(
                f"Dia supports English only — ignoring requested language '{language}'"
            )

        speaker_audio_s1: Optional[str] = kwargs.get("speaker_audio_s1")
        speaker_audio_s2: Optional[str] = kwargs.get("speaker_audio_s2")

        # Determine whether the text already uses Dia's multi-speaker format
        has_speaker_tags = "[S1]" in text or "[S2]" in text
        if not has_speaker_tags:
            tag = SPEAKER_TAG.get(voice_id, SPEAKER_TAG["S1"])
            formatted_text = f"{tag} {text}"
        else:
            formatted_text = text

        logger.info(
            f"Dia synthesising, voice='{voice_id}', "
            f"multi-speaker={has_speaker_tags}, "
            f"text='{formatted_text[:80]}...'"
        )

        try:
            audio_np = await self._generate(
                formatted_text, speaker_audio_s1, speaker_audio_s2
            )
            if audio_np is not None and AudioProcessor.validate_audio(audio_np):
                # Resample from Dia's native 44.1 kHz to configured target
                if self.sample_rate_output != self.config.sample_rate:
                    audio_np = AudioProcessor.resample_audio(
                        audio_np,
                        self.sample_rate_output,
                        self.config.sample_rate,
                    )
                yield audio_np
            else:
                logger.warning("Dia produced empty audio")
        except Exception as exc:
            logger.error(f"Dia synthesis error: {exc}", exc_info=True)

    async def synthesize_dialogue(self, turns: List[dict], **kwargs) -> List[np.ndarray]:
        """
        Synthesize a multi-turn dialogue.

        Args:
            turns: List of dicts with keys "speaker" ("S1" or "S2") and "text".
                   Example:
                       [
                           {"speaker": "S1", "text": "Hello, how are you?"},
                           {"speaker": "S2", "text": "I'm doing great! (laughs) Thanks!"},
                       ]
            **kwargs: Forwarded to synthesize_streaming (e.g. speaker_audio_s1,
                      speaker_audio_s2 for voice cloning of each speaker).

        Returns:
            List of audio chunks (float32, sample_rate from config).
        """
        if not turns:
            return []

        # Format all turns into a single Dia-compatible dialogue string
        dialogue_text = self._format_dialogue(turns)
        logger.info(
            f"synthesize_dialogue: {len(turns)} turn(s) -> '{dialogue_text[:120]}...'"
        )

        chunks: List[np.ndarray] = []
        async for chunk in self.synthesize_streaming(
            dialogue_text,
            voice_id="S1",  # voice_id is ignored when [S1]/[S2] tags are present
            **kwargs,
        ):
            chunks.append(chunk)

        if not chunks:
            logger.warning("synthesize_dialogue produced no audio")
            chunks.append(AudioProcessor.generate_silence(0.5, self.config.sample_rate))

        return chunks

    def _format_dialogue(self, turns: List[dict]) -> str:
        """
        Convert a list of turn dicts into a [S1]/[S2]-tagged Dia dialogue string.

        Example output:
            "[S1] Hello, how are you? [S2] I'm doing great! (laughs)"
        """
        parts: List[str] = []
        for turn in turns:
            speaker = str(turn.get("speaker", "S1")).upper()
            text = str(turn.get("text", "")).strip()
            tag = SPEAKER_TAG.get(speaker, SPEAKER_TAG["S1"])
            if text:
                parts.append(f"{tag} {text}")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Internal generation helpers
    # ------------------------------------------------------------------

    async def _generate(
        self,
        text: str,
        speaker_audio_s1: Optional[str],
        speaker_audio_s2: Optional[str],
    ) -> Optional[np.ndarray]:
        """Dispatch to the appropriate backend."""
        if DIA_PACKAGE and self._dia_model is not None:
            return await self._generate_via_package(
                text, speaker_audio_s1, speaker_audio_s2
            )
        return await self._generate_via_transformers(text)

    async def _generate_via_package(
        self,
        text: str,
        speaker_audio_s1: Optional[str],
        speaker_audio_s2: Optional[str],
    ) -> Optional[np.ndarray]:
        """Use the official dia-tts package."""
        call_kwargs: dict = {}
        if speaker_audio_s1:
            call_kwargs["audio_prompt"] = speaker_audio_s1
        if speaker_audio_s2:
            # Dia uses a single audio_prompt_s2 key for S2 reference
            call_kwargs["audio_prompt_s2"] = speaker_audio_s2

        with torch.inference_mode():
            result = self._dia_model.generate(text, **call_kwargs)

        # Dia may return a numpy array or a torch tensor
        if isinstance(result, torch.Tensor):
            audio = result.squeeze().cpu().numpy()
        elif isinstance(result, np.ndarray):
            audio = result.squeeze()
        elif isinstance(result, dict):
            raw = result.get("audio") or result.get("waveform") or next(iter(result.values()))
            audio = (
                raw.squeeze().cpu().numpy()
                if isinstance(raw, torch.Tensor)
                else np.asarray(raw).squeeze()
            )
        else:
            audio = np.asarray(result).squeeze()

        return AudioProcessor.normalize_to_float32(audio.astype(np.float32))

    async def _generate_via_transformers(self, text: str) -> Optional[np.ndarray]:
        """
        Fallback transformers-based generation.

        Dia is a codec language model; without the official package the audio
        post-processing chain may be incomplete.  We attempt generation and log
        clear guidance if decoding fails.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Transformers model not loaded — call initialize() first")

        inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            )

        # Attempt to decode — the processor should handle audio codec output
        if hasattr(self._processor, "decode_audio"):
            audio = self._processor.decode_audio(output)
            if isinstance(audio, torch.Tensor):
                audio = audio.squeeze().cpu().numpy()
            return AudioProcessor.normalize_to_float32(np.asarray(audio, dtype=np.float32))

        # If no audio decoder is available, inform the user clearly
        logger.error(
            "Cannot decode Dia audio output without the official dia-tts package. "
            "Install with: pip install dia-tts"
        )
        return None

    # ------------------------------------------------------------------
    # Voice / language helpers
    # ------------------------------------------------------------------

    def list_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Return the two Dia speaker roles."""
        voices = list(DIA_VOICES.values())
        if language is not None and language != "en":
            return []
        return voices

    def get_default_voice(self, language: str) -> str:
        """Return default speaker role (S1)."""
        if language != "en":
            logger.warning(
                f"Dia supports English only — ignoring language '{language}'"
            )
        return DEFAULT_VOICE_ID

    async def cleanup(self):
        """Release model resources and free GPU memory."""
        self._dia_model = None
        self._model = None
        self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_initialized = False
        logger.info("Dia TTS resources released")
