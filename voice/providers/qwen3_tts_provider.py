"""
Qwen3-TTS Provider
Alibaba Qwen3-TTS: streaming TTS with voice cloning, VoiceDesign and custom voices.
https://github.com/QwenLM/Qwen3-TTS  (released 2026-01-22, Apache 2.0)

Official package API (pip install -U qwen-tts):
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(model_id, device_map="cuda:0", dtype=torch.bfloat16)
    wavs, sr = model.generate_custom_voice(text=..., language="English", speaker="Ryan", instruct=...)
    wavs, sr = model.generate_voice_design(text=..., language="English", instruct="<free-form description>")
    wavs, sr = model.generate_voice_clone(text=..., language="English", ref_audio=..., ref_text=...)

The CustomVoice and VoiceDesign capabilities live in *separate* model
checkpoints, so the right checkpoint is loaded lazily depending on the request:
- voice_description present -> VoiceDesign checkpoint (generate_voice_design)
- ref_audio_path present    -> CustomVoice checkpoint (generate_voice_clone)
- otherwise                 -> CustomVoice checkpoint (generate_custom_voice)
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

# Official package — class lives in `qwen_tts` (pip install -U qwen-tts)
try:
    from qwen_tts import Qwen3TTSModel
    QWEN3_PACKAGE = True
    logger.info("qwen_tts package found — using official Qwen3TTSModel")
except ImportError:
    Qwen3TTSModel = None
    QWEN3_PACKAGE = False
    logger.debug(
        "qwen_tts package not found. Install with: pip install -U qwen-tts"
    )

# ---------------------------------------------------------------------------
# Static voice catalogue — one representative voice per supported language
# ---------------------------------------------------------------------------
QWEN3_VOICES: Dict[str, Voice] = {
    "en": Voice(id="en_default", name="English Default", language="en", gender="neutral", quality_grade="A", description="Qwen3-TTS default English voice"),
    "zh": Voice(id="zh_default", name="Chinese Default", language="zh", gender="neutral", quality_grade="A", description="Qwen3-TTS default Mandarin Chinese voice"),
    "ja": Voice(id="ja_default", name="Japanese Default", language="ja", gender="neutral", quality_grade="A", description="Qwen3-TTS default Japanese voice"),
    "ko": Voice(id="ko_default", name="Korean Default", language="ko", gender="neutral", quality_grade="A", description="Qwen3-TTS default Korean voice"),
    "de": Voice(id="de_default", name="German Default", language="de", gender="neutral", quality_grade="A", description="Qwen3-TTS default German voice"),
    "fr": Voice(id="fr_default", name="French Default", language="fr", gender="neutral", quality_grade="A", description="Qwen3-TTS default French voice"),
    "ru": Voice(id="ru_default", name="Russian Default", language="ru", gender="neutral", quality_grade="A", description="Qwen3-TTS default Russian voice"),
    "pt": Voice(id="pt_default", name="Portuguese Default", language="pt", gender="neutral", quality_grade="A", description="Qwen3-TTS default Portuguese voice"),
    "es": Voice(id="es_default", name="Spanish Default", language="es", gender="neutral", quality_grade="A", description="Qwen3-TTS default Spanish voice"),
    "it": Voice(id="it_default", name="Italian Default", language="it", gender="neutral", quality_grade="A", description="Qwen3-TTS default Italian voice"),
}

# ISO 639-1 -> full language name expected by the qwen_tts API
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}

# CustomVoice premium speaker timbres (from the official model card).
# Used as the default speaker when the caller does not pass one.
DEFAULT_SPEAKER = "Ryan"

# Capability -> HuggingFace checkpoint suffix
_MODEL_SUFFIX = {"custom": "CustomVoice", "design": "VoiceDesign"}

# Qwen3-TTS output sample rate (24 kHz waveform; "12Hz" refers to codec frames)
NATIVE_SAMPLE_RATE = 24000


class Qwen3TTSProvider(TTSProvider):
    """Qwen3-TTS provider backed by the official `qwen_tts` package."""

    def __init__(self, config: TTSConfig, model_variant: str = "1.7B"):
        super().__init__(config)

        if model_variant not in ("0.6B", "1.7B"):
            raise ValueError(
                f"Unknown model_variant '{model_variant}'. Valid options: 0.6B, 1.7B"
            )

        self.model_variant = model_variant
        self.sample_rate_output = NATIVE_SAMPLE_RATE

        # Lazily-loaded checkpoints keyed by capability ("custom" / "design").
        # Only one is kept resident at a time to bound VRAM usage.
        self._models: Dict[str, object] = {}
        self._resident_capability: Optional[str] = None

    # ------------------------------------------------------------------
    # TTSProvider interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "qwen3-tts"

    @property
    def description(self) -> str:
        return (
            "Alibaba Qwen3-TTS: streaming TTS with custom voices, voice cloning "
            "and free-form VoiceDesign"
        )

    @property
    def supported_languages(self) -> List[str]:
        return ["en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]

    def _model_id(self, capability: str) -> str:
        return f"Qwen/Qwen3-TTS-12Hz-{self.model_variant}-{_MODEL_SUFFIX[capability]}"

    async def initialize(self):
        """Verify the package is available; checkpoints load lazily per request."""
        if self.is_initialized:
            return

        if not QWEN3_PACKAGE:
            raise ImportError(
                "Qwen3-TTS requires the 'qwen-tts' package. "
                "Install with: pip install -U qwen-tts"
            )

        logger.info(
            f"Qwen3-TTS ({self.model_variant}) ready on {self.config.device} "
            "(checkpoints load on first use)"
        )
        self.is_initialized = True

    def _load_model(self, capability: str):
        """Load (and cache) the checkpoint for a capability, evicting the other."""
        if capability in self._models:
            return self._models[capability]

        model_id = self._model_id(capability)
        dtype = self.config.torch_dtype
        if dtype is None:
            dtype = torch.bfloat16 if self.config.device != "cpu" else torch.float32

        logger.info(f"Loading Qwen3-TTS checkpoint: {model_id} (dtype={dtype})")

        # Evict the other capability's model first to bound VRAM.
        if self._resident_capability and self._resident_capability != capability:
            self._evict(self._resident_capability)

        device_map = self.config.device if self.config.device != "cpu" else "cpu"
        try:
            model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=device_map,
                dtype=dtype,
                attn_implementation="flash_attention_2",
            )
        except (ImportError, ValueError, RuntimeError) as exc:
            # flash-attention not installed/supported — retry with the default kernel
            logger.warning(f"flash_attention_2 unavailable ({exc}); using default attention")
            model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=device_map,
                dtype=dtype,
            )

        self._models[capability] = model
        self._resident_capability = capability
        return model

    def _evict(self, capability: str):
        model = self._models.pop(capability, None)
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Evicted Qwen3-TTS '{capability}' checkpoint")

    async def synthesize(
        self, text: str, voice_id: str, language: str = "en", **kwargs,
    ) -> List[np.ndarray]:
        chunks: List[np.ndarray] = []
        async for chunk in self.synthesize_streaming(text, voice_id, language, **kwargs):
            chunks.append(chunk)
        if not chunks:
            logger.warning(f"No audio generated for text: '{text[:80]}'")
            chunks.append(AudioProcessor.generate_silence(0.5, self.config.sample_rate))
        return chunks

    async def synthesize_streaming(
        self, text: str, voice_id: str, language: str = "en", **kwargs,
    ) -> AsyncGenerator[np.ndarray, None]:
        if not self.is_initialized:
            await self.initialize()

        voice_description: Optional[str] = kwargs.get("voice_description")
        ref_audio_path: Optional[str] = kwargs.get("ref_audio_path")
        ref_text: Optional[str] = kwargs.get("ref_text")
        speaker: str = kwargs.get("speaker") or DEFAULT_SPEAKER
        instruct: Optional[str] = kwargs.get("instruct")

        lang_name = LANGUAGE_NAMES.get(language, "English")

        # Decide capability + checkpoint once for the whole utterance.
        if voice_description:
            capability = "design"
        else:
            capability = "custom"
        model = self._load_model(capability)

        sentences = TextProcessor.split_sentences(text, max_length=150)
        logger.info(
            f"Qwen3-TTS [{capability}] {len(sentences)} sentence(s), language='{lang_name}'"
        )

        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            try:
                audio_np = self._generate(
                    model, capability, sentence, lang_name,
                    speaker=speaker, instruct=instruct,
                    voice_description=voice_description,
                    ref_audio_path=ref_audio_path, ref_text=ref_text,
                )
                if audio_np is not None and AudioProcessor.validate_audio(audio_np):
                    if self.sample_rate_output != self.config.sample_rate:
                        audio_np = AudioProcessor.resample_audio(
                            audio_np, self.sample_rate_output, self.config.sample_rate
                        )
                    yield audio_np
                else:
                    logger.warning(f"Empty audio for sentence: '{sentence[:50]}'")
            except Exception as exc:
                logger.error(f"Error synthesising '{sentence[:50]}': {exc}", exc_info=True)
                continue

    def _generate(
        self, model, capability: str, text: str, lang_name: str,
        speaker: str, instruct: Optional[str],
        voice_description: Optional[str],
        ref_audio_path: Optional[str], ref_text: Optional[str],
    ) -> Optional[np.ndarray]:
        """Run one inference call and return a float32 mono waveform."""
        with torch.inference_mode():
            if capability == "design":
                wavs, sr = model.generate_voice_design(
                    text=text, language=lang_name, instruct=voice_description,
                )
            elif ref_audio_path:
                wavs, sr = model.generate_voice_clone(
                    text=text, language=lang_name,
                    ref_audio=ref_audio_path, ref_text=ref_text,
                )
            else:
                call = dict(text=text, language=lang_name, speaker=speaker)
                if instruct:
                    call["instruct"] = instruct
                wavs, sr = model.generate_custom_voice(**call)

        if sr:
            self.sample_rate_output = int(sr)

        # The API returns a batch (wavs[0]) or a single waveform.
        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().detach().cpu().numpy()
        return AudioProcessor.normalize_to_float32(np.asarray(audio, dtype=np.float32))

    def list_voices(self, language: Optional[str] = None) -> List[Voice]:
        voices = list(QWEN3_VOICES.values())
        if language is not None:
            voices = [v for v in voices if v.language == language]
        return voices

    def get_default_voice(self, language: str) -> str:
        voice = QWEN3_VOICES.get(language)
        if voice is None:
            logger.warning(f"Language '{language}' not in Qwen3-TTS catalogue — using English.")
            return "en_default"
        return voice.id

    async def cleanup(self):
        for cap in list(self._models.keys()):
            self._evict(cap)
        self._resident_capability = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_initialized = False
        logger.info("Qwen3-TTS resources released")
