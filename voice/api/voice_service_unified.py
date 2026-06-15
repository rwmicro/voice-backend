"""
Unified Voice Service - Multi-provider TTS + STT with faster-whisper
Supports: Kokoro, Chatterbox, F5-TTS, Qwen3-TTS, Orpheus, Dia + GPU Queue Management
STT: faster-whisper (default) with word timestamps and VAD
"""

import asyncio
import torch
import numpy as np
from typing import Optional, Generator, Dict, Any, List
import time
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

# TTS Provider imports
from ..providers import (
    get_tts_provider,
    list_available_providers,
    TTSConfig,
    TTSProvider,
)

# Centralised configuration (single source of truth)
from config.settings import settings

# Language detection
from voice.utils.language_detector import detect_language

# Latency metrics
from voice.utils.metrics import get_metrics_tracker

# Logger
from voice.utils.logger import get_logger

logger = get_logger(__name__)

# GPU Queue Manager
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../utils"))
    from gpu_queue_manager import get_gpu_queue_manager
    GPU_QUEUE_AVAILABLE = True
except ImportError:
    print("[Voice] GPU Queue Manager not available - GPU management disabled")
    GPU_QUEUE_AVAILABLE = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Disabled on purpose: STT/TTS inputs have highly variable lengths, so cuDNN
# autotuning re-benchmarks kernels on every new shape and hurts latency.
torch.backends.cudnn.benchmark = False


def get_device_config():
    """Intelligently detect the best device and configuration based on user settings"""
    device_config = (settings.STT_DEVICE or settings.TTS_DEVICE or "auto").lower()
    use_fp16 = settings.TTS_USE_FP16
    use_quantization = settings.STT_USE_QUANTIZATION

    print(f"Device Config: {device_config}")
    print(f"FP16 Enabled: {use_fp16}")
    print(f"Quantization Enabled: {use_quantization}")

    if device_config == "cpu":
        print("Using CPU (as configured)")
        return "cpu", False, torch.float32

    if device_config == "cuda":
        if not torch.cuda.is_available():
            print("CUDA requested but not available - falling back to CPU")
            return "cpu", False, torch.float32
        dtype = torch.float16 if use_fp16 else torch.float32
        return "cuda", use_quantization, dtype

    if not torch.cuda.is_available():
        print("CUDA not available - using CPU")
        return "cpu", False, torch.float32

    try:
        capability = torch.cuda.get_device_capability()
        major, minor = capability
        cuda_capability = major + minor / 10
        gpu_name = torch.cuda.get_device_name(0)

        if cuda_capability >= 7.0:
            dtype = torch.float16 if use_fp16 else torch.float32
            print(f"GPU detected: {gpu_name} (CUDA {cuda_capability})")
            return "cuda", use_quantization, dtype
        else:
            print(f"GPU {gpu_name} has CUDA capability {cuda_capability} (requires 7.0+), using CPU")
            return "cpu", False, torch.float32
    except Exception as e:
        print(f"Error detecting GPU: {e}")
        return "cpu", False, torch.float32


_device, _use_quantization, torch_dtype = get_device_config()


@dataclass
class AudioConfig:
    sample_rate: int = 24000
    channels: int = 1
    dtype: np.dtype = np.float32
    chunk_size: int = 1024


@dataclass
class ModelConfig:
    stt_model: str = settings.STT_MODEL
    stt_backend: str = settings.STT_BACKEND
    stt_profile: str = settings.STT_PROFILE
    tts_provider: str = settings.TTS_DEFAULT_PROVIDER
    device: str = _device
    use_quantization: bool = _use_quantization
    enable_gpu_queue: bool = settings.ENABLE_GPU_QUEUE
    stt_word_timestamps: bool = settings.STT_WORD_TIMESTAMPS
    stt_vad_filter: bool = settings.STT_VAD_FILTER
    stt_language: Optional[str] = settings.STT_LANGUAGE


class VoiceServiceUnified:
    """
    Unified Voice Service with Multi-TTS Provider Support + GPU Queue Management

    STT backends:
    - faster-whisper (default): 2-4x faster, word timestamps, built-in VAD
    - transformers: legacy fallback

    TTS providers:
    - kokoro, chatterbox, f5-tts, qwen3-tts, orpheus, dia
    """

    def __init__(self, audio_config: AudioConfig = None, model_config: ModelConfig = None):
        self.audio_config = audio_config or AudioConfig()
        self.model_config = model_config or ModelConfig()

        # STT - unified interface (faster-whisper or transformers)
        self.stt_backend = None          # FasterWhisperSTT instance
        self.stt_pipeline = None         # Legacy transformers model
        self.stt_processor = None        # Legacy transformers processor

        # TTS - Provider system
        self.tts_provider: Optional[TTSProvider] = None
        self.tts_providers: Dict[str, TTSProvider] = {}
        self.current_provider_name = self.model_config.tts_provider

        # Metrics
        self.metrics = get_metrics_tracker()

        # GPU Queue Manager (optional)
        self.gpu_manager = None
        if (
            self.model_config.enable_gpu_queue
            and GPU_QUEUE_AVAILABLE
            and self.model_config.device == "cuda"
        ):
            self.gpu_manager = get_gpu_queue_manager()
            print("GPU Queue Management: ENABLED")
        else:
            print("GPU Queue Management: DISABLED")

        self.executor = ThreadPoolExecutor(max_workers=4)

        # Serializes all GPU work (provider/STT switches, TTS synthesis, STT
        # transcription). The service is a shared singleton on a single GPU, so
        # this both prevents state corruption from concurrent switches and acts
        # as a GPU-concurrency guard against OOM.
        self._inference_lock = asyncio.Lock()

    async def initialize_models(self):
        """Initialize STT and TTS models"""
        print("Initializing Unified Voice Service...")

        if self.gpu_manager:
            print("[GPU-Queue] Requesting GPU for voice models...")
            await self.gpu_manager.request_gpu_for_tts("voice-service", estimated_vram_mb=2500)

        try:
            async with asyncio.timeout(300):  # 5 minute timeout for model download
                await self.initialize_stt()
                await self.initialize_tts(self.model_config.tts_provider)
        except asyncio.TimeoutError:
            raise RuntimeError("Model initialization timed out after 5 minutes. Check your internet connection or model cache.")

        print("Unified Voice Service ready!")

    async def initialize_stt(self):
        """Initialize Speech-to-Text model"""
        backend = self.model_config.stt_backend

        if backend == "faster-whisper":
            await self._initialize_faster_whisper()
        else:
            await self._initialize_transformers_whisper()

    async def _initialize_faster_whisper(self):
        """Initialize faster-whisper STT backend"""
        if self.stt_backend is not None:
            return

        profile = self.model_config.stt_profile
        print(f"[STT] Loading faster-whisper (profile: {profile})...")

        try:
            from voice.stt.faster_whisper_stt import FasterWhisperSTT
            self.stt_backend = FasterWhisperSTT(
                model_name=profile,
                device=self.model_config.device,
                language=self.model_config.stt_language,
            )
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.stt_backend.load)
            print(f"[STT] faster-whisper loaded (profile: {profile})")
        except ImportError:
            print("[STT] faster-whisper not installed, falling back to transformers")
            self.model_config.stt_backend = "transformers"
            await self._initialize_transformers_whisper()

    async def _initialize_transformers_whisper(self):
        """Initialize legacy transformers Whisper"""
        if self.stt_pipeline is not None:
            return

        print(f"[STT] Loading transformers Whisper: {self.model_config.stt_model}...")

        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig

            self.stt_processor = AutoProcessor.from_pretrained(self.model_config.stt_model)

            if self.model_config.use_quantization and self.model_config.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
                )
                self.stt_pipeline = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_config.stt_model,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
            else:
                self.stt_pipeline = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_config.stt_model, torch_dtype=torch_dtype
                ).to(self.model_config.device)

            self.stt_pipeline.eval()
            print(f"[STT] Transformers Whisper loaded")
        except Exception as e:
            print(f"[STT] Error loading model: {e}")
            raise

    async def initialize_tts(self, provider_name: str, **kwargs):
        """Initialize TTS provider"""
        if provider_name not in list_available_providers():
            raise ValueError(
                f"Unknown TTS provider: {provider_name}. "
                f"Available: {list_available_providers()}"
            )

        cache_key = provider_name
        if kwargs:
            cache_key = f"{provider_name}_{hash(frozenset(kwargs.items()))}"

        if cache_key in self.tts_providers:
            self.tts_provider = self.tts_providers[cache_key]
            self.current_provider_name = provider_name
            print(f"[TTS] Switched to cached {provider_name} provider")
            return

        print(f"[TTS] Loading {provider_name} provider...")

        try:
            tts_config = TTSConfig(
                device=self.model_config.device,
                use_quantization=self.model_config.use_quantization,
                sample_rate=24000,
                torch_dtype=torch_dtype,
            )

            provider = get_tts_provider(provider_name, tts_config, **kwargs)
            await provider.initialize()

            self.tts_providers[cache_key] = provider
            self.tts_provider = provider
            self.current_provider_name = provider_name

            # Evict the oldest cached provider when over the limit
            MAX_CACHED_PROVIDERS = 3
            if len(self.tts_providers) > MAX_CACHED_PROVIDERS:
                oldest_key = next(
                    k for k in self.tts_providers
                    if k != cache_key
                )
                old_provider = self.tts_providers.pop(oldest_key)
                await old_provider.cleanup()
                logger.info(f"[TTS] Evicted cached provider: {oldest_key} (cache limit={MAX_CACHED_PROVIDERS})")

            print(f"[TTS] {provider_name} provider loaded")
        except Exception as e:
            print(f"[TTS] Error loading {provider_name}: {e}")
            raise

    async def switch_tts_provider(self, provider_name: str, **kwargs):
        """Switch to a different TTS provider (serialized against inference)."""
        async with self._inference_lock:
            await self.initialize_tts(provider_name, **kwargs)

    async def _ensure_provider(self, provider_name: Optional[str], **kwargs):
        """Load/switch provider if needed. Caller must hold the inference lock."""
        if provider_name and provider_name != self.current_provider_name:
            logger.info(f"[TTS] Switching provider to: {provider_name}")
            await self.initialize_tts(provider_name, **kwargs)

    def get_available_tts_providers(self) -> List[str]:
        return list_available_providers()

    def get_current_tts_provider_info(self) -> Dict[str, Any]:
        if self.tts_provider:
            return self.tts_provider.get_info()
        return {}

    @property
    def stt_is_ready(self) -> bool:
        return self.stt_backend is not None or self.stt_pipeline is not None

    async def transcribe_audio(
        self,
        audio: np.ndarray,
        word_timestamps: bool = None,
        vad_filter: bool = None,
        language: Optional[str] = None,
    ) -> dict:
        """
        Transcribe audio to text

        Args:
            audio: Audio array (float32, any sample rate)
            word_timestamps: Enable word-level timestamps (overrides config)
            vad_filter: Enable VAD filtering (overrides config)
            language: Force language for this call only (None = use config /
                      auto-detect). Passed per-call, never mutates shared state.

        Returns:
            dict with: text, language, time, words (if timestamps enabled)
        """
        if not self.stt_is_ready:
            raise RuntimeError("STT model not initialized")

        if self.gpu_manager:
            self.gpu_manager.mark_model_used("voice-service")

        # Apply config defaults
        use_word_ts = word_timestamps if word_timestamps is not None else self.model_config.stt_word_timestamps
        use_vad = vad_filter if vad_filter is not None else self.model_config.stt_vad_filter
        use_language = language if language is not None else self.model_config.stt_language

        start_time = time.time()

        # Ensure mono float32
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        try:
            async with self._inference_lock:
                if self.stt_backend is not None:
                    # Use faster-whisper (preferred)
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor,
                        lambda: self.stt_backend.transcribe(
                            audio,
                            sample_rate=self.audio_config.sample_rate,
                            word_timestamps=use_word_ts,
                            vad_filter=use_vad,
                            language=use_language,
                        )
                    )
                else:
                    # Legacy transformers path
                    result = await self._transcribe_transformers(audio, language=use_language)

            elapsed = time.time() - start_time
            result["time"] = elapsed

            # Record metric
            self.metrics.record("stt", self.model_config.stt_backend, elapsed * 1000)

            return result

        except Exception as e:
            print(f"[STT] Error: {e}")
            raise

    async def _transcribe_transformers(self, audio: np.ndarray, language: Optional[str] = None) -> dict:
        """Legacy transformers Whisper transcription.

        Args:
            language: Force a language (ISO 639-1). None lets Whisper auto-detect.
        """
        inputs = self.stt_processor(
            audio, sampling_rate=self.audio_config.sample_rate, return_tensors="pt"
        )

        input_features = inputs.input_features.to(self.model_config.device)
        if input_features.dtype != torch.float32:
            input_features = input_features.to(torch.float32)

        gen_kwargs: Dict[str, Any] = {
            # Whisper decodes at most ~448 tokens per 30s window; 128 truncated
            # longer utterances. Use the model's full budget.
            "max_new_tokens": 440,
        }
        # Force the target language/task when requested so short clips aren't
        # mis-detected by the decoder.
        if language:
            try:
                forced = self.stt_processor.get_decoder_prompt_ids(
                    language=language, task="transcribe"
                )
                gen_kwargs["forced_decoder_ids"] = forced
            except Exception:
                pass

        with torch.no_grad():
            predicted_ids = self.stt_pipeline.generate(input_features, **gen_kwargs)

        text = self.stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        if language:
            detected_lang = language
        else:
            try:
                detected_lang = detect_language(text) if text.strip() else "en"
            except Exception:
                detected_lang = "en"

        return {"text": text, "language": detected_lang, "words": None}

    async def synthesize_speech(
        self,
        text: str,
        language: str = "en",
        voice_id: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> List[np.ndarray]:
        """Synthesize speech using current TTS provider.

        Provider selection and synthesis happen atomically under the inference
        lock so concurrent requests can't switch the provider mid-synthesis.
        """
        if self.gpu_manager:
            self.gpu_manager.mark_model_used("voice-service")

        language = self._maybe_detect_language(text, language)
        start_time = time.time()

        async with self._inference_lock:
            await self._ensure_provider(provider)
            prov = self.tts_provider
            provider_name = self.current_provider_name
            if prov is None:
                raise RuntimeError("TTS provider not initialized")

            if voice_id is None:
                voice_id = prov.get_default_voice(language)

            if not prov.supports_language(language):
                logger.warning(
                    f"[TTS] {provider_name} doesn't support '{language}', "
                    "using default voice for fallback language"
                )
                language = prov.supported_languages[0] if prov.supported_languages else "en"

            audio_chunks = await prov.synthesize(text, voice_id, language, **kwargs)

        elapsed = time.time() - start_time
        self.metrics.record("tts", provider_name, elapsed * 1000)

        print(f"TTS ({provider_name}): {elapsed:.2f}s [{language}] {len(audio_chunks)} chunks")

        return audio_chunks

    def _maybe_detect_language(self, text: str, language: str) -> str:
        """Auto-detect language only when the caller left the default ("en").

        An explicitly chosen non-default language is respected — auto-detection
        on short texts is unreliable and must not override a deliberate choice.
        """
        if language and language != "en":
            return language
        try:
            detected = detect_language(text)
            if detected != language:
                logger.info(f"[TTS] Auto-detected language '{detected}'")
            return detected
        except Exception:
            return language

    async def synthesize_speech_streaming(
        self,
        text: str,
        language: str = "en",
        voice_id: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        """Synthesize speech with streaming.

        The inference lock is held for the whole stream so the active provider
        can't be swapped while chunks are being generated (single-GPU service).
        """
        if self.gpu_manager:
            self.gpu_manager.mark_model_used("voice-service")

        language = self._maybe_detect_language(text, language)
        start_time = time.time()

        async with self._inference_lock:
            await self._ensure_provider(provider)
            prov = self.tts_provider
            provider_name = self.current_provider_name
            if prov is None:
                raise RuntimeError("TTS provider not initialized")

            if voice_id is None:
                voice_id = prov.get_default_voice(language)

            if not prov.supports_language(language):
                print(f"[TTS] {provider_name} doesn't support '{language}'")

            chunk_count = 0
            if hasattr(prov, "synthesize_streaming"):
                async for chunk in prov.synthesize_streaming(text, voice_id, language, **kwargs):
                    chunk_count += 1
                    yield chunk
            else:
                for chunk in await prov.synthesize(text, voice_id, language, **kwargs):
                    chunk_count += 1
                    yield chunk

        elapsed = time.time() - start_time
        self.metrics.record("tts", provider_name, elapsed * 1000)
        print(f"TTS Streaming ({provider_name}): {elapsed:.2f}s [{language}] {chunk_count} chunks")

    async def synthesize_dialogue(
        self,
        turns: List[Dict[str, str]],
        language: str = "en",
        provider: Optional[str] = None,
        **kwargs
    ) -> List[np.ndarray]:
        """
        Synthesize multi-speaker dialogue (requires Dia provider)

        Args:
            turns: List of {"speaker": "S1"|"S2", "text": "..."}
            language: Language code
            provider: Provider to switch to before synthesis (e.g. "dia")
            **kwargs: Provider-specific params (e.g. speaker_audio_s1/s2)

        Returns:
            List of audio chunks
        """
        start_time = time.time()

        async with self._inference_lock:
            await self._ensure_provider(provider)
            prov = self.tts_provider
            provider_name = self.current_provider_name
            if prov is None:
                raise RuntimeError("TTS provider not initialized")

            if not hasattr(prov, "synthesize_dialogue"):
                raise ValueError(
                    f"Provider '{provider_name}' does not support multi-speaker dialogue. "
                    "Switch to the 'dia' provider: POST /api/voice/tts/switch"
                )

            audio_chunks = await prov.synthesize_dialogue(turns, **kwargs)

        elapsed = time.time() - start_time
        self.metrics.record("tts", f"{provider_name}_dialogue", elapsed * 1000)

        return audio_chunks

    async def list_voices(self, language: Optional[str] = None):
        if self.tts_provider is None:
            return []
        return self.tts_provider.list_voices(language)

    def get_latency_stats(self, category: str = None) -> Dict[str, Any]:
        """Get p50/p95/p99 latency stats per provider"""
        return self.metrics.get_stats(category=category)

    async def switch_stt_profile(self, profile: str):
        """
        Switch STT model profile at runtime

        Args:
            profile: "fast" | "default" | "accurate"
        """
        from voice.stt.faster_whisper_stt import STT_PROFILES
        if profile not in STT_PROFILES:
            raise ValueError(f"Unknown STT profile: {profile}. Available: {list(STT_PROFILES.keys())}")

        print(f"[STT] Switching profile to: {profile}")

        async with self._inference_lock:
            # Unload current backend
            if self.stt_backend is not None:
                self.stt_backend.unload()
                self.stt_backend = None

            self.model_config.stt_profile = profile

            await self._initialize_faster_whisper()
        print(f"[STT] Profile switched to: {profile}")

    async def unload_models(self):
        """Unload TTS models to free GPU memory"""
        print("Unloading TTS models...")

        for provider in self.tts_providers.values():
            await provider.cleanup()

        self.tts_providers.clear()
        self.tts_provider = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        print("TTS models unloaded")

    async def cleanup(self):
        """Cleanup all resources"""
        print("Cleaning up Unified Voice Service...")

        # Cleanup STT
        if self.stt_backend is not None:
            self.stt_backend.unload()
            self.stt_backend = None

        if self.stt_pipeline is not None:
            del self.stt_pipeline
            self.stt_pipeline = None
        if self.stt_processor is not None:
            del self.stt_processor
            self.stt_processor = None

        for provider in self.tts_providers.values():
            await provider.cleanup()

        self.tts_providers.clear()
        self.tts_provider = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.executor.shutdown(wait=True)
        print("Cleanup complete")

    def get_gpu_status(self) -> Dict[str, Any]:
        if self.gpu_manager:
            return self.gpu_manager.get_status()
        return {"gpu_queue": "disabled"}
