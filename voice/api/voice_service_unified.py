"""
Unified Voice Service - Consolidation of all voice service implementations
Supports: Multi-TTS providers (Kokoro, Chatterbox, F5-TTS) + GPU Queue Management
"""

import asyncio
import torch
import numpy as np
from typing import Optional, Generator, Dict, Any
import time
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Core ML imports for STT
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, BitsAndBytesConfig
from langdetect import detect

# TTS Provider imports
from ..providers import (
    get_tts_provider,
    list_available_providers,
    TTSConfig,
    TTSProvider,
)

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
torch.backends.cudnn.benchmark = True


def get_device_config():
    """Intelligently detect the best device and configuration based on user settings"""
    # Read user configuration from environment
    device_config = os.getenv("STT_DEVICE", os.getenv("TTS_DEVICE", "auto")).lower()
    use_fp16 = os.getenv("TTS_USE_FP16", "true").lower() == "true"
    # Allow disabling quantization (useful for new GPUs not supported by bitsandbytes)
    use_quantization = os.getenv("STT_USE_QUANTIZATION", "true").lower() == "true"

    print(f"🔧 Device Config: {device_config}")
    print(f"🔧 FP16 Enabled: {use_fp16}")
    print(f"🔧 Quantization Enabled: {use_quantization}")

    # Force CPU if requested
    if device_config == "cpu":
        print("💻 Using CPU (as configured)")
        return "cpu", False, torch.float32

    # Force CUDA if requested
    if device_config == "cuda":
        if not torch.cuda.is_available():
            print("❌ CUDA requested but not available - falling back to CPU")
            return "cpu", False, torch.float32
        print(f"🚀 Using CUDA (as configured)")
        dtype = torch.float16 if use_fp16 else torch.float32
        return "cuda", use_quantization, dtype

    # Auto-detect (default)
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - using CPU")
        return "cpu", False, torch.float32

    try:
        capability = torch.cuda.get_device_capability()
        major, minor = capability
        cuda_capability = major + minor / 10
        gpu_name = torch.cuda.get_device_name(0)

        if cuda_capability >= 7.0:
            dtype = torch.float16 if use_fp16 else torch.float32
            precision = "FP16" if use_fp16 else "FP32"
            print(f"✅ GPU detected: {gpu_name} (CUDA {cuda_capability})")
            print(f"🚀 Using GPU with {precision} precision")
            if use_fp16:
                print(f"⚡ Expected speedup: 2-3x faster, 50% less VRAM")
            return "cuda", use_quantization, dtype
        else:
            print(
                f"⚠️  GPU {gpu_name} has CUDA capability {cuda_capability} (requires 7.0+)"
            )
            print("⚠️  Falling back to CPU")
            return "cpu", False, torch.float32
    except Exception as e:
        print(f"⚠️  Error detecting GPU: {e}")
        return "cpu", False, torch.float32


_device, _use_quantization, torch_dtype = get_device_config()


@dataclass
class AudioConfig:
    sample_rate: int = 24000  # Match Chatterbox native quality
    channels: int = 1
    dtype: np.dtype = np.float32
    chunk_size: int = 1024


@dataclass
class ModelConfig:
    stt_model: str = os.getenv("STT_MODEL", "openai/whisper-small")
    tts_provider: str = os.getenv("TTS_DEFAULT_PROVIDER", "chatterbox")
    device: str = _device
    use_quantization: bool = _use_quantization
    enable_gpu_queue: bool = os.getenv("ENABLE_GPU_QUEUE", "false").lower() == "true"


class VoiceServiceUnified:
    """
    Unified Voice Service with Multi-TTS Provider Support + GPU Queue Management

    Features:
    - Multi-TTS providers (Kokoro, Chatterbox, F5-TTS)
    - Automatic GPU memory management
    - Coordination with Ollama for GPU sharing
    - Configurable via environment variables
    """

    def __init__(
        self, audio_config: AudioConfig = None, model_config: ModelConfig = None
    ):
        self.audio_config = audio_config or AudioConfig()
        self.model_config = model_config or ModelConfig()

        # STT
        self.stt_pipeline = None
        self.stt_processor = None

        # TTS - Provider system
        self.tts_provider: Optional[TTSProvider] = None
        self.tts_providers: Dict[str, TTSProvider] = {}  # Cache of loaded providers
        self.current_provider_name = self.model_config.tts_provider

        # GPU Queue Manager (optional)
        self.gpu_manager = None
        if (
            self.model_config.enable_gpu_queue
            and GPU_QUEUE_AVAILABLE
            and self.model_config.device == "cuda"
        ):
            self.gpu_manager = get_gpu_queue_manager()
            print("🔄 GPU Queue Management: ENABLED")
        else:
            print("🔄 GPU Queue Management: DISABLED")

        # Executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def initialize_models(self):
        """Initialize STT and TTS models"""
        print("🚀 Initializing Unified Voice Service...")

        # If GPU queue is enabled, unload Ollama models first
        if self.gpu_manager:
            print("[GPU-Queue] Requesting GPU for voice models...")
            await self.gpu_manager.request_gpu_for_tts(
                "voice-service", estimated_vram_mb=2500
            )

        # Initialize STT
        await self.initialize_stt()

        # Initialize default TTS provider
        await self.initialize_tts(self.model_config.tts_provider)

        print("✅ Unified Voice Service ready!")

    async def initialize_stt(self):
        """Initialize Speech-to-Text model"""
        if self.stt_pipeline is not None:
            return

        print(f"[STT] Loading {self.model_config.stt_model}...")

        try:
            # Load processor
            self.stt_processor = AutoProcessor.from_pretrained(
                self.model_config.stt_model
            )

            # Load model with quantization if available
            if (
                self.model_config.use_quantization
                and self.model_config.device == "cuda"
            ):
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
            print(f"[STT] ✓ Model loaded")

        except Exception as e:
            print(f"[STT] ✗ Error loading model: {e}")
            raise

    async def initialize_tts(self, provider_name: str, **kwargs):
        """
        Initialize TTS provider

        Args:
            provider_name: Name of the TTS provider
            **kwargs: Provider-specific arguments
        """
        if provider_name not in list_available_providers():
            raise ValueError(
                f"Unknown TTS provider: {provider_name}. "
                f"Available: {list_available_providers()}"
            )

        # Create cache key with kwargs for unique provider instances
        cache_key = provider_name
        if kwargs:
            cache_key = f"{provider_name}_{hash(frozenset(kwargs.items()))}"

        # Check if already loaded
        if cache_key in self.tts_providers:
            self.tts_provider = self.tts_providers[cache_key]
            self.current_provider_name = provider_name
            print(f"[TTS] Switched to cached {provider_name} provider")
            return

        print(f"[TTS] Loading {provider_name} provider...")
        if kwargs:
            print(f"[TTS] With options: {kwargs}")

        try:
            # Create TTS config
            tts_config = TTSConfig(
                device=self.model_config.device,
                use_quantization=self.model_config.use_quantization,
                sample_rate=24000,
                torch_dtype=torch_dtype,
            )

            # Get provider with additional kwargs
            provider = get_tts_provider(provider_name, tts_config, **kwargs)

            # Initialize provider
            await provider.initialize()

            # Cache and set as current
            self.tts_providers[cache_key] = provider
            self.tts_provider = provider
            self.current_provider_name = provider_name

            print(f"[TTS] ✓ {provider_name} provider loaded")

        except Exception as e:
            print(f"[TTS] ✗ Error loading {provider_name}: {e}")
            raise

    async def switch_tts_provider(self, provider_name: str, **kwargs):
        """Switch to a different TTS provider"""
        await self.initialize_tts(provider_name, **kwargs)
        print(f"[TTS] Switched to {provider_name}")

    def get_available_tts_providers(self) -> list:
        """Get list of available TTS providers"""
        return list_available_providers()

    def get_current_tts_provider_info(self) -> Dict[str, Any]:
        """Get information about current TTS provider"""
        if self.tts_provider:
            return self.tts_provider.get_info()
        return {}

    async def transcribe_audio(self, audio: np.ndarray) -> dict:
        """Transcribe audio to text"""
        if self.stt_pipeline is None:
            raise RuntimeError("STT model not initialized")

        # Mark GPU usage if queue management is enabled
        if self.gpu_manager:
            self.gpu_manager.mark_model_used("voice-service")

        start_time = time.time()

        try:
            # Prepare audio
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Process with Whisper
            inputs = self.stt_processor(
                audio, sampling_rate=self.audio_config.sample_rate, return_tensors="pt"
            )

            # Move to device and ensure correct dtype
            input_features = inputs.input_features.to(self.model_config.device)
            if input_features.dtype != torch.float32:
                input_features = input_features.to(torch.float32)

            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.stt_pipeline.generate(
                    input_features, max_new_tokens=128
                )

            # Decode
            text = self.stt_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            # Detect language
            try:
                detected_lang = detect(text) if text.strip() else "en"
            except:
                detected_lang = "en"

            elapsed = time.time() - start_time

            return {"text": text, "language": detected_lang, "time": elapsed}

        except Exception as e:
            print(f"[STT] Error: {e}")
            raise

    async def synthesize_speech(
        self, text: str, language: str = "en", voice_id: Optional[str] = None, **kwargs
    ) -> Generator[np.ndarray, None, None]:
        """
        Synthesize speech using current TTS provider

        Args:
            text: Text to synthesize
            language: Language code
            voice_id: Voice ID (optional)
            **kwargs: Provider-specific parameters

        Returns:
            Generator of audio chunks
        """
        if self.tts_provider is None:
            raise RuntimeError("TTS provider not initialized")

        # Mark GPU usage if queue management is enabled
        if self.gpu_manager:
            self.gpu_manager.mark_model_used("voice-service")

        # Auto-detect language
        try:
            detected_lang = detect(text)
            if detected_lang != language:
                print(
                    f"[TTS] Language mismatch: provided '{language}', detected '{detected_lang}', using detected"
                )
                language = detected_lang
        except Exception as e:
            print(f"[TTS] Could not detect language: {e}, using provided '{language}'")

        start_time = time.time()

        # Get default voice if not specified
        if voice_id is None:
            voice_id = self.tts_provider.get_default_voice(language)

        # Check language support
        if not self.tts_provider.supports_language(language):
            print(
                f"[TTS] Warning: {self.current_provider_name} doesn't support '{language}', "
                f"using default"
            )

        # Generate audio
        audio_chunks = await self.tts_provider.synthesize(
            text, voice_id, language, **kwargs
        )

        elapsed = time.time() - start_time
        print(
            f"⚡ TTS ({self.current_provider_name}): {elapsed:.2f}s - "
            f"[{language}] {len(audio_chunks)} chunks"
        )

        return audio_chunks

    async def synthesize_speech_streaming(
        self, text: str, language: str = "en", voice_id: Optional[str] = None, **kwargs
    ):
        """
        Synthesize speech with true streaming

        Args:
            text: Text to synthesize
            language: Language code
            voice_id: Voice ID (optional)
            **kwargs: Provider-specific parameters

        Yields:
            Audio chunks as they are generated
        """
        if self.tts_provider is None:
            raise RuntimeError("TTS provider not initialized")

        # Mark GPU usage if queue management is enabled
        if self.gpu_manager:
            self.gpu_manager.mark_model_used("voice-service")

        # Auto-detect language
        try:
            detected_lang = detect(text)
            if detected_lang != language:
                print(
                    f"[TTS] Language mismatch: provided '{language}', detected '{detected_lang}', using detected"
                )
                language = detected_lang
        except Exception as e:
            print(f"[TTS] Could not detect language: {e}, using provided '{language}'")

        start_time = time.time()

        # Get default voice if not specified
        if voice_id is None:
            voice_id = self.tts_provider.get_default_voice(language)

        # Check language support
        if not self.tts_provider.supports_language(language):
            print(
                f"[TTS] Warning: {self.current_provider_name} doesn't support '{language}', "
                f"using default"
            )

        # Check if provider supports streaming
        if hasattr(self.tts_provider, "synthesize_streaming"):
            # Use streaming method
            chunk_count = 0
            async for chunk in self.tts_provider.synthesize_streaming(
                text, voice_id, language, **kwargs
            ):
                chunk_count += 1
                yield chunk

            elapsed = time.time() - start_time
            print(
                f"⚡ TTS Streaming ({self.current_provider_name}): {elapsed:.2f}s - "
                f"[{language}] {chunk_count} chunks"
            )
        else:
            # Fallback to non-streaming
            audio_chunks = await self.tts_provider.synthesize(
                text, voice_id, language, **kwargs
            )
            for chunk in audio_chunks:
                yield chunk

            elapsed = time.time() - start_time
            print(
                f"⚡ TTS ({self.current_provider_name}): {elapsed:.2f}s - "
                f"[{language}] {len(audio_chunks)} chunks"
            )

    async def list_voices(self, language: Optional[str] = None):
        """List available voices for current TTS provider"""
        if self.tts_provider is None:
            return []

        return self.tts_provider.list_voices(language)

    async def unload_models(self):
        """Unload TTS models to free GPU memory"""
        print("🧹 Unloading TTS models...")

        # Cleanup all TTS providers
        for provider_name, provider in self.tts_providers.items():
            await provider.cleanup()

        self.tts_providers.clear()
        self.tts_provider = None

        # Cleanup GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc

            gc.collect()

        print("✅ TTS models unloaded")

    async def cleanup(self):
        """Cleanup all resources"""
        print("🧹 Cleaning up Unified Voice Service...")

        # Cleanup STT
        if self.stt_pipeline is not None:
            del self.stt_pipeline
            self.stt_pipeline = None
        if self.stt_processor is not None:
            del self.stt_processor
            self.stt_processor = None

        # Cleanup all TTS providers
        for provider_name, provider in self.tts_providers.items():
            await provider.cleanup()

        self.tts_providers.clear()
        self.tts_provider = None

        # Cleanup GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Cleanup executor
        self.executor.shutdown(wait=True)

        print("✅ Cleanup complete")

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU queue status (if enabled)"""
        if self.gpu_manager:
            return self.gpu_manager.get_status()
        return {"gpu_queue": "disabled"}
