"""
Unified Voice Service API
FastAPI server with Multi-TTS providers + GPU Queue Management

TTS providers: Kokoro, Chatterbox, F5-TTS, Qwen3-TTS, Orpheus, Dia
STT backend:   faster-whisper (word timestamps, VAD) + transformers fallback

New endpoints (v4.0):
  POST /api/voice/tts/design     - VoiceDesign: describe voice in natural language
  POST /api/voice/tts/dialogue   - Multi-speaker dialogue (Dia provider)
  GET  /api/voice/stats          - Latency p50/p95/p99 per provider
  GET  /api/voice/stt/profiles   - Available STT profiles
  POST /api/voice/stt/switch     - Switch STT profile at runtime
"""

from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    HTTPException,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
import numpy as np
import json
from typing import Optional, List, Dict, Any
import base64
import io
import uuid
from datetime import datetime

from voice.api.voice_service_unified import (
    VoiceServiceUnified,
    AudioConfig,
    ModelConfig,
)
from voice.api.middleware import (
    RequestTrackingMiddleware,
    ErrorHandlingMiddleware,
    RateLimitMiddleware,
    RequestTimeoutMiddleware,
)
from voice.utils.security import validate_audio_path, sanitize_error_message
from config.settings import settings, ensure_directories

# Setup structured logging
try:
    from voice.utils.logger import setup_logger, get_logger
    setup_logger(log_file=settings.LOG_FILE, level=settings.LOG_LEVEL)
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

voice_service: Optional[VoiceServiceUnified] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global voice_service
    logger.info("=" * 60)
    logger.info("Starting Voice Backend API v4.0.0")
    logger.info("TTS: Kokoro | Chatterbox | F5-TTS | Qwen3-TTS | Orpheus | Dia")
    logger.info("STT: faster-whisper (word timestamps + VAD)")
    logger.info("=" * 60)

    ensure_directories()

    audio_config = AudioConfig()
    model_config = ModelConfig(
        stt_model=settings.STT_MODEL,
        stt_backend=settings.STT_BACKEND,
        stt_profile=settings.STT_PROFILE,
        tts_provider=settings.TTS_DEFAULT_PROVIDER,
        enable_gpu_queue=settings.ENABLE_GPU_QUEUE,
        stt_word_timestamps=settings.STT_WORD_TIMESTAMPS,
        stt_vad_filter=settings.STT_VAD_FILTER,
        stt_language=settings.STT_LANGUAGE,
    )

    voice_service = VoiceServiceUnified(audio_config, model_config)
    await voice_service.initialize_models()

    logger.info("=" * 60)
    logger.info("Voice Backend API ready!")
    logger.info(f"Port: {settings.PORT}")
    logger.info(f"TTS: {voice_service.current_provider_name}")
    logger.info(f"STT: {model_config.stt_backend} / profile={model_config.stt_profile}")
    logger.info("=" * 60)

    yield

    if voice_service:
        await voice_service.cleanup()


# Initialize FastAPI app
app = FastAPI(
    title="Unified Voice Service API",
    version="4.0.0",
    description=(
        "Multi-TTS provider voice service with GPU queue management.\n\n"
        "**TTS Providers:** Kokoro, Chatterbox, F5-TTS, Qwen3-TTS, Orpheus, Dia\n"
        "**STT Backend:** faster-whisper (word timestamps, VAD) + transformers fallback\n"
        "**New in v4:** VoiceDesign, multi-speaker dialogue, latency metrics, STT profiles"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestTimeoutMiddleware, timeout_seconds=settings.REQUEST_TIMEOUT_SECONDS)
app.add_middleware(RateLimitMiddleware, max_requests=settings.RATE_LIMIT_REQUESTS, window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestTrackingMiddleware)


# ===========================
# Request / Response models
# ===========================

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    language: str = Field(
        default="en",
        pattern=r"^[a-z]{2}(-[a-z]{2})?$",
        description="Language code (ISO 639-1)",
    )
    provider: Optional[str] = Field(None, description="TTS provider to use")
    voice_id: Optional[str] = Field(None, max_length=100, description="Voice ID")

    # Common params
    speed: Optional[float] = Field(1.0, gt=0.5, lt=2.0, description="Speech speed")

    # F5-TTS params
    ref_audio: Optional[str] = Field(None, description="Reference audio path for F5-TTS")
    ref_text: Optional[str] = Field(None, max_length=1000, description="Reference text for F5-TTS")

    # Chatterbox params
    audio_prompt_path: Optional[str] = Field(None, description="Audio prompt path for Chatterbox")
    exaggeration: Optional[float] = Field(0.5, ge=0.25, le=2.0, description="Chatterbox expressiveness")
    temperature: Optional[float] = Field(0.8, ge=0.05, le=5.0, description="Chatterbox temperature")
    cfg_weight: Optional[float] = Field(0.5, ge=0.2, le=1.0, description="Chatterbox CFG weight")

    # Qwen3-TTS params
    voice_description: Optional[str] = Field(
        None, max_length=500,
        description="Qwen3-TTS VoiceDesign: natural language voice description "
                    "(e.g. 'a calm British male with a deep voice')"
    )

    # Orpheus params
    emotion: Optional[str] = Field(
        None,
        description="Orpheus emotion tag: happy, sad, angry, whispering, laughing, sighing, crying, surprised"
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v):
        if v is not None:
            allowed = ["kokoro", "chatterbox", "f5-tts", "qwen3-tts", "orpheus", "dia"]
            if v not in allowed:
                raise ValueError(f"Provider must be one of: {', '.join(allowed)}")
        return v

    @field_validator("emotion")
    @classmethod
    def validate_emotion(cls, v):
        if v is not None:
            allowed = ["happy", "sad", "angry", "whispering", "laughing", "sighing", "crying", "surprised"]
            if v.lower() not in allowed:
                raise ValueError(f"Emotion must be one of: {', '.join(allowed)}")
        return v.lower() if v else v


class VoiceDesignRequest(BaseModel):
    """Request for VoiceDesign endpoint (Qwen3-TTS)"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_description: str = Field(
        ..., min_length=5, max_length=500,
        description="Natural language description of the desired voice. "
                    "Examples: 'a calm British male with a deep voice', "
                    "'a cheerful young French woman', 'a professional news anchor'"
    )
    language: str = Field(default="en", pattern=r"^[a-z]{2}$")
    speed: Optional[float] = Field(1.0, gt=0.5, lt=2.0)
    ref_audio_path: Optional[str] = Field(None, description="Optional reference audio for additional voice style")


class DialogueTurn(BaseModel):
    speaker: str = Field(..., pattern=r"^S[12]$", description="Speaker ID: S1 or S2")
    text: str = Field(..., min_length=1, max_length=1000)


class DialogueRequest(BaseModel):
    """Request for multi-speaker dialogue synthesis (Dia provider)"""
    turns: List[DialogueTurn] = Field(..., min_length=1, max_length=50)
    speaker_audio_s1: Optional[str] = Field(None, description="Reference audio path for speaker S1")
    speaker_audio_s2: Optional[str] = Field(None, description="Reference audio path for speaker S2")
    language: str = Field(default="en", pattern=r"^[a-z]{2}$")


class STTRequest(BaseModel):
    word_timestamps: bool = Field(False, description="Return word-level timestamps")
    vad_filter: bool = Field(True, description="Apply Voice Activity Detection filtering")
    language: Optional[str] = Field(None, description="Force language (None = auto-detect)")


class STTResponse(BaseModel):
    text: str
    language: str
    language_probability: Optional[float] = None
    duration: Optional[float] = None
    time: float
    words: Optional[List[Dict[str, Any]]] = None
    segments: Optional[List[Dict[str, Any]]] = None


class SwitchProviderRequest(BaseModel):
    provider: str
    model_variant: Optional[str] = None


class SwitchSTTProfileRequest(BaseModel):
    profile: str = Field(..., description="STT profile: fast, default, or accurate")


# ===========================
# Helpers
# ===========================

def _concat_chunks(audio_chunks: List[np.ndarray]) -> np.ndarray:
    """Concatenate audio chunks, raising a clear error when none were produced."""
    chunks = [c for c in audio_chunks if c is not None and len(c) > 0]
    if not chunks:
        raise HTTPException(
            status_code=502,
            detail="TTS provider produced no audio. The model may be unavailable "
                   "or the input could not be synthesized.",
        )
    return np.concatenate(chunks)


def _encode_wav(audio_chunks: List[np.ndarray]) -> bytes:
    """Concatenate chunks and encode them as a single in-memory WAV file."""
    import soundfile as sf
    audio_data = _concat_chunks(audio_chunks)
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, voice_service.audio_config.sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def _provider_unavailable_detail(provider: Optional[str], error: Exception) -> str:
    """Human-readable 503 detail when a provider's package/model is missing."""
    name = provider or (voice_service.current_provider_name if voice_service else "selected")
    return (
        f"TTS provider '{name}' is unavailable on this server "
        f"(missing package or model weights). {error}"
    )


async def _ws_send_chunk(websocket: WebSocket, chunk: np.ndarray, index: int, fmt: str):
    """Send one TTS audio chunk over the WebSocket.

    Default format is raw little-endian int16 PCM sent as a *binary* frame
    (preceded by a JSON header). This avoids the ~2-2.7x overhead of the legacy
    path that wrapped every chunk in a full WAV file and base64-encoded it.
    Set fmt="wav" for backward-compatible base64 WAV chunks.
    """
    from voice.utils.audio import AudioProcessor
    sr = voice_service.audio_config.sample_rate

    if fmt == "wav":
        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, chunk, sr, format="WAV")
        buffer.seek(0)
        await websocket.send_json({
            "type": "tts_chunk",
            "audio": base64.b64encode(buffer.read()).decode("utf-8"),
            "chunk_index": index,
            "format": "wav",
        })
        return

    if fmt == "opus":
        # Opus gives ~5-10x less bandwidth than PCM. Encoded via libsndfile
        # (already a dependency); falls back to PCM if unsupported by the build.
        try:
            import soundfile as sf
            buffer = io.BytesIO()
            sf.write(buffer, chunk, sr, format="OGG", subtype="OPUS")
            buffer.seek(0)
            opus_bytes = buffer.read()
            await websocket.send_json({
                "type": "tts_chunk",
                "chunk_index": index,
                "format": "ogg_opus",
                "sample_rate": sr,
                "channels": 1,
                "bytes": len(opus_bytes),
            })
            await websocket.send_bytes(opus_bytes)
            return
        except Exception as e:
            logger.warning(f"Opus encoding unavailable ({e}); falling back to PCM")

    pcm = AudioProcessor.convert_to_int16(chunk).tobytes()
    await websocket.send_json({
        "type": "tts_chunk",
        "chunk_index": index,
        "format": "pcm_s16le",
        "sample_rate": sr,
        "channels": 1,
        "bytes": len(pcm),
    })
    await websocket.send_bytes(pcm)


# ===========================
# Health & Info
# ===========================

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Unified Voice Service API",
        "version": "4.0.0",
        "tts_providers": ["kokoro", "chatterbox", "f5-tts", "qwen3-tts", "orpheus", "dia"],
        "stt_backends": ["faster-whisper", "transformers"],
        "new_features": [
            "VoiceDesign: POST /api/voice/tts/design",
            "Multi-speaker dialogue: POST /api/voice/tts/dialogue",
            "Latency stats: GET /api/voice/stats",
            "STT profiles: GET /api/voice/stt/profiles",
            "Word timestamps: POST /api/voice/stt with word_timestamps=true",
        ],
        "endpoints": {
            "health": "GET /api/voice/health",
            "stt": "POST /api/voice/stt",
            "stt_profiles": "GET /api/voice/stt/profiles",
            "stt_switch": "POST /api/voice/stt/switch",
            "tts": "POST /api/voice/tts",
            "tts_design": "POST /api/voice/tts/design",
            "tts_dialogue": "POST /api/voice/tts/dialogue",
            "providers": "GET /api/voice/tts/providers",
            "switch": "POST /api/voice/tts/switch",
            "voices": "GET /api/voice/tts/voices",
            "stats": "GET /api/voice/stats",
            "unload": "POST /api/voice/unload",
            "gpu_status": "GET /api/voice/gpu/status",
            "stream": "WS /api/voice/stream",
        },
    }


@app.get("/api/voice/health")
async def health_check():
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    device_info = {}
    try:
        from voice.utils.device import get_detailed_device_info
        device_info = get_detailed_device_info()
    except Exception:
        device_info = {"device": voice_service.model_config.device}

    gpu_status = "N/A"
    if voice_service.gpu_manager:
        try:
            gpu_status = voice_service.get_gpu_status()
        except Exception as e:
            gpu_status = {"error": str(e)}

    import shutil
    try:
        cache_usage = shutil.disk_usage(settings.VOICE_CACHE_DIR)
        disk_info = {
            "cache_dir": settings.VOICE_CACHE_DIR,
            "total_gb": round(cache_usage.total / (1024**3), 2),
            "used_gb": round(cache_usage.used / (1024**3), 2),
            "free_gb": round(cache_usage.free / (1024**3), 2),
            "usage_percent": round((cache_usage.used / cache_usage.total) * 100, 2),
        }
    except Exception as e:
        disk_info = {"error": str(e)}

    # STT info
    stt_info = {
        "backend": voice_service.model_config.stt_backend,
        "profile": voice_service.model_config.stt_profile,
        "loaded": voice_service.stt_is_ready,
    }
    if voice_service.stt_backend is not None:
        stt_info["model"] = voice_service.stt_backend.model_name

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0",
        "device": device_info,
        "configuration": {
            "quantization_enabled": voice_service.model_config.use_quantization,
            "gpu_queue_enabled": voice_service.model_config.enable_gpu_queue,
            "sample_rate": voice_service.audio_config.sample_rate,
        },
        "models": {
            "stt": stt_info,
            "tts": {
                "loaded": voice_service.tts_provider is not None,
                "provider": voice_service.current_provider_name,
                "available_providers": voice_service.get_available_tts_providers(),
            },
        },
        "resources": {"gpu": gpu_status, "disk": disk_info},
    }


# ===========================
# STT Endpoints
# ===========================

@app.post("/api/voice/stt", response_model=STTResponse)
async def speech_to_text(
    file: UploadFile = File(...),
    word_timestamps: bool = Query(False, description="Return word-level timestamps"),
    vad_filter: bool = Query(True, description="Apply VAD filtering"),
    language: Optional[str] = Query(None, description="Force language code (e.g. 'en', 'fr')"),
):
    """
    Speech-to-Text transcription.

    Supports word-level timestamps and Voice Activity Detection.
    Uses faster-whisper backend for 2-4x speedup vs transformers.
    """
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

        # Reject oversized uploads up-front (via Content-Length) before reading
        # the whole body into memory; fall back to a post-read check when the
        # size is not advertised.
        if file.size is not None and file.size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB"
            )

        audio_bytes = await file.read()

        if len(audio_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB"
            )

        import soundfile as sf
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Language is passed per-call (never mutates shared backend state),
        # so concurrent requests can't leak languages into one another.
        result = await voice_service.transcribe_audio(
            audio_data,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            language=language,
        )

        return STTResponse(
            text=result["text"],
            language=result.get("language", "en"),
            language_probability=result.get("language_probability"),
            duration=result.get("duration"),
            time=result["time"],
            words=result.get("words"),
            segments=result.get("segments"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@app.get("/api/voice/stt/profiles")
async def get_stt_profiles():
    """List available STT profiles with model info and capabilities."""
    try:
        from voice.stt.faster_whisper_stt import STT_PROFILES
        profiles_info = {
            "fast": {
                "model": STT_PROFILES["fast"],
                "description": "Distil-Whisper: English only, ~300x RTFx, fastest",
                "languages": ["en"],
                "rtfx": "~300x",
                "use_case": "Real-time English transcription",
            },
            "default": {
                "model": STT_PROFILES["default"],
                "description": "Whisper Large V3 Turbo: 99+ languages, 216x RTFx",
                "languages": "99+ languages",
                "rtfx": "216x",
                "use_case": "Multilingual production (recommended)",
            },
            "accurate": {
                "model": STT_PROFILES["accurate"],
                "description": "Whisper Large V3: Highest accuracy, 60x RTFx",
                "languages": "99+ languages",
                "rtfx": "~60x",
                "use_case": "Maximum accuracy when speed is not critical",
            },
        }
        return {
            "current_profile": voice_service.model_config.stt_profile if voice_service else "unknown",
            "current_backend": voice_service.model_config.stt_backend if voice_service else "unknown",
            "profiles": profiles_info,
        }
    except ImportError:
        return {"error": "faster-whisper not installed", "profiles": {}}


@app.post("/api/voice/stt/switch")
async def switch_stt_profile(request: SwitchSTTProfileRequest):
    """
    Switch STT profile at runtime.

    - **fast**: Distil-Whisper (English only, ~300x RTFx)
    - **default**: Whisper Large V3 Turbo (multilingual, 216x RTFx)
    - **accurate**: Whisper Large V3 (multilingual, highest accuracy)
    """
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        await voice_service.switch_stt_profile(request.profile)
        return {
            "success": True,
            "profile": request.profile,
            "message": f"STT profile switched to '{request.profile}'",
        }
    except Exception as e:
        logger.error(f"STT switch error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


# ===========================
# TTS Endpoints
# ===========================

@app.post("/api/voice/tts")
async def text_to_speech(request: TTSRequest):
    """
    Text-to-Speech synthesis.

    Supports all providers: Kokoro, Chatterbox, F5-TTS, Qwen3-TTS, Orpheus, Dia.

    Provider-specific parameters:
    - **Chatterbox**: exaggeration, temperature, cfg_weight, audio_prompt_path
    - **F5-TTS**: ref_audio, ref_text
    - **Qwen3-TTS**: voice_description (VoiceDesign)
    - **Orpheus**: emotion (happy/sad/angry/whispering/laughing/sighing/crying/surprised)
    """
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        kwargs = {}
        try:
            if request.ref_audio:
                kwargs["ref_audio"] = validate_audio_path(request.ref_audio, "ref_audio")
            if request.audio_prompt_path:
                kwargs["audio_prompt_path"] = validate_audio_path(request.audio_prompt_path, "audio_prompt_path")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if request.ref_text:
            kwargs["ref_text"] = request.ref_text
        if request.speed != 1.0:
            kwargs["speed"] = request.speed
        if request.exaggeration != 0.5:
            kwargs["exaggeration"] = request.exaggeration
        if request.temperature != 0.8:
            kwargs["temperature"] = request.temperature
        if request.cfg_weight != 0.5:
            kwargs["cfg_weight"] = request.cfg_weight
        if request.voice_description:
            kwargs["voice_description"] = request.voice_description
        if request.emotion:
            kwargs["emotion"] = request.emotion

        audio_chunks = await voice_service.synthesize_speech(
            text=request.text,
            language=request.language,
            voice_id=request.voice_id,
            provider=request.provider,
            **kwargs,
        )

        audio_bytes = _encode_wav(audio_chunks)

        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "language": request.language,
            "provider": voice_service.current_provider_name,
        }

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=503, detail=_provider_unavailable_detail(request.provider, e))
    except Exception as e:
        logger.error(f"TTS Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@app.post("/api/voice/tts/design")
async def voice_design(request: VoiceDesignRequest):
    """
    VoiceDesign: Synthesize speech with a voice described in natural language.

    **Requires Qwen3-TTS provider** (automatically switched to if not active).

    Examples:
    - `"a calm British male with a deep voice"`
    - `"a cheerful young French woman speaking energetically"`
    - `"a professional news anchor, clear and neutral"`
    - `"an elderly Japanese man, warm and gentle"`
    """
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        kwargs = {
            "voice_description": request.voice_description,
        }
        if request.speed != 1.0:
            kwargs["speed"] = request.speed
        try:
            if request.ref_audio_path:
                kwargs["ref_audio_path"] = validate_audio_path(request.ref_audio_path, "ref_audio_path")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Auto-switch to Qwen3-TTS atomically with synthesis (handled in service)
        audio_chunks = await voice_service.synthesize_speech(
            text=request.text,
            language=request.language,
            provider="qwen3-tts",
            **kwargs,
        )

        audio_bytes = _encode_wav(audio_chunks)

        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "language": request.language,
            "provider": "qwen3-tts",
            "voice_description": request.voice_description,
        }

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=503, detail=_provider_unavailable_detail("qwen3-tts", e))
    except Exception as e:
        logger.error(f"VoiceDesign Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@app.post("/api/voice/tts/dialogue")
async def synthesize_dialogue(request: DialogueRequest):
    """
    Multi-speaker dialogue synthesis.

    **Requires Dia provider** (automatically switched to if not active).

    Dia natively generates:
    - Multi-speaker conversations ([S1]/[S2] format)
    - Nonverbal sounds: (laughs), (sighs), (breathes), (clears throat), (pauses)

    Example input:
    ```json
    {
      "turns": [
        {"speaker": "S1", "text": "Hey, how are you doing today?"},
        {"speaker": "S2", "text": "I'm great! (laughs) Just finished a big project."},
        {"speaker": "S1", "text": "That's wonderful! Congratulations!"}
      ]
    }
    ```
    """
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        turns = [{"speaker": t.speaker, "text": t.text} for t in request.turns]

        kwargs = {}
        try:
            if request.speaker_audio_s1:
                kwargs["speaker_audio_s1"] = validate_audio_path(request.speaker_audio_s1, "speaker_audio_s1")
            if request.speaker_audio_s2:
                kwargs["speaker_audio_s2"] = validate_audio_path(request.speaker_audio_s2, "speaker_audio_s2")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Auto-switch to Dia atomically with synthesis (handled in service)
        audio_chunks = await voice_service.synthesize_dialogue(
            turns=turns,
            language=request.language,
            provider="dia",
            **kwargs,
        )

        audio_bytes = _encode_wav(audio_chunks)

        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "language": request.language,
            "provider": "dia",
            "speakers": list({t.speaker for t in request.turns}),
            "total_turns": len(request.turns),
        }

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=503, detail=_provider_unavailable_detail("dia", e))
    except Exception as e:
        logger.error(f"Dialogue Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@app.get("/api/voice/tts/providers")
async def get_tts_providers():
    """Get available TTS providers with metadata."""
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        from voice.providers import get_provider_info
        providers = voice_service.get_available_tts_providers()
        voices = await voice_service.list_voices()
        provider_info = get_provider_info()

        return {
            "providers": providers,
            "current": voice_service.current_provider_name,
            "provider_metadata": provider_info,
            "available_voices": voices,
        }
    except Exception as e:
        logger.error(f"Error getting providers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@app.post("/api/voice/tts/switch")
async def switch_tts_provider(request: SwitchProviderRequest):
    """Switch TTS provider."""
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        kwargs = {}
        if request.model_variant:
            kwargs["model_variant"] = request.model_variant

        await voice_service.switch_tts_provider(request.provider, **kwargs)

        return {
            "success": True,
            "provider": voice_service.current_provider_name,
            "message": f"Switched to {request.provider}",
        }
    except ImportError as e:
        raise HTTPException(status_code=503, detail=_provider_unavailable_detail(request.provider, e))
    except Exception as e:
        logger.error(f"Error switching provider: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@app.get("/api/voice/tts/voices")
async def get_voices(language: Optional[str] = Query(None)):
    """Get available voices for current TTS provider."""
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        voices = await voice_service.list_voices(language)
        return {"voices": voices, "provider": voice_service.current_provider_name}
    except Exception as e:
        logger.error(f"Error getting voices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


# ===========================
# Metrics & Stats
# ===========================

@app.get("/api/voice/stats")
async def get_latency_stats(category: Optional[str] = Query(None, description="Filter by 'tts' or 'stt'")):
    """
    Get latency statistics (p50/p95/p99) per provider.

    Returns rolling window stats from the last 1000 requests per provider.
    """
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    stats = voice_service.get_latency_stats(category=category)
    return {
        "stats": stats,
        "current_tts_provider": voice_service.current_provider_name,
        "current_stt_backend": voice_service.model_config.stt_backend,
        "current_stt_profile": voice_service.model_config.stt_profile,
    }


# ===========================
# GPU & Resource Management
# ===========================

@app.post("/api/voice/unload")
async def unload_models():
    """Unload TTS models from GPU memory."""
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        await voice_service.unload_models()
        return {"success": True, "message": "TTS models unloaded"}
    except Exception as e:
        logger.error(f"Error unloading models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@app.get("/api/voice/gpu/status")
async def get_gpu_status():
    """Get GPU queue status (if enabled)."""
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return voice_service.get_gpu_status()


# ===========================
# WebSocket Streaming
# ===========================

@app.websocket("/api/voice/stream")
async def voice_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming voice conversation.

    Message types (client → server):
    - `{"type": "stt", "audio": "<base64>", "word_timestamps": false, "language": null}`
    - `{"type": "tts", "text": "...", "language": "en", "voice_id": null, "emotion": null, "provider": null, "format": "pcm"}`
    - `{"type": "tts_design", "text": "...", "voice_description": "...", "format": "pcm"}`
    - `{"type": "stt_switch", "profile": "fast|default|accurate"}`
    - `{"type": "ping"}`

    Audio format ("format" field, TTS): "pcm" (default) streams raw 16-bit
    little-endian mono PCM as a binary frame preceded by a JSON header
    (low-overhead); "opus" streams Ogg/Opus binary frames (~5-10x smaller,
    falls back to PCM if the libsndfile build lacks Opus); "wav" returns legacy
    base64-encoded WAV chunks.

    Message types (server → client):
    - `{"type": "stt_result", "text": "...", "language": "...", "words": [...]}`
    - PCM: `{"type": "tts_chunk", "chunk_index": 0, "format": "pcm_s16le", "sample_rate": 24000, "channels": 1, "bytes": N}` followed by a binary frame
    - WAV: `{"type": "tts_chunk", "audio": "<base64>", "chunk_index": 0, "format": "wav"}`
    - `{"type": "tts_complete", "total_chunks": N}`
    - `{"type": "pong"}`
    - `{"type": "error", "message": "..."}`
    """
    if voice_service is None:
        await websocket.close(code=1011, reason="Service not initialized")
        return

    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "stt":
                audio_base64 = data.get("audio")
                if not audio_base64:
                    await websocket.send_json({"type": "error", "message": "No audio data"})
                    continue

                # Bound decoded size (base64 inflates ~4/3) to avoid memory DoS.
                max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
                if len(audio_base64) > max_bytes * 4 // 3 + 4:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Audio too large (max {settings.MAX_UPLOAD_SIZE_MB}MB)",
                    })
                    continue

                audio_bytes = base64.b64decode(audio_base64)
                import soundfile as sf
                audio_data, _ = sf.read(io.BytesIO(audio_bytes))

                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                result = await voice_service.transcribe_audio(
                    audio_data,
                    word_timestamps=data.get("word_timestamps", False),
                    vad_filter=data.get("vad_filter", True),
                    language=data.get("language"),
                )

                await websocket.send_json({
                    "type": "stt_result",
                    "text": result["text"],
                    "language": result.get("language", "en"),
                    "time": result["time"],
                    "words": result.get("words"),
                    "segments": result.get("segments"),
                })

            elif msg_type == "tts":
                text = data.get("text")
                if not text:
                    await websocket.send_json({"type": "error", "message": "No text provided"})
                    continue

                language = data.get("language", "en")
                voice_id = data.get("voice_id")
                fmt = data.get("format", "pcm")
                kwargs = {}
                if data.get("emotion"):
                    kwargs["emotion"] = data["emotion"]
                if data.get("voice_description"):
                    kwargs["voice_description"] = data["voice_description"]

                chunk_count = 0
                async for audio_chunk in voice_service.synthesize_speech_streaming(
                    text=text, language=language, voice_id=voice_id,
                    provider=data.get("provider"), **kwargs
                ):
                    await _ws_send_chunk(websocket, audio_chunk, chunk_count, fmt)
                    chunk_count += 1

                await websocket.send_json({"type": "tts_complete", "total_chunks": chunk_count})

            elif msg_type == "tts_design":
                text = data.get("text")
                voice_description = data.get("voice_description")
                if not text or not voice_description:
                    await websocket.send_json({"type": "error", "message": "text and voice_description required"})
                    continue

                fmt = data.get("format", "pcm")
                chunk_count = 0
                async for audio_chunk in voice_service.synthesize_speech_streaming(
                    text=text,
                    language=data.get("language", "en"),
                    provider="qwen3-tts",
                    voice_description=voice_description,
                ):
                    await _ws_send_chunk(websocket, audio_chunk, chunk_count, fmt)
                    chunk_count += 1

                await websocket.send_json({"type": "tts_complete", "total_chunks": chunk_count})

            elif msg_type == "stt_switch":
                profile = data.get("profile")
                if not profile:
                    await websocket.send_json({"type": "error", "message": "profile required"})
                    continue
                try:
                    await voice_service.switch_stt_profile(profile)
                    await websocket.send_json({
                        "type": "stt_switched",
                        "profile": profile,
                        "message": f"STT profile switched to '{profile}'",
                    })
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}. "
                               "Valid: stt, tts, tts_design, stt_switch, ping"
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Voice Backend API on {settings.HOST}:{settings.PORT}")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level=settings.LOG_LEVEL.lower())
