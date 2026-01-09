"""
Unified Voice Service API
FastAPI server with Multi-TTS providers + GPU Queue Management

This is the consolidated version of all voice APIs.
Use this instead of voice_api.py, voice_api_v2.py, or voice_api_gpu_managed.py
"""

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
from pydantic import BaseModel, Field, validator
import numpy as np
import json
from typing import Optional, List
import base64
import io
import uuid
from datetime import datetime

from voice.api.voice_service_unified import (
    VoiceServiceUnified,
    AudioConfig,
    ModelConfig,
)
from config.settings import settings, ensure_directories

# Setup structured logging
try:
    from voice.utils.logger import setup_logger, get_logger

    # Configure logging
    setup_logger(
        log_file=settings.LOG_FILE,
        level=settings.LOG_LEVEL,
    )
    logger = get_logger(__name__)
except ImportError:
    # Fallback to print if loguru not available
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Unified Voice Service API",
    version="3.0.0",
    description="Multi-TTS provider voice service with GPU queue management",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize voice service
voice_service: Optional[VoiceServiceUnified] = None


# Request/Response models with validation
class TTSRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=5000, description="Text to synthesize"
    )
    language: str = Field(
        default="en",
        pattern=r"^[a-z]{2}(-[a-z]{2})?$",
        description="Language code (ISO 639-1)",
    )
    provider: Optional[str] = Field(
        None, description="TTS provider (kokoro, chatterbox, f5-tts)"
    )
    voice_id: Optional[str] = Field(None, max_length=100, description="Voice ID")
    # Provider-specific parameters
    ref_audio: Optional[str] = Field(
        None, description="Reference audio path for F5-TTS"
    )
    ref_text: Optional[str] = Field(
        None, max_length=1000, description="Reference text for F5-TTS"
    )
    speed: Optional[float] = Field(1.0, gt=0.5, lt=2.0, description="Speech speed")
    audio_prompt_path: Optional[str] = Field(
        None, description="Audio prompt path for Chatterbox"
    )
    exaggeration: Optional[float] = Field(
        0.5, ge=0.25, le=2.0, description="Chatterbox expressiveness"
    )
    temperature: Optional[float] = Field(
        0.8, ge=0.05, le=5.0, description="Chatterbox temperature"
    )
    cfg_weight: Optional[float] = Field(
        0.5, ge=0.2, le=1.0, description="Chatterbox CFG weight"
    )

    @validator("provider")
    def validate_provider(cls, v):
        if v is not None:
            allowed_providers = ["kokoro", "chatterbox", "f5-tts"]
            if v not in allowed_providers:
                raise ValueError(
                    f"Provider must be one of: {', '.join(allowed_providers)}"
                )
        return v


class STTResponse(BaseModel):
    text: str
    language: str
    time: float


class TTSResponse(BaseModel):
    audio_base64: str
    language: str
    time: float


class TTSProvidersResponse(BaseModel):
    providers: List[str]
    current: str
    available_voices: List[dict]


class SwitchProviderRequest(BaseModel):
    provider: str
    model_variant: Optional[str] = None  # For F5-TTS


@app.on_event("startup")
async def startup_event():
    """Initialize voice service on startup"""
    global voice_service
    logger.info("=" * 60)
    logger.info("🚀 Starting Voice Backend API")
    logger.info("📦 Version: 3.0.0")
    logger.info("🎤 Voice Service (TTS/STT)")
    logger.info("=" * 60)

    # Ensure directories exist
    ensure_directories()
    logger.debug("Required directories created/verified")

    audio_config = AudioConfig()
    model_config = ModelConfig(
        stt_model=settings.STT_MODEL,
        tts_provider=settings.TTS_DEFAULT_PROVIDER,
        enable_gpu_queue=settings.ENABLE_GPU_QUEUE,
    )

    logger.info(f"Audio Config: sample_rate={audio_config.sample_rate}")
    logger.info(
        f"Model Config: stt={model_config.stt_model}, tts={model_config.tts_provider}"
    )

    voice_service = VoiceServiceUnified(audio_config, model_config)
    await voice_service.initialize_models()

    logger.info("=" * 60)
    logger.info("✅ Voice Backend API ready!")
    logger.info(f"📍 Port: {settings.PORT}")
    logger.info(f"📍 TTS Provider: {voice_service.current_provider_name}")
    logger.info(
        f"🔄 GPU Queue: {'ENABLED' if voice_service.gpu_manager else 'DISABLED'}"
    )
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global voice_service
    if voice_service:
        await voice_service.cleanup()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Unified Voice Service API",
        "version": "3.0.0",
        "features": [
            "Multi-TTS providers (Kokoro, Chatterbox, F5-TTS)",
            "GPU Queue Management",
            "Voice selection",
            "Zero-shot cloning (F5-TTS)",
            "Streaming support",
        ],
        "endpoints": {
            "health": "GET /api/voice/health",
            "stt": "POST /api/voice/stt",
            "tts": "POST /api/voice/tts",
            "providers": "GET /api/voice/tts/providers",
            "switch": "POST /api/voice/tts/switch",
            "voices": "GET /api/voice/tts/voices",
            "unload": "POST /api/voice/unload",
            "gpu_status": "GET /api/voice/gpu/status",
            "stream": "WS /api/voice/stream",
        },
    }


@app.get("/api/voice/health")
async def health_check():
    """Detailed health check with comprehensive system information"""
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Get detailed device info if available
    device_info = {}
    try:
        from voice.utils.device import get_detailed_device_info

        device_info = get_detailed_device_info()
    except Exception as e:
        logger.warning(f"Could not get detailed device info: {e}")
        device_info = {"device": voice_service.model_config.device}

    # GPU status
    gpu_status = "N/A"
    if voice_service.gpu_manager:
        try:
            gpu_status = voice_service.get_gpu_status()
        except Exception as e:
            logger.warning(f"Could not get GPU status: {e}")
            gpu_status = {"error": str(e)}

    # Check disk space for cache
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
        logger.warning(f"Could not get disk info: {e}")
        disk_info = {"error": str(e)}

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "device": device_info,
        "configuration": {
            "quantization_enabled": voice_service.model_config.use_quantization,
            "gpu_queue_enabled": voice_service.model_config.enable_gpu_queue,
            "sample_rate": voice_service.audio_config.sample_rate,
        },
        "models": {
            "stt": {
                "loaded": voice_service.stt_pipeline is not None,
                "model": voice_service.model_config.stt_model,
            },
            "tts": {
                "loaded": voice_service.tts_provider is not None,
                "provider": voice_service.current_provider_name,
                "available_providers": voice_service.get_available_tts_providers()
                if hasattr(voice_service, "get_available_tts_providers")
                else [],
            },
        },
        "resources": {"gpu": gpu_status, "disk": disk_info},
    }


@app.post("/api/voice/stt", response_model=STTResponse)
async def speech_to_text(file: UploadFile = File(...)):
    """Speech-to-Text endpoint"""
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Read audio file
        audio_bytes = await file.read()

        # Convert to numpy array
        import soundfile as sf

        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Ensure float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Transcribe
        result = await voice_service.transcribe_audio(audio_data)

        return STTResponse(
            text=result["text"], language=result["language"], time=result["time"]
        )

    except Exception as e:
        logger.error(f"STT Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/tts")
async def text_to_speech(request: TTSRequest):
    """Text-to-Speech endpoint"""
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Switch provider if requested
        if request.provider and request.provider != voice_service.current_provider_name:
            logger.info(f"Switching TTS provider to: {request.provider}")
            await voice_service.switch_tts_provider(request.provider)

        # Prepare kwargs for provider-specific parameters
        kwargs = {}
        if request.ref_audio:
            kwargs["ref_audio"] = request.ref_audio
        if request.ref_text:
            kwargs["ref_text"] = request.ref_text
        if request.speed != 1.0:
            kwargs["speed"] = request.speed
        if request.audio_prompt_path:
            kwargs["audio_prompt_path"] = request.audio_prompt_path
        if request.exaggeration != 0.5:
            kwargs["exaggeration"] = request.exaggeration
        if request.temperature != 0.8:
            kwargs["temperature"] = request.temperature
        if request.cfg_weight != 0.5:
            kwargs["cfg_weight"] = request.cfg_weight

        # Synthesize
        audio_chunks = await voice_service.synthesize_speech(
            text=request.text,
            language=request.language,
            voice_id=request.voice_id,
            **kwargs,
        )

        # Concatenate chunks
        audio_data = np.concatenate(audio_chunks)

        # Convert to bytes
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(
            buffer, audio_data, voice_service.audio_config.sample_rate, format="WAV"
        )
        buffer.seek(0)
        audio_bytes = buffer.read()

        # Return as base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "audio_base64": audio_base64,
            "language": request.language,
            "provider": voice_service.current_provider_name,
        }

    except Exception as e:
        logger.error(f"TTS Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voice/tts/providers", response_model=TTSProvidersResponse)
async def get_tts_providers():
    """Get available TTS providers"""
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        providers = voice_service.get_available_tts_providers()
        current = voice_service.current_provider_name
        voices = await voice_service.list_voices()

        return TTSProvidersResponse(
            providers=providers, current=current, available_voices=voices
        )

    except Exception as e:
        logger.error(f"Error getting providers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/tts/switch")
async def switch_tts_provider(request: SwitchProviderRequest):
    """Switch TTS provider"""
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

    except Exception as e:
        logger.error(f"Error switching provider: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voice/tts/voices")
async def get_voices(language: Optional[str] = Query(None)):
    """Get available voices"""
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        voices = await voice_service.list_voices(language)
        return {"voices": voices}

    except Exception as e:
        logger.error(f"Error getting voices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/unload")
async def unload_models():
    """Unload TTS models to free GPU memory"""
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        await voice_service.unload_models()
        return {"success": True, "message": "TTS models unloaded"}

    except Exception as e:
        logger.error(f"Error unloading models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voice/gpu/status")
async def get_gpu_status():
    """Get GPU queue status (if enabled)"""
    if voice_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return voice_service.get_gpu_status()


@app.websocket("/api/voice/stream")
async def voice_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming voice conversation"""
    if voice_service is None:
        await websocket.close(code=1011, reason="Service not initialized")
        return

    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "stt":
                # Speech-to-Text
                audio_base64 = data.get("audio")
                if not audio_base64:
                    await websocket.send_json(
                        {"type": "error", "message": "No audio data"}
                    )
                    continue

                # Decode audio
                audio_bytes = base64.b64decode(audio_base64)
                import soundfile as sf

                audio_data, _ = sf.read(io.BytesIO(audio_bytes))

                # Ensure mono and float32
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Transcribe
                result = await voice_service.transcribe_audio(audio_data)

                await websocket.send_json(
                    {
                        "type": "stt_result",
                        "text": result["text"],
                        "language": result["language"],
                        "time": result["time"],
                    }
                )

            elif msg_type == "tts":
                # Text-to-Speech (streaming)
                text = data.get("text")
                language = data.get("language", "en")
                voice_id = data.get("voice_id")

                if not text:
                    await websocket.send_json(
                        {"type": "error", "message": "No text provided"}
                    )
                    continue

                # Stream audio chunks
                chunk_count = 0
                async for audio_chunk in voice_service.synthesize_speech_streaming(
                    text=text, language=language, voice_id=voice_id
                ):
                    # Convert to bytes
                    import soundfile as sf

                    buffer = io.BytesIO()
                    sf.write(
                        buffer,
                        audio_chunk,
                        voice_service.audio_config.sample_rate,
                        format="WAV",
                    )
                    buffer.seek(0)
                    audio_bytes = buffer.read()

                    # Send chunk
                    await websocket.send_json(
                        {
                            "type": "tts_chunk",
                            "audio": base64.b64encode(audio_bytes).decode("utf-8"),
                            "chunk_index": chunk_count,
                        }
                    )
                    chunk_count += 1

                # Send completion
                await websocket.send_json(
                    {"type": "tts_complete", "total_chunks": chunk_count}
                )

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown message type: {msg_type}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception as send_error:
            # WebSocket already closed, cannot send error message
            logger.debug(f"Could not send error message to client: {send_error}")


if __name__ == "__main__":
    import uvicorn

    host = settings.HOST
    port = settings.PORT

    logger.info(f"🚀 Starting Voice Backend API on {host}:{port}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port, log_level=settings.LOG_LEVEL.lower())
