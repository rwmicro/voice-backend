#!/bin/bash
set -e

echo "========================================"
echo "RAG Voice Backend v4.0.0"
echo "========================================"
echo "Device Type: ${DEVICE_TYPE:-auto}"
echo "Python Version: $(python --version)"
echo ""

# Create necessary directories if they don't exist
# This handles the case where bind mounts create root-owned dirs
echo "Ensuring directories exist with correct permissions..."

# Application directories (bind mounts — always writable by appuser)
mkdir -p "${DATA_DIR:-/app/data}" \
         "${VOICE_CACHE_DIR:-/app/voice_cache}" \
         "${LOG_DIR:-/app/logs}" 2>/dev/null || true

# Model cache subdirectories — live inside ./models bind mount (host-owned).
# Should be writable by appuser, but guard against edge cases.
if mkdir -p /app/models/transformers \
            /app/models/huggingface \
            /app/models/torch 2>/dev/null; then
    echo "✓ Model cache dirs ready (/app/models)"
else
    # Fallback: redirect model cache to /app/data which is always writable
    echo "Warning: Cannot write to /app/models — redirecting model cache to /app/data/models"
    export HF_HOME="${DATA_DIR:-/app/data}/models/huggingface"
    export TRANSFORMERS_CACHE="${DATA_DIR:-/app/data}/models/transformers"
    export TORCH_HOME="${DATA_DIR:-/app/data}/models/torch"
    mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" 2>/dev/null || true
    echo "✓ Model cache dirs ready (fallback: ${DATA_DIR:-/app/data}/models)"
fi

# Ensure writable permissions for current user (works with both root and non-root)
if [ -w "$(dirname "${DATA_DIR:-/app/data}")" ]; then
    chmod -R u+w "${DATA_DIR:-/app/data}" "${VOICE_CACHE_DIR:-/app/voice_cache}" "${LOG_DIR:-/app/logs}" 2>/dev/null || true
fi

echo "✓ Directories ready"

# Detect GPU availability
if python -c "import torch; print('GPU Available:', torch.cuda.is_available())" 2>/dev/null; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        export TTS_DEVICE=${TTS_DEVICE:-cuda}
        export STT_DEVICE=${STT_DEVICE:-cuda}
        echo "✓ CUDA Available"
        python -c "import torch; print('  Device:', torch.cuda.get_device_name(0))" 2>/dev/null || true
        python -c "import torch; print('  CUDA Version:', torch.version.cuda)" 2>/dev/null || true
    else
        export TTS_DEVICE=cpu
        export STT_DEVICE=cpu
        echo "✓ Running on CPU"
    fi
else
    export TTS_DEVICE=cpu
    export STT_DEVICE=cpu
    echo "✓ Running on CPU"
fi

echo ""
echo "Configuration:"
echo "  Port: ${PORT}"
echo "  TTS Provider: ${TTS_DEFAULT_PROVIDER}"
echo "  TTS Device: ${TTS_DEVICE}"
echo "  STT Device: ${STT_DEVICE}"
echo "  STT Backend: ${STT_BACKEND:-faster-whisper}"
echo "  STT Profile: ${STT_PROFILE:-default}"
echo "  HF_HOME: ${HF_HOME:-/app/models/huggingface}"
echo "========================================"
echo ""

# Execute main command
exec "$@"
