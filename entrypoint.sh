#!/bin/bash
set -e

echo "========================================"
echo "RAG Voice Backend v3.0.0"
echo "========================================"
echo "Device Type: ${DEVICE_TYPE:-auto}"
echo "Python Version: $(python --version)"
echo ""

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
echo "========================================"
echo ""

# Execute main command
exec "$@"
