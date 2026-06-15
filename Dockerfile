# ==============================================================================
# RAG Voice Backend - Production-Ready Multi-Stage Dockerfile
# Supports both GPU (NVIDIA CUDA) and CPU deployments
# ==============================================================================
# Resolves gradio dependency conflict by installing F5-TTS without gradio
# ==============================================================================

ARG DEVICE_TYPE=gpu
ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.8.1

# ==============================================================================
# GPU Builder Stage
# ==============================================================================
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder-gpu

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION

# Install system dependencies
# Ubuntu 24.04 comes with Python 3.12, use system python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    ca-certificates \
    libsndfile1-dev \
    portaudio19-dev \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment FIRST (before upgrading pip - PEP 668)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in venv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /tmp
COPY requirements.txt ./

# Install PyTorch with CUDA 12.8 support (required for Blackwell RTX 50xx)
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Install big dependencies first to simplify pip resolver
RUN pip install --no-cache-dir \
    numpy>=1.26.0,\<2.0.0 \
    scipy>=1.11.4 \
    transformers>=4.51.0

# Install main application dependencies with legacy resolver (Python 3.12 compatibility)
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# Install F5-TTS WITHOUT gradio to avoid version conflict
# F5-TTS requires gradio>=6.0.0 but chatterbox needs 5.44.1
# We don't need gradio for API-only deployment
RUN pip install --no-cache-dir --no-deps f5-tts>=1.1.15

# Install F5-TTS core dependencies (excluding gradio)
# Based on f5-tts requirements but skipping gradio
RUN pip install --no-cache-dir \
    cached-path>=1.6.2 \
    jiwer>=3.0.5 \
    jieba>=0.42.1 \
    vocos>=0.1.0 \
    pydub>=0.25.1 \
    cn2an>=0.5.22 \
    inflect>=7.4.0 \
    g2p-en>=2.1.0

# Install spaCy English model for Kokoro (required at runtime)
RUN python -m spacy download en_core_web_sm

# Clean up
RUN find /opt/venv -type d -name __pycache__ -print0 | xargs -0 rm -rf 2>/dev/null || true && \
    find /opt/venv -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

# ==============================================================================
# CPU Builder Stage
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder-cpu

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libsndfile1-dev \
    portaudio19-dev \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment FIRST (before upgrading pip - PEP 668)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in venv
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /tmp
COPY requirements.txt ./

# Install PyTorch CPU-only version FIRST (largest dependency, cache separately)
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install big dependencies first to simplify pip resolver
RUN pip install --no-cache-dir \
    numpy>=1.26.0,\<2.0.0 \
    scipy>=1.11.4 \
    transformers>=4.51.0

# Install main application dependencies with legacy resolver (Python 3.12 compatibility)
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

# Install F5-TTS WITHOUT gradio to avoid version conflict
RUN pip install --no-cache-dir --no-deps f5-tts>=1.1.15

# Install F5-TTS core dependencies (excluding gradio)
RUN pip install --no-cache-dir \
    cached-path>=1.6.2 \
    jiwer>=3.0.5 \
    jieba>=0.42.1 \
    vocos>=0.1.0 \
    pydub>=0.25.1 \
    cn2an>=0.5.22 \
    inflect>=7.4.0 \
    g2p-en>=2.1.0

# Install spaCy English model for Kokoro (required at runtime)
RUN python -m spacy download en_core_web_sm

# Clean up
RUN find /opt/venv -type d -name __pycache__ -print0 | xargs -0 rm -rf 2>/dev/null || true && \
    find /opt/venv -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

# ==============================================================================
# GPU Runtime Stage
# ==============================================================================
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime-gpu

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION

LABEL maintainer="RAG Voice Team"
LABEL description="RAG Voice Backend - TTS/STT Service with GPU support"
LABEL version="3.0.0"

# Install runtime dependencies only
# Ubuntu 24.04 comes with Python 3.12, use system python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    libgomp1 \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    espeak-ng \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder-gpu /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# NVIDIA GPU Configuration
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PyTorch GPU optimizations
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDNN_BENCHMARK=1

# ==============================================================================
# CPU Runtime Stage
# ==============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime-cpu

ENV DEBIAN_FRONTEND=noninteractive

LABEL maintainer="RAG Voice Team"
LABEL description="RAG Voice Backend - TTS/STT Service CPU-only"
LABEL version="3.0.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    espeak-ng \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder-cpu /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU Performance tuning
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV TORCH_NUM_THREADS=4

# ==============================================================================
# Final Stage - Common for both CPU and GPU
# ==============================================================================
FROM runtime-${DEVICE_TYPE} AS final

# Security: Create non-root user (let system assign UID to avoid conflicts)
RUN groupadd -r appuser && \
    useradd -r -g appuser -m -s /sbin/nologin appuser

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser voice/ /app/voice/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser entrypoint.sh /app/entrypoint.sh

# Create application directories with correct permissions AFTER copying files
RUN mkdir -p /app/models /app/data /app/logs /app/voice_cache && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Application configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Model cache directories
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch

# Voice backend specific
ENV DATA_DIR=/app/data
ENV VOICE_CACHE_DIR=/app/voice_cache
ENV LOG_DIR=/app/logs
ENV LOG_FILE=/app/logs/voice-backend.log

# Default configuration
ENV PORT=8002
ENV HOST=0.0.0.0
ENV LOG_LEVEL=INFO
ENV TTS_DEFAULT_PROVIDER=chatterbox
ENV TTS_DEVICE=auto

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/api/voice/health || exit 1

# Make entrypoint executable and set it
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["python", "-u", "voice/api/main.py"]
