"""
GPU Model Queue Manager for Voice Backend
Manages GPU memory for TTS/STT models
Can optionally coordinate with Ollama LLMs if enabled
"""

import torch
import gc
import asyncio
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    type: str  # 'tts', 'stt', 'llm', 'embedding'
    vram_mb: int
    last_used: datetime
    priority: int  # Higher = keep in memory longer
    is_small: bool = False  # True for embedding models (can coexist with TTS)


class GPUQueueManager:
    """
    Manages GPU memory for voice backend
    Tracks TTS/STT models and optionally coordinates with Ollama
    """

    def __init__(
        self,
        max_vram_mb: int = 6000,  # Reserve 6GB for main model
        idle_timeout_seconds: int = 300,  # Unload after 5 minutes idle
        enable_ollama_coordination: bool = False,
        ollama_url: str = "http://localhost:11434",
    ):
        self.max_vram_mb = max_vram_mb
        self.idle_timeout_seconds = idle_timeout_seconds
        self.enable_ollama_coordination = enable_ollama_coordination
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.lock = asyncio.Lock()

        # Ollama API endpoint (optional)
        self.ollama_url = ollama_url

        # Embedding model patterns (small models that can coexist with TTS)
        self.embedding_patterns = [
            'embed', 'embedding', 'nomic', 'mxbai', 'bge', 'minilm',
            'all-minilm', 'qwen3-embedding', 'snowflake'
        ]

    def _is_embedding_model(self, model_name: str) -> bool:
        """Check if a model is an embedding model (small, can coexist with TTS)"""
        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in self.embedding_patterns)

    async def request_gpu_for_tts(self, model_name: str, estimated_vram_mb: int = 2000):
        """
        Request GPU memory for TTS model
        Will unload Ollama models if coordination is enabled
        """
        async with self.lock:
            print(f"[GPU-Queue] Request: TTS model '{model_name}' needs ~{estimated_vram_mb}MB")

            # Check if we need to free memory
            current_usage = self._get_current_vram_usage()
            required = estimated_vram_mb
            available = self.max_vram_mb - current_usage

            print(f"[GPU-Queue] Current VRAM: {current_usage}MB, Available: {available}MB, Required: {required}MB")

            if available < required:
                print(f"[GPU-Queue] Insufficient VRAM, need to free {required - available}MB")

                # Try to unload Ollama models if coordination is enabled
                if self.enable_ollama_coordination:
                    await self._unload_ollama_models()

                await asyncio.sleep(1)  # Wait for cleanup
                torch.cuda.empty_cache()
                gc.collect()

            # Register TTS model as loaded
            self.loaded_models[model_name] = ModelInfo(
                name=model_name,
                type='tts',
                vram_mb=estimated_vram_mb,
                last_used=datetime.now(),
                priority=2  # Medium priority
            )

            print(f"[GPU-Queue] ✅ GPU ready for TTS")
            return True

    async def request_gpu_for_stt(self, model_name: str, estimated_vram_mb: int = 1500):
        """
        Request GPU memory for STT model (Whisper)
        """
        async with self.lock:
            print(f"[GPU-Queue] Request: STT model '{model_name}' needs ~{estimated_vram_mb}MB")

            # Check if we need to free memory
            current_usage = self._get_current_vram_usage()
            required = estimated_vram_mb
            available = self.max_vram_mb - current_usage

            print(f"[GPU-Queue] Current VRAM: {current_usage}MB, Available: {available}MB, Required: {required}MB")

            if available < required:
                print(f"[GPU-Queue] Insufficient VRAM, need to free {required - available}MB")

                # Try to unload Ollama models if coordination is enabled
                if self.enable_ollama_coordination:
                    await self._unload_ollama_models()

                await asyncio.sleep(1)
                torch.cuda.empty_cache()
                gc.collect()

            # Register STT model as loaded
            self.loaded_models[model_name] = ModelInfo(
                name=model_name,
                type='stt',
                vram_mb=estimated_vram_mb,
                last_used=datetime.now(),
                priority=2
            )

            print(f"[GPU-Queue] ✅ GPU ready for STT")
            return True

    async def _unload_ollama_models(self):
        """
        Unload Ollama LLM models from GPU (keeps embedding models loaded)
        Only called if ollama coordination is enabled
        """
        if not self.enable_ollama_coordination:
            print("[GPU-Queue] Ollama coordination disabled, skipping")
            return

        print("[GPU-Queue] Unloading Ollama LLM models...")

        try:
            # Get list of loaded models
            response = requests.get(f"{self.ollama_url}/api/ps", timeout=5)
            if response.ok:
                data = response.json()
                models = data.get('models', [])

                unloaded_count = 0
                kept_count = 0

                for model in models:
                    model_name = model.get('name', '')
                    if model_name:
                        # Skip embedding models - they can coexist with TTS
                        if self._is_embedding_model(model_name):
                            print(f"[GPU-Queue] Keeping embedding model loaded: {model_name}")
                            kept_count += 1
                            continue

                        print(f"[GPU-Queue] Stopping LLM model: {model_name}")
                        # Unload model via Ollama API
                        try:
                            requests.post(
                                f"{self.ollama_url}/api/generate",
                                json={"model": model_name, "keep_alive": 0},
                                timeout=10
                            )
                            unloaded_count += 1
                        except Exception as e:
                            print(f"[GPU-Queue] Error stopping {model_name}: {e}")

                # Remove only LLM models from tracking (keep embeddings)
                self.loaded_models = {
                    k: v for k, v in self.loaded_models.items()
                    if v.type != 'llm' or v.is_small
                }

                print(f"[GPU-Queue] ✅ Unloaded {unloaded_count} LLM(s), kept {kept_count} embedding model(s)")
        except Exception as e:
            print(f"[GPU-Queue] Error unloading Ollama: {e}")
            print(f"[GPU-Queue] This is expected if Ollama is not running")

    async def unload_tts_models(self):
        """
        Unload TTS models from GPU
        Called internally by this backend when models need to be freed
        """
        print("[GPU-Queue] Unloading TTS models...")

        # Clear tracking
        self.loaded_models = {
            k: v for k, v in self.loaded_models.items()
            if v.type not in ('tts', 'stt')
        }

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        print("[GPU-Queue] ✅ TTS models unloaded")

    def _get_current_vram_usage(self) -> int:
        """Get current VRAM usage in MB across all processes"""
        try:
            import subprocess
            # Use nvidia-smi to get actual GPU memory usage across all processes
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                vram_used = int(result.stdout.strip())
                return vram_used
        except Exception as e:
            print(f"[GPU-Queue] Warning: Could not get GPU memory via nvidia-smi: {e}")

        # Fallback to torch (only shows current process)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            return int(allocated)
        return 0

    def mark_model_used(self, model_name: str):
        """Update last used timestamp for a model"""
        if model_name in self.loaded_models:
            self.loaded_models[model_name].last_used = datetime.now()

    async def cleanup_idle_models(self):
        """Background task to unload idle models"""
        while True:
            await asyncio.sleep(60)  # Check every minute

            async with self.lock:
                now = datetime.now()
                idle_threshold = timedelta(seconds=self.idle_timeout_seconds)

                for model_name, info in list(self.loaded_models.items()):
                    idle_time = now - info.last_used
                    if idle_time > idle_threshold:
                        print(f"[GPU-Queue] Model '{model_name}' idle for {idle_time.seconds}s, unloading...")

                        if info.type in ('tts', 'stt'):
                            await self.unload_tts_models()

    def get_status(self) -> Dict[str, Any]:
        """Get current GPU queue status"""
        vram_used = self._get_current_vram_usage()
        vram_available = self.max_vram_mb - vram_used

        return {
            "vram_total_mb": self.max_vram_mb,
            "vram_used_mb": vram_used,
            "vram_available_mb": vram_available,
            "ollama_coordination_enabled": self.enable_ollama_coordination,
            "loaded_models": [
                {
                    "name": info.name,
                    "type": info.type,
                    "vram_mb": info.vram_mb,
                    "idle_seconds": (datetime.now() - info.last_used).seconds,
                }
                for info in self.loaded_models.values()
            ]
        }


# Global instance
_gpu_queue_manager: Optional[GPUQueueManager] = None


def get_gpu_queue_manager(
    enable_ollama_coordination: bool = False,
    ollama_url: str = "http://localhost:11434",
) -> GPUQueueManager:
    """Get or create global GPU queue manager"""
    global _gpu_queue_manager
    if _gpu_queue_manager is None:
        _gpu_queue_manager = GPUQueueManager(
            enable_ollama_coordination=enable_ollama_coordination,
            ollama_url=ollama_url,
        )
    return _gpu_queue_manager
