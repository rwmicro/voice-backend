"""
Device configuration utilities
Centralized device detection and configuration
"""

import os
from typing import Tuple, Dict, Any


def get_device_config() -> Tuple[str, bool]:
    """
    Auto-detect optimal device configuration

    Returns:
        Tuple of (device, use_quantization)
        - device: 'cuda', 'mps', or 'cpu'
        - use_quantization: True if quantization should be enabled
    """
    try:
        import torch

        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = "cuda"

            # Get GPU memory
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # Use quantization for GPUs with < 8GB VRAM
            use_quantization = gpu_mem_gb < 8.0

            print(f"[Device] ✓ CUDA available")
            print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[Device] VRAM: {gpu_mem_gb:.1f} GB")
            print(
                f"[Device] Quantization: {'enabled' if use_quantization else 'disabled'}"
            )

            return device, use_quantization

        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            use_quantization = False  # MPS doesn't support quantization well

            print(f"[Device] ✓ Apple Silicon (MPS) available")
            print(f"[Device] Quantization: disabled (MPS doesn't support it)")

            return device, use_quantization

        # Fallback to CPU
        print(f"[Device] ℹ No GPU available, using CPU")
        print(f"[Device] Warning: CPU inference will be significantly slower")
        print(f"[Device] Consider using a GPU for better performance")

        return "cpu", False

    except ImportError:
        print(f"[Device] ✗ PyTorch not installed, defaulting to CPU")
        return "cpu", False


def get_detailed_device_info() -> Dict[str, Any]:
    """
    Get detailed device information

    Returns:
        Dictionary with device information
    """
    info = {
        "device": "cpu",
        "device_name": "CPU",
        "has_cuda": False,
        "has_mps": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_memory_gb": 0.0,
        "recommended_batch_size": 1,
        "recommended_fp16": False,
    }

    try:
        import torch

        # CUDA information
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["has_cuda"] = True
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )

            # Recommendations based on GPU memory
            if info["gpu_memory_gb"] >= 16:
                info["recommended_batch_size"] = 4
                info["recommended_fp16"] = True
            elif info["gpu_memory_gb"] >= 8:
                info["recommended_batch_size"] = 2
                info["recommended_fp16"] = True
            else:
                info["recommended_batch_size"] = 1
                info["recommended_fp16"] = True

        # MPS information
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["device"] = "mps"
            info["has_mps"] = True
            info["device_name"] = "Apple Silicon"
            info["recommended_batch_size"] = 2
            info["recommended_fp16"] = False

        # CPU fallback
        else:
            info["device"] = "cpu"
            info["device_name"] = "CPU"
            info["recommended_batch_size"] = 1
            info["recommended_fp16"] = False

    except ImportError:
        pass

    return info


def check_gpu_availability() -> bool:
    """
    Check if GPU is available

    Returns:
        True if GPU is available, False otherwise
    """
    try:
        import torch

        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        return False


def get_optimal_device() -> str:
    """
    Get the optimal device string for PyTorch

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    device, _ = get_device_config()
    return device
