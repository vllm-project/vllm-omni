from __future__ import annotations

import torch
from vllm.platforms import current_platform


def detect_device_type() -> str:
    device_type = getattr(current_platform, "device_type", None)
    if isinstance(device_type, str) and device_type:
        return device_type.lower()
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
        return "npu"
    return "cpu"


def is_npu() -> bool:
    return detect_device_type() == "npu"


def is_rocm() -> bool:
    return current_platform.is_rocm()


def get_device_control_env_var() -> str:
    """Return the environment variable name for device visibility control."""
    if hasattr(current_platform, "device_control_env_var"):
        env_var = getattr(current_platform, "device_control_env_var", None)
        if isinstance(env_var, str) and env_var:
            return env_var

    device_type = detect_device_type()
    if device_type == "npu":
        return "ASCEND_RT_VISIBLE_DEVICES"
    return "CUDA_VISIBLE_DEVICES"  # fallback


def get_diffusion_worker_class() -> type:
    """Get the appropriate diffusion WorkerProc class based on current device type.

    Returns:
        The WorkerProc class for the detected device type (either NPUWorkerProc or WorkerProc).

    Raises:
        ImportError: If the worker module for the detected device is not available.
    """
    device_type = detect_device_type()

    if device_type == "npu":
        from vllm_omni.diffusion.worker.npu.npu_worker import NPUWorkerProc

        return NPUWorkerProc
    else:
        # Default to GPU worker for cuda and other devices
        from vllm_omni.diffusion.worker.gpu_diffusion_worker import WorkerProc

        return WorkerProc
