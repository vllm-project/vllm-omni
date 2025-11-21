from __future__ import annotations

from typing import Callable

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


def get_device_control_env_var() -> str:
    """Return the environment variable name for device visibility control."""
    if hasattr(current_platform, "device_control_env_var"):
        env_var = getattr(current_platform, "device_control_env_var", None)
        if isinstance(env_var, str) and env_var:
            return env_var
    
    device_type = detect_device_type()
    if device_type == "npu":
        return "ASCEND_RT_VISIBLE_DEVICES"
    return "CUDA_VISIBLE_DEVICES" # fallback