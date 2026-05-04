# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from typing import Any

import torch

_MEMORY_LOG_ENV = "VLLM_OMNI_DIFFUSION_LOG_MEMORY"


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _is_diffusion_debug_logging_enabled() -> bool:
    logger = logging.getLogger("vllm_omni.core.sched.omni_generation_scheduler")
    return logger.isEnabledFor(logging.DEBUG)


def is_memory_profiling_enabled() -> bool:
    return _is_truthy(os.environ.get(_MEMORY_LOG_ENV)) or _is_diffusion_debug_logging_enabled()


def get_memory_log_env_var() -> str:
    return _MEMORY_LOG_ENV


def _bytes_to_gib(value: int) -> float:
    return value / float(1024**3)


def _device_index(device: int | str | torch.device | None) -> int:
    """Normalize *device* to a torch.device and return its integer index."""
    if device is None:
        device = torch.cuda.current_device()
    # Build a torch.device and extract its index (avoids pyright narrowing issues
    # with torch.device("cuda", int) → int | torch.device)
    d = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    idx = d.index
    return idx if idx is not None else torch.cuda.current_device()


def capture_cuda_memory_snapshot(device: int | str | torch.device | None = None) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        return None

    device_index = _device_index(device)

    return {
        "device": device_index,
        "allocated_bytes": torch.cuda.memory_allocated(device_index),
        "reserved_bytes": torch.cuda.memory_reserved(device_index),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
    }


def format_cuda_memory_snapshot(snapshot: dict[str, Any] | None) -> str:
    if snapshot is None:
        return "cuda=unavailable"

    return (
        f"cuda:{snapshot['device']} "
        f"allocated={_bytes_to_gib(int(snapshot['allocated_bytes'])):.2f}GiB "
        f"reserved={_bytes_to_gib(int(snapshot['reserved_bytes'])):.2f}GiB "
        f"max_allocated={_bytes_to_gib(int(snapshot['max_allocated_bytes'])):.2f}GiB "
        f"max_reserved={_bytes_to_gib(int(snapshot['max_reserved_bytes'])):.2f}GiB"
    )


def reset_cuda_peak_memory_stats(device: int | str | torch.device | None = None) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats(_device_index(device))
