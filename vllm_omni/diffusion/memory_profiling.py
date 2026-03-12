# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch

_MEMORY_LOG_ENV = "VLLM_OMNI_DIFFUSION_LOG_MEMORY"


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def is_memory_profiling_enabled() -> bool:
    return _is_truthy(os.environ.get(_MEMORY_LOG_ENV))


def get_memory_log_env_var() -> str:
    return _MEMORY_LOG_ENV


def _bytes_to_gib(value: int) -> float:
    return value / float(1024**3)


def capture_cuda_memory_snapshot(device: int | str | torch.device | None = None) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        return None

    if device is None:
        device = torch.cuda.current_device()
    torch_device = torch.device("cuda", device) if not isinstance(device, torch.device) else device
    device_index = torch_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

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

    if device is None:
        device = torch.cuda.current_device()
    torch_device = torch.device("cuda", device) if not isinstance(device, torch.device) else device
    device_index = torch_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device_index)
