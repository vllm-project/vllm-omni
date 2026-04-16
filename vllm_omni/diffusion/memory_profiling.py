# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Memory profiling helpers for diffusion pipeline logging.

Provides utilities to capture and format CUDA memory snapshots, and to
determine whether memory profiling is enabled (via the
``VLLM_OMNI_DIFFUSION_LOG_MEMORY`` environment variable or DEBUG log level).
"""

from __future__ import annotations

import logging
import os
from typing import Any

__all__ = [
    "get_memory_log_env_var",
    "is_memory_profiling_enabled",
    "capture_cuda_memory_snapshot",
    "format_cuda_memory_snapshot",
]

_MEMORY_LOG_ENV_VAR = "VLLM_OMNI_DIFFUSION_LOG_MEMORY"
_MEMORY_LOGGER_NAME = "vllm_omni.core.sched.omni_generation_scheduler"


def get_memory_log_env_var() -> str:
    """Return the environment variable name that controls memory profiling."""
    return _MEMORY_LOG_ENV_VAR


def is_memory_profiling_enabled() -> bool:
    """Return True when memory profiling is active.

    Profiling is enabled when either:
    - The ``VLLM_OMNI_DIFFUSION_LOG_MEMORY`` environment variable is set to a
      truthy value (``1``, ``true``, ``yes``, case-insensitive).
    - The relevant logger has DEBUG log level.
    """
    val = os.environ.get(_MEMORY_LOG_ENV_VAR, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    logger = logging.getLogger(_MEMORY_LOGGER_NAME)
    return logger.isEnabledFor(logging.DEBUG)


def capture_cuda_memory_snapshot() -> dict[str, Any] | None:
    """Capture a snapshot of current CUDA memory usage.

    Returns:
        A dictionary with keys:
        - ``device`` (int): CUDA device ordinal.
        - ``allocated_bytes`` (int): Current allocated memory in bytes.
        - ``reserved_bytes`` (int): Current reserved memory in bytes.
        - ``max_allocated_bytes`` (int): Peak allocated memory in bytes.
        - ``max_reserved_bytes`` (int): Peak reserved memory in bytes.

        Returns ``None`` when CUDA is not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
        return {
            "device": device,
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    except Exception:
        return None


def format_cuda_memory_snapshot(snapshot: dict[str, Any] | None) -> str:
    """Format a CUDA memory snapshot into a human-readable string.

    Args:
        snapshot: A snapshot as returned by :func:`capture_cuda_memory_snapshot`,
            or ``None``.

    Returns:
        A string such as
        ``"cuda:0 allocated=1.50GiB reserved=2.00GiB max_allocated=3.00GiB max_reserved=4.00GiB"``
        or ``"cuda=unavailable"`` when ``snapshot`` is ``None``.
    """
    if snapshot is None:
        return "cuda=unavailable"
    GiB = 1024**3
    return (
        f"cuda:{snapshot['device']} "
        f"allocated={snapshot['allocated_bytes'] / GiB:.2f}GiB "
        f"reserved={snapshot['reserved_bytes'] / GiB:.2f}GiB "
        f"max_allocated={snapshot['max_allocated_bytes'] / GiB:.2f}GiB "
        f"max_reserved={snapshot['max_reserved_bytes'] / GiB:.2f}GiB"
    )
