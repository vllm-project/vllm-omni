# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Operator limits and colour options owned by SeedVR2."""

import os
import tempfile
from collections.abc import Callable
from typing import Any

COLOR_CORRECTION_METHODS = ("lab", "wavelet", "adain", "none")
DEFAULT_COLOR_CORRECTION_METHOD = "lab"


def _positive_int(name: str, default: int | None) -> int | None:
    """Read an operator-tuned limit, rejecting values that would disable it."""
    value = os.environ.get(name)
    if value is None:
        return default
    if not value.strip().isdigit() or not int(value):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


environment_variables: dict[str, Callable[[], Any]] = {
    # ================== SeedVR2 Restoration Env Vars ==================
    # Admission budgets for the sharded serving profile. The defaults are
    # calibrated for the smallest qualified device, so larger accelerators raise
    # them here; setting them beyond what the device holds surfaces as a CUDA
    # OOM during the forward pass. See docs/models/seedvr2.md.
    "VLLM_OMNI_SEEDVR2_MAX_FRAMES": lambda: _positive_int("VLLM_OMNI_SEEDVR2_MAX_FRAMES", 257),
    "VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS": lambda: _positive_int(
        "VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS", 2560 * 1472
    ),
    # Unset, the clip budget scales with device memory from the calibrated
    # 32 GB value; see seedvr2.pipeline_seedvr2.sharded_budget.
    "VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS": lambda: _positive_int("VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS", None),
    # Bounds for the long-video restoration route.
    "VLLM_OMNI_SEEDVR2_LONG_MAX_UPLOAD_BYTES": lambda: _positive_int(
        "VLLM_OMNI_SEEDVR2_LONG_MAX_UPLOAD_BYTES", 8 * 1024**3
    ),
    # Longest model window. The clip budget usually binds first; this caps it
    # at 30 latent frames, the DiT's largest temporal attention window.
    "VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW": lambda: _positive_int("VLLM_OMNI_SEEDVR2_LONG_MAX_WINDOW", 121),
    "VLLM_OMNI_SEEDVR2_LONG_JOB_TTL_SECONDS": lambda: _positive_int("VLLM_OMNI_SEEDVR2_LONG_JOB_TTL_SECONDS", 3600),
    "VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR": lambda: os.environ.get(
        "VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR", tempfile.gettempdir()
    ),
}


def __getattr__(name):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
