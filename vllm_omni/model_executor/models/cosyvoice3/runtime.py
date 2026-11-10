# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runtime toggles shared by the CosyVoice3 stages."""

import os
from contextlib import nullcontext

import torch

_FALSE_ENV_VALUES = ("0", "false", "False", "")


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in _FALSE_ENV_VALUES


def _positive_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}")
    return value


def cosyvoice3_batch_flow_enabled() -> bool:
    """Return whether cross-request Stage-1 flow batching is enabled."""
    return _env_flag("COSYVOICE3_BATCH_FLOW", "0")


def cosyvoice3_batch_flow_debug() -> bool:
    """Return whether Stage-1 batching diagnostics are enabled."""
    return _env_flag("COSYVOICE3_BATCH_FLOW_DEBUG", "0")


def cosyvoice3_batch_flow_profile(name: str):
    """Create a profiler scope only when batching diagnostics are enabled."""
    npu = getattr(torch, "npu", None)
    is_capturing = getattr(npu, "is_current_stream_capturing", None)
    capturing = False
    if callable(is_capturing):
        try:
            capturing = bool(is_capturing())
        except (RuntimeError, TypeError):
            capturing = False
    if cosyvoice3_batch_flow_debug() and not capturing:
        return torch.profiler.record_function(name)
    return nullcontext()


def cosyvoice3_stage1_npu_graph_enabled() -> bool:
    """Return whether the opt-in Stage-1 NPU graph adapter is enabled."""
    return _env_flag("COSYVOICE3_STAGE1_NPU_GRAPH", "0")


def cosyvoice3_stage1_npu_graph_max() -> int:
    """Return the validated per-region Stage-1 NPU graph cache limit."""
    return _positive_env_int("COSYVOICE3_STAGE1_NPU_GRAPH_MAX", 8)


def cosyvoice3_full_response_enabled() -> bool:
    """Opt-in Hopper full-response path; other devices retain the standard path."""
    return (
        _env_flag("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "0")
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
    )


def cosyvoice3_packed_streaming_enabled() -> bool:
    """Opt-in Hopper packed, chunk-causal Flow with request-owned GPU HiFT state."""
    return (
        _env_flag("COSYVOICE3_PACKED_STREAMING", "0")
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
    )


def cosyvoice3_packed_inference_enabled() -> bool:
    """Either optimized profile needs live conditioning outside CUDA graphs."""
    return cosyvoice3_full_response_enabled() or cosyvoice3_packed_streaming_enabled()


def cosyvoice3_standard_sampling(config: object) -> bool:
    """Select ordinary sampling; RAS remains the default checkpoint behavior.

    The optional HF override is shared by API stop configuration and the model
    sampler. Reject misspellings instead of silently selecting another policy.
    """
    mode = getattr(config, "cosyvoice3_sampling_mode", "ras")
    if mode not in ("ras", "standard"):
        raise ValueError("cosyvoice3_sampling_mode must be 'ras' or 'standard'")
    return mode == "standard"
