# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runtime toggles shared by the CosyVoice3 stages."""

import os
from contextlib import nullcontext
from functools import cache

import torch

_FALSE_ENV_VALUES = ("0", "false", "False", "")


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in _FALSE_ENV_VALUES


def cosyvoice3_batch_flow_enabled() -> bool:
    """Return whether cross-request Stage-1 flow batching is enabled."""
    return _env_flag("COSYVOICE3_BATCH_FLOW", "0")


def cosyvoice3_batch_flow_debug() -> bool:
    """Return whether Stage-1 batching diagnostics are enabled."""
    return _env_flag("COSYVOICE3_BATCH_FLOW_DEBUG", "0")


def cosyvoice3_batch_flow_profile(name: str):
    """Create a profiler scope only when batching diagnostics are enabled."""
    if cosyvoice3_batch_flow_debug():
        return torch.profiler.record_function(name)
    return nullcontext()


@cache
def _fa_version_supported(version: int) -> bool:
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported
    except ImportError:
        # Builds without vLLM's CUDA FlashAttention extensions, such as ROCm.
        return False
    return is_fa_version_supported(version)


def cosyvoice3_packed_fa_version() -> int | None:
    """FlashAttention version for packed Flow: FA3 on Hopper, else FA2 (sm80+).

    COSYVOICE3_PACKED_FA_VERSION pins one version, e.g. FA2 on Hopper to
    compare it against FA3. None means no packed backend runs here.
    """
    pinned = os.environ.get("COSYVOICE3_PACKED_FA_VERSION", "")
    if pinned not in ("", "2", "3"):
        raise ValueError("COSYVOICE3_PACKED_FA_VERSION must be 2 or 3")
    if not torch.cuda.is_available():
        return None
    candidates = (int(pinned),) if pinned else (3, 2)
    for version in candidates:
        if _fa_version_supported(version):
            return version
    return None


def cosyvoice3_full_response_enabled() -> bool:
    """Opt-in packed full-response path; devices without FA2/FA3 retain the standard path."""
    return _env_flag("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "0") and cosyvoice3_packed_fa_version() is not None


def cosyvoice3_packed_streaming_enabled() -> bool:
    """Opt-in packed, chunk-causal Flow with request-owned GPU HiFT state."""
    return _env_flag("COSYVOICE3_PACKED_STREAMING", "0") and cosyvoice3_packed_fa_version() is not None


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
