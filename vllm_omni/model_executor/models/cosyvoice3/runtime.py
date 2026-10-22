# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runtime toggles shared by the CosyVoice3 stages."""

import importlib.util
import os
from contextlib import nullcontext
from functools import cache

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

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


@cache
def _flashinfer_available() -> bool:
    return importlib.util.find_spec("flashinfer") is not None


PACKED_ATTENTION_BACKENDS = ("fa3", "fa2", "flashinfer")


def _backend_supported(backend: str) -> bool:
    if backend == "flashinfer":
        return _flashinfer_available()
    return _fa_version_supported(int(backend[2]))


def cosyvoice3_packed_attention_backend() -> str | None:
    """Attention backend for packed Flow: FA3 on Hopper, else FA2 (sm80+).

    COSYVOICE3_PACKED_ATTENTION_BACKEND pins ``fa3``, ``fa2`` or ``flashinfer``
    (FlashInfer is only used when pinned). None means no packed backend runs
    here.
    """
    pinned = os.environ.get("COSYVOICE3_PACKED_ATTENTION_BACKEND", "")
    if pinned and pinned not in PACKED_ATTENTION_BACKENDS:
        raise ValueError("COSYVOICE3_PACKED_ATTENTION_BACKEND must be fa3, fa2 or flashinfer")
    if not torch.cuda.is_available():
        return None
    candidates = (pinned,) if pinned else ("fa3", "fa2")
    for backend in candidates:
        if _backend_supported(backend):
            return backend
    return None


def cosyvoice3_full_response_enabled() -> bool:
    """Opt-in packed full-response path; devices without a packed backend retain the standard path."""
    return _packed_profile_enabled("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS")


def cosyvoice3_packed_streaming_enabled() -> bool:
    """Opt-in packed, chunk-causal Flow with request-owned GPU HiFT state."""
    return _packed_profile_enabled("COSYVOICE3_PACKED_STREAMING")


def _packed_profile_enabled(environment_variable: str) -> bool:
    """Resolve an opted-in packed profile and explain unsupported fallbacks."""
    if not _env_flag(environment_variable, "0"):
        return False

    if cosyvoice3_packed_attention_backend() is not None:
        return True

    pinned = os.environ.get("COSYVOICE3_PACKED_ATTENTION_BACKEND", "")
    if pinned:
        logger.warning_once(
            "%s is enabled but pinned attention backend %s is unavailable on this GPU/build; "
            "CosyVoice3 will use the stock Flow path",
            environment_variable,
            pinned,
        )
    else:
        logger.warning_once(
            "%s is enabled but no supported CUDA attention backend is available; "
            "CosyVoice3 will use the stock Flow path",
            environment_variable,
        )
    return False


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
