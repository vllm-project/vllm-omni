# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SVDQuant backend dispatch + hardware gate.

The SVDQuant on-disk format is canonical row-major NVFP4 (or
INT4-nibble), backend-agnostic. The runtime kernel backend is picked
at LinearMethod construction based on CUDA compute capability:

    SM_75–89, SM_120 → nunchaku (in this PR)
    SM_100/103       → FlashInfer (planned; not yet integrated)
    SM_90 (Hopper)   → unsupported (no validated kernel family)

Add a new backend by writing a `vllm_omni/quantization/svdquant_<name>.py`
module exposing `supports(cap, precision) -> bool`, `prepare_weights(
layer, precision) -> None`, and `apply(layer, x, bias) -> Tensor`, then
appending it to `_candidate_backends()` below.
"""

from __future__ import annotations

from types import ModuleType
from typing import Literal

from vllm.platforms import current_platform

SVDQuantPrecision = Literal["int4", "nvfp4"]


def _candidate_backends() -> list[ModuleType]:
    """Backends to try, in priority order.

    When FlashInfer lands, prepend it here so it takes precedence on
    its supported caps before falling back to nunchaku.
    """
    from . import svdquant_nunchaku

    return [svdquant_nunchaku]


def select_backend(precision: SVDQuantPrecision) -> ModuleType:
    """Return the first backend that supports (current platform, precision).

    Defense in depth — callers normally call `assert_svdquant_supported`
    first, which raises a more actionable error for unsupported
    platforms. This raises a generic error if you somehow skipped the
    gate.
    """
    cap = current_platform.get_device_capability() if current_platform.is_cuda() else None
    for backend in _candidate_backends():
        if backend.supports(cap, precision):
            return backend
    raise RuntimeError(
        f"No SVDQuant backend supports precision={precision!r} on "
        f"{current_platform.device_name!r}. Call "
        "assert_svdquant_supported() for a detailed diagnostic."
    )


def assert_svdquant_supported(precision: SVDQuantPrecision) -> None:
    """Raise a precise error if the active platform cannot run SVDQuant."""
    if not current_platform.is_cuda():
        raise RuntimeError(
            f"SVDQuant has no available backend on platform "
            f"{current_platform.device_name!r}. CUDA + a SVDQuant backend "
            "(nunchaku for consumer GPUs, FlashInfer for SM_100/103 — "
            "planned) is required."
        )

    cap = current_platform.get_device_capability()
    sm = f"SM_{cap.to_int()}" if cap is not None else "<unknown>"

    if current_platform.is_device_capability_family(90):
        raise RuntimeError(
            "SVDQuant W4A4 is not supported on Hopper (SM_90). Use a "
            "consumer GPU (SM_75–SM_89, SM_120) with nunchaku, or wait "
            "for the datacenter Blackwell (SM_100/103) path planned in "
            "FlashInfer."
        )

    if current_platform.is_device_capability_family(100):
        raise RuntimeError(
            f"SVDQuant on {sm} (B200/GB300) is not yet integrated; the datacenter path is planned in FlashInfer."
        )

    if not current_platform.has_device_capability((7, 5)):
        raise RuntimeError(f"Unsupported CUDA compute capability for SVDQuant: {sm}")

    # nvfp4 needs SM_100+ tensor units; pre-Blackwell consumer cards
    # (Turing/Ampere/Ada) cannot run it.
    if precision == "nvfp4" and not current_platform.has_device_capability(100):
        raise ValueError(f"NVFP4 SVDQuant requires SM_100+ or SM_120; got {sm}. Use precision='int4'.")

    # Backend-level missing-package check (current single backend = nunchaku).
    from . import svdquant_nunchaku

    if not svdquant_nunchaku.has_nunchaku_w4a4():
        # The PyPI `nunchaku` package is an unrelated Bayesian library;
        # SVDQuant kernels ship as GitHub release wheels only.
        raise ImportError(
            f"SVDQuant on {sm} requires nunchaku-ai's W4A4 wheels from "
            "https://github.com/nunchaku-ai/nunchaku/releases "
            "(not `pip install nunchaku`, which is a different project)."
        )


__all__ = ["SVDQuantPrecision", "assert_svdquant_supported", "select_backend"]
