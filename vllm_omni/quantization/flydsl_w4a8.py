# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FlyDSL-backed W4A8 GEMM for diffusion transformers on ROCm gfx950.

Two variants, both MXFP4 weights with E8M0 per-32 scales and *dynamically*
quantized MXFP8 activations:

  plain  y = Q(x) @ Q(W).T + bias
  svd    y = Q(x) @ Q(W).T + (x @ L1.T) @ L2.T + bias

The low-rank up-projection of the ``svd`` variant is fused into the GEMM
epilogue, so both variants are a single kernel launch. ``d = x @ L1.T`` is
computed in bf16 by torch inside the custom op.

The kernel is reached through a single provider interface so the underlying
source can move without changes to the linear methods:

    quark  -> quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear
    flydsl -> upstream FlyDSL, once it ships the SVD epilogue in a released wheel

Selection is automatic (prefer ``flydsl`` when it exposes the epilogue, else
``quark``) and overridable with ``VLLM_OMNI_SVDQUANT_PROVIDER=quark|flydsl``.

Note on the Quark provider: it currently binds *private* Quark symbols
(``_gemm_flydsl_a8w4`` / ``_gemm_flydsl_svdquant``) because Quark exposes no
public alias for them yet. That coupling is deliberately confined to
``_load_quark_provider`` so promoting it to a public ``quark.flydsl`` namespace
is a one-line change here.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import NamedTuple

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# The kernel multiplies MXFP4 operands with CDNA4 scaled MFMA, which CDNA3
# (gfx942 / MI325X) does not have. Quark's kernel factory raises on non-gfx950
# too; this gate exists so the user sees an actionable message instead of an
# MLIR backend stack trace.
_SUPPORTED_ARCHS = ("gfx950",)

# The preshuffle GEMM does not mask N/K tiles that overrun the matrix, and the
# SVD epilogue additionally requires both dimensions to be tileable at >=256.
# Layers violating this must fall back to BF16 rather than emit garbage.
_SVD_DIM_MULTIPLE = 256

# Smallest tile_n the plain a8w4 path falls back to (Quark tries 128, 64, 32 in
# order). It is also the MXFP4 scale group size, so N must be a multiple of it
# regardless of tiling.
_DIM_MULTIPLE = 32

# The plain a8w4 kernel validates K itself and *raises* below this
# (``_validate_a8w4_inputs``: "requires K >= 256 and divisible by 256"), because
# the preshuffled B operand carries a padded scale layout of eight groups. Screen
# for it here so such layers fall back to BF16 instead of aborting generation.
_K_MULTIPLE = 256


def _arch() -> str:
    index = torch.accelerator.current_device_index()
    return torch.cuda.get_device_properties(index).gcnArchName


@functools.lru_cache(maxsize=1)
def supports() -> tuple[bool, str]:
    """Return ``(usable, reason)``.

    A missing provider is a capability answer, not an error. An unknown value of
    ``VLLM_OMNI_SVDQUANT_PROVIDER`` is user misconfiguration and propagates as
    ``ValueError``.
    """
    from vllm_omni.platforms import current_omni_platform

    if not current_omni_platform.is_rocm():
        return False, "FlyDSL W4A8 backend requires ROCm"
    if not torch.cuda.is_available():
        return False, "FlyDSL W4A8 backend requires an available GPU"
    arch = _arch()
    if not any(a in arch for a in _SUPPORTED_ARCHS):
        return False, f"requires one of {_SUPPORTED_ARCHS}, detected {arch}"
    try:
        provider = _provider()
    except ImportError as exc:
        return False, f"no FlyDSL W4A8 kernel provider available: {exc}"
    logger.info("FlyDSL W4A8 kernel provider: %s", provider.name)
    return True, ""


def supports_shape(in_features: int, out_features: int) -> bool:
    """Whether the plain W4A8 kernel can run this layer's shape.

    Two independent constraints:

    * K (``in_features``) must be >= 256 and a multiple of 256 -- Quark's
      ``_validate_a8w4_inputs`` raises otherwise, and ``_pack_weight_asm``
      silently switches to an *unshuffled* layout below 256 that the preshuffle
      kernel cannot read.
    * N (``out_features``) must be a multiple of 32, the smallest tile_n Quark
      falls back to; a non-dividing N makes the grid overrun the matrix.
    """
    return in_features >= _K_MULTIPLE and in_features % _K_MULTIPLE == 0 and out_features % _DIM_MULTIPLE == 0


def supports_svd_shape(in_features: int, out_features: int) -> bool:
    """Whether the fused SVD epilogue can run this layer's shape.

    Quark refuses in/out features below 256 or not a multiple of 256 because the
    preshuffle B layout plus the E8M0 256-K scale granularity produce garbage
    otherwise (Wan's ``proj_out``, out=192, is the motivating case). Callers must
    route rejected layers to an unquantized method.
    """
    return (
        in_features >= _SVD_DIM_MULTIPLE
        and out_features >= _SVD_DIM_MULTIPLE
        and in_features % _SVD_DIM_MULTIPLE == 0
        and out_features % _SVD_DIM_MULTIPLE == 0
    )


# --- provider resolution -----------------------------------------------------


class _Provider(NamedTuple):
    name: str
    gemm: Callable[..., torch.Tensor]
    svd_gemm: Callable[..., torch.Tensor]
    pack_weight: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    # Split of pack_weight for TP: quantize to MXFP4 in *natural* (shardable)
    # order, then shuffle a per-rank shard into the kernel layout at load.
    pack_weight_unshuffled: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    shuffle_for_kernel: Callable[..., tuple[torch.Tensor, torch.Tensor]]


@functools.lru_cache(maxsize=1)
def _provider() -> _Provider:
    requested = os.environ.get("VLLM_OMNI_SVDQUANT_PROVIDER", "auto").lower()
    loaders = {"quark": _load_quark_provider, "flydsl": _load_flydsl_provider}

    if requested in loaders:
        return loaders[requested]()
    if requested != "auto":
        raise ValueError(f"unknown VLLM_OMNI_SVDQUANT_PROVIDER={requested!r}")

    errors = []
    for name in ("flydsl", "quark"):
        try:
            return loaders[name]()
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError("; ".join(errors))


def _load_quark_provider() -> _Provider:
    """Phase 1 provider, backed by Quark's vendored FlyDSL kernels."""
    try:
        import aiter
        from aiter.ops.shuffle import shuffle_weight
        from quark.torch.quantization.nn.modules.aiter_fp4_inference_linear import _pack_weight_asm
        from quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear import (
            _gemm_flydsl_a8w4,
            _gemm_flydsl_svdquant,
            _shuffle_e8m0_scale,
        )
    except ImportError as exc:
        raise ImportError(
            f"Quark with vendored FlyDSL a8w4 kernels is required ({exc}). The released "
            f"amd-quark wheel does not ship them; install Quark PR #6079 "
            f"(branch xiaoyu/svd-quant-flydsl) from source."
        ) from exc

    _quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)

    def _pack_unshuffled(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Natural (N, K/2) MXFP4 + (N, K/32) E8M0 -- no weight/scale shuffle, so a
        # per-rank slice along N or K is still a valid sub-matrix.
        w_q, w_s = _quant(weight, shuffle=False)
        return w_q.view(torch.uint8), w_s.view(torch.uint8)

    def _shuffle_for_kernel(w_q: torch.Tensor, w_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Applied to a per-rank shard at load: shuffle_weight + _shuffle_e8m0_scale
        # reproduce _pack_weight_asm's preshuffled layout exactly (verified equal).
        kernel_w = shuffle_weight(w_q, layout=(16, 16)).view(torch.uint8).contiguous()
        kernel_s = _shuffle_e8m0_scale(w_s.view(torch.uint8)).view(torch.uint8).contiguous()
        return kernel_w, kernel_s

    return _Provider(
        name="quark",
        gemm=_gemm_flydsl_a8w4,
        svd_gemm=_gemm_flydsl_svdquant,
        pack_weight=_pack_weight_asm,
        pack_weight_unshuffled=_pack_unshuffled,
        shuffle_for_kernel=_shuffle_for_kernel,
    )


def _load_flydsl_provider() -> _Provider:
    """Phase 3 provider. The released flydsl wheel ships the compiler only.

    Note that a top-level ``kernels`` module from an unrelated project is
    importable in the ROCm image, so this must not probe ``kernels.gemm``
    directly -- it would resolve to the wrong package and fail confusingly.
    """
    raise ImportError("upstream FlyDSL does not package its kernels yet; use the quark provider")


# --- weight preparation (load time) ------------------------------------------


def pack_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16/FP16 weight to MXFP4 and shuffle it into kernel layout.

    Returns ``(packed_weight, packed_scale)`` as uint8 views, ready to hand to
    the GEMM entrypoints below.
    """
    packed, scale = _provider().pack_weight(weight)
    return packed.view(torch.uint8), scale.view(torch.uint8)


def pack_weight_unshuffled(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to MXFP4 in *natural* order (no kernel shuffle), for TP.

    Returns ``(weight_packed (N, K/2), weight_scale (N, K/32))`` uint8. Unlike
    :func:`pack_weight`, a per-rank slice of these along N (output) or K (input)
    is still a valid sub-matrix, so vLLM can shard them; each rank then calls
    :func:`shuffle_for_kernel` on its local shard.
    """
    return _provider().pack_weight_unshuffled(weight)


def shuffle_for_kernel(weight_packed: torch.Tensor, weight_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle an unshuffled MXFP4 shard into the GEMM's preshuffled layout.

    Run per rank on the local shard. Reproduces :func:`pack_weight`'s bytes
    exactly, so at TP=1 the result is bit-identical to packing the whole weight.
    """
    return _provider().shuffle_for_kernel(weight_packed, weight_scale)


# --- custom ops --------------------------------------------------------------
# The GEMM must sit behind a torch.compile-opaque custom op with a correct
# register_fake, mirroring vllm_omni::rocm_mxfp4_gemm in mxfp4_config.py.


def register_ops() -> None:
    """Register the W4A8 custom ops. Idempotent."""
    if hasattr(torch.ops.vllm_omni, "flydsl_w4a8_gemm"):
        return

    @torch.library.custom_op("vllm_omni::flydsl_w4a8_gemm", mutates_args=())
    def _flydsl_w4a8_gemm(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
    ) -> torch.Tensor:
        del out_features
        # Activations are quantized to MXFP8 inside the Quark entrypoint.
        #
        # ``epilogue`` is not optional when a bias exists: the entrypoint gates
        # the bias operand on ``bias is not None and epilogue != "none"``, so
        # passing bias alone drops it on the floor without any error.
        if bias is None:
            return _provider().gemm(x, weight, weight_scale, torch.bfloat16)
        return _provider().gemm(x, weight, weight_scale, torch.bfloat16, bias=bias, epilogue="bias")

    @_flydsl_w4a8_gemm.register_fake
    def _flydsl_w4a8_gemm_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
    ) -> torch.Tensor:
        del weight, weight_scale, bias
        return torch.empty(x.shape[0], out_features, dtype=torch.bfloat16, device=x.device)

    @torch.library.custom_op("vllm_omni::flydsl_w4a8_svd_gemm", mutates_args=())
    def _flydsl_w4a8_svd_gemm(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        proj_down: torch.Tensor,
        proj_up: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
    ) -> torch.Tensor:
        del out_features
        # d = x @ L1.T, (M, rank) in bf16; the kernel fuses d @ L2.T (+ bias).
        d = torch.nn.functional.linear(x, proj_down)
        return _provider().svd_gemm(x, weight, weight_scale, d, proj_up, torch.bfloat16, bias=bias)

    @_flydsl_w4a8_svd_gemm.register_fake
    def _flydsl_w4a8_svd_gemm_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        proj_down: torch.Tensor,
        proj_up: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
    ) -> torch.Tensor:
        del weight, weight_scale, proj_down, proj_up, bias
        return torch.empty(x.shape[0], out_features, dtype=torch.bfloat16, device=x.device)
