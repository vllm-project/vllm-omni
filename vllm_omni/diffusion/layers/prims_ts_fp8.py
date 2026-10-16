# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable FP8 operations backed by FlashInfer PrimsTS GEMMs."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeAlias

import torch
from torch.library import Library
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.diffusion.layers.fp8_per_token_quant import per_token_fp8_quant

_ONE_PASS_MAX_COLS = 4096
_ONE_PASS_MIN_ROWS = 16384

if TYPE_CHECKING:
    from flashinfer.prims_ts.gemm import PrimsTsGemmConfig


ConfigProvider = Callable[[torch.Tensor, torch.Tensor, str, int | None], "PrimsTsGemmConfig | None"]
Fp8Activation: TypeAlias = tuple[torch.Tensor, torch.Tensor]
_CONFIG_PROVIDERS: dict[str, ConfigProvider] = {}


@functools.lru_cache(maxsize=16)
def token_positions(rows: int, device: torch.device) -> torch.Tensor:
    """Identity RoPE positions (token i reads cos/sin row i), built once per row count.

    The fused QKV ops build these themselves when called without positions, so compiled
    graphs do not launch an ``arange`` per attention block.
    """
    return torch.arange(rows, device=device, dtype=torch.int64)


def prepare_prims_ts_fp8_weights(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    layout: Literal["linear", "swiglu"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert post-load FP8 linear weights to the PrimsTS operand layout.

    The source weight is ``[K, N]`` with one scale per output channel; PrimsTS
    consumes ``[N, K]``. For ``swiglu``, source channels are laid out as
    ``[gate | up]`` and the kernel consumes adjacent ``[up, gate]`` pairs.
    """
    if weight.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"PrimsTS FP8 preparation expected float8_e4m3fn weights, got {weight.dtype}")
    if weight.ndim != 2 or weight_scale.ndim != 1 or weight_scale.numel() != weight.shape[1]:
        raise ValueError("FP8 weights must be [K, N] with one scale per output channel")
    if layout == "swiglu":
        n = weight.shape[1]
        if n % 2:
            raise ValueError(f"PrimsTS FP8 SwiGLU requires an even output width, got {n}")
        half = n // 2
        transposed = weight.t()
        packed_weight = torch.stack((transposed[half:], transposed[:half]), dim=1).reshape(n, weight.shape[0])
        packed_scale = torch.stack((weight_scale[half:], weight_scale[:half]), dim=1).reshape(n)
    else:
        packed_weight, packed_scale = weight.t(), weight_scale
    return packed_weight.contiguous(), packed_scale.contiguous()


def register_prims_ts_fp8_config_provider(name: str, provider: ConfigProvider) -> None:
    """Register an optional consumer tactic policy for the FP8 GEMM operations."""
    if not name:
        raise ValueError("Config provider name must not be empty")
    existing = _CONFIG_PROVIDERS.setdefault(name, provider)
    if existing is not provider:
        raise ValueError(f"PrimsTS FP8 config provider {name!r} is already registered")


def _config_for(
    provider_name: str,
    activation: torch.Tensor,
    weight: torch.Tensor,
    epilogue: str,
    head_dim: int | None = None,
) -> PrimsTsGemmConfig | None:
    if not provider_name:
        return None
    try:
        provider = _CONFIG_PROVIDERS[provider_name]
    except KeyError as exc:
        raise RuntimeError(f"PrimsTS FP8 config provider {provider_name!r} is not registered") from exc
    return provider(activation, weight, epilogue, head_dim)


def prims_ts_fp8_quantize(
    activation: torch.Tensor,
    *,
    use_per_token_if_dynamic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an activation to FP8 for PrimsTS GEMMs (per token by default)."""
    activation_2d = activation.reshape(-1, activation.shape[-1]).detach()
    rows, cols = activation_2d.shape
    # The one-pass kernel is up to 1.8x faster for many rows of moderate width (GB300,
    # K=3072, M >= 16k); vLLM's kernel is as fast for wide rows and faster for few rows.
    if use_per_token_if_dynamic and cols <= _ONE_PASS_MAX_COLS and rows >= _ONE_PASS_MIN_ROWS:
        return per_token_fp8_quant(activation_2d)
    from vllm import _custom_ops as ops

    return ops.scaled_fp8_quant(activation_2d.contiguous(), use_per_token_if_dynamic=use_per_token_if_dynamic)


def _prepare_fp8_activation(
    activation: torch.Tensor | Fp8Activation,
    *,
    use_per_token_if_dynamic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(activation, tuple):
        if len(activation) != 2:
            raise ValueError("prequantized FP8 activation must be an (activation, scale) pair")
        packed, scale = activation
        if packed.dtype is not torch.float8_e4m3fn:
            raise TypeError("prequantized activation must use torch.float8_e4m3fn")
        if packed.ndim != 2:
            raise ValueError("prequantized activation must have shape [M, K]")
        return packed, scale.reshape(-1)
    if activation.dtype is torch.float8_e4m3fn:
        raise TypeError("pass prequantized FP8 input as an (activation, scale) pair")
    return prims_ts_fp8_quantize(activation, use_per_token_if_dynamic=use_per_token_if_dynamic)


def _fp8_linear_impl(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    config_provider: str,
) -> torch.Tensor:
    from flashinfer.prims_ts import gemm

    return gemm.fp8_linear(
        activation,
        weight,
        activation_scale.reshape(-1),
        weight_scale,
        config=_config_for(config_provider, activation, weight, "linear"),
    )


def _fp8_linear_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    config_provider: str,
) -> torch.Tensor:
    del activation_scale, weight_scale, config_provider
    return torch.empty((activation.shape[0], weight.shape[0]), device=activation.device, dtype=torch.bfloat16)


def _fp8_swiglu_impl(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    config_provider: str,
) -> torch.Tensor:
    from flashinfer.prims_ts import gemm

    return gemm.fp8_linear_swiglu(
        activation,
        weight,
        activation_scale.reshape(-1),
        weight_scale,
        config=_config_for(config_provider, activation, weight, "swiglu"),
    )


def _fp8_swiglu_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    config_provider: str,
) -> torch.Tensor:
    del activation_scale, weight_scale, config_provider
    return torch.empty((activation.shape[0], weight.shape[0] // 2), device=activation.device, dtype=torch.bfloat16)


def _fp8_qkv_qknorm_rope_impl(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor | None,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    config_provider: str,
) -> torch.Tensor:
    from flashinfer.prims_ts import gemm

    if positions is None:
        positions = token_positions(activation.shape[0], activation.device)

    return gemm.fp8_qkv_qknorm_rope(
        activation.detach(),
        weight.detach(),
        activation_scale.reshape(-1).detach(),
        weight_scale.detach(),
        q_norm.detach(),
        k_norm.detach(),
        cos_sin.detach(),
        positions.detach(),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_neox=False,
        config=_config_for(config_provider, activation, weight, "qkv_qknorm_rope", head_dim),
    )


def _fp8_qkv_qknorm_rope_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor | None,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    config_provider: str,
) -> torch.Tensor:
    del activation_scale, weight_scale, q_norm, k_norm, cos_sin, positions
    del num_q_heads, num_kv_heads, head_dim, config_provider
    return torch.empty((activation.shape[0], weight.shape[0]), device=activation.device, dtype=torch.bfloat16)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "prims_ts_fp8_linear"):
    direct_register_custom_op(
        op_name="prims_ts_fp8_linear",
        op_func=_fp8_linear_impl,
        fake_impl=_fp8_linear_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )
if not hasattr(torch.ops.vllm_omni, "prims_ts_fp8_swiglu"):
    direct_register_custom_op(
        op_name="prims_ts_fp8_swiglu",
        op_func=_fp8_swiglu_impl,
        fake_impl=_fp8_swiglu_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )
if not hasattr(torch.ops.vllm_omni, "prims_ts_fp8_qkv_qknorm_rope"):
    direct_register_custom_op(
        op_name="prims_ts_fp8_qkv_qknorm_rope",
        op_func=_fp8_qkv_qknorm_rope_impl,
        fake_impl=_fp8_qkv_qknorm_rope_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )

_LINEAR_OP = torch.ops.vllm_omni.prims_ts_fp8_linear
_SWIGLU_OP = torch.ops.vllm_omni.prims_ts_fp8_swiglu
_QKV_QKNORM_ROPE_OP = torch.ops.vllm_omni.prims_ts_fp8_qkv_qknorm_rope


def prims_ts_fp8_linear(
    activation: torch.Tensor | Fp8Activation,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    use_per_token_if_dynamic: bool = True,
    config_provider: str | None = None,
) -> torch.Tensor:
    """Apply an FP8 linear projection using prepared ``[N, K]`` weights.

    Pass a floating-point tensor to quantize on demand, or a prequantized
    ``(activation, scale)`` pair to reuse a quantization across projections.
    """
    activation, activation_scale = _prepare_fp8_activation(
        activation,
        use_per_token_if_dynamic=use_per_token_if_dynamic,
    )
    return _LINEAR_OP(activation, weight, activation_scale, weight_scale, config_provider or "")


def prims_ts_fp8_swiglu(
    activation: torch.Tensor | Fp8Activation,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    use_per_token_if_dynamic: bool = True,
    config_provider: str | None = None,
) -> torch.Tensor:
    """Apply FP8 linear+SwiGLU, optionally quantizing a floating-point input.

    Pass a floating-point tensor to quantize on demand, or a prequantized
    ``(activation, scale)`` pair to reuse a quantization.
    """
    activation, activation_scale = _prepare_fp8_activation(
        activation,
        use_per_token_if_dynamic=use_per_token_if_dynamic,
    )
    return _SWIGLU_OP(activation, weight, activation_scale, weight_scale, config_provider or "")


def prims_ts_fp8_qkv_qknorm_rope(
    activation: torch.Tensor | Fp8Activation,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor | None,
    *,
    use_per_token_if_dynamic: bool = True,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    config_provider: str | None = None,
) -> torch.Tensor:
    """Project FP8 QKV and apply fused Q/K RMSNorm plus adjacent-pair RoPE.

    Inputs use packed Q/K/V ``[N, K]`` weights, per-output-channel weight
    scales, ``[head_dim]`` norm weights, and a packed float32 ``[M, head_dim]``
    cosine/sine table with one position per token (``positions=None`` means token i
    uses row i). Pass floating-point input to
    quantize on demand, or an ``(activation, scale)`` pair to reuse a prior
    quantization.
    The current FlashInfer specialization requires equal Q/KV head counts,
    head_dim 128, and uses its internal RMSNorm epsilon.
    """
    activation, activation_scale = _prepare_fp8_activation(
        activation,
        use_per_token_if_dynamic=use_per_token_if_dynamic,
    )
    if activation.ndim != 2 or weight.ndim != 2 or activation.shape[1] != weight.shape[1]:
        raise ValueError("activation and weight must have compatible [M, K] and [N, K] shapes")
    if weight.dtype is not torch.float8_e4m3fn:
        raise TypeError("weight must use torch.float8_e4m3fn")
    if head_dim != 128:
        raise ValueError("PrimsTS fused QKV currently requires head_dim=128")
    if num_q_heads != num_kv_heads or weight.shape[0] != 3 * num_q_heads * head_dim:
        raise ValueError("weight must contain equal Q/K/V groups for the requested heads")
    if q_norm.shape != (head_dim,) or k_norm.shape != (head_dim,):
        raise ValueError("Q/K norm weights must each have shape [head_dim]")
    if cos_sin.shape != (activation.shape[0], head_dim) or (
        positions is not None and positions.shape != (activation.shape[0],)
    ):
        raise ValueError("cos_sin and positions must have one row or index per token")
    if activation_scale.numel() != activation.shape[0] or weight_scale.numel() != weight.shape[0]:
        raise ValueError("activation and weight scales must contain one value per token and output channel")
    return _QKV_QKNORM_ROPE_OP(
        activation,
        weight,
        activation_scale,
        weight_scale,
        q_norm,
        k_norm,
        cos_sin,
        positions,
        num_q_heads,
        num_kv_heads,
        head_dim,
        config_provider or "",
    )


__all__ = [
    "prepare_prims_ts_fp8_weights",
    "Fp8Activation",
    "prims_ts_fp8_linear",
    "prims_ts_fp8_quantize",
    "prims_ts_fp8_qkv_qknorm_rope",
    "prims_ts_fp8_swiglu",
    "register_prims_ts_fp8_config_provider",
]
