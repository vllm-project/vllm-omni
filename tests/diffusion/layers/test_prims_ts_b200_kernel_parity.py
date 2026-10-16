# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B200-only parity checks for the Flux3 PrimsTS kernel building blocks."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import prims_ts_fp8, prims_ts_nvfp4

pytestmark = [pytest.mark.core_model]

_TOKENS = 128
_HIDDEN = 512
_HEADS = 4
_HEAD_DIM = 128
_QKV = 3 * _HEADS * _HEAD_DIM
_MLP = 1024


def _require_prims_ts_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("FlashInfer PrimsTS kernels require Blackwell-class CUDA")
    pytest.importorskip("flashinfer.prims_ts.gemm")


def _bf16_randn(shape: tuple[int, ...], *, device: torch.device, scale: float = 0.15) -> torch.Tensor:
    return (torch.randn(shape, device=device, dtype=torch.float32) * scale).to(torch.bfloat16)


def _rope_table(tokens: int, head_dim: int, *, device: torch.device) -> torch.Tensor:
    freqs = torch.arange(head_dim // 2, device=device, dtype=torch.float32) / head_dim
    positions = torch.arange(tokens, device=device, dtype=torch.float32).unsqueeze(1)
    angles = positions * freqs.unsqueeze(0)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps=1e-6).to(x.dtype)


def _apply_interleaved_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    rotary_dim = rope.shape[-1]
    half = rotary_dim // 2
    x_float = x.float()
    cos = rope[:, :half].unsqueeze(1)
    sin = rope[:, half:].unsqueeze(1)
    first = x_float[..., 0:rotary_dim:2]
    second = x_float[..., 1:rotary_dim:2]
    rotated = torch.stack((first * cos - second * sin, second * cos + first * sin), dim=-1).flatten(-2)
    return torch.cat((rotated, x_float[..., rotary_dim:]), dim=-1).to(x.dtype)


def _qkv_qknorm_rope_reference(
    projected: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    rope: torch.Tensor,
) -> torch.Tensor:
    rows = projected.shape[0]
    head_width = _HEADS * _HEAD_DIM
    q = projected[:, :head_width].reshape(rows, _HEADS, _HEAD_DIM)
    k = projected[:, head_width : 2 * head_width].reshape(rows, _HEADS, _HEAD_DIM)
    v = projected[:, 2 * head_width :].reshape(rows, _HEADS, _HEAD_DIM)

    q = _apply_interleaved_rope(_rms_norm(q, q_norm).to(v.dtype), rope)
    k = _apply_interleaved_rope(_rms_norm(k, k_norm).to(v.dtype), rope)
    return torch.cat((q.reshape(rows, -1), k.reshape(rows, -1), v.reshape(rows, -1)), dim=-1)


def _fp8_weight(source_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = source_weight.float().abs().amax(dim=0).clamp(min=1e-6) / fp8_max
    weight = (source_weight.float() / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    # Match ModelOpt's post-load [K, N] column-major layout for CUTLASS.
    return weight.t().contiguous().t(), scale.contiguous()


def _fp8_fallback_linear(
    activation_pair: prims_ts_fp8.Fp8Activation,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    activation, activation_scale = activation_pair
    return ops.cutlass_scaled_mm(activation, weight, activation_scale, weight_scale, torch.bfloat16)


def _nvfp4_quantize_cutlass(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops

    global_scale = torch.ones((1,), device=x.device, dtype=torch.float32)
    return ops.scaled_fp4_quant(x, global_scale, is_sf_swizzled_layout=True)


def _nvfp4_prims_pair(pair: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    packed, scale = pair
    return packed, scale.view(torch.uint8).reshape(-1).contiguous()


def _nvfp4_weight(
    out_features: int,
    in_features: int,
    *,
    device: torch.device,
    scale: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _nvfp4_quantize_cutlass(_bf16_randn((out_features, in_features), device=device, scale=scale))


def _nvfp4_fallback_linear(
    activation_pair: tuple[torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    activation, activation_scale = activation_pair
    alpha = torch.ones((1,), device=activation.device, dtype=torch.float32)
    return ops.cutlass_scaled_fp4_mm(activation, weight, activation_scale, weight_scale, alpha, torch.bfloat16)


def _nvfp4_dequantize(pair: tuple[torch.Tensor, torch.Tensor], width: int) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import dequantize_to_dtype

    packed, scale = pair
    global_scale = torch.ones((1,), device=packed.device, dtype=torch.float32)
    return dequantize_to_dtype(packed, scale, global_scale, torch.bfloat16, swizzle=True)[:, :width]


@hardware_test(res={"cuda": "B200"}, num_cards=1)
@torch.no_grad()
def test_prims_ts_fp8_ops_match_cutlass_fallback() -> None:
    _require_prims_ts_gpu()
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = _bf16_randn((_TOKENS, _HIDDEN), device=device)
    activation_pair = prims_ts_fp8.prims_ts_fp8_quantize(x)
    rope = _rope_table(_TOKENS, _HEAD_DIM, device=device)
    positions = torch.arange(_TOKENS, device=device, dtype=torch.int64)
    q_norm = (1 + _bf16_randn((_HEAD_DIM,), device=device, scale=0.01)).contiguous()
    k_norm = (1 + _bf16_randn((_HEAD_DIM,), device=device, scale=0.01)).contiguous()

    qkv_weight, qkv_scale = _fp8_weight(_bf16_randn((_HIDDEN, _QKV), device=device))
    prims_qkv_weight, prims_qkv_scale = prims_ts_fp8.prepare_prims_ts_fp8_weights(
        qkv_weight,
        qkv_scale,
        layout="linear",
    )
    actual_qkv = prims_ts_fp8.prims_ts_fp8_qkv_qknorm_rope(
        activation_pair,
        prims_qkv_weight,
        prims_qkv_scale,
        q_norm,
        k_norm,
        rope,
        positions,
        num_q_heads=_HEADS,
        num_kv_heads=_HEADS,
        head_dim=_HEAD_DIM,
    )
    expected_qkv = _qkv_qknorm_rope_reference(
        _fp8_fallback_linear(activation_pair, qkv_weight, qkv_scale),
        q_norm,
        k_norm,
        rope,
    )
    torch.testing.assert_close(actual_qkv, expected_qkv, rtol=2e-2, atol=0.0625)

    linear_weight, linear_scale = _fp8_weight(_bf16_randn((_HIDDEN, _HIDDEN), device=device))
    prims_linear_weight, prims_linear_scale = prims_ts_fp8.prepare_prims_ts_fp8_weights(
        linear_weight,
        linear_scale,
        layout="linear",
    )
    torch.testing.assert_close(
        prims_ts_fp8.prims_ts_fp8_linear(activation_pair, prims_linear_weight, prims_linear_scale),
        _fp8_fallback_linear(activation_pair, linear_weight, linear_scale),
        rtol=2e-2,
        atol=0.0625,
    )

    swiglu_weight, swiglu_scale = _fp8_weight(_bf16_randn((_HIDDEN, 2 * _MLP), device=device))
    prims_swiglu_weight, prims_swiglu_scale = prims_ts_fp8.prepare_prims_ts_fp8_weights(
        swiglu_weight,
        swiglu_scale,
        layout="swiglu",
    )
    actual_swiglu = prims_ts_fp8.prims_ts_fp8_swiglu(activation_pair, prims_swiglu_weight, prims_swiglu_scale)
    gate, up = _fp8_fallback_linear(activation_pair, swiglu_weight, swiglu_scale).chunk(2, dim=-1)
    torch.testing.assert_close(actual_swiglu, F.silu(gate) * up, rtol=2e-2, atol=0.0625)


@hardware_test(res={"cuda": "B200"}, num_cards=1)
@torch.no_grad()
def test_prims_ts_nvfp4_ops_match_cutlass_fallback() -> None:
    _require_prims_ts_gpu()
    torch.manual_seed(1)
    device = torch.device("cuda")
    owner = nn.Module()
    cache_id = prims_ts_nvfp4.register_prims_ts_nvfp4_layer(owner)
    x = _bf16_randn((_TOKENS, _HIDDEN), device=device)
    activation_pair = _nvfp4_quantize_cutlass(x)
    prims_activation_pair = _nvfp4_prims_pair(activation_pair)
    rope = _rope_table(_TOKENS, _HEAD_DIM, device=device)
    positions = torch.arange(_TOKENS, device=device, dtype=torch.int64)
    q_norm = (1 + _bf16_randn((_HEAD_DIM,), device=device, scale=0.01)).contiguous()
    k_norm = (1 + _bf16_randn((_HEAD_DIM,), device=device, scale=0.01)).contiguous()

    qkv_weight, qkv_scale = _nvfp4_weight(_QKV, _HIDDEN, device=device)
    prims_qkv_weight, prims_qkv_scale = prims_ts_nvfp4.prepare_prims_ts_nvfp4_weights(
        qkv_weight,
        qkv_scale,
        layout="linear",
    )
    actual_qkv = prims_ts_nvfp4.prims_ts_nvfp4_qkv_qknorm_rope(
        prims_activation_pair,
        prims_qkv_weight,
        prims_qkv_scale,
        1.0,
        1.0,
        torch.ones((3,), device=device, dtype=torch.float32),
        q_norm,
        k_norm,
        rope,
        positions,
        cache_id,
    )
    expected_qkv = _qkv_qknorm_rope_reference(
        _nvfp4_fallback_linear(activation_pair, qkv_weight, qkv_scale),
        q_norm,
        k_norm,
        rope,
    )
    torch.testing.assert_close(actual_qkv, expected_qkv, rtol=2e-2, atol=0.0625)

    linear_weight, linear_scale = _nvfp4_weight(_HIDDEN, _HIDDEN, device=device)
    prims_linear_weight, prims_linear_scale = prims_ts_nvfp4.prepare_prims_ts_nvfp4_weights(
        linear_weight,
        linear_scale,
        layout="linear",
    )
    torch.testing.assert_close(
        prims_ts_nvfp4.prims_ts_nvfp4_linear(
            prims_activation_pair,
            prims_linear_weight,
            prims_linear_scale,
            1.0,
            1.0,
            cache_id,
        ),
        _nvfp4_fallback_linear(activation_pair, linear_weight, linear_scale),
        rtol=2e-2,
        atol=0.0625,
    )

    # Keep the fused SwiGLU fixture below the range where adjacent NVFP4 output
    # bins dominate the parity signal.
    swiglu_weight, swiglu_scale = _nvfp4_weight(2 * _MLP, _HIDDEN, device=device, scale=0.05)
    prims_swiglu_weight, prims_swiglu_scale = prims_ts_nvfp4.prepare_prims_ts_nvfp4_weights(
        swiglu_weight,
        swiglu_scale,
        layout="swiglu",
    )
    actual_swiglu = prims_ts_nvfp4.prims_ts_nvfp4_swiglu(
        prims_activation_pair,
        prims_swiglu_weight,
        prims_swiglu_scale,
        1.0,
        1.0,
        cache_id,
    )
    gate, up = _nvfp4_fallback_linear(activation_pair, swiglu_weight, swiglu_scale).chunk(2, dim=-1)
    expected_swiglu = _nvfp4_quantize_cutlass(F.silu(gate) * up)
    torch.testing.assert_close(
        _nvfp4_dequantize(actual_swiglu, _MLP),
        _nvfp4_dequantize(expected_swiglu, _MLP),
        rtol=2e-2,
        atol=0.0625,
    )
