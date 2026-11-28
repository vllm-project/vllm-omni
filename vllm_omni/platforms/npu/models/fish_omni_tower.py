# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend kernels for the Drama 3 codebook tower."""

import importlib
from collections.abc import Callable
from typing import cast

import torch
from vllm.logger import init_logger
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

from vllm_omni.model_executor.models.fish_omni import kernels
from vllm_omni.platforms.npu.layers.rotary_embedding import npu_rotary_mul_with_bsnd_fallback

logger = init_logger(__name__)

# Limit the tower fast paths to A2/A3.
_FUSED_TOWER_SUPPORTED = get_ascend_device_type() in (AscendDeviceType.A2, AscendDeviceType.A3)
_gemma_op: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None
if _FUSED_TOWER_SUPPORTED:
    try:
        importlib.import_module("vllm_ascend.vllm_ascend_C")
    except ImportError as error:
        logger.warning("fish_omni: could not import Ascend custom ops: %s", error)
    _gemma_op = getattr(torch.ops._C_ascend, "npu_gemma_rms_norm", None)
    if _gemma_op is None:
        logger.warning("fish_omni: Ascend Gemma RMSNorm is unavailable; using eager tower norms")


def _rope(x: torch.Tensor, cos_sin_cache: torch.Tensor, positions: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """NeoX partial RoPE in fp32, reusing the shared NPU layout fallback."""
    half = rotary_dim // 2
    tokens = x.shape[0]
    picked = cos_sin_cache.index_select(0, positions.to(torch.long)).float()
    cos = picked[:, :half].repeat(1, 2).view(1, tokens, rotary_dim)
    sin = picked[:, half:rotary_dim].repeat(1, 2).view(1, tokens, rotary_dim)
    rotated = (
        npu_rotary_mul_with_bsnd_fallback(
            x[..., :rotary_dim].float().transpose(0, 1).unsqueeze(0), cos, sin, unsqueeze_dim=1
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    return torch.cat((rotated.to(dtype=x.dtype), x[..., rotary_dim:]), dim=-1)


def qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | tuple[int, int, int] | None = None,
    norm_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gemma QK norm, partial RoPE and gate split for bf16 tower inputs."""
    if (
        not _FUSED_TOWER_SUPPORTED
        or norm_beta != 1.0
        or mrope_section is not None
        or positions.ndim != 1
        or q_gate.shape[0] == 0
        or rotary_dim == 0
        or q_gate.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or q_weight.dtype != torch.bfloat16
        or k_weight.dtype != torch.bfloat16
    ):
        return kernels.eager_qk_rmsnorm_rope_gate(
            q_gate,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            mrope_section,
            norm_beta,
        )
    tokens = q_gate.shape[0]
    q_heads = q_gate.view(tokens, num_q_heads, 2 * head_dim)
    gate = q_heads[..., head_dim:]
    # Flatten the head axes to use the same two-dimensional norm path.
    # The fallback allocates its output, preserving q_gate and k even
    # when contiguous() returns an alias (e.g. single-token decoding).
    query = cast(torch.Tensor, gemma_rms_norm(q_heads[..., :head_dim].reshape(-1, head_dim), q_weight, eps))
    key = cast(torch.Tensor, gemma_rms_norm(k.reshape(-1, head_dim), k_weight, eps))
    query = _rope(query.view(tokens, num_q_heads, head_dim), cos_sin_cache, positions, rotary_dim)
    key = _rope(key.view(tokens, num_kv_heads, head_dim), cos_sin_cache, positions, rotary_dim)
    return (
        query.reshape(tokens, num_q_heads * head_dim),
        key.reshape(tokens, num_kv_heads * head_dim),
        gate.reshape(tokens, num_q_heads * head_dim).contiguous(),
    )


def gemma_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Use the fused norm without a residual. Its fp32 reduction may differ at bf16 rounding boundaries."""
    if (
        _gemma_op is not None
        and residual is None
        and x.shape[0] > 0
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
    ):
        return _gemma_op(x.contiguous(), weight, eps)[0]
    return kernels.gemma_rms_norm(x, weight, eps, residual)


def tower_mix_norm(
    hidden: torch.Tensor,
    code: torch.Tensor,
    code_mask: torch.Tensor,
    table: torch.Tensor,
    extra: torch.Tensor | None,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Mix code embeddings and normalize both halves using Gemma RMSNorm."""
    if (
        _gemma_op is None
        or hidden.shape[0] == 0
        or hidden.dtype != torch.bfloat16
        or table.dtype != torch.bfloat16
        or enorm_weight.dtype != torch.bfloat16
        or hnorm_weight.dtype != torch.bfloat16
        or (extra is not None and extra.dtype != torch.bfloat16)
    ):
        return kernels.tower_mix_norm(hidden, code, code_mask, table, extra, enorm_weight, hnorm_weight, eps)
    idx = code.clamp(0, table.shape[0] - 1)
    # The tower mask is boolean, so masking in bf16 is exact.
    kept = table.index_select(0, idx) * code_mask.to(table.dtype).unsqueeze(-1)
    if extra is not None:
        kept = kept + extra
    return torch.cat((_gemma_op(kept, enorm_weight, eps)[0], _gemma_op(hidden, hnorm_weight, eps)[0]), dim=-1)
