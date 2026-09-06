# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model-facing integration for exact LTX-2 denoiser operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..platform import is_ltx2_ops_eligible
from .attention_gate import try_attention_gate_exact
from .perturbation_blend import try_perturbation_blend_attention_gate_exact
from .qknorm_split_rope import try_qknorm_split_rope_exact
from .residual_gate_add import (
    try_masked_residual_gate_add_exact,
    try_residual_gate_add_exact,
)
from .rms_norm_modulate import (
    try_rms_norm_dual_modulate_exact,
    try_rms_norm_modulate_exact,
)


@dataclass(frozen=True)
class DeferredModulation:
    """Unmaterialized per-sample modulation plus its per-layer table."""

    value: torch.Tensor
    table: torch.Tensor


Modulation = torch.Tensor | DeferredModulation


def should_defer_modulation(hidden_states: torch.Tensor) -> bool:
    """Return whether exact eager kernels can consume deferred modulation."""

    return hidden_states.dtype is torch.bfloat16 and is_ltx2_ops_eligible(hidden_states)


def get_deferred_mod_params(
    scale_shift_table: torch.Tensor,
    temb: torch.Tensor,
    batch_size: int,
) -> tuple[DeferredModulation, ...]:
    """Split modulation inputs without materializing their broadcasted sum."""

    num_ada_params = scale_shift_table.shape[0]
    values = temb.reshape(
        batch_size,
        temb.shape[1],
        num_ada_params,
        -1,
    ).unbind(dim=2)
    tables = scale_shift_table.to(temb.device).unbind(dim=0)
    return tuple(DeferredModulation(value=value, table=table) for value, table in zip(values, tables, strict=True))


def materialize_modulation(value: Modulation) -> torch.Tensor:
    """Materialize a deferred modulation using the reference addition."""

    if isinstance(value, DeferredModulation):
        return value.table + value.value
    return value


def modulate_scale_shift(
    hidden_states: torch.Tensor,
    scale: Modulation,
    shift: Modulation,
) -> torch.Tensor:
    """Apply the reference LTX-2 scale/shift expression."""

    scale = materialize_modulation(scale)
    shift = materialize_modulation(shift)
    return hidden_states * (1 + scale) + shift


def _modulation_parts(value: Modulation) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(value, DeferredModulation):
        return value.value, value.table
    return value, None


def rms_norm_modulate(
    norm: nn.Module,
    hidden_states: torch.Tensor,
    scale: Modulation,
    shift: Modulation,
) -> torch.Tensor:
    """Apply exact eager RMSNorm modulation with the model fallback."""

    if not torch.compiler.is_compiling() and getattr(norm, "weight", None) is None:
        scale_value, scale_table = _modulation_parts(scale)
        shift_value, shift_table = _modulation_parts(shift)
        optimized = try_rms_norm_modulate_exact(
            hidden_states,
            scale_value,
            shift_value,
            float(getattr(norm, "eps", 1e-6)),
            scale_table,
            shift_table,
        )
        if optimized is not None:
            return optimized
    return modulate_scale_shift(norm(hidden_states), scale, shift)


def rms_norm_dual_modulate(
    norm: nn.Module,
    hidden_states: torch.Tensor,
    scale_a: Modulation,
    shift_a: Modulation,
    scale_b: Modulation,
    shift_b: Modulation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply exact eager RMSNorm modulation for two output branches."""

    if not torch.compiler.is_compiling() and getattr(norm, "weight", None) is None:
        scale_a_value, scale_a_table = _modulation_parts(scale_a)
        shift_a_value, shift_a_table = _modulation_parts(shift_a)
        scale_b_value, scale_b_table = _modulation_parts(scale_b)
        shift_b_value, shift_b_table = _modulation_parts(shift_b)
        optimized = try_rms_norm_dual_modulate_exact(
            hidden_states,
            scale_a_value,
            shift_a_value,
            scale_b_value,
            shift_b_value,
            float(getattr(norm, "eps", 1e-6)),
            scale_a_table,
            shift_a_table,
            scale_b_table,
            shift_b_table,
        )
        if optimized is not None:
            return optimized
    normalized = norm(hidden_states)
    return (
        modulate_scale_shift(normalized, scale_a, shift_a),
        modulate_scale_shift(normalized, scale_b, shift_b),
    )


def residual_gate_add(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: Modulation,
    perturbation_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply an exact eager gated residual update with the model fallback."""

    if not torch.compiler.is_compiling():
        gate_value, gate_table = _modulation_parts(gate)
        if perturbation_mask is None:
            optimized = try_residual_gate_add_exact(
                residual,
                update,
                gate_value,
                gate_table,
            )
        else:
            optimized = try_masked_residual_gate_add_exact(
                residual,
                update,
                gate_value,
                perturbation_mask,
                gate_table,
            )
        if optimized is not None:
            return optimized
    if perturbation_mask is not None:
        update = update * perturbation_mask
    return residual + update * materialize_modulation(gate)


def try_qknorm_split_rope(
    attn: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Adapt an LTX attention module to the exact QK/RoPE kernel."""

    if (
        torch.compiler.is_compiling()
        or attn.rope_type != "split"
        or query_rotary_emb is None
        or key_rotary_emb is None
        or not query.is_cuda
        or not isinstance(attn.norm_q, torch.nn.RMSNorm)
        or not isinstance(attn.norm_k, torch.nn.RMSNorm)
    ):
        return None
    q_cos, q_sin = query_rotary_emb
    k_cos, k_sin = key_rotary_emb
    eps = float(attn.norm_q.eps)
    if eps != float(attn.norm_k.eps):
        return None
    return try_qknorm_split_rope_exact(
        query,
        q_cos,
        q_sin,
        attn.norm_q.weight,
        key,
        k_cos,
        k_sin,
        attn.norm_k.weight,
        eps,
        attn.heads,
        attn.head_dim,
    )


def try_perturbation_blend_attention_gate(
    hidden_states: torch.Tensor,
    value: torch.Tensor,
    perturbation_mask: torch.Tensor,
    gate_logits: torch.Tensor,
    head_dim: int,
) -> torch.Tensor | None:
    """Try the exact combined perturbation-blend and attention-gate path."""

    if torch.compiler.is_compiling():
        return None
    return try_perturbation_blend_attention_gate_exact(
        hidden_states,
        value,
        perturbation_mask,
        gate_logits,
        head_dim,
    )


def apply_attention_gate(
    hidden_states: torch.Tensor,
    gate_logits: torch.Tensor,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply the exact eager attention gate with the model fallback."""

    optimized = None
    if not torch.compiler.is_compiling():
        optimized = try_attention_gate_exact(hidden_states, gate_logits, head_dim)
    if optimized is not None:
        return optimized
    hidden_states = hidden_states.unflatten(2, (heads, head_dim))
    gates = 2.0 * torch.sigmoid(gate_logits)
    return (hidden_states * gates.unsqueeze(-1)).flatten(2, 3)


__all__ = [
    "Modulation",
    "apply_attention_gate",
    "get_deferred_mod_params",
    "materialize_modulation",
    "modulate_scale_shift",
    "residual_gate_add",
    "rms_norm_dual_modulate",
    "rms_norm_modulate",
    "should_defer_modulation",
    "try_perturbation_blend_attention_gate",
    "try_qknorm_split_rope",
]
