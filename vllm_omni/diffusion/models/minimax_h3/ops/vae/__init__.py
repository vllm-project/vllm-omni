# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Install decoder operator optimizations for the MiniMax H3 video VAE."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from types import FunctionType, MethodType
from typing import Any

import torch
import torch.nn as nn

from vllm_omni.diffusion.layers.activation import SiluAndMul
from vllm_omni.platforms.npu import is_a5

from .dispatch import resolve_h3_vae_operators
from .qk_norm_rope import try_rms_norm_npu, try_rotary_npu


def _is_boolean_flag(value: Any) -> bool:
    """Accept bools and the JSON-derived 0/1 flags used by remote VAE code."""

    return isinstance(value, bool) or type(value) is int and value in (0, 1)


def _uses_fp32_attention_norm(attention: nn.Module) -> bool:
    """Probe the loaded remote module's Q/K normalization semantics."""

    forward = getattr(type(attention), "forward", None)
    namespace = getattr(forward, "__globals__", None)
    norm_input = namespace.get("_vit_norm_input") if isinstance(namespace, dict) else None
    if not callable(norm_input):
        return False

    probe = torch.empty(1, dtype=torch.float16)
    try:
        normalized = norm_input(attention.norm_q, probe)
    except Exception:
        return False
    return (
        isinstance(normalized, torch.Tensor) and normalized.shape == probe.shape and normalized.dtype == torch.float32
    )


def _optimized_feed_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return type(self).forward(self, hidden_states)

    hidden_states = self.w1(hidden_states)
    if hidden_states.is_cuda and hidden_states.dtype == torch.float16:
        hidden_states = self._omni_silu_and_mul(hidden_states)
    else:
        gate, hidden_states = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act_fn(gate) * hidden_states
    return self.w2(hidden_states)


def _optimized_attention(
    self: nn.Module,
    hidden_states: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    pack_info: dict[str, Any] | None = None,
) -> torch.Tensor:
    pack_info = {} if pack_info is None else pack_info
    if torch.compiler.is_compiling() or self.spatial_parallel or rotary_pos_emb is None:
        return type(self).forward(self, hidden_states, rotary_pos_emb, pack_info)

    batch_size, sequence, _ = hidden_states.shape
    qkv = self.to_qkv(hidden_states).view(
        batch_size,
        sequence,
        -1,
        3 * self.dim_head,
    )
    query, key, value = qkv.chunk(3, dim=-1)
    optimized_qk = self._omni_qk_norm_rope(
        query,
        key,
        rotary_pos_emb,
        float(self.norm_q.eps),
    )
    if optimized_qk is None:
        return type(self).forward(self, hidden_states, rotary_pos_emb, pack_info)

    query, key = optimized_qk
    hidden_states = self.perform_attention(query, key, value, pack_info)
    hidden_states = hidden_states.reshape(batch_size, sequence, -1)
    return self.to_out(hidden_states)


def _optimized_transformer_block(
    self: nn.Module,
    hidden_states: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    pack_info: dict[str, Any] | None = None,
) -> torch.Tensor:
    pack_info = {} if pack_info is None else pack_info
    if torch.compiler.is_compiling() or hidden_states.dtype != torch.float32:
        return type(self).forward(self, hidden_states, rotary_pos_emb, pack_info)

    normalized = self.norm1(hidden_states.float()).to(hidden_states.dtype)
    attention_output = self.attn(normalized, rotary_pos_emb, pack_info)
    updated = self._omni_scaled_residual(
        hidden_states,
        attention_output,
        self.scale1,
    )
    hidden_states = hidden_states + attention_output * self.scale1 if updated is None else updated

    normalized = self.norm2(hidden_states.float()).to(hidden_states.dtype)
    feed_forward_output = self.ff(normalized)
    updated = self._omni_scaled_residual(
        hidden_states,
        feed_forward_output,
        self.scale2,
    )
    return hidden_states + feed_forward_output * self.scale2 if updated is None else updated


def _decoder_block_linears(decoder: nn.Module) -> tuple[nn.Linear, ...] | None:
    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        return None

    linears: list[nn.Linear] = []
    for block in blocks:
        attention = getattr(block, "attn", None)
        feed_forward = getattr(block, "ff", None)
        to_qkv = getattr(attention, "to_qkv", None)
        to_out = getattr(attention, "to_out", None)
        w1 = getattr(feed_forward, "w1", None)
        w2 = getattr(feed_forward, "w2", None)
        candidates = (to_qkv, to_out, w1, w2)
        dim_head = getattr(attention, "dim_head", None)
        scale1 = getattr(block, "scale1", None)
        scale2 = getattr(block, "scale2", None)
        if (
            not all(isinstance(linear, nn.Linear) for linear in candidates)
            or not _is_boolean_flag(getattr(attention, "spatial_parallel", None))
            or not isinstance(dim_head, int)
            or dim_head <= 0
            or not callable(getattr(attention, "perform_attention", None))
            or not isinstance(getattr(attention, "norm_q", None), nn.RMSNorm)
            or not isinstance(getattr(attention, "norm_k", None), nn.RMSNorm)
            or attention.norm_q.weight is not None
            or attention.norm_k.weight is not None
            or attention.norm_q.eps != attention.norm_k.eps
            or not _uses_fp32_attention_norm(attention)
            or not isinstance(getattr(block, "norm1", None), nn.RMSNorm)
            or not isinstance(getattr(block, "norm2", None), nn.RMSNorm)
            or block.norm1.weight is None
            or block.norm2.weight is None
            or not getattr(block, "use_scale", False)
            or not isinstance(scale1, torch.Tensor)
            or not isinstance(scale2, torch.Tensor)
            or not getattr(feed_forward, "use_gated", False)
            or not isinstance(getattr(feed_forward, "act_fn", None), nn.SiLU)
            or not hasattr(feed_forward, "_compile_forward_enabled")
            or getattr(feed_forward, "_compile_forward_enabled")
            or not hasattr(feed_forward, "_compile_forward_fatal")
            or not hasattr(feed_forward, "_compiled_forward")
        ):
            return None

        hidden_size = to_qkv.in_features
        if (
            to_qkv.out_features != 3 * to_out.in_features
            or to_out.in_features % dim_head != 0
            or to_out.out_features != hidden_size
            or w1.in_features != hidden_size
            or w1.out_features != 2 * w2.in_features
            or w2.out_features != hidden_size
            or attention.norm_q.normalized_shape != (dim_head,)
            or attention.norm_k.normalized_shape != (dim_head,)
            or block.norm1.normalized_shape != (hidden_size,)
            or block.norm2.normalized_shape != (hidden_size,)
            or scale1.shape != (hidden_size,)
            or scale2.shape != (hidden_size,)
        ):
            return None
        linears.extend(candidates)
    return tuple(linears)


def install_h3_vae_optimizations(
    decoder: nn.Module | None,
    *,
    device: torch.device,
) -> bool:
    """Install decoder operators for the selected backend once."""
    if decoder is None:
        return False
    if device.type == "npu":
        return _install_npu_operators(decoder) if is_a5(device) else False
    if getattr(decoder, "_omni_h3_vae_optimizations_installed", False):
        return True
    operators = resolve_h3_vae_operators(device)
    if operators is None:
        return False

    linears = _decoder_block_linears(decoder)
    if linears is None or any(linear.weight.dtype != torch.float32 for linear in linears):
        return False

    # The H3 decode path always uses FP16 CUDA autocast. Persisting these
    # rounded decoder-block weights avoids rebuilding the same casts per tile.
    for linear in linears:
        linear.to(dtype=torch.float16)

    for block in decoder.transformer_blocks:
        block.ff._omni_silu_and_mul = SiluAndMul()
        block.ff.forward = MethodType(_optimized_feed_forward, block.ff)
        block.attn._omni_qk_norm_rope = operators.qk_norm_rope
        block.attn.forward = MethodType(_optimized_attention, block.attn)
        block._omni_scaled_residual = operators.scaled_residual
        block.forward = MethodType(_optimized_transformer_block, block)

    decoder._omni_h3_vae_optimizations_installed = True
    return True


def _npu_rms_norm(self: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    gamma = self.weight if self.weight is not None else self._omni_unit_gamma
    result = try_rms_norm_npu(x, gamma, self.eps, unit_gamma=self.weight is None)
    return type(self).forward(self, x) if result is None else result


def _npu_rotary(
    native: Callable[[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    x: torch.Tensor,
    rotary: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    result = try_rotary_npu(x, rotary)
    return native(x, rotary) if result is None else result


def _npu_swiglu_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return self.w2(self._omni_silu_and_mul(self.w1(x)))


def _install_npu_operators(decoder: nn.Module) -> bool:
    """Install on the native decoder before quantization and residency snapshots."""
    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError("MiniMax H3 VAE NPU operators require native decoder transformer blocks")
    if getattr(decoder, "_omni_h3_vae_npu_operators_installed", False):
        return True

    rotary_forwards: dict[FunctionType, FunctionType] = {}
    for block in blocks:
        attn = getattr(block, "attn", None)
        for owner, names in ((block, ("norm1", "norm2")), (attn, ("norm_q", "norm_k"))):
            if not isinstance(owner, nn.Module):
                continue
            for name in names:
                norm = getattr(owner, name, None)
                if (
                    type(norm) is not nn.RMSNorm
                    or len(norm.normalized_shape) != 1
                    or norm.eps is None
                    or "forward" in vars(norm)
                ):
                    continue
                if norm.weight is None:
                    reference = next(owner.parameters(), None)
                    device = reference.device if reference is not None else torch.device("cpu")
                    norm.register_buffer(
                        "_omni_unit_gamma",
                        torch.ones(norm.normalized_shape, device=device, dtype=torch.float32),
                        persistent=False,
                    )
                norm.forward = MethodType(_npu_rms_norm, norm)

        forward = getattr(getattr(attn, "forward", None), "__func__", None)
        if (
            isinstance(forward, FunctionType)
            and forward is type(attn).forward
            and "apply_rotary_pos_emb" in forward.__code__.co_names
        ):
            if forward not in rotary_forwards:
                native = forward.__globals__.get("apply_rotary_pos_emb")
                if callable(native):
                    # Bind the NPU RoPE to these decoder instances. Keep the
                    # checkpoint's attention, masks, and collectives intact.
                    namespace = dict(forward.__globals__, apply_rotary_pos_emb=partial(_npu_rotary, native))
                    patched = FunctionType(
                        forward.__code__, namespace, forward.__name__, forward.__defaults__, forward.__closure__
                    )
                    patched.__kwdefaults__ = forward.__kwdefaults__
                    rotary_forwards[forward] = patched
            if forward in rotary_forwards:
                attn.forward = MethodType(rotary_forwards[forward], attn)

        ff = getattr(block, "ff", None)
        if (
            isinstance(ff, nn.Module)
            and getattr(ff, "use_gated", False)
            and type(getattr(ff, "act_fn", None)) is nn.SiLU
            and callable(getattr(ff, "_forward_impl", None))
            and not getattr(ff, "_compile_forward_enabled", False)
            and getattr(ff, "_compiled_forward", None) is None
            and "_forward_impl" not in vars(ff)
        ):
            ff._omni_silu_and_mul = SiluAndMul()
            ff._forward_impl = MethodType(_npu_swiglu_forward, ff)

    decoder._omni_h3_vae_npu_operators_installed = True
    return True


__all__ = ["install_h3_vae_optimizations"]
