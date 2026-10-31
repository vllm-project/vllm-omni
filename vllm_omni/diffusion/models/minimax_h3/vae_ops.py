# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in Omni decoder operators for the checkpoint-owned H3 Video VAE.

Install before INT8 conversion and residency snapshots. Linear attributes are
resolved at forward time so the existing online INT8 adapter remains in use.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MethodType

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def _check_uncompiled(module: nn.Module, path: str) -> None:
    if (
        hasattr(module, "_orig_mod")
        or getattr(module, "_compiled_call_impl", None) is not None
        or getattr(module, "_compiled_forward", None) is not None
        or getattr(module, "_compile_forward_enabled", False)
    ):
        raise ValueError(f"{path}: install H3 VAE Omni ops before compiling the decoder")


def _check_rms_norm(module: nn.Module | None, path: str) -> None:
    if not isinstance(module, nn.RMSNorm) or module.weight is None or module.eps is None:
        return
    if (
        module.weight.ndim != 1
        or tuple(module.normalized_shape) != tuple(module.weight.shape)
        or module.weight.is_meta
        or not module.weight.is_floating_point()
    ):
        raise ValueError(f"{path}: H3 VAE Omni ops require materialized, floating 1D RMSNorm weights")


def _replace_rms_norm(parent: nn.Module, name: str) -> bool:
    """Keep the checkpoint key and preserve non-affine/dynamic-epsilon norms."""
    old = getattr(parent, name, None)
    _check_rms_norm(old, name)
    if not isinstance(old, nn.RMSNorm) or old.weight is None or old.eps is None:
        return False
    with torch.device(old.weight.device):
        new = RMSNorm(old.weight.numel(), eps=float(old.eps), dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight)
    new.weight.requires_grad_(old.weight.requires_grad)
    new.train(old.training)
    setattr(parent, name, new)
    return True


def _apply_h3_omni_rope(
    rope: RotaryEmbedding,
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply complete NeoX coefficients to the rotary prefix, retaining batch.

    H3 normally rotates 48 of 64 head dimensions. Neither width is hardcoded.
    Generic Omni native RoPE assumes shared batch positions, so use the H3
    formula here when MindIE is unavailable or outside the initial B=1 scope.
    """
    if tensor.ndim != 4:
        raise ValueError("H3 RoPE input must be [B, S, H, D]")
    if cos.shape != sin.shape:
        raise ValueError(f"H3 RoPE cos/sin shapes differ: {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if cos.ndim not in (3, 4):
        raise ValueError("H3 RoPE cos/sin must be [B, S, R] or [B, S, 1, R]")
    if cos.ndim == 4 and cos.shape[2] != 1:
        raise ValueError("H3 RoPE head axis must be singleton")
    if cos.shape[0] not in (1, tensor.shape[0]) or cos.shape[1] != tensor.shape[1]:
        raise ValueError("H3 RoPE batch must broadcast to the input and sequence lengths must match")
    rotary_dim = cos.shape[-1]
    if not 0 < rotary_dim <= tensor.shape[-1] or rotary_dim % 2:
        raise ValueError("H3 RoPE rotary_dim must be positive, even, and no larger than head_dim")
    if not all(value.is_floating_point() for value in (tensor, cos, sin)):
        raise ValueError("H3 RoPE input and cos/sin must be floating tensors")
    if cos.device != tensor.device or sin.device != tensor.device:
        raise ValueError("H3 RoPE input and cos/sin must be on the same device")
    if cos.ndim == 3:
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    cos, sin = cos.to(dtype=tensor.dtype), sin.to(dtype=tensor.dtype)
    rotary, suffix = tensor[..., :rotary_dim], tensor[..., rotary_dim:]

    if tensor.device.type != "npu":
        reason = "non_npu_tensor"
    elif not current_omni_platform.is_npu():
        reason = "non_npu_platform"
    elif not rope.has_mindie:
        reason = "no_mindie"
    elif tensor.shape[0] != 1 or cos.shape[0] != 1:
        reason = "batch_policy"
    elif tensor.numel() == 0:
        reason = "empty_input"
    elif tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        reason = "dtype"
    else:
        reason = "mindie"

    if reason == "mindie":
        # Q/K chunks and their partial rotary prefixes can have non-dense
        # strides. Normalize only this kernel boundary, not the native path.
        rotated = rope(rotary.contiguous(), cos.contiguous(), sin.contiguous())
    else:
        first, second = rotary.chunk(2, dim=-1)
        rotated = rotary * cos + torch.cat((-second, first), dim=-1) * sin
    logger.info_once(
        "MiniMax H3 VAE RoPE: backend=%s, reason=%s",
        "mindie" if reason == "mindie" else "h3_native",
        reason,
    )
    return torch.cat((rotated, suffix), dim=-1)


def _patched_attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    pack_info: dict[str, object] | None = None,
) -> torch.Tensor:
    """Mirror remote Attention, redirecting only its two RoPE calls."""
    if pack_info is None:
        pack_info = {}
    batch_size, seq_len, _ = hidden_states.shape
    qkv = self.to_qkv(hidden_states)
    qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
    query, key, value = torch.chunk(qkv, 3, dim=-1)

    if self.spatial_parallel:
        group = self._minimax_h3_omni_get_parallel_state()["sp_process_group"]
        query = self._minimax_h3_omni_all_to_all_4d(query, 2, 1, group=group)
        key = self._minimax_h3_omni_all_to_all_4d(key, 2, 1, group=group)
        value = self._minimax_h3_omni_all_to_all_4d(value, 2, 1, group=group)
    if self.norm_q is not None:
        query = self.norm_q(self._minimax_h3_omni_norm_input(self.norm_q, query)).to(query.dtype)
    if self.norm_k is not None:
        key = self.norm_k(self._minimax_h3_omni_norm_input(self.norm_k, key)).to(key.dtype)
    if rotary_pos_emb is not None:
        cos, sin = rotary_pos_emb
        query = _apply_h3_omni_rope(self.omni_rope, query, cos, sin)
        key = _apply_h3_omni_rope(self.omni_rope, key, cos, sin)
    hidden_states = self.perform_attention(query, key, value, pack_info)
    if self.spatial_parallel:
        hidden_states = self._minimax_h3_omni_all_to_all_4d(hidden_states, 1, 2, group=group)
    hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
    return self.to_out(hidden_states)


def _attention_helpers(attn: nn.Module, path: str) -> tuple[Callable, Callable, Callable]:
    forward = attn.forward
    if not isinstance(forward, MethodType):
        raise RuntimeError(f"{path}: unsupported H3 VAE attention forward; expected a bound Python method")
    namespace = forward.__func__.__globals__
    names = ("all_to_all_4D", "get_parallel_state", "_vit_norm_input")
    helpers = tuple(namespace.get(name) for name in names)
    if not all(callable(helper) for helper in helpers):
        raise RuntimeError(f"{path}: unsupported H3 VAE attention remote code; required helper is missing")
    return helpers


def _patch_attention(attn: nn.Module) -> None:
    if getattr(attn, "_minimax_h3_omni_ops_patched", False):
        return
    all_to_all, parallel_state, norm_input = _attention_helpers(attn, "attention")
    attn.omni_rope = RotaryEmbedding(is_neox_style=True, half_head_dim=False)
    attn.omni_rope.train(attn.training)
    attn._minimax_h3_omni_all_to_all_4d = all_to_all
    attn._minimax_h3_omni_get_parallel_state = parallel_state
    attn._minimax_h3_omni_norm_input = norm_input
    attn.forward = MethodType(_patched_attention_forward, attn)
    attn._minimax_h3_omni_ops_patched = True


def patch_minimax_h3_video_vae(model: nn.Module) -> None:
    """Install idempotently, validating all blocks before changing any module."""
    decoder = getattr(model, "decoder", None)
    blocks = getattr(decoder, "transformer_blocks", None)
    if not isinstance(decoder, nn.Module) or not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError("H3 VAE Omni ops require a nonempty decoder.transformer_blocks ModuleList")
    _check_uncompiled(model, "video_vae")
    _check_uncompiled(decoder, "video_vae.decoder")
    for index, block in enumerate(blocks):
        path = f"video_vae.decoder.transformer_blocks.{index}"
        _check_uncompiled(block, path)
        attn = getattr(block, "attn", None)
        if not isinstance(attn, nn.Module):
            raise ValueError(f"{path}: missing attention module")
        _check_uncompiled(attn, f"{path}.attn")
        if (
            not isinstance(getattr(attn, "dim_head", None), int)
            or attn.dim_head <= 0
            or not hasattr(attn, "spatial_parallel")
            or not callable(getattr(attn, "perform_attention", None))
            or not isinstance(getattr(attn, "to_qkv", None), nn.Module)
            or not isinstance(getattr(attn, "to_out", None), nn.Module)
        ):
            raise ValueError(f"{path}.attn: unsupported H3 VAE attention structure")
        for owner, names in ((block, ("norm1", "norm2")), (attn, ("norm_q", "norm_k"))):
            for name in names:
                norm = getattr(owner, name, None)
                if not hasattr(owner, name) or (norm is None and owner is block):
                    raise ValueError(f"{path}.{name}: missing H3 VAE norm")
                if norm is not None:
                    if not isinstance(norm, nn.Module):
                        raise ValueError(f"{path}.{name}: expected a norm module or None")
                    _check_uncompiled(norm, f"{path}.{name}")
                    _check_rms_norm(norm, f"{path}.{name}")
        if not getattr(attn, "_minimax_h3_omni_ops_patched", False):
            _attention_helpers(attn, f"{path}.attn")

    replaced = retained = patched = 0
    for block in blocks:
        attn = block.attn
        for owner, names in ((block, ("norm1", "norm2")), (attn, ("norm_q", "norm_k"))):
            for name in names:
                if _replace_rms_norm(owner, name):
                    replaced += 1
                else:
                    retained += 1
        if not getattr(attn, "_minimax_h3_omni_ops_patched", False):
            _patch_attention(attn)
            patched += 1
    logger.info(
        "Installed MiniMax H3 VAE Omni ops: blocks=%d, replaced_norms=%d, retained_norms=%d, patched_attentions=%d",
        len(blocks),
        replaced,
        retained,
        patched,
    )


__all__ = ["patch_minimax_h3_video_vae"]
