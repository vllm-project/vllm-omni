# SPDX-License-Identifier: Apache-2.0
"""Local optimized decoder operators for the MiniMax H3 remote VAE.

The VAE itself remains checkpoint-owned remote code.  This module only swaps
the decoder's RMSNorm modules and binds an attention ``forward`` implementation
which applies Omni's RoPE at the same point as the remote implementation.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn as nn

from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.platforms import current_omni_platform


def _replace_rms_norm(parent: nn.Module, name: str) -> bool:
    """Replace an affine ``nn.RMSNorm`` without changing its state-dict key.

    Omni RMSNorm always has a scale parameter.  H3's Q/K norms are configured
    without one, so they must remain untouched rather than gaining a new,
    non-checkpoint parameter.
    """
    old = getattr(parent, name, None)
    if not isinstance(old, nn.RMSNorm) or old.weight is None:
        return False
    if old.weight.ndim != 1:
        raise ValueError(f"MiniMax H3 VAE only supports 1D RMSNorm weights, got {tuple(old.weight.shape)}")

    eps = old.eps if old.eps is not None else torch.finfo(torch.float32).eps
    new = RMSNorm(
        old.weight.numel(),
        eps=float(eps),
        dtype=old.weight.dtype,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight)
    new.weight.requires_grad_(old.weight.requires_grad)
    setattr(parent, name, new)
    return True


def _apply_h3_omni_rope(
    rope: RotaryEmbedding,
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply H3's partial, full-dimension 3D RoPE with an Omni operator.

    H3 uses a 48-dimensional rotary prefix in each 64-dimensional attention
    head.  Its remote ``RotaryEmbeddingND`` returns the complete rotation
    coefficients as ``[B, S, 1, 48]``.  Split the head locally so the NPU
    fused operator receives matching widths, then preserve the 16-dimensional
    non-rotary suffix exactly.
    """
    if cos.shape != sin.shape:
        raise ValueError(f"H3 RoPE cos/sin shapes differ: {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if cos.dim() not in (3, 4):
        raise ValueError(f"H3 RoPE cos/sin must be [B, S, D] or [B, S, 1, D], got {tuple(cos.shape)}")
    if cos.dim() == 4 and cos.shape[2] != 1:
        raise ValueError(f"H3 RoPE head axis must be singleton, got {tuple(cos.shape)}")

    rotary_dim = cos.shape[-1]
    if rotary_dim > tensor.shape[-1]:
        raise ValueError(f"H3 RoPE rotary_dim ({rotary_dim}) exceeds attention head_dim ({tensor.shape[-1]})")
    rotary, passthrough = tensor[..., :rotary_dim], tensor[..., rotary_dim:]

    # MindIE accepts the native H3 [B, S, 1, D] coefficients.  CUDA/native
    # Omni paths consume shared [B, S, D] coefficients instead.
    if not current_omni_platform.is_npu() and cos.dim() == 4:
        cos = cos.squeeze(2)
        sin = sin.squeeze(2)
    rotary = rope(rotary, cos.to(rotary.dtype), sin.to(rotary.dtype))
    return torch.cat((rotary, passthrough), dim=-1)


def _patched_attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    rotary_pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    pack_info: dict[str, Any] | None = None,
) -> torch.Tensor:
    """H3 remote ``Attention.forward`` with only Q/K RoPE redirected.

    This deliberately mirrors the remote-code ordering, including its spatial
    all-to-all boundaries.  The functions are saved per instance while
    patching, avoiding any mutation of the dynamic module's global helpers.
    """
    if pack_info is None:
        pack_info = {}
    batch_size, seq_len, _ = hidden_states.shape

    qkv = self.to_qkv(hidden_states)
    qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
    query, key, value = torch.chunk(qkv, 3, dim=-1)

    if self.spatial_parallel:
        local_process_group = self._minimax_h3_omni_get_parallel_state()["sp_process_group"]
        query = self._minimax_h3_omni_all_to_all_4d(query, 2, 1, group=local_process_group)
        key = self._minimax_h3_omni_all_to_all_4d(key, 2, 1, group=local_process_group)
        value = self._minimax_h3_omni_all_to_all_4d(value, 2, 1, group=local_process_group)

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
        hidden_states = self._minimax_h3_omni_all_to_all_4d(
            hidden_states,
            1,
            2,
            group=local_process_group,
        )

    hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
    return self.to_out(hidden_states)


def _patch_attention(attn: nn.Module) -> None:
    if bool(getattr(attn, "_minimax_h3_omni_ops_patched", False)):
        return

    forward_globals = attn.forward.__func__.__globals__
    try:
        all_to_all_4d = forward_globals["all_to_all_4D"]
        get_parallel_state = forward_globals["get_parallel_state"]
        norm_input = forward_globals["_vit_norm_input"]
    except KeyError as exc:
        raise RuntimeError("unsupported MiniMax H3 VAE attention remote code; required helper is missing") from exc

    # ``half_head_dim=False`` declares that H3 supplies its complete rotary
    # dimension.  RotaryEmbedding retains the full H3 [B, S, 1, D] layout.
    attn.omni_rope = RotaryEmbedding(is_neox_style=True, half_head_dim=False)
    attn._minimax_h3_omni_all_to_all_4d = all_to_all_4d
    attn._minimax_h3_omni_get_parallel_state = get_parallel_state
    attn._minimax_h3_omni_norm_input = norm_input
    attn.forward = MethodType(_patched_attention_forward, attn)
    attn._minimax_h3_omni_ops_patched = True


def patch_minimax_h3_video_vae(model: nn.Module) -> None:
    """Inject Omni RMSNorm and RoPE into a loaded MiniMax H3 video VAE.

    The operation is idempotent.  ``LayerNorm`` and non-affine RMSNorm are
    intentionally retained as remote modules because replacing either would
    change the checkpoint parameter contract.
    """
    decoder = getattr(model, "decoder", None)
    blocks = getattr(decoder, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("MiniMax H3 VAE decoder.transformer_blocks is required")

    for block in blocks:
        _replace_rms_norm(block, "norm1")
        _replace_rms_norm(block, "norm2")
        attn = getattr(block, "attn", None)
        if attn is None:
            raise ValueError("MiniMax H3 VAE transformer block is missing attention")
        _replace_rms_norm(attn, "norm_q")
        _replace_rms_norm(attn, "norm_k")
        _patch_attention(attn)


__all__ = ["patch_minimax_h3_video_vae"]
