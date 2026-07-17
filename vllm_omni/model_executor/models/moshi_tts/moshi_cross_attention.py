# SPDX-License-Identifier: Apache-2.0
"""Cross-attention block and transformer layer wrapper for Moshi TTS.

Implements Self-Attn → Cross-Attn → FFN ordering matching the moshi/
reference (moshi/modules/transformer.py:StreamingTransformerLayer).

``MoshiTransformerLayer`` borrows sub-modules from the original
``LlamaDecoderLayer`` by reference and calls them in the correct order,
adding cross-attention between self-attention and the FFN.  The per-request
speaker context is injected via the ``cross_attention_src`` attribute before
each model forward call.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoshiCrossAttentionBlock(nn.Module):
    """Q from normed hidden states, K/V from projected speaker context.

    All projections are square (hidden_size → hidden_size); the speaker
    context is already projected to hidden_size by the conditioner's
    output_proj before being passed here.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, cross_src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: ``[T, H]`` or ``[B, T, H]`` — normed query input.
            cross_src:     ``[B, S, H]`` or ``[S, H]`` — speaker context.

        Returns:
            Attention output with the same shape as ``hidden_states``.
        """
        squeeze = hidden_states.dim() == 2
        if squeeze:
            hidden_states = hidden_states.unsqueeze(0)  # [1, T, H]
        if cross_src.dim() == 2:
            cross_src = cross_src.unsqueeze(0)  # [1, S, H]

        B, T, H = hidden_states.shape
        S = cross_src.shape[1]

        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(cross_src).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(cross_src).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        out = self.o_proj(out)
        return out.squeeze(0) if squeeze else out


class MoshiTransformerLayer(nn.Module):
    """Self-Attn → Cross-Attn → FFN layer wrapping a LlamaDecoderLayer.

    Sub-modules (self_attn, mlp, norms) are borrowed by reference from the
    original ``LlamaDecoderLayer`` so all vLLM-initialised weights remain
    intact.  Cross-attention and its LayerNorm are added as new parameters
    whose weights are loaded from the checkpoint.

    The speaker context is set on ``cross_attention_src`` by
    ``MoshiTTSTalkerForConditionalGeneration.forward()`` before the model
    forward pass and cleared afterwards.
    """

    def __init__(self, llama_layer: nn.Module, cross_attn: MoshiCrossAttentionBlock) -> None:
        super().__init__()
        self.input_layernorm = llama_layer.input_layernorm
        self.self_attn = llama_layer.self_attn
        self.post_attention_layernorm = llama_layer.post_attention_layernorm
        self.mlp = llama_layer.mlp

        self.cross_attn = cross_attn
        # LayerNorm (with bias) matching moshi/'s norm_cross
        self.cross_attn_layernorm = nn.LayerNorm(cross_attn.q_proj.in_features, eps=1e-5)

        # Set per forward pass by the talker
        self.cross_attention_src: torch.Tensor | None = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = hidden_states if residual is None else hidden_states + residual
        normed = self.input_layernorm(combined)
        sa_out = self.self_attn(positions=positions, hidden_states=normed, **kwargs)
        combined = combined + sa_out

        import os as _os

        _dbg = _os.environ.get("MOSHI_TTS_DBG") == "1"
        if self.cross_attention_src is not None:
            normed_cross = self.cross_attn_layernorm(combined)
            cross_out = self.cross_attn(normed_cross, self.cross_attention_src)
            if _dbg:
                print(
                    f"[MOSHI:cross_out] ctx_shape={tuple(self.cross_attention_src.shape)}"
                    f" ctx_norm={self.cross_attention_src.float().norm():.4f}"
                    f" cross_out_norm={cross_out.float().norm():.4f}"
                    f" cross_out[:4]={cross_out.reshape(-1)[:4].tolist()}",
                    flush=True,
                )
            combined = combined + cross_out
        elif _dbg:
            print("[MOSHI:cross_out] SKIPPED (cross_attention_src is None)", flush=True)

        normed_mlp = self.post_attention_layernorm(combined)
        hidden_states = self.mlp(normed_mlp)
        full_hidden = hidden_states + combined
        return full_hidden, torch.zeros_like(full_hidden)
