# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RVQ depth decoder for MiniMax Music 3 (codebooks c1..c7).

Each audio frame is eight codebooks deep. The backbone's own ``lm_head``
emits ``c0``; this four-layer decoder walks ``c1..c7``, conditioning each code
on the previous ones. Its per-codebook hidden states are also the acoustic
stage's conditioning signal, so this module returns them alongside the codes.

The sequence fed to the decoder is::

    [proj(backbone_hidden), proj(embed(c0)), proj(embed(c1)), ...]

and after step ``i`` the head ``audio_heads[i - 1]`` predicts ``c_i``. The
whole thing runs over at most eight positions, so it is written as a dense
causal block rather than anything with a KV cache.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .constants import AUDIO_VOCAB_SIZE, BACKBONE_HIDDEN_SIZE, NUM_CODEBOOKS

# cuDNN's fused attention is not bit-reproducible on this hardware: two
# identical replays of the same input can differ by one bfloat16 ULP, which is
# enough to flip a sampled code and send the rest of the song somewhere else.
# The choice is pinned per call rather than through a process-global switch so
# it cannot be undone by whatever else shares the process.
_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]


class DepthDecoderBlock(nn.Module):
    """Pre-norm transformer block with fused QKV and a SwiGLU MLP."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        # Stored fused so the three projections are one matmul; the checkpoint
        # ships them split and ``load_checkpoint_state`` concatenates.
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.input_layernorm(x)
        bsz, seq, hidden = h.shape
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
        q = q.reshape(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        with sdpa_kernel(_SDPA_BACKENDS):
            h = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h.transpose(1, 2).contiguous().reshape(bsz, seq, hidden)
        x = x + self.o_proj(h)
        h = self.post_attention_layernorm(x)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x + h


class RVQDepthDecoder(nn.Module):
    """Depth-autoregressive decoder over the seven residual codebooks."""

    def __init__(
        self,
        *,
        hidden_size: int = BACKBONE_HIDDEN_SIZE,
        num_layers: int = 4,
        num_heads: int = 16,
        intermediate_size: int = 6144,
        audio_vocab_size: int = AUDIO_VOCAB_SIZE,
        num_codebooks: int = NUM_CODEBOOKS,
        max_seq_len: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.audio_vocab_size = audio_vocab_size
        self.num_codebooks = num_codebooks
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList(
            DepthDecoderBlock(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)
        )
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.audio_heads = nn.ModuleList(
            nn.Linear(hidden_size, audio_vocab_size, bias=False) for _ in range(num_codebooks - 1)
        )

    def forward(self, seq: Tensor) -> Tensor:
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = seq + self.pos_embedding(positions).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def load_checkpoint_state(self, state: dict[str, Tensor]) -> None:
        """Load the component checkpoint, failing loudly on any mismatch.

        The checkpoint keeps ``q``/``k``/``v`` split and names attention
        projections ``attn.to_*``; both are reshaped here. A silent partial
        load would produce plausible-sounding but wrong audio, so every key is
        accounted for and anything left over is an error.

        Args:
            state: Tensors from the ``rvq_depth_decoder`` component, with the
                ``audio_embeddings.weight`` entry already removed by the
                caller.

        Raises:
            ValueError: If any expected key is missing, any shape disagrees,
                or the checkpoint carries keys this module does not consume.
        """
        loaded: set[str] = set()
        missing: list[str] = []
        mismatched: list[str] = []

        def take(key: str) -> Tensor | None:
            value = state.get(key)
            if value is None:
                missing.append(key)
            return value

        def assign(key: str, target: Tensor, source: Tensor, keys: tuple[str, ...]) -> None:
            if tuple(source.shape) != tuple(target.shape):
                mismatched.append(f"{key}: checkpoint={tuple(source.shape)} model={tuple(target.shape)}")
                return
            target.data.copy_(source.to(dtype=target.dtype))
            loaded.update(keys)

        def copy_one(key: str, target: Tensor) -> None:
            source = take(key)
            if source is not None:
                assign(key, target, source, (key,))

        copy_one("projection.weight", self.projection.weight)
        copy_one("pos_embedding.weight", self.pos_embedding.weight)
        copy_one("norm.weight", self.norm.weight)
        for index, head in enumerate(self.audio_heads):
            copy_one(f"audio_heads.{index}.weight", head.weight)

        for index, layer in enumerate(self.layers):
            prefix = f"layers.{index}."
            copy_one(prefix + "input_layernorm.weight", layer.input_layernorm.weight)
            copy_one(
                prefix + "post_attention_layernorm.weight",
                layer.post_attention_layernorm.weight,
            )
            qkv_keys = tuple(prefix + f"attn.to_{name}.weight" for name in ("q", "k", "v"))
            parts = [take(key) for key in qkv_keys]
            if all(part is not None for part in parts):
                assign(
                    prefix + "attn.qkv",
                    layer.qkv_proj.weight,
                    torch.cat(parts, dim=0),
                    qkv_keys,
                )
            copy_one(prefix + "attn.to_out.weight", layer.o_proj.weight)
            copy_one(prefix + "gate_proj.weight", layer.gate_proj.weight)
            copy_one(prefix + "up_proj.weight", layer.up_proj.weight)
            copy_one(prefix + "down_proj.weight", layer.down_proj.weight)

        unexpected = sorted(set(state) - loaded)
        if missing or mismatched or unexpected:
            raise ValueError(
                "MiniMax Music 3 RVQ depth decoder failed to load: "
                f"missing={missing[:8]}, mismatched={mismatched[:8]}, "
                f"unexpected={unexpected[:8]}"
            )


__all__ = ["DepthDecoderBlock", "RVQDepthDecoder"]
