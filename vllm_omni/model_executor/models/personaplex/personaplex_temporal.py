# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming Helium temporal transformer for PersonaPlex (moshi-free, plain torch).

Frame-clocked duplex needs a stateful per-frame forward: one 80 ms step consumes
one summed frame embedding and returns the temporal hidden + text logits, with a
sliding-window KV carried across the conversation. vLLM's engine path serves the
offline/staged flavor of this model (``modeling_helium.HeliumForCausalLM``); this
module is the REAL-TIME flavor: no engine context, a fixed-capacity ring KV, and
static shapes so the whole step is CUDA-graphable.

Semantics mirror the Moshi streaming transformer exactly (MIT; the reference for
the ``nvidia/personaplex-7b-v1`` checkpoint), op for op, so greedy decoding is
bit-comparable against recorded reference outputs:

- fp32 RMSNorm with ``alpha`` weights (norm1/norm2/out_norm)
- fused in-proj attention, interleaved (non-neox) RoPE applied at the ABSOLUTE
  offset in fp32, relative-position causal mask over a ring KV with
  ``context``-window truncation
- SiLU gating MLP with fused ``linear_in`` (gate/up) and ``linear_out``
- ``transformer_out`` is the POST-``out_norm`` hidden; ``text_logits`` is
  ``text_linear`` of that hidden

Weights load directly from the moshi checkpoint's fused layout
(``transformer.layers.{i}.self_attn.in_proj_weight`` etc.), keeping the exact
computation order of the reference.

Per-slot elastic recycle (concurrent serving) uses the same per-row
``start_offset`` masking verified for the moshi-hosted path: writes stay uniform
because every slot ticks in lockstep; ``reset_slot`` only moves that row's
valid-window start. The LM-specific "+1 sacrifice tick" bump is the CALLER's
job (see the fullduplex PersonaPlex stage0), matching the split introduced when
the +1 was found to be wrong for codec streams.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.personaplex.personaplex_depformer import _rms_norm_f32


def _apply_rope(q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor, max_period: float = 10_000.0):
    """Interleaved RoPE at absolute ``offset``, fp32 rotation (moshi ``apply_rope``)."""
    B, H, T, D = q.shape
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = offset.float() + torch.arange(T, device=q.device, dtype=torch.float32)
    ts = ts.view(1, -1, 1)

    dims = q.shape[:-1]
    q = q.view(*dims, D // 2, 2)
    k = k.view(*dims, D // 2, 2)
    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()
    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr
    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)
    return qo.view(*dims, D), ko.view(*dims, D)


class _RingKV:
    """Fixed-capacity KV ring with per-row valid-window start (elastic recycle).

    Uniform writes (all rows tick together) + a per-row mask via absolute
    positions; with ``start_offset == 0`` everywhere this is the plain streaming
    cache. All shapes static -> CUDA-graph safe.
    """

    def __init__(self, batch_size: int, num_heads: int, dim_per_head: int, capacity: int, device, dtype):
        self.capacity = capacity
        self.cache = torch.zeros((2, batch_size, num_heads, capacity, dim_per_head), device=device, dtype=dtype)
        self.end_offset = torch.zeros(1, device=device, dtype=torch.long)
        self.start_offset = torch.zeros(batch_size, device=device, dtype=torch.long)

    def reset(self) -> None:
        self.end_offset.zero_()
        self.start_offset.zero_()

    def reset_slot(self, b: int) -> None:
        # Mask everything written so far for row b; the row's next write is its
        # first visible entry. (LM sacrifice-tick +1 is applied by the caller.)
        self.start_offset[b] = self.end_offset.clone()

    def bump_slot_start(self, b: int) -> None:
        self.start_offset[b] += 1

    def complete(self, k: torch.Tensor, v: torch.Tensor):
        B, H, T, D = k.shape
        indexes = torch.arange(T, device=self.end_offset.device, dtype=self.end_offset.dtype) + self.end_offset
        indexes = indexes % self.capacity
        self.cache[0].index_copy_(2, indexes, k)
        self.cache[1].index_copy_(2, indexes, v)
        self.end_offset.add_(T)

        idx = torch.arange(self.capacity, device=self.end_offset.device, dtype=torch.long)
        invalid = idx >= self.end_offset
        end_index = self.end_offset % self.capacity
        delta = idx - end_index
        # `delta <= 0` (not `< 0`) is moshi's exact convention (transformer.py
        # RingKVCache.complete). It labels the just-past-newest slot as the future
        # write position, so once the ring has wrapped the single oldest in-window
        # cell is excluded and the effective window is capacity-1. This is inherited
        # verbatim from the reference and only shows after the window fills (Helium
        # ~3000 frames / 240 s); it costs one frame out of thousands. Do NOT change
        # this to `< 0`: it would diverge from moshi and break greedy bit-parity.
        positions = torch.where(delta <= 0, self.end_offset + delta, self.end_offset + delta - self.capacity)
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)
        positions = positions.view(1, -1)  # [1, capacity]
        below = positions < self.start_offset.view(-1, 1)  # [B, capacity]
        positions = torch.where(below, torch.full_like(positions, -1), positions)
        return self.cache[0], self.cache[1], positions


class _TemporalLayer(nn.Module):
    """One Helium decoder layer in the checkpoint's fused layout."""

    def __init__(self, dim: int, num_heads: int, hidden: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.out_proj_weight = nn.Parameter(torch.empty(dim, dim))
        self.gating_in = nn.Parameter(torch.empty(2 * hidden, dim))
        self.gating_out = nn.Parameter(torch.empty(dim, hidden))
        self.norm1_alpha = nn.Parameter(torch.ones(1, 1, dim))
        self.norm2_alpha = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x: torch.Tensor, kv: _RingKV, offset: torch.Tensor, context: int) -> torch.Tensor:
        B, T, _ = x.shape
        h = _rms_norm_f32(x, self.norm1_alpha, 1e-8)
        qkv = F.linear(h, self.in_proj_weight)
        # Layout mirrors moshi's einops "b t (p h d) -> p b h t d" exactly, so the
        # downstream kernels see identical strides (bit-level replay agreement).
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _apply_rope(q, k, offset)

        keys, values, pos_k = kv.complete(k, v)
        pos_k = pos_k.view(pos_k.shape[0], 1, pos_k.shape[1])  # [B, 1, cap]
        pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(1, -1, 1)
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0) & (delta < context)
        attn_bias = attn_bias.unsqueeze(1)  # [B, 1, T, cap]
        attn = F.scaled_dot_product_attention(q, keys, values, attn_bias, dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(B, T, self.dim)
        x = x + F.linear(attn, self.out_proj_weight)

        h = _rms_norm_f32(x, self.norm2_alpha, 1e-8)
        a, b = F.linear(h, self.gating_in).chunk(2, dim=-1)
        return x + F.linear(F.silu(a) * b, self.gating_out)


class PersonaPlexTemporalStreaming(nn.Module):
    """The 7B Helium temporal backbone as a stateful per-frame stepper.

    ``step(frame_embedding [B, 1, dim]) -> (transformer_out [B, 1, dim],
    text_logits [B, 1, 1, text_card])``, matching moshi's
    ``LMModel.forward_embeddings`` contract so the existing native depformer and
    input embeddings compose unchanged.
    """

    def __init__(
        self,
        dim: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        hidden: int = 11264,
        context: int = 3000,
        text_card: int = 32000,
        max_period: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.context = context
        self.max_period = max_period
        self.layers = nn.ModuleList([_TemporalLayer(dim, num_heads, hidden) for _ in range(num_layers)])
        self.out_norm_alpha = nn.Parameter(torch.ones(1, 1, dim))
        self.text_linear = nn.Parameter(torch.empty(text_card, dim))
        self._kv: list[_RingKV] | None = None
        self._offset: torch.Tensor | None = None

    # -- streaming state ----------------------------------------------------

    def streaming_init(self, batch_size: int) -> None:
        p = next(self.parameters())
        heads = self.layers[0].num_heads
        head_dim = self.layers[0].head_dim
        # Capacity = context window (the mask truncates at `context` anyway).
        self._kv = [_RingKV(batch_size, heads, head_dim, self.context, p.device, p.dtype) for _ in self.layers]
        self._offset = torch.zeros(1, device=p.device, dtype=torch.long)

    def reset_streaming(self) -> None:
        assert self._kv is not None, "call streaming_init first"
        for kv in self._kv:
            kv.reset()
        self._offset.zero_()

    def reset_slot(self, b: int) -> None:
        assert self._kv is not None
        for kv in self._kv:
            kv.reset_slot(b)

    def bump_slot_start(self, b: int) -> None:
        """LM sacrifice-tick semantics: mask the one spurious post-recycle write."""
        assert self._kv is not None
        for kv in self._kv:
            kv.bump_slot_start(b)

    # -- per-frame step -------------------------------------------------------

    @torch.no_grad()
    def step(self, frame_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._kv is not None, "call streaming_init first"
        x = frame_embedding
        for layer, kv in zip(self.layers, self._kv):
            x = layer(x, kv, self._offset, self.context)
        self._offset.add_(x.shape[1])
        out = _rms_norm_f32(x, self.out_norm_alpha, 1e-8)
        text_logits = F.linear(out, self.text_linear)
        return out, text_logits[:, None]

    # -- weights --------------------------------------------------------------

    def load_weights(self, state_dict: dict[str, torch.Tensor]) -> int:
        """Load from the moshi checkpoint layout (fused, exact tensor reuse)."""
        loaded = 0
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                base = f"transformer.layers.{i}"
                pairs = [
                    (layer.in_proj_weight, f"{base}.self_attn.in_proj_weight"),
                    (layer.out_proj_weight, f"{base}.self_attn.out_proj.weight"),
                    (layer.gating_in, f"{base}.gating.linear_in.weight"),
                    (layer.gating_out, f"{base}.gating.linear_out.weight"),
                    (layer.norm1_alpha, f"{base}.norm1.alpha"),
                    (layer.norm2_alpha, f"{base}.norm2.alpha"),
                ]
                for param, name in pairs:
                    src = state_dict[name]
                    param.data.copy_(src.reshape(param.shape).to(param.dtype))
                    loaded += 1
            self.out_norm_alpha.data.copy_(state_dict["out_norm.alpha"].reshape(self.out_norm_alpha.shape))
            self.text_linear.data.copy_(state_dict["text_linear.weight"].to(self.text_linear.dtype))
            loaded += 2
        return loaded
