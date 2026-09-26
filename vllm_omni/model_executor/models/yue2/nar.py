# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Acoustic flow matching for YuE2-3B, ported from upstream ``nar.py``.

One AR prefill per original chunk, then 32 midpoint ODE steps that walk only
the NAR path. The AR prefill re-uses the vLLM backbone's own projections
(fused ``qkv_proj`` split by head counts, ``q_norm``/``k_norm``, the layer's
``o_proj``/``mlp``), with RoPE and SDPA computed here exactly like the
reference: NAR positions attend the full AR prefix plus all NAR positions
bidirectionally, which a per-chunk prefill + concatenated K/V reproduces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import torch
import torch.nn.functional as F

from .constants import CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END


def chunk_ranges(frames: int, prefix_tokens: int, context: int = CONTEXT):
    size = min((context - prefix_tokens - 3) // 2, CONTEXT)
    if frames < 1 or size < 1:
        raise ValueError("Empty codec or prefix leaves no acoustic context")
    return [(a, min(a + size, frames)) for a in range(0, frames, size)]


@dataclass
class Chunk:
    ar_tokens: list[int]
    noise: torch.Tensor


def song_chunks(prefix: Sequence[int], codec: Sequence[int], seed: int, context: int = CONTEXT):
    """Draw the whole-song noise once, then cut it at the historical boundaries."""
    prefix = [int(v) for v in prefix]
    codec = [int(v) for v in codec]
    if min(prefix) < 0 or not codec or min(codec) < 0 or max(codec) >= CODEC_SIZE:
        raise ValueError("Token IDs are outside their allowed vocabulary")
    ranges = chunk_ranges(len(codec), len(prefix), int(context))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=generator)
    return [
        Chunk(
            prefix + [value + CODEC_OFFSET for value in codec[a:b]] + [MUSIC_END],
            noise[a:b],
        )
        for a, b in ranges
    ]


def _linear(module, x):
    out = module(x)
    return out[0] if isinstance(out, tuple) else out


def _attention(q, k, v, *, causal: bool, query_chunk_size: int | None = None):
    """SDPA over [tokens, heads, dim]; query tiling only bounds temporaries."""
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("Expected Q/K/V [tokens, heads, dim] with matching K/V")
    if min(q.shape) < 1 or min(k.shape) < 1 or q.shape[1] % k.shape[1]:
        raise ValueError("Invalid attention lengths or grouped-query head count")
    if causal and len(q) != len(k):
        raise ValueError("Causal prefill requires matching Q/K sequence lengths")
    block = query_chunk_size or (len(q) if q.device.type == "cuda" else 256)
    query = q.transpose(0, 1).unsqueeze(0)
    key = k.transpose(0, 1).unsqueeze(0)
    value = v.transpose(0, 1).unsqueeze(0)
    grouped = query.shape[1] != key.shape[1]
    outputs = []
    for start in range(0, len(q), block):
        end = min(start + block, len(q))
        used_key = key[..., :end, :] if causal else key
        used_value = value[..., :end, :] if causal else value
        mask = None
        if causal and start:
            mask = torch.arange(end, device=q.device)[None, :] <= torch.arange(start, end, device=q.device)[:, None]
        outputs.append(
            F.scaled_dot_product_attention(
                query[..., start:end, :],
                used_key,
                used_value,
                attn_mask=mask,
                is_causal=causal and start == 0,
                enable_gqa=grouped,
            )
        )
    return torch.cat(outputs, dim=-2)[0].transpose(0, 1)


class CachedNAR:
    """One acoustic chunk: AR prefix K/V prefilled once, ODE walks the NAR path."""

    def __init__(self, model, chunk: Chunk, query_chunk_size: int | None = None):
        self.model = model
        self.chunk = chunk
        self.query_chunk_size = query_chunk_size
        weight = next(model.vae2llm.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        if chunk.noise.ndim != 2 or chunk.noise.shape[1] != 64 or len(chunk.noise) < 1:
            raise ValueError("Expected nonempty acoustic noise [frames,64]")
        self.ar_length, self.nar_length = len(chunk.ar_tokens), len(chunk.noise) + 2
        if self.ar_length < 1 or max(chunk.ar_tokens) >= model.vocab_size:
            raise ValueError("AR prefix is empty or outside the model vocabulary")
        if self.ar_length + self.nar_length > model.max_position_embeddings:
            raise ValueError("Acoustic chunk exceeds the model context")
        positions = torch.arange(self.ar_length, self.ar_length + self.nar_length, device=self.device)[None]
        self.cos, self.sin = model.rotary(positions)
        local = torch.arange(self.nar_length, device=self.device).clamp(max=model.max_latent_frames - 1)
        self.pos_emb = model.latent_pos_embed(local)[None]
        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._prefill()

    def _prefill(self):
        backbone = self.model.model
        ids = torch.tensor([self.chunk.ar_tokens], dtype=torch.long, device=self.device)
        positions = torch.arange(self.ar_length, device=self.device)[None]
        cos, sin = self.model.rotary(positions)
        x = backbone.embed_tokens(ids)[0]  # [T, H]
        for layer in backbone.layers:
            # Upstream: layer.self_attn.project_qkv(layer.input_layernorm(x), ...)
            # The layernorm is load-bearing: without it the cached AR K/V
            # conditioning is garbage and the ODE latents blow up (clipped audio).
            q, k, v = self.model.ar_project(layer, layer.input_layernorm(x), cos, sin)
            self.cache.append((k, v))
            h = _attention(q, k, v, causal=True, query_chunk_size=self.query_chunk_size)
            x = x + _linear(layer.self_attn.o_proj, h.flatten(1))
            x = x + _linear(layer.mlp, layer.post_attention_layernorm(x))

    @torch.inference_mode()
    def velocity(self, state: torch.Tensor, raw_t: float) -> torch.Tensor:
        model = self.model
        x_nar = F.pad(state, (0, 0, 1, 1))
        shifted = model.shift_t(raw_t, self.device, self.dtype)
        x = model.vae2llm(x_nar[None])
        x = x + model.time_embedder(shifted.expand(self.nar_length))[None]
        x = x + self.pos_emb
        for layer, (ar_k, ar_v) in zip(model.nar_layers, self.cache):
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), self.cos, self.sin)
            k, v = torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))
            h = _attention(q[0], k, v, causal=False, query_chunk_size=self.query_chunk_size)
            x = x + _linear(layer.self_attn.o_proj, h.flatten(1)[None])
            x = x + layer.mlp(layer.pre_mlp_layernorm(x))
        return model.llm2vae(model.model.norm(x))[0, 1:-1]

    @torch.inference_mode()
    def solve(self, steps: int = 32) -> torch.Tensor:
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("steps must be a positive integer")
        state = self.chunk.noise.to(device=self.device, dtype=self.dtype)
        dt = 1.0 / steps
        for step in range(steps):
            t = 1.0 - step * dt
            raw = torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            first = self.velocity(state, raw)
            mid = state - first * (dt / 2)
            raw_mid = torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            state = state - self.velocity(mid, raw_mid) * dt
        result = state.float().cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result

    def close(self):
        self.cache.clear()
        self.cos = self.sin = self.pos_emb = None


@torch.inference_mode()
def synthesize(model, prefix: Sequence[int], codec: Sequence[int], seed: int, steps: int = 32) -> torch.Tensor:
    """CPU FP32 latents [frames,64] for the whole song, chunk by chunk."""
    if model.training:
        raise ValueError("synthesize requires model.eval()")
    output = []
    for chunk in song_chunks(prefix, codec, seed):
        engine = CachedNAR(model, chunk)
        try:
            output.append(engine.solve(steps))
        finally:
            engine.close()
        del engine
    return torch.cat(output, dim=0)


__all__ = ["CachedNAR", "chunk_ranges", "song_chunks", "synthesize"]
