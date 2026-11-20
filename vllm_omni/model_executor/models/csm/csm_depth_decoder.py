# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CSM depth decoder: completes one audio frame (codebooks 1..N-1) for a batch of requests.

Position 0 is the backbone hidden state and position ``k + 1`` is the embedding
of codebook ``k``. Parameter names follow transformers' ``depth_decoder.*``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

_UNIFORM_EPS = 1e-20


def apply_sampling_filters(
    scores: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Temperature, top-k, then top-p, matching transformers' logits warpers (top-k keeps ties)."""
    if temperature != 1.0:
        scores = scores / temperature
    if top_k > 0:
        kth = torch.topk(scores, min(top_k, scores.shape[-1]))[0][..., -1, None]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=False)
        cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative <= (1.0 - top_p)
        remove[..., -1:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        scores = scores.masked_fill(remove, float("-inf"))
    return scores


def sample_gumbel(scores: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    """Draw from ``softmax(scores)`` by Gumbel-max using precomputed uniforms."""
    return (scores.float() - torch.log(-torch.log(uniforms))).argmax(dim=-1)


def draw_frame_uniforms(
    batch: int,
    steps: int,
    vocab_size: int,
    *,
    device: torch.device | str,
    generators: Sequence[torch.Generator | None] | None = None,
) -> torch.Tensor:
    """Uniform noise for one frame, ``[batch, steps, vocab_size]``.

    Each row draws from its own generator when given one, so seeded rows do
    not depend on the rest of the batch.
    """
    uniforms = torch.empty((batch, steps, vocab_size), dtype=torch.float32, device=device)
    if generators is None or all(generator is None for generator in generators):
        return uniforms.uniform_(_UNIFORM_EPS, 1.0 - _UNIFORM_EPS)
    if len(generators) != batch:
        raise ValueError(f"expected {batch} generators, got {len(generators)}")
    for row, generator in enumerate(generators):
        uniforms[row : row + 1].uniform_(_UNIFORM_EPS, 1.0 - _UNIFORM_EPS, generator=generator)
    return uniforms


def _rope_inverse_frequencies(config: Any, head_dim: int) -> tuple[torch.Tensor, float]:
    """Inverse frequencies and cos/sin scale for the config's RoPE type (``llama3`` in sesame/csm-1b)."""
    rope = dict(getattr(config, "rope_parameters", None) or {})
    rope_type = rope.get("rope_type", "default")
    if rope_type == "default":
        theta = float(rope.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
        return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)), 1.0
    if rope_type == "dynamic":
        raise NotImplementedError("CSM depth decoder does not support dynamic RoPE")
    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, None)
    return inv_freq, float(attention_scaling)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class CsmRMSNorm(nn.Module):
    """RMS norm computed in float32."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize ``hidden_states`` over the last dimension."""
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(dtype)


class CsmDepthDecoderAttention(nn.Module):
    """Grouped-query attention over a fixed-length, frame-local key/value buffer."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // self.num_heads)
        self.scaling = self.head_dim**-0.5
        bias = bool(getattr(config, "attention_bias", False))
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attend the new positions; ``mask`` hides later slots, so stale cache entries are never read."""
        bsz, num_new, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(bsz, num_new, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(bsz, num_new, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(bsz, num_new, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        k_cache.index_copy_(2, positions, key.to(k_cache.dtype))
        v_cache.index_copy_(2, positions, value.to(v_cache.dtype))

        # Broadcast each key/value head over its query heads instead of repeating it.
        groups = self.num_heads // self.num_kv_heads
        query = query.reshape(bsz, self.num_kv_heads, groups, num_new, self.head_dim)
        weights = torch.matmul(query, k_cache.unsqueeze(2).transpose(-1, -2)) * self.scaling
        weights = F.softmax(weights + mask, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(weights, v_cache.unsqueeze(2))
        output = output.reshape(bsz, self.num_heads, num_new, self.head_dim).transpose(1, 2)
        return self.o_proj(output.reshape(bsz, num_new, -1))


class CsmDepthDecoderMLP(nn.Module):
    """Gated feed-forward block."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        bias = bool(getattr(config, "mlp_bias", False))
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward block."""
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class CsmDepthDecoderLayer(nn.Module):
    """Pre-norm decoder layer: attention, then feed-forward."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.self_attn = CsmDepthDecoderAttention(config)
        self.mlp = CsmDepthDecoderMLP(config)
        self.input_layernorm = CsmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CsmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the layer on the new positions."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, k_cache, v_cache, positions, mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class CsmDepthDecoderModel(nn.Module):
    """Embedding table, layers and final norm of the depth decoder."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        # One table for all codebooks: codebook k uses rows [k * vocab_size, (k + 1) * vocab_size).
        self.embed_tokens = nn.Embedding(config.num_codebooks * config.vocab_size, config.backbone_hidden_size)
        self.layers = nn.ModuleList([CsmDepthDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = CsmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.inputs_embeds_projector = nn.Linear(config.backbone_hidden_size, config.hidden_size, bias=False)


class CsmCodebooksHead(nn.Module):
    """One output projection per predicted codebook: ``weight[k]`` scores codebook ``k + 1``."""

    def __init__(self, hidden_size: int, num_codebooks: int, vocab_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks - 1, hidden_size, vocab_size))


class CsmDepthDecoder(nn.Module):
    """Batched depth decoder that completes audio frames."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.num_codebooks = int(config.num_codebooks)
        self.vocab_size = int(config.vocab_size)
        self.num_steps = self.num_codebooks - 1
        self.max_positions = self.num_codebooks
        if int(config.max_position_embeddings) < self.max_positions:
            raise ValueError(
                f"depth decoder needs {self.max_positions} positions, config allows {config.max_position_embeddings}"
            )
        self.model = CsmDepthDecoderModel(config)
        self.codebooks_head = CsmCodebooksHead(config.hidden_size, self.num_codebooks, self.vocab_size)

        inv_freq, attention_scaling = _rope_inverse_frequencies(config, self.model.layers[0].self_attn.head_dim)
        freqs = torch.arange(self.max_positions, dtype=torch.float)[:, None] * inv_freq
        angles = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", angles.cos() * attention_scaling, persistent=False)
        self.register_buffer("rope_sin", angles.sin() * attention_scaling, persistent=False)
        causal = torch.full((self.max_positions, self.max_positions), float("-inf")).triu(diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)
        self.register_buffer("positions", torch.arange(self.max_positions), persistent=False)
        self.register_buffer("codebook_offsets", torch.arange(self.num_codebooks) * self.vocab_size, persistent=False)
        # [layers, 2 (k/v), batch capacity, kv_heads, max_positions, head_dim]; overwritten every frame.
        self._kv_scratch: torch.Tensor | None = None

    def reserve(self, max_batch: int) -> None:
        """Allocate the key/value scratch for up to ``max_batch`` rows ahead of the first frame."""
        weight = self.model.inputs_embeds_projector.weight
        self._scratch(max_batch, weight.device, weight.dtype)

    def _scratch(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scratch = self._kv_scratch
        if scratch is None or scratch.shape[2] < batch or scratch.device != device or scratch.dtype != dtype:
            attn = self.model.layers[0].self_attn
            scratch = torch.zeros(
                (len(self.model.layers), 2, batch, attn.num_kv_heads, self.max_positions, attn.head_dim),
                device=device,
                dtype=dtype,
            )
            self._kv_scratch = scratch
        return scratch

    def embed_frames(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum of the per-codebook embeddings of ``codes`` (``[..., num_codebooks]``)."""
        return self.model.embed_tokens(codes + self.codebook_offsets).sum(dim=-2)

    def _forward_positions(
        self,
        inputs_embeds: torch.Tensor,
        start: int,
        scratch: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_new, _ = inputs_embeds.shape
        stop = start + num_new
        hidden_states = self.model.inputs_embeds_projector(inputs_embeds)
        positions = self.positions[start:stop]
        mask = self.causal_mask[start:stop]
        for index, layer in enumerate(self.model.layers):
            hidden_states = layer(
                hidden_states,
                cos[start:stop],
                sin[start:stop],
                scratch[index, 0, :bsz],
                scratch[index, 1, :bsz],
                positions,
                mask,
            )
        return self.model.norm(hidden_states)

    def generate_frame(
        self,
        backbone_hidden: torch.Tensor,
        first_codes: torch.Tensor,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        uniforms: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Complete one frame for every row.

        Args:
            backbone_hidden: ``[B, backbone_hidden]``, the backbone output after its final norm.
            first_codes: ``[B]`` codebook-0 ids in ``[0, vocab_size)``.
            uniforms: ``[B, num_codebooks - 1, vocab_size]`` noise from :func:`draw_frame_uniforms`.

        Returns:
            ``codes`` ``[B, num_codebooks]`` and ``frame_embeds`` ``[B, backbone_hidden]``,
            the backbone's next input.
        """
        bsz = first_codes.shape[0]
        table = self.model.embed_tokens.weight
        dtype = self.model.inputs_embeds_projector.weight.dtype
        scratch = self._scratch(bsz, table.device, dtype)
        cos = self.rope_cos.to(dtype)
        sin = self.rope_sin.to(dtype)
        if do_sample and uniforms is None:
            uniforms = draw_frame_uniforms(bsz, self.num_steps, self.vocab_size, device=table.device)

        codes = first_codes.new_empty((bsz, self.num_codebooks))
        codes[:, 0] = first_codes
        first_embeds = F.embedding(first_codes, table)
        inputs = torch.stack((backbone_hidden.to(dtype), first_embeds.to(dtype)), dim=1)
        hidden = self._forward_positions(inputs, 0, scratch, cos, sin)[:, -1]

        for step in range(self.num_steps):
            logits = F.linear(hidden, self.codebooks_head.weight[step].T)
            if do_sample:
                scores = apply_sampling_filters(logits.float(), temperature=temperature, top_k=top_k, top_p=top_p)
                code = sample_gumbel(scores, uniforms[:, step])
            else:
                code = logits.argmax(dim=-1)
            codes[:, step + 1] = code
            if step + 1 < self.num_steps:
                embeds = F.embedding(code + (step + 1) * self.vocab_size, table)
                hidden = self._forward_positions(embeds.to(dtype)[:, None], step + 2, scratch, cos, sin)[:, -1]

        return codes, self.embed_frames(codes)
