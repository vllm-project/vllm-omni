# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GO-1-Air policy, top-level model, and checkpoint-aligned submodules.

The submodule layout mirrors the upstream safetensors index so a
``model.load_state_dict`` call lines up directly: ``vision_model.*``,
``mlp1.*``, ``language_model.*``, ``action_model.*``, ``k_proj_layers.*``,
``v_proj_layers.*``, ``time_embedder.*``, ``freq_embedder.*``,
``state_adaptor.*``, ``action_adaptor.*``, ``final_layer.*`` are all direct
children of :class:`Go1Air`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from torch import nn
from vllm.logger import init_logger

from .config import OBS_IMAGES, OBS_STATE, OBS_TASK, Go1AirConfig

logger = init_logger(__name__)

# ----- InternLM2 shared blocks -----------------------------------------


@dataclass
class InternLM2BlockSpec:
    """Hyperparameters for one InternLM2-style transformer stack."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int | None = None
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    rope_scaling_factor: float = 2.0
    rope_scaling_type: str = "dynamic"
    max_position_embeddings: int = 32_768
    attn_implementation: str = "eager"

    def resolved_head_dim(self) -> int:
        return self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        # Number of query heads sharing one KV head.
        return self.num_attention_heads // self.num_key_value_heads


class InternLM2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)


def _build_inv_freq(head_dim: int, base: float, device: torch.device) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))


class InternLM2RotaryEmbedding(nn.Module):
    """RoPE with optional NTK-aware dynamic scaling.

    Dynamic scaling adjusts the base when the sequence exceeds the trained
    context window. For GO-1-Air's typical input length (vision + state +
    action tokens, well under 32K) this short-circuits to the static path.
    """

    def __init__(self, head_dim: int, spec: InternLM2BlockSpec) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = float(spec.rope_theta)
        self.scaling_factor = float(spec.rope_scaling_factor)
        self.scaling_type = spec.rope_scaling_type
        self.max_position_embeddings = int(spec.max_position_embeddings)
        inv_freq = _build_inv_freq(head_dim, self.base, torch.device("cpu"))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = position_ids.device
        seq_max = int(position_ids.max().item()) + 1 if position_ids.numel() > 0 else 1
        if self.scaling_type == "dynamic" and seq_max > self.max_position_embeddings:
            scale = self.scaling_factor * seq_max / self.max_position_embeddings - (self.scaling_factor - 1.0)
            new_base = self.base * (scale ** (self.head_dim / (self.head_dim - 2)))
            inv_freq = _build_inv_freq(self.head_dim, new_base, device)
        else:
            inv_freq = self.inv_freq.to(device)

        freqs = position_ids.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    b, h, t, d = x.shape
    return x.unsqueeze(2).expand(b, h, repeats, t, d).reshape(b, h * repeats, t, d)


class InternLM2Attention(nn.Module):
    """Self-attention with combined ``wqkv`` + grouped-query attention.

    The combined projection lays out per kv-group as ``num_kv_groups`` query
    heads followed by one K head and one V head, then flattens. We undo that
    layout in :meth:`forward` to obtain Q/K/V tensors compatible with
    standard scaled dot-product attention.
    """

    def __init__(self, spec: InternLM2BlockSpec) -> None:
        super().__init__()
        self.spec = spec
        self.head_dim = spec.resolved_head_dim()
        self.num_q_heads = spec.num_attention_heads
        self.num_kv_heads = spec.num_key_value_heads
        self.num_kv_groups = spec.num_kv_groups
        per_group_heads = self.num_kv_groups + 2
        self.wqkv = nn.Linear(
            spec.hidden_size,
            self.num_kv_heads * per_group_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(self.num_q_heads * self.head_dim, spec.hidden_size, bias=False)

    def split_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = hidden_states.shape
        per_group_heads = self.num_kv_groups + 2
        qkv = self.wqkv(hidden_states).view(
            bsz,
            seqlen,
            self.num_kv_heads,
            per_group_heads,
            self.head_dim,
        )
        # Q: first num_kv_groups slices per kv-group → fold groups into a single head axis.
        q = qkv[..., : self.num_kv_groups, :].reshape(
            bsz, seqlen, self.num_kv_heads * self.num_kv_groups, self.head_dim
        )
        k = qkv[..., -2, :]
        v = qkv[..., -1, :]
        # (B, H, T, D)
        return (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: InternLM2RotaryEmbedding,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self.split_qkv(hidden_states)
        cos, sin = rotary_emb(position_ids)
        q, k = apply_rope(q, k, cos, sin)
        attn_out = scaled_dot_product(
            q,
            k,
            v,
            num_kv_groups=self.num_kv_groups,
            mask=attention_mask,
            implementation=self.spec.attn_implementation,
        )
        bsz, _, seqlen, _ = q.shape
        out = attn_out.transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.wo(out), k, v


def scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_kv_groups: int,
    mask: torch.Tensor | None,
    implementation: str,
) -> torch.Tensor:
    k_full = repeat_kv(k, num_kv_groups)
    v_full = repeat_kv(v, num_kv_groups)
    if implementation == "sdpa":
        # Callers own causal/layout masking; keep SDPA and eager semantics
        # identical when no explicit additive mask is provided.
        return F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask, is_causal=False)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    probs = scores.softmax(dim=-1).to(v_full.dtype)
    return torch.matmul(probs, v_full)


class InternLM2FeedForward(nn.Module):
    """SwiGLU MLP with InternLM2 parameter naming (w1 gate, w3 up, w2 down)."""

    def __init__(self, spec: InternLM2BlockSpec) -> None:
        super().__init__()
        self.w1 = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=False)
        self.w3 = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=False)
        self.w2 = nn.Linear(spec.intermediate_size, spec.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class InternLM2Block(nn.Module):
    """One pre-norm decoder block: attention → FFN, both residual."""

    def __init__(self, spec: InternLM2BlockSpec) -> None:
        super().__init__()
        self.attention_norm = InternLM2RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.attention = InternLM2Attention(spec)
        self.ffn_norm = InternLM2RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.feed_forward = InternLM2FeedForward(spec)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_emb: InternLM2RotaryEmbedding,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_input = self.attention_norm(hidden_states)
        attn_out, k, v = self.attention(attn_input, position_ids, rotary_emb, attention_mask)
        hidden_states = hidden_states + attn_out
        ffn_input = self.ffn_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(ffn_input)
        return hidden_states, k, v


# ----- InternViT vision encoder ----------------------------------------


@dataclass
class InternViTSpec:
    image_size: int
    patch_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    qkv_bias: bool
    qk_normalization: bool
    norm_type: str

    @classmethod
    def from_config(cls, config: Go1AirConfig) -> InternViTSpec:
        return cls(
            image_size=config.image_resolution[0],
            patch_size=config.vision_patch_size,
            hidden_size=config.vision_hidden_size,
            intermediate_size=config.vision_intermediate_size,
            num_hidden_layers=config.vision_num_hidden_layers,
            num_attention_heads=config.vision_num_attention_heads,
            qkv_bias=config.vision_qkv_bias,
            qk_normalization=config.vision_qk_normalization,
            norm_type=config.vision_norm_type,
        )

    @property
    def num_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def _build_norm(spec: InternViTSpec) -> nn.Module:
    if spec.norm_type == "layer_norm":
        return nn.LayerNorm(spec.hidden_size, eps=1e-6)
    if spec.norm_type == "rms_norm":
        return nn.RMSNorm(spec.hidden_size, eps=1e-6)
    raise ValueError(f"Unsupported InternViT norm type: {spec.norm_type}")


class InternViTPatchEmbeddings(nn.Module):
    def __init__(self, spec: InternViTSpec) -> None:
        super().__init__()
        self.spec = spec
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, spec.hidden_size))
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=spec.hidden_size,
            kernel_size=spec.patch_size,
            stride=spec.patch_size,
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, spec.num_patches + 1, spec.hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch = pixel_values.shape[0]
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        cls = self.class_embedding.expand(batch, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        return tokens + self.position_embedding[:, : tokens.shape[1]].to(tokens.dtype)


class InternViTAttention(nn.Module):
    def __init__(self, spec: InternViTSpec) -> None:
        super().__init__()
        self.spec = spec
        self.num_heads = spec.num_attention_heads
        self.head_dim = spec.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(spec.hidden_size, spec.hidden_size * 3, bias=spec.qkv_bias)
        self.proj = nn.Linear(spec.hidden_size, spec.hidden_size, bias=True)
        if spec.qk_normalization:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).view(bsz, seqlen, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        if self.spec.qk_normalization:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.proj(out)


class InternViTMLP(nn.Module):
    def __init__(self, spec: InternViTSpec) -> None:
        super().__init__()
        self.fc1 = nn.Linear(spec.hidden_size, spec.intermediate_size, bias=True)
        self.fc2 = nn.Linear(spec.intermediate_size, spec.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class InternViTLayer(nn.Module):
    def __init__(self, spec: InternViTSpec) -> None:
        super().__init__()
        self.norm1 = _build_norm(spec)
        self.attn = InternViTAttention(spec)
        self.norm2 = _build_norm(spec)
        self.mlp = InternViTMLP(spec)
        self.ls1 = nn.Parameter(torch.ones(spec.hidden_size))
        self.ls2 = nn.Parameter(torch.ones(spec.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class InternViTEncoder(nn.Module):
    def __init__(self, spec: InternViTSpec) -> None:
        super().__init__()
        self.layers = nn.ModuleList([InternViTLayer(spec) for _ in range(spec.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        select_layer: int,
    ) -> torch.Tensor:
        # ``select_layer`` follows the InternVL convention: -1 means take the
        # last layer's output; positive indices select an explicit layer.
        if select_layer < 0:
            select_layer = len(self.layers) + select_layer
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if idx == select_layer:
                break
        return hidden_states


class InternVisionModel(nn.Module):
    """GO-1-Air vision tower; output keeps the ``[CLS]`` token (consumer drops it)."""

    def __init__(self, spec: InternViTSpec) -> None:
        super().__init__()
        self.spec = spec
        self.embeddings = InternViTPatchEmbeddings(spec)
        self.encoder = InternViTEncoder(spec)

    def forward(self, pixel_values: torch.Tensor, select_layer: int = -1) -> torch.Tensor:
        tokens = self.embeddings(pixel_values)
        return self.encoder(tokens, select_layer=select_layer)


def pixel_shuffle(features: torch.Tensor, scale: float, version: str = "v2") -> torch.Tensor:
    """Spatial-to-channel shuffle used to compress vision tokens into the LLM stream.

    For ``scale=0.5`` this groups every 2x2 patch block into a single token
    while quadrupling the channel dimension. Mirrors the InternVL ``ps_v2``
    layout (the input tile order is rotated before reshape so adjacent
    patches stay adjacent in memory).
    """
    if scale >= 1.0:
        return features
    bsz, num_tokens, channels = features.shape
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"pixel_shuffle expects a square token grid; got {num_tokens}.")
    new_side = int(side * scale)
    block = int(round(1.0 / scale))
    grid = features.view(bsz, side, side, channels)
    if version == "v2":
        grid = grid.permute(0, 2, 1, 3).contiguous()
    grid = grid.view(bsz, side, new_side, channels * block)
    grid = grid.permute(0, 2, 1, 3).contiguous()
    grid = grid.view(bsz, new_side, new_side, channels * (block * block))
    if version == "v2":
        grid = grid.permute(0, 2, 1, 3).contiguous()
    return grid.view(bsz, new_side * new_side, channels * (block * block))


# ----- InternLM2-GO1 language stack ------------------------------------


@dataclass
class LanguageStackOutput:
    last_hidden_state: torch.Tensor
    layer_kv: list[tuple[torch.Tensor, torch.Tensor]]


def language_block_spec(config: Go1AirConfig) -> InternLM2BlockSpec:
    return InternLM2BlockSpec(
        hidden_size=config.llm_hidden_size,
        intermediate_size=config.llm_intermediate_size,
        num_attention_heads=config.llm_num_attention_heads,
        num_key_value_heads=config.llm_num_key_value_heads,
        rms_norm_eps=config.llm_rms_norm_eps,
        rope_theta=config.llm_rope_theta,
        rope_scaling_factor=config.llm_rope_scaling_factor,
        rope_scaling_type=config.llm_rope_scaling_type,
        max_position_embeddings=config.llm_max_position_embeddings,
        attn_implementation=config.attn_implementation,
    )


class InternLM2Trunk(nn.Module):
    """The ``language_model.model.*`` subtree: embeddings + blocks + final norm."""

    def __init__(self, config: Go1AirConfig) -> None:
        super().__init__()
        self.config = config
        self.spec = language_block_spec(config)
        self.tok_embeddings = nn.Embedding(config.llm_vocab_size, config.llm_hidden_size)
        self.layers = nn.ModuleList([InternLM2Block(self.spec) for _ in range(config.llm_num_hidden_layers)])
        self.norm = InternLM2RMSNorm(config.llm_hidden_size, eps=config.llm_rms_norm_eps)
        head_dim = self.spec.resolved_head_dim()
        self.rotary_emb = InternLM2RotaryEmbedding(head_dim, self.spec)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> LanguageStackOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")
        hidden = inputs_embeds if inputs_embeds is not None else self.tok_embeddings(input_ids)
        bsz, seqlen, _ = hidden.shape
        position_ids = torch.arange(seqlen, device=hidden.device).expand(bsz, seqlen)

        # Decoder-only InternLM2 needs causal masking; combine with padding
        # mask (if provided) into a (B, 1, T, T) additive attention mask.
        neg = torch.finfo(hidden.dtype).min
        causal = torch.triu(
            torch.full((seqlen, seqlen), neg, device=hidden.device, dtype=hidden.dtype),
            diagonal=1,
        )
        attn_bias = causal.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seqlen, seqlen).contiguous()
        if attention_mask is not None and attention_mask.dim() == 2:
            invalid = (attention_mask == 0).view(bsz, 1, 1, seqlen)
            attn_bias = attn_bias.masked_fill(invalid, neg)

        layer_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.layers:
            hidden, k, v = block(hidden, position_ids, self.rotary_emb, attn_bias)
            layer_kv.append((k, v))
        hidden = self.norm(hidden)
        return LanguageStackOutput(last_hidden_state=hidden, layer_kv=layer_kv)

    def set_attention_implementation(self, implementation: str) -> None:
        self.spec.attn_implementation = implementation
        for block in self.layers:
            block.attention.spec.attn_implementation = implementation


class InternLM2GO1LanguageModel(nn.Module):
    """The full ``language_model`` subtree: trunk + vocabulary head."""

    def __init__(self, config: Go1AirConfig) -> None:
        super().__init__()
        self.config = config
        self.model = InternLM2Trunk(config)
        self.output = nn.Linear(
            config.llm_hidden_size,
            config.llm_vocab_size,
            bias=False,
        )
        if config.llm_tie_word_embeddings:
            self.output.weight = self.model.tok_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> LanguageStackOutput:
        return self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    def set_attention_implementation(self, implementation: str) -> None:
        self.model.set_attention_implementation(implementation)


# ----- GO-1-Air action expert ------------------------------------------


def action_block_spec(config: Go1AirConfig) -> InternLM2BlockSpec:
    return InternLM2BlockSpec(
        hidden_size=config.act_hidden_size,
        intermediate_size=config.act_intermediate_size,
        num_attention_heads=config.act_num_attention_heads,
        num_key_value_heads=config.act_num_key_value_heads,
        head_dim=config.act_head_dim,
        rms_norm_eps=config.act_rms_norm_eps,
        rope_theta=config.act_rope_theta,
        rope_scaling_factor=config.act_rope_scaling_factor,
        rope_scaling_type=config.act_rope_scaling_type,
        max_position_embeddings=config.act_max_position_embeddings,
        attn_implementation=config.attn_implementation,
    )


class SinusoidalScalarEmbedding(nn.Module):
    """Sinusoidal embedding for a scalar input (timestep, frequency)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Sinusoidal embedding dim must be even; got {dim}.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        result = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return result.to(t.dtype if t.dtype.is_floating_point else torch.float32)


class TimestepEmbedder(nn.Module):
    """Sinusoidal -> 2-layer MLP with parameter names ``mlp.0`` and ``mlp.2``.

    The sinusoidal embedding has a fixed 256-dim output; the MLP then lifts
    it to ``hidden_size``. This matches the upstream checkpoint shape
    ``mlp.0.weight = [hidden_size, 256]``.
    """

    SINUSOIDAL_DIM = 256

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalScalarEmbedding(self.SINUSOIDAL_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(self.SINUSOIDAL_DIM, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Sinusoidal output is fp32 when ``t`` is integer; cast to match the
        # MLP's parameter dtype so the Linear matmul doesn't reject the input.
        out = self.sinusoidal(t)
        return self.mlp(out.to(self.mlp[0].weight.dtype))


def make_state_action_adaptor(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """3-layer MLP with parameter indices 0/2/4 (interleaved with tanh GELU)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=True),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden_dim, hidden_dim, bias=True),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden_dim, out_dim, bias=True),
    )


class FinalLayer(nn.Module):
    """Output head: RMSNorm -> 2-layer MLP that maps to ``action_dim``.

    The intermediate width equals ``hidden_size`` (not the action expert's
    ``intermediate_size``), matching the upstream checkpoint shape
    ``fc1.weight = [hidden_size, hidden_size]`` / ``fc2.weight = [action_dim, hidden_size]``.
    """

    def __init__(self, hidden_size: int, action_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm_final = InternLM2RMSNorm(hidden_size, eps=eps)
        self.ffn_final = nn.Sequential()
        self.ffn_final.add_module("fc1", nn.Linear(hidden_size, hidden_size, bias=True))
        self.ffn_final.add_module("act", nn.GELU(approximate="tanh"))
        self.ffn_final.add_module("fc2", nn.Linear(hidden_size, action_dim, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_final(self.norm_final(x))


class ActionExpertAttention(nn.Module):
    """Action-side self-attention that concatenates VLM K/V into the joint key/value."""

    def __init__(self, spec: InternLM2BlockSpec) -> None:
        super().__init__()
        self.spec = spec
        self.head_dim = spec.resolved_head_dim()
        self.num_q_heads = spec.num_attention_heads
        self.num_kv_heads = spec.num_key_value_heads
        self.num_kv_groups = spec.num_kv_groups
        per_group_heads = self.num_kv_groups + 2
        self.wqkv = nn.Linear(
            spec.hidden_size,
            self.num_kv_heads * per_group_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(self.num_q_heads * self.head_dim, spec.hidden_size, bias=False)

    def split_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = hidden_states.shape
        per_group_heads = self.num_kv_groups + 2
        qkv = self.wqkv(hidden_states).view(
            bsz,
            seqlen,
            self.num_kv_heads,
            per_group_heads,
            self.head_dim,
        )
        q = qkv[..., : self.num_kv_groups, :].reshape(
            bsz, seqlen, self.num_kv_heads * self.num_kv_groups, self.head_dim
        )
        k = qkv[..., -2, :]
        v = qkv[..., -1, :]
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_position_ids: torch.Tensor,
        vlm_position_ids: torch.Tensor,
        rotary_emb: InternLM2RotaryEmbedding,
        vlm_k: torch.Tensor,
        vlm_v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k_act, v_act = self.split_qkv(hidden_states)
        cos_act, sin_act = rotary_emb(action_position_ids)
        q, k_act = apply_rope(q, k_act, cos_act, sin_act)

        # Re-stamp absolute VLM positions onto the projected VLM keys so action
        # queries can use cross-modal positional information. Cos/sin from the
        # rotary embedding are fp32; cast the result back to vlm_k's dtype so
        # the joint K stays in the same precision as the action-side K.
        orig_dtype = vlm_k.dtype
        cos_vlm, sin_vlm = rotary_emb(vlm_position_ids)
        cos_vlm = cos_vlm.unsqueeze(1)
        sin_vlm = sin_vlm.unsqueeze(1)
        half = vlm_k.shape[-1] // 2
        rotated = torch.cat((-vlm_k[..., half:], vlm_k[..., :half]), dim=-1)
        vlm_k = ((vlm_k * cos_vlm) + (rotated * sin_vlm)).to(orig_dtype)

        joint_k = torch.cat([vlm_k, k_act], dim=2)
        joint_v = torch.cat([vlm_v, v_act], dim=2)

        attn_out = scaled_dot_product(
            q,
            joint_k,
            joint_v,
            num_kv_groups=self.num_kv_groups,
            mask=attention_mask,
            implementation=self.spec.attn_implementation,
        )
        bsz, _, seqlen, _ = q.shape
        out = attn_out.transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.wo(out)


class ActionExpertBlock(nn.Module):
    """Action-side decoder block: cross-attention into joint VLM/action keys, then FFN."""

    def __init__(self, spec: InternLM2BlockSpec) -> None:
        super().__init__()
        self.attention_norm = InternLM2RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.attention = ActionExpertAttention(spec)
        self.ffn_norm = InternLM2RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.feed_forward = InternLM2FeedForward(spec)

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_position_ids: torch.Tensor,
        vlm_position_ids: torch.Tensor,
        rotary_emb: InternLM2RotaryEmbedding,
        vlm_k: torch.Tensor,
        vlm_v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.attention(
            self.attention_norm(hidden_states),
            action_position_ids,
            vlm_position_ids,
            rotary_emb,
            vlm_k,
            vlm_v,
            attention_mask,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


# ----- GO-1-Air top-level model and policy -----------------------------


def _build_mlp1(vision_dim: int, scale: float, llm_hidden: int) -> nn.Sequential:
    """Vision-to-LLM projector matching the safetensors layout (LN, Linear, GELU, Linear)."""
    inv_scale_sq = int(round(1.0 / scale)) ** 2
    projector_in = vision_dim * inv_scale_sq
    return nn.Sequential(
        nn.LayerNorm(projector_in),
        nn.Linear(projector_in, llm_hidden, bias=True),
        nn.GELU(),
        nn.Linear(llm_hidden, llm_hidden, bias=True),
    )


class Go1Air(nn.Module):
    """Top-level GO-1-Air model.

    Forward path: vision_model -> pixel_shuffle -> mlp1 -> inject into LLM
    embeddings at ``img_context_token_id`` -> language_model (collect layer KV)
    -> action expert (state/action/time/freq embeddings + 24 cross-attention
    blocks + final_layer) -> diffusion sampling loop -> ``[B, chunk, action_dim]``.
    """

    def __init__(self, config: Go1AirConfig) -> None:
        super().__init__()
        self.config = config

        vision_spec = InternViTSpec.from_config(config)
        self.vision_model = InternVisionModel(vision_spec)
        self.mlp1 = _build_mlp1(
            vision_dim=config.vision_hidden_size,
            scale=config.downsample_ratio,
            llm_hidden=config.llm_hidden_size,
        )
        self.language_model = InternLM2GO1LanguageModel(config)

        spec = action_block_spec(config)
        adaptor_hidden = config.act_hidden_size
        self.state_adaptor = make_state_action_adaptor(
            in_dim=config.max_state_dim,
            hidden_dim=adaptor_hidden,
            out_dim=config.act_hidden_size,
        )
        self.action_adaptor = make_state_action_adaptor(
            in_dim=config.max_action_dim,
            hidden_dim=adaptor_hidden,
            out_dim=config.act_hidden_size,
        )
        self.time_embedder = TimestepEmbedder(config.act_hidden_size)
        self.freq_embedder = TimestepEmbedder(config.act_hidden_size)

        self.action_model = nn.ModuleDict(
            {
                "layers": nn.ModuleList([ActionExpertBlock(spec) for _ in range(config.act_num_hidden_layers)]),
                "norm": InternLM2RMSNorm(config.act_hidden_size, eps=config.act_rms_norm_eps),
            }
        )

        # Per-layer per-head projections from LLM head_dim → action expert head_dim.
        # The Linear is applied along the last (head_dim) axis of vlm_k/vlm_v;
        # the kv-head count is preserved (must be equal between the two stacks).
        head_dim_llm = config.llm_hidden_size // config.llm_num_attention_heads
        head_dim_act = spec.resolved_head_dim()
        self.k_proj_layers = nn.ModuleList(
            [nn.Linear(head_dim_llm, head_dim_act, bias=True) for _ in range(config.act_num_hidden_layers)]
        )
        self.v_proj_layers = nn.ModuleList(
            [nn.Linear(head_dim_llm, head_dim_act, bias=True) for _ in range(config.act_num_hidden_layers)]
        )

        self.final_layer = FinalLayer(
            hidden_size=config.act_hidden_size,
            action_dim=config.max_action_dim,
            eps=config.act_rms_norm_eps,
        )

        self._action_spec = spec
        self._action_rotary_emb = InternLM2RotaryEmbedding(spec.resolved_head_dim(), spec)
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
        )
        if config.compile_model:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

    # ----- runtime knobs --------------------------------------------------

    def set_attention_implementation(self, implementation: str) -> None:
        self.config.attn_implementation = implementation
        self._action_spec.attn_implementation = implementation
        self.language_model.set_attention_implementation(implementation)
        for block in self.action_model["layers"]:
            block.attention.spec.attn_implementation = implementation

    # ----- vision side ----------------------------------------------------

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run vision tower, drop CLS, apply pixel shuffle, project to LLM hidden size."""
        features = self.vision_model(pixel_values, select_layer=self.config.select_layer)
        # Drop the [CLS] token before pixel-shuffling so the patch grid is square.
        patches = features[:, 1:, :]
        compressed = pixel_shuffle(patches, self.config.downsample_ratio, self.config.pixel_shuffle_version)
        return self.mlp1(compressed)

    # ----- language prefix ------------------------------------------------

    def _inject_vision(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """Replace embeddings at ``img_context_token_id`` positions with vision features."""
        out = input_embeds.clone()
        flat_mask = input_ids == self.config.img_context_token_id
        flat_features = vision_features.reshape(-1, vision_features.shape[-1]).to(out.dtype)
        expected_slots = int(flat_mask.sum().item())
        if expected_slots != flat_features.shape[0]:
            raise ValueError(
                f"Go1Air vision injection mismatch: prompt has {expected_slots} "
                f"img_context_token positions but vision tower produced "
                f"{flat_features.shape[0]} tokens."
            )
        out[flat_mask] = flat_features
        return out

    def encode_prefix(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Embed text + inject vision tokens, run LLM, return last hidden state + per-layer KV."""
        embeds = self.language_model.model.tok_embeddings(input_ids)
        if pixel_values is not None:
            vision_features = self.encode_vision(pixel_values)
            embeds = self._inject_vision(embeds, input_ids, vision_features)
        out = self.language_model.model(inputs_embeds=embeds, attention_mask=attention_mask)
        return out.last_hidden_state, out.layer_kv, embeds

    # ----- action expert --------------------------------------------------

    def _project_vlm_kv(
        self,
        layer_idx: int,
        vlm_k: torch.Tensor,
        vlm_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vlm_k/vlm_v shape: (B, num_kv_heads, T, head_dim_llm).
        # The per-layer Linear acts along the last axis, mapping head_dim_llm
        # to head_dim_act while preserving the kv_heads / batch / time axes.
        proj_k = self.k_proj_layers[layer_idx](vlm_k)
        proj_v = self.v_proj_layers[layer_idx](vlm_v)
        return proj_k, proj_v

    def _build_joint_tokens(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        control_freq: torch.Tensor,
    ) -> torch.Tensor:
        bsz = actions.shape[0]
        state_token = self.state_adaptor(state).unsqueeze(1).expand(bsz, self.config.state_token_num, -1).contiguous()
        action_tokens = self.action_adaptor(actions)
        time_tok = self.time_embedder(timesteps).unsqueeze(1)
        freq_tok = self.freq_embedder(control_freq).unsqueeze(1)
        return torch.cat([time_tok, freq_tok, state_token, action_tokens], dim=1)

    def predict_clean(
        self,
        actions: torch.Tensor,
        state: torch.Tensor,
        timesteps: torch.Tensor,
        control_freq: torch.Tensor,
        vlm_layer_kv: list[tuple[torch.Tensor, torch.Tensor]],
        vlm_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        joint = self._build_joint_tokens(state, actions, timesteps, control_freq)
        bsz, joint_len, _ = joint.shape
        prefix_len = 2 + self.config.state_token_num
        vlm_len = vlm_layer_kv[0][0].shape[2]

        action_position_ids = torch.arange(joint_len, device=joint.device).expand(bsz, joint_len) + vlm_len
        vlm_position_ids = torch.arange(vlm_len, device=joint.device).expand(bsz, vlm_len)

        attention_mask = self._build_joint_attention_mask(
            joint_len=joint_len,
            vlm_len=vlm_len,
            vlm_attention_mask=vlm_attention_mask,
            device=joint.device,
            dtype=joint.dtype,
        )

        hidden = joint
        for idx, block in enumerate(self.action_model["layers"]):
            proj_k, proj_v = self._project_vlm_kv(idx, vlm_layer_kv[idx][0], vlm_layer_kv[idx][1])
            hidden = block(
                hidden,
                action_position_ids,
                vlm_position_ids,
                self._action_rotary_emb,
                proj_k,
                proj_v,
                attention_mask,
            )

        hidden = self.action_model["norm"](hidden)
        action_hidden = hidden[:, prefix_len:, :]
        return self.final_layer(action_hidden)

    def _build_joint_attention_mask(
        self,
        *,
        joint_len: int,
        vlm_len: int,
        vlm_attention_mask: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if vlm_attention_mask is None:
            return None
        bsz = vlm_attention_mask.shape[0]
        total_kv = vlm_len + joint_len
        neg = torch.finfo(dtype).min
        mask = torch.zeros((bsz, 1, joint_len, total_kv), device=device, dtype=dtype)
        invalid = (vlm_attention_mask == 0).view(bsz, 1, 1, vlm_len).expand(bsz, 1, joint_len, vlm_len)
        mask[..., :vlm_len] = mask[..., :vlm_len].masked_fill(invalid, neg)
        return mask

    @torch.inference_mode()
    def sample_actions(
        self,
        state: torch.Tensor,
        control_freq: torch.Tensor,
        vlm_layer_kv: list[tuple[torch.Tensor, torch.Tensor]],
        vlm_attention_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = state.shape[0]
        device = state.device
        dtype = state.dtype if state.is_floating_point() else torch.float32
        shape = (bsz, self.config.chunk_size, self.config.max_action_dim)
        if noise is None:
            noise = torch.randn(shape, device=device, dtype=dtype)
        elif noise.shape != shape:
            raise ValueError(f"Go1Air noise shape must be {shape}, got {tuple(noise.shape)}.")
        x = noise.to(device=device, dtype=dtype)

        self.noise_scheduler_sample.set_timesteps(self.config.num_inference_steps, device=device)
        for timestep in self.noise_scheduler_sample.timesteps:
            t = timestep.expand(bsz).to(device=device)
            x0_pred = self.predict_clean(
                x,
                state,
                t,
                control_freq,
                vlm_layer_kv,
                vlm_attention_mask,
            ).to(dtype)
            x = self.noise_scheduler_sample.step(x0_pred, timestep, x).prev_sample.to(dtype)

        return x


class Go1AirPolicy(nn.Module):
    """Pipeline-facing wrapper: holds a tokenizer + ``Go1Air`` model."""

    def __init__(
        self,
        config: Go1AirConfig,
        *,
        processor_model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.processor_model_name = processor_model_name
        self.model = Go1Air(config)
        self._tokenizer = None
        self._has_weights = False

    # ----- construction ---------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        config: Go1AirConfig,
        processor_model_name: str | None = None,
        strict: bool = False,
    ) -> Go1AirPolicy:
        policy = cls(config, processor_model_name=processor_model_name)
        policy._load_weights(Path(model_dir), strict=strict)
        policy._maybe_load_tokenizer(Path(model_dir))
        if policy._has_weights and policy._tokenizer is None:
            raise RuntimeError(
                "Go1AirPolicy: checkpoint weights loaded but tokenizer is missing; "
                "real-checkpoint inference cannot continue."
            )
        return policy

    def _load_weights(self, model_dir: Path, *, strict: bool) -> None:
        index_path = model_dir / "model.safetensors.index.json"
        single_path = model_dir / "model.safetensors"
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required to load GO-1-Air checkpoints.") from exc

        state_dict: dict[str, torch.Tensor] = {}
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
            shards = sorted(set(index["weight_map"].values()))
            for shard in shards:
                state_dict.update(load_file(str(model_dir / shard)))
        elif single_path.exists():
            state_dict.update(load_file(str(single_path)))
        else:
            logger.warning("Go1AirPolicy: no safetensors found under %s; running uninitialised.", model_dir)
            return

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self._has_weights = True
        if missing:
            logger.warning("Go1AirPolicy: %d missing weights (showing first 5): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("Go1AirPolicy: %d unexpected weights (showing first 5): %s", len(unexpected), unexpected[:5])
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Go1AirPolicy strict load failed: {len(missing)} missing, {len(unexpected)} unexpected."
            )

    def _maybe_load_tokenizer(self, model_dir: Path) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            logger.warning("transformers not available; GO-1-Air policy will run without a tokenizer.")
            return

        # ``model_dir`` ships GO-1-Air's own InternLM2 tokenizer (added_tokens
        # contain ``<IMG_CONTEXT>``); always prefer it. ``processor_model_name``
        # is only used as a fallback for users whose checkpoint directory lacks
        # tokenizer files.
        sources: list[str] = [str(model_dir)]
        if self.processor_model_name and str(self.processor_model_name) != str(model_dir):
            sources.append(str(self.processor_model_name))

        last_err: Exception | None = None
        for source in sources:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    source,
                    trust_remote_code=True,
                )
                logger.info("Go1AirPolicy: tokenizer loaded from %s.", source)
                return
            except Exception as exc:
                last_err = exc
        logger.warning(
            "Go1AirPolicy: tokenizer load failed from all sources %s (last err: %s); falling back to stub mode.",
            sources,
            last_err,
        )
        self._tokenizer = None

    # ----- forward --------------------------------------------------------

    def forward(
        self,
        batch_inputs: dict[str, Any],
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = self._infer_batch_size(batch_inputs)
        device = self._model_device()
        dtype = self._model_dtype()

        # Stub path: no weights loaded → return zero actions of the right shape.
        # Keeps pipeline plumbing exercisable end-to-end without a checkpoint.
        if not self._has_weights:
            return torch.zeros(
                (bsz, self.config.chunk_size, self.config.max_action_dim),
                device=device,
                dtype=noise.dtype if noise is not None else torch.float32,
            )

        state = self._prepare_state(batch_inputs, batch_size=bsz, device=device, dtype=dtype)

        pixel_values, vlm_attention_mask, input_ids = self._build_llm_inputs(
            batch_inputs,
            batch_size=bsz,
            device=device,
            dtype=dtype,
        )
        _, layer_kv, _ = self.model.encode_prefix(input_ids, pixel_values, vlm_attention_mask)

        control_freq = self._prepare_control_freq(batch_inputs, batch_size=bsz, device=device)

        return self.model.sample_actions(
            state=state,
            control_freq=control_freq,
            vlm_layer_kv=layer_kv,
            vlm_attention_mask=vlm_attention_mask,
            noise=noise,
        )

    # ----- input plumbing -------------------------------------------------

    def _infer_batch_size(self, batch_inputs: dict[str, Any]) -> int:
        state = batch_inputs.get(OBS_STATE)
        if isinstance(state, torch.Tensor):
            if state.ndim == 0:
                raise ValueError(f"Go1AirPolicy expects '{OBS_STATE}' to include a batch dimension.")
            return int(state.shape[0])
        for key, value in batch_inputs.items():
            if key.startswith(f"{OBS_IMAGES}.") and not key.endswith("_mask"):
                if isinstance(value, torch.Tensor) and value.ndim >= 1:
                    return int(value.shape[0])
        return 1

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _prepare_state(
        self,
        batch_inputs: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if OBS_STATE not in batch_inputs:
            raise ValueError(f"Go1AirPolicy expects '{OBS_STATE}' in batch_inputs.")
        state = batch_inputs[OBS_STATE]
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"Go1AirPolicy expects '{OBS_STATE}' as a tensor, got {type(state)!r}.")
        if state.ndim != 2 or state.shape != (batch_size, self.config.max_state_dim):
            raise ValueError(
                f"Go1AirPolicy expects '{OBS_STATE}' shape "
                f"({batch_size}, {self.config.max_state_dim}), got {tuple(state.shape)}."
            )
        return state.to(device=device, dtype=dtype)

    def _prepare_control_freq(
        self,
        batch_inputs: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if "control_freq" not in batch_inputs:
            return torch.full((batch_size,), 30.0, device=device, dtype=torch.float32)
        control_freq = batch_inputs["control_freq"]
        if not isinstance(control_freq, torch.Tensor):
            raise TypeError(f"Go1AirPolicy expects 'control_freq' as a tensor, got {type(control_freq)!r}.")
        control_freq = control_freq.to(device=device, dtype=torch.float32)
        if control_freq.ndim == 0:
            return control_freq.expand(batch_size)
        if control_freq.shape != (batch_size,):
            raise ValueError(
                f"Go1AirPolicy expects 'control_freq' shape ({batch_size},), got {tuple(control_freq.shape)}."
            )
        return control_freq

    def _build_llm_inputs(
        self,
        batch_inputs: dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """Collect pixel_values + tokenize task with ``<IMG_CONTEXT>`` placeholders.

        Each batch row gets ``images_per_row × vision_tokens_per_image`` copies
        of ``img_context_token_id`` prepended to the tokenised task string.
        ``Go1Air._inject_vision`` then replaces those positions with the
        ``mlp1``-projected vision features at forward time.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Go1AirPolicy.forward requires a tokenizer for the language prefix; "
                "load via from_pretrained(model_dir) or run the pipeline in stub mode."
            )

        # Group cameras (deterministic order) and pair each with its mask.
        # ``observation.images.*_mask`` is a (B,) bool tensor signalling whether
        # that camera is valid on each batch row; padded / unavailable cameras
        # must be dropped so they don't get encoded as a real visual prefix.
        camera_keys = sorted(
            k
            for k, v in batch_inputs.items()
            if k.startswith(f"{OBS_IMAGES}.") and not k.endswith("_mask") and isinstance(v, torch.Tensor)
        )

        per_row_frames: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        for cam_key in camera_keys:
            cam_tensor = batch_inputs[cam_key]
            if cam_tensor.ndim != 5:
                raise ValueError(
                    f"Go1AirPolicy expects '{cam_key}' shape (B, history, 3, H, W), got {tuple(cam_tensor.shape)}."
                )
            bsz_cam, history = cam_tensor.shape[:2]
            if bsz_cam != batch_size:
                raise ValueError(f"Go1AirPolicy expects '{cam_key}' batch {batch_size}, got {bsz_cam}.")
            expected_shape = (3, self.config.image_resolution[0], self.config.image_resolution[1])
            if tuple(cam_tensor.shape[2:]) != expected_shape:
                raise ValueError(
                    f"Go1AirPolicy expects '{cam_key}' per-frame shape {expected_shape}, "
                    f"got {tuple(cam_tensor.shape[2:])}."
                )
            cam_tensor = cam_tensor.to(device=device, dtype=dtype)
            mask = batch_inputs.get(f"{cam_key}_mask")
            for row in range(bsz_cam):
                for h in range(history):
                    if self._is_valid_image_frame(
                        mask, cam_key=cam_key, row=row, history_idx=h, batch_size=batch_size, history=history
                    ):
                        per_row_frames[row].append(cam_tensor[row, h])

        flat_frames: list[torch.Tensor] = []
        images_per_row: list[int] = []
        for row_imgs in per_row_frames:
            images_per_row.append(len(row_imgs))
            flat_frames.extend(row_imgs)
        pixel_values = torch.stack(flat_frames, dim=0).to(device=device) if flat_frames else None

        # Vision tokens per kept image after pixel_shuffle:
        #   patches_per_side = image_size / patch_size
        #   block = round(1 / downsample_ratio)
        #   tokens = (patches_per_side / block) ** 2
        side_in_patches = self.config.image_resolution[0] // self.config.vision_patch_size
        block = max(1, int(round(1.0 / self.config.downsample_ratio)))
        vision_tokens_per_image = (side_in_patches // block) ** 2

        task_strings = self._get_task_strings(batch_inputs, batch_size=batch_size)

        img_context_id = int(self.config.img_context_token_id)
        pad_id = int(self.config.llm_pad_token_id)

        input_ids_rows: list[list[int]] = []
        for row_idx, task in enumerate(task_strings):
            prefix = [img_context_id] * (images_per_row[row_idx] * vision_tokens_per_image)
            task_ids = self._tokenizer(
                task,
                add_special_tokens=True,
                return_tensors="pt",
            )["input_ids"][0].tolist()
            input_ids_rows.append(prefix + task_ids)

        max_len = max(len(row) for row in input_ids_rows)
        row_lengths = [len(row) for row in input_ids_rows]
        padded = [row + [pad_id] * (max_len - len(row)) for row in input_ids_rows]
        input_ids = torch.tensor(padded, device=device, dtype=torch.long)
        positions = torch.arange(max_len, device=device).unsqueeze(0)
        attention_mask = positions < torch.tensor(row_lengths, device=device).unsqueeze(1)

        return pixel_values, attention_mask, input_ids

    def _is_valid_image_frame(
        self,
        mask: Any,
        *,
        cam_key: str,
        row: int,
        history_idx: int,
        batch_size: int,
        history: int,
    ) -> bool:
        if mask is None:
            return True
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"Go1AirPolicy expects '{cam_key}_mask' as a tensor, got {type(mask)!r}.")
        if mask.ndim == 0:
            return bool(mask.item())
        if mask.ndim == 1:
            if mask.shape[0] != batch_size:
                raise ValueError(
                    f"Go1AirPolicy expects '{cam_key}_mask' shape ({batch_size},) "
                    f"or ({batch_size}, {history}), got {tuple(mask.shape)}."
                )
            return bool(mask[row].item())
        if mask.ndim == 2:
            if mask.shape != (batch_size, history):
                raise ValueError(
                    f"Go1AirPolicy expects '{cam_key}_mask' shape ({batch_size}, {history}), got {tuple(mask.shape)}."
                )
            return bool(mask[row, history_idx].item())
        raise ValueError(
            f"Go1AirPolicy expects '{cam_key}_mask' as scalar, (B,), or (B, history), got {tuple(mask.shape)}."
        )

    def _get_task_strings(self, batch_inputs: dict[str, Any], *, batch_size: int) -> list[str]:
        if OBS_TASK not in batch_inputs:
            raise ValueError(f"Go1AirPolicy expects '{OBS_TASK}' in batch_inputs.")
        task_payload = batch_inputs[OBS_TASK]
        if isinstance(task_payload, str):
            if batch_size != 1:
                raise ValueError(f"Go1AirPolicy expects {batch_size} task strings, got a single string.")
            return [task_payload]
        if not isinstance(task_payload, (list, tuple)):
            raise TypeError(f"Go1AirPolicy expects '{OBS_TASK}' as a string/list/tuple, got {type(task_payload)!r}.")
        if len(task_payload) != batch_size:
            raise ValueError(f"Go1AirPolicy expects {batch_size} task strings, got {len(task_payload)}.")
        tasks: list[str] = []
        for idx, item in enumerate(task_payload):
            if not isinstance(item, str):
                raise TypeError(f"Go1AirPolicy expects '{OBS_TASK}[{idx}]' as a string, got {type(item)!r}.")
            tasks.append(item)
        return tasks
