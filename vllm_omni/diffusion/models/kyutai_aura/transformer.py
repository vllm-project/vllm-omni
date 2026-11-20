# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# WanRMSNorm, WanLayerNorm, the attention projections, the block, Head and unpatchify are adapted from
# Wan2.1 (https://github.com/Wan-Video/Wan2.1, wan/modules/model.py, Apache-2.0),
# Copyright 2024-2025 The Alibaba Wan Team Authors.
"""Kyutai AURA diffusion transformer (https://arxiv.org/abs/2611.26540).

AURA's student is a Wan2.1 DiT with three additions:

- the reference frame's tokens share the sequence with the noisy chunk: they get their own
  modulation row (timestep 0) and a token-type embedding, and they never enter the history;
- speech cross-attention after a fixed set of blocks, with the Wan2.2-S2V audio modules;
- self-attention over a request-owned rolling history of earlier chunks (see ``kv_cache``).

The Wan parts are adapted from the Apache-2.0 Wan2.1 code (wan/modules/model.py: the RMS and
layer norms, the attention projections, the block, the head and unpatchify). The speech encoder
is vLLM-Omni's Wan2.2-S2V ``CausalAudioEncoder`` and the AdaIN layer is diffusers'
``AdaLayerNorm``. Parameter names are those of the released checkpoint (original Wan layout).

Precision, as the released weights expect: fp32 parameters under bf16 autocast; the time
embedding, the modulation and the head run in fp32 regions, the attention in bf16 and the rotary
embedding in float64.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

import torch
from diffusers.models.attention import AdaLayerNorm
from einops import rearrange
from torch import nn

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.models.kyutai_aura.config import KyutaiAuraConfig
from vllm_omni.diffusion.models.kyutai_aura.kv_cache import ChunkPositions, RollingChunkKVCache
from vllm_omni.diffusion.models.kyutai_aura.rope import apply_rope, gather_freqs, rope_table
from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer import CausalAudioEncoder, sinusoidal_embedding_1d

# Rows of ``trainable_cond_mask`` used while streaming.
NOISY_TOKENS, REFERENCE_TOKENS = 0, 1

# One entry per block: the layer's (keys, values) history and the rows [start, end) of the chunk.
LayerHistory = tuple[torch.Tensor, torch.Tensor, int, int]


def _fp32_region() -> torch.autocast:
    """Wan2.1 computes the modulation, the residual updates and the head under fp32 autocast."""
    return torch.autocast(device_type="cuda", dtype=torch.float32)


RowOp = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _per_token_type(x: torch.Tensor, num_noisy: int, fn: RowOp, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply ``fn(tokens, a_row, b_row)`` with row 0 of a, b ([1, 2, C]) to the noisy tokens
    x[:, :num_noisy] and row 1 to the reference-frame tokens after them."""
    noisy = fn(x[:, :num_noisy], a[:, 0:1], b[:, 0:1])
    reference = fn(x[:, num_noisy:], a[:, 1:2], b[:, 1:2])
    return torch.cat([noisy, reference], dim=1)


def _scale_shift(x: torch.Tensor, num_noisy: int, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return _per_token_type(x, num_noisy, lambda t, s, c: t * (1 + c) + s, shift, scale)


def _gated(x: torch.Tensor, num_noisy: int, gate: torch.Tensor) -> torch.Tensor:
    return _per_token_type(x, num_noisy, lambda t, g, _: t * g, gate, gate)


class WanRMSNorm(nn.Module):
    """RMS norm of Wan2.1: normalized in fp32, scaled by the fp32 weight (so the result is fp32)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C] -> normalized x, scaled by the weight."""
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """Layer norm of Wan2.1, computed in fp32 and returned in the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C] -> layer-normalized x in the dtype of x."""
        return super().forward(x.float()).type_as(x)


class WanAttentionProjections(nn.Module):
    """q, k, v, o of a Wan2.1 attention layer with RMS norms on q and k, over the full width.
    The attention itself runs in bf16 through vLLM-Omni's Attention layer (BSND layout)."""

    def __init__(self, dim: int, num_heads: int, eps: float, *, role: str, prefix: str) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)
        cross = {"computation": "cross", "cross_parallel_layout": "local_q_replicated_kv"} if role == "cross" else {}
        self.attn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            role=role,
            qkv_layout="BSND",
            skip_sequence_parallel=True,
            disable_kv_quant=True,
            prefix=f"{prefix}.attn",
            **cross,
        )

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """[B, L, heads, head_dim] inputs; returns [B, Lq, dim] in the dtype of q."""
        out = self.attn(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16))
        out = out.reshape(q.shape[0], -1, self.num_heads, self.head_dim)
        return out.type_as(q).flatten(2)


class WanCrossAttention(WanAttentionProjections):
    """Cross-attention of Wan2.1 (to the text tokens, or to the speech tokens of one frame)."""

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """x: [B, Lq, C] queries; context: [B, Lk, C] keys and values. Returns [B, Lq, C]."""
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        return self.o(self.attention(q, k, v))


class HistorySelfAttention(WanAttentionProjections):
    """Self-attention of one chunk over the request's history.

    The sequence holds the chunk's noisy tokens, then the reference frame's tokens. The noisy
    tokens' keys and values are written into the history (before RoPE) at the chunk's rows; the
    reference-frame tokens take part in the attention but are never written. Keys are rotated with
    the history's positions at every step."""

    def forward(
        self,
        x: torch.Tensor,
        num_noisy: int,
        history: LayerHistory,
        freqs_q: torch.Tensor,
        freqs_k: torch.Tensor,
    ) -> torch.Tensor:
        """x: [1, L, C], the ``num_noisy`` noisy tokens first. Returns [1, L, C]."""
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(x)).view(b, -1, n, d)
        v = self.v(x).view(b, -1, n, d)
        cached_k, cached_v, row_start, row_end = history
        cached_k[:, row_start:row_end] = k[:, :num_noisy]
        cached_v[:, row_start:row_end] = v[:, :num_noisy]
        visible_k = torch.cat([cached_k[:, :row_end], k[:, num_noisy:]], dim=1)
        visible_v = torch.cat([cached_v[:, :row_end], v[:, num_noisy:]], dim=1)
        out = self.attention(apply_rope(q, freqs_q), apply_rope(visible_k, freqs_k), visible_v)
        return self.o(out)


class AuraBlock(nn.Module):
    """Wan2.1 attention block (self-attention, text cross-attention, FFN, six modulation vectors)
    with two modulation rows, one for the noisy tokens and one for the reference-frame tokens."""

    def __init__(self, cfg: KyutaiAuraConfig, prefix: str) -> None:
        super().__init__()
        dim, eps = cfg.dim, cfg.eps
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = HistorySelfAttention(dim, cfg.num_heads, eps, role="self", prefix=f"{prefix}.self_attn")
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = WanCrossAttention(dim, cfg.num_heads, eps, role="cross", prefix=f"{prefix}.cross_attn")
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, cfg.ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(cfg.ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        num_noisy: int,
        text_context: torch.Tensor,
        history: LayerHistory,
        freqs_q: torch.Tensor,
        freqs_k: torch.Tensor,
    ) -> torch.Tensor:
        """x: [1, L, C], the noisy tokens first. e: [1, 6, 2, C] (row 0 noisy, row 1 reference)."""
        with _fp32_region():
            e = (self.modulation.unsqueeze(2) + e).unbind(dim=1)

        y = self.self_attn(
            _scale_shift(self.norm1(x).float(), num_noisy, e[0], e[1]), num_noisy, history, freqs_q, freqs_k
        )
        with _fp32_region():
            x = x + _gated(y, num_noisy, e[2])

        x = x + self.cross_attn(self.norm3(x), text_context)
        y = self.ffn(_scale_shift(self.norm2(x).float(), num_noisy, e[3], e[4]))
        with _fp32_region():
            x = x + _gated(y, num_noisy, e[5])
        return x


class SpeechInjector(nn.Module):
    """Per injection point: an AdaIN of the noisy tokens on the global speech token, then
    cross-attention from each latent frame's tokens to that frame's local speech tokens."""

    def __init__(self, cfg: KyutaiAuraConfig) -> None:
        super().__init__()
        count = len(cfg.audio_inject_layers)
        self.injector = nn.ModuleList(
            WanCrossAttention(cfg.dim, cfg.num_heads, cfg.eps, role="cross", prefix=f"audio_injector.injector.{i}")
            for i in range(count)
        )
        self.injector_adain_layers = nn.ModuleList(
            AdaLayerNorm(embedding_dim=cfg.dim, output_dim=2 * cfg.dim, chunk_dim=1) for _ in range(count)
        )


class Head(nn.Module):
    """Output head of Wan2.1: layer norm, modulation by the timestep embedding, linear."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float) -> None:
        super().__init__()
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C] tokens; e: [B, C] time embedding. Returns [B, L, C_out * prod(patch)]."""
        with _fp32_region():
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            return self.head(self.norm(x) * (1 + e[1]) + e[0])


class KyutaiAuraTransformer(nn.Module):
    """Velocity of one chunk of noisy latents, given the text context, the reference-frame latent,
    the chunk's speech tokens and the request's rolling history."""

    # Regional compilation (vllm_omni/diffusion/compile.py) compiles each of these blocks.
    _repeated_blocks = ["AuraBlock"]

    def __init__(self, cfg: KyutaiAuraConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dim = dim = cfg.dim
        self.patch_embedding = nn.Conv3d(cfg.latent_channels, dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(cfg.text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(nn.Linear(cfg.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(AuraBlock(cfg, prefix=f"blocks.{i}") for i in range(cfg.num_layers))
        self.head = Head(dim, cfg.latent_channels, cfg.patch_size, cfg.eps)
        self.trainable_cond_mask = nn.Embedding(2, dim)  # token types: noisy, reference frame
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=cfg.audio_dim,
            num_layers=cfg.audio_layers,
            out_dim=dim,
            num_token=cfg.audio_local_tokens,
            need_global=True,
        )
        self.audio_injector = SpeechInjector(cfg)
        self.speech_slot = {block: slot for slot, block in enumerate(cfg.audio_inject_layers)}
        self.positions = ChunkPositions.for_config(cfg)
        self.register_buffer("rope_freqs", rope_table(cfg.head_dim), persistent=False)

    # ---------------------------------------------------------------------------------- speech
    def encode_audio(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Speech tokens of one chunk: features [1, num_layers, dim, frames] at the video rate ->
        (global [1, t, 1, C], local [1, t, n + 1, C]) per latent frame t.

        The teacher saw ``audio_motion_frames`` frames of motion context before each clip; the
        student keeps that input length by repeating the chunk's first feature in front, and the
        latent frames that the repeated part produces are dropped."""
        context_frames = self.cfg.audio_motion_frames
        repeated = features[..., :1].expand(-1, -1, -1, context_frames)
        with _fp32_region():
            global_tokens, local_tokens = self.casual_audio_encoder(torch.cat([repeated, features], dim=-1).float())
        skip = (context_frames + 3) // 4  # the encoder downsamples time by 4 (two stride-2 causal convs)
        return global_tokens[:, skip:], local_tokens[:, skip:]

    def _inject_speech(
        self, slot: int, x: torch.Tensor, num_noisy: int, speech: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        global_tokens, local_tokens = speech
        frames = local_tokens.shape[1]
        noisy = rearrange(x[:, :num_noisy], "b (t n) c -> (b t) n c", t=frames)
        noisy = self.audio_injector.injector_adain_layers[slot](
            noisy, temb=rearrange(global_tokens, "b t n c -> (b t n) c")
        )
        update = self.audio_injector.injector[slot](noisy, rearrange(local_tokens, "b t n c -> (b t) n c"))
        update = rearrange(update, "(b t) n c -> b (t n) c", t=frames)
        return torch.cat([x[:, :num_noisy] + update, x[:, num_noisy:]], dim=1)

    # --------------------------------------------------------------------------------- forward
    def _patchify(self, latents: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """[1, C, F, H, W] -> tokens [1, F * h * w, dim] and the token grid (F, h, w)."""
        tokens = self.patch_embedding(latents)
        grid = tuple(tokens.shape[2:])
        return tokens.flatten(2).transpose(1, 2), grid

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        ref_latent: torch.Tensor,
        audio: tuple[torch.Tensor, torch.Tensor],
        history: RollingChunkKVCache,
    ) -> torch.Tensor:
        """x: [C, F, H, W] noisy latents of the current chunk; timestep: [1]; context: [L, 4096]
        (unpadded); ref_latent: [1, C, 1, H, W]; audio: the chunk's speech tokens. Returns the
        velocity [C, F, H, W] in fp32."""
        cfg = self.cfg
        noisy, grid = self._patchify(x.unsqueeze(0))
        reference, _ = self._patchify(ref_latent)
        num_noisy = noisy.shape[1]
        type_rows = self.trainable_cond_mask.weight.to(noisy.dtype)
        x = torch.cat([noisy + type_rows[NOISY_TOKENS], reference + type_rows[REFERENCE_TOKENS]], dim=1)

        # Time embedding as in Wan2.1: the chunk's timestep for the noisy tokens, 0 for the reference.
        with _fp32_region():
            t = torch.cat([timestep, timestep.new_zeros(1)])
            e = self.time_embedding(sinusoidal_embedding_1d(cfg.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        e0 = e0.transpose(0, 1).unsqueeze(0).contiguous()  # [1, 6, 2, C]

        context = self.text_embedding(
            torch.stack([torch.cat([context, context.new_zeros(cfg.text_len - context.size(0), context.size(1))])])
        )

        _, h, w = grid
        freqs_q = gather_freqs(self.rope_freqs, self.positions.query_positions(history.chunk), h, w)
        freqs_k = gather_freqs(self.rope_freqs, self.positions.key_positions(history.chunk), h, w)
        row_start, row_end = history.current_rows()
        for index, block in enumerate(self.blocks):
            layer_history = (history.keys[index], history.values[index], row_start, row_end)
            x = block(x, e0, num_noisy, context, layer_history, freqs_q, freqs_k)
            slot = self.speech_slot.get(index)
            if slot is not None:
                x = self._inject_speech(slot, x, num_noisy, audio)

        x = self.head(x[:, :num_noisy], e[:1])
        return self.unpatchify(x[0], grid).float()

    def unpatchify(self, x: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """Tokens [F * h * w, C_out * prod(patch)] -> latents [C_out, F, H, W] (Wan2.1)."""
        c, patch = self.cfg.latent_channels, self.cfg.patch_size
        u = x[: math.prod(grid)].view(*grid, *patch, c)
        u = torch.einsum("fhwpqrc->cfphqwr", u)
        return u.reshape(c, *[i * j for i, j in zip(grid, patch)])

    # --------------------------------------------------------------------------------- weights
    def checkpoint_names(self) -> set[str]:
        """Parameter names that the released checkpoint must provide (not the attention backends')."""
        return {name for name, _ in self.named_parameters() if ".attn." not in name}

    def load_checkpoint(
        self, weights: Iterable[tuple[str, torch.Tensor]], skip_prefixes: Sequence[str] = ("frame_packer.",)
    ) -> set[str]:
        """Strictly copy checkpoint tensors into the parameters (dtype kept as fp32)."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name.startswith(tuple(skip_prefixes)):
                continue
            if name == "trainable_cond_mask.weight":
                tensor = tensor[:2]  # token types used when streaming: noisy, reference frame
            if name not in params:
                raise KeyError(f"unexpected checkpoint tensor {name}")
            param = params[name]
            if param.shape != tensor.shape:
                raise ValueError(f"{name}: checkpoint shape {tuple(tensor.shape)} != model {tuple(param.shape)}")
            with torch.no_grad():
                param.copy_(tensor.to(param.dtype))
            loaded.add(name)
        missing = self.checkpoint_names() - loaded
        if missing:
            raise KeyError(f"checkpoint misses {len(missing)} tensors, e.g. {sorted(missing)[:5]}")
        return loaded
