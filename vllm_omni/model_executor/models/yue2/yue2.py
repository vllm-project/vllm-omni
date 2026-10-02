# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2-3B text-to-music: single-stage native-AR model, ported from upstream
YuE2 (bd90e4c, Apache-2.0).

One file per the single-stage layout (see auk/): constants, the AR model with
its model-owned sampler, the terminal NAR flow-matching pass, weight routing
and the tiled VAE decoder all live here because they all run inside stage 0.
The text tokenizer and token-id prompt construction live in
``vllm_omni/tokenizers/`` (driver-side only, never used by the model).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm
from transformers import PretrainedConfig, PreTrainedModel
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.v1.outputs import SamplerOutput

from vllm_omni.model_executor.models.output_templates import OmniOutput

# -------------------- token/protocol constants --------------------


"""Token/protocol constants for YuE2-3B, pinned by the checkpoint.

Values mirror the upstream ``yue2/protocol.py`` (bd90e4c) exactly; changing any
of them desynchronizes generation from the reference implementation. The model
is single-codebook: one codec token per latent frame, 25 frames per second,
so ``max_tokens`` in engine terms equals frames/25 seconds of music.
"""

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
LATENT_START, LATENT_END, LATENT_PAD = 184621, 184622, 184623
VOCAB_SIZE, CONTEXT = 184704, 24576

# Engine-side stop ids: the union of both phase ends. The model's own sampler
# only ever emits the end token of the request's current phase, so a request
# never stops on the other phase's end by accident.
STOP_TOKEN_IDS = [ABC_END, MUSIC_END]

SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2
FRAMES_PER_SECOND = 25
SAMPLES_PER_FRAME = 1920
LATENT_DIM = 64

# Reference sampling presets (upstream yue2_generation_config.json).
ABC_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 30,
    "repetition_penalty": 1.005,
    "penalty_window": 100,
    "min_tokens": 32,
    "max_tokens": 4096,
}
SEMANTIC_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 100,
    "repetition_penalty": 1.2,
    "penalty_window": 50,
    "min_tokens": 200,
    "max_tokens": 9000,
}

ODE_STEPS = 32
ODE_METHOD = "midpoint"
VAE_CORE_FRAMES = 1024
VAE_HALO_FRAMES = 16
DEFAULT_VAE_ID = "m-a-p/YuE2-Vae"

# extra_args keys the model reads per request.
KEY_PHASE = "yue2_phase"  # "abc" | "semantic"
KEY_TEMPERATURE = "yue2_temperature"
KEY_TOP_P = "yue2_top_p"
KEY_TOP_K = "yue2_top_k"
KEY_REPETITION_PENALTY = "yue2_repetition_penalty"
KEY_PENALTY_WINDOW = "yue2_penalty_window"
KEY_MIN_TOKENS = "yue2_min_tokens"
KEY_MAX_AUDIO_FRAMES = "yue2_max_audio_frames"
KEY_SEED = "yue2_seed"
KEY_SKIP_SYNTHESIS = "yue2_skip_synthesis"  # abc phase: tokens only, no audio
# Full prompt token ids, shipped by the driver. The NAR conditioning needs the
# whole prefix, but under a KV prefix-cache hit the engine schedules only the
# uncached tail, so the scheduled input_ids slice cannot rebuild it.
KEY_PREFIX_IDS = "yue2_prefix_ids"


# -------------------- checkpoint weight routing --------------------


"""AR/side routing for YuE2's single-file MoT checkpoint.

The checkpoint interleaves AR-path, NAR-path and projection tensors under one
namespace. AR tensors go through the vLLM backbone loader unchanged; NAR and
projection tensors are hand-loaded, with ``model.layers.N.nar_*`` names
remapped onto the ``nar_layers.N.*`` module layout.
"""

_SIDE_TOP_LEVEL = {"vae2llm", "llm2vae", "time_embedder"}


def partition_checkpoint_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    """Split checkpoint tensors into (backbone AR pairs, remapped side pairs)."""
    ar: list[tuple[str, torch.Tensor]] = []
    side: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        if name == "latent_pos_embed.pe":
            continue  # deterministic sinusoid, rebuilt in __init__
        if ".nar_" in name or name.split(".", 1)[0] in _SIDE_TOP_LEVEL:
            new = name
            if name.startswith("model.layers."):
                rest = name[len("model.layers.") :]
                layer_no, _, tail = rest.partition(".")
                if tail.startswith("nar_"):
                    tail = tail[len("nar_") :]
                new = f"nar_layers.{layer_no}.{tail}"
            side.append((new, tensor))
            continue
        ar.append((name, tensor))
    return ar, side


# -------------------- model-owned sampler --------------------


"""Model-owned sampling for YuE2-3B, ported from upstream ``sampling.py``.

The model claims ``prefer_model_sampler`` and reproduces the reference
request-local arithmetic exactly: phase masking (abc = text vocabulary + its
end; semantic = codec span + its end), windowed repetition penalty over the
request's own history, then temperature/top-k/top-p with a seeded multinomial.
Default CFG is off (guidance 1.0 for full/melody, 1.01 for off); the upstream
torch backend treats those the same, so Phase 1 samples a single row.
"""


def window_penalty(logits: torch.Tensor, recent_ids: list[int], penalty: float) -> torch.Tensor:
    """Upweight/downweight ids seen in the window, upstream arithmetic."""
    if penalty == 1.0 or not recent_ids:
        return logits
    recent = torch.as_tensor(recent_ids, dtype=torch.long, device=logits.device)
    if logits.dim() > 1:
        recent = recent.reshape(1, -1)
    freq = torch.zeros_like(logits)
    freq.scatter_add_(-1, recent, torch.ones_like(recent, dtype=logits.dtype))
    alpha = penalty**freq
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def distribution(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    penalty_window: int,
    history: list[int],
    step: int,
    min_tokens: int,
    phase: str,
) -> torch.Tensor:
    """Masked/penalized/shaped scores for one row (float32 path).

    Accepts either one row ``[vocab]`` (what the model's sampler sees) or a
    batch ``[rows, vocab]`` and returns the same shape it was given.
    """
    single = logits.dim() == 1
    logits = logits.unsqueeze(0) if single else logits
    scores = logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, float("-inf"))
    if phase == "abc":
        allowed[..., :EOD] = 0
    else:
        allowed[..., CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end] = 0
    scores = scores + allowed
    if step < min_tokens:
        scores[..., end] = -torch.inf
    scores = window_penalty(scores, history[-penalty_window:], repetition_penalty)
    if temperature == 0:
        return scores.squeeze(0) if single else scores
    if temperature != 1:
        scores = scores / temperature
    threshold = scores.topk(min(top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > top_p
        removed[..., :1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores.squeeze(0) if single else scores


def sample_row(scores: torch.Tensor, generator: torch.Generator, *, greedy: bool = False) -> int:
    """Draw one token id from prepared scores with the request's generator."""
    if not torch.isfinite(scores).any():
        raise RuntimeError("sampling scores are all -inf; check phase masking")
    if greedy:
        return int(scores.argmax().item())
    probabilities = scores.softmax(-1)
    next_id = torch.multinomial(probabilities, 1, generator=generator)
    return int(next_id.item())


# -------------------- NAR flow-matching pass --------------------


"""Acoustic flow matching for YuE2-3B, ported from upstream ``nar.py``.

One AR prefill per original chunk, then 32 midpoint ODE steps that walk only
the NAR path. The AR prefill re-uses the vLLM backbone's own projections
(fused ``qkv_proj`` split by head counts, ``q_norm``/``k_norm``, the layer's
``o_proj``/``mlp``), with RoPE and SDPA computed here exactly like the
reference: NAR positions attend the full AR prefix plus all NAR positions
bidirectionally, which a per-chunk prefill + concatenated K/V reproduces.
"""


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


# -------------------- tiled VAE decoder --------------------


"""Portable FP32 YuE2 Oobleck VAE with exact-boundary tiled decoding.

Oobleck and SnakeBeta derived from stable-audio-tools a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45.
Copyright (c) 2023 Stability AI; Copyright (c) 2022 NVIDIA CORPORATION.
MIT: see THIRD_PARTY_NOTICES.md shipped with this module/model repository.
"""


def checkpoint(function, *args, **kwargs):
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * torch.pow(torch.sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features,
        alpha=1.0,
        alpha_trainable=True,
        alpha_logscale=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return snake_beta(x, alpha, beta)


def get_activation(activation: Literal["elu", "snake", "none"], channels=None) -> nn.Module:
    if activation == "elu":
        return nn.ELU()
    if activation == "snake":
        return SnakeBeta(channels)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, act_type):
        super().__init__()
        self.dilation = dilation
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            get_activation(act_type, channels=out_channels),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
            ),
            get_activation(act_type, channels=out_channels),
            WNConv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        residual = x
        if self.training:
            x = checkpoint(self.layers, x)
        else:
            x = self.layers(x)
        return x + residual


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, in_channels, 1, act_type),
            ResidualUnit(in_channels, in_channels, 3, act_type),
            ResidualUnit(in_channels, in_channels, 9, act_type),
            get_activation(act_type, channels=in_channels),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super().__init__()
        upsample_layer = WNConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.layers = nn.Sequential(
            get_activation(act_type, channels=in_channels),
            upsample_layer,
            ResidualUnit(out_channels, out_channels, 1, act_type),
            ResidualUnit(out_channels, out_channels, 3, act_type),
            ResidualUnit(out_channels, out_channels, 9, act_type),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=(1, 2, 4, 8),
        strides=(2, 4, 8, 8),
        use_snake=False,
        antialias_activation=False,
    ):
        super().__init__()
        if antialias_activation:
            raise ValueError("The released encoder does not use antialias_activation")
        self.in_channels = in_channels
        c_mults = [1] + list(c_mults)
        self.depth = len(c_mults)
        layers = [
            WNConv1d(
                in_channels=in_channels,
                out_channels=c_mults[0] * channels,
                kernel_size=7,
                padding=3,
            )
        ]
        act_type = "snake" if use_snake else "elu"
        for i in range(self.depth - 1):
            layers.append(
                EncoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i + 1] * channels,
                    stride=strides[i],
                    act_type=act_type,
                )
            )
        layers.extend(
            [
                get_activation(act_type, channels=c_mults[-1] * channels),
                WNConv1d(
                    in_channels=c_mults[-1] * channels,
                    out_channels=latent_dim,
                    kernel_size=3,
                    padding=1,
                ),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=(1, 2, 4, 8),
        strides=(2, 4, 8, 8),
        use_snake=False,
        snake_type="vanilla",
        antialias_activation=False,
        use_nearest_upsample=False,
        use_filter=False,
        final_tanh=True,
    ):
        super().__init__()
        if antialias_activation or use_nearest_upsample or use_filter:
            raise ValueError("Unsupported option for the released decoder")
        if use_snake and snake_type != "vanilla":
            raise ValueError("The released decoder uses vanilla SnakeBeta")
        self.out_channels = out_channels
        c_mults = [1] + list(c_mults)
        self.depth = len(c_mults)
        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults[-1] * channels,
                kernel_size=7,
                padding=3,
            )
        ]
        act_type = "snake" if use_snake else "elu"
        for i in range(self.depth - 1, 0, -1):
            layers.append(
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    act_type=act_type,
                )
            )
        layers.extend(
            [
                get_activation(act_type, channels=c_mults[0] * channels),
                WNConv1d(
                    in_channels=c_mults[0] * channels,
                    out_channels=out_channels,
                    kernel_size=7,
                    padding=3,
                    bias=False,
                ),
                nn.Tanh() if final_tanh else nn.Identity(),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class YuE2VAEConfig(PretrainedConfig):
    """Configuration shared by YuE2-Vae and YuE2-Vae-legacy.

    Use ``standard`` for listening and ``legacy`` for the paper metric baseline.
    """

    model_type = "yue2_vae"

    _hf_fields = frozenset(
        {
            "model_type",
            "architectures",
            "auto_map",
            "transformers_version",
            "dtype",
            "torch_dtype",
            "return_dict",
            "output_hidden_states",
            "output_attentions",
            "use_cache",
            "tie_word_embeddings",
            "torchscript",
            "is_decoder",
            "is_encoder_decoder",
            "add_cross_attention",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "decoder_start_token_id",
            "attn_implementation",
        }
    )

    def to_dict(self):
        return {
            key: value
            for key, value in super().to_dict().items()
            if key in self._hf_fields or key in self._inference_fields
        }

    _inference_fields = frozenset(
        [
            "encoder_config",
            "decoder_config",
            "sample_rate",
            "latent_dim",
            "downsampling_ratio",
            "audio_channels",
            "release_variant",
            "decode_core_frames",
            "decode_halo_frames",
        ]
    )

    def __init__(
        self,
        encoder_config=None,
        decoder_config=None,
        sample_rate=48000,
        latent_dim=64,
        downsampling_ratio=1920,
        audio_channels=2,
        release_variant="standard",
        decode_core_frames=1024,
        decode_halo_frames=16,
        **kwargs,
    ):
        kwargs = {key: value for key, value in kwargs.items() if key in self._hf_fields}
        kwargs.setdefault("architectures", ["YuE2VAE"])
        kwargs.setdefault(
            "auto_map",
            {
                "AutoConfig": "modeling_vae.YuE2VAEConfig",
                "AutoModel": "modeling_vae.YuE2VAE",
            },
        )
        super().__init__(**kwargs)
        self.encoder_config = encoder_config or dict(
            in_channels=2,
            channels=64,
            c_mults=[1, 2, 4, 8, 16, 32],
            strides=[2, 2, 4, 4, 5, 6],
            latent_dim=128,
            use_snake=True,
        )
        self.decoder_config = decoder_config or dict(
            out_channels=2,
            channels=64,
            c_mults=[1, 2, 4, 8, 16, 32],
            strides=[2, 2, 4, 4, 5, 6],
            latent_dim=64,
            use_snake=True,
            snake_type="vanilla",
            use_filter=False,
            final_tanh=False,
        )
        encoder_fields = {
            "in_channels",
            "channels",
            "latent_dim",
            "c_mults",
            "strides",
            "use_snake",
            "antialias_activation",
        }
        decoder_fields = {
            "out_channels",
            "channels",
            "latent_dim",
            "c_mults",
            "strides",
            "use_snake",
            "snake_type",
            "antialias_activation",
            "use_nearest_upsample",
            "use_filter",
            "final_tanh",
        }
        self.encoder_config = {key: value for key, value in self.encoder_config.items() if key in encoder_fields}
        self.decoder_config = {key: value for key, value in self.decoder_config.items() if key in decoder_fields}
        self.sample_rate = int(sample_rate)
        self.latent_dim = int(latent_dim)
        self.downsampling_ratio = int(downsampling_ratio)
        self.audio_channels = int(audio_channels)
        self.release_variant = release_variant
        self.decode_core_frames = int(decode_core_frames)
        self.decode_halo_frames = int(decode_halo_frames)
        if self.decode_core_frames < 1 or self.decode_halo_frames < 0:
            raise ValueError("Invalid VAE core/halo configuration")
        if math.prod(self.decoder_config["strides"]) != self.downsampling_ratio:
            raise ValueError("Decoder strides do not match downsampling_ratio")
        if self.decoder_config["latent_dim"] != self.latent_dim:
            raise ValueError("Decoder input channels do not match latent_dim")


def _dependency_interval(module, low, high):
    """Inclusive input support of an output interval; no waveform blending."""
    if isinstance(module, (nn.Sequential, OobleckDecoder, DecoderBlock)):
        layers = module if isinstance(module, nn.Sequential) else module.layers
        for child in reversed(list(layers)):
            low, high = _dependency_interval(child, low, high)
        return low, high
    if isinstance(module, ResidualUnit):
        a, b = _dependency_interval(module.layers, low, high)
        return min(a, low), max(b, high)
    if isinstance(module, nn.ConvTranspose1d):
        s, p, d, k = (module.stride[0], module.padding[0], module.dilation[0], module.kernel_size[0])
        return -(-(low + p - d * (k - 1)) // s), (high + p) // s
    if isinstance(module, nn.Conv1d):
        s, p, d, k = (module.stride[0], module.padding[0], module.dilation[0], module.kernel_size[0])
        return low * s - p, high * s - p + d * (k - 1)
    if isinstance(module, (SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return low, high
    raise TypeError(f"No audited support rule for {type(module).__name__}")


def _output_length(module, length):
    if isinstance(module, (nn.Sequential, OobleckDecoder, DecoderBlock)):
        layers = module if isinstance(module, nn.Sequential) else module.layers
        for child in layers:
            length = _output_length(child, length)
        return length
    if isinstance(module, nn.ConvTranspose1d):
        return (
            (length - 1) * module.stride[0]
            - 2 * module.padding[0]
            + module.dilation[0] * (module.kernel_size[0] - 1)
            + module.output_padding[0]
            + 1
        )
    if isinstance(module, nn.Conv1d):
        return (length + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[
            0
        ] + 1
    if isinstance(module, (ResidualUnit, SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return length
    raise TypeError(f"No audited length rule for {type(module).__name__}")


class YuE2VAE(PreTrainedModel):
    """Strict EMA model; only the decoder needs to reside on an accelerator.

    ``decode_tiled`` preserves finite receptive-field context and writes cropped
    cores to CPU. The mathematical waveform is the full decoder's waveform;
    convolution kernel choices may cause small FP32 rounding differences.
    """

    config_class = YuE2VAEConfig
    base_model_prefix = ""
    main_input_name = "audio"

    def __init__(self, config, decoder_only=False):
        super().__init__(config)
        self.decoder_only = bool(decoder_only)
        if not self.decoder_only:
            self.encoder = OobleckEncoder(**config.encoder_config)
        self.decoder = OobleckDecoder(**config.decoder_config)
        self.eval().requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        decoder_only=False,
        device="cpu",
        torch_dtype=None,
        dtype=None,
        revision=None,
        token=None,
        cache_dir=None,
        local_files_only=False,
        force_download=False,
        subfolder="",
        **kwargs,
    ):
        """Load a local export or Hub repository, selecting decoder tensors.

        Complete exports use unprefixed EMA keys. ``decoder_only=True`` avoids
        constructing the encoder or loading encoder tensors into RAM/GPU.
        """
        from safetensors import safe_open

        requested_dtype = dtype if dtype is not None else torch_dtype
        if requested_dtype not in (None, "auto", "float32", torch.float32):
            raise ValueError("The validated VAE requires FP32; quantize the LM separately")
        device_map = kwargs.pop("device_map", None)
        if device_map is not None:
            if device_map == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif isinstance(device_map, str):
                device = device_map
            elif isinstance(device_map, dict) and set(device_map) == {""}:
                device = device_map[""]
            else:
                raise ValueError("Use decoder_only=True and a single device for the VAE")
        for name in (
            "trust_remote_code",
            "low_cpu_mem_usage",
            "_from_auto",
            "_from_pipeline",
            "_commit_hash",
            "adapter_kwargs",
            "_fast_init",
            "weights_only",
            "use_safetensors",
        ):
            kwargs.pop(name, None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        if kwargs or model_args:
            raise TypeError(f"Unsupported VAE loading options: {sorted(kwargs)}")
        path = Path(pretrained_model_name_or_path).expanduser()
        if not path.is_dir():
            from vllm_omni.transformers_utils.repo_utils import hf_api

            path = Path(
                hf_api().snapshot_download(
                    str(pretrained_model_name_or_path),
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    force_download=force_download,
                    allow_patterns=[
                        f"{subfolder + '/' if subfolder else ''}{pattern}"
                        for pattern in ("config.json", "*.safetensors", "*.safetensors.index.json")
                    ],
                )
            )
        path = path / subfolder
        if config is None:
            config = YuE2VAEConfig.from_pretrained(path, local_files_only=True)
        # vLLM may construct us under a BF16 default dtype. Widen before copying
        # the FP32 checkpoint so load_state_dict does not round its values.
        model = cls(config, decoder_only=decoder_only).float()
        index = path / "model.safetensors.index.json"
        if index.exists():
            mapping = json.loads(index.read_text())["weight_map"]
            files = sorted({name for key, name in mapping.items() if not decoder_only or key.startswith("decoder.")})
        else:
            files = ["model.safetensors"]
        state = {}
        for name in files:
            with safe_open(path / name, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if not decoder_only or key.startswith("decoder."):
                        if key in state:
                            raise ValueError(f"Duplicate VAE tensor: {key}")
                        state[key] = handle.get_tensor(key)
        expected = set(model.state_dict())
        if set(state) != expected:
            raise ValueError(
                f"VAE tensor mismatch: missing={sorted(expected - set(state))}, "
                f"unexpected={sorted(set(state) - expected)}"
            )
        if any(value.dtype != torch.float32 for value in state.values()):
            raise ValueError("VAE export contains tensors that are not FP32")
        model.load_state_dict(state, strict=True)
        model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        if output_loading_info:
            return model, dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
        return model

    def save_pretrained(self, save_directory, *args, **kwargs):
        if self.decoder_only:
            raise ValueError("Reload decoder_only=False to save a complete VAE repository")
        if kwargs.get("safe_serialization", True) is False:
            raise ValueError("YuE2 VAE release exports require safetensors")
        return super().save_pretrained(save_directory, *args, **kwargs)

    @property
    def decoder_device(self):
        return next(self.decoder.parameters()).device

    def _latent(self, latent):
        latent = torch.as_tensor(latent)
        if latent.ndim != 3 or latent.shape[1] != self.config.latent_dim or latent.shape[0] < 1 or latent.shape[-1] < 1:
            raise ValueError(f"Expected nonempty [B,{self.config.latent_dim},T] latents")
        if not torch.isfinite(latent).all():
            raise ValueError("VAE latents contain non-finite values")
        if next(self.decoder.parameters()).dtype != torch.float32:
            raise ValueError("VAE decoder weights must remain FP32")
        return latent

    @torch.inference_mode()
    def encode(self, audio, sample=False, generator=None, return_info=False):
        """Encode FP32 stereo audio; posterior mean by default.

        Set ``sample=True`` with a per-request ``torch.Generator`` to reproduce
        stochastic posterior sampling. This audio VAE is not the unreleased
        semantic audio tokenizer semantic tokenizer.
        """
        if self.decoder_only:
            raise RuntimeError("Encoder not loaded; reload with decoder_only=False")
        audio = torch.as_tensor(audio)
        if (
            audio.ndim != 3
            or audio.shape[1] != self.config.audio_channels
            or audio.shape[-1] < self.config.downsampling_ratio
        ):
            raise ValueError("Expected audio [B,2,S] with at least one latent frame")
        if not torch.isfinite(audio).all():
            raise ValueError("Audio contains non-finite values")
        device = next(self.encoder.parameters()).device
        pre = self.encoder(audio.to(device=device, dtype=torch.float32))
        mean, scale = pre.chunk(2, dim=1)
        stdev = torch.nn.functional.softplus(scale) + 1e-4
        if sample:
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=device, generator=generator)
            latent = noise * stdev + mean
        else:
            latent = mean
        if return_info:
            return latent, dict(mean=mean, scale=scale, stdev=stdev)
        return latent

    @torch.inference_mode()
    def decode(self, latent):
        """Full waveform, FP32 [B,2,1920*T-64], without clipping."""
        latent = self._latent(latent)
        with torch.autocast(device_type=self.decoder_device.type, enabled=False):
            return self.decoder(latent.to(device=self.decoder_device, dtype=torch.float32))

    def natural_output_length(self, frames):
        if int(frames) < 1:
            raise ValueError("frames must be positive")
        return _output_length(self.decoder, int(frames))

    def required_halo(self, core_frames=None):
        core_frames = self.config.decode_core_frames if core_frames is None else core_frames
        ratio = self.config.downsampling_ratio
        low, high = _dependency_interval(self.decoder, 0, core_frames * ratio - 1)
        return max(0, -low, high - core_frames + 1)

    @torch.inference_mode()
    def decode_tiled(
        self,
        latent,
        core_frames=None,
        halo_frames=None,
        output_device="cpu",
        on_progress: Callable[[int, int], None] | None = None,
    ):
        """Decode bounded tiles, retaining exact cores with natural end length.

        Each crop has enough left/right context for every dependency. There is
        no crossfade or boundary smoothing, and no zero padding of final audio.
        CPU output prevents an entire song from accumulating on the GPU.
        ``on_progress(completed, total)`` runs after each existing crop copy;
        no extra synchronization is added. With a CUDA output device, queued
        work may still be executing. Callback exceptions propagate.
        """
        latent = self._latent(latent)
        core_frames = self.config.decode_core_frames if core_frames is None else core_frames
        halo_frames = self.config.decode_halo_frames if halo_frames is None else halo_frames
        if not isinstance(core_frames, int) or core_frames < 1:
            raise ValueError("core_frames must be a positive integer")
        required = self.required_halo(core_frames)
        if not isinstance(halo_frames, int) or halo_frames < required:
            raise ValueError(f"halo_frames must be at least {required} for this decoder")
        frames = latent.shape[-1]
        ratio = self.config.downsampling_ratio
        total = self.natural_output_length(frames)
        audio = torch.empty(
            (latent.shape[0], self.config.audio_channels, total), dtype=torch.float32, device=output_device
        )
        tiles = (frames + core_frames - 1) // core_frames
        for tile_index, start in enumerate(range(0, frames, core_frames)):
            end = min(frames, start + core_frames)
            left = max(0, start - halo_frames)
            right = min(frames, end + halo_frames)
            tile = self.decode(latent[..., left:right])
            out_start, out_end = start * ratio, min(end * ratio, total)
            crop_start = (start - left) * ratio
            crop = tile[..., crop_start : crop_start + out_end - out_start]
            if crop.shape[-1] != out_end - out_start:
                raise RuntimeError("VAE tile did not cover its requested output core")
            audio[..., out_start:out_end].copy_(crop.to(output_device))
            del tile, crop
            if on_progress is not None:
                on_progress(tile_index + 1, tiles)
        return audio

    def decode_audio(self, latent, chunked=True, **kwargs):
        return self.decode_tiled(latent, **kwargs) if chunked else self.decode(latent)

    def forward(self, audio, sample=False, generator=None):
        return self.decode(self.encode(audio, sample=sample, generator=generator))


YuE2VAEConfig.register_for_auto_class("AutoConfig")
YuE2VAE.register_for_auto_class("AutoModel")


# -------------------- the model --------------------


"""YuE2-3B text-to-music model for vllm-omni: single-stage native-AR.

The checkpoint is one Mixture-of-Transformers: a Qwen3-shaped AR path (which
is exactly a Qwen3-1.7B-class backbone over an extended vocabulary) plus a
parallel NAR path (``nar_self_attn``/``nar_mlp`` per layer) that a 32-step
midpoint flow-matching solver walks over 64-dim VAE latents. Both paths and
the projection heads live in the same ``model.safetensors``, so a single
vLLM stage loads everything once and no weights are duplicated.

Sampling is model-owned (``prefer_model_sampler``), reproducing the upstream
request-local arithmetic: per-phase vocabulary masking, windowed repetition
penalty, seeded multinomial, ``min_tokens`` end-token suppression. One song
is one or two engine requests — the abc phase (``cot=full|melody``) first,
then the semantic phase; the driver stitches them. When the semantic phase's
end token is drawn (or its frame budget is hit), the model solves the ODE and
decodes 48 kHz stereo audio inside the step and ships it as the request's
final multimodal payload, so the whole song arrives in one piece.
"""
logger = init_logger(__name__)

__all__ = ["Yue2ForCausalLM"]

DEFAULT_SEED = 831001


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class _RotaryEmbedding(nn.Module):
    """Checkpoint-compatible RoPE: cos/sin over [B, T, head_dim/2]."""

    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self._inv_freq: torch.Tensor | None = None

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = position_ids.device
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
            )
        angles = position_ids.float().unsqueeze(-1) * self._inv_freq
        return angles.cos(), angles.sin()


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class _NARAttention(nn.Module):
    """NAR-path attention with checkpoint-native separate projections."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)

    def project_qkv(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        rc, rs = cos.unsqueeze(2), sin.unsqueeze(2)
        return _apply_rotary(q, rc, rs), _apply_rotary(k, rc, rs), v


class _NARMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _NARLayer(nn.Module):
    """One NAR path layer; checkpoint ``model.layers.N.nar_*`` remaps onto it."""

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = _NARAttention(config)
        self.pre_mlp_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = _NARMLP(config)


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(next(self.parameters()).dtype))


class _AudioPositionEmbedding(nn.Module):
    def __init__(self, max_frames: int, hidden_size: int):
        super().__init__()
        pe = torch.zeros(max_frames, hidden_size)
        position = torch.arange(0, max_frames, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, position_ids):
        return self.pe[position_ids]


def _request_key(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


@dataclass
class _RowConstants:
    """Per-request values fixed at admission (a prefill step, never replayed)."""

    request_id: str
    phase: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    penalty_window: int
    min_tokens: int
    max_audio_frames: int
    seed: int
    skip_synthesis: bool


@dataclass
class _RequestState:
    """Everything one song request accumulates across decode steps."""

    request_id: str
    constants: _RowConstants
    prompt_tokens: int = 0
    prompt_len: int = 0  # full prompt length; a completing prefill row reaches it
    prefix_ids: list[int] = field(default_factory=list)
    history: list[int] = field(default_factory=list)
    generator: torch.Generator | None = None
    finished: bool = False
    truncated: bool = False


class Yue2ForCausalLM(nn.Module):
    """YuE2-3B: vLLM-native Qwen3 AR backbone + NAR flow-matching + VAE."""

    have_multimodal_outputs = True
    prefer_model_sampler = True
    has_postprocess = False
    # The song is decoded in-model and shipped via make_omni_output; per-step
    # hidden states must NOT ride the pooler payload, or the output pipeline
    # remaps the accumulated "hidden" rows to the audio modality key and the
    # driver receives hidden states instead of the decoded waveform
    # (single-stage precedent: minimax_music3 talker).
    omni_pooler_payload_include_hidden: bool = False
    # No downstream stage consumes prefix-cached hidden/mm tensors for YuE2;
    # opting out keeps the merged prefix-cache view from rebuilding "hidden"
    # payloads (KV-block prefix caching for cot=full stays on).
    requires_full_prefix_cached_hidden_states: bool = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        # The NAR pass keeps every KV head on each rank (ar_project splits by
        # the per-rank head count), so TP>1 builds a server whose every song
        # fails after its full AR decode. Pipeline parallelism has the same
        # problem: AR and NAR must live in one stage. Fail at startup instead.
        tp = int(getattr(vllm_config.parallel_config, "tensor_parallel_size", 1) or 1)
        pp = int(getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) or 1)
        if tp > 1 or pp > 1:
            raise ValueError(
                f"Yue2 is single-GPU only: the NAR path is replicated, so "
                f"tensor_parallel_size={tp} and pipeline_parallel_size={pp} are "
                "unsupported; use 1/1."
            )
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.config = config
        hidden = int(config.hidden_size)

        self.model = Qwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                int(config.vocab_size),
                hidden,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(int(config.vocab_size))

        self.nar_layers = nn.ModuleList([_NARLayer(config) for _ in range(int(config.num_hidden_layers))])
        self.vae2llm = nn.Linear(int(getattr(config, "latent_dim", LATENT_DIM)), hidden)
        self.llm2vae = nn.Linear(hidden, int(getattr(config, "latent_dim", LATENT_DIM)))
        self.time_embedder = _TimestepEmbedder(hidden)
        self.latent_pos_embed = _AudioPositionEmbedding(
            int(getattr(config, "max_latent_frames", config.max_position_embeddings)),
            hidden,
        )
        self.rotary_emb = _RotaryEmbedding(int(config.head_dim), float(config.rope_theta))

        self.vocab_size = int(config.vocab_size)
        self.max_position_embeddings = int(config.max_position_embeddings)
        self.max_latent_frames = int(getattr(config, "max_latent_frames", self.max_position_embeddings))
        self._timestep_shift = float(getattr(config, "timestep_shift", 1.0))

        self._states: dict[str, _RequestState] = {}
        self._row_constants: dict[str, _RowConstants] = {}
        self._audio_queue: list[tuple[str, torch.Tensor, bool, bool]] = []
        self._deferred_cleanup_ids: set[str] = set()
        self._step_rows: list[tuple[str, int, int]] = []  # (req_id, computed, scheduled)
        self._decode_t0: dict[str, float] = {}  # first decode-step wall time, for AR tok/s
        self._last_mm: dict[str, Any] | None = None  # current step's make_omni_output payload
        self._vae: YuE2VAE | None = None
        self._vae_device: torch.device | None = None

    # ------------------------------------------------------------ sampling

    def shift_t(self, raw_t: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_sig = torch.sigmoid(torch.tensor(raw_t, dtype=dtype, device=device))
        shift = self._timestep_shift
        return shift * t_sig / (1 + (shift - 1) * t_sig)

    def rotary(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb(positions)

    def ar_project(self, layer, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """AR-path Q/K/V from the vLLM backbone's fused projection."""
        attn = layer.self_attn
        qkv = attn.qkv_proj(x)
        if isinstance(qkv, tuple):
            qkv = qkv[0]
        num_heads, num_kv_heads, head_dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
        q, k, v = torch.split(
            qkv,
            [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim],
            dim=-1,
        )
        T = x.shape[-2]
        q = attn.q_norm(q.view(T, num_heads, head_dim))
        k = attn.k_norm(k.view(T, num_kv_heads, head_dim))
        rc, rs = cos.unsqueeze(2), sin.unsqueeze(2)
        # q/k are [T, heads, head_dim] (3D); rc/rs are [1, T, 1, head_dim/2]
        # (4D). Broadcasting them directly promotes q/k to 4D, which the NAR
        # attention rejects. Add the batch axis explicitly, then drop it.
        q = _apply_rotary(q.unsqueeze(0), rc, rs).squeeze(0)
        k = _apply_rotary(k.unsqueeze(0), rc, rs).squeeze(0)
        return q, k, v.view(T, num_kv_heads, head_dim)

    # ------------------------------------------------------------ weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ar_pairs, remapped = partition_checkpoint_weights(weights)
        missing, unexpected = self.load_state_dict(dict(remapped), strict=False)
        missing = [k for k in missing if not k.startswith(("model.", "lm_head"))]
        if missing or unexpected:
            raise RuntimeError(f"YuE2 NAR/projection weight mismatch: missing={missing}, unexpected={unexpected}")

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(iter(ar_pairs))
        # Load the VAE now, not lazily at the first finishing request: a bad
        # path or a failed download must surface at startup, and the decoder
        # weights must sit on the GPU before vLLM's memory profiling sizes
        # the KV cache.
        device = f"cuda:{torch.accelerator.current_device_index()}" if torch.cuda.is_available() else "cpu"
        self._vae_model(torch.device(device))
        # The NAR/projection modules were populated by hand above;
        # DefaultModelLoader.track_weights_loading diffs every named
        # parameter against the returned set, so the side keys must be
        # reported too or the loader flags them as uninitialized.
        return loaded | {name for name, _ in remapped}

    # ------------------------------------------------------------ hooks

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def prepare_runner_inputs(
        self,
        *,
        req_ids: list[str],
        num_computed_tokens: Any,
        num_scheduled_tokens: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bind rows to requests for the step; prompt tokens are captured in
        ``forward``, where the state is created (see ``_capture_constants``).
        Decode rows feed tokens the model itself sampled, so they never
        contribute to the prompt prefix."""
        computed = [int(v) for v in num_computed_tokens]
        scheduled = [int(v) for v in num_scheduled_tokens]
        self._step_rows = [(str(rid), computed[i], scheduled[i]) for i, rid in enumerate(req_ids)]
        # make_omni_output runs every step; without this reset the first step's
        # payload would stay live forever and the _audio_queue fallback in
        # _ship_audio could never trigger.
        self._last_mm = None
        return input_ids, positions

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del intermediate_tensors
        self._capture_constants(kwargs, input_ids)
        hidden = self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        self._flush_deferred_cleanup()
        return hidden

    def _capture_constants(self, kwargs: dict[str, Any], input_ids: torch.Tensor | None) -> None:
        """Create per-request state on the request's FIRST scheduled step.

        With KV prefix caching the first step may arrive with ``comp > 0``
        (a shared head was cached by an earlier request), so ``comp == 0``
        cannot be the trigger. The scheduled ``input_ids`` slice then holds
        only the uncached tail, so the driver ships the full prompt ids in
        the request's extra args (KEY_PREFIX_IDS); they are the source of
        truth for the NAR conditioning prefix and for the prompt length the
        sampler uses to tell a completing prefill row from a mid-chunk one.
        """
        extra_args = kwargs.get("sampling_extra_args")
        if extra_args is None or input_ids is None:
            return
        offset = 0
        for row, (req_id, comp, span) in enumerate(self._step_rows):
            if row >= len(extra_args):
                break
            if req_id in self._states:
                offset += span
                continue
            args = extra_args[row] or {}
            prefix = args.get(KEY_PREFIX_IDS)
            if isinstance(prefix, (list, tuple)) and prefix:
                prefix_ids = [int(v) for v in prefix]
                prompt_len = len(prefix_ids)
            elif comp == 0 and span >= 1:
                # Legacy path (no ids in extra args): only correct when the
                # whole prompt is scheduled in one chunk with no cache hit.
                prefix_ids = input_ids[offset : offset + span].tolist()
                prompt_len = len(prefix_ids)
            else:
                logger.warning(
                    "YuE2 request %s arrived with %d cached tokens but no "
                    "yue2_prefix_ids in its extra args; cannot rebuild its "
                    "full prompt (prefix-cache hit on the first request?).",
                    req_id,
                    comp,
                )
                offset += span
                continue
            phase = str(args.get(KEY_PHASE, "semantic"))
            preset = ABC_SAMPLING if phase == "abc" else SEMANTIC_SAMPLING
            seed = args.get(KEY_SEED, DEFAULT_SEED)
            constants = _RowConstants(
                request_id=req_id,
                phase=phase,
                temperature=float(args.get(KEY_TEMPERATURE, preset["temperature"])),
                top_p=float(args.get(KEY_TOP_P, preset["top_p"])),
                top_k=int(args.get(KEY_TOP_K, preset["top_k"])),
                repetition_penalty=float(args.get(KEY_REPETITION_PENALTY, preset["repetition_penalty"])),
                penalty_window=int(args.get(KEY_PENALTY_WINDOW, preset["penalty_window"])),
                min_tokens=int(args.get(KEY_MIN_TOKENS, preset["min_tokens"])),
                max_audio_frames=int(args.get(KEY_MAX_AUDIO_FRAMES, preset["max_tokens"])),
                seed=int(seed),
                skip_synthesis=bool(args.get(KEY_SKIP_SYNTHESIS, phase == "abc")),
            )
            device = input_ids.device
            generator = torch.Generator(device=device if device.type != "mps" else "cpu")
            generator.manual_seed(constants.seed)
            self._row_constants[req_id] = constants
            self._states[req_id] = _RequestState(
                request_id=req_id,
                constants=constants,
                prompt_tokens=prompt_len,
                prompt_len=prompt_len,
                prefix_ids=prefix_ids,
                generator=generator,
            )
            offset += span

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    # ------------------------------------------------------------ sampler

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> SamplerOutput | None:
        """Draw one token per decoding row with the request's own arithmetic."""
        del sampling_metadata
        rows = int(logits.shape[0])
        # The runner's input_ids buffer is int32; long ids crash its scatter
        # when concurrent requests re-index rows.
        token_ids = torch.zeros((rows, 1), dtype=torch.int32, device=logits.device)
        if not self._step_rows:
            return SamplerOutput(sampled_token_ids=token_ids, logprobs_tensors=None)

        for row, (req_id, comp, span) in enumerate(self._step_rows):
            if row >= rows:
                break
            state = self._states.get(req_id)
            if state is None or state.finished:
                continue
            if comp + span < state.prompt_len:
                # Mid-chunk prefill rows: their sampled token is discarded.
                # (A completing prefill row reaches the full prompt length
                # even when a prefix-cache hit left comp > 0.)
                continue
            constants = state.constants
            if state.request_id not in self._decode_t0:
                self._decode_t0[state.request_id] = time.perf_counter()
            scores = distribution(
                logits[row],
                temperature=constants.temperature,
                top_p=constants.top_p,
                top_k=constants.top_k,
                repetition_penalty=constants.repetition_penalty,
                penalty_window=constants.penalty_window,
                history=state.history,
                step=len(state.history),
                min_tokens=constants.min_tokens,
                phase=constants.phase,
            )
            token = sample_row(scores, state.generator, greedy=constants.temperature == 0)
            end = ABC_END if constants.phase == "abc" else MUSIC_END
            if token == end:
                state.finished = True
                state.truncated = False
                self._finish_request_safely(state, hit_end=True)
                token_ids[row, 0] = end
            else:
                state.history.append(token)
                budget = constants.max_audio_frames
                if constants.phase == "semantic" and len(state.history) >= budget:
                    state.finished = True
                    state.truncated = True
                    self._finish_request_safely(state, hit_end=False)
                    token_ids[row, 0] = end
                else:
                    token_ids[row, 0] = token
        return SamplerOutput(sampled_token_ids=token_ids, logprobs_tensors=None)

    # ------------------------------------------------------------ audio

    def _finish_request_safely(self, state: _RequestState, *, hit_end: bool) -> None:
        """Keep an NAR/VAE failure from escaping ``sample``.

        The runner calls ``model.sample()`` with no error handling, so an
        exception here (OOM in the ODE solve, non-finite VAE output) would
        kill Stage-0 and take every live request down with it. Fail only
        this request: ship an empty error-flagged clip that serving turns
        into a 500.
        """
        try:
            self._finish_request(state, hit_end=hit_end)
        except Exception:
            logger.exception("YuE2 synthesis failed for req=%s; failing only this request", state.request_id)
            self._ship_audio(state.request_id, torch.zeros((2, 0)), state.truncated, error=True)

    def _finish_request(self, state: _RequestState, *, hit_end: bool) -> None:
        """Solve the ODE and decode the song; abc-phase requests skip this."""
        constants = state.constants
        if constants.skip_synthesis or not state.history:
            return
        codec = [t - CODEC_OFFSET for t in state.history]
        if min(codec) < 0 or max(codec) >= CODEC_SIZE:
            raise RuntimeError("semantic history contains non-codec tokens")
        t0 = self._decode_t0.pop(state.request_id, None)
        t_nar0 = time.perf_counter()
        latents = synthesize(self, state.prefix_ids, codec, constants.seed, steps=ODE_STEPS)
        logger.info(
            "YUE2_LATENTS frames=%d absmax=%.4f mean=%.5f std=%.4f",
            latents.shape[0],
            float(latents.abs().max()),
            float(latents.mean()),
            float(latents.std()),
        )
        t_nar = time.perf_counter() - t_nar0
        t_vae0 = time.perf_counter()
        audio = self._decode_latents(latents)
        t_vae = time.perf_counter() - t_vae0
        t_ar = (t_nar0 - t0) if t0 is not None else float("nan")
        logger.info(
            "YUE2_TIMING req=%s phase=%s ar_tokens=%d ar_s=%.2f ar_tps=%.1f nar_s=%.2f vae_s=%.2f",
            state.request_id,
            constants.phase,
            len(state.history),
            t_ar,
            len(state.history) / t_ar if t_ar else float("nan"),
            t_nar,
            t_vae,
        )
        self._ship_audio(state.request_id, audio, state.truncated)
        del hit_end

    def _ship_audio(self, req_id: str, audio: torch.Tensor, truncated: bool, *, error: bool = False) -> None:
        """Append the finished song to the CURRENT step's mm payload in place.

        make_omni_output already ran for this step (it ships the sparse marker
        with empty lists), and the runner assembles per-request payloads only
        after sample(), so mutating the same dict here is picked up by the
        sparse routing on the request's last step in the output batch. Leaving
        the payload for a later step would drop it: the request is gone by
        then (gepard_talker flushes at its last step for the same reason).
        """
        mm = self._last_mm
        if mm is None:
            # No forward built a payload this step; fall back to the queue.
            self._audio_queue.append((req_id, audio, truncated, error))
            return
        mm["model_outputs"].append(audio)
        mm["sr"].append(torch.tensor(SAMPLE_RATE, dtype=torch.int32))
        meta = mm["meta"]
        meta["req_id"].append(req_id)
        # Keep the flags ints, not strs: the wire payload is tensor-only
        # (_ensure_tensor_values) and a string would be dropped before serving.
        meta["truncated"].append(int(truncated))
        meta["error"].append(int(error))

    def _vae_model(self, device: torch.device) -> YuE2VAE:
        if self._vae is None:
            vae_path = os.environ.get("YUE2_VAE", DEFAULT_VAE_ID)
            logger.info("Loading YuE2 VAE decoder from %s", vae_path)
            # Keep the VAE out of the module tree (__dict__, not an nn.Module
            # attribute): the checkpoint has no VAE keys, so a registered
            # submodule would trip DefaultModelLoader.track_weights_loading,
            # and a model-wide dtype/device pass would change VAE numerics.
            self.__dict__["_vae"] = YuE2VAE.from_pretrained(vae_path, decoder_only=True, device="cpu")
        if self._vae_device != device:
            self._vae.to(device)
            self._vae_device = device
        return self._vae

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """[frames, 64] FP32 CPU latents -> channels-first stereo [2, samples]."""
        device = f"cuda:{torch.accelerator.current_device_index()}" if torch.cuda.is_available() else "cpu"
        model = self._vae_model(torch.device(device))
        z = latents.T.unsqueeze(0)  # [1, 64, T]
        with torch.inference_mode():
            audio = model.decode_tiled(
                z,
                core_frames=VAE_CORE_FRAMES,
                halo_frames=VAE_HALO_FRAMES,
                output_device="cpu",
            )
        if not torch.isfinite(audio).all():
            raise RuntimeError("VAE produced non-finite audio")
        logger.info("YUE2_AUDIO_PRECLAMP absmax=%.4f", float(audio.abs().max()))
        # Channels-first [2, T], the same layout MiniMax Music 3 ships in
        # model_outputs; serving's create_audio transposes it for soundfile.
        return audio[0].float().clamp(-1, 1).contiguous()

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        by_req: dict[str, torch.Tensor] = {}
        truncated: dict[str, bool] = {}
        failed: dict[str, bool] = {}
        for req_id, audio, is_truncated, is_error in self._audio_queue:
            by_req[req_id] = audio
            truncated[req_id] = is_truncated
            failed[req_id] = is_error
        self._audio_queue.clear()
        ready = list(by_req)
        sr = torch.tensor(SAMPLE_RATE, dtype=torch.int32)
        # The sparse marker rides EVERY step, even ones that decoded nothing
        # (req_id=[] is the legal zero-audio shape): without it the runner
        # falls back to the dense pooler payload, whose per-step hidden states
        # land under this stage's "audio" key and the driver receives hidden
        # states instead of the decoded waveform (gepard_talker precedent).
        mm: dict[str, Any] = {
            "model_outputs": [by_req[r] for r in ready],
            "sr": [sr for _ in ready],
            "meta": {
                "req_id": ready,
                "sparse_audio": ["1"],
                "truncated": [int(truncated[r]) for r in ready],
                "error": [int(failed[r]) for r in ready],
            },
        }
        # sample() runs after this each step and appends finished songs into
        # this very dict (see _ship_audio); keep the handle for it.
        self._last_mm = mm
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=mm)

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        # Fires before forward; defer the free so the in-flight step can read.
        for rid in finished_req_ids:
            self._deferred_cleanup_ids.add(_request_key(rid))

    def _flush_deferred_cleanup(self) -> None:
        for req_id in self._deferred_cleanup_ids:
            state = self._states.pop(req_id, None)
            if state is not None and not state.finished:
                logger.warning(
                    "YuE2 request %s ended without an end token or its frame "
                    "budget (aborted or preempted); no audio was produced.",
                    req_id,
                )
            self._row_constants.pop(req_id, None)
        self._deferred_cleanup_ids.clear()
