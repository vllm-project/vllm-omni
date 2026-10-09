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
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
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
so the frame budget divided by 25 is the maximum duration in seconds.
The engine token budget also includes synthesis HOLD and terminal tokens.
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
# Emitted in place of a freshly drawn MUSIC_END: the sampler reads its draws
# back one step late, so the request stays alive for one more step, in which
# the song is synthesized and MUSIC_END is emitted.
HOLD_TOKEN = LATENT_PAD

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
# How many steps a finishing song may emit HOLD_TOKEN while its NAR/VAE pass
# runs on a side stream (0: synthesize synchronously in the finishing step).
# The caller must leave that many tokens of max_tokens headroom.
KEY_MAX_HOLD_STEPS = "yue2_max_hold_steps"
# What the serving adapter and the offline example grant. A song holds while
# it waits for, and runs on, the side stream; under load the wait behind
# earlier songs dominates (about a thousand steps with 32-48 songs in flight
# on one H200). A request out of budget waits for its song in that step.
SYNTHESIS_HOLD_STEPS = 4096


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
torch backend treats those the same, so a request samples a single row.

Rows sharing a preset are sampled as one batch over their phase's columns
only, as a CUDA graph per row count; the multinomial is drawn the way
``torch.multinomial`` draws one sample, so each request's random stream and
every drawn token match the per-row reference arithmetic.
"""


# The semantic phase may only emit MUSIC_END or a codec token, and MUSIC_END
# sits right below the codec span, so its whole vocabulary is one slice.
assert MUSIC_END + 1 == CODEC_OFFSET


class PhaseVocab:
    """Columns one phase may sample, in increasing token-id order.

    Phase masking sets every other column to -inf, so dropping them changes
    no finite score, and keeping the id order keeps sort/argmax tie breaking
    the same as on the full vocabulary.
    """

    def __init__(self, phase: str):
        self.phase = phase
        if phase == "abc":
            self.end_col = EOD  # columns [0, EOD) are ids; column EOD is ABC_END
            self.num_cols = EOD + 1
        else:
            self.end_col = 0  # column c is id MUSIC_END + c
            self.num_cols = CODEC_OFFSET + CODEC_SIZE - MUSIC_END
        self._index: dict[torch.device, torch.Tensor] = {}

    def take(self, rows: torch.Tensor) -> torch.Tensor:
        """Allowed columns of ``rows`` [R, vocab] as a new float32 tensor."""
        if self.phase == "abc":
            index = self._index.get(rows.device)
            if index is None:
                index = torch.cat((torch.arange(EOD), torch.tensor([ABC_END]))).to(rows.device)
                self._index[rows.device] = index
            return rows.index_select(-1, index).float()
        return rows[..., MUSIC_END : CODEC_OFFSET + CODEC_SIZE].to(torch.float32, copy=True)

    def col(self, token_id: int) -> int:
        if self.phase == "abc":
            return self.end_col if token_id == ABC_END else token_id
        return token_id - MUSIC_END

    def ids(self, cols: torch.Tensor) -> torch.Tensor:
        if self.phase == "abc":
            return torch.where(cols == self.end_col, ABC_END, cols)
        return cols + MUSIC_END


PHASE_VOCAB = {"abc": PhaseVocab("abc"), "semantic": PhaseVocab("semantic")}


def _sample_core(
    logits: torch.Tensor,
    noise: torch.Tensor,
    vocab: PhaseVocab,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    window_cols: torch.Tensor | None,
    block_end: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked, penalized, shaped scores and the multinomial draw, given the
    Exp(1) noise [R, vocab]."""
    rows = logits.shape[0]
    scores = vocab.take(logits)
    if block_end is not None:
        scores[:, vocab.end_col].masked_fill_(block_end, -torch.inf)
    if repetition_penalty != 1.0 and window_cols is not None:
        # The padding column is counted, then dropped.
        freq = torch.zeros((rows, vocab.num_cols + 1), dtype=scores.dtype, device=scores.device)
        freq.scatter_add_(-1, window_cols, torch.ones_like(window_cols, dtype=scores.dtype))
        alpha = repetition_penalty ** freq[:, : vocab.num_cols]
        scores = torch.where(scores < 0, scores * alpha, scores / alpha)
    bad = ~torch.isfinite(scores).any(-1)
    if temperature == 0:
        return vocab.ids(scores.argmax(-1)), bad
    if temperature != 1:
        scores = scores / temperature
    threshold = scores.topk(min(top_k, vocab.num_cols)).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if top_p < 1:
        values, indices = scores.sort(descending=True, stable=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > top_p
        removed[..., :1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    probabilities = scores.softmax(-1)
    return vocab.ids((probabilities / vocab.take(noise)).argmax(-1)), bad


def draw_noise(noise: torch.Tensor, generators: Sequence[torch.Generator]) -> torch.Tensor:
    """Row r = the Exp(1) draw ``torch.multinomial`` makes from generator r."""
    for row, generator in enumerate(generators):
        noise[row].exponential_(1, generator=generator)
    return noise


def sample_rows(
    logits: torch.Tensor,
    vocab: PhaseVocab,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    window_cols: torch.Tensor | None,
    block_end: torch.Tensor | None,
    generators: list[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one token per row for rows sharing one preset.

    Works on the phase's allowed columns only and never synchronizes: every
    input is already on the device (``window_cols`` [R, W] are the recent
    ids as vocab columns, padded with ``vocab.num_cols``; ``block_end`` [R]
    masks the end token) and it returns device tensors (token ids [R],
    all-masked flags [R]). The multinomial draw is written out the way
    ``torch.multinomial`` draws one sample (``argmax(p / q)`` with
    ``q ~ Exp(1)`` over the full vocabulary), so each request's generator
    advances exactly as in the per-row path.
    """
    noise = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    draw_noise(noise, generators)
    return _sample_core(logits, noise, vocab, temperature, top_p, top_k, repetition_penalty, window_cols, block_end)


class SamplerGraph:
    """``sample_rows`` for a row-count bucket and preset, replayed as one graph.

    The sampler is ~30 small kernels per step, each a host-side launch; in a
    batch-1 decode step that host time exceeds the backbone's GPU time. The
    Exp(1) noise is still drawn per row from each request's own generator
    outside the graph, and an all-False ``block_end`` / padded ``window_cols``
    leave every score as the eager path computes it, so draws are unchanged.
    """

    def __init__(self, vocab, rows, vocab_size, logits_dtype, preset, window, device, pool):
        self.vocab = vocab
        self.preset = preset  # (temperature, top_p, top_k, repetition_penalty)
        self.logits = torch.zeros((rows, vocab_size), dtype=logits_dtype, device=device)
        self.noise = torch.ones((rows, vocab_size), dtype=torch.float32, device=device)
        self.block_end = torch.zeros(rows, dtype=torch.bool, device=device)
        self.window_cols = (
            torch.full((rows, window), vocab.num_cols, dtype=torch.long, device=device) if window else None
        )
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(2):
                self._run()
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool, capture_error_mode="thread_local"):
            self.ids, self.bad = self._run()

    def _run(self):
        return _sample_core(self.logits, self.noise, self.vocab, *self.preset, self.window_cols, self.block_end)

    def __call__(self, logits, window_cols, block_end, generators):
        rows = logits.shape[0]
        if rows > self.logits.shape[0] or len(generators) != rows:
            raise ValueError("Sampler inputs exceed the captured bucket or lack a request generator")
        # A replay reads its inputs from the addresses it was captured with.
        # Padding rows have independent arithmetic and never advance a real
        # request's generator. Only the active output rows are returned.
        self.logits[:rows].copy_(logits)
        draw_noise(self.noise, generators)
        if self.window_cols is not None:
            self.window_cols[:rows].copy_(window_cols)
        if block_end is None:
            self.block_end.zero_()
        else:
            self.block_end[:rows].copy_(block_end)
        self.graph.replay()
        return self.ids[:rows], self.bad[:rows]


# -------------------- NAR flow-matching pass --------------------


"""Acoustic flow matching for YuE2-3B, ported from upstream ``nar.py``.

One AR prefill per original chunk, then 32 midpoint ODE steps that walk only
the NAR path: NAR positions attend the full AR prefix plus all NAR positions
bidirectionally. The prefill re-uses the vLLM backbone's own projections and
writes the prefix K/V once into a per-layer buffer; every ODE evaluation
writes its NAR K/V right after it and attends one contiguous sequence. An
evaluation is FlashAttention-3 (SDPA without it) plus the per-layer halves in
``_nar_pre_attention``/``_nar_post_attention`` (compiled), and after the
first call a CUDA graph replay: all 64 evaluations of a chunk share shapes.
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


def _attention(q, k, v, *, causal: bool) -> torch.Tensor:
    """SDPA over [tokens, heads, dim] Q/K/V (grouped K/V heads allowed)."""
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        is_causal=causal,
        enable_gqa=q.shape[1] != k.shape[1],
    )
    return out[0].transpose(0, 1)


def _drain(work: Iterator[None]) -> torch.Tensor:
    """Run a work generator to completion and return its result."""
    while True:
        try:
            next(work)
        except StopIteration as done:
            return done.value


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Same arithmetic as _RMSNorm.forward.
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype) * weight


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """_apply_rotary for [T, heads, dim] with [T, dim/2] cos/sin."""
    return _apply_rotary(x, cos[:, None, :], sin[:, None, :])


def _nar_pre_attention(x, w_norm, w_qkv, w_q_norm, w_k_norm, cos, sin, eps: float, heads: int, kv_heads: int):
    """input norm -> fused q/k/v projection -> q/k norm -> RoPE, for [T, hidden]."""
    tokens = x.shape[0]
    head_dim = w_q_norm.shape[0]
    qkv = F.linear(_rms(x, w_norm, eps), w_qkv)
    q, k, v = qkv.split([heads * head_dim, kv_heads * head_dim, kv_heads * head_dim], dim=-1)
    q = _rope(_rms(q.reshape(tokens, heads, head_dim), w_q_norm, eps), cos, sin)
    k = _rope(_rms(k.reshape(tokens, kv_heads, head_dim), w_k_norm, eps), cos, sin)
    return q, k, v.reshape(tokens, kv_heads, head_dim)


def _nar_post_attention(x, attn, w_o, w_norm, w_gate_up, w_down, eps: float):
    """o_proj + residual -> pre-MLP norm -> fused gate/up SiLU MLP + residual."""
    x = x + F.linear(attn.flatten(1), w_o)
    gate, up = F.linear(_rms(x, w_norm, eps), w_gate_up).chunk(2, dim=-1)
    return x + F.linear(F.silu(gate) * up, w_down)


_COMPILED: dict[str, Callable] = {}


def _nar_kernels() -> tuple[Callable, Callable]:
    """The per-layer NAR halves, compiled once for any token count.

    Compilation fuses the RMSNorm/RoPE/SiLU elementwise chains that dominate
    the eager NAR pass after its GEMMs and attention.
    """
    if "pre" not in _COMPILED:
        pre, post = _nar_pre_attention, _nar_post_attention
        if torch.cuda.is_available():
            pre = torch.compile(pre, dynamic=True, fullgraph=True, options=_COMPILE_OPTIONS)
            post = torch.compile(post, dynamic=True, fullgraph=True, options=_COMPILE_OPTIONS)
        _COMPILED.update(pre=pre, post=post)
    return _COMPILED["pre"], _COMPILED["post"]


def _flash_attention_fn() -> Callable | None:
    """FlashAttention-3 varlen from vLLM on Hopper, else None (use SDPA)."""
    if "fa3" not in _COMPILED:
        fn = None
        if torch.cuda.is_available():
            try:
                from vllm.vllm_flash_attn import flash_attn_interface, flash_attn_varlen_func
            except ImportError:
                pass
            else:
                if flash_attn_interface.is_fa_version_supported(3):
                    fn = flash_attn_varlen_func
        _COMPILED["fa3"] = fn
    return _COMPILED["fa3"]


def _sequence_bounds(length: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([0, length], dtype=torch.int32).to(device, non_blocking=True)


def _flash_attention(fa3, q, k, v, *, causal: bool, bounds_q=None, bounds_k=None) -> torch.Tensor:
    """One sequence of [tokens, heads, dim] Q/K/V (grouped K/V heads allowed).

    ``fa3`` is FlashAttention-3 varlen, or None for PyTorch SDPA.
    """
    if fa3 is None:
        return _attention(q, k, v, causal=causal)
    bounds_q = _sequence_bounds(q.shape[0], q.device) if bounds_q is None else bounds_q
    bounds_k = _sequence_bounds(k.shape[0], q.device) if bounds_k is None else bounds_k
    return fa3(
        q,
        k,
        v,
        max_seqlen_q=q.shape[0],
        cu_seqlens_q=bounds_q,
        max_seqlen_k=k.shape[0],
        cu_seqlens_k=bounds_k,
        causal=causal,
        fa_version=3,
    )


def fuse_nar_projections(layers: nn.ModuleList) -> list[tuple[torch.Tensor, ...]]:
    """Per NAR layer: norm/qkv/q-norm/k-norm/o/norm/gate-up/down weights.

    q/k/v and gate/up are concatenated so each runs as one GEMM; the original
    ``nn.Linear`` weights become views into the fused tensors, so the module
    tree stays intact at no extra memory.
    """
    fused = []
    for layer in layers:
        attn, mlp = layer.self_attn, layer.mlp
        projections = (attn.q_proj, attn.k_proj, attn.v_proj)
        qkv = torch.cat([p.weight.data for p in projections])
        offset = 0
        for p in projections:
            rows = p.weight.shape[0]
            p.weight.data = qkv[offset : offset + rows]
            offset += rows
        gate_up = torch.cat([mlp.gate_proj.weight.data, mlp.up_proj.weight.data])
        rows = mlp.gate_proj.weight.shape[0]
        mlp.gate_proj.weight.data = gate_up[:rows]
        mlp.up_proj.weight.data = gate_up[rows:]
        fused.append(
            (
                layer.input_layernorm.weight,
                qkv,
                attn.q_norm.weight,
                attn.k_norm.weight,
                attn.o_proj.weight,
                layer.pre_mlp_layernorm.weight,
                gate_up,
                mlp.down_proj.weight,
            )
        )
    return fused


class CachedNAR:
    """One acoustic chunk: AR prefix K/V prefilled once, ODE walks the NAR path."""

    def __init__(self, model, chunk: Chunk, pool: tuple[int, int] | None = None):
        self.model = model
        self.chunk = chunk
        self.pool = pool
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
        cos, sin = model.rotary(positions)
        self.cos, self.sin = cos[0], sin[0]  # [NAR, dim/2]
        local = torch.arange(self.nar_length, device=self.device).clamp(max=model.max_latent_frames - 1)
        self.pos_emb = model.latent_pos_embed(local)[None]
        self.fa3 = _flash_attention_fn()
        self.bounds_q = _sequence_bounds(self.nar_length, self.device)
        self.bounds_k = _sequence_bounds(self.ar_length + self.nar_length, self.device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.last_use: torch.cuda.Event | None = None
        self._prefill()

    def _prefill(self):
        model = self.model
        backbone = model.model
        config = model.config
        total = self.ar_length + self.nar_length
        shape = (len(backbone.layers), total, int(config.num_key_value_heads), int(config.head_dim))
        self.keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        self.values = torch.empty(shape, dtype=self.dtype, device=self.device)
        ids = torch.tensor(self.chunk.ar_tokens, dtype=torch.long).to(self.device, non_blocking=True)
        cos, sin = model.rotary(torch.arange(self.ar_length, device=self.device)[None])
        x = backbone.embed_tokens(ids)  # [T, H]
        last = len(backbone.layers) - 1
        for index, layer in enumerate(backbone.layers):
            # The layernorm is load-bearing: without it the cached AR K/V
            # conditioning is garbage and the ODE latents blow up (clipped audio).
            q, k, v = model.ar_project(layer, layer.input_layernorm(x), cos, sin)
            self.keys[index, : self.ar_length] = k
            self.values[index, : self.ar_length] = v
            if index == last:
                break  # the last layer's output feeds nothing
            h = _flash_attention(self.fa3, q, k, v, causal=True)
            x = x + _linear(layer.self_attn.o_proj, h.flatten(1))
            x = x + _linear(layer.mlp, layer.post_attention_layernorm(x))

    @torch.inference_mode()
    def velocity(self, state: torch.Tensor, raw_t: float) -> torch.Tensor:
        """One ODE evaluation; from the second call on, a CUDA graph replay.

        After one eager call (compilation, cuBLAS/FA3 setup) the evaluation is
        captured once and replayed from static state/time inputs, replacing
        ~30 host-side launches per layer with one launch per evaluation.
        """
        if self.device.type != "cuda":
            return self._velocity(state, torch.tensor(raw_t, dtype=self.dtype, device=self.device))
        if self.graph is None:
            if not hasattr(self, "state_in"):
                self.state_in = torch.empty_like(state)
                self.time_in = torch.empty((), dtype=self.dtype, device=self.device)
                self.state_in.copy_(state)
                self.time_in.fill_(raw_t)
                return self._velocity(self.state_in, self.time_in)
            self.graph = torch.cuda.CUDAGraph()
            if torch.cuda.current_stream(self.device) == torch.cuda.default_stream(self.device):
                with torch.cuda.graph(self.graph, pool=self.pool, capture_error_mode="thread_local"):
                    self.velocity_out = self._velocity(self.state_in, self.time_in)
            else:
                # Already on a side stream (the async synthesis path): capture
                # in place, without torch.cuda.graph's device-wide sync, which
                # would stall the decode loop behind every queued kernel.
                self.graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
                try:
                    self.velocity_out = self._velocity(self.state_in, self.time_in)
                finally:
                    self.graph.capture_end()
        # The graph reads its inputs from the addresses it was captured with.
        self.state_in.copy_(state)
        self.time_in.fill_(raw_t)
        self.graph.replay()
        return self.velocity_out

    def _velocity(self, state: torch.Tensor, raw_t: torch.Tensor) -> torch.Tensor:
        model = self.model
        pre, post = _nar_kernels()
        config = model.config
        eps = float(config.rms_norm_eps)
        heads, kv_heads = int(config.num_attention_heads), int(config.num_key_value_heads)
        x_nar = F.pad(state, (0, 0, 1, 1))
        # The timestep shift, from a device scalar so a graph replay can update it.
        t_sig = torch.sigmoid(raw_t)
        shift = model._timestep_shift
        shifted = shift * t_sig / (1 + (shift - 1) * t_sig)
        x = model.vae2llm(x_nar[None])
        x = x + model.time_embedder(shifted.expand(self.nar_length))[None]
        x = (x + self.pos_emb)[0]
        for index, weights in enumerate(model.nar_weights()):
            w_norm, w_qkv, w_q_norm, w_k_norm, w_o, w_mlp_norm, w_gate_up, w_down = weights
            q, k, v = pre(x, w_norm, w_qkv, w_q_norm, w_k_norm, self.cos, self.sin, eps, heads, kv_heads)
            self.keys[index, self.ar_length :] = k
            self.values[index, self.ar_length :] = v
            h = _flash_attention(
                self.fa3,
                q,
                self.keys[index],
                self.values[index],
                causal=False,
                bounds_q=self.bounds_q,
                bounds_k=self.bounds_k,
            )
            x = post(x, h, w_o, w_mlp_norm, w_gate_up, w_down, eps)
        return model.llm2vae(model.model.norm(x[None]))[0, 1:-1]

    def solve_iter(self, steps: int = 32) -> Iterator[None]:
        """The ODE solve as a work generator: yields after enqueuing each
        evaluation and returns the (still in flight) device latents."""
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("steps must be a positive integer")
        # Non-blocking: a blocking copy would wait for the queued prefill.
        state = self.chunk.noise.to(device=self.device, dtype=self.dtype, non_blocking=True)
        dt = 1.0 / steps
        for step in range(steps):
            t = 1.0 - step * dt
            raw = torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            first = self.velocity(state, raw)
            yield
            mid = state - first * (dt / 2)
            raw_mid = torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()
            state = state - self.velocity(mid, raw_mid) * dt
            yield
        return state.float()

    @torch.inference_mode()
    def solve(self, steps: int = 32) -> torch.Tensor:
        result = _drain(self.solve_iter(steps)).cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result

    def close(self):
        self.keys = self.values = self.graph = None
        self.cos = self.sin = self.pos_emb = None
        self.state_in = self.time_in = self.velocity_out = None
        self.bounds_q = self.bounds_k = None
        self.chunk = None


class _NARGraphPool:
    """Allocator pool shared by graphs on the serialized synthesis stream."""

    def __init__(self) -> None:
        self.pool: tuple[int, int] | None = None
        self._anchor: torch.cuda.CUDAGraph | None = None
        self._anchor_buffer: torch.Tensor | None = None

    def handle(self, device: torch.device) -> tuple[int, int]:
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
            # The handle alone does not keep the allocator pool alive. A tiny
            # graph owns it after a chunk is released, including across aborts.
            self._anchor = torch.cuda.CUDAGraph()
            self._anchor.capture_begin(pool=self.pool, capture_error_mode="thread_local")
            try:
                self._anchor_buffer = torch.zeros((), dtype=torch.int32, device=device)
            finally:
                self._anchor.capture_end()
        return self.pool


def synthesize_iter(
    model, prefix: Sequence[int], codec: Sequence[int], seed: int, keep: list, graph_pool: _NARGraphPool | None = None
) -> Iterator[None]:
    """``synthesize`` as a work generator (one unit per prefill or ODE
    evaluation); returns the in-flight device latents, unchecked. The solver
    engines go to ``keep``: their graphs and buffers must outlive the work."""
    output = []
    for chunk in song_chunks(prefix, codec, seed):
        # Graphs share allocator storage only on this serialized stream.
        # Their state and K/V are retired after each chunk, with no engine cache.
        pool = None if graph_pool is None else graph_pool.handle(next(model.vae2llm.parameters()).device)
        engine = CachedNAR(model, chunk, pool)
        keep.append(engine)
        yield
        output.append((yield from engine.solve_iter(ODE_STEPS)))
        engine.last_use = torch.cuda.Event()
        engine.last_use.record()
    return torch.cat(output, dim=0)


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


_SNAKE: list[Callable] = []
# Inductor otherwise benchmarks a few launch configs per reduction kernel and
# keeps the fastest; timing noise then changes the reduction order, and with it
# the song's last bits, from one process to the next. Deterministic mode keeps
# "same seed, same song" across server restarts.
_COMPILE_OPTIONS = {"deterministic": True}


def _snake_kernel() -> Callable:
    """snake_beta fused into one kernel: it runs on every [C, samples] FP32
    activation of the decoder, where each unfused elementwise op is a full
    memory pass."""
    if not _SNAKE:
        fused = torch.cuda.is_available()
        _SNAKE.append(
            torch.compile(snake_beta, dynamic=True, fullgraph=True, options=_COMPILE_OPTIONS) if fused else snake_beta
        )
    return _SNAKE[0]


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
        if x.is_cuda and not self.training:
            return _snake_kernel()(x, alpha, beta)
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

    def _latent(self, latent, check_finite=True):
        latent = torch.as_tensor(latent)
        if latent.ndim != 3 or latent.shape[1] != self.config.latent_dim or latent.shape[0] < 1 or latent.shape[-1] < 1:
            raise ValueError(f"Expected nonempty [B,{self.config.latent_dim},T] latents")
        # The check reads the values back, a device sync; an async caller
        # checks the decoded audio on the host instead.
        if check_finite and not torch.isfinite(latent).all():
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
    def decode(self, latent, check_finite=True):
        """Full waveform, FP32 [B,2,1920*T-64], without clipping."""
        latent = self._latent(latent, check_finite)
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
        check_finite=True,
    ):
        return _drain(
            self.decode_tiled_iter(latent, core_frames, halo_frames, output_device, on_progress, check_finite)
        )

    def decode_tiled_iter(
        self,
        latent,
        core_frames=None,
        halo_frames=None,
        output_device="cpu",
        on_progress: Callable[[int, int], None] | None = None,
        check_finite=True,
    ):
        """Decode bounded tiles, retaining exact cores with natural end length.

        Each crop has enough left/right context for every dependency. There is
        no crossfade or boundary smoothing, and no zero padding of final audio.
        CPU output prevents an entire song from accumulating on the GPU.
        ``on_progress(completed, total)`` runs after each existing crop copy;
        no extra synchronization is added. With a CUDA output device, queued
        work may still be executing. Callback exceptions propagate.
        """
        latent = self._latent(latent, check_finite)
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
            tile = self.decode(latent[..., left:right], check_finite)
            out_start, out_end = start * ratio, min(end * ratio, total)
            crop_start = (start - left) * ratio
            crop = tile[..., crop_start : crop_start + out_end - out_start]
            if crop.shape[-1] != out_end - out_start:
                raise RuntimeError("VAE tile did not cover its requested output core")
            audio[..., out_start:out_end].copy_(crop.to(output_device))
            del tile, crop
            if on_progress is not None:
                on_progress(tile_index + 1, tiles)
            yield
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
    max_hold_steps: int = 0


@dataclass
class _RequestState:
    """Everything one song request accumulates across decode steps."""

    request_id: str
    constants: _RowConstants
    prompt_len: int = 0  # full prompt length; a completing prefill row reaches it
    prefix_ids: list[int] = field(default_factory=list)
    history: list[int] = field(default_factory=list)
    generator: torch.Generator | None = None
    finished: bool = False
    truncated: bool = False
    # Set when the host learns (one step after the draw) that the request
    # drew its end token or filled its frame budget; the next scheduled step
    # runs the finishing pass and emits the end token.
    finish_ready: bool = False
    end_drawn: bool = False  # the engine counts a drawn end (or its HOLD) too
    job: _SynthesisJob | None = None  # async synthesis in flight
    hold_steps: int = 0  # HOLD_TOKEN steps emitted while it runs
    # Engine output count once the emitted end token is accepted, and the
    # shipped (audio, truncated, error). A preemption can drop the step that
    # delivered both; the resumed request then delivers them again.
    end_len: int | None = None
    delivered: tuple[torch.Tensor, bool, bool] | None = None


SongWork = Callable[[list], Generator[None, None, torch.Tensor]]


class _SynthesisJob:
    """One song's NAR + VAE pass on a side stream, fed a little per step.

    The pass is a generator of work units (the prefill, each ODE evaluation,
    each VAE tile). ``advance`` enqueues units only while fewer than
    ``MAX_INFLIGHT`` are queued ahead of the GPU and never waits: enqueuing
    the whole song at once fills the stream's launch queue, after which every
    launch blocks the host until the GPU drains, and the decode loop with it.
    The decode steps call ``advance`` every step, so the side stream stays
    busy while the other requests keep decoding; kernels of the two streams
    run concurrently within the process. The finishing request emits
    HOLD_TOKEN until ``done()``.
    """

    MAX_INFLIGHT = 2

    def __init__(self, request_id: str, song: SongWork):
        """``song(keep)`` yields once per enqueued work unit and returns the
        device audio [2, samples]; what it appends to ``keep`` stays alive
        until the job is released."""
        self.request_id = request_id
        self.song = song
        self.stream: torch.cuda.Stream | None = None
        self.work: Iterator[None] | None = None
        self.inflight: list[torch.cuda.Event] = []
        self.event: torch.cuda.Event | None = None  # recorded after the last unit
        self.audio: torch.Tensor | None = None  # pinned host copy
        self.finite: torch.Tensor | None = None  # pinned pre-clamp validity flag
        self.error: BaseException | None = None
        self.cancelled = False
        self._released = False
        # Solver engines (captured graphs, K/V buffers), latents and audio:
        # they must outlive the queued kernels that use them.
        self.keep: list[CachedNAR | torch.Tensor] = []
        self.t_start: float | None = None
        self.t_done: float | None = None

    @property
    def started(self) -> bool:
        return self.t_start is not None

    def _run(self) -> Iterator[None]:
        audio = yield from self.song(self.keep)
        # Check before clipping: clipping maps infinities to finite +/-1.
        # Both copies target pinned allocations and complete before self.event.
        self.finite = torch.empty((), dtype=torch.bool, pin_memory=True)
        self.finite.copy_(torch.isfinite(audio).all(), non_blocking=True)
        audio = audio.clamp(-1, 1)
        # Pinned, so the device-to-host copy stays async on the side stream.
        self.audio = torch.empty(audio.shape, dtype=audio.dtype, pin_memory=True)
        self.audio.copy_(audio, non_blocking=True)
        self.keep.append(audio)

    def start(self, stream: torch.cuda.Stream) -> None:
        self.stream = stream
        self.t_start = time.perf_counter()
        self.work = self._run()
        self.advance()

    def advance(self) -> None:
        """Enqueue work units up to MAX_INFLIGHT ahead of the GPU; never waits."""
        if self.work is None or self.event is not None or self.error is not None:
            return
        with torch.cuda.stream(self.stream), torch.inference_mode():
            while True:
                # Completed chunks need not keep all their K/V and private
                # graph pools alive until the whole song finishes.
                for item in self.keep[:]:
                    if isinstance(item, CachedNAR) and item.last_use is not None and item.last_use.query():
                        item.close()
                        self.keep.remove(item)
                while self.inflight and self.inflight[0].query():
                    self.inflight.pop(0)
                if len(self.inflight) >= self.MAX_INFLIGHT:
                    return
                try:
                    next(self.work)
                except StopIteration:
                    self.event = torch.cuda.Event()
                    self.event.record(self.stream)
                    self.inflight.clear()
                    return
                except (RuntimeError, ValueError) as exc:
                    # CUDA errors and OOM are RuntimeErrors; song_chunks
                    # rejects a bad song with ValueError. Either fails only
                    # this request (see finish); anything else is a bug.
                    self.error = exc
                    return
                unit = torch.cuda.Event()
                unit.record(self.stream)
                self.inflight.append(unit)

    def done(self) -> bool:
        if self.cancelled or self.error is not None:
            return True
        self.advance()
        finished = self.event is not None and self.event.query()
        if finished and self.t_done is None:
            self.t_done = time.perf_counter()
        return finished

    def drain(self) -> None:
        """Enqueue and wait for everything (a request out of hold budget)."""
        while self.work is not None and self.event is None and self.error is None:
            if self.inflight:
                self.inflight[0].synchronize()
            self.advance()

    def release(self) -> None:
        """Release once, after this job's last submitted kernel has finished.

        The queue may already be running a later job on the same stream when
        this job is delivered. Waiting on the stream would wait for that job.
        """
        if self._released:
            return
        if self.stream is not None and self.event is None:
            # An error or cancellation can precede the normal terminal event.
            self.event = torch.cuda.Event()
            self.event.record(self.stream)
        if self.event is not None:
            self.event.synchronize()
        if self.work is not None:
            self.work.close()
            self.work = None
        for item in self.keep:
            if isinstance(item, CachedNAR):
                item.close()
        self.keep.clear()
        self.stream = None
        self._released = True

    def cancel(self) -> None:
        """Stop submitting work and release the already submitted units.

        Waiting is bounded by MAX_INFLIGHT, not the rest of the song. Cleanup
        happens here even if the engine becomes idle after the cancellation.
        """
        self.cancelled = True
        self.release()

    def finish(self) -> tuple[torch.Tensor, bool]:
        """Wait if needed, release, and return (audio [2, T] on the CPU, failed)."""
        self.drain()
        self.release()
        if self.t_done is None:
            self.t_done = time.perf_counter()
        if self.cancelled:
            return torch.zeros((2, 0)), True
        if self.error is not None:
            logger.error("YuE2 synthesis failed for req=%s: %r", self.request_id, self.error)
            return torch.zeros((2, 0)), True
        if self.finite is None or not bool(self.finite):
            logger.error("YuE2 synthesis produced non-finite audio for req=%s", self.request_id)
            return torch.zeros((2, 0)), True
        return self.audio, False


class _SynthesisQueue:
    """Songs waiting for, or running on, the synthesis side stream.

    One song runs at a time: the NAR pass is compute-bound, so a second one
    would only double the memory. ``pump`` runs every decode step; it feeds
    the running song and starts the next once it is done.
    """

    def __init__(self) -> None:
        self._stream: torch.cuda.Stream | None = None
        self.graph_pool = _NARGraphPool()
        self.active: _SynthesisJob | None = None
        self.waiting: list[_SynthesisJob] = []

    def stream(self) -> torch.cuda.Stream:
        """The side stream, at the highest priority the device offers.

        Once several songs are in flight the synthesis pass, not the decode,
        bounds throughput, and its kernels leave no room to co-schedule the
        decode's anyway: they fill every SM. So synthesis goes first and the
        decode fills the gaps.
        """
        if self._stream is None:
            self._stream = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
        return self._stream

    def submit(self, job: _SynthesisJob) -> None:
        self.waiting.append(job)
        self.pump()

    def pump(self) -> None:
        """Feed the running song; start the next queued one once it is done."""
        if self.active is not None:
            if not self.active.done():  # done() also feeds it
                return
            self.active.release()  # frees its buffers; the audio stays
            self.active = None
        while self.waiting:
            job = self.waiting.pop(0)
            if job.error is not None:
                continue  # fails at completion, without touching the GPU
            job.start(self.stream())
            self.active = job
            return

    def complete(self, job: _SynthesisJob) -> tuple[torch.Tensor, bool, bool]:
        """Finish ``job`` now, ahead of the queue if it has not started.

        Returns (audio [2, T] on the CPU, failed, whether it had to wait).
        """
        if job.cancelled:
            return torch.zeros((2, 0)), True, False
        if job in self.waiting:
            self.waiting.remove(job)
        if not job.started and job.error is None:
            if self.active is not None:
                # Finish the running song first (its own request ships it).
                self.active.drain()
                self.active.release()
                self.active = None
            job.start(self.stream())
            self.active = job
        waited = not job.done()
        audio, failed = job.finish()
        if self.active is job:
            self.active = None
        self.pump()
        return audio, failed, waited

    def cancel(self, job: _SynthesisJob) -> None:
        """Drop an aborted song and release its bounded in-flight work."""
        if job in self.waiting:
            self.waiting.remove(job)
        job.cancel()
        if self.active is job:
            self.active = None


class Yue2ForCausalLM(nn.Module):
    """YuE2-3B: vLLM-native Qwen3 AR backbone + NAR flow-matching + VAE."""

    have_multimodal_outputs = True
    prefer_model_sampler = True
    has_postprocess = False
    # Request state is created in prepare_runner_inputs, which runs on every
    # step; forward() is skipped on a full CUDA graph replay.
    accepts_runner_sampling_extra_args = True
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
    # _ship_audio appends the song in sample(), after the runner saved the
    # step for the omni prefix cache; with prefix caching on, the payload is
    # built from these live outputs rather than the step snapshot.
    mm_outputs_written_in_sample: bool = True

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
        # Fused NAR projection weights (fuse_nar_projections); see nar_weights.
        self.nar_fused: list[tuple[torch.Tensor, ...]] | None = None
        self._vae: YuE2VAE | None = None
        self._vae_device: torch.device | None = None

        self._states: dict[str, _RequestState] = {}
        self._audio_queue: list[tuple[str, torch.Tensor, bool, bool]] = []
        self._step_rows: list[tuple[str, int, int]] = []  # (req_id, computed, scheduled)
        self._step_discard: list[bool] | None = None
        self._decode_t0: dict[str, float] = {}  # first decode-step wall time, for AR tok/s
        self._last_mm: dict[str, Any] | None = None  # current step's make_omni_output payload
        # Tokens drawn by the last sample() call, still on their way to the
        # host: (copy-done event, pinned [2, rows] ids/flags, states).
        self._pending: tuple[torch.Event, torch.Tensor, list[_RequestState]] | None = None
        self._pinned: torch.Tensor | None = None
        self._stage: torch.Tensor | None = None
        self._logits: torch.Tensor | None = None  # semantic-only logits buffer
        self._sampler_graphs: dict[tuple, SamplerGraph] = {}
        self._sampler_graph_pool: tuple[int, int] | None = None
        self._sampler_capacity = int(vllm_config.scheduler_config.max_num_seqs)
        self._max_model_len = int(vllm_config.model_config.max_model_len)
        self._synthesis = _SynthesisQueue()

    # ------------------------------------------------------------ sampling

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

    def nar_weights(self) -> list[tuple[torch.Tensor, ...]]:
        """The fused NAR layer weights, fused on first use.

        load_weights fuses them, but a dummy-weight load (load_format=dummy,
        as the CI smoke tests run) never calls it.
        """
        if self.nar_fused is None:
            self.nar_fused = fuse_nar_projections(self.nar_layers)
        return self.nar_fused

    def rotary(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb(positions)

    def synthesize_song(
        self, prefix_ids: Sequence[int], codec: Sequence[int], seed: int
    ) -> tuple[torch.Tensor, float, float]:
        """Channels-first stereo audio for one song, plus NAR and VAE seconds."""
        t0 = time.perf_counter()
        latents = synthesize(self, prefix_ids, codec, seed, steps=ODE_STEPS)
        logger.debug(
            "YUE2_LATENTS frames=%d absmax=%.4f mean=%.5f std=%.4f",
            latents.shape[0],
            float(latents.abs().max()),
            float(latents.mean()),
            float(latents.std()),
        )
        t1 = time.perf_counter()
        audio = self._decode_latents(latents)
        return audio, t1 - t0, time.perf_counter() - t1

    @torch.inference_mode()
    def _warm_up_synthesis(self) -> None:
        """Compile the NAR kernels and touch the VAE once at startup.

        Otherwise the first finishing song pays the compile (seconds) inside
        a decode step. One ODE step over a few frames covers every shape:
        the NAR kernels are compiled for dynamic token counts.
        """
        if not torch.cuda.is_available() or self.nar_fused is None:
            return
        t0 = time.perf_counter()
        training, default_dtype = self.training, torch.get_default_dtype()
        # load_weights runs before the loader switches to eval, and under the
        # loader's bf16 default dtype, which the compiled kernels guard on:
        # warm up in the serving-time state or the first song recompiles.
        self.eval()
        torch.set_default_dtype(torch.float32)
        try:
            latents = synthesize(self, [EOD, ABC_START, ABC_END, MUSIC_START], list(range(16)), DEFAULT_SEED, steps=1)
            self._decode_latents(latents)
        finally:
            torch.set_default_dtype(default_dtype)
            self.train(training)
        logger.info("YuE2 synthesis warm-up took %.1f s", time.perf_counter() - t0)

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

    def _decode_latents_iter(self, latents: torch.Tensor) -> Iterator[None]:
        """Device latents -> device stereo [2, samples] as a work generator
        (one unit per VAE tile); nothing is read back to the host."""
        model = self._vae_model(latents.device)
        audio = yield from model.decode_tiled_iter(
            latents.T.unsqueeze(0),
            core_frames=VAE_CORE_FRAMES,
            halo_frames=VAE_HALO_FRAMES,
            output_device=latents.device,
            check_finite=False,
        )
        # _SynthesisJob checks finite values before clipping and transferring.
        return audio[0].contiguous()

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
        logger.debug("YUE2_AUDIO_PRECLAMP absmax=%.4f", float(audio.abs().max()))
        # Channels-first [2, T], the same layout MiniMax Music 3 ships in
        # model_outputs; serving's create_audio transposes it for soundfile.
        return audio[0].float().clamp(-1, 1).contiguous()

    # ------------------------------------------------------------ weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ar_pairs, remapped = partition_checkpoint_weights(weights)
        missing, unexpected = self.load_state_dict(dict(remapped), strict=False)
        missing = [k for k in missing if not k.startswith(("model.", "lm_head"))]
        if missing or unexpected:
            raise RuntimeError(f"YuE2 NAR/projection weight mismatch: missing={missing}, unexpected={unexpected}")

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(iter(ar_pairs))
        self.nar_fused = fuse_nar_projections(self.nar_layers)
        # Load the VAE now, not lazily at the first finishing request: a bad
        # path or a failed download must surface at startup, and the decoder
        # weights must sit on the GPU before vLLM's memory profiling sizes
        # the KV cache.
        device = f"cuda:{torch.accelerator.current_device_index()}" if torch.cuda.is_available() else "cpu"
        self._vae_model(torch.device(device))
        self._warm_up_synthesis()
        self._warm_up_sampler()
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
        sampling_extra_args: list[dict[str, Any]] | None = None,
        discard_mask: Sequence[bool] | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bind rows to requests and create new request state for the step.

        This runs before the backbone on every step, including CUDA graph
        replays, so request admission never depends on forward() running.
        Decode rows feed tokens the model itself sampled, so they never
        contribute to the prompt prefix."""
        computed = [int(v) for v in num_computed_tokens]
        scheduled = [int(v) for v in num_scheduled_tokens]
        self._step_rows = [(str(rid), computed[i], scheduled[i]) for i, rid in enumerate(req_ids)]
        # The runner's own verdict on which rows' samples it keeps: a partial
        # prefill, including a recompute after preemption (prompt + output
        # tokens replayed), must not draw from the request's generator.
        self._step_discard = None if discard_mask is None else [bool(v) for v in discard_mask]
        # make_omni_output runs every step; without this reset the first step's
        # payload would stay live forever and the _audio_queue fallback in
        # _ship_audio could never trigger.
        self._last_mm = None
        self._capture_constants(sampling_extra_args, input_ids)
        return input_ids, positions

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del intermediate_tensors, kwargs
        hidden = self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden

    def _capture_constants(self, extra_args: list[dict[str, Any]] | None, input_ids: torch.Tensor | None) -> None:
        """Create per-request state on the request's FIRST scheduled step.

        With KV prefix caching the first step may arrive with ``comp > 0``
        (a shared head was cached by an earlier request), so ``comp == 0``
        cannot be the trigger. The scheduled ``input_ids`` slice then holds
        only the uncached tail, so the driver ships the full prompt ids in
        the request's extra args (KEY_PREFIX_IDS); they are the source of
        truth for the NAR conditioning prefix and for the prompt length the
        sampler uses to tell a completing prefill row from a mid-chunk one.
        """
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
                max_hold_steps=int(args.get(KEY_MAX_HOLD_STEPS, 0) or 0),
            )
            device = input_ids.device
            generator = torch.Generator(device=device if device.type != "mps" else "cpu")
            generator.manual_seed(constants.seed)
            self._states[req_id] = _RequestState(
                request_id=req_id,
                constants=constants,
                prompt_len=prompt_len,
                prefix_ids=prefix_ids,
                generator=generator,
            )
            offset += span

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        if not self._semantic_only_step():
            return self.logits_processor(self.lm_head, hidden_states)
        # Every row samples the semantic phase, whose sampler reads only the
        # MUSIC_END + codec columns (18% of the vocabulary): project onto
        # those alone and leave the rest of the row at -inf.
        rows = hidden_states.shape[0]
        if self._logits is None or self._logits.shape[0] < rows or self._logits.device != hidden_states.device:
            self._logits = torch.full(
                (max(rows, 8), self.vocab_size), -torch.inf, dtype=hidden_states.dtype, device=hidden_states.device
            )
        logits = self._logits[:rows]
        lo, hi = MUSIC_END, CODEC_OFFSET + CODEC_SIZE
        torch.matmul(hidden_states, self.lm_head.weight[lo:hi].T, out=logits[:, lo:hi])
        return logits

    def _semantic_only_step(self) -> bool:
        """True when every request in the step is a semantic-phase request."""
        if not self._step_rows or not hasattr(self.lm_head, "weight"):
            return False
        for req_id, _, _ in self._step_rows:
            state = self._states.get(req_id)
            if state is None or state.constants.phase != "semantic":
                return False
        return True

    # ------------------------------------------------------------ sampler

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> SamplerOutput | None:
        """Draw one token per decoding row with the request's own arithmetic.

        The draw stays on the device: its ids reach the host through an
        async copy that the NEXT call resolves, so the host never waits for
        the GPU inside a decode step and async scheduling can overlap the
        next step's preparation with this step's kernels. Every draw still
        sees the full history (the previous draw is resolved first); only
        acting on an end token or a full frame budget moves one step later.
        """
        del sampling_metadata
        rows = int(logits.shape[0])
        # The runner's input_ids buffer is int32; long ids crash its scatter
        # when concurrent requests re-index rows.
        # Build fixed HOLD/end outputs on the host and transfer the whole batch
        # once. Assigning a Python scalar to a CUDA row performs a blocking H2D
        # copy. The pinned allocator tracks this copy's lifetime, so the buffer
        # can be retired without waiting for it or rewriting an in-flight copy.
        host_token_ids = torch.zeros((rows, 1), dtype=torch.int32, pin_memory=logits.is_cuda)
        self._resolve_pending()
        self._synthesis.pump()
        if not self._step_rows:
            return SamplerOutput(
                sampled_token_ids=host_token_ids.to(logits.device, non_blocking=True), logprobs_tensors=None
            )

        # Rows sharing a phase and preset are sampled as one batch.
        groups: dict[tuple, list[tuple[int, _RequestState]]] = {}
        for row, (req_id, comp, span) in enumerate(self._step_rows):
            if row >= rows:
                break
            state = self._states.get(req_id)
            if state is None:
                continue
            if self._step_discard is not None and row < len(self._step_discard):
                discarded = self._step_discard[row]
            else:
                # A completing prefill row reaches the full prompt length even
                # when a prefix-cache hit left comp > 0.
                discarded = comp + span < state.prompt_len
            if discarded:
                continue
            accepted = max(0, comp + span - state.prompt_len)
            self._reconcile_history(state, accepted)
            c = state.constants
            end = ABC_END if c.phase == "abc" else MUSIC_END
            if state.finished:
                # Normally an async lookahead row the engine discards. If the
                # scheduler dropped the step that emitted the end token (the
                # request was preempted with it in flight), that step's audio
                # was dropped too: deliver both again.
                if state.end_len is not None and accepted < state.end_len:
                    if state.delivered is not None:
                        audio, truncated, error = state.delivered
                        self._ship_audio(state.request_id, audio, truncated, error=error)
                    state.end_len = accepted + 1
                    host_token_ids[row, 0] = end
                continue
            if state.finish_ready:
                state.finish_ready = False
                if self._hold_allowed(state):
                    self._queue_synthesis(state)
                    host_token_ids[row, 0] = HOLD_TOKEN
                    continue
                state.finished = True
                state.end_len = accepted + 1
                self._finish_request_safely(state, hit_end=not state.truncated)
                host_token_ids[row, 0] = end
                continue
            if state.job is not None:
                if state.job.done() or not self._hold_allowed(state):
                    self._complete_synthesis(state)
                    state.end_len = accepted + 1
                    host_token_ids[row, 0] = end
                else:
                    state.hold_steps += 1
                    host_token_ids[row, 0] = HOLD_TOKEN
                continue
            if state.request_id not in self._decode_t0:
                self._decode_t0[state.request_id] = time.perf_counter()
            # With no penalty the window has no effect. Canonicalize it so
            # equivalent groups cannot replay and overwrite the same graph's
            # output buffers before their draws are collected below.
            window = 0 if c.repetition_penalty == 1.0 else c.penalty_window
            key = (c.phase, c.temperature, c.top_p, c.top_k, c.repetition_penalty, window)
            groups.setdefault(key, []).append((row, state))
        token_ids = host_token_ids.to(logits.device, non_blocking=True)
        if not groups:
            return SamplerOutput(sampled_token_ids=token_ids, logprobs_tensors=None)

        # Stage every small host input of this step in one pinned buffer and
        # move it with a single async copy: a pageable copy or a list index
        # would make the host wait for the GPU. The previous step's copies
        # have landed (_resolve_pending waited on its event), so the staging
        # buffer can be rewritten.
        plans = []
        flat: list[int] = []
        for (phase, temperature, top_p, top_k, penalty, window), members in groups.items():
            vocab = PHASE_VOCAB[phase]
            # With no penalty there is no history input: [-0:] would walk the
            # entire history every step before discarding it at width=0.
            windows = (
                [] if penalty == 1.0 else [[vocab.col(t) for t in state.history[-window:]] for _, state in members]
            )
            # A fixed width (the preset window) keeps the sampler graph static.
            width = 0 if penalty == 1.0 else window if window > 0 else max(map(len, windows))
            block = [len(state.history) < state.constants.min_tokens for _, state in members]
            # A request that still needs its song must not stop on its end
            # token when drawn: the host only sees the draw next step.
            end = ABC_END if phase == "abc" else MUSIC_END
            held = [-1 if state.constants.skip_synthesis else end for _, state in members]
            base = len(flat)
            flat += [row for row, _ in members] + block + held
            for cols in windows if width else ():
                flat += cols + [vocab.num_cols] * (width - len(cols))
            plans.append((vocab, (temperature, top_p, top_k, penalty), members, base, width, any(block)))
        if self._stage is None or self._stage.numel() < len(flat):
            self._stage = torch.empty(max(len(flat), 1024), dtype=torch.long, pin_memory=True)
        self._stage[: len(flat)].copy_(torch.as_tensor(flat))
        staged = self._stage[: len(flat)].to(logits.device, non_blocking=True)

        # Graphs share a pool and may replay in a different order from capture.
        # Preserve each group's outputs outside that pool before replaying the
        # next graph: its intermediates may alias the previous graph's outputs.
        count = sum(len(members) for members in groups.values())
        drawn = torch.empty((2, count), dtype=torch.long, device=logits.device)
        emitted: list[torch.Tensor] = []
        drawn_rows: list[int] = []
        states: list[_RequestState] = []
        for vocab, (temperature, top_p, top_k, penalty), members, base, width, blocks in plans:
            n = len(members)
            index = staged[base : base + n]
            rows_logits = logits if n == rows else logits.index_select(0, index)
            window_cols = staged[base + 3 * n : base + 3 * n + n * width].view(n, width) if width else None
            block_end = staged[base + n : base + 2 * n].bool() if blocks else None
            generators = [state.generator for _, state in members]
            graph = self._sampler_graph(vocab, n, rows_logits, (temperature, top_p, top_k, penalty), width)
            if graph is not None:
                ids, bad = graph(rows_logits, window_cols, block_end, generators)
            else:
                ids, bad = sample_rows(
                    rows_logits,
                    vocab,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=penalty,
                    window_cols=window_cols,
                    block_end=block_end,
                    generators=generators,
                )
            offset = len(drawn_rows)
            drawn[0, offset : offset + n].copy_(ids)
            drawn[1, offset : offset + n].copy_(bad)
            emitted.append(torch.where(ids == staged[base + 2 * n : base + 3 * n], HOLD_TOKEN, ids))
            drawn_rows += [row for row, _ in members]
            states += [state for _, state in members]
        out = torch.cat(emitted).to(torch.int32)
        if drawn_rows == list(range(rows)):
            token_ids[:, 0] = out
        else:
            index = torch.cat([staged[base : base + len(m)] for _, _, m, base, _, _ in plans])
            token_ids.view(-1).index_copy_(0, index, out)

        if self._pinned is None or self._pinned.shape[1] < count:
            self._pinned = torch.empty((2, max(count, 64)), dtype=torch.long, pin_memory=True)
        host = self._pinned[:, :count]
        # Async into pinned memory; the next sample() call resolves it.
        host.copy_(drawn, non_blocking=True)
        event = torch.Event()
        event.record()
        self._pending = (event, host, states)
        return SamplerOutput(sampled_token_ids=token_ids, logprobs_tensors=None)

    def _sampler_graph(self, vocab, rows, logits, preset, width) -> SamplerGraph | None:
        """Use a power-of-two bucket; bound captures for custom sampling presets.

        The serving presets are captured before vLLM profiles memory. Other
        presets run eagerly once the small custom-preset allowance is full.
        """
        if logits.device.type != "cuda":
            return None
        bucket = 1 << (rows - 1).bit_length()
        key = (vocab.phase, bucket, logits.dtype, preset, width)
        graph = self._sampler_graphs.get(key)
        if graph is None:
            if len(self._sampler_graphs) >= 2 * self._sampler_capacity.bit_length() + 8:
                return None
            if self._sampler_graph_pool is None:
                self._sampler_graph_pool = torch.cuda.graph_pool_handle()
            graph = SamplerGraph(
                vocab, bucket, logits.shape[1], logits.dtype, preset, width, logits.device, self._sampler_graph_pool
            )
            self._sampler_graphs[key] = graph
        return graph

    def _warm_up_sampler(self) -> None:
        """Capture default phase/preset buckets before serving and KV profiling.

        Lazy captures use torch.cuda.graph, which synchronizes the device.
        Admission and batch-size changes should replay pre-existing graphs.
        """
        weight = self.lm_head.weight
        if weight.device.type != "cuda":
            return
        bucket = 1
        while bucket < self._sampler_capacity * 2:
            for phase, preset in (("semantic", SEMANTIC_SAMPLING), ("abc", ABC_SAMPLING)):
                shape = (preset["temperature"], preset["top_p"], preset["top_k"], preset["repetition_penalty"])
                logits = torch.empty((bucket, self.vocab_size), dtype=weight.dtype, device=weight.device)
                self._sampler_graph(PHASE_VOCAB[phase], bucket, logits, shape, preset["penalty_window"])
            if bucket >= self._sampler_capacity:
                break
            bucket *= 2

    # ------------------------------------------------------------ async synthesis

    def _hold_allowed(self, state: _RequestState) -> bool:
        """May this request emit one more HOLD_TOKEN step?

        Needs the caller's hold budget (and its max_tokens headroom) and room
        in the context for the HOLD and the final end token.
        """
        c = state.constants
        if c.skip_synthesis or state.hold_steps >= c.max_hold_steps or not torch.cuda.is_available():
            return False
        # prompt + frames + the drawn-end HOLD + holds so far + this HOLD + end
        return state.prompt_len + len(state.history) + state.hold_steps + 3 <= self._max_model_len

    def _song_iter(
        self, prefix_ids: list[int], codec: list[int], seed: int, keep: list
    ) -> Generator[None, None, torch.Tensor]:
        """NAR + VAE for one song as work units; returns the device audio."""
        latents = yield from synthesize_iter(self, prefix_ids, codec, seed, keep, self._synthesis.graph_pool)
        keep.append(latents)
        return (yield from self._decode_latents_iter(latents))

    def _queue_synthesis(self, state: _RequestState) -> None:
        codec = [t - CODEC_OFFSET for t in state.history]
        prefix_ids, seed = state.prefix_ids, state.constants.seed
        job = _SynthesisJob(state.request_id, lambda keep: self._song_iter(prefix_ids, codec, seed, keep))
        if not codec or min(codec) < 0 or max(codec) >= CODEC_SIZE:
            job.error = RuntimeError("semantic history is empty or contains non-codec tokens")
        state.job = job
        state.hold_steps = 1
        self._synthesis.submit(job)

    def _complete_synthesis(self, state: _RequestState) -> None:
        """Ship the song on this step (waiting for it if the hold ran out)."""
        job = state.job
        assert job is not None
        audio, failed, waited = self._synthesis.complete(job)
        state.job = None
        state.finished = True
        t0 = self._decode_t0.pop(state.request_id, None)
        t_ar = (job.t_start - t0) if t0 is not None and job.t_start is not None else float("nan")
        logger.debug(
            "YUE2_TIMING req=%s phase=%s ar_tokens=%d ar_s=%.2f ar_tps=%.1f synth_s=%.2f hold_steps=%d waited=%s",
            state.request_id,
            state.constants.phase,
            len(state.history),
            t_ar,
            len(state.history) / t_ar if t_ar else float("nan"),
            (job.t_done - job.t_start) if job.t_start is not None and job.t_done is not None else 0.0,
            state.hold_steps,
            waited,
        )
        self._ship_audio(state.request_id, audio, state.truncated, error=failed)

    def _reconcile_history(self, state: _RequestState, accepted: int) -> None:
        """Drop in-flight draws the scheduler discarded on preemption.

        Only a non-discarded row has replayed the complete accepted sequence;
        partial recompute chunks cannot establish this boundary. A drawn EOS
        occupies one engine token even though it is not in the codec history.
        """
        if accepted >= len(state.history) + int(state.end_drawn):
            return
        if state.job is not None:
            self._synthesis.cancel(state.job)
            state.job = None
        del state.history[accepted:]
        state.end_drawn = False
        state.finished = False
        state.end_len = None
        state.delivered = None
        state.hold_steps = 0
        state.truncated = state.constants.phase == "semantic" and len(state.history) >= state.constants.max_audio_frames
        state.finish_ready = state.truncated

    def _resolve_pending(self) -> None:
        """Apply the previous step's draws to the host-side request state."""
        if self._pending is None:
            return
        event, host, states = self._pending
        self._pending = None
        event.synchronize()
        ids, bad = host.tolist()
        if any(bad):
            raise RuntimeError("sampling scores are all -inf; check phase masking")
        for state, token in zip(states, ids):
            constants = state.constants
            if token == (ABC_END if constants.phase == "abc" else MUSIC_END):
                state.end_drawn = True
                state.truncated = False
                if constants.skip_synthesis:
                    # Emitted as drawn: the engine has already stopped it.
                    state.finished = True
                else:
                    state.finish_ready = True
                continue
            state.history.append(token)
            if constants.phase == "semantic" and len(state.history) >= constants.max_audio_frames:
                state.truncated = True
                state.finish_ready = True

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
        except (RuntimeError, ValueError, FloatingPointError):
            # CPU synthesis also reports non-finite latents as FloatingPointError.
            # Programming errors propagate on both CUDA completion paths.
            logger.exception("YuE2 synthesis failed for req=%s; failing only this request", state.request_id)
            self._ship_audio(state.request_id, torch.zeros((2, 0)), state.truncated, error=True)

    def _finish_request(self, state: _RequestState, *, hit_end: bool) -> None:
        """Solve the ODE and decode the song; abc-phase requests skip this."""
        constants = state.constants
        if constants.skip_synthesis or not state.history:
            return
        if torch.cuda.is_available():
            # No hold headroom means wait now, but still use the serialized
            # synthesis stream and its bounded in-flight work. Independent sync
            # graphs would overlap a running job and accumulate private pools.
            self._queue_synthesis(state)
            state.hold_steps = 0
            self._complete_synthesis(state)
            return
        codec = [t - CODEC_OFFSET for t in state.history]
        if min(codec) < 0 or max(codec) >= CODEC_SIZE:
            raise RuntimeError("semantic history contains non-codec tokens")
        t0 = self._decode_t0.pop(state.request_id, None)
        t_nar0 = time.perf_counter()
        audio, t_nar, t_vae = self.synthesize_song(state.prefix_ids, codec, constants.seed)
        t_ar = (t_nar0 - t0) if t0 is not None else float("nan")
        logger.debug(
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
        The request keeps a reference until the engine finishes it, in case
        the scheduler drops this step (see ``_RequestState.end_len``).
        """
        state = self._states.get(req_id)
        if state is not None:
            state.delivered = (audio, truncated, error)
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
        # Runs at the start of execute_model, after the previous step's
        # sample(), so the state can be freed right away.
        finished = {_request_key(rid) for rid in finished_req_ids}
        for req_id in finished:
            state = self._states.pop(req_id, None)
            if state is not None and state.job is not None:
                # Aborted mid-synthesis.
                self._synthesis.cancel(state.job)
                state.job = None
            if state is not None and not state.finished:
                logger.warning(
                    "YuE2 request %s ended without an end token or its frame "
                    "budget (aborted or preempted); no audio was produced.",
                    req_id,
                )
            self._decode_t0.pop(req_id, None)
        self._step_rows = [row for row in self._step_rows if row[0] not in finished]
