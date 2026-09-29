# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in packed Flow for full-response and streaming inference on Hopper GPUs.

Adapted from SGLang-Omni packed_dit.py, commit
127f34b57446a3cb5588ec987da0771a3be675be (Apache-2.0):
https://github.com/sgl-project/sglang-omni
Full-context and chunk-causal attention use vLLM's FA3 binding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch._dynamo as dynamo
import torch.nn.functional as F
from vllm.vllm_flash_attn import flash_attn_varlen_func

from .packed_conv import pack_conv_weight, packed_causal_conv_mish

FA3_PAGE_SIZE = 1


class RaggedRowAttention:
    """FA3 page metadata isolating each request and each causal query chunk."""

    def __init__(
        self,
        rows: PackedRows,
        *,
        heads: int,
        head_dim: int,
        chunk_size: int | None = None,
        width: int | None = None,
    ) -> None:
        self.heads, self.head_dim = heads, head_dim
        device = rows.row_ids.device
        segment_rows, segment_ends, offsets = [], [], [0]
        for row, length in enumerate(rows.lengths):
            if length == 0 and width is not None:
                # Bucketed layouts keep one (empty) segment per row slot so
                # metadata shapes are identical across batches in a bucket.
                segment_rows.append(row)
                segment_ends.append(0)
                offsets.append(offsets[-1])
                continue
            span = chunk_size or length
            for start in range(0, length, span):
                end = min(start + span, length)
                segment_rows.append(row)
                segment_ends.append(end)
                offsets.append(offsets[-1] + end - start)
        self.cache_seqlens = torch.tensor(segment_ends, dtype=torch.int32, device=device)
        self.cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
        self.max_seqlen_q = max(b - a for a, b in zip(offsets, offsets[1:]))
        starts = rows.starts_host[segment_rows].to(device)
        page = torch.arange(width or max(segment_ends), dtype=torch.int32, device=device)
        self.page_table = torch.where(page[None] < self.cache_seqlens[:, None], starts[:, None] + page[None], 0)


class PackedDiT:
    """Adapt a loaded DiT to compiled, variable-length inference with FA3."""

    def __init__(self, dit: torch.nn.Module) -> None:
        self.dit = dit
        self.compiled_full_forward = None
        self.graph_buckets = False
        self.graph_runner: FlowGraphRunner | None = None
        conv_pos = dit.input_embed.conv_pos_embed
        group = conv_pos.conv1[0].in_channels // conv_pos.conv1[0].groups
        # The packed kernel's tensor-core tiles need a power-of-two group of at
        # least 16 channels in half precision; other shapes keep cuDNN.
        self.conv_weights = (
            (pack_conv_weight(conv_pos.conv1[0]), pack_conv_weight(conv_pos.conv2[0]))
            if group >= 16
            and group & (group - 1) == 0
            and conv_pos.conv1[0].weight.dtype in (torch.float16, torch.bfloat16)
            else None
        )
        self._modulations: dict = {}

    @torch.inference_mode()
    def step_modulations(self, time_span: torch.Tensor, key: tuple) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Per-step adaLN modulations, identical for every request and batch.

        They depend only on the Euler time, which follows a fixed schedule.
        Replay the solver's own time recurrence and compute each step's
        modulation exactly as the forward would (same M=1 shapes and autocast),
        once per schedule, instead of 23 GEMVs and their activations per call.
        """
        cached = self._modulations.get(key)
        if cached is not None:
            return cached
        dit = self.dit
        blocks, finals = [], []
        flow_time = torch.zeros(1, device=time_span.device, dtype=time_span.dtype)
        t, dt = time_span[0], time_span[1] - time_span[0]
        with torch.autocast("cuda", dtype=time_span.dtype):
            for step in range(1, len(time_span)):
                flow_time[:] = t
                emb = dit.time_embed(flow_time)
                blocks.append(
                    torch.cat([block.attn_norm.linear(block.attn_norm.silu(emb)) for block in dit.transformer_blocks])
                )
                finals.append(dit.norm_out.linear(dit.norm_out.silu(emb)))
                t = t + dt
                if step < len(time_span) - 1:
                    dt = time_span[step + 1] - t
        cached = self._modulations[key] = (blocks, finals)
        return cached

    def compile(self, dtype: torch.dtype) -> None:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Packed Flow requires FP16 or BF16")
        import os

        options = {"emulate_precision_casts": True}
        if os.environ.get("COSYVOICE3_FLOW_CUDAGRAPH") == "1":
            # Inductor CUDA graphs replay one recording per concrete shape;
            # full-context batches are padded to recurring buckets.
            options["triton.cudagraphs"] = True
            self.graph_buckets = True
        self.compiled_full_forward = torch.compile(
            self.forward_full,
            backend="inductor",
            dynamic=True,
            fullgraph=True,
            options=options,
        )
        if (
            os.environ.get("COSYVOICE3_FLOW_GRAPH", "0") == "1"
            and not self.graph_buckets
            and self.conv_weights is not None
        ):
            self.graph_runner = FlowGraphRunner(self)

    def row_attention(self, rows: PackedRows, *, streaming: bool, width: int | None = None) -> RaggedRowAttention:
        attn = self.dit.transformer_blocks[0].attn
        result = RaggedRowAttention(
            rows,
            heads=attn.heads,
            head_dim=attn.inner_dim // attn.heads,
            chunk_size=self.dit.static_chunk_size if streaming else None,
            width=width,
        )
        mark_packed_compile_metadata(rows, result)
        return result

    def forward_full(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        rows: PackedRows,
        attention: RaggedRowAttention,
        block_modulation: torch.Tensor | None = None,
        final_modulation: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return forward_packed_tensor_geometry(
            self,
            x,
            mu,
            spks,
            cond,
            t,
            rows,
            attention,
            max_seqlen_q=attention.page_table.shape[1],
            block_modulation=block_modulation,
            final_modulation=final_modulation,
            rope=rope,
        )


@torch.library.custom_op(
    "vllm_omni_cosyvoice3_perf::packed_fa3",
    mutates_args=(),
    device_types="cuda",
)
def packed_fa3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
) -> torch.Tensor:
    """Alias-free FA3 boundary for the compiled PackedDiT path."""
    return flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=page_table.shape[1],
        seqused_k=cache_seqlens,
        block_table=page_table,
        causal=False,
        fa_version=3,
    )


@packed_fa3.register_fake
def fake_packed_fa3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op(
    "vllm_omni_cosyvoice3_perf::native_mish",
    mutates_args=(),
    device_types="cuda",
)
def native_mish(x: torch.Tensor) -> torch.Tensor:
    """Preserve eager CUDA Mish arithmetic across the Inductor boundary."""
    return F.mish(x)


@native_mish.register_fake
def fake_native_mish(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "vllm_omni_cosyvoice3_perf::packed_conv_mish",
    mutates_args=(),
    device_types="cuda",
)
def packed_conv_mish(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Grouped causal conv + Mish on packed rows (see packed_conv.py)."""
    return packed_causal_conv_mish(x, weight, bias, positions)


@packed_conv_mish.register_fake
def fake_packed_conv_mish(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "vllm_omni_cosyvoice3_perf::native_layer_norm",
    mutates_args=(),
    device_types="cuda",
)
def native_layer_norm(
    x: torch.Tensor,
    normalized_size: int,
    eps: float,
) -> torch.Tensor:
    """Keep CUDA autocast's eager FP32 LayerNorm contract."""
    return F.layer_norm(x.float(), (normalized_size,), None, None, eps)


@native_layer_norm.register_fake
def fake_native_layer_norm(
    x: torch.Tensor,
    normalized_size: int,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x, dtype=torch.float32)


@dataclass(frozen=True)
class PackedRows:
    lengths: tuple[int, ...]
    starts_host: torch.Tensor
    row_ids: torch.Tensor
    positions: torch.Tensor

    @property
    def total(self) -> int:
        return sum(self.lengths)

    @property
    def width(self) -> int:
        return max(self.lengths)


def pack_rows(lengths: Sequence[int], device: torch.device) -> PackedRows:
    lengths = tuple(int(length) for length in lengths)
    starts_host = F.pad(torch.tensor(lengths, dtype=torch.int64).cumsum(0), (1, 0))
    starts = starts_host.to(device)
    total = int(starts_host[-1])
    row_ids = torch.repeat_interleave(
        torch.arange(len(lengths), device=device),
        torch.tensor(lengths, dtype=torch.int64, device=device),
        output_size=total,
    )
    positions = torch.arange(total, device=device) - starts[row_ids]
    return PackedRows(
        lengths=lengths,
        starts_host=starts_host.to(torch.int32),
        row_ids=row_ids,
        positions=positions,
    )


def gather_rows(padded: torch.Tensor, rows: PackedRows) -> torch.Tensor:
    """(rows, width, channels) -> (1, total, channels), each row's first
    length frames in row order."""
    width = padded.shape[1]
    flat = padded.reshape(padded.shape[0] * width, padded.shape[2])
    return flat[rows.row_ids * width + rows.positions].unsqueeze(0)


def scatter_rows(packed: torch.Tensor, rows: PackedRows, width: int) -> torch.Tensor:
    """(1, total, channels) -> (rows, width, channels), zero past each row's
    length."""
    channels = packed.shape[2]
    flat = packed.new_zeros(len(rows.lengths) * width, channels)
    flat[rows.row_ids * width + rows.positions] = packed[0]
    return flat.view(len(rows.lengths), width, channels)


def mark_packed_compile_metadata(rows: PackedRows, attention: RaggedRowAttention) -> None:
    dynamo.mark_dynamic(attention.page_table, (0, 1))
    dynamo.mark_dynamic(attention.cu_seqlens_q, 0)
    dynamo.mark_dynamic(attention.cache_seqlens, 0)
    dynamo.mark_dynamic(rows.starts_host, 0)
    dynamo.mark_dynamic(rows.row_ids, 0)
    dynamo.mark_dynamic(rows.positions, 0)


def gather_rows_tensor_geometry(padded: torch.Tensor, rows: PackedRows) -> torch.Tensor:
    width = padded.shape[1]
    flat = padded.reshape(padded.shape[0] * width, padded.shape[2])
    return flat[rows.row_ids * width + rows.positions].unsqueeze(0)


def scatter_rows_tensor_geometry(packed: torch.Tensor, rows: PackedRows, row_count: int, width: int) -> torch.Tensor:
    channels = packed.shape[2]
    flat = packed.new_zeros(row_count * width, channels)
    flat[rows.row_ids * width + rows.positions] = packed[0]
    return flat.view(row_count, width, channels)


def bucketed_width(width):
    """Round a frame count up so convolution shapes recur across batches.

    cuDNN and cuFFT build a plan for every new shape; live batches have unique
    lengths. Right padding is exact for the causal convolutions it feeds.
    """
    step = 32 if width <= 512 else (64 if width <= 2048 else 128)
    return (width + step - 1) // step * step


def conv_pos_embed_tensor_geometry(
    estimator: PackedDiT,
    h: torch.Tensor,
    rows: PackedRows,
    attention: RaggedRowAttention,
) -> torch.Tensor:
    module = estimator.dit.input_embed.conv_pos_embed
    if estimator.conv_weights is not None:
        # Both causal convolutions run on the packed sequence: taps before a
        # row start are masked (the left zero padding), so no padded layout,
        # transposes or per-group cuDNN launches are needed.
        first, second = estimator.conv_weights
        embedded = packed_conv_mish(h[0], first, module.conv1[0].bias, rows.positions)
        embedded = packed_conv_mish(embedded, second, module.conv2[0].bias, rows.positions)
        return embedded.unsqueeze(0)
    row_count = rows.starts_host.shape[0] - 1
    # Branch-free rounding keeps this symbolic under dynamic compilation; a
    # size comparison here would add guards and recompile mid-serving.
    width = (attention.page_table.shape[1] + 63) // 64 * 64
    padded = scatter_rows_tensor_geometry(h, rows, row_count, width)
    embedded = padded.permute(0, 2, 1)
    embedded = F.pad(embedded, (module.kernel_size - 1, 0, 0, 0))
    embedded = module.conv1[0](embedded)
    embedded = native_mish(embedded)
    embedded = F.pad(embedded, (module.kernel_size - 1, 0, 0, 0))
    embedded = module.conv2[0](embedded)
    embedded = native_mish(embedded)
    embedded = embedded.permute(0, 2, 1)
    return gather_rows_tensor_geometry(embedded, rows)


def rope_tensor_geometry(
    estimator: PackedDiT,
    rows: PackedRows,
    attention: RaggedRowAttention,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = attention.page_table.shape[1]
    freqs, scale = estimator.dit.rotary_embed.forward_from_seq_len(width)
    assert not isinstance(scale, torch.Tensor), "the DiT's RoPE has no xpos scale"
    freqs = freqs[:, rows.positions]
    return freqs.cos(), freqs.sin()


def ragged_attention_tensor_geometry(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention: RaggedRowAttention,
    max_seqlen_q: int,
) -> torch.Tensor:
    page_shape = (-1, FA3_PAGE_SIZE, attention.heads, attention.head_dim)
    output = packed_fa3(
        query[0].reshape(-1, attention.heads, attention.head_dim),
        key[0].reshape(page_shape),
        value[0].reshape(page_shape),
        attention.cache_seqlens,
        attention.page_table,
        attention.cu_seqlens_q,
        max_seqlen_q,
    )
    return output.reshape(1, -1, attention.heads * attention.head_dim)


def attend_tensor_geometry(
    attn: torch.nn.Module,
    x: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
    attention: RaggedRowAttention,
    max_seqlen_q: int,
) -> torch.Tensor:
    # note (ratish): under autocast to_q, to_k and to_v would each cast the
    # float32 norm output again.
    x = x.to(attn.to_q.weight.dtype)
    query = attn.to_q(x)
    key = attn.to_k(x)
    value = attn.to_v(x)
    rotate_in_place(query, *rope)
    rotate_in_place(key, *rope)
    output = ragged_attention_tensor_geometry(query, key, value, attention, max_seqlen_q).to(query.dtype)
    return attn.to_out[1](attn.to_out[0](output))


_FUSED_LAYER_NORM = __import__("os").environ.get("COSYVOICE3_FUSED_LAYERNORM", "1") != "0"


def layer_norm_preserving_eager(layer_norm: torch.nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    normalized_size = int(layer_norm.normalized_shape[0])
    if _FUSED_LAYER_NORM:
        # Same FP32 LayerNorm contract as autocast, but visible to Inductor so
        # the upcast, normalization, modulation and downcast fuse into one
        # kernel instead of a LayerNorm, two copies and elementwise passes.
        return F.layer_norm(x.float(), (normalized_size,), None, None, float(layer_norm.eps))
    return native_layer_norm(x, normalized_size, float(layer_norm.eps))


def attn_norm_preserving_eager(
    block: torch.nn.Module,
    h: torch.Tensor,
    time_embedding: torch.Tensor | None,
    modulation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if modulation is None:
        modulation = block.attn_norm.linear(block.attn_norm.silu(time_embedding))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(modulation, 6, dim=1)
    normalized = layer_norm_preserving_eager(block.attn_norm.norm, h)
    normalized = normalized * (1 + scale_msa[:, None]) + shift_msa[:, None]
    return normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp


def final_norm_preserving_eager(
    norm_out: torch.nn.Module,
    h: torch.Tensor,
    time_embedding: torch.Tensor | None,
    modulation: torch.Tensor | None = None,
) -> torch.Tensor:
    if modulation is None:
        modulation = norm_out.linear(norm_out.silu(time_embedding))
    scale, shift = torch.chunk(modulation, 2, dim=1)
    normalized = layer_norm_preserving_eager(norm_out.norm, h)
    return normalized * (1 + scale)[:, None, :] + shift[:, None, :]


def forward_packed_tensor_geometry(
    estimator: PackedDiT,
    x: torch.Tensor,
    mu: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    t: torch.Tensor,
    rows: PackedRows,
    attention: RaggedRowAttention,
    *,
    max_seqlen_q: int,
    block_modulation: torch.Tensor | None = None,
    final_modulation: torch.Tensor | None = None,
    rope: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    dit = estimator.dit
    t = dit.time_embed(t) if block_modulation is None else None
    h = dit.input_embed.proj(torch.cat((x, cond, mu, spks), dim=-1))
    h = conv_pos_embed_tensor_geometry(estimator, h, rows, attention) + h
    if rope is None:
        rope = rope_tensor_geometry(estimator, rows, attention)
    residual = h
    for index, block in enumerate(dit.transformer_blocks):
        modulation = None if block_modulation is None else block_modulation[index : index + 1]
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = attn_norm_preserving_eager(block, h, t, modulation)
        h = h + gate_msa.unsqueeze(1) * attend_tensor_geometry(block.attn, norm, rope, attention, max_seqlen_q)
        ff_norm = layer_norm_preserving_eager(block.ff_norm, h)
        ff_norm = ff_norm * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        h = h + gate_mlp.unsqueeze(1) * block.ff(ff_norm)
    if dit.long_skip_connection is not None:
        h = dit.long_skip_connection(torch.cat((h, residual), dim=-1))
    h = final_norm_preserving_eager(dit.norm_out, h, t, final_modulation)
    return dit.proj_out(h)


def rotate_in_place(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """x: (1, total, heads * head_dim). Interleaved RoPE in float32 on the
    rotary dims, rounded back into x."""
    # note (ratish): the DiT rotates only the first rotary dims of the
    # flattened heads, so the rest of x is never copied.
    rotary = x[..., : cos.shape[-1]]
    half = torch.stack((-rotary[..., 1::2], rotary[..., ::2]), dim=-1).flatten(-2)
    rotary.copy_(rotary * cos + half * sin)


def solve_flow_euler_packed(
    estimator: PackedDiT,
    noise: torch.Tensor,
    time_span: torch.Tensor,
    mu: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    rows: PackedRows,
    *,
    cfg_rate: float,
    streaming: bool,
    modulation_key: tuple | None = None,
) -> torch.Tensor:
    """Euler steps over a packed sequence with classifier free guidance: the
    conditional rows and their unconditional twins share one DiT call."""
    if estimator.graph_runner is not None and modulation_key is not None:
        solved = estimator.graph_runner.run(
            noise, time_span, mu, spks, cond, rows, cfg_rate=cfg_rate, streaming=streaming, key=modulation_key
        )
        if solved is not None:
            return solved
    real_total = noise.shape[1]
    width = None
    if estimator.graph_buckets and not streaming:
        rows, noise, mu, spks, cond, width = pad_to_graph_bucket(rows, noise, mu, spks, cond)
    twin_rows = pack_rows(rows.lengths * 2, noise.device)
    attention = estimator.row_attention(twin_rows, streaming=streaming, width=width)
    mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=1)
    cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=1)
    spks_cfg = torch.cat((spks, torch.zeros_like(spks)), dim=0)
    spks_cfg = spks_cfg[twin_rows.row_ids].unsqueeze(0)
    flow_time = torch.zeros(1, device=noise.device, dtype=spks.dtype)
    forward = estimator.compiled_full_forward
    if forward is None:
        raise ValueError("Packed Flow requires a compiled packed path")
    modulations = None if modulation_key is None else estimator.step_modulations(time_span, modulation_key)
    x = euler_steps(
        forward, noise, time_span, mu_cfg, spks_cfg, cond_cfg, flow_time, twin_rows, attention, modulations, cfg_rate
    )
    return x[:, :real_total].float()


def euler_steps(
    forward,
    noise: torch.Tensor,
    time_span: torch.Tensor,
    mu_cfg: torch.Tensor,
    spks_cfg: torch.Tensor,
    cond_cfg: torch.Tensor,
    flow_time: torch.Tensor,
    twin_rows: PackedRows,
    attention: RaggedRowAttention,
    modulations: tuple[list[torch.Tensor], list[torch.Tensor]] | None,
    cfg_rate: float,
) -> torch.Tensor:
    total = noise.shape[1]
    x = noise
    t, dt = time_span[0], time_span[1] - time_span[0]
    for step in range(1, len(time_span)):
        flow_time[:] = t
        vector_field = forward(
            torch.cat((x, x), dim=1),
            mu_cfg,
            spks_cfg,
            cond_cfg,
            flow_time,
            twin_rows,
            attention,
            None if modulations is None else modulations[0][step - 1],
            None if modulations is None else modulations[1][step - 1],
        )
        conditional = vector_field[:, :total]
        unconditional = vector_field[:, total:]
        x = x + dt * ((1.0 + cfg_rate) * conditional - cfg_rate * unconditional)
        t = t + dt
        if step < len(time_span) - 1:
            dt = time_span[step + 1] - t
    return x


_GRAPH_ROW_BUCKETS = (1, 2, 4, 8, 16)
_GRAPH_TOTAL_STEP = 512
_GRAPH_WIDTH_STEP = 256


def pad_to_graph_bucket(rows, noise, mu, spks, cond):
    """Pad a full-context batch so CUDA-graph shapes recur across batches.

    Rows are padded to a bucketed count with empty rows, and the packed length
    to a bucketed total with one trailing padding row; attention isolates every
    row, so padding never reaches real frames, which stay first in the packing.
    """
    n = len(rows.lengths)
    slots = next((b for b in _GRAPH_ROW_BUCKETS if b >= n), n)
    total = rows.total
    pad = -total % _GRAPH_TOTAL_STEP
    width = max(max(rows.lengths), pad, 1)
    width = -(-width // _GRAPH_WIDTH_STEP) * _GRAPH_WIDTH_STEP
    lengths = tuple(rows.lengths) + (0,) * (slots - n) + (pad,)
    padded_rows = pack_rows(lengths, noise.device)
    extend = (0, 0, 0, pad)
    spks = torch.cat((spks, spks.new_zeros((slots - n + 1, spks.shape[1]))))
    return (
        padded_rows,
        F.pad(noise, extend),
        F.pad(mu, extend),
        spks,
        F.pad(cond, extend),
        width,
    )


_FLOW_GRAPH_SLOTS = (1, 2, 4, 8)
_FLOW_GRAPH_MAX_TOTAL = int(__import__("os").environ.get("COSYVOICE3_FLOW_GRAPH_MAX_TOTAL", "16384"))
_FLOW_GRAPH_SEGMENT_STEP = 16


def flow_graph_total(total: int) -> int:
    """Packed-length bucket: at most ~12% padding, few recurring layouts."""
    step = 128 if total <= 1024 else (256 if total <= 4096 else 512)
    return -(-total // step) * step


@dataclass
class _FlowGraph:
    graph: torch.cuda.CUDAGraph
    noise: torch.Tensor
    mu: torch.Tensor
    cond: torch.Tensor
    spks: torch.Tensor
    time_span: torch.Tensor
    table: torch.Tensor
    row_ids: torch.Tensor
    positions: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor
    output: torch.Tensor | None = None


class FlowGraphRunner:
    """Replay the whole CFG Euler solve as one CUDA graph per padded layout.

    Small codec batches are launch bound: every step walks Dynamo guards, the
    Inductor wrapper and one FA3 wrapper per block, several hundred host
    microseconds per step before the GPU sees work. The layout (row slots,
    packed length, attention segments) is padded to recurring buckets exactly
    like the bucketed layouts above: padding rows are isolated by attention
    and by the positional masks of the packed convolution, so real frames only
    see their own row. Metadata lives in static buffers that are rewritten
    before each replay.
    """

    def __init__(self, estimator: PackedDiT) -> None:
        self.estimator = estimator
        self.graphs: dict[tuple, _FlowGraph] = {}
        self.pool = None
        attn = estimator.dit.transformer_blocks[0].attn
        self.heads, self.head_dim = attn.heads, attn.inner_dim // attn.heads

    def _layout(self, lengths: tuple[int, ...], streaming: bool):
        n, total = len(lengths), sum(lengths)
        slots = next((b for b in _FLOW_GRAPH_SLOTS if b >= n), None)
        padded_total = flow_graph_total(total)
        if slots is None or padded_total > _FLOW_GRAPH_MAX_TOTAL:
            return None
        twin = (tuple(lengths) + (0,) * (slots - n) + (padded_total - total,)) * 2
        chunk = self.estimator.dit.static_chunk_size if streaming else None
        seg_rows, seg_ends, offsets = [], [], [0]
        for row, length in enumerate(twin):
            if length == 0:
                seg_rows.append(row)
                seg_ends.append(0)
                offsets.append(offsets[-1])
                continue
            span = chunk or length
            for start in range(0, length, span):
                end = min(start + span, length)
                seg_rows.append(row)
                seg_ends.append(end)
                offsets.append(offsets[-1] + end - start)
        segments = len(seg_ends)
        if streaming:
            segments = -(-segments // _FLOW_GRAPH_SEGMENT_STEP) * _FLOW_GRAPH_SEGMENT_STEP
        extra = segments - len(seg_ends)
        seg_rows += [0] * extra
        seg_ends += [0] * extra
        offsets += [offsets[-1]] * extra
        return (streaming, slots, padded_total, segments), twin, seg_rows, seg_ends, offsets

    def _allocate(self, key, noise, spks, time_span) -> _FlowGraph:
        _, slots, padded_total, segments = key[:4]
        device, dtype = noise.device, noise.dtype
        channels = noise.shape[-1]
        rows = 2 * (slots + 1)
        return _FlowGraph(
            graph=torch.cuda.CUDAGraph(),
            noise=torch.zeros((1, padded_total, channels), device=device, dtype=dtype),
            mu=torch.zeros((1, padded_total, channels), device=device, dtype=dtype),
            cond=torch.zeros((1, padded_total, channels), device=device, dtype=dtype),
            spks=torch.zeros((slots + 1, spks.shape[-1]), device=device, dtype=spks.dtype),
            time_span=torch.zeros_like(time_span),
            table=torch.zeros(rows + 3 * segments + 1, device=device, dtype=torch.int32),
            row_ids=torch.zeros(2 * padded_total, device=device, dtype=torch.int64),
            positions=torch.zeros(2 * padded_total, device=device, dtype=torch.int64),
            cache_seqlens=torch.zeros(segments, device=device, dtype=torch.int32),
            cu_seqlens_q=torch.zeros(segments + 1, device=device, dtype=torch.int32),
            page_table=torch.zeros((segments, padded_total), device=device, dtype=torch.int32),
        )

    @staticmethod
    def _load(state: _FlowGraph, twin, seg_rows, seg_ends, offsets, noise, mu, cond, spks, time_span) -> None:
        total = noise.shape[1]
        host = torch.tensor(list(twin) + seg_rows + seg_ends + offsets, dtype=torch.int32, pin_memory=True)
        state.table.copy_(host, non_blocking=True)
        rows, segments = len(twin), len(seg_ends)
        lengths = state.table[:rows].long()
        starts = F.pad(lengths.cumsum(0), (1, 0))[:-1]
        width = state.row_ids.shape[0]
        row_ids = torch.repeat_interleave(torch.arange(rows, device=noise.device), lengths, output_size=width)
        state.row_ids.copy_(row_ids)
        state.positions.copy_(torch.arange(width, device=noise.device) - starts[row_ids])
        seg_rows_t = state.table[rows : rows + segments]
        ends = state.table[rows + segments : rows + 2 * segments]
        state.cache_seqlens.copy_(ends)
        state.cu_seqlens_q.copy_(state.table[rows + 2 * segments :])
        page = torch.arange(state.page_table.shape[1], device=noise.device, dtype=torch.int32)
        state.page_table.copy_(
            torch.where(page[None] < ends[:, None], starts[seg_rows_t.long()].int()[:, None] + page[None], 0)
        )
        state.noise[:, :total].copy_(noise)
        state.mu[:, :total].copy_(mu)
        state.cond[:, :total].copy_(cond)
        state.spks.zero_()
        state.spks[: spks.shape[0]].copy_(spks)
        state.time_span.copy_(time_span)

    def _solve(self, state: _FlowGraph, cfg_rate: float, key: tuple) -> torch.Tensor:
        estimator = self.estimator
        rows = PackedRows(lengths=(), starts_host=state.table, row_ids=state.row_ids, positions=state.positions)
        attention = object.__new__(RaggedRowAttention)
        attention.heads, attention.head_dim = self.heads, self.head_dim
        attention.cache_seqlens, attention.cu_seqlens_q = state.cache_seqlens, state.cu_seqlens_q
        attention.page_table = state.page_table
        attention.max_seqlen_q = state.page_table.shape[1]
        mark_packed_compile_metadata(rows, attention)
        mu_cfg = torch.cat((state.mu, torch.zeros_like(state.mu)), dim=1)
        cond_cfg = torch.cat((state.cond, torch.zeros_like(state.cond)), dim=1)
        spks_cfg = torch.cat((state.spks, torch.zeros_like(state.spks)), dim=0)[state.row_ids].unsqueeze(0)
        flow_time = torch.zeros(1, device=state.noise.device, dtype=state.spks.dtype)
        modulations = estimator.step_modulations(state.time_span, key[4])
        return euler_steps(
            estimator.compiled_full_forward,
            state.noise,
            state.time_span,
            mu_cfg,
            spks_cfg,
            cond_cfg,
            flow_time,
            rows,
            attention,
            modulations,
            cfg_rate,
        )

    def run(
        self,
        noise: torch.Tensor,
        time_span: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        rows: PackedRows,
        *,
        cfg_rate: float,
        streaming: bool,
        key: tuple,
    ) -> torch.Tensor | None:
        layout = self._layout(rows.lengths, streaming)
        if layout is None or torch.cuda.is_current_stream_capturing():
            return None
        graph_key, twin, seg_rows, seg_ends, offsets = layout
        graph_key = graph_key + (key, float(cfg_rate))
        state = self.graphs.get(graph_key)
        fresh = state is None
        if fresh:
            state = self._allocate(graph_key, noise, spks, time_span)
        self._load(state, twin, seg_rows, seg_ends, offsets, noise, mu, cond, spks, time_span)
        if fresh:
            # Warm up (and compile new dynamic shapes) off the capture, then
            # record the solve into the shared pool.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._solve(state, cfg_rate, graph_key)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(state.graph, pool=self.pool):
                state.output = self._solve(state, cfg_rate, graph_key)
            if self.pool is None:
                self.pool = state.graph.pool()
            self.graphs[graph_key] = state
        state.graph.replay()
        return state.output[:, : noise.shape[1]].float()
