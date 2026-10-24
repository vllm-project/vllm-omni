# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed incremental Flow with shared eager and CUDA Graph execution."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import torch
import torch.nn.functional as F

from .packed_dit import (
    _FLOW_GRAPH_MAX_TOTAL,
    _FLOW_GRAPH_SLOTS,
    PackedDiT,
    attn_norm_preserving_eager,
    final_norm_preserving_eager,
    flow_graph_total,
    layer_norm_preserving_eager,
    packed_conv_mish,
    ragged_attention_tensor_geometry,
    rotate_in_place,
)


@dataclass
class FlowStepCache:
    conv1_cache: torch.Tensor
    conv2_cache: torch.Tensor
    kv: list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class IncrementalFlowCache:
    frames: int
    schedule: tuple
    steps: list[FlowStepCache]
    mel: torch.Tensor


def _segment_metadata(lengths, prefixes, slots, qcap, chunk, segments):
    """Build fixed-capacity CFG segment metadata; unused Q segments are empty."""
    padded_lengths = lengths + [0] * (slots - len(lengths) - 1) + [qcap - sum(lengths)]
    segment_kv_slots, ends, offsets = [], [], [0]
    for row, length in enumerate(padded_lengths):
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            segment_kv_slots.append(row)
            ends.append(max(1, (prefixes[row] if row < len(prefixes) else 0) + end))
            offsets.append(offsets[-1] + end - start)
    extra = segments - len(segment_kv_slots)
    segment_kv_slots += [slots - 1] * extra
    ends += [1] * extra
    offsets += [qcap] * extra

    return (
        segment_kv_slots
        + [row + slots for row in segment_kv_slots]
        + ends * 2
        + offsets
        + [value + qcap for value in offsets[1:]]
    )


@dataclass
class _FlowState:
    segment_metadata: torch.Tensor
    prefixes: torch.Tensor
    lengths: torch.Tensor
    noise: torch.Tensor
    mu: torch.Tensor
    cond: torch.Tensor
    speakers: torch.Tensor
    time_span: torch.Tensor
    frequencies: torch.Tensor
    kv: list[list[tuple[torch.Tensor, torch.Tensor]]]
    conv1_cache: list[torch.Tensor]
    conv2_cache: list[torch.Tensor]
    graph: torch.cuda.CUDAGraph | None = None
    output: torch.Tensor | None = None

    @property
    def segment_kv_slots(self):
        count = (self.segment_metadata.numel() - 1) // 3
        return self.segment_metadata[:count]

    @property
    def cache_seqlens(self):
        count = (self.segment_metadata.numel() - 1) // 3
        return self.segment_metadata[count : 2 * count]

    @property
    def cu_seqlens_q(self):
        count = (self.segment_metadata.numel() - 1) // 3
        return self.segment_metadata[2 * count :]


class IncrementalFlowRunner:
    """One per-slot Q/KV layout and solver for eager and graph execution.

    Q rows are tightly packed; graphs round only their total capacity.
    KV histories retain independent per-slot capacity.
    Distributions vary through the same metadata and cache operations.

    Keep at most two bucket recordings: static per-layer/per-step KV working
    buffers are large, unlike the original stateless solve's input buffers.
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.graphs = OrderedDict()
        attention = estimator.dit.transformer_blocks[0].attn
        self.heads, self.head_dim = attention.heads, attention.dim_head
        self.page_size = 16 if getattr(estimator, "fa_version", 3) == 2 else 1

    def _layout(self, frames, cached, *, graph):
        slots = next((value for value in _FLOW_GRAPH_SLOTS if value >= len(frames)), None) if graph else len(frames)
        total = sum(n - a for n, a in zip(frames, cached))
        qcap = flow_graph_total(total) if graph else total
        kv_cap = max(frames)
        # One extra slot owns only the total-capacity padding.
        kv_cap = max(kv_cap, qcap - total)
        if graph:
            kv_cap = flow_graph_total(kv_cap)
            if slots is None or (slots + 1) * kv_cap > _FLOW_GRAPH_MAX_TOTAL:
                return None
        else:
            kv_cap = -(-kv_cap // self.page_size) * self.page_size
        return slots, qcap, kv_cap

    def _allocate(self, layout, noise, speakers, time_span):
        slots, qcap, kv_cap = layout
        slots += 1  # Isolate total-capacity padding from real requests.
        chunk = self.estimator.dit.static_chunk_size
        # Per CFG branch, sum(lengths) == qcap, including the padding slot.
        # Each slot needs floor(length / chunk) full segments and at most
        # one partial segment. Thus sum(ceil(length / chunk)) is bounded by
        # floor(qcap / chunk) + slots, regardless of the length distribution.
        # Reserve that bound so a graph can replay different distributions;
        # unused entries have zero-length Q ranges.
        segments = qcap // chunk + slots
        device = noise.device
        channels = noise.shape[-1]
        dim = self.estimator.dit.dim
        margin = self.estimator.dit.input_embed.conv_pos_embed.kernel_size - 1
        steps = len(time_span) - 1
        kv = [
            [
                (
                    noise.new_zeros(2 * slots, kv_cap, self.heads, self.head_dim),
                    noise.new_zeros(2 * slots, kv_cap, self.heads, self.head_dim),
                )
                for _ in self.estimator.dit.transformer_blocks
            ]
            for _ in range(steps)
        ]
        frequencies, _ = self.estimator.dit.rotary_embed.forward_from_seq_len(kv_cap)
        return _FlowState(
            segment_metadata=torch.zeros(6 * segments + 1, device=device, dtype=torch.int32),
            prefixes=torch.zeros(slots, device=device, dtype=torch.long),
            lengths=torch.zeros(slots, device=device, dtype=torch.long),
            noise=noise.new_zeros(1, qcap, channels),
            mu=noise.new_zeros(1, qcap, channels),
            cond=noise.new_zeros(1, qcap, channels),
            speakers=speakers.new_zeros(slots, speakers.shape[-1]),
            time_span=torch.zeros_like(time_span),
            frequencies=frequencies,
            kv=kv,
            conv1_cache=[noise.new_zeros(2 * slots, margin, dim) for _ in range(steps)],
            conv2_cache=[noise.new_zeros(2 * slots, margin, dim) for _ in range(steps)],
        )

    def _load(self, state, noise, time_span, mu, speakers, cond, caches):
        slots = state.prefixes.shape[0]
        prefixes = [0 if cache is None else cache.frames for cache in caches]
        lengths = [value.shape[1] - prefix for value, prefix in zip(noise, prefixes)]
        metadata = torch.tensor(
            [prefixes + [0] * (slots - len(prefixes)), lengths + [0] * (slots - len(lengths))],
            dtype=torch.long,
            pin_memory=state.noise.device.type == "cuda",
        ).to(state.noise.device, non_blocking=True)
        segment_metadata = _segment_metadata(
            lengths,
            prefixes,
            slots,
            state.noise.shape[1],
            self.estimator.dit.static_chunk_size,
            state.segment_kv_slots.numel() // 2,
        )
        host = torch.tensor(segment_metadata, dtype=torch.int32, pin_memory=state.noise.device.type == "cuda")
        state.segment_metadata.copy_(host, non_blocking=True)
        state.prefixes.copy_(metadata[0])
        state.lengths.copy_(metadata[1])
        for value in (state.noise, state.mu, state.cond, state.speakers):
            value.zero_()
        state.time_span.copy_(time_span)
        offset = 0
        for row, (prefix, length, cache) in enumerate(zip(prefixes, lengths, caches)):
            state.noise[0, offset : offset + length].copy_(noise[row][0, prefix:])
            state.mu[0, offset : offset + length].copy_(mu[row][0, prefix:])
            state.cond[0, offset : offset + length].copy_(cond[row][0, prefix:])
            offset += length
            state.speakers[row].copy_(speakers[row][0])
            for step in range(len(state.kv)):
                for branch in range(2):
                    slot = row + branch * slots
                    if cache is None:
                        state.conv1_cache[step][slot].zero_()
                        state.conv2_cache[step][slot].zero_()
                    else:
                        old = cache.steps[step]
                        state.conv1_cache[step][slot].copy_(old.conv1_cache[branch])
                        state.conv2_cache[step][slot].copy_(old.conv2_cache[branch])
                        for layer, (key, value) in enumerate(state.kv[step]):
                            key[slot, :prefix].copy_(old.kv[layer][0][branch])
                            value[slot, :prefix].copy_(old.kv[layer][1][branch])

    def _solve(self, state, cfg_rate):
        dit = self.estimator.dit
        qcap = state.noise.shape[1]
        slots = state.prefixes.shape[0]
        kv_cap = state.kv[0][0][0].shape[1]
        chunk = dit.static_chunk_size
        device = state.noise.device
        # Reuse the packed layout's length/cumsum mapping, without building
        # or copying token-sized Python lists before every graph replay.
        lengths = torch.cat((state.lengths[:-1], (qcap - state.lengths.sum()).reshape(1)))
        rows = torch.repeat_interleave(torch.arange(slots, device=device), lengths, output_size=qcap)
        starts = F.pad(lengths.cumsum(0), (1, 0))[:-1]
        local = (torch.arange(qcap, device=device) - starts[rows]).repeat(2)
        row_ids = torch.cat((rows, rows + slots))
        prefix = state.prefixes.repeat(2)
        length = state.lengths.repeat(2)
        positions = prefix[row_ids] + local
        complete = ((prefix + length) // chunk * chunk - prefix).clamp_min(0)
        page = torch.arange(kv_cap // self.page_size, device=device, dtype=torch.int32)
        attention = SimpleNamespace(
            heads=self.heads,
            head_dim=self.head_dim,
            page_table=state.segment_kv_slots[:, None] * page.numel() + page[None],
            cache_seqlens=state.cache_seqlens,
            cu_seqlens_q=state.cu_seqlens_q,
            max_seqlen_q=chunk,
            fa_version=3 if self.page_size == 1 else 2,
        )
        frequencies = state.frequencies[:, positions]
        rope = frequencies.cos(), frequencies.sin()
        mu = torch.cat((state.mu, torch.zeros_like(state.mu)), dim=1)
        cond = torch.cat((state.cond, torch.zeros_like(state.cond)), dim=1)
        speaker = torch.cat((state.speakers, torch.zeros_like(state.speakers)))[row_ids].unsqueeze(0)
        module = dit.input_embed.conv_pos_embed
        margin = module.kernel_size - 1
        conv_width = min(qcap, kv_cap)
        conv_positions = torch.arange(margin + conv_width, device=device).repeat(2 * slots)
        history_indices = (complete[:, None] + torch.arange(margin, device=device)[None])[:, :, None]

        def convolution(h, step):
            value = h.new_zeros(2 * slots * conv_width, h.shape[-1])
            value.index_copy_(0, row_ids * conv_width + local, h[0])
            value = value.reshape(2 * slots, conv_width, -1)
            for layer, history, packed_weight in (
                (
                    module.conv1[0],
                    state.conv1_cache[step],
                    None if self.estimator.conv_weights is None else self.estimator.conv_weights[0],
                ),
                (
                    module.conv2[0],
                    state.conv2_cache[step],
                    None if self.estimator.conv_weights is None else self.estimator.conv_weights[1],
                ),
            ):
                joined = torch.cat((history, value), dim=1)
                if packed_weight is None:
                    value = F.mish(layer(joined.transpose(1, 2))).transpose(1, 2)
                else:
                    value = packed_conv_mish(
                        joined.reshape(-1, joined.shape[-1]), packed_weight, layer.bias, conv_positions
                    ).reshape_as(joined)[:, margin:]
                history.copy_(joined.gather(1, history_indices.expand(-1, -1, joined.shape[-1])))
            return value[row_ids, local].unsqueeze(0)

        def update_kv(k, v, step, layer):
            key, val = state.kv[step][layer]
            index = row_ids * kv_cap + positions
            key.view(-1, self.heads, self.head_dim).index_copy_(0, index, k.reshape(-1, self.heads, self.head_dim))
            val.view(-1, self.heads, self.head_dim).index_copy_(0, index, v.reshape(-1, self.heads, self.head_dim))
            return key.reshape(1, -1, k.shape[-1]), val.reshape(1, -1, v.shape[-1])

        x = state.noise
        time_span = state.time_span
        valid = (rows != slots - 1).reshape(1, qcap, 1)
        total = x.shape[1]
        t, dt = time_span[0], time_span[1] - time_span[0]
        for step in range(len(time_span) - 1):
            timestep = dit.time_embed(t.reshape(1))
            h = dit.input_embed.proj(torch.cat((torch.cat((x, x), dim=1), cond, mu, speaker), dim=-1))
            h = h + convolution(h, step)
            residual = h
            for layer, block in enumerate(dit.transformer_blocks):
                norm, gate, shift, scale, gate_mlp = attn_norm_preserving_eager(block, h, timestep)
                attn = block.attn
                norm = norm.to(attn.to_q.weight.dtype)
                q, k, v = attn.to_q(norm), attn.to_k(norm), attn.to_v(norm)
                rotate_in_place(q, *rope)
                rotate_in_place(k, *rope)
                k, v = update_kv(k, v, step, layer)
                attended = ragged_attention_tensor_geometry(q, k, v, attention, attention.max_seqlen_q).to(q.dtype)
                h = h + gate[:, None] * attn.to_out(attended)
                normalized = layer_norm_preserving_eager(block.ff_norm, h)
                normalized = normalized * (1 + scale[:, None]) + shift[:, None]
                h = h + gate_mlp[:, None] * block.ff(normalized)
            if dit.long_skip_connection is not None:
                h = dit.long_skip_connection(torch.cat((h, residual), dim=-1))
            vector = dit.proj_out(final_norm_preserving_eager(dit.norm_out, h, timestep))
            x = x + dt * ((1.0 + cfg_rate) * vector[:, :total] - cfg_rate * vector[:, total:])
            x = torch.where(valid, x, 0)
            t = t + dt
            if step + 2 < len(time_span):
                dt = time_span[step + 2] - t
        return x.float()

    def run(self, noise, time_span, mu, speakers, cond, *, cfg_rate, schedule, caches):
        frames = [value.shape[1] for value in noise]
        cached = [0 if cache is None else cache.frames for cache in caches]
        graph = noise[0].is_cuda and os.getenv("COSYVOICE3_FLOW_GRAPH", "0") == "1"
        if graph and torch.cuda.is_current_stream_capturing():
            graph = False
        layout = self._layout(frames, cached, graph=graph)
        if layout is None:
            graph = False
            layout = self._layout(frames, cached, graph=False)
        if not graph:
            state = self._allocate(layout, noise[0], speakers[0], time_span)
            self._load(state, noise, time_span, mu, speakers, cond, caches)
            state.output = self._solve(state, cfg_rate)
            return self._results(state, frames, cached, schedule, caches)
        key = (*layout, schedule, float(cfg_rate))
        state = self.graphs.get(key)
        if state is None:
            if len(self.graphs) >= 2:
                self.graphs.popitem(last=False)
            state = self._allocate(layout, noise[0], speakers[0], time_span)
            self._load(state, noise, time_span, mu, speakers, cond, caches)
            # Warmup mutates only graph-owned KV/conv buffers. Reload the
            # caller's original histories before recording the actual solve.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._solve(state, cfg_rate)
            torch.cuda.current_stream().wait_stream(stream)
            self._load(state, noise, time_span, mu, speakers, cond, caches)
            state.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(state.graph):
                state.output = self._solve(state, cfg_rate)
            self.graphs[key] = state
        else:
            self.graphs.move_to_end(key)
        # Capture itself ran once and changed graph buffers: replay must start
        # from the same input histories, just like every subsequent invocation.
        self._load(state, noise, time_span, mu, speakers, cond, caches)
        state.graph.replay()
        return self._results(state, frames, cached, schedule, caches)

    def _results(self, state, frames, cached, schedule, caches):
        slots = state.prefixes.shape[0]
        offset = 0
        chunk = self.estimator.dit.static_chunk_size
        output, updated = [], []
        for row, (frames, prefix, cache) in enumerate(zip(frames, cached, caches)):
            if frames == prefix and cache is not None:
                output.append(cache.mel)
                updated.append(cache)
                continue
            suffix = state.output[:, offset : offset + frames - prefix].clone()
            offset += frames - prefix
            mel = suffix if cache is None else torch.cat((cache.mel, suffix), dim=1)
            complete = frames // chunk * chunk
            steps = [
                FlowStepCache(
                    torch.stack((state.conv1_cache[step][row], state.conv1_cache[step][row + slots])),
                    torch.stack((state.conv2_cache[step][row], state.conv2_cache[step][row + slots])),
                    [
                        (
                            torch.stack((k[row, :complete], k[row + slots, :complete])),
                            torch.stack((v[row, :complete], v[row + slots, :complete])),
                        )
                        for k, v in state.kv[step]
                    ],
                )
                for step in range(len(state.kv))
            ]
            output.append(mel)
            updated.append(IncrementalFlowCache(complete, schedule, steps, mel[:, :complete].clone()))
        return output, updated


@torch.inference_mode()
def solve_flow_euler_incremental(
    estimator: PackedDiT,
    noise: list[torch.Tensor],
    time_span: torch.Tensor,
    mu: list[torch.Tensor],
    speakers: list[torch.Tensor],
    cond: list[torch.Tensor],
    *,
    cfg_rate: float,
    schedule: tuple,
    caches: list[IncrementalFlowCache | None],
) -> tuple[list[torch.Tensor], list[IncrementalFlowCache]]:
    """Pack every request's new/partial blocks, with conditional then CFG rows.

    Complete histories remain request-owned. Historical Transformer blocks
    are reused; convolution uses fixed left histories, and the current
    incomplete attention block is recomputed.
    Final/full-context solves must continue through the original solver.
    """
    frames_all = [value.shape[1] for value in noise]
    caches = [
        cache if cache is not None and cache.schedule == schedule and cache.frames <= frames else None
        for cache, frames in zip(caches, frames_all)
    ]
    active = [i for i, (cache, frames) in enumerate(zip(caches, frames_all)) if cache is None or cache.frames < frames]
    if not active:
        ready = cast(list[IncrementalFlowCache], caches)
        return [cache.mel for cache in ready], ready
    runner = getattr(estimator, "incremental_runner", None)
    if runner is None:
        runner = estimator.incremental_runner = IncrementalFlowRunner(estimator)
    return runner.run(noise, time_span, mu, speakers, cond, cfg_rate=cfg_rate, schedule=schedule, caches=caches)
