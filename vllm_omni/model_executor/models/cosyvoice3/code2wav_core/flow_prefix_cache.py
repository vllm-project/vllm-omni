# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned paged prefix cache for the packed streaming Flow.

Non-final streaming hops run the DiT with chunk-causal attention on
position-fixed noise, and every mu frame they condition on is final (its
lookahead tokens are known). A frame in a completed attention chunk therefore
sees exactly the same inputs on every later hop, so its whole trajectory
through the Euler solve is final too. Later hops only need the new frames,
provided they can still read the prefix's per-step DiT state:

* the K/V of every transformer layer, which new queries attend over;
* the input of the causal convolution position embedding, whose two stacked
  convolutions look ``2 * (kernel - 1)`` frames back into the prefix.

Both are kept like an LLM decode KV cache: each request owns a list of pages,
a block table maps its frames to pages and a slot mapping writes new frames.
The solve has ``2 * steps`` independent sequences per request (every Euler
step and CFG branch sees different inputs). One page holds ``PAGE_FRAMES``
frames of all of them, so a single page list addresses every sequence: block
``page * sequences + 2 * step + branch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    rotate_in_place,
    to_device_async,
)

PAGE_FRAMES = 16
# Page 0 is never handed out: graph padding rows write and read it.
_SCRATCH_PAGE = 0

# The pools are reached from inside the custom ops, the way vLLM attention ops
# reach the KV cache through the forward context: Inductor never sees the
# pools as graph inputs, so it can neither copy nor reorder their updates.
_ACTIVE_CACHE: FlowPrefixCache | None = None


@dataclass(eq=False)
class FlowPrefixState:
    """Cache pages owned by one streaming request.

    ``committed`` leading frames hold final values for every Euler step and
    CFG branch. Pages past it may hold a partial chunk that the next hop
    recomputes.
    """

    pages: list[int] = field(default_factory=list)
    committed: int = 0


@dataclass
class PrefixQuery:
    """New frames ``[start, end)`` of one request; tensors are ``[end - start, channels]``."""

    state: FlowPrefixState
    start: int
    end: int
    noise: torch.Tensor
    mu: torch.Tensor
    cond: torch.Tensor
    spks: torch.Tensor  # [1, channels]


@dataclass
class _Entry:
    pages: list[int] | None  # None marks the padding entry
    start: int
    end: int
    context: int


@dataclass
class _PlanSizes:
    query: int  # query rows per CFG branch
    context: int  # cached convolution-context rows per CFG branch
    segments: int
    width: int  # block-table entries per segment
    rows: int  # request slots, without the padding entry
    windows: int  # non-empty convolution windows (eager fallback only)
    window_width: int


@dataclass
class PrefixPlan:
    """Device-side layout of one solve, as views into two flat tables.

    Query rows are ordered CFG branch first, then request; slots, page tables
    and the convolution context address step 0 and are shifted per step.
    """

    slots: torch.Tensor
    context_slots: torch.Tensor
    conv_order: torch.Tensor  # cat(context, rows) -> per-sequence windows
    conv_positions: torch.Tensor
    conv_window_ids: torch.Tensor
    conv_take: torch.Tensor  # query rows inside the windows
    row_ids: torch.Tensor  # speaker row of each query row
    positions: torch.Tensor  # absolute frame of each query row (RoPE)
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor
    max_seqlen_q: int
    windows: int
    window_width: int


def _plan_views(table64: torch.Tensor, table32: torch.Tensor, sizes: _PlanSizes, chunk: int) -> PrefixPlan:
    query, context = 2 * sizes.query, 2 * sizes.context
    window = query + context
    parts = table64.split((query, context, window, window, window, query, query, query))
    seqlens, cu, table = table32.split((sizes.segments, sizes.segments + 1, sizes.segments * sizes.width))
    return PrefixPlan(
        *parts,
        cache_seqlens=seqlens,
        cu_seqlens_q=cu,
        page_table=table.view(sizes.segments, sizes.width),
        max_seqlen_q=chunk,
        windows=sizes.windows,
        window_width=sizes.window_width,
    )


@torch.library.custom_op("vllm_omni_cosyvoice3_perf::prefix_cache_conv_input", mutates_args=())
def prefix_cache_conv_input(h: torch.Tensor, slots: torch.Tensor, context_slots: torch.Tensor) -> torch.Tensor:
    """Read the cached convolution context, store this hop's inputs, and
    return both: ``[context; h]``.

    The op declares no mutation, so a compiled graph keeps it only for its
    result. On a request's first hop the context is empty; returning it
    alone would let Inductor drop the call, and this hop's store with it.
    """
    pool = _ACTIVE_CACHE.conv_input
    context = pool[context_slots]
    pool[slots] = h
    return torch.cat((context, h))


@prefix_cache_conv_input.register_fake
def _(h: torch.Tensor, slots: torch.Tensor, context_slots: torch.Tensor) -> torch.Tensor:
    return h.new_empty((context_slots.shape[0] + h.shape[0], h.shape[1]))


@torch.library.custom_op("vllm_omni_cosyvoice3_perf::prefix_attention", mutates_args=())
def prefix_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: int,
    slots: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
) -> torch.Tensor:
    """Store the new K/V rows of one layer, then attend over the paged prefix."""
    key_cache, value_cache = _ACTIVE_CACHE.kv[layer]
    key_cache.view(-1, *k.shape[1:])[slots] = k
    value_cache.view(-1, *v.shape[1:])[slots] = v
    if not q.is_cuda:
        return paged_attention_reference(q, key_cache, value_cache, cache_seqlens, page_table, cu_seqlens_q)
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(
        q,
        key_cache,
        value_cache,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=page_table.shape[1] * key_cache.shape[1],
        seqused_k=cache_seqlens,
        block_table=page_table,
        causal=False,
        fa_version=3,
    )


@prefix_attention.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: int,
    slots: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
) -> torch.Tensor:
    return torch.empty_like(q)


def paged_attention_reference(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> torch.Tensor:
    """Eager equivalent of FA3's paged varlen attention, for CPU runs."""
    page = key_cache.shape[1]
    keys, values = key_cache.flatten(0, 1), value_cache.flatten(0, 1)
    offsets = torch.arange(page, device=q.device)
    bounds = cu_seqlens_q.tolist()
    output = torch.zeros_like(q)
    for segment, length in enumerate(cache_seqlens.tolist()):
        begin, end = bounds[segment], bounds[segment + 1]
        if begin == end:
            continue
        slots = (page_table[segment].long()[:, None] * page + offsets).flatten()[:length]
        attended = F.scaled_dot_product_attention(
            q[begin:end].transpose(0, 1), keys[slots].transpose(0, 1), values[slots].transpose(0, 1)
        )
        output[begin:end] = attended.transpose(0, 1)
    return output


def conv_pos_embed_windows(estimator: PackedDiT, window: torch.Tensor, plan: PrefixPlan) -> torch.Tensor:
    """Causal conv position embedding over [cached context; new rows] windows.

    Positions restart at each window, so taps before it read the convolution's
    zero padding. A window carries the full receptive field (or starts at frame
    0), so the new rows come out exactly as in a full-sequence convolution.
    """
    module = estimator.dit.input_embed.conv_pos_embed
    if estimator.conv_weights is not None:
        first, second = estimator.conv_weights
        embedded = packed_conv_mish(window, first, module.conv1[0].bias, plan.conv_positions)
        return packed_conv_mish(embedded, second, module.conv2[0].bias, plan.conv_positions)
    # FP32 / CPU: scatter windows into a left-aligned padded batch for nn.Conv1d.
    index = plan.conv_window_ids * plan.window_width + plan.conv_positions
    padded = window.new_zeros((plan.windows * plan.window_width, window.shape[1])).index_copy(0, index, window)
    embedded = padded.view(plan.windows, plan.window_width, -1).transpose(1, 2)
    for conv in (module.conv1[0], module.conv2[0]):
        embedded = F.mish(conv(F.pad(embedded, (module.kernel_size - 1, 0))))
    return embedded.transpose(1, 2).reshape(plan.windows * plan.window_width, -1)[index]


def attend_prefix(
    attn: torch.nn.Module,
    x: torch.Tensor,
    layer: int,
    plan: PrefixPlan,
    rope: tuple[torch.Tensor, torch.Tensor],
    slots: torch.Tensor,
    page_table: torch.Tensor,
) -> torch.Tensor:
    x = x.to(attn.to_q.weight.dtype)
    query, key, value = attn.to_q(x), attn.to_k(x), attn.to_v(x)
    rotate_in_place(query, *rope)
    rotate_in_place(key, *rope)
    head_dim = attn.inner_dim // attn.heads
    output = prefix_attention(
        query[0].reshape(-1, attn.heads, head_dim),
        key[0].reshape(-1, attn.heads, head_dim),
        value[0].reshape(-1, attn.heads, head_dim),
        layer,
        slots,
        plan.cache_seqlens,
        page_table,
        plan.cu_seqlens_q,
        plan.max_seqlen_q,
    )
    output = output.reshape(1, -1, attn.inner_dim).to(query.dtype)
    return attn.to_out[1](attn.to_out[0](output))


def forward_prefix(
    estimator: PackedDiT,
    x: torch.Tensor,
    mu: torch.Tensor,
    spks: torch.Tensor,
    cond: torch.Tensor,
    plan: PrefixPlan,
    rope: tuple[torch.Tensor, torch.Tensor],
    slots: torch.Tensor,
    context_slots: torch.Tensor,
    page_table: torch.Tensor,
    block_modulation: torch.Tensor,
    final_modulation: torch.Tensor,
) -> torch.Tensor:
    """One DiT evaluation of the new rows only; mirrors forward_packed_tensor_geometry."""
    dit = estimator.dit
    h = dit.input_embed.proj(torch.cat((x, cond, mu, spks), dim=-1))
    window = prefix_cache_conv_input(h[0], slots, context_slots)[plan.conv_order]
    h = conv_pos_embed_windows(estimator, window, plan)[plan.conv_take].unsqueeze(0) + h
    residual = h
    for index, block in enumerate(dit.transformer_blocks):
        modulation = block_modulation[index : index + 1]
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = attn_norm_preserving_eager(block, h, None, modulation)
        h = h + gate_msa.unsqueeze(1) * attend_prefix(block.attn, norm, index, plan, rope, slots, page_table)
        ff_norm = layer_norm_preserving_eager(block.ff_norm, h)
        ff_norm = ff_norm * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        h = h + gate_mlp.unsqueeze(1) * block.ff(ff_norm)
    if dit.long_skip_connection is not None:
        h = dit.long_skip_connection(torch.cat((h, residual), dim=-1))
    h = final_norm_preserving_eager(dit.norm_out, h, None, final_modulation)
    return dit.proj_out(h)


@dataclass
class _PrefixGraph:
    graph: torch.cuda.CUDAGraph
    table64: torch.Tensor
    table32: torch.Tensor
    noise: torch.Tensor
    mu: torch.Tensor
    cond: torch.Tensor
    spks: torch.Tensor
    time_span: torch.Tensor
    plan: PrefixPlan | None = None
    output: torch.Tensor | None = None


class FlowPrefixCache:
    """Page pool, allocator and incremental solver of the streaming Flow."""

    def __init__(
        self,
        estimator: PackedDiT,
        *,
        steps: int,
        capacity_bytes: int,
        compiled: bool,
        graphs: bool,
    ) -> None:
        global _ACTIVE_CACHE
        dit = estimator.dit
        attn = dit.transformer_blocks[0].attn
        conv = dit.input_embed.conv_pos_embed
        self.estimator = estimator
        self.steps = steps
        self.sequences = 2 * steps
        self.chunk = int(dit.static_chunk_size)
        self.conv_context = 2 * (conv.kernel_size - 1)
        weight = attn.to_k.weight
        layers = len(dit.transformer_blocks)
        page_bytes = self.sequences * PAGE_FRAMES * (2 * layers * attn.inner_dim + dit.dim) * weight.element_size()
        self.num_pages = int(capacity_bytes // page_bytes)
        if self.num_pages < 2:
            raise ValueError(f"Flow prefix cache needs at least {2 * page_bytes} bytes, got {capacity_bytes}")
        blocks = self.num_pages * self.sequences
        head_dim = attn.inner_dim // attn.heads
        self.kv = weight.new_zeros((layers, 2, blocks, PAGE_FRAMES, attn.heads, head_dim))
        self.conv_input = weight.new_zeros((blocks * PAGE_FRAMES, dit.dim))
        self.free_pages = [page for page in range(self.num_pages - 1, -1, -1) if page != _SCRATCH_PAGE]
        self.forward = forward_prefix
        if compiled:
            self.forward = torch.compile(
                forward_prefix,
                backend="inductor",
                dynamic=True,
                fullgraph=True,
                options={"emulate_precision_casts": True},
            )
        self.graphs: dict[tuple, _PrefixGraph] | None = {} if graphs else None
        self.pool = None
        _ACTIVE_CACHE = self

    @property
    def used_pages(self) -> int:
        return self.num_pages - 1 - len(self.free_pages)

    def reserve(self, state: FlowPrefixState, frames: int) -> bool:
        """Grow ``state`` to cover ``frames``; False (and no change) when the pool is short."""
        missing = -(-frames // PAGE_FRAMES) - len(state.pages)
        if missing > len(self.free_pages):
            return False
        for _ in range(missing):
            state.pages.append(self.free_pages.pop())
        return True

    def release(self, state: FlowPrefixState | None) -> None:
        """Return a request's pages; idempotent, so every owner of the state may call it."""
        if state is None:
            return
        self.free_pages.extend(state.pages)
        state.pages = []
        state.committed = 0

    def solve(
        self,
        queries: list[PrefixQuery],
        time_span: torch.Tensor,
        *,
        cfg_rate: float,
        modulation_key: tuple,
    ) -> list[torch.Tensor]:
        """Euler-solve the new frames of each query; returns float ``[end - start, channels]``."""
        global _ACTIVE_CACHE
        _ACTIVE_CACHE = self
        outputs: list[torch.Tensor] = []
        group_size = _FLOW_GRAPH_SLOTS[-1]
        for begin in range(0, len(queries), group_size):
            outputs += self._solve_group(queries[begin : begin + group_size], time_span, cfg_rate, modulation_key)
        return outputs

    def capture(
        self,
        rows: int,
        total: int,
        like: torch.Tensor,
        speaker_dim: int,
        time_span: torch.Tensor,
        *,
        cfg_rate: float,
        modulation_key: tuple,
    ) -> None:
        """Record the solve of ``rows`` row slots and ``total`` query frames (a
        bucket) before serving; only the padding entry runs, on the scratch page."""
        global _ACTIVE_CACHE
        _ACTIVE_CACHE = self
        empty = like.new_zeros((1, 0, like.shape[-1]))
        speakers = like.new_zeros((0, speaker_dim))
        self._replay([], rows, total, empty, empty, empty, speakers, time_span, cfg_rate, modulation_key)

    def _solve_group(
        self, queries: list[PrefixQuery], time_span: torch.Tensor, cfg_rate: float, modulation_key: tuple
    ) -> list[torch.Tensor]:
        entries = [_Entry(q.state.pages, q.start, q.end, min(self.conv_context, q.start)) for q in queries]
        lengths = [q.end - q.start for q in queries]
        query = sum(lengths)
        noise = torch.cat([q.noise for q in queries]).unsqueeze(0)
        mu = torch.cat([q.mu for q in queries]).unsqueeze(0)
        cond = torch.cat([q.cond for q in queries]).unsqueeze(0)
        spks = torch.cat([q.spks for q in queries])
        rows = next((slots for slots in _FLOW_GRAPH_SLOTS if slots >= len(queries)), None)
        total = flow_graph_total(query)
        if (
            self.graphs is not None
            and rows is not None
            and total <= _FLOW_GRAPH_MAX_TOTAL
            and not torch.cuda.is_current_stream_capturing()
        ):
            solved = self._replay(entries, rows, total, noise, mu, cond, spks, time_span, cfg_rate, modulation_key)
        else:
            sizes = self._sizes(
                entries,
                rows=len(entries),
                query=query,
                context=sum(entry.context for entry in entries),
                segments=None,
                width=max(len(entry.pages) for entry in entries),
            )
            table64, table32 = self._build_tables(entries, sizes)
            table64, table32 = (to_device_async(table, mu.device) for table in (table64, table32))
            plan = _plan_views(table64, table32, sizes, self.chunk)
            spks = torch.cat((spks, spks.new_zeros((1, spks.shape[1]))))
            modulations = self.estimator.step_modulations(time_span, modulation_key)
            solved = self._euler(plan, noise, mu, cond, spks, time_span, cfg_rate, modulations)
        return list(solved[0, :query].float().split(lengths))

    def _sizes(
        self, entries: list[_Entry], *, rows: int, query: int, context: int, segments: int | None, width: int
    ) -> _PlanSizes:
        """Plan sizes; ``segments=None`` sizes them exactly instead of to a graph bucket."""
        real_query = sum(entry.end - entry.start for entry in entries)
        real_context = sum(entry.context for entry in entries)
        pad_query, pad_context = query - real_query, context - real_context
        if segments is None:
            spans = [(e.start, e.end) for e in entries] + [(pad_context, pad_context + pad_query)]
            segments = 2 * sum(len(self._segment_ends(start, end)) for start, end in spans)
        spans = [e.context + e.end - e.start for e in entries] + [pad_context + pad_query]
        windows = 2 * sum(1 for span in spans if span > 0)
        return _PlanSizes(query, context, segments, max(width, 1), rows, windows, max(spans))

    def _segment_ends(self, start: int, end: int) -> list[int]:
        """Ends of the attention chunks covering frames [start, end)."""
        if end <= start:
            return []
        return list(range((start // self.chunk + 1) * self.chunk, end, self.chunk)) + [end]

    def _build_tables(self, entries: list[_Entry], sizes: _PlanSizes) -> tuple[torch.Tensor, torch.Tensor]:
        """Host int64 / int32 tables backing a PrefixPlan (see _plan_views for the layout)."""
        real_query = sum(entry.end - entry.start for entry in entries)
        real_context = sum(entry.context for entry in entries)
        pad_query, pad_context = sizes.query - real_query, sizes.context - real_context
        empty = [_Entry([], 0, 0, 0)] * (sizes.rows - len(entries))
        pad = _Entry(None, pad_context, pad_context + pad_query, pad_context)
        entries = entries + empty + [pad]
        count, page, sequences = len(entries), PAGE_FRAMES, self.sequences
        columns: dict[str, list[torch.Tensor]] = {name: [] for name in _COLUMNS}
        seqlens: list[int] = []
        cu = [0]
        table: list[list[int]] = []
        query_offset, context_offset, window_start, window_index = 0, 2 * sizes.context, 0, 0
        for branch in (0, 1):
            context_base = branch * sizes.context
            for index, entry in enumerate(entries):
                frames = torch.arange(entry.start, entry.end)
                prior = torch.arange(entry.start - entry.context, entry.start)
                if entry.pages is None:
                    # Padding rows only ever touch (and attend to) the scratch page.
                    blocks = [_SCRATCH_PAGE * sequences + branch]
                    columns["slots"].append(blocks[0] * page + frames % page)
                    columns["context_slots"].append(blocks[0] * page + prior % page)
                    columns["positions"].append(frames - entry.start)
                else:
                    pages = torch.tensor(entry.pages, dtype=torch.long)
                    blocks = (pages * sequences + branch).tolist()
                    columns["slots"].append((pages[frames // page] * sequences + branch) * page + frames % page)
                    columns["context_slots"].append((pages[prior // page] * sequences + branch) * page + prior % page)
                    columns["positions"].append(frames)
                length = entry.end - entry.start
                window = entry.context + length
                columns["conv_order"].append(
                    torch.cat(
                        (
                            context_base + torch.arange(entry.context),
                            context_offset + query_offset + torch.arange(length),
                        )
                    )
                )
                columns["conv_positions"].append(torch.arange(window))
                columns["conv_window_ids"].append(torch.full((window,), window_index))
                columns["conv_take"].append(window_start + entry.context + torch.arange(length))
                columns["row_ids"].append(torch.full((length,), branch * count + index))
                previous = entry.start
                for end in self._segment_ends(entry.start, entry.end):
                    cu.append(cu[-1] + end - previous)
                    seqlens.append(1 if entry.pages is None else end)
                    table.append(blocks + [0] * (sizes.width - len(blocks)))
                    previous = end
                context_base += entry.context
                query_offset += length
                window_start += window
                window_index += window > 0
        if len(seqlens) > sizes.segments:
            raise ValueError(f"prefix plan needs {len(seqlens)} attention segments, bucket has {sizes.segments}")
        for _ in range(sizes.segments - len(seqlens)):
            seqlens.append(0)
            cu.append(cu[-1])
            table.append([0] * sizes.width)
        table64 = torch.cat([torch.cat(columns[name]).long() for name in _COLUMNS])
        table32 = torch.tensor(seqlens + cu + [block for row in table for block in row], dtype=torch.int32)
        return table64, table32

    def _euler(
        self,
        plan: PrefixPlan,
        noise: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        spks: torch.Tensor,
        time_span: torch.Tensor,
        cfg_rate: float,
        modulations: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> torch.Tensor:
        """CFG Euler loop of solve_flow_euler_packed; step ``s`` uses sequences ``2s`` and ``2s + 1``."""
        freqs, _ = self.estimator.dit.rotary_embed(plan.positions)
        rope = (freqs.cos(), freqs.sin())
        mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=1)
        cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=1)
        spks_cfg = torch.cat((spks, torch.zeros_like(spks)))[plan.row_ids].unsqueeze(0)
        total = noise.shape[1]
        x = noise
        t, dt = time_span[0], time_span[1] - time_span[0]
        for step in range(1, len(time_span)):
            sequence = 2 * (step - 1)
            vector_field = self.forward(
                self.estimator,
                torch.cat((x, x), dim=1),
                mu_cfg,
                spks_cfg,
                cond_cfg,
                plan,
                rope,
                plan.slots + sequence * PAGE_FRAMES,
                plan.context_slots + sequence * PAGE_FRAMES,
                plan.page_table + sequence,
                modulations[0][step - 1],
                modulations[1][step - 1],
            )
            conditional = vector_field[:, :total]
            unconditional = vector_field[:, total:]
            x = x + dt * ((1.0 + cfg_rate) * conditional - cfg_rate * unconditional)
            t = t + dt
            if step < len(time_span) - 1:
                dt = time_span[step + 1] - t
        return x

    def _replay(
        self,
        entries: list[_Entry],
        rows: int,
        total: int,
        noise: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        spks: torch.Tensor,
        time_span: torch.Tensor,
        cfg_rate: float,
        modulation_key: tuple,
    ) -> torch.Tensor:
        """Replay the whole solve as one CUDA graph per (row slots, padded length).

        Every other dimension is fixed per bucket: the context covers each slot's
        full receptive field, segments their worst case, and block tables span
        the whole pool. Padding lives in its own entry on the scratch page.
        """
        sizes = self._sizes(
            entries,
            rows=rows,
            query=total,
            context=rows * self.conv_context,
            segments=2 * (total // self.chunk + 2 * rows + 2),
            width=self.num_pages,
        )
        table64, table32 = self._build_tables(entries, sizes)
        key = (rows, total, modulation_key, float(cfg_rate))
        state = self.graphs.get(key)
        fresh = state is None
        if fresh:
            channels = noise.shape[-1]
            state = _PrefixGraph(
                graph=torch.cuda.CUDAGraph(),
                table64=torch.zeros_like(table64, device=mu.device),
                table32=torch.zeros_like(table32, device=mu.device),
                noise=noise.new_zeros((1, total, channels)),
                mu=mu.new_zeros((1, total, channels)),
                cond=cond.new_zeros((1, total, channels)),
                spks=spks.new_zeros((rows + 1, spks.shape[1])),
                time_span=torch.zeros_like(time_span),
            )
            state.plan = _plan_views(state.table64, state.table32, sizes, self.chunk)
        state.table64.copy_(table64.pin_memory(), non_blocking=True)
        state.table32.copy_(table32.pin_memory(), non_blocking=True)
        query = noise.shape[1]
        for static, value in ((state.noise, noise), (state.mu, mu), (state.cond, cond)):
            static[:, query:].zero_()
            static[:, :query].copy_(value)
        state.spks.zero_()
        state.spks[: spks.shape[0]].copy_(spks)
        state.time_span.copy_(time_span)
        if fresh:
            modulations = self.estimator.step_modulations(state.time_span, modulation_key)
            arguments = (state.plan, state.noise, state.mu, state.cond, state.spks, state.time_span, cfg_rate)
            # Warm up (and compile new dynamic shapes) off the capture.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._euler(*arguments, modulations)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(state.graph, pool=self.pool):
                state.output = self._euler(*arguments, modulations)
            if self.pool is None:
                self.pool = state.graph.pool()
            self.graphs[key] = state
        state.graph.replay()
        return state.output


_COLUMNS = (
    "slots",
    "context_slots",
    "conv_order",
    "conv_positions",
    "conv_window_ids",
    "conv_take",
    "row_ids",
    "positions",
)
