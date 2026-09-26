# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Window-aligned sequence parallelism for SeedVR2 (Plan A).

SeedVR2's DiT attends only inside 3D windows, so a shard that owns *whole*
windows needs no attention communication at all: every rank keeps all
attention heads and runs plain local attention over its own windows.  What does
cost communication is that consecutive layers use different window layouts
(regular / shifted), so the token -> rank ownership changes at layer
boundaries and the video activations have to be re-sharded.

This module implements that path, deliberately scoped to SeedVR2:

* a deterministic, token-balanced window -> rank planner (LPT with a fixed
  tie-break),
* rank-local metadata (window ids, canonical token ids, ``cu_seqlens``),
* bidirectional (regular -> shifted **and** shifted -> regular) Plan A
  redistribution plans built from canonical token membership,
* a variable-split ``all_to_all_single`` runtime over an explicit process
  group, plus a local-permutation fast path when no rank crosses a boundary,
* the global window-mean reduction used to keep the replicated text stream
  identical on every rank.

Design notes that matter for correctness:

* A layout is an exact partition of the post-patch ``(T, H, W)`` token grid;
  see :mod:`vllm_omni.diffusion.models.seedvr2.window_geometry`.
* The number of layout transitions is derived from the actual layer schedule
  (``ensure_layout``), never hard-coded.  For the 3B checkpoint's alternating
  schedule that is 31 inter-layer transitions plus entry/final handling.
* Every collective on the window-SP group is executed by *all* members in the
  same order, including ranks that own no window at all.
* The planner cache key includes the SP world size: the same geometry maps to a
  different assignment on a different number of ranks.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .window_geometry import (
    DEFAULT_WINDOW,
    DEFAULT_WINDOW_METHODS,
    GEOMETRY_VERSION,
    geometry_fingerprint,
)

#: Bumped whenever assignment or routing semantics change.
PLANNER_VERSION = 1

#: int32 ``cu_seqlens`` bounds (torch's varlen kernels take int32 offsets).
INT32_MAX = 2**31 - 1


# =============================================================================
# Layout / assignment / plan data structures (CPU metadata)
# =============================================================================


@dataclass(frozen=True)
class WindowLayoutKey:
    """Identity of one window layout (shape + geometry, not ownership)."""

    token_grid: tuple[int, int, int]
    window_method: str
    window_shape: tuple[int, int, int]
    geometry_version: int
    geometry_fingerprint: str


@dataclass(frozen=True, eq=False)
class WindowLayout:
    """A partition of the video tokens into windows, in window-packed order.

    ``window_token_ids`` holds canonical token ids (``(t * H + h) * W + w``) of
    every packed row; ``window_offsets`` delimits the windows inside it.
    """

    key: WindowLayoutKey
    num_tokens: int
    window_offsets: torch.Tensor  # CPU int64 [W + 1]
    window_token_ids: torch.Tensor  # CPU int64 [N]
    window_shapes: torch.Tensor  # CPU int64 [W, 3]

    @property
    def num_windows(self) -> int:
        return int(self.window_offsets.numel() - 1)

    def window_lengths(self) -> torch.Tensor:
        return (self.window_offsets[1:] - self.window_offsets[:-1]).to(torch.int64)

    def window_token_ids_of(self, window_ids: torch.Tensor) -> torch.Tensor:
        """Canonical ids of the given windows, concatenated in that order."""
        if window_ids.numel() == 0:
            return torch.empty(0, dtype=torch.int64)
        ids = window_ids.tolist()
        pieces = [self.window_token_ids[int(self.window_offsets[w]) : int(self.window_offsets[w + 1])] for w in ids]
        return torch.cat(pieces)


@dataclass(frozen=True, eq=False)
class WindowAssignment:
    """Which rank owns which window (a whole window belongs to one rank)."""

    layout_key: WindowLayoutKey
    world_size: int
    planner_version: int
    owner_rank: torch.Tensor  # CPU int64 [W]
    rank_window_ids: tuple[torch.Tensor, ...]
    rank_token_counts: tuple[int, ...]
    rank_window_counts: tuple[int, ...]
    max_window_tokens: int = 0

    @property
    def max_rank_tokens(self) -> int:
        return max(self.rank_token_counts) if self.rank_token_counts else 0

    @property
    def ideal_lower_bound(self) -> int:
        """Largest window, or perfectly balanced load, whichever is bigger."""
        total = sum(self.rank_token_counts)
        return max(self.max_window_tokens, math.ceil(total / self.world_size))

    def load_imbalance(self) -> float:
        total = sum(self.rank_token_counts)
        if total == 0:
            return 1.0
        return self.max_rank_tokens / (total / self.world_size)


@dataclass(frozen=True, eq=False)
class RankWindowPlan:
    """Rank-local view of a layout: what this rank owns and how it is packed."""

    layout_key: WindowLayoutKey
    rank: int
    window_ids: torch.Tensor  # CPU int64 [W_r]
    global_token_ids: torch.Tensor  # CPU int64 [N_r], canonical ids in local row order
    video_cu_seqlens: torch.Tensor  # CPU int32 [W_r + 1]
    window_shapes: torch.Tensor  # CPU int64 [W_r, 3]

    @property
    def num_local_tokens(self) -> int:
        return int(self.global_token_ids.numel())

    @property
    def num_local_windows(self) -> int:
        return int(self.window_ids.numel())


@dataclass(frozen=True, eq=False)
class WindowRedistributionPlan:
    """Plan A routing contract for one rank of one layout transition.

    ``input_split_sizes[d]`` is the number of rows this rank sends to rank ``d``
    and ``output_split_sizes[s]`` the number of rows it receives from rank
    ``s`` (feature rows, not bytes).  ``send_indices`` selects rows of the
    local source tensor in send order; ``recv_to_dst_indices`` permutes the
    received rows into destination-local order.
    """

    src_key: WindowLayoutKey
    dst_key: WindowLayoutKey
    world_size: int
    rank: int
    input_split_sizes: tuple[int, ...]
    output_split_sizes: tuple[int, ...]
    send_indices: torch.Tensor  # CPU int64 [N_src]
    recv_to_dst_indices: torch.Tensor  # CPU int64 [N_dst]
    network_exchange_required: bool
    planner_version: int = PLANNER_VERSION

    @property
    def num_send_rows(self) -> int:
        return int(self.send_indices.numel())

    @property
    def num_recv_rows(self) -> int:
        return int(self.recv_to_dst_indices.numel())


@dataclass(frozen=True)
class DeviceWindowRedistributionPlan:
    """Device-side materialisation of :class:`WindowRedistributionPlan`."""

    send_indices: torch.Tensor  # int64 [N_src]
    recv_to_dst_indices: torch.Tensor  # int64 [N_dst]
    input_split_sizes: tuple[int, ...]
    output_split_sizes: tuple[int, ...]
    num_recv_rows: int
    network_exchange_required: bool


# =============================================================================
# Layouts and assignment
# =============================================================================


def build_window_layout(
    token_grid: tuple[int, int, int],
    window_method: str,
    *,
    window: tuple[int, int, int] = DEFAULT_WINDOW,
    geometry_version: int = GEOMETRY_VERSION,
) -> WindowLayout:
    """Build the CPU layout of one layer's window partition."""
    from .window_geometry import window_layout_geometry

    window_offsets, window_token_ids, window_shapes = window_layout_geometry(token_grid, window_method, window)
    key = WindowLayoutKey(
        token_grid=(int(token_grid[0]), int(token_grid[1]), int(token_grid[2])),
        window_method=window_method,
        window_shape=(int(window[0]), int(window[1]), int(window[2])),
        geometry_version=int(geometry_version),
        geometry_fingerprint=geometry_fingerprint(token_grid, window_method, window, geometry_version),
    )
    return WindowLayout(
        key=key,
        num_tokens=int(window_token_ids.numel()),
        window_offsets=window_offsets,
        window_token_ids=window_token_ids,
        window_shapes=window_shapes,
    )


def build_window_assignment(
    layout: WindowLayout,
    world_size: int,
    *,
    planner_version: int = PLANNER_VERSION,
) -> WindowAssignment:
    """Deterministic token-balanced window -> rank assignment (LPT).

    Windows are considered in ``(-video_token_count, window_id)`` order and each
    is placed on the rank with the smallest ``(assigned_tokens, rank_id)``.  The
    result is independent of container iteration order.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    lengths = [int(v) for v in layout.window_lengths().tolist()]
    if not lengths:
        raise ValueError("layout has no windows")
    if any(n <= 0 for n in lengths):
        raise ValueError(f"every window must have a positive token count, got {lengths}")

    loads = [0] * world_size
    buckets: list[list[int]] = [[] for _ in range(world_size)]
    for w in sorted(range(len(lengths)), key=lambda w: (-lengths[w], w)):
        r = min(range(world_size), key=lambda r: (loads[r], r))
        buckets[r].append(w)
        loads[r] += lengths[w]

    rank_window_ids = tuple(torch.tensor(sorted(b), dtype=torch.int64) for b in buckets)
    owner_rank = torch.empty(len(lengths), dtype=torch.int64)
    for r, ids in enumerate(rank_window_ids):
        if ids.numel():
            owner_rank[ids] = r

    return WindowAssignment(
        layout_key=layout.key,
        world_size=world_size,
        planner_version=planner_version,
        owner_rank=owner_rank,
        rank_window_ids=rank_window_ids,
        rank_token_counts=tuple(loads),
        rank_window_counts=tuple(int(ids.numel()) for ids in rank_window_ids),
        max_window_tokens=max(lengths),
    )


def build_rank_window_plan(layout: WindowLayout, assignment: WindowAssignment, rank: int) -> RankWindowPlan:
    """Rank-local metadata for one layout (used for windows/attention packing)."""
    _check_same_layout(layout, assignment.layout_key)
    if not 0 <= rank < assignment.world_size:
        raise ValueError(f"rank {rank} out of range for world_size {assignment.world_size}")

    window_ids = assignment.rank_window_ids[rank]
    lengths = layout.window_lengths()
    if window_ids.numel():
        local_lengths = lengths[window_ids]
        local_counts = torch.cumsum(local_lengths, dim=0)
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), local_counts])
        global_token_ids = layout.window_token_ids_of(window_ids)
        window_shapes = layout.window_shapes[window_ids]
    else:
        offsets = torch.zeros(1, dtype=torch.int64)
        global_token_ids = torch.empty(0, dtype=torch.int64)
        window_shapes = torch.empty((0, 3), dtype=torch.int64)

    video_cu_seqlens = to_cu_seqlens(offsets, context=f"rank {rank} video packing")
    return RankWindowPlan(
        layout_key=layout.key,
        rank=rank,
        window_ids=window_ids,
        global_token_ids=global_token_ids,
        video_cu_seqlens=video_cu_seqlens,
        window_shapes=window_shapes,
    )


def to_cu_seqlens(offsets: torch.Tensor, *, context: str = "") -> torch.Tensor:
    """Convert int64 offsets to int32 ``cu_seqlens`` with an explicit overflow check."""
    offsets = offsets.to(torch.int64)
    if offsets.numel() == 0:
        raise ValueError("cu_seqlens must contain at least the leading zero")
    if int(offsets[0]) != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {int(offsets[0])} ({context})")
    if int(offsets[-1]) > INT32_MAX:
        raise ValueError(f"sequence length {int(offsets[-1])} exceeds int32 cu_seqlens range ({context})")
    if bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError(f"cu_seqlens must be non-decreasing ({context})")
    return offsets.to(torch.int32)


def joint_cu_seqlens(video_cu_seqlens: torch.Tensor, text_len: int) -> torch.Tensor:
    """``[W + 1]`` int32 joint (video + replicated text) offsets for local attention.

    ``video_cu_seqlens`` describes video packing only and must not be used
    directly as joint Q/K offsets.  The offsets are accumulated in int64 on the
    input's device and converted to int32 only after validation, so CUDA
    metadata never round-trips through the host and an overflow is caught before
    a wrapped offset can be produced.  ``text_len`` may be 0; an input of ``[0]``
    (a rank without windows) yields ``[0]``.
    """
    if text_len < 0:
        raise ValueError(f"text_len must be >= 0, got {text_len}")
    video = video_cu_seqlens.to(torch.int64)
    if video.numel() == 0:
        raise ValueError("video_cu_seqlens must contain at least the leading zero")
    if bool((video[0] != 0).item()):
        raise ValueError(f"video_cu_seqlens must start at 0, got {int(video[0])}")
    if bool((video[1:] < video[:-1]).any().item()):
        raise ValueError("video_cu_seqlens must be non-decreasing")
    windows = video.numel() - 1
    text_offsets = torch.arange(windows + 1, device=video.device, dtype=torch.int64) * int(text_len)
    return to_cu_seqlens(video + text_offsets, context="joint video+text packing")


# =============================================================================
# Plan A routing
# =============================================================================


def _check_same_layout(layout: WindowLayout, key: WindowLayoutKey) -> None:
    if layout.key != key:
        raise ValueError(
            "layout/assignment mismatch: "
            f"layout={layout.key.token_grid}/{layout.key.window_method}, "
            f"assignment={key.token_grid}/{key.window_method}"
        )


def _rank_global_token_ids(layout: WindowLayout, assignment: WindowAssignment) -> list[torch.Tensor]:
    return [layout.window_token_ids_of(assignment.rank_window_ids[r]) for r in range(assignment.world_size)]


def build_redistribution_plans(
    src_layout: WindowLayout,
    src_assignment: WindowAssignment,
    dst_layout: WindowLayout,
    dst_assignment: WindowAssignment,
    *,
    planner_version: int = PLANNER_VERSION,
) -> tuple[WindowRedistributionPlan, ...]:
    """Build the bidirectional Plan A routing plans for every rank.

    Returns one :class:`WindowRedistributionPlan` per rank, indexed by SP-local
    rank.  Both directions (regular -> shifted and shifted -> regular) are plain
    calls to this function; neither is derived from the other by transposing.
    """
    _check_same_layout(src_layout, src_assignment.layout_key)
    _check_same_layout(dst_layout, dst_assignment.layout_key)
    if src_assignment.world_size != dst_assignment.world_size:
        raise ValueError(
            f"source world_size {src_assignment.world_size} != destination world_size {dst_assignment.world_size}"
        )
    if src_layout.num_tokens != dst_layout.num_tokens:
        raise ValueError(f"layouts cover different token counts: {src_layout.num_tokens} != {dst_layout.num_tokens}")
    world_size = src_assignment.world_size

    src_ids = _rank_global_token_ids(src_layout, src_assignment)
    dst_ids = _rank_global_token_ids(dst_layout, dst_assignment)
    num_tokens = src_layout.num_tokens

    # global token id -> (source owner rank, local row)
    src_owner = torch.full((num_tokens,), -1, dtype=torch.int64)
    src_local_pos = torch.full((num_tokens,), -1, dtype=torch.int64)
    for r, ids in enumerate(src_ids):
        if ids.numel():
            src_owner[ids] = r
            src_local_pos[ids] = torch.arange(ids.numel(), dtype=torch.int64)
    if bool((src_owner < 0).any()):
        raise ValueError("source assignment does not cover every token exactly once")

    counts = torch.zeros((world_size, world_size), dtype=torch.int64)  # counts[src, dst]
    send_rows: dict[tuple[int, int], torch.Tensor] = {}
    dst_positions: dict[tuple[int, int], torch.Tensor] = {}

    for d in range(world_size):
        z = dst_ids[d]
        if z.numel():
            owners = src_owner[z]
            if bool((owners < 0).any()):
                raise ValueError(f"destination rank {d} requires tokens missing from the source layouts")
            order = torch.argsort(owners, stable=True)
            offsets = torch.cumsum(torch.bincount(owners, minlength=world_size), dim=0)
        else:
            order = torch.empty(0, dtype=torch.int64)
            offsets = torch.zeros(world_size, dtype=torch.int64)

        start = 0
        for r in range(world_size):
            end = int(offsets[r])
            seg = order[start:end]
            start = end
            counts[r, d] = end - (int(offsets[r - 1]) if r else 0)
            if seg.numel():
                send_rows[(r, d)] = src_local_pos[z[seg]]
                dst_positions[(r, d)] = seg
            else:
                send_rows[(r, d)] = torch.empty(0, dtype=torch.int64)
                dst_positions[(r, d)] = torch.empty(0, dtype=torch.int64)

    # Group consistency: computed from the global count matrix, so every rank
    # reaches the same decision and no rank skips a collective alone.
    off_diagonal = counts.clone()
    off_diagonal.fill_diagonal_(0)
    network_exchange_required = bool(int(off_diagonal.sum()) > 0)

    plans: list[WindowRedistributionPlan] = []
    for rank in range(world_size):
        send_indices = torch.cat([send_rows[(rank, d)] for d in range(world_size)])
        received_dst_positions = torch.cat([dst_positions[(s, rank)] for s in range(world_size)])
        num_src_rows = int(src_ids[rank].numel())
        num_dst_rows = int(dst_ids[rank].numel())
        if send_indices.numel() != num_src_rows:
            raise ValueError(f"rank {rank} routes {send_indices.numel()} rows but owns {num_src_rows} source rows")
        if received_dst_positions.numel() != num_dst_rows:
            raise ValueError(
                f"rank {rank} receives {received_dst_positions.numel()} rows but needs {num_dst_rows} destination rows"
            )
        if num_dst_rows and not torch.equal(
            torch.sort(received_dst_positions).values, torch.arange(num_dst_rows, dtype=torch.int64)
        ):
            raise ValueError(f"rank {rank} received positions are not a permutation of its destination rows")
        recv_to_dst_indices = torch.argsort(received_dst_positions)

        input_split_sizes = tuple(int(v) for v in counts[rank].tolist())
        output_split_sizes = tuple(int(v) for v in counts[:, rank].tolist())
        if sum(input_split_sizes) != num_src_rows or sum(output_split_sizes) != num_dst_rows:
            raise ValueError(
                f"rank {rank} split sizes are inconsistent with its row counts "
                f"(send {sum(input_split_sizes)}/{num_src_rows}, recv {sum(output_split_sizes)}/{num_dst_rows})"
            )

        plans.append(
            WindowRedistributionPlan(
                src_key=src_layout.key,
                dst_key=dst_layout.key,
                world_size=world_size,
                rank=rank,
                input_split_sizes=input_split_sizes,
                output_split_sizes=output_split_sizes,
                send_indices=send_indices,
                recv_to_dst_indices=recv_to_dst_indices,
                network_exchange_required=network_exchange_required,
                planner_version=planner_version,
            )
        )
    return tuple(plans)


# =============================================================================
# Cache
# =============================================================================


class WindowPlanCache:
    """Small LRU cache for layouts, assignments, rank plans and routes.

    The assignment / routing identity includes the SP world size, the planner
    version and the geometry fingerprint, so a change in any of them cannot hit
    a stale entry.  Runtime (device) objects are intentionally *not* cached
    here: they are owned by the caller and must be released when the process
    group or device changes.
    """

    def __init__(self, capacity: int = 32) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._layouts: OrderedDict[WindowLayoutKey, WindowLayout] = OrderedDict()
        self._assignments: OrderedDict[tuple, WindowAssignment] = OrderedDict()
        self._rank_plans: OrderedDict[tuple, RankWindowPlan] = OrderedDict()
        self._routes: OrderedDict[tuple, tuple[WindowRedistributionPlan, ...]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _touch(self, cache: OrderedDict, key):
        value = cache.pop(key)
        cache[key] = value
        return value

    def _store(self, cache: OrderedDict, key, value):
        cache[key] = value
        while len(cache) > self._capacity:
            cache.popitem(last=False)
        return value

    def clear(self) -> None:
        self._layouts.clear()
        self._assignments.clear()
        self._rank_plans.clear()
        self._routes.clear()

    def layout(self, token_grid, window_method, window=DEFAULT_WINDOW, geometry_version=GEOMETRY_VERSION):
        key = (tuple(token_grid), window_method, tuple(window), int(geometry_version))
        if key in self._layouts:
            self.hits += 1
            return self._touch(self._layouts, key)
        self.misses += 1
        layout = build_window_layout(token_grid, window_method, window=window, geometry_version=geometry_version)
        return self._store(self._layouts, key, layout)

    def assignment(self, layout: WindowLayout, world_size: int, planner_version: int = PLANNER_VERSION):
        key = (layout.key, int(world_size), int(planner_version))
        if key in self._assignments:
            self.hits += 1
            return self._touch(self._assignments, key)
        self.misses += 1
        return self._store(
            self._assignments, key, build_window_assignment(layout, world_size, planner_version=planner_version)
        )

    def rank_plan(self, layout: WindowLayout, assignment: WindowAssignment, rank: int):
        key = (
            layout.key,
            assignment.layout_key,
            int(assignment.world_size),
            int(assignment.planner_version),
            int(rank),
        )
        if key in self._rank_plans:
            self.hits += 1
            return self._touch(self._rank_plans, key)
        self.misses += 1
        return self._store(self._rank_plans, key, build_rank_window_plan(layout, assignment, rank))

    def route(
        self,
        src_layout: WindowLayout,
        src_assignment: WindowAssignment,
        dst_layout: WindowLayout,
        dst_assignment: WindowAssignment,
    ):
        key = (
            src_layout.key,
            src_assignment.world_size,
            src_assignment.planner_version,
            dst_layout.key,
            dst_assignment.world_size,
            dst_assignment.planner_version,
        )
        if key in self._routes:
            self.hits += 1
            return self._touch(self._routes, key)
        self.misses += 1
        plans = build_redistribution_plans(src_layout, src_assignment, dst_layout, dst_assignment)
        return self._store(self._routes, key, plans)


# =============================================================================
# Runtime (device)
# =============================================================================


def materialize_redistribution_plan(
    plan: WindowRedistributionPlan, *, device: torch.device | str
) -> DeviceWindowRedistributionPlan:
    """Move one CPU redistribution plan to the device (index tensors only)."""
    device = torch.device(device)
    return DeviceWindowRedistributionPlan(
        send_indices=plan.send_indices.to(device=device, dtype=torch.int64, non_blocking=True),
        recv_to_dst_indices=plan.recv_to_dst_indices.to(device=device, dtype=torch.int64, non_blocking=True),
        input_split_sizes=plan.input_split_sizes,
        output_split_sizes=plan.output_split_sizes,
        num_recv_rows=plan.num_recv_rows,
        network_exchange_required=plan.network_exchange_required,
    )


def redistribute_window_rows(
    hidden: torch.Tensor,
    plan: DeviceWindowRedistributionPlan,
    *,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Re-shard local video rows from one window layout to another (Plan A).

    ``hidden`` is ``[N_src, C]`` (contiguous trailing feature dimension) in the
    source layout's rank-local row order; the result is ``[N_dst, C]`` in the
    destination layout's rank-local row order.  Only row placement and order
    change -- values are moved bit-exactly.

    The first version is deliberately synchronous: no extra streams, no
    ``cuda.synchronize()`` / barrier "fixups", and no hidden catch-up all-gather.
    """
    if hidden.dim() != 2:
        raise ValueError(f"hidden must be 2D [rows, features], got shape {tuple(hidden.shape)}")
    expected_src = sum(plan.input_split_sizes)
    if hidden.shape[0] != expected_src:
        raise ValueError(f"hidden has {hidden.shape[0]} rows but the plan sends {expected_src}")

    if not plan.network_exchange_required:
        # No rank crosses a boundary: every token stays local, only the packed
        # order changes (e.g. SP = 1, or identical layouts).
        if plan.send_indices.numel() == hidden.shape[0]:
            return hidden.index_select(0, plan.send_indices)
        return hidden

    send = hidden.index_select(0, plan.send_indices).contiguous()
    recv = torch.empty(
        (plan.num_recv_rows, hidden.shape[1]),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=list(plan.output_split_sizes),
        input_split_sizes=list(plan.input_split_sizes),
        group=group,
        async_op=False,
    )
    if plan.recv_to_dst_indices.numel() == plan.num_recv_rows:
        return recv.index_select(0, plan.recv_to_dst_indices)
    return recv


def reduction_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulator dtype for a window reduction (fp16/bf16 reduce in fp32)."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def global_window_mean(
    local_window_sum: torch.Tensor,
    global_window_count: int,
    *,
    group: dist.ProcessGroup | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Reference text reduction: ``(sum over all windows) / global_window_count``.

    Every rank contributes the *sum* of its own windows' text attention output
    (a zero tensor on ranks without windows) and all ranks divide by the global
    window count.  This is neither a token-weighted mean nor a mean of rank
    means.

    ``group=None`` means *local* reduction: no collective is issued, and ``None``
    is never forwarded to ``all_reduce`` (which would silently use the default
    process group).  fp16/bf16 inputs are accumulated in fp32, fp32/fp64 keep
    their own precision, the caller's tensor is not modified, and ``dtype`` is
    applied only after the reduction and the division.
    """
    if global_window_count <= 0:
        raise ValueError(f"global_window_count must be positive, got {global_window_count}")
    accumulate = reduction_dtype(local_window_sum.dtype)
    if group is None:
        total = local_window_sum.to(accumulate)
    else:
        total = local_window_sum.to(accumulate).clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    total = total / float(global_window_count)
    return total.to(dtype) if dtype is not None else total


def transition_count(methods: Sequence[str], num_layers: int | None = None) -> int:
    """Number of layout transitions for a layer schedule (diagnostic only).

    The 3B schedule alternates, so every consecutive layer pair is a transition;
    the entry distribution and the final reconstruction are counted separately
    by the caller.
    """
    schedule = [methods[i % len(methods)] for i in range(num_layers if num_layers is not None else len(methods))]
    return sum(1 for a, b in zip(schedule, schedule[1:]) if a != b)


def schedule_methods(num_layers: int, methods: Iterable[str] = DEFAULT_WINDOW_METHODS) -> tuple[str, ...]:
    methods = tuple(methods)
    return tuple(methods[i % len(methods)] for i in range(num_layers))


# Every key here carries the geometry fingerprint, the SP world size and the
# planner version, and nothing device-owned is stored, so one cache can serve
# every request in the process. Rebuilding it per request costs seconds on a
# large token grid, which is pure host time with the accelerators idle.
_SHARED_PLAN_CACHE = WindowPlanCache(capacity=64)


def shared_plan_cache() -> WindowPlanCache:
    """The process-wide plan cache that requests reuse across identical geometry."""
    return _SHARED_PLAN_CACHE


class WindowLayoutManager:
    """Drives ``ensure_layout`` across a model's per-layer window schedule.

    One manager per request / per model instance.  It owns the device plans of
    the current layout pair and is the only place that decides whether a
    transition is needed and in which direction.  The CPU plan cache is shared
    process-wide, so a second request with the same geometry does not rebuild
    layouts and routing plans that cost seconds on a large token grid.
    """

    def __init__(
        self,
        token_grid: tuple[int, int, int],
        *,
        group: dist.ProcessGroup | None,
        world_size: int,
        rank: int,
        window: tuple[int, int, int] = DEFAULT_WINDOW,
        methods: Sequence[str] = DEFAULT_WINDOW_METHODS,
        num_layers: int = 32,
        cache: WindowPlanCache | None = None,
        planner_version: int = PLANNER_VERSION,
    ) -> None:
        if group is not None and world_size != dist.get_world_size(group):
            raise ValueError(
                f"world_size {world_size} does not match the window-SP group size {dist.get_world_size(group)}"
            )
        if group is None and world_size != 1:
            raise ValueError("a window-SP group is required when world_size > 1")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        self.token_grid = (int(token_grid[0]), int(token_grid[1]), int(token_grid[2]))
        self.group = group
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.window = tuple(window)
        self.methods = tuple(methods)
        self.num_layers = int(num_layers)
        self.planner_version = int(planner_version)
        self.cache = shared_plan_cache() if cache is None else cache
        self.transitions = 0
        self._device: torch.device | None = None
        self._device_plans: dict[tuple[WindowLayoutKey, WindowLayoutKey], DeviceWindowRedistributionPlan] = {}

    # -- layouts -----------------------------------------------------------
    def layer_method(self, layer_index: int) -> str:
        return self.methods[layer_index % len(self.methods)]

    def layout(self, window_method: str) -> WindowLayout:
        return self.cache.layout(self.token_grid, window_method, self.window)

    def layer_layout(self, layer_index: int) -> WindowLayout:
        return self.layout(self.layer_method(layer_index))

    def layout_for_key(self, key: WindowLayoutKey) -> WindowLayout:
        return self.cache.layout(key.token_grid, key.window_method, key.window_shape)

    def assignment(self, layout: WindowLayout) -> WindowAssignment:
        return self.cache.assignment(layout, self.world_size, self.planner_version)

    def rank_plan(self, layout: WindowLayout, assignment: WindowAssignment | None = None) -> RankWindowPlan:
        return self.rank_plan_for(layout, self.rank, assignment)

    def rank_plan_for(
        self, layout: WindowLayout, rank: int, assignment: WindowAssignment | None = None
    ) -> RankWindowPlan:
        assignment = assignment or self.assignment(layout)
        return self.cache.rank_plan(layout, assignment, rank)

    # -- transitions -------------------------------------------------------
    def device_plan(self, src_layout: WindowLayout, dst_layout: WindowLayout) -> DeviceWindowRedistributionPlan:
        key = (src_layout.key, dst_layout.key)
        plan = self._device_plans.get(key)
        if plan is None:
            src_assignment = self.assignment(src_layout)
            dst_assignment = self.assignment(dst_layout)
            routes = self.cache.route(src_layout, src_assignment, dst_layout, dst_assignment)
            plan = materialize_redistribution_plan(routes[self.rank], device=self.device)
            self._device_plans[key] = plan
        return plan

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("WindowLayoutManager device is unset; call set_device() first")
        return self._device

    def set_device(self, device: torch.device | str) -> None:
        device = torch.device(device)
        if self._device != device:
            self._device = device
            self._device_plans.clear()

    def ensure_layout(
        self,
        hidden: torch.Tensor,
        src_layout_key: WindowLayoutKey,
        dst_layout_key: WindowLayoutKey,
    ) -> torch.Tensor:
        """Return ``hidden`` re-sharded into the destination layout (or as-is)."""
        if src_layout_key == dst_layout_key:
            return hidden
        self.set_device(hidden.device)
        src_layout = self.cache.layout(
            src_layout_key.token_grid, src_layout_key.window_method, src_layout_key.window_shape
        )
        dst_layout = self.cache.layout(
            dst_layout_key.token_grid, dst_layout_key.window_method, dst_layout_key.window_shape
        )
        plan = self.device_plan(src_layout, dst_layout)
        if not plan.network_exchange_required and plan.send_indices.numel() == hidden.shape[0]:
            if torch.equal(plan.send_indices, torch.arange(hidden.shape[0], device=hidden.device)):
                return hidden
        self.transitions += 1
        return redistribute_window_rows(hidden, plan, group=self.group)
