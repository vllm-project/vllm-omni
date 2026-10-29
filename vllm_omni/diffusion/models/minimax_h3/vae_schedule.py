# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded integer geometry scheduling for independent H3 ViT decode calls.

No timing model, calibration or device operation belongs here. The caller
supplies real task geometry and a group-minimum incremental memory budget.
"""

from collections import defaultdict
from dataclasses import dataclass
from math import gcd, prod


@dataclass(frozen=True)
class TileTask:
    clip: int
    spatial: int
    batch: int
    shape: tuple[int, int, int, int, int]
    context: tuple[int, int, int, int]  # latent start/stop, head tokens, tail tokens
    output: tuple[int, int]  # native spatial origin

    @property
    def cost(self) -> int:
        return prod(self.shape)


@dataclass(frozen=True)
class TileBucket:
    shape: tuple[int, int, int, int, int]
    ranks: tuple[tuple[int, ...], ...]  # indices into the supplied task tuple


@dataclass(frozen=True)
class WindowAssignment:
    window: int
    waste_units: int
    extra_bytes: int
    rank_tasks: tuple[int, ...]
    rank_work_units: tuple[int, ...]
    reason: str = "eligible"


@dataclass(frozen=True)
class TemporalPlan:
    window: int
    reason: str
    balancing_period: int
    candidates: tuple[WindowAssignment, ...] = ()

    @property
    def optimized(self) -> bool:
        return self.window > 1


def allocate_tiles(tasks: tuple[TileTask, ...], world_size: int) -> tuple[TileBucket, ...]:
    """LPT across shape buckets; equal-shaped tasks alone share a collective.

    Undersized buckets execute replicated, matching the native safe fallback.
    All other buckets assign at least one task per rank. Padding is OUTPUT
    padding only; it never increases a decoder's batch N or computes dummies.
    """
    if type(world_size) is not int or world_size < 1:
        raise ValueError("video group size must be a positive integer")
    groups: dict[tuple[int, int, int, int, int], list[int]] = defaultdict(list)
    for i, task in enumerate(tasks):
        if (
            len(task.shape) != 5
            or any(type(d) is not int or d <= 0 for d in task.shape)
            or type(task.batch) is not int
            or task.batch != task.shape[0]
        ):
            raise ValueError("invalid VAE tile shape or original batch")
        if (
            type(task.clip) is not int
            or task.clip < 0
            or type(task.spatial) is not int
            or task.spatial < 0
            or len(task.context) != 4
            or any(type(d) is not int or d < 0 for d in task.context)
            or task.context[1] <= task.context[0]
            or len(task.output) != 2
            or any(type(d) is not int or d < 0 for d in task.output)
        ):
            raise ValueError("invalid VAE tile identity, context or output position")
        groups[task.shape].append(i)
    loads = [0] * world_size
    buckets = []
    for shape, indices in sorted(groups.items(), key=lambda item: (-prod(item[0]), item[0])):
        assigned: list[list[int]] = [[] for _ in range(world_size)]
        if len(indices) < world_size:
            for rank in range(world_size):
                assigned[rank] = indices.copy()
                loads[rank] += sum(tasks[i].cost for i in indices)
        else:
            for i in sorted(indices, key=lambda i: (-tasks[i].cost, i)):
                # Seed every rank before LPT to satisfy native gather's nonempty contract.
                empty = [r for r in range(world_size) if not assigned[r]]
                rank = min(empty or range(world_size), key=lambda r: (loads[r], r))
                assigned[rank].append(i)
                loads[rank] += tasks[i].cost
        buckets.append(TileBucket(shape, tuple(tuple(a) for a in assigned)))
    return tuple(buckets)


def plan_temporal_schedule(
    clips: tuple[tuple[TileTask, ...], ...],
    world_size: int,
    *,
    budget_bytes: int,
    output_bytes_per_latent_pixel: int,
    input_element_bytes: int,
) -> TemporalPlan:
    """Minimize integer assignment waste, then window size, within W<=16.

    Work units are latent elements, never predicted milliseconds. Shape buckets
    execute serially; replicated undersized buckets count as wasted work too.
    The entire request, including short tails, participates in each candidate.
    Sixteen clips bound buffered outputs/metadata independently of group size;
    the incremental memory check can impose a smaller limit.
    """
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("video group size must be a positive integer")
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("invalid VAE planner memory budget")
    if any(type(v) is not int or v <= 0 for v in (output_bytes_per_latent_pixel, input_element_bytes)):
        raise ValueError("invalid VAE tile storage geometry")
    if not clips or any(not c for c in clips):
        raise ValueError("VAE planner requires nonempty temporal clips and spatial tiles")
    for clip in clips:
        allocate_tiles(clip, world_size)
    period = world_size // gcd(len(clips[0]), world_size)
    if world_size == 1 or len(clips) == 1:
        return TemporalPlan(1, "single_rank_or_clip", period)

    def evaluate(window: int) -> WindowAssignment:
        capacity = useful = peak = 0
        rank_tasks, rank_work = [0] * world_size, [0] * world_size
        for start in range(0, len(clips), window):
            chunk = clips[start : start + window]
            tasks = tuple(t for clip in chunk for t in clip)
            sizes = [prod(t.shape[:1] + t.shape[2:]) * output_bytes_per_latent_pixel for t in tasks]
            if len(chunk) == 1:
                # W1 actually uses native row-major ownership, not shape buckets.
                shares = tuple(tuple(range(r, len(tasks), world_size)) for r in range(world_size))
                if len(tasks) < world_size:
                    shares = (tuple(range(len(tasks))),) * world_size
                buckets = (TileBucket(tasks[0].shape, shares),)
            else:
                buckets = allocate_tiles(tasks, world_size)
            wire_peak = send_peak = 0
            for bucket in buckets:
                work = [sum(tasks[i].cost for i in share) for share in bucket.ranks]
                capacity += max(work) * world_size
                for rank, share in enumerate(bucket.ranks):
                    rank_tasks[rank] += len(share)
                    rank_work[rank] += work[rank]
                if len({i for share in bucket.ranks for i in share}) >= world_size:
                    send = max(sum(sizes[i] for i in share) for share in bucket.ranks)
                    send_peak = max(send_peak, send)
                    wire_peak = max(wire_peak, send * world_size)
            useful += sum(t.cost for t in tasks)
            raw = sum(sizes)
            # Raw outputs and stitched canvases, send stack, persistent receive
            # plus unpack clones. Reserve input copies too (head/tail concat).
            # Each decoder forward retains original N, so its activation
            # workspace does not multiply with the temporal window.
            live = 2 * raw + 2 * wire_peak + send_peak + sum(t.cost for t in tasks) * input_element_bytes
            peak = max(peak, live)
        return WindowAssignment(window, capacity - useful, peak, tuple(rank_tasks), tuple(rank_work))

    native = evaluate(1)
    candidates = []
    for window in range(1, min(16, len(clips)) + 1):
        item = native if window == 1 else evaluate(window)
        extra = max(0, item.extra_bytes - native.extra_bytes)
        candidates.append(
            WindowAssignment(
                window,
                item.waste_units,
                extra,
                item.rank_tasks,
                item.rank_work_units,
                "eligible" if extra <= budget_bytes else "incremental_memory_budget",
            )
        )
    best = min((c for c in candidates if c.reason == "eligible"), key=lambda c: (c.waste_units, c.window))
    reason = "minimum_assignment_waste" if best.window > 1 else "native_assignment_or_tail"
    if best.window == 1 and all(c.reason != "eligible" for c in candidates[1:]):
        reason = "incremental_memory_budget"
    return TemporalPlan(best.window, reason, period, tuple(candidates))
