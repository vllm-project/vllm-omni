# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pure-function cell timetable, Latest-KV visibility, and transfer plan.

No torch. ``S ∈ {1, T+1}``. Schedule/KV unit is cell ``(chunk, step, block)``.

* Weight split: PP layer-groups ``G``; each coarse ``(c,s)`` expands to local blocks.
* ``S = 1``: layer PP. ``S = T+1``: one replica per denoise/clean step.
* ``SELF_FORCING``: one chunk in flight (Self-Forcing / clean history);
  ``LATEST_KV``: diagonal Latest-KV.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

# (chunk, step, block) — schedule task and KV version.
Cell = tuple[int, int, int]


class Ordering(str, Enum):
    SELF_FORCING = "self_forcing"
    LATEST_KV = "latest_kv"
    # Accepted string aliases for request / deploy configs.
    SERIAL = "self_forcing"
    INTERLEAVED = "latest_kv"


@dataclass(frozen=True)
class KVSource:
    version: Cell
    owner: int


@dataclass(frozen=True)
class KVTransfer:
    version: Cell
    src: int
    dst: int


@dataclass(frozen=True)
class RequestKVTransfer:
    req: str
    version: Cell
    src: int
    dst: int


@dataclass(frozen=True)
class CellSchedule:
    chunks: int
    num_denoise_steps: int
    stages: int
    layer_groups: int
    ordering: Ordering
    kv_history_chunks: int
    #: Total DiT blocks. Default ``layer_groups`` ⇒ one block per group (K=1).
    blocks: int | None = None
    #: Contiguous blocks per rank in a stage. Default ``1``.
    blocks_per_rank: int | None = None

    def __post_init__(self) -> None:
        if self.chunks < 1:
            raise ValueError(f"chunks must be positive, got {self.chunks}")
        if self.num_denoise_steps < 1:
            raise ValueError(f"num_denoise_steps must be positive, got {self.num_denoise_steps}")
        if self.layer_groups < 1:
            raise ValueError(f"layer_groups must be positive, got {self.layer_groups}")
        if self.kv_history_chunks < 0:
            raise ValueError(f"kv_history_chunks must be non-negative, got {self.kv_history_chunks}")
        t_plus_1 = self.num_denoise_steps + 1
        if self.stages not in (1, t_plus_1):
            raise ValueError(f"stages must be 1 or num_denoise_steps+1 ({t_plus_1}), got {self.stages}")
        if self.stages > 1 and self.kv_history_chunks < 1:
            raise ValueError("vertical slice (S > 1) requires kv_history_chunks > 0")
        blocks = self.layer_groups if self.blocks is None else self.blocks
        k = 1 if self.blocks_per_rank is None else self.blocks_per_rank
        if blocks < 1:
            raise ValueError(f"blocks must be positive, got {blocks}")
        if k < 1:
            raise ValueError(f"blocks_per_rank must be positive, got {k}")
        if self.blocks is not None:
            # Deploy may set G independently of ceil(B/K); only reject empty groups.
            start0, end0 = _block_range(blocks, 0, self.layer_groups)
            if end0 - start0 < 1:
                raise ValueError(f"layer_groups={self.layer_groups} leaves group 0 empty for blocks={blocks}")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "blocks_per_rank", k)

    @property
    def num_blocks(self) -> int:
        assert self.blocks is not None
        return self.blocks


@dataclass(frozen=True)
class Inflight:
    req: str
    t0: int
    plan: CellPlan


class CellPlan:
    """Immutable per-request timetable plus derived Latest-KV tables."""

    def __init__(
        self,
        schedule: CellSchedule,
        slots: tuple[tuple[Cell | None, ...], ...],
        completions: dict[Cell, int],
        sources: dict[tuple[int, Cell], tuple[KVSource, ...]],
        transfers: dict[int, tuple[KVTransfer, ...]],
        last_use: dict[int, dict[Cell, int]],
        wait_ready: dict[int, dict[int, frozenset[Cell]]],
    ) -> None:
        self.schedule = schedule
        self._slots = slots
        self._completions = completions
        self._sources = sources
        self._transfers = transfers
        self._last_use = last_use
        self._wait_ready = wait_ready
        self.num_slots = len(slots)
        self.num_ticks = self.num_slots
        self.world = schedule.stages * schedule.layer_groups

    def task(self, slot: int, rank: int) -> Cell | None:
        if slot < 0 or slot >= self.num_slots:
            return None
        if rank < 0 or rank >= self.world:
            raise ValueError(f"rank {rank} out of range for world {self.world}")
        return self._slots[slot][rank]

    def completion_tick(self, version: Cell) -> int:
        return self._completions[version]

    def completion_slot(self, version: Cell, layer_group: int | None = None) -> int:
        """Alias of ``completion_tick`` (layer_group ignored; kept for call sites)."""
        del layer_group
        return self.completion_tick(version)

    def sources(self, task: Cell, rank: int) -> tuple[KVSource, ...]:
        return self._sources.get((rank, task), ())

    def transfers(self, slot: int) -> tuple[KVTransfer, ...]:
        return self._transfers.get(slot, ())

    def last_use(self, rank: int) -> dict[Cell, int]:
        return dict(self._last_use.get(rank, {}))

    def wait_ready(self, slot: int, rank: int) -> frozenset[Cell]:
        """Remote source versions that ``rank`` first consumes at ``slot + 1`` (await by end of ``slot``)."""
        return self._wait_ready.get(rank, {}).get(slot, frozenset())


def stage_of(step: int, schedule: CellSchedule) -> int:
    return 0 if schedule.stages == 1 else step


@lru_cache(maxsize=128)
def _block_range(num_blocks: int, group: int, groups: int) -> tuple[int, int]:
    """Layer partition; must match Wan ``get_pp_indices`` / ``stage_layer_range``."""
    from vllm.distributed.utils import get_pp_indices

    if groups < 1:
        raise ValueError("groups must be positive")
    if not 0 <= group < groups:
        raise ValueError(f"group {group} out of range for {groups}")
    return get_pp_indices(num_blocks, group, groups)


def group_of_block(block: int, schedule: CellSchedule) -> int:
    for g in range(schedule.layer_groups):
        start, end = _block_range(schedule.num_blocks, g, schedule.layer_groups)
        if start <= block < end:
            return g
    raise ValueError(f"block {block} out of range for blocks={schedule.num_blocks}")


def rank_of(step: int, layer_group: int, schedule: CellSchedule) -> int:
    if schedule.stages == 1:
        return layer_group
    return step * schedule.layer_groups + layer_group


def rank_of_block(step: int, block: int, schedule: CellSchedule) -> int:
    return rank_of(step, group_of_block(block, schedule), schedule)


def is_group_boundary(cell: Cell, schedule: CellSchedule) -> bool:
    """True when ``cell`` is the last local block of its layer group (activation edge)."""
    _chunk, _step, block = cell
    g = group_of_block(block, schedule)
    _start, end = _block_range(schedule.num_blocks, g, schedule.layer_groups)
    return block == end - 1


def max_local_blocks(schedule: CellSchedule) -> int:
    return max(
        _block_range(schedule.num_blocks, g, schedule.layer_groups)[1]
        - _block_range(schedule.num_blocks, g, schedule.layer_groups)[0]
        for g in range(schedule.layer_groups)
    )


def _s1_jobs(schedule: CellSchedule) -> list[tuple[int, int]]:
    steps = schedule.num_denoise_steps + (1 if schedule.kv_history_chunks > 0 else 0)
    chunks = schedule.chunks
    if schedule.ordering is Ordering.SELF_FORCING:
        return [(chunk, step) for chunk in range(chunks) for step in range(steps)]
    if schedule.ordering is Ordering.LATEST_KV:
        world = schedule.layer_groups
        return [
            (chunk, step)
            for first in range(0, chunks, world)
            for step in range(steps)
            for chunk in range(first, min(first + world, chunks))
        ]
    raise ValueError(f"unsupported ordering {schedule.ordering}")


def _plan_s1_coarse(schedule: CellSchedule) -> tuple[tuple[tuple[int, int] | None, ...], ...]:
    world = schedule.layer_groups
    jobs = _s1_jobs(schedule)
    serial = schedule.ordering is Ordering.SELF_FORCING
    slots: list[tuple[tuple[int, int] | None, ...]] = []
    completed: set[tuple[int, int]] = set()
    carry: list[tuple[int, int] | None] = [None] * (world - 1)
    cursor = 0
    while cursor < len(jobs) or any(task is not None for task in carry):
        launched: tuple[int, int] | None = None
        if cursor < len(jobs) and not (serial and any(task is not None for task in carry)):
            chunk, step = jobs[cursor]
            dependencies = {(chunk, step - 1)} if step else set()
            if dependencies <= completed:
                launched = jobs[cursor]
                cursor += 1
        slot = (launched, *carry)
        if all(task is None for task in slot):
            raise RuntimeError("Chunk schedule cannot satisfy the next task's dependencies")
        slots.append(slot)
        if slot[-1] is not None:
            completed.add(slot[-1])
        carry = list(slot[:-1])
    return tuple(slots)


def _plan_vertical_coarse(schedule: CellSchedule) -> tuple[tuple[tuple[int, int] | None, ...], ...]:
    """Diagonal Latest-KV at layer-group granularity (before block expand)."""
    n = schedule.chunks
    s = schedule.stages
    g = schedule.layer_groups
    world = s * g
    num_slots = n + world - 1
    slots = []
    for t in range(num_slots):
        row: list[tuple[int, int] | None] = [None] * world
        for rank in range(world):
            chunk = t - rank
            if 0 <= chunk < n:
                row[rank] = (chunk, rank // g)
        slots.append(tuple(row))
    return tuple(slots)


def _plan_vertical_serial_coarse(schedule: CellSchedule) -> tuple[tuple[tuple[int, int] | None, ...], ...]:
    unit = _plan_vertical_coarse(
        CellSchedule(
            chunks=1,
            num_denoise_steps=schedule.num_denoise_steps,
            stages=schedule.stages,
            layer_groups=schedule.layer_groups,
            ordering=Ordering.LATEST_KV,
            kv_history_chunks=schedule.kv_history_chunks,
            blocks=schedule.num_blocks,
            blocks_per_rank=schedule.blocks_per_rank,
        )
    )
    slots: list[tuple[tuple[int, int] | None, ...]] = []
    for chunk in range(schedule.chunks):
        for row in unit:
            slots.append(tuple((chunk, task[1]) if task is not None else None for task in row))
    return tuple(slots)


def _expand_to_cells(
    coarse: tuple[tuple[tuple[int, int] | None, ...], ...],
    schedule: CellSchedule,
) -> tuple[tuple[Cell | None, ...], ...]:
    """Expand each coarse ``(c,s)`` on a rank into consecutive local-block ticks."""
    g_size = schedule.layer_groups
    max_k = max_local_blocks(schedule)
    out: list[tuple[Cell | None, ...]] = []
    for row in coarse:
        for k in range(max_k):
            fine: list[Cell | None] = []
            for rank, task in enumerate(row):
                if task is None:
                    fine.append(None)
                    continue
                chunk, step = task
                g = rank if schedule.stages == 1 else rank % g_size
                start, end = _block_range(schedule.num_blocks, g, g_size)
                if k < end - start:
                    fine.append((chunk, step, start + k))
                else:
                    fine.append(None)
            if any(cell is not None for cell in fine):
                out.append(tuple(fine))
    return tuple(out)


def _completions(slots: tuple[tuple[Cell | None, ...], ...]) -> dict[Cell, int]:
    out: dict[Cell, int] = {}
    for t, row in enumerate(slots):
        for task in row:
            if task is not None:
                out[task] = t
    return out


def _sources_and_transfers(
    slots: tuple[tuple[Cell | None, ...], ...],
    completions: dict[Cell, int],
    schedule: CellSchedule,
) -> tuple[dict[tuple[int, Cell], tuple[KVSource, ...]], dict[int, tuple[KVTransfer, ...]]]:
    h = schedule.kv_history_chunks
    t_clean = schedule.num_denoise_steps
    sources: dict[tuple[int, Cell], tuple[KVSource, ...]] = {}
    transfer_acc: dict[int, list[KVTransfer]] = {}
    seen: set[tuple[Cell, int]] = set()
    if h < 1:
        return sources, {}

    for t, row in enumerate(slots):
        for rank, task in enumerate(row):
            if task is None:
                continue
            chunk, _step, block = task
            found: list[KVSource] = []
            for prev in range(max(0, chunk - h), chunk):
                candidates = [
                    s_prime for s_prime in range(t_clean + 1) if completions.get((prev, s_prime, block), 10**9) < t
                ]
                if not candidates:
                    continue
                s_star = max(candidates)
                version = (prev, s_star, block)
                owner = rank_of_block(s_star, block, schedule)
                found.append(KVSource(version=version, owner=owner))
                if owner != rank:
                    key = (version, rank)
                    if key not in seen:
                        seen.add(key)
                        prod = completions[version]
                        transfer_acc.setdefault(prod, []).append(KVTransfer(version=version, src=owner, dst=rank))
            sources[(rank, task)] = tuple(found)

    transfers = {
        slot: tuple(sorted(items, key=lambda x: (x.src, x.dst, x.version))) for slot, items in transfer_acc.items()
    }
    return sources, transfers


def incoming_transfers(
    transfers: dict[int, tuple[KVTransfer, ...]],
    slot: int,
    rank: int,
) -> tuple[KVTransfer, ...]:
    return tuple(xfer for xfer in transfers.get(slot, ()) if xfer.dst == rank)


def next_consumed(
    slots: tuple[tuple[Cell | None, ...], ...],
    sources: dict[tuple[int, Cell], tuple[KVSource, ...]],
    slot: int,
    rank: int,
) -> set[Cell]:
    upcoming = _slot_or_empty(slots, slot + 1)
    if rank >= len(upcoming) or upcoming[rank] is None:
        return set()
    return {src.version for src in sources.get((rank, upcoming[rank]), ())}


def _slot_or_empty(
    slots: tuple[tuple[Cell | None, ...], ...],
    slot: int,
) -> tuple[Cell | None, ...]:
    if slot < 0 or slot >= len(slots):
        return ()
    return slots[slot]


def _last_use(
    slots: tuple[tuple[Cell | None, ...], ...],
    sources: dict[tuple[int, Cell], tuple[KVSource, ...]],
    transfers: dict[int, tuple[KVTransfer, ...]],
    schedule: CellSchedule,
) -> dict[int, dict[Cell, int]]:
    world = schedule.stages * schedule.layer_groups
    last: dict[int, dict[Cell, int]] = {rank: {} for rank in range(world)}
    for t, row in enumerate(slots):
        for rank, task in enumerate(row):
            if task is None:
                continue
            last[rank][task] = max(last[rank].get(task, -1), t)
            for src in sources.get((rank, task), ()):
                last[rank][src.version] = max(last[rank].get(src.version, -1), t)
        for xfer in transfers.get(t, ()):
            last[xfer.src][xfer.version] = max(last[xfer.src].get(xfer.version, -1), t)
            last[xfer.dst][xfer.version] = max(last[xfer.dst].get(xfer.version, -1), t)
    return last


def _assert_invariants(plan: CellPlan) -> None:
    schedule = plan.schedule
    h = schedule.kv_history_chunks
    for t in range(plan.num_slots):
        for rank in range(plan.world):
            task = plan.task(t, rank)
            if task is None or h < 1:
                continue
            for src in plan.sources(task, rank):
                p = plan.completion_tick(src.version)
                if not p < t:
                    raise AssertionError(f"I2 violated: {src.version} P={p} not < consume {t}")
                if src.owner != rank_of_block(src.version[1], src.version[2], schedule):
                    raise AssertionError(f"I3 violated: owner mismatch for {src}")
                if src.version[2] != task[2]:
                    raise AssertionError(f"I10 violated: source block {src.version[2]} != task block {task[2]}")
        posted: set[tuple[Cell, int]] = set()
        for xfer in plan.transfers(t):
            if plan.completion_tick(xfer.version) != t:
                raise AssertionError(f"I4 violated: transfer {xfer} not at production tick")
            key = (xfer.version, xfer.dst)
            if key in posted:
                raise AssertionError(f"I4 violated: duplicate transfer {key} at slot {t}")
            posted.add(key)


def build_cell_plan(schedule: CellSchedule) -> CellPlan:
    if schedule.stages == 1:
        coarse = _plan_s1_coarse(schedule)
    elif schedule.ordering is Ordering.SELF_FORCING:
        coarse = _plan_vertical_serial_coarse(schedule)
    else:
        coarse = _plan_vertical_coarse(schedule)
    slots = _expand_to_cells(coarse, schedule)
    completions = _completions(slots)
    sources, transfers = _sources_and_transfers(slots, completions, schedule)
    last_use = _last_use(slots, sources, transfers, schedule)
    world = schedule.stages * schedule.layer_groups
    # Await remote KV at the tick before first consumption — not only when the
    # transfer lands in the same slot as next-tick consume (cell expand breaks that).
    wait_ready: dict[int, dict[int, frozenset[Cell]]] = {rank: {} for rank in range(world)}
    first_remote_consume: dict[int, dict[Cell, int]] = {rank: {} for rank in range(world)}
    for t, row in enumerate(slots):
        for rank, task in enumerate(row):
            if task is None:
                continue
            for src in sources.get((rank, task), ()):
                if src.owner == rank:
                    continue
                first_remote_consume[rank].setdefault(src.version, t)
    for rank, version_slots in first_remote_consume.items():
        by_wait: dict[int, set[Cell]] = {}
        for version, consume_slot in version_slots.items():
            wait_slot = consume_slot - 1
            if wait_slot < 0:
                # Consumed at slot 0: executor also awaits in prepare path.
                wait_slot = 0
            by_wait.setdefault(wait_slot, set()).add(version)
        for wait_slot, versions in by_wait.items():
            wait_ready[rank][wait_slot] = frozenset(versions)
    plan = CellPlan(schedule, slots, completions, sources, transfers, last_use, wait_ready)
    _assert_invariants(plan)
    return plan


def rank_work(inflight: tuple[Inflight, ...], slot: int, rank: int) -> tuple[tuple[str, Cell], ...]:
    out: list[tuple[str, Cell]] = []
    for item in inflight:
        local = slot - item.t0
        task = item.plan.task(local, rank)
        if task is not None:
            out.append((item.req, task))
    return tuple(out)


def union_transfers(inflight: tuple[Inflight, ...], slot: int) -> tuple[RequestKVTransfer, ...]:
    acc: list[RequestKVTransfer] = []
    for item in inflight:
        local = slot - item.t0
        for xfer in item.plan.transfers(local):
            acc.append(RequestKVTransfer(req=item.req, version=xfer.version, src=xfer.src, dst=xfer.dst))
    acc.sort(key=lambda x: (x.src, x.dst, x.req, x.version))
    return tuple(acc)


def union_wait_ready(
    inflight: tuple[Inflight, ...],
    slot: int,
    rank: int,
) -> frozenset[tuple[str, Cell]]:
    acc: set[tuple[str, Cell]] = set()
    for item in inflight:
        local = slot - item.t0
        for version in item.plan.wait_ready(local, rank):
            acc.add((item.req, version))
    return frozenset(acc)


def can_admit(
    inflight: tuple[Inflight, ...],
    slot: int,
    plan: CellPlan,
    *,
    max_batch_size: int,
) -> bool:
    """Admit when every stage-0 rank still has a free microbatch lane."""
    if max_batch_size < 1:
        return False
    for rank in range(plan.schedule.layer_groups):
        if len(rank_work(inflight, slot, rank)) >= max_batch_size:
            return False
    return True
