# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pure-function chunk timetable, Latest-KV visibility, and transfer plan.

No torch. ``S ∈ {1, T+1}`` only.

* ``S = 1``: traditional layer-split (``SERIAL`` / ``INTERLEAVED``).
* ``S = T+1``: one denoise-stage replica per denoise/clean step. ``SERIAL`` keeps a single
  chunk in flight (Self Forcing baseline on the same topology); non-serial
  uses the diagonal Latest-KV pipeline (multiple chunks overlapped).

``stages`` / ``stage_of`` name the denoise-stage axis ``S`` (weight replicas), not Omni
deploy ``stage_id``. Orchestration modules are ``chunk_schedule`` / ``chunk_executor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

ChunkStep = tuple[int, int]


class Ordering(str, Enum):
    SERIAL = "serial"
    INTERLEAVED = "interleaved"


@dataclass(frozen=True)
class KVSource:
    version: ChunkStep
    owner: int


@dataclass(frozen=True)
class KVTransfer:
    version: ChunkStep
    src: int
    dst: int


@dataclass(frozen=True)
class RequestKVTransfer:
    req: str
    version: ChunkStep
    src: int
    dst: int


@dataclass(frozen=True)
class ChunkSchedule:
    chunks: int
    num_denoise_steps: int
    stages: int
    layer_groups: int
    ordering: Ordering
    kv_history_chunks: int

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


@dataclass(frozen=True)
class Inflight:
    req: str
    t0: int
    plan: ChunkPlan


class ChunkPlan:
    """Immutable per-request timetable plus derived Latest-KV tables."""

    def __init__(
        self,
        schedule: ChunkSchedule,
        slots: tuple[tuple[ChunkStep | None, ...], ...],
        completions: dict[tuple[ChunkStep, int], int],
        sources: dict[tuple[int, ChunkStep], tuple[KVSource, ...]],
        transfers: dict[int, tuple[KVTransfer, ...]],
        last_use: dict[int, dict[ChunkStep, int]],
        wait_ready: dict[int, dict[int, frozenset[ChunkStep]]],
    ) -> None:
        self.schedule = schedule
        self._slots = slots
        self._completions = completions
        self._sources = sources
        self._transfers = transfers
        self._last_use = last_use
        self._wait_ready = wait_ready
        self.num_slots = len(slots)
        self.world = schedule.stages * schedule.layer_groups

    def task(self, slot: int, rank: int) -> ChunkStep | None:
        if slot < 0 or slot >= self.num_slots:
            return None
        if rank < 0 or rank >= self.world:
            raise ValueError(f"rank {rank} out of range for world {self.world}")
        return self._slots[slot][rank]

    def completion_slot(self, version: ChunkStep, layer_group: int) -> int:
        return self._completions[(version, layer_group)]

    def sources(self, task: ChunkStep, rank: int) -> tuple[KVSource, ...]:
        return self._sources.get((rank, task), ())

    def transfers(self, slot: int) -> tuple[KVTransfer, ...]:
        return self._transfers.get(slot, ())

    def last_use(self, rank: int) -> dict[ChunkStep, int]:
        return dict(self._last_use.get(rank, {}))

    def wait_ready(self, slot: int, rank: int) -> frozenset[ChunkStep]:
        """Incoming versions of ``slot`` that ``rank`` must await before ``slot + 1``."""
        return self._wait_ready.get(rank, {}).get(slot, frozenset())


def stage_of(step: int, schedule: ChunkSchedule) -> int:
    return 0 if schedule.stages == 1 else step


def rank_of(step: int, layer_group: int, schedule: ChunkSchedule) -> int:
    if schedule.stages == 1:
        return layer_group
    return step * schedule.layer_groups + layer_group


def _s1_jobs(schedule: ChunkSchedule) -> list[ChunkStep]:
    steps = schedule.num_denoise_steps + (1 if schedule.kv_history_chunks > 0 else 0)
    chunks = schedule.chunks
    if schedule.ordering is Ordering.SERIAL:
        return [(chunk, step) for chunk in range(chunks) for step in range(steps)]
    if schedule.ordering is Ordering.INTERLEAVED:
        world = schedule.layer_groups
        return [
            (chunk, step)
            for first in range(0, chunks, world)
            for step in range(steps)
            for chunk in range(first, min(first + world, chunks))
        ]
    raise ValueError(f"unsupported ordering {schedule.ordering}")


def _plan_s1_slots(schedule: ChunkSchedule) -> tuple[tuple[ChunkStep | None, ...], ...]:
    world = schedule.layer_groups
    jobs = _s1_jobs(schedule)
    serial = schedule.ordering is Ordering.SERIAL
    slots: list[tuple[ChunkStep | None, ...]] = []
    completed: set[ChunkStep] = set()
    carry: list[ChunkStep | None] = [None] * (world - 1)
    cursor = 0
    while cursor < len(jobs) or any(task is not None for task in carry):
        launched: ChunkStep | None = None
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


def _plan_vertical_slots(schedule: ChunkSchedule) -> tuple[tuple[ChunkStep | None, ...], ...]:
    """Diagonal Latest-KV: chunk ``c`` and chunk ``c+1`` overlap across ranks."""
    n = schedule.chunks
    s = schedule.stages
    g = schedule.layer_groups
    world = s * g
    num_slots = n + world - 1
    slots = []
    for t in range(num_slots):
        row: list[ChunkStep | None] = [None] * world
        for rank in range(world):
            chunk = t - rank
            if 0 <= chunk < n:
                row[rank] = (chunk, rank // g)
        slots.append(tuple(row))
    return tuple(slots)


def _plan_vertical_serial_slots(schedule: ChunkSchedule) -> tuple[tuple[ChunkStep | None, ...], ...]:
    """Same ``S×G`` topology as Latest-KV, but one chunk finishes every cell first.

    Implemented as ``N`` concatenated single-chunk vertical waves so layer-group
    PP within a stage still fills, while chunk ``c+1`` never overlaps chunk ``c``.
    """
    unit = _plan_vertical_slots(
        ChunkSchedule(
            chunks=1,
            num_denoise_steps=schedule.num_denoise_steps,
            stages=schedule.stages,
            layer_groups=schedule.layer_groups,
            ordering=Ordering.INTERLEAVED,
            kv_history_chunks=schedule.kv_history_chunks,
        )
    )
    slots: list[tuple[ChunkStep | None, ...]] = []
    for chunk in range(schedule.chunks):
        for row in unit:
            slots.append(tuple((chunk, task[1]) if task is not None else None for task in row))
    return tuple(slots)


def _completions(
    slots: tuple[tuple[ChunkStep | None, ...], ...],
    schedule: ChunkSchedule,
) -> dict[tuple[ChunkStep, int], int]:
    out: dict[tuple[ChunkStep, int], int] = {}
    g_size = schedule.layer_groups
    for t, row in enumerate(slots):
        for rank, task in enumerate(row):
            if task is None:
                continue
            g = rank % g_size if schedule.stages > 1 else rank
            out[(task, g)] = t
    return out


def _sources_and_transfers(
    slots: tuple[tuple[ChunkStep | None, ...], ...],
    completions: dict[tuple[ChunkStep, int], int],
    schedule: ChunkSchedule,
) -> tuple[dict[tuple[int, ChunkStep], tuple[KVSource, ...]], dict[int, tuple[KVTransfer, ...]]]:
    h = schedule.kv_history_chunks
    t_clean = schedule.num_denoise_steps
    g_size = schedule.layer_groups
    sources: dict[tuple[int, ChunkStep], tuple[KVSource, ...]] = {}
    transfer_acc: dict[int, list[KVTransfer]] = {}
    seen: set[tuple[ChunkStep, int]] = set()
    if h < 1:
        return sources, {}

    for t, row in enumerate(slots):
        for rank, task in enumerate(row):
            if task is None:
                continue
            chunk, step = task
            g = rank % g_size if schedule.stages > 1 else rank
            found: list[KVSource] = []
            for prev in range(max(0, chunk - h), chunk):
                candidates = [
                    s_prime for s_prime in range(t_clean + 1) if completions.get(((prev, s_prime), g), 10**9) < t
                ]
                if not candidates:
                    continue
                s_star = max(candidates)
                version = (prev, s_star)
                owner = rank_of(s_star, g, schedule)
                found.append(KVSource(version=version, owner=owner))
                if owner != rank:
                    key = (version, rank)
                    if key not in seen:
                        seen.add(key)
                        prod = completions[(version, g)]
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
    """Transfers produced exactly at ``slot`` whose receiver is ``rank``.

    Used for the narrowed wait: a rank only blocks on versions it produced at
    the previous slot minus those the next schedule step consumes itself.
    """
    return tuple(xfer for xfer in transfers.get(slot, ()) if xfer.dst == rank)


def next_consumed(
    slots: tuple[tuple[ChunkStep | None, ...], ...],
    sources: dict[tuple[int, ChunkStep], tuple[KVSource, ...]],
    slot: int,
    rank: int,
) -> set[ChunkStep]:
    """Versions of ``slot`` incoming transfers consumed by ``rank`` at ``slot + 1``."""
    upcoming = _slot_or_empty(slots, slot + 1)
    if rank >= len(upcoming) or upcoming[rank] is None:
        return set()
    return {src.version for src in sources.get((rank, upcoming[rank]), ())}


def _slot_or_empty(
    slots: tuple[tuple[ChunkStep | None, ...], ...],
    slot: int,
) -> tuple[ChunkStep | None, ...]:
    if slot < 0 or slot >= len(slots):
        return ()
    return slots[slot]


def _last_use(
    slots: tuple[tuple[ChunkStep | None, ...], ...],
    sources: dict[tuple[int, ChunkStep], tuple[KVSource, ...]],
    transfers: dict[int, tuple[KVTransfer, ...]],
    schedule: ChunkSchedule,
) -> dict[int, dict[ChunkStep, int]]:
    world = schedule.stages * schedule.layer_groups
    last: dict[int, dict[ChunkStep, int]] = {rank: {} for rank in range(world)}
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


def _assert_invariants(plan: ChunkPlan) -> None:
    schedule = plan.schedule
    h = schedule.kv_history_chunks
    g_size = schedule.layer_groups
    for t in range(plan.num_slots):
        for rank in range(plan.world):
            task = plan.task(t, rank)
            if task is None or h < 1:
                continue
            g = rank % g_size if schedule.stages > 1 else rank
            for src in plan.sources(task, rank):
                p = plan.completion_slot(src.version, g)
                if not p < t:
                    raise AssertionError(f"I2 violated: {src.version} P_g={p} not < consume {t}")
                if src.owner != rank_of(src.version[1], g, schedule):
                    raise AssertionError(f"I3 violated: owner mismatch for {src}")
        posted: set[tuple[ChunkStep, int]] = set()
        for xfer in plan.transfers(t):
            g = xfer.src % g_size if schedule.stages > 1 else xfer.src
            if plan.completion_slot(xfer.version, g) != t:
                raise AssertionError(f"I4 violated: transfer {xfer} not at production slot")
            key = (xfer.version, xfer.dst)
            if key in posted:
                raise AssertionError(f"I4 violated: duplicate transfer {key} at slot {t}")
            posted.add(key)


def build_chunk_plan(schedule: ChunkSchedule) -> ChunkPlan:
    if schedule.stages == 1:
        slots = _plan_s1_slots(schedule)
    elif schedule.ordering is Ordering.SERIAL:
        slots = _plan_vertical_serial_slots(schedule)
    else:
        slots = _plan_vertical_slots(schedule)
    completions = _completions(slots, schedule)
    sources, transfers = _sources_and_transfers(slots, completions, schedule)
    last_use = _last_use(slots, sources, transfers, schedule)
    world = schedule.stages * schedule.layer_groups
    wait_ready: dict[int, dict[int, frozenset[ChunkStep]]] = {rank: {} for rank in range(world)}
    for slot in range(len(slots)):
        for rank in range(world):
            consumed = next_consumed(slots, sources, slot, rank)
            if not consumed:
                continue
            pending = frozenset(
                xfer.version for xfer in incoming_transfers(transfers, slot, rank) if xfer.version in consumed
            )
            if pending:
                wait_ready[rank][slot] = pending
    plan = ChunkPlan(schedule, slots, completions, sources, transfers, last_use, wait_ready)
    _assert_invariants(plan)
    return plan


def rank_work(inflight: tuple[Inflight, ...], slot: int, rank: int) -> tuple[tuple[str, ChunkStep], ...]:
    out: list[tuple[str, ChunkStep]] = []
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
) -> frozenset[tuple[str, ChunkStep]]:
    """``(req, version)`` whose inbound handles ``rank`` must wait before ``slot + 1``."""
    acc: set[tuple[str, ChunkStep]] = set()
    for item in inflight:
        local = slot - item.t0
        for version in item.plan.wait_ready(local, rank):
            acc.add((item.req, version))
    return frozenset(acc)


def can_admit(
    inflight: tuple[Inflight, ...],
    slot: int,
    plan: ChunkPlan,
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
