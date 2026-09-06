# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Runs WriteTasks and the step D2H staging pool.

The manager owns request/slot identity and when to submit. This class
owns the staging pool, the GPU-byte cap, the copy queues, and the
single committer that scatters into the CPU block pool.

Two D2H paths — the step path does not write `seg.host` in the committer:

    JOIN_NEXT_STEP   save already launched a whole-step D2H into a
                     staging slot and hung `seg.host` as views.
                     Committer waits that `step_d2h_event`, then scatters.
    JOIN_ON_FINISH   committer copies GPU freeze → owned host tensors,
                     then scatters.

Async: hi/lo queues, then scatter.
Eager: submit() does wait+scatter inline (CPU tests / no CUDA).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import torch

from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
from vllm_omni.core.prefix_cache.interface import (
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    WriteSchedule,
)

logger = logging.getLogger(__name__)


class StagingBufferHolder(NamedTuple):
    """One holder of a D2H staging-buffer slot. The slot is free when none remain.

    Not a buffer state — concurrent owners share the same slot:
    - for_step: claimed at save, released when materialize/discard consumes the ctx
    - for_task: bound before WriteTask submit, released when that task completes
    """

    kind: Literal["step", "task"]
    owner_id: int

    @classmethod
    def for_step(cls, step_id: int) -> StagingBufferHolder:
        return cls("step", step_id)

    @classmethod
    def for_task(cls, tid: int) -> StagingBufferHolder:
        return cls("task", tid)


@dataclass
class _GpuFreezeAlloc:
    """One whole-step GPU freeze clone.

    Several per-request views share this storage. The GPU-byte cap is
    charged once and released when the last view is dropped.
    """

    nbytes: int
    holders: set[int] = field(default_factory=set)


@dataclass
class _Segment:
    """One contiguous save's rows: slots + per-key frozen tensors."""

    slots_cpu: torch.Tensor  # int64 flat row ids, in token order
    tensors: dict[str, torch.Tensor]  # frozen (GPU) or eager CPU tensors
    host: dict[str, torch.Tensor] = field(default_factory=dict)  # key -> host view of rows [0:n)
    # Shared whole-step clone this view hangs on (deferred). None on
    # the immediate path.
    gpu_alloc: _GpuFreezeAlloc | None = None


@dataclass
class WriteTask:
    """One write of (slot, key) rows for a single request.

    Identity: `tid` is the handle in the manager's (slot, key) tables.
    `req_id` + `write_n` mark whose write this is and the nth time that
    request opened a write. One write may cover several keys.

    Pipeline:

        queued / GPU-staged
            -> copy claimed (`d2h_claimed`)
            -> `host_ready`  (D2H complete; GPU freeze refs may drop)
            -> scatter
            -> `done`        (in the CPU mirror; `failed` instead on error)

    How `host_ready` is reached:
    - JOIN_NEXT_STEP: `seg.host` is a staging view hung at save.
      `_copy_task` only waits `step_d2h_event` (does not write host).
    - JOIN_ON_FINISH: committer D2H writes owned tensors into
      `seg.host`, then sets `host_ready`.

    `JOIN_NEXT_STEP` starts on the hi copy queue and is joined at the
    next save (`host_ready` only). `JOIN_ON_FINISH` stays on the lo
    queue, or unqueued (deferred `submit(queued=False)`), until finish
    or cap pressure `escalate`s it onto hi — once. Cap flush takes the
    unfinished task with the oldest `enqueued_time`.

    Concurrent readers/writers:
    - Staging readers (materialize clone, committer scatter) all wait
      the same `step_d2h_event` before touching the view.
    - Remount: a later task taking the same (slot, key) pushes those
      rows into `reassigned`; the old scatter omits them so the two
      writes stay disjoint (no join edge).
    - Append: `append_segment` loses if copy already claimed or `done`
      — caller opens a fresh task rather than mutating a closed one.
    - `lock` covers `reassigned` / `d2h_claimed` / `append_segment` /
      host↔freeze / `slot_to_row`. `host_ready` and `done` are their
      own events (`set_host_tensor` / `mark_host_ready` / `mark_failed` /
      `mark_done`). `scatter_rows` snapshots `reassigned`.
    """

    tid: int
    req_id: str
    write_n: int  # 1-based: nth write opened by this request
    schedule: WriteSchedule
    segments: list[_Segment]
    # Compute-stream D2D freeze completion. Copy/read streams must wait
    # this before touching `segments[].tensors`, or they can read the
    # next CUDA-graph static-buffer overwrite.
    freeze_event: object | None = None
    nbytes: int = 0  # GPU-staging cap accounting only; not a correctness signal
    # Promoted from the lazy queue onto the hi queue (finish / cap). Once.
    escalated: bool = False
    # Single claimer for the copy stage. Staging: wait `step_d2h_event`.
    # Deferred: committer writes `seg.host`.
    d2h_claimed: bool = False
    # Committer could not land the write; manager fail-fasts on next entry.
    failed: bool = False
    # Slots this write no longer owns (a newer write took them over).
    reassigned: dict[str, torch.Tensor] = field(default_factory=dict)
    # Per-write CPU event: this write's host rows are ready (`join_host_ready`).
    host_ready: threading.Event = field(default_factory=threading.Event)
    # Scatter into the CPU mirror has finished (strictly after host_ready).
    done: threading.Event = field(default_factory=threading.Event)
    # Guards reassigned / d2h_claimed / append / host↔freeze. Not host_ready or done.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # time.monotonic() at submit; cap flush picks the smallest of these.
    enqueued_time: float = field(default_factory=time.monotonic)
    # Immediate path: this write's view into the shared step D2H page.
    staging_slot: int | None = None
    # Immediate path: this step's CUDA D2H event (shared). None if deferred.
    step_d2h_event: object | None = None
    # slot -> (which segment, row in that segment's tensor). Built on demand
    # for multi-segment tasks; scatter uses slot, the tensor uses row.
    _slot_to_row: dict[int, tuple[int, int]] | None = None

    def add_reassigned(self, key: str, slots: torch.Tensor) -> None:
        with self.lock:
            prev = self.reassigned.get(key)
            self.reassigned[key] = slots.clone() if prev is None else torch.cat([prev, slots])

    def try_claim_d2h(self) -> bool:
        """Single claimer for the copy stage. True if this caller won."""
        with self.lock:
            if self.d2h_claimed:
                return False
            self.d2h_claimed = True
            return True

    def is_done(self) -> bool:
        return self.done.is_set()

    def is_host_ready(self) -> bool:
        return self.host_ready.is_set()

    def ready_to_scatter(self) -> bool:
        return self.is_host_ready() and not self.is_done()

    def append_segment(self, seg: _Segment, freeze_event: object | None = None) -> bool:
        """Grow this write with one step's rows. False if copy already claimed.

        `freeze_event` is stored in the same snapshot: events on one compute
        stream are ordered, so the newest also covers every earlier clone.
        """
        nbytes = sum(t.numel() * t.element_size() for t in seg.tensors.values())
        with self.lock:
            if self.d2h_claimed or self.is_done():
                return False
            self.segments.append(seg)
            self._slot_to_row = None
            if freeze_event is not None:
                self.freeze_event = freeze_event
            # Shared clones are charged on the alloc, not per-view.
            if seg.gpu_alloc is None:
                self.nbytes += nbytes
            return True

    def get_host_tensor(self, si: int, key: str) -> torch.Tensor | None:
        """`segments[si]` host if written, else freeze. One snapshot."""
        with self.lock:
            seg = self.segments[si]
            src = seg.host.get(key)
            if src is None:
                src = seg.tensors.get(key)
            return src

    def set_host_tensor(self, rows: list[tuple[_Segment, str, torch.Tensor]]) -> None:
        """Write these host tensors, drop the GPU freeze, set `host_ready`."""
        with self.lock:
            for seg, key, tensor in rows:
                seg.host[key] = tensor
            self._clear_tensor()
        self.host_ready.set()

    def mark_host_ready(self) -> None:
        """Host already written (staging views). Wait the step D2H, drop freeze."""
        if self.step_d2h_event is not None:
            self.step_d2h_event.synchronize()
        self.clear_tensor()
        self.host_ready.set()

    def mark_failed(self) -> None:
        """Unblock joiners. Host may be missing; manager fail-fasts."""
        self.failed = True
        self.clear_tensor()
        self.host_ready.set()
        self.done.set()

    def mark_done(self) -> None:
        self.done.set()

    def scatter_rows(self) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        """`(key, slots, host)` to write. Omits slots in `reassigned`."""
        with self.lock:
            reassigned = {k: s.clone() for k, s in self.reassigned.items()}
        out: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for seg in self.segments:
            for k, host in seg.host.items():
                taken = reassigned.get(k)
                if taken is not None and taken.numel():
                    keep = ~torch.isin(seg.slots_cpu, taken)
                    if not bool(keep.any()):
                        continue
                    out.append((k, seg.slots_cpu[keep], host[keep]))
                else:
                    out.append((k, seg.slots_cpu, host))
        return out

    def clear_tensor(self) -> None:
        """Drop the GPU freeze. Host is unchanged (staging wait / fail)."""
        with self.lock:
            self._clear_tensor()

    def _clear_tensor(self) -> None:
        for seg in self.segments:
            seg.tensors = {}

    def keys(self) -> set[str]:
        ks: set[str] = set()
        for s in self.segments:
            ks.update(s.tensors.keys())
        return ks

    def slot_to_row(self) -> dict[int, tuple[int, int]]:
        with self.lock:
            if self._slot_to_row is None:
                m: dict[int, tuple[int, int]] = {}
                for si, seg in enumerate(self.segments):
                    for ri, slot in enumerate(seg.slots_cpu.tolist()):
                        m[slot] = (si, ri)
                self._slot_to_row = m
            return self._slot_to_row


class StagingBufferPool:
    """Reusable pinned landing zone for ONE whole-step D2H at save.

    Per-task `seg.host` is a row-range view into a slot, so the committer
    skips per-task D2H. Slots recycle; this is not the CPU block pool.

    A slot stays busy while anyone still holds it: the step (until
    materialize/discard) and each immediate write that views the page
    (until that write is retired). Prefix hits do not hold a slot —
    they wait for scatter and read the durable pool.
    """

    def __init__(self, depth: int, capacity: int):
        self.depth = depth
        self.capacity = capacity  # rows per slot; a larger step fails fast
        self._bufs: dict[str, torch.Tensor] = {}  # key -> [depth*capacity, width]
        self._busy: list[set[StagingBufferHolder]] = [set() for _ in range(depth)]
        self._lock = threading.Lock()

    def _buf(self, key: str, width: int, dtype: torch.dtype, pin: bool) -> torch.Tensor:
        buf = self._bufs.get(key)
        if buf is None or buf.shape[-1] != width or buf.dtype != dtype:
            buf = torch.empty((self.depth * self.capacity, width), dtype=dtype, pin_memory=pin)
            self._bufs[key] = buf
        return buf

    def try_claim(self, holder: StagingBufferHolder) -> int | None:
        """Grab a free slot for `holder` (the step holder); None if all busy."""
        with self._lock:
            for slot in range(self.depth):
                if not self._busy[slot]:
                    self._busy[slot].add(holder)
                    return slot
        return None

    def bind(self, slot: int, holder: StagingBufferHolder) -> None:
        with self._lock:
            self._busy[slot].add(holder)

    def release(self, slot: int, holder: StagingBufferHolder) -> None:
        with self._lock:
            self._busy[slot].discard(holder)

    def views(self, slot: int, key: str, n: int, width: int, dtype: torch.dtype, pin: bool) -> torch.Tensor:
        base = slot * self.capacity
        return self._buf(key, width, dtype, pin)[base : base + n]


class OmniPrefixCacheController:
    """Staging pool + committer. Step D2H is launched at save; this
    thread waits that event (JOIN_NEXT_STEP) or copies deferred rows
    (JOIN_ON_FINISH), then scatters into the CPU pool.
    """

    def __init__(self, pool: PrefixBlockPool, config: PrefixCacheConfig, eager: bool | None = None):
        self._pool = pool
        self._config = config
        self._eager = (not torch.cuda.is_available()) if eager is None else eager
        self._tasks: dict[int, WriteTask] = {}
        self._completed: deque[int] = deque()  # scattered, awaiting manager drain
        self._failed: deque[int] = deque()  # write failed; manager must fail-fast
        self._staged_bytes = 0
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue_hi: deque[int] = deque()  # join-next-step + escalated
        self._queue_lo: deque[int] = deque()  # join-on-finish trickle
        self._blocked: list[int] = []  # copy done, awaiting scatter
        self._shutdown = False
        self._copy_stream: torch.cuda.Stream | None = None
        self._read_stream: torch.cuda.Stream | None = None
        self._worker: threading.Thread | None = None
        self._staging_pool = StagingBufferPool(config.staging_depth, config.staging_capacity_tokens)
        if not self._eager:
            self._copy_stream = torch.cuda.Stream()
            self._read_stream = torch.cuda.Stream()
            self._worker = threading.Thread(target=self._worker_loop, name="omni-prefix-cache-committer", daemon=True)
            self._worker.start()

    # --------------------------------------------------------- step D2H staging

    def _d2h_on_stream(
        self,
        stream: torch.cuda.Stream,
        freeze_event: object | None,
        copy: Callable[[], None],
    ) -> torch.cuda.Event:
        """Issue D2H on `stream` after freeze; return the done event."""
        with torch.cuda.stream(stream):
            if freeze_event is not None:
                stream.wait_event(freeze_event)
            copy()
            ev = torch.cuda.Event()
            ev.record()
        return ev

    def stage_step_host(
        self, tensors: dict[str, torch.Tensor], n: int, freeze_event: object | None, step_holder: StagingBufferHolder
    ) -> tuple[int, dict[str, torch.Tensor], object | None]:
        """Launch ONE whole-step D2H into a staging slot, ahead of consumption.

        Returns (slot, key -> host view of rows [0:n), d2h event). The
        caller only invokes this for a non-empty packed step. Too large a
        step, or no free slot, is a contract break — not a second D2H
        path. The caller binds task holders after submit; `step_holder`
        is released by materialize/discard via staging_release.
        """
        if not tensors or n <= 0:
            raise OmniPrefixCacheUnmatchError(
                f"stage_step_host called with n={n} keys={list(tensors)}; only a non-empty packed step may launch D2H"
            )
        if n > self._staging_pool.capacity:
            raise OmniPrefixCacheUnmatchError(
                f"step has {n} tokens; staging capacity is {self._staging_pool.capacity} "
                "(size staging_capacity_tokens to max_num_batched_tokens)"
            )
        slot = self._staging_pool.try_claim(step_holder)
        if slot is None:
            raise OmniPrefixCacheUnmatchError(
                "D2H staging pool exhausted; unconsumed steps, leaked holders, "
                f"or committer backlog (in_flight_tasks={len(self._tasks)})"
            )
        try:
            pin = not self._eager
            views: dict[str, torch.Tensor] = {}
            event: object | None = None
            if self._eager or all(t.device.type == "cpu" for t in tensors.values()):
                for key, src in tensors.items():
                    v = self._staging_pool.views(slot, key, n, int(src.shape[-1]), src.dtype, pin)
                    v.copy_(src)
                    views[key] = v
            else:
                assert self._copy_stream is not None

                def _copy_to_staging() -> None:
                    for key, src in tensors.items():
                        v = self._staging_pool.views(slot, key, n, int(src.shape[-1]), src.dtype, pin)
                        v.copy_(src, non_blocking=True)
                        views[key] = v

                event = self._d2h_on_stream(self._copy_stream, freeze_event, _copy_to_staging)
            return slot, views, event
        except Exception:
            self._staging_pool.release(slot, step_holder)
            raise

    def staging_bind(self, slot: int, holder: StagingBufferHolder) -> None:
        self._staging_pool.bind(slot, holder)

    def staging_release(self, slot: int, holder: StagingBufferHolder) -> None:
        self._staging_pool.release(slot, holder)

    # ------------------------------------------------------------------ submit

    def submit(self, task: WriteTask, queued: bool = True) -> None:
        """Register a task. queued=False (deferred tasks) stays GPU-staged
        until escalated or cap-flushed.

        Caller must reserve() the task bytes first (cap flush can block;
        the manager does that outside the state lock).
        """
        # Immediate (no gpu_alloc): charge/release via task.nbytes.
        # Deferred C→1 views: charge the shared alloc once at reserve().
        task.nbytes = sum(
            t.numel() * t.element_size() for s in task.segments if s.gpu_alloc is None for t in s.tensors.values()
        )
        task.enqueued_time = time.monotonic()
        with self._lock:
            self._tasks[task.tid] = task
        if self._eager:
            if queued:
                self._run_eager(task)
            return
        if queued:
            with self._wake:
                (self._queue_hi if task.schedule is WriteSchedule.JOIN_NEXT_STEP else self._queue_lo).append(task.tid)
                self._wake.notify_all()

    def append_segment(self, task: WriteTask, seg: _Segment, freeze_event: object | None = None) -> bool:
        return task.append_segment(seg, freeze_event)

    def pin_gpu_freeze(self, alloc: _GpuFreezeAlloc, tid: int) -> None:
        """Record that ``tid`` holds a view of this step's deferred clone."""
        with self._lock:
            alloc.holders.add(tid)

    def reserve(self, nbytes: int, exclude: set[int] | None = None) -> None:
        """Public cap reservation; blocking flush happens here, so callers
        must not hold the manager's state lock."""
        self._reserve_bytes(nbytes, exclude=exclude)

    def _release_staged_bytes(self, task: WriteTask) -> None:
        """Drop this task's GPU-freeze charge.

        Shared C→1 clones release only when the last holder tid drops.
        Immediate tasks (no ``gpu_alloc``) still release ``task.nbytes``.
        """
        allocs: list[_GpuFreezeAlloc] = []
        seen: set[int] = set()
        for seg in task.segments:
            alloc = seg.gpu_alloc
            if alloc is not None and id(alloc) not in seen:
                seen.add(id(alloc))
                allocs.append(alloc)
        with self._wake:
            if allocs:
                for alloc in allocs:
                    if task.tid in alloc.holders:
                        alloc.holders.discard(task.tid)
                        if not alloc.holders:
                            self._staged_bytes -= alloc.nbytes
            else:
                self._staged_bytes -= task.nbytes

    def _reserve_bytes(self, nbytes: int, exclude: set[int] | None = None) -> None:
        # Cap backpressure: force-flush oldest pending tasks until under
        # budget. Bounded block: their D2H has usually long completed.
        exclude = exclude or set()
        while True:
            with self._lock:
                pending = [tid for tid, t in self._tasks.items() if not t.is_done() and tid not in exclude]
                if self._staged_bytes + nbytes <= self._config.gpu_staging_bytes or not pending:
                    # Under budget or no pending tasks; admit reservation.
                    self._staged_bytes += nbytes
                    return
                oldest = min(pending, key=lambda tid: self._tasks[tid].enqueued_time)
            logger.warning("omni prefix cache: staging cap hit, force-flushing task %d", oldest)
            self.escalate([oldest])
            self.join([oldest])

    # ------------------------------------------------------------- lifecycle

    def escalate(self, tids: list[int]) -> None:
        if self._eager:
            for tid in tids:
                task = self._tasks.get(tid)
                if task is not None and not task.is_done():
                    self._run_eager(task)
            return
        with self._wake:
            for tid in tids:
                task = self._tasks.get(tid)
                if task is None or task.escalated or task.is_done():
                    continue
                task.escalated = True
                try:
                    self._queue_lo.remove(tid)
                    self._queue_hi.appendleft(tid)
                except ValueError:
                    # Not in the lazy queue: either an unqueued deferred
                    # task (queue it now) or already claimed/queued-hi.
                    if tid not in self._queue_hi and tid not in self._blocked and not task.d2h_claimed:
                        self._queue_hi.appendleft(tid)
            self._wake.notify_all()

    def join(self, tids: list[int]) -> None:
        """Block until each task has finished scatter (or failed)."""
        for tid in tids:
            task = self._tasks.get(tid)
            if task is not None:
                task.done.wait()

    def join_host_ready(self, tids: list[int]) -> None:
        """Block until each task's D2H is complete (`host_ready`).

        Staging: committer has waited `step_d2h_event`. Deferred: committer
        has written `seg.host`. Does not wait scatter.
        """
        for tid in tids:
            task = self._tasks.get(tid)
            if task is not None:
                task.host_ready.wait()

    def drain_completed(self) -> list[int]:
        """Pop scattered tasks from `_completed` and drop them from `_tasks`.

        WriteTask holders release HERE — the same locked drain that flips
        state to committed — not at scatter: a hit plan that still sees rows
        in-transit must be able to hold the slot before it is reclaimable.
        """
        out: list[int] = []
        with self._lock:
            while self._completed:
                out.append(self._completed.popleft())
            tasks = [self._tasks.pop(tid, None) for tid in out]
        for task in tasks:
            if task is not None and task.staging_slot is not None:
                self._staging_pool.release(task.staging_slot, StagingBufferHolder.for_task(task.tid))
        return out

    def drain_failed(self) -> list[int]:
        """Pop failed task ids from `_failed`. Does not drop `_tasks`."""
        out: list[int] = []
        with self._lock:
            while self._failed:
                out.append(self._failed.popleft())
        return out

    def get_task(self, tid: int) -> WriteTask | None:
        return self._tasks.get(tid)

    def shutdown(self) -> None:
        with self._wake:
            self._shutdown = True
            self._wake.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=5.0)

    # ------------------------------------------------------------ fetch_host

    @torch.inference_mode()
    def fetch_host(self, task: WriteTask, slots: torch.Tensor, key: str) -> torch.Tensor:
        """Rows for `slots` of one in-flight JOIN_ON_FINISH task.

        `_plan_rows` puts JOIN_NEXT_STEP owners in `join_tids` (join then
        pool). This path reads committer-written `seg.host`, or the GPU
        freeze if that D2H has not landed.
        """
        if task.step_d2h_event is not None:
            task.step_d2h_event.synchronize()
        return self._rows_from(task, slots, key)

    def _rows_from(self, task: WriteTask, slots: torch.Tensor, key: str) -> torch.Tensor:
        """Map `slots` to rows across one or more segments; preserve caller order."""
        s2r = task.slot_to_row()
        try:
            idx = [s2r[int(s)] for s in slots.tolist()]
        except KeyError:
            raise KeyError(f"slots not covered by task {task.tid}") from None
        parts: list[torch.Tensor] = []
        order: list[int] = []
        pos = 0
        seg_groups: dict[int, list[tuple[int, int]]] = {}
        for si, ri in idx:
            seg_groups.setdefault(si, []).append((pos, ri))
            pos += 1
        for si, items in seg_groups.items():
            src = task.get_host_tensor(si, key)
            if src is None:
                continue
            rows_idx = torch.tensor([ri for _, ri in items], dtype=torch.long)
            picked = self._slice_rows(task, src, rows_idx, host=(src.device.type == "cpu"))
            parts.append(picked)
            order.extend(p for p, _ in items)
        if not parts:
            raise KeyError(f"key {key} not present in task {task.tid}")
        cat = torch.cat(parts, dim=0)
        out = torch.empty_like(cat)
        out[torch.tensor(order, dtype=torch.long)] = cat
        return out

    def _slice_rows(self, task: WriteTask, src: torch.Tensor, rows_idx: torch.Tensor, host: bool) -> torch.Tensor:
        n = int(rows_idx.numel())
        # Ascending-run check without materializing an arange: endpoints plus
        # a monotonic diff are enough, and the common case is one long run.
        contiguous = (
            n > 0 and int(rows_idx[-1]) - int(rows_idx[0]) == n - 1 and (n < 2 or bool((rows_idx.diff() == 1).all()))
        )

        def _pick() -> torch.Tensor:
            return src[rows_idx[0] : rows_idx[0] + n] if contiguous else src.index_select(0, rows_idx)

        if host or src.device.type == "cpu":
            return _pick()
        if self._read_stream is None:
            return _pick().detach().cpu()
        cpu: torch.Tensor | None = None

        def _copy_to_cpu() -> None:
            nonlocal cpu
            cpu = _pick().to("cpu", non_blocking=True)

        ev = self._d2h_on_stream(self._read_stream, task.freeze_event, _copy_to_cpu)
        ev.synchronize()
        assert cpu is not None
        return cpu

    # ------------------------------------------------------------ eager mode

    @torch.inference_mode()
    def _run_eager(self, task: WriteTask) -> None:
        if not task.try_claim_d2h():
            if task.ready_to_scatter():
                self._scatter(task)
            return
        if task.schedule is WriteSchedule.JOIN_NEXT_STEP:
            # Host is already a staging view (copied at save). Drop freeze.
            # `step_d2h_event` is None on CPU; `mark_host_ready` skips wait.
            task.mark_host_ready()
        else:
            # JOIN_ON_FINISH: no copy stream; freeze → owned host inline.
            task.set_host_tensor(
                [
                    (seg, k, t.detach().cpu() if t.device.type != "cpu" else t.clone())
                    for seg in task.segments
                    for k, t in seg.tensors.items()
                ]
            )
        self._release_staged_bytes(task)
        self._scatter(task)

    # ---------------------------------------------------------- worker loop

    def _worker_loop(self) -> None:
        # A dying committer would strand every join() forever, so the loop
        # never propagates: it fails the offending task and keeps serving.
        while True:
            tid = None
            try:
                with self._wake:
                    while not self._shutdown and not self._queue_hi and not self._queue_lo:
                        if self._blocked:
                            break
                        # submit / escalate / shutdown all notify.
                        self._wake.wait()
                    if self._shutdown and not self._queue_hi and not self._queue_lo and not self._blocked:
                        return
                    if self._queue_hi:
                        tid = self._queue_hi.popleft()
                    elif self._queue_lo:
                        tid = self._queue_lo.popleft()
                if tid is not None:
                    task = self._tasks.get(tid)
                    if task is not None:
                        self._copy_task(task)
                        with self._wake:
                            if tid not in self._blocked:
                                self._blocked.append(tid)
                self._scatter_host_ready()
            except BaseException:
                logger.exception("omni prefix cache committer failed on task %s; releasing waiters", tid)
                self._fail_task(tid)

    @torch.inference_mode()
    def _copy_task(self, task: WriteTask) -> None:
        """Reach `host_ready`. Staging: wait the save-time D2H event.
        Deferred: this is the D2H into owned `seg.host` tensors.
        """
        if not task.try_claim_d2h():
            return
        if task.schedule is WriteSchedule.JOIN_NEXT_STEP:
            # `seg.host` is already a staging view; D2H flew at save.
            # `mark_host_ready` waits `step_d2h_event` if one was recorded.
            # No per-task copy. Scatter is `_scatter_host_ready`.
            task.mark_host_ready()
            self._release_staged_bytes(task)
            return
        assert self._copy_stream is not None
        chunk_bytes = self._config.copy_chunk_bytes
        pending_host: list[tuple[_Segment, str, torch.Tensor]] = []
        pending_cats: list[tuple[_Segment, str, list[torch.Tensor]]] = []
        with torch.cuda.stream(self._copy_stream):
            if task.freeze_event is not None:
                self._copy_stream.wait_event(task.freeze_event)
            for seg in task.segments:
                for k, src in seg.tensors.items():
                    if src.numel() * src.element_size() > chunk_bytes:
                        rows_per_chunk = max(1, chunk_bytes // max(1, src.shape[-1] * src.element_size()))
                        parts = [
                            src[start : start + rows_per_chunk].to("cpu", non_blocking=True)
                            for start in range(0, src.shape[0], rows_per_chunk)
                        ]
                        pending_cats.append((seg, k, parts))
                    else:
                        pending_host.append((seg, k, src.to("cpu", non_blocking=True)))
            ev = torch.cuda.Event()
            ev.record()
        ev.synchronize()
        task.set_host_tensor([(seg, k, torch.cat(parts, dim=0)) for seg, k, parts in pending_cats] + pending_host)
        self._release_staged_bytes(task)

    def _fail_task(self, tid: int | None) -> None:
        """Release waiters for a task the committer could not complete.

        Idempotent, and only releases bytes the copy stage has not already
        released: a raise AFTER a successful _copy_task (host_ready set)
        must not subtract this task's bytes a second time.
        """
        task = self._tasks.get(tid) if tid is not None else None
        if task is None or task.is_done():
            return
        if not task.is_host_ready():
            self._release_staged_bytes(task)
        with self._wake:
            if tid in self._blocked:
                self._blocked.remove(tid)
        task.mark_failed()
        if task.staging_slot is not None:
            self._staging_pool.release(task.staging_slot, StagingBufferHolder.for_task(task.tid))
        with self._lock:
            # Publish the failure: rows behind already-published block hashes
            # never landed, which the manager must turn into a fail-fast (a
            # silently poisoned span would crash on every future hit instead).
            self._failed.append(task.tid)

    @torch.inference_mode()
    def _scatter_host_ready(self) -> None:
        """Scatter `_blocked` tasks whose `host_ready` is set."""
        with self._wake:
            ready = [tid for tid in self._blocked if (t := self._tasks.get(tid)) and t.is_host_ready()]
            for tid in ready:
                self._blocked.remove(tid)
        for tid in ready:
            task = self._tasks.get(tid)
            if task is None:
                continue
            try:
                self._scatter(task)
            except BaseException:
                # Attribute the failure to THIS task: letting it propagate
                # would fail whichever entry the worker loop happened to be
                # copying, double-release its bytes, and strand this one's
                # join() forever.
                logger.exception("omni prefix cache scatter failed on task %s; releasing waiters", tid)
                self._fail_task(tid)

    @torch.inference_mode()
    def _scatter(self, task: WriteTask) -> None:
        for key, slots, host in task.scatter_rows():
            self._pool.write(key, slots, host)
        task.mark_done()
        with self._lock:
            self._completed.append(task.tid)
