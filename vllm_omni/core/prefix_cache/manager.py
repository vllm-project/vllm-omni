# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni prefix cache, manager side.

Owns (slot, key) occupancy, the hit/span registry, per-step snapshots,
and the merge. The controller owns the staging pool, copy queues, and
pool scatter. The state lock covers those tables only — never a join,
a cap flush, or a copy.

Two host stores:

    StagingBufferPool   reusable step-sized pages. save launches ONE
                        whole-step D2H here for immediately-cached keys
                        (hidden + non-deferred mm). Per-req `seg.host`
                        is a view into that page, not a second copy.
    PrefixBlockPool     durable (kv_slot, key) prefix cache. The
                        committer only scatters into it.

Two write paths (schedule is the key split, not token count):

    JOIN_NEXT_STEP      immediately-cached keys. D2H is already in
                        flight at submit; the committer waits
                        `step_d2h_event` then H2H-scatters. Joined at the
                        next save (`host_ready` only).
    JOIN_ON_FINISH      deferred mm. Stays on the GPU freeze; the
                        committer does that D2H, then scatters.
                        Escalated on finish/abort or cap pressure.

Per real scheduler_output, engine-thread order:

    new_step_starts   before _update_states drops finished requests
                      (register hits, start prefix prefetch)
    forward
    save_outputs      D2D freeze + launch staging D2H; returns step id
    materialize XOR discard_step   consume that id exactly once

materialize may run on the async output builder while the engine is
already in the next step. Warmup/dummy runs are never fed.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.core.prefix_cache.block_pool import PrefixBlockPool
from vllm_omni.core.prefix_cache.controller import (
    OmniPrefixCacheController,
    StagingBufferHolder,
    WriteTask,
    _GpuFreezeAlloc,
    _Segment,
)
from vllm_omni.core.prefix_cache.interface import (
    HIDDEN_KEY,
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
    StageCacheOutputs,
    WriteSchedule,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_omni.core.prefix_cache.group_view import FullAttentionGroupView

logger = logging.getLogger(__name__)


class _Occupancy(IntEnum):
    ABSENT = 0
    IN_TRANSIT = 1
    COMMITTED = 2


_ABSENT, _IN_TRANSIT, _COMMITTED = (
    _Occupancy.ABSENT,
    _Occupancy.IN_TRANSIT,
    _Occupancy.COMMITTED,
)


class MmValueKind(Enum):
    """How one ``mm_flat`` value relates to this step's scheduled tokens.

    Shared by freeze, deferred, and leftover so the three sites cannot
    drift onto different type/shape predicates.
    """

    # 2D+ tensor whose first dim is the unpadded *or* CUDA-graph padded length.
    TOKEN_MAJOR = "token_major"
    # Per-request list/tuple (Higgs ``codes.audio``). Not a pool key.
    REQ_LIST = "req_list"
    # Everything else: ``codes.ref``, 1D meta, dicts, scalars.
    PASSTHROUGH = "passthrough"


def classify_mm_value(
    val: Any,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
) -> MmValueKind:
    if isinstance(val, (list, tuple)):
        return MmValueKind.REQ_LIST
    if isinstance(val, torch.Tensor) and val.ndim >= 2:
        rows = int(val.shape[0])
        if rows == num_tokens_unpadded or rows == num_tokens_padded:
            return MmValueKind.TOKEN_MAJOR
    return MmValueKind.PASSTHROUGH


def _snapshot_leftover_mm_cpu(
    mm_flat: dict[str, Any],
    frozen_keys: set[str],
    num_tokens_unpadded: int,
    num_tokens_padded: int | None = None,
) -> dict[str, Any]:
    """Copy leftover mm onto CPU.

    Keys already in ``frozen_keys`` go through staging D2H. Everything else
    (deferred tails + uncached passthrough) is copied here so materialize
    never reads live graph buffers. Only ``TOKEN_MAJOR`` tensors whose
    first dim equals ``num_tokens_unpadded`` are sliced; ``>= n`` would
    truncate leftover tensors whose first dim is unrelated (``codes.ref``).
    """
    n = num_tokens_unpadded
    padded = n if num_tokens_padded is None else int(num_tokens_padded)

    def _copy(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            kind = classify_mm_value(val, n, padded)
            t = val[:n] if kind is MmValueKind.TOKEN_MAJOR and int(val.shape[0]) == n else val
            copied = t.detach()
            # .cpu() copies device tensors; CPU/pinned views still share storage.
            copied = copied.cpu() if copied.device.type != "cpu" else copied.clone()
            return copied.contiguous()
        if isinstance(val, Mapping):
            return {k: _copy(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_copy(v) for v in val]
        if isinstance(val, tuple):
            return tuple(_copy(v) for v in val)
        return val

    return {key: _copy(val) for key, val in mm_flat.items() if key not in frozen_keys and key != HIDDEN_KEY}


def _locked(fn):
    """Serialize facade entry points: the async output builder calls
    materialize() while the engine thread is in the next step."""

    def wrapper(self, *args, **kwargs):
        with self._state_lock:
            return fn(self, *args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@dataclass
class _RowSource:
    """Row sources resolved under the state lock, fetched outside it.

    Schedule split, not a single read tier:
    - JOIN_NEXT_STEP in-transit: ``join_tids``. Fetch waits ``done``,
      drains, then reads the pool. Staging views are never sliced.
    - JOIN_ON_FINISH in-transit: ``staged_list`` task refs. Fetch uses
      ``fetch_host`` (GPU freeze / committer host).
    - Already scattered: ``already_staged`` → pool.

    A JOIN_NEXT_STEP task may disappear between plan and join (another
    facade already drained it); ``join`` no-ops and the pool rows persist.
    """

    slots: torch.Tensor
    key: str
    req_id: str
    already_staged: bool  # this key is already in the CPU pool
    staged_list: list[tuple[WriteTask, torch.Tensor]]  # JOIN_ON_FINISH only
    join_tids: list[int] = field(default_factory=list)  # JOIN_NEXT_STEP in-transit


@dataclass
class _StepD2HClaim:
    """This step's claim on D2H staging. None if the step had no rows."""

    slot: int
    views: dict[str, torch.Tensor]  # key -> host rows [0:n)
    event: Any = None  # torch.cuda.Event; None on eager/CPU


@dataclass(kw_only=True)
class _StepContext:
    """Save-time snapshot of one step; consumed exactly once.

    Built on the engine thread in save_outputs so materialize (possibly on
    the async builder) never reads the live batch. Host rows live on `d2h`
    (None only if the step had no rows). materialize XOR discard_step pops
    it; leaking contexts fails fast at a later save.
    """

    # Packed layout in batch order: req_id -> [start, end) of this step's rows.
    spans: dict[str, tuple[int, int]]
    num_tokens_unpadded: int = 0

    # Hits snapshotted at new_step_starts. Prefetch fills [hit | empty tail]
    # during forward; materialize writes the tail.
    hits: dict[str, tuple[int, list[int] | None]]  # req_id -> (hit_upto, blocks)
    hit_prefetch: dict[str, dict[str, Future]] = field(default_factory=dict)

    # Key split frozen at save: recompute at materialize races ensure_key.
    cached_keys: set[str] = field(default_factory=set)
    # Leftover mm copied to CPU at save (deferred tails + uncached
    # passthrough). Never live graph-buffer refs: materialize may run
    # after the next forward has overwritten those buffers.
    mm_cpu_snapshot: dict[str, Any] = field(default_factory=dict)

    # Staging claim if this step had rows.
    d2h: _StepD2HClaim | None = None


class OmniPrefixCacheManager:
    def __init__(
        self,
        config: PrefixCacheConfig,
        view: FullAttentionGroupView,
        *,
        eager: bool | None = None,
    ):
        self._config = config
        self._view = view
        self._pool = PrefixBlockPool(config)
        self._controller = OmniPrefixCacheController(self._pool, config, eager=eager)
        self._policy = ModelCachePolicy()
        # Serializes engine vs async-builder facade entries. Non-reentrant:
        # those entries never call each other, and the lock must not cover
        # a join, cap flush, or D2H.
        self._state_lock = threading.Lock()

        # (slot, key) occupancy, lazily allocated per key. Hidden and a
        # deferred mm key on the same slot can be in different states.
        self._num_slots = config.num_blocks * config.block_size
        self._key_state: dict[str, torch.Tensor] = {}  # key -> int8[num_slots]
        self._key_owner: dict[str, torch.Tensor] = {}  # key -> int64[num_slots]

        # WriteTask identity and who still owes a host join.
        self._next_tid = 1
        self._write_n: dict[str, int] = {}  # req_id -> last write_n issued
        self._task_slots: dict[int, torch.Tensor] = {}
        self._task_keys: dict[int, tuple[str, ...]] = {}
        self._req_tasks: dict[str, set[int]] = {}  # req_id -> live tids
        self._deferred_tasks: dict[str, WriteTask] = {}
        self._join_next_step_tids: list[int] = []
        self._finished_join: set[int] = set()  # escalated on finish/abort

        # Between new_step_starts and save: live batch + this step's hits.
        self._live_reqs: set[str] = set()
        self._cur_num_scheduled: dict[str, int] = {}
        self._hit_spans: dict[str, tuple[int, list[int]]] = {}  # req -> (upto, blocks)
        self._hit_prefetch: dict[str, dict[str, Future]] = {}  # req -> key -> gather Future

        # Prefix gather during forward (CPU work releases the GIL).
        self._prefetch_pool = ThreadPoolExecutor(1, thread_name_prefix="omni-prefix-cache-prefetch")
        self._prefetch_jobs: list[tuple[Future, _RowSource]] = []

        # Consume-exactly-once snapshots (materialize XOR discard_step).
        self._next_step_id = 1
        self._step_ctxs: dict[int, _StepContext] = {}

    # ------------------------------------------------------------- facade

    def register_policy(self, policy: ModelCachePolicy) -> None:
        self._policy = policy

    @_locked
    @torch.inference_mode()
    def new_step_starts(self, scheduler_output: SchedulerOutput) -> None:
        """Consume one scheduler_output (lifecycle stream).

        Engine thread only; before _update_states removes finished
        requests; exactly once per real step. Registers new-request prefix
        hits (snapshotting their block tables) and escalates the writes of
        finished/aborted requests — a block hash that entered the batch
        must land in the cache, abort included.
        """
        # 1. Publish writes the committer has already scattered.
        self._commit_drained_writes()

        # 2. Finished/aborted reqs: escalate leftover writes (no rollback) and
        #    join their host copy at the next save.
        finished = getattr(scheduler_output, "finished_req_ids", None) or ()
        for req_id in finished:
            self._live_reqs.discard(req_id)
            self._write_n.pop(req_id, None)
            eids = self._req_tasks.pop(req_id, set())
            dtask = self._deferred_tasks.pop(req_id, None)
            if dtask is not None:
                eids.add(dtask.tid)
            pending = [e for e in eids if self._controller.get_task(e) is not None]
            if pending:
                # Abort included: complete the writes (never roll back) so
                # still-hashed blocks stay servable.
                self._controller.escalate(pending)
                self._finished_join.update(pending)

        # 3. Snapshot prefix hits for brand-new reqs (block table dies in
        #    _update_states / the next batch; materialize cannot reread it).
        self._hit_spans.clear()
        self._hit_prefetch.clear()
        for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_id = new_req.req_id
            if req_id in self._live_reqs:
                # Streaming continuation (async_chunk): parity with legacy —
                # no hit marking; span/delivered_upto refinement is Phase 2.
                continue
            self._live_reqs.add(req_id)
            num_computed = int(getattr(new_req, "num_computed_tokens", 0) or 0)
            if num_computed > 0:
                # Snapshot the hit block table now: materialize must not read
                # the live input_batch (it may have advanced under the async
                # builder). block_ids is per-kv-group; group 0 only.
                blocks = getattr(new_req, "block_ids", None)
                if blocks is not None and len(blocks) > 0 and not isinstance(blocks[0], int):
                    blocks = blocks[0]
                if not blocks:
                    # Fail at the cause: a hit we cannot snapshot now would
                    # crash at materialize time with less context (materialize is
                    # forbidden from reading the live batch).
                    raise OmniPrefixCacheUnmatchError(
                        f"prefix hit for req {req_id} ({num_computed} tokens) carries no block_ids"
                    )
                hit_blocks = list(blocks[: num_computed // self._config.block_size])
                self._hit_spans[req_id] = (num_computed, hit_blocks)

        # 4. Gather those spans on the prefetch thread; overlaps this forward.
        self._prefetch_jobs = [(f, s) for f, s in self._prefetch_jobs if not f.done()]
        self._cur_num_scheduled = dict(scheduler_output.num_scheduled_tokens)
        if self._hit_spans:
            self._prefetch_hit_spans()

    @torch.inference_mode()
    def save_outputs(
        self,
        hidden_states: torch.Tensor | None,
        mm_outputs: dict[str, Any] | None,
        *,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> int:
        """Write this step's outputs into the cache; returns the step id.

        Engine thread only, after the forward and before materialize.
        Immediately-cached rows: one D2D freeze, one whole-step D2H into
        the staging pool, then one JOIN_NEXT_STEP WriteTask per request
        whose `seg.host` is a view of that page. Deferred rows stay on
        the GPU freeze (JOIN_ON_FINISH); the committer copies them later.
        Leftover mm (deferred tails, uncached passthrough) is copied to
        CPU here so materialize never reads live graph buffers.
        Snapshots everything materialize needs. The returned step id MUST
        be consumed exactly once — by materialize() or discard_step();
        leaking contexts fails fast at a later save.

        The state lock never covers a blocking wait: the previous step's
        JOIN_NEXT_STEP join, the clone build, and the cap reservation (which may
        flush) all run unlocked.
        """
        # 1. Join the previous step's host copies (unlocked).
        with self._state_lock:
            join_ids = list(self._finished_join)
            join_ids.extend(self._join_next_step_tids)
            self._finished_join.clear()
            self._join_next_step_tids = []
        if join_ids:
            self._controller.join_host_ready(join_ids)

        # 2. Packed batch layout for this step (req -> [start, end)).
        req_order = self._view.batch_req_ids()
        num_sched = {r: int(self._cur_num_scheduled.get(r, 0)) for r in req_order}
        query_start: dict[str, int] = {}
        current_start_idx = 0
        for req_id in req_order:
            query_start[req_id] = current_start_idx
            current_start_idx += num_sched[req_id]

        slots_cpu: torch.Tensor | None = None
        mm_flat = mm_outputs or {}
        frozen_rows: dict[str, torch.Tensor] = {}
        deferred_segs: list[tuple[str, _Segment]] = []
        freeze_event = None

        # 3. Freeze this step's rows (D2D) and reserve GPU staging bytes.
        if num_tokens_unpadded > 0:
            # Derive the slot mapping on CPU: reading the device one back
            # would need a stream sync that waits on the whole forward.
            slots_cpu = self._view.step_slots_cpu(req_order, num_sched)
            if int(slots_cpu.numel()) != num_tokens_unpadded:
                # Fail at the cause: skipping the save would leave rows absent
                # behind hashes vLLM already published — a delayed crash at
                # some future hit instead of a debuggable one here.
                raise OmniPrefixCacheUnmatchError(
                    f"slot mapping covers {int(slots_cpu.numel())} of {num_tokens_unpadded} scheduled tokens; "
                    "CPU-side slot derivation out of sync with the batch"
                )
            frozen_rows = self._freeze_step_rows(hidden_states, mm_flat, num_tokens_unpadded, num_tokens_padded)
            deferred_segs = self._build_deferred_segments(
                mm_flat,
                slots_cpu,
                req_order,
                num_sched,
                query_start,
                num_tokens_unpadded,
                num_tokens_padded,
            )

            staged = [t for t in frozen_rows.values()] + [t for _, seg in deferred_segs for t in seg.tensors.values()]
            if staged:
                if torch.cuda.is_available() and any(t.is_cuda for t in staged):
                    freeze_event = torch.cuda.Event()
                    freeze_event.record()
                # Charge unique allocations: immediate clones + one deferred
                # C→1 clone. Do not sum per-req views — they share storage.
                immediate_bytes = sum(t.numel() * t.element_size() for t in frozen_rows.values())
                deferred_alloc = deferred_segs[0][1].gpu_alloc if deferred_segs else None
                deferred_bytes = deferred_alloc.nbytes if deferred_alloc is not None else 0
                # Cap reservation may block on a flush: outside the lock. The
                # flush must not close the deferred entries we are about to
                # append to (main-thread-only reads, safe unlocked).
                exclude = {self._deferred_tasks[r].tid for r, _ in deferred_segs if r in self._deferred_tasks}
                self._controller.reserve(immediate_bytes + deferred_bytes, exclude=exclude)

        # 4. Leftover mm D2H (unlocked). Keys already in frozen_rows go
        #    through staging; deferred tails and uncached passthrough must
        #    be on CPU before the next forward can overwrite graph buffers.
        leftover_mm = _snapshot_leftover_mm_cpu(mm_flat, set(frozen_rows), num_tokens_unpadded, num_tokens_padded)

        # 5. Fail-fast if the runner leaked prior step contexts *before*
        #    claiming a staging slot — otherwise a full pool raises a
        #    holder-exhaustion error and hides the consume-once ids.
        #    Then optional D2H (unlocked), then submit + hang ctx (locked).

        self._raise_if_unconsumed_ctxs_at_capacity()
        staging_slot: int | None = None
        host_views: dict[str, torch.Tensor] | None = None
        step_d2h_event = None
        step_holder = StagingBufferHolder.for_step(self._next_step_id)
        transferred = False
        bound_tids: list[int] = []
        try:
            if frozen_rows:
                staging_slot, host_views, step_d2h_event = self._controller.stage_step_host(
                    frozen_rows, num_tokens_unpadded, freeze_event, step_holder
                )

            with self._state_lock:
                self._commit_drained_writes()
                d2h = None
                if frozen_rows:
                    assert slots_cpu is not None and host_views is not None and staging_slot is not None
                    self._submit_step_writes(
                        req_order,
                        query_start,
                        num_sched,
                        frozen_rows,
                        slots_cpu,
                        host_views,
                        freeze_event,
                        staging_slot,
                        step_d2h_event,
                        bound_tids,
                    )
                    d2h = _StepD2HClaim(slot=staging_slot, views=host_views, event=step_d2h_event)

                self._stage_deferred(deferred_segs, freeze_event)

                step_id = self._next_step_id
                self._next_step_id += 1
                self._step_ctxs[step_id] = _StepContext(
                    spans={r: (query_start[r], query_start[r] + num_sched[r]) for r in req_order},
                    num_tokens_unpadded=num_tokens_unpadded,
                    hits=dict(self._hit_spans),
                    hit_prefetch=dict(self._hit_prefetch),
                    cached_keys=(self._pool.keys() - {HIDDEN_KEY}) & set(mm_flat.keys()),
                    mm_cpu_snapshot=leftover_mm,
                    d2h=d2h,
                )
                self._hit_spans.clear()
                self._hit_prefetch.clear()
                transferred = True
                return step_id
        finally:
            # Slot claim is outside the lock; a later raise must drop holders.
            if not transferred and staging_slot is not None:
                self._controller.staging_release(staging_slot, step_holder)
                for tid in bound_tids:
                    self._controller.staging_release(staging_slot, StagingBufferHolder.for_task(tid))

    @torch.inference_mode()
    def materialize(self, step_id: int, req_ids: list[str]) -> StageCacheOutputs:
        """Per-request merged outputs for the step saved as `step_id`.

        Any thread. `req_ids` must be (a subset of) the save-time snapshot;
        an outside id means the caller is reading the live batch (debug
        assert). A request without a hit is a plain miss and gets exactly
        this step's rows — normal path, nothing logged. A hit span that
        resolves to absent rows raises OmniPrefixCacheUnmatchError: fatal
        by contract, never a degrade.

        Two phases: under the lock, drain completions and pin every row
        source (task refs + masks, absent checks included) — the storage
        tier is NOT baked in. Unlocked: wait this step's `step_d2h_event`,
        clone the staging views (then drop the step holder), and merge.
        The engine thread never waits on this thread's PCIe.
        """
        ctx = None
        step_released = False
        try:
            with self._state_lock:
                ctx = self._take_step_ctx(step_id)
                self._commit_drained_writes()

                # The builder must pass (a subset of) the req list captured at
                # save time — an id outside the snapshot means it is reading the
                # live batch, which the contract forbids (debug assert, not a
                # degrade path).
                assert set(req_ids) <= set(ctx.spans), (
                    f"materialize(step {step_id}) got req ids outside the save snapshot: "
                    f"{sorted(set(req_ids) - set(ctx.spans))[:8]}"
                )

                if not self._policy.needs_full_hidden_states and not ctx.hits:
                    # Nothing will read the views; drop the step holder here
                    # or the slot leaks (consume-exactly-once ends with us).
                    self._release_step_staging_buffer(ctx, step_id)
                    step_released = True
                    return StageCacheOutputs(hidden_states=None, mm_outputs={})

                want_hidden = self._policy.needs_full_hidden_states
                cached_keys = ctx.cached_keys

                hit_sources: dict[tuple[str, str], _RowSource | Future] = {}
                try:
                    for req_id in req_ids:
                        hit = ctx.hits.get(req_id)
                        if not hit:
                            continue
                        hit_upto, hit_blocks = hit
                        prefetched = ctx.hit_prefetch.get(req_id, {})
                        keys = ([HIDDEN_KEY] if want_hidden else []) + sorted(cached_keys)
                        for key in keys:
                            fut = prefetched.get(key)
                            if fut is not None:
                                hit_sources[(req_id, key)] = fut
                                continue
                            hit_sources[(req_id, key)] = self._plan_hit_rows(
                                req_id, hit_upto, hit_blocks, key, strict=(key == HIDDEN_KEY)
                            )
                except Exception:
                    logger.critical("omni prefix cache unmatch during materialize", exc_info=True)
                    raise

            # ---- unlocked: data movement + merge ----
            current: dict[str, torch.Tensor] = {}
            if ctx.d2h is not None:
                # Whole-step D2H was launched at save. One event wait (usually
                # already complete), then a contiguous copy-out per key — the
                # copy detaches consumers from the reusable staging slot.
                if ctx.d2h.event is not None:
                    ctx.d2h.event.synchronize()
                current = {k: v.clone() for k, v in ctx.d2h.views.items()}
                self._release_step_staging_buffer(ctx, step_id)
                step_released = True

            hidden_out: dict[str, torch.Tensor] | None = None
            if want_hidden and HIDDEN_KEY in current:
                hidden_out = {}
                for req_id in req_ids:
                    hidden_out[req_id] = self._merge_cached_for_req(
                        ctx, req_id, HIDDEN_KEY, current[HIDDEN_KEY], hit_sources
                    )

            mm_out: dict[str, dict[str, Any]] = {}
            for key in cached_keys:
                cur = current.get(key)
                if cur is None:
                    val = ctx.mm_cpu_snapshot.get(key)
                    if not isinstance(val, torch.Tensor):
                        continue
                    # Leftover snapshot already classified; do not re-slice.
                    cur = val
                mm_out[key] = {
                    req_id: self._merge_cached_for_req(ctx, req_id, key, cur, hit_sources) for req_id in req_ids
                }

            self._merge_uncached_mm(ctx, req_ids, cached_keys, mm_out)
            return StageCacheOutputs(hidden_states=hidden_out, mm_outputs=mm_out)
        finally:
            if ctx is not None and not step_released:
                self._release_step_staging_buffer(ctx, step_id)

    @_locked
    def discard_step(self, step_id: int) -> None:
        """Consume the step context when nothing will materialize it.

        Any thread; same exactly-once contract as materialize (unknown or
        duplicate id fails fast). Only the read-side snapshot is dropped —
        the cache write proceeds unchanged.
        """
        ctx = self._take_step_ctx(step_id)
        self._release_step_staging_buffer(ctx, step_id)

    def shutdown(self) -> None:
        self._prefetch_pool.shutdown(wait=False, cancel_futures=True)
        self._prefetch_jobs.clear()
        self._controller.shutdown()

    # ------------------------------------------------------ new_step

    def _prefetch_hit_spans(self) -> None:
        """Plan each hit span now (we hold the state lock) and gather it on
        the prefetch thread, overlapping the forward. A span that fails to
        plan — same-step hits resolve rows this step's save has not
        registered yet — is left to materialize, which owns the fail-fast.
        """
        keys = sorted(self._pool.keys() - {HIDDEN_KEY})
        if self._policy.needs_full_hidden_states:
            keys = [HIDDEN_KEY, *keys]
        for req_id, (hit_upto, hit_blocks) in self._hit_spans.items():
            n_new = int(self._cur_num_scheduled.get(req_id, 0))
            futs: dict[str, Future] = {}
            for key in keys:
                try:
                    src = self._plan_hit_rows(req_id, hit_upto, hit_blocks, key, strict=(key == HIDDEN_KEY))
                except Exception:
                    continue
                fut = self._prefetch_pool.submit(self._prefetch_hit, src, n_new)
                self._prefetch_jobs.append((fut, src))
                futs[key] = fut
            if futs:
                self._hit_prefetch[req_id] = futs

    @torch.inference_mode()
    def _prefetch_hit(self, src: _RowSource, n_new: int) -> torch.Tensor:
        """Prefetch thread: gather the hit span and pre-build the merged
        buffer with the prefix filled. materialize writes only this step's
        rows at the tail — the gather AND the prefix copy both happen while
        the forward runs, and the cat leaves the critical path."""
        rows = self._fetch_source(src)
        out = torch.empty((rows.shape[0] + n_new, rows.shape[-1]), dtype=rows.dtype)
        out[: rows.shape[0]] = rows
        return out

    # ---------------------------------------------------------- save

    def _freeze_step_rows(
        self,
        hidden_states: torch.Tensor | None,
        mm_flat: dict[str, Any],
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> dict[str, torch.Tensor]:
        """D2D-clone this step's immediately-cached rows.

        Deferred keys stay on the GPU freeze (``_build_deferred_segments``).
        Immediate mm is ``TOKEN_MAJOR`` only: first dim is unpadded *or*
        CUDA-graph padded. Talker ``codes.audio`` is a cat of scheduled
        rows and stays unpadded while hidden is padded; both must open a
        pool key. Anything else is leftover passthrough.
        """
        n = num_tokens_unpadded
        out: dict[str, torch.Tensor] = {}
        if hidden_states is not None and self._policy.needs_full_hidden_states:
            if hidden_states.ndim < 2 or hidden_states.shape[0] < n:
                rows = 0 if hidden_states.ndim < 2 else int(hidden_states.shape[0])
                raise OmniPrefixCacheUnmatchError(f"hidden_states has {rows} rows, need {n}")
            self._pool.ensure_key(HIDDEN_KEY, hidden_states.dtype, int(hidden_states.shape[-1]))
            out[HIDDEN_KEY] = hidden_states[:n].clone()

        reserved = self._policy.deferred_keys | {HIDDEN_KEY}
        for key, val in mm_flat.items():
            if key in reserved:
                continue
            if classify_mm_value(val, n, num_tokens_padded) is not MmValueKind.TOKEN_MAJOR:
                continue
            self._pool.ensure_key(key, val.dtype, int(val.shape[-1]))
            out[key] = val[:n].clone()
        return out

    def _build_deferred_segments(
        self,
        mm_flat: dict[str, Any],
        slots_cpu: torch.Tensor,
        req_order: list[str],
        num_sched: dict[str, int],
        query_start: dict[str, int],
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> list[tuple[str, _Segment]]:
        """Clone this step's deferred rows (build phase, no lock held).

        One whole-step D2D per token-major key, then per-req views — same
        pattern as the immediate freeze. List-valued deferred keys
        (Higgs ``codes.audio``) are ``REQ_LIST`` and stay leftover.
        """
        n = num_tokens_unpadded
        step_rows: dict[str, torch.Tensor] = {}
        for key in self._policy.deferred_keys:
            val = mm_flat.get(key)
            if classify_mm_value(val, n, num_tokens_padded) is not MmValueKind.TOKEN_MAJOR:
                continue
            if not self._pool.has_key(key):
                self._pool.ensure_key(key, val.dtype, int(val.shape[-1]))
            step_rows[key] = val[:n].clone()
        if not step_rows:
            return []
        alloc = _GpuFreezeAlloc(nbytes=sum(t.numel() * t.element_size() for t in step_rows.values()))
        segs: list[tuple[str, _Segment]] = []
        for req_id in req_order:
            sched = num_sched[req_id]
            if sched <= 0:
                continue
            start = query_start[req_id]
            end = start + sched
            segs.append(
                (
                    req_id,
                    _Segment(
                        slots_cpu=slots_cpu[start:end],
                        tensors={k: v[start:end] for k, v in step_rows.items()},
                        gpu_alloc=alloc,
                    ),
                )
            )
        return segs

    def _submit_step_writes(
        self,
        req_order: list[str],
        query_start: dict[str, int],
        num_sched: dict[str, int],
        frozen_rows: dict[str, torch.Tensor],
        slots_cpu: torch.Tensor,
        host_views: dict[str, torch.Tensor],
        freeze_event,
        staging_slot: int,
        step_d2h_event,
        bound_tids: list[int],
    ) -> None:
        """One queued WriteTask per request (locked phase).

        Per-req views of the shared frozen clone: one D2D freeze, req-scoped
        finish/abort, skip masks, and completion. Appends bound tids to
        `bound_tids` as it goes so a mid-loop raise still unwinds holders.
        """
        for req_id in req_order:
            start = query_start[req_id]
            end = start + num_sched[req_id]
            if end == start:
                continue
            req_rows = {k: v[start:end] for k, v in frozen_rows.items()}
            tid = self._alloc_task_id()
            seg = _Segment(slots_cpu=slots_cpu[start:end], tensors=req_rows)
            # Host rows are views into the slot; the committer only waits
            # the shared step event. D2H is already in flight.
            seg.host = {k: v[start:end] for k, v in host_views.items()}
            task = WriteTask(
                tid=tid,
                req_id=req_id,
                write_n=self._next_write_n(req_id),
                schedule=WriteSchedule.JOIN_NEXT_STEP,
                segments=[seg],
                freeze_event=freeze_event,
                staging_slot=staging_slot,
                step_d2h_event=step_d2h_event,
            )
            self._map_slots(slots_cpu[start:end], tid, req_rows.keys())
            # Bind before submit: the slot must never be holder-free
            # while the task is live (freed at completion drain).
            self._controller.staging_bind(staging_slot, StagingBufferHolder.for_task(tid))
            bound_tids.append(tid)
            self._controller.submit(task)
            self._req_tasks.setdefault(req_id, set()).add(tid)
            self._join_next_step_tids.append(tid)

    def _stage_deferred(self, segs: list[tuple[str, _Segment]], freeze_event) -> None:
        """Register pre-built deferred segments (locked phase; bytes already
        reserved by save_outputs)."""
        for req_id, seg in segs:
            task = self._deferred_tasks.get(req_id)
            if task is not None and not self._controller.append_segment(task, seg, freeze_event):
                # Entry closed under us (cap flush / escalation): start a new one.
                task = None
            if task is None:
                task = WriteTask(
                    tid=self._alloc_task_id(),
                    req_id=req_id,
                    write_n=self._next_write_n(req_id),
                    schedule=WriteSchedule.JOIN_ON_FINISH,
                    segments=[seg],
                    freeze_event=freeze_event,
                )
                self._deferred_tasks[req_id] = task
                self._req_tasks.setdefault(req_id, set()).add(task.tid)
                self._controller.submit(task, queued=False)
            if seg.gpu_alloc is not None:
                self._controller.pin_gpu_freeze(seg.gpu_alloc, task.tid)
            # Block reuse across deferred tenants (preemption path) is
            # handled inside _map_slots: the old tenant's rows are skipped.
            self._map_slots(seg.slots_cpu, task.tid, seg.tensors.keys())

    # ----------------------------------------------------- occupancy

    def _alloc_task_id(self) -> int:
        tid = self._next_tid
        self._next_tid += 1
        return tid

    def _next_write_n(self, req_id: str) -> int:
        n = self._write_n.get(req_id, 0) + 1
        self._write_n[req_id] = n
        return n

    def _tables(self, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._key_state.get(key)
        if state is None:
            state = self._key_state[key] = torch.zeros(self._num_slots, dtype=torch.int8)
            self._key_owner[key] = torch.zeros(self._num_slots, dtype=torch.int64)
        return state, self._key_owner[key]

    def _map_slots(self, slots: torch.Tensor, tid: int, keys: Iterable[str]) -> None:
        """Hang `tid` on these (slot, key); reassignment = task swap.

        A (slot, key) still in-transit under another task means block reuse
        (the old request was preempted/aborted): push those rows into the old
        task's skip set — its mirror write skips them, ours lands, so the
        writes are disjoint and need no ordering edge.
        """
        keys = tuple(keys)
        for key in keys:
            state, owner = self._tables(key)
            cur = owner[slots]
            stale = (state[slots] == _IN_TRANSIT) & (cur != tid) & (cur != 0)
            if bool(stale.any()):
                for old in {int(o) for o in cur[stale].tolist()}:
                    old_task = self._controller.get_task(old)
                    if old_task is not None:
                        old_task.add_reassigned(key, slots[stale & (cur == old)])
            state[slots] = _IN_TRANSIT
            owner[slots] = tid
        prev = self._task_slots.get(tid)
        if prev is None:
            self._task_slots[tid] = slots
            self._task_keys[tid] = keys
        else:
            # Deferred tasks grow one segment per step.
            self._task_slots[tid] = torch.cat([prev, slots])
            self._task_keys[tid] = tuple(dict.fromkeys(self._task_keys[tid] + keys))

    @torch.inference_mode()
    def _commit_drained_writes(self) -> None:
        """Fold controller drain into the (slot, key) tables.

        Failed writes raise here (hashes already published, rows unrecoverable).
        Successful scatters flip still-owned slots to COMMITTED.
        """
        failed = self._controller.drain_failed()
        if failed:
            # A failed write leaves rows absent behind hashes vLLM already
            # published — unservable and unrecoverable, so fatal. Raise here,
            # once, at the earliest facade entry instead of poisoning every
            # future hit that touches these slots.
            raise OmniPrefixCacheUnmatchError(
                f"prefix cache write failed for task(s) {failed}; cached rows lost behind published hashes"
            )
        drained = self._controller.drain_completed()
        for tid in drained:
            slots = self._task_slots.pop(tid, None)
            keys = self._task_keys.pop(tid, ())
            if slots is None:
                continue
            for key in keys:
                state, owner = self._tables(key)
                # Only publish slots we still own: a later entry may have
                # taken them over (block reuse), and its write must win.
                still_owned = owner[slots] == tid
                idx = slots[still_owned]
                state[idx] = _COMMITTED
                owner[idx] = 0
        if drained:
            # Drop completed ids so per-request sets stay bounded over long
            # streaming requests.
            done = set(drained)
            for req_id, eids in self._req_tasks.items():
                eids -= done

    # --------------------------------------------------- consume-once

    def _take_step_ctx(self, step_id: int) -> _StepContext:
        """Pop the context for this step id (consume-exactly-once)."""
        ctx = self._step_ctxs.pop(step_id, None)
        if ctx is None:
            raise OmniPrefixCacheUnmatchError(
                f"step context {step_id} missing (have {sorted(self._step_ctxs)}); already consumed or never saved"
            )
        return ctx

    def _raise_if_unconsumed_ctxs_at_capacity(self) -> None:
        """Unconsumed contexts at `staging_depth` means the runner skipped
        both materialize and discard_step. Checked before claiming a
        staging slot so a full pool does not hide the leaked ids.

        This bound currently equals the slot-pool depth (same config field).
        They are different failures; splitting them is a follow-up.
        """
        with self._state_lock:
            if len(self._step_ctxs) >= self._config.staging_depth:
                raise OmniPrefixCacheUnmatchError(
                    f"{len(self._step_ctxs)} unconsumed step contexts (ids={sorted(self._step_ctxs)}); "
                    "runner violated the consume-exactly-once contract"
                )

    def _release_step_staging_buffer(self, ctx: _StepContext, step_id: int) -> None:
        """Drop this step's holder on the staging-pool slot (no-op if no rows)."""
        if ctx.d2h is not None:
            self._controller.staging_release(ctx.d2h.slot, StagingBufferHolder.for_step(step_id))

    # -------------------------------------------------- plan / fetch

    def _plan_hit_rows(
        self, req_id: str, hit_upto: int, hit_blocks: list[int] | None, key: str, strict: bool
    ) -> _RowSource:
        """Resolve a hit span into a row-source plan (locked phase)."""
        bs = self._config.block_size
        assert hit_upto % bs == 0, (
            f"prefix hit not block aligned (req={req_id}, hit_upto={hit_upto}, block_size={bs}); "
            "vLLM invariant violated"
        )
        if hit_blocks is None:
            # The hit block table is snapshotted in new_step_starts precisely
            # so materialize never reads the live batch; a
            # missing snapshot is a registration bug, not a fallback case.
            raise OmniPrefixCacheUnmatchError(
                f"no hit-block snapshot for req {req_id} (hit_upto={hit_upto}); hit registered without block_ids"
            )
        block_ids = torch.tensor(hit_blocks, dtype=torch.int64)
        slots = (block_ids.unsqueeze(1) * bs + torch.arange(bs)).reshape(-1)[:hit_upto]

        state = self._key_state.get(key)
        states = state[slots] if state is not None else torch.zeros(int(slots.numel()), dtype=torch.int8)
        if strict and bool((states == _ABSENT).any()):
            # Same-step prefetch hits this before save registers the rows;
            # prefetch swallows it. materialize owns the fail-fast log.
            absent = slots[states == _ABSENT]
            named = [_Occupancy(int(s)).name for s in states[:64].tolist()]
            raise OmniPrefixCacheUnmatchError(
                f"hit span for req {req_id} key={key} hit_upto={hit_upto} "
                f"resolved to {int((states == _ABSENT).sum())} absent slots "
                f"(absent={absent[:32].tolist()}, states={named})"
            )

        return self._plan_rows(slots, key, strict, req_id, states)

    def _plan_rows(
        self,
        slots: torch.Tensor,
        key: str,
        strict: bool,
        req_id: str,
        states: torch.Tensor | None = None,
    ) -> _RowSource:
        """Pin the row sources for `slots` (locked phase; no data movement).

        In-transit rows win over the mirror: their rows may not have been
        scattered yet, and a mirror read would return zero/stale values.
        Per-key absent semantics: rows never registered for this key fall to
        the mirror baseline (legitimate for sparse/deferred keys — the strict
        caller has already rejected absent); rows registered in-transit whose
        entry cannot serve them are a bookkeeping error, never silent zeros.

        JOIN_NEXT_STEP owners are recorded as tids (join-then-pool at
        fetch). JOIN_ON_FINISH owners stay as task refs for fetch_host.
        """
        n = int(slots.numel())
        if states is None:
            state = self._key_state.get(key)
            states = state[slots] if state is not None else torch.zeros(n, dtype=torch.int8)
        owner_table = self._key_owner.get(key)
        owners = owner_table[slots] if owner_table is not None else torch.zeros(n, dtype=torch.int64)
        staged_mask = states == _IN_TRANSIT

        staged: list[tuple[WriteTask, torch.Tensor]] = []
        join_tids: list[int] = []
        for owner in {int(o) for o in owners[staged_mask].tolist()}:
            task = self._controller.get_task(owner) if owner != 0 else None
            if task is None:
                # in-transit implies a live owner entry (state flips to
                # committed in the same locked drain that retires the task).
                # Zeros here would ship silently downstream.
                raise OmniPrefixCacheUnmatchError(
                    f"(slot, {key}) rows of req {req_id} are in-transit but entry {owner} cannot serve them"
                )
            if task.schedule is WriteSchedule.JOIN_NEXT_STEP:
                join_tids.append(task.tid)
            else:
                staged.append((task, staged_mask & (owners == owner)))

        already_staged = self._pool.has_key(key)
        if not already_staged and not staged and not join_tids:
            if strict:
                raise OmniPrefixCacheUnmatchError(f"no data source for hit span of req {req_id}, key {key}")
            raise KeyError(f"key {key} has no cache mirror")
        return _RowSource(
            slots=slots,
            key=key,
            req_id=req_id,
            already_staged=already_staged,
            staged_list=staged,
            join_tids=join_tids,
        )

    def _fetch_source(self, src: _RowSource) -> torch.Tensor:
        """Fetch a planned row source (execute phase, no lock).

        JOIN_NEXT_STEP: wait ``done`` (not host_ready — pool is still zero
        after D2H), drain so manager tables flip to COMMITTED, then read
        the pool. JOIN_ON_FINISH: fetch_host on the GPU freeze.
        """
        if src.join_tids:
            self._controller.join(src.join_tids)
            with self._state_lock:
                self._commit_drained_writes()
        n = int(src.slots.numel())
        out: torch.Tensor | None = None
        if src.already_staged or src.join_tids:
            if self._pool.has_key(src.key):
                out = self._pool.rows(src.key, src.slots)
        for task, mask in src.staged_list:
            try:
                rows = self._controller.fetch_host(task, src.slots[mask], src.key)
            except KeyError:
                raise OmniPrefixCacheUnmatchError(
                    f"(slot, {src.key}) rows of req {src.req_id} are staged in entry "
                    f"{task.tid} (req {task.req_id}, write_n {task.write_n}) but the task cannot serve them"
                ) from None
            if out is None:
                out = torch.zeros((n, rows.shape[-1]), dtype=rows.dtype)
            out[mask] = rows
        assert out is not None  # _plan_rows guarantees a source
        in_transit = None
        for _, mask in src.staged_list:
            in_transit = mask if in_transit is None else in_transit | mask
        self._ensure_not_reassigned(src.slots, src.key, in_transit_mask=in_transit, req_id=src.req_id)
        return out

    def _ensure_not_reassigned(
        self,
        slots: torch.Tensor,
        key: str,
        *,
        in_transit_mask: torch.Tensor | None = None,
        req_id: str = "?",
    ) -> None:
        """Post-fetch validation: pool rows read without a lock may have
        been remounted mid-read (block reuse). A torn pool read must
        fail-fast. JOIN_ON_FINISH slots already in-transit at plan time
        are excluded; JOIN_NEXT_STEP slots must be COMMITTED after drain.
        """
        with self._state_lock:
            state = self._key_state.get(key)
            if state is None:
                return
            violated = state[slots] == _IN_TRANSIT
            if in_transit_mask is not None:
                violated &= ~in_transit_mask
            if bool(violated.any()):
                raise OmniPrefixCacheUnmatchError(
                    f"(slot, {key}) rows of req {req_id} were reassigned to a new entry during "
                    f"materialize ({int(violated.sum())} slots; block reuse mid-read)"
                )

    # ---------------------------------------------------------- merge

    def _merge_cached_for_req(
        self,
        ctx: _StepContext,
        req_id: str,
        key: str,
        current_cpu: torch.Tensor,
        hit_sources: dict[tuple[str, str], _RowSource | Future],
    ) -> torch.Tensor:
        """Hit prefix + this step's rows for one (req, key).

        No hit → this step's slice only. Prefetch Future → write the
        slice into the reserved tail. Else cat(fetch, new).
        """
        if req_id not in ctx.spans:
            # The caller passed a req the step context never saw: a silent
            # empty slice here would ship a zero-row payload downstream.
            raise OmniPrefixCacheUnmatchError(f"req {req_id} not in this step's context (had {list(ctx.spans)[:8]})")
        start, end = ctx.spans[req_id]
        new_rows = current_cpu[start:end]
        src = hit_sources.get((req_id, key))
        if src is None:
            return new_rows
        if isinstance(src, Future):
            # Prefetched during the forward, prefix already in place; only
            # this step's rows land here. result() re-raises fetch/validation
            # errors — the fail-fast contract survives the thread hop.
            merged = src.result()
            merged[merged.shape[0] - new_rows.shape[0] :] = new_rows
            return merged
        cached = self._fetch_source(src)
        return torch.cat([cached, new_rows], dim=0)

    def _merge_uncached_mm(
        self,
        ctx: _StepContext,
        req_ids: list[str],
        cached_keys: set[str],
        mm_out: dict[str, dict[str, Any]],
    ) -> None:
        """Write mm keys that are not in the prefix cache into mm_out.

        No hit concat: leftover mm was already copied to CPU at save
        (``ctx.mm_cpu_snapshot``). cached_keys already went through
        _merge_cached_for_req. ``req_ids`` is a subset of ``ctx.spans``.
        """
        uncached = {k: v for k, v in ctx.mm_cpu_snapshot.items() if k not in cached_keys and k != HIDDEN_KEY}
        if not uncached:
            return
        from vllm_omni.utils.mm_outputs import to_payload_element

        order = list(ctx.spans)
        total = sum(e - s for s, e in ctx.spans.values())
        for key, val in uncached.items():
            per_req: dict[str, Any] = {}
            for req_id in req_ids:
                idx = order.index(req_id)
                start, end = ctx.spans[req_id]
                per_req[req_id] = to_payload_element(
                    val,
                    idx,
                    start=start,
                    end=end,
                    pass_lists_through=True,
                    seq_len=total,
                )
            mm_out[key] = per_req
