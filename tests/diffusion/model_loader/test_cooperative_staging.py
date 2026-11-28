# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import threading
import time

import pytest
import torch

from vllm_omni.diffusion.model_loader import cooperative_staging, pinned_staging
from vllm_omni.diffusion.model_loader.cooperative_staging import (
    _BucketPlanner,
    cooperative_staging_weights_iterator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeGroup:
    """In-process rendezvous standing in for a TP process group: rank threads
    meet at barriers for all_reduce/broadcast, mirroring NCCL's blocking
    collective semantics (including the deadlock-on-divergence property the
    protocol is designed around — a barrier timeout IS the hang)."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size, timeout=5)
        self.vals = [[0]] * world_size
        self.payload = None


class _FakeComm:
    def __init__(self, group: _FakeGroup, rank: int):
        self.group = group
        self.rank = rank
        self.world_size = group.world_size
        self.device = torch.device("cpu")

    def all_reduce_max(self, values: list[int]) -> list[int]:
        self.group.vals[self.rank] = list(values)
        self.group.barrier.wait()
        result = [max(v[i] for v in self.group.vals) for i in range(len(values))]
        self.group.barrier.wait()
        return result

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        if self.rank == src:
            self.group.payload = tensor
        self.group.barrier.wait()
        if self.rank != src:
            tensor.copy_(self.group.payload)
        self.group.barrier.wait()


def _run_ranks(
    world_size,
    make_stream,
    per_rank=None,
    return_errors=False,
    comm_factory=_FakeComm,
    **iter_kwargs,
):
    """One iterator per rank on its own thread; asserts nobody deadlocked or
    raised, and returns each rank's collected (name, dtype, device, value).
    ``per_rank`` optionally supplies each rank's kwargs instead."""
    group = _FakeGroup(world_size)
    results = [None] * world_size
    errors = [None] * world_size

    def _rank_main(rank):
        try:
            it = cooperative_staging_weights_iterator(
                make_stream(rank), comm=comm_factory(group, rank), **(per_rank[rank] if per_rank else iter_kwargs)
            )
            results[rank] = [(n, t.dtype, t.device.type, t.clone()) for n, t in it]
        except RuntimeError as exc:
            errors[rank] = exc

    threads = [threading.Thread(target=_rank_main, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "rank thread deadlocked"
    if return_errors:
        return results, errors
    assert all(e is None for e in errors), f"rank raised: {errors}"
    return results


# 96 KiB fp32 tensors: above pinned staging's tiny-tensor threshold so the
# fallback paths stage them too.
def _stream(n=12, numel=24576):
    for i in range(n):
        yield f"layer.{i}.weight", torch.full((numel,), float(i), dtype=torch.float32)


def _assert_complete_bf16(rank_out, n, ordered=True):
    names = [name for name, *_ in rank_out]
    want = [f"layer.{i}.weight" for i in range(n)]
    assert (names if ordered else sorted(names)) == (want if ordered else sorted(want))
    for name, dtype, _, value in rank_out:
        i = int(name.split(".")[1])
        assert dtype is torch.bfloat16
        assert torch.equal(value, torch.full((24576,), float(i), dtype=torch.bfloat16))


def test_bucket_plan_determinism_and_ownership():
    """The plan is a pure function of the metadata stream: every rank computes
    identical bucket boundaries and owners with no communication. Ownership is
    greedy least-loaded (balanced bytes), an oversized tensor both evicts the
    open bucket and closes solo (the one add() that returns two buckets), and
    packing offsets are 16-byte aligned with the fused-cast OUTPUT sizes."""

    def _plan(world):
        planner = _BucketPlanner(world, 1 << 20, target_dtypes={"big": torch.float32}, default_dtype=torch.bfloat16)
        buckets = []
        for name, t in [
            ("a", torch.zeros(1000, dtype=torch.float32)),  # -> bf16, 2000 B
            ("b", torch.zeros(300000, dtype=torch.float32)),  # -> bf16, 600 KB
            ("big", torch.zeros(1 << 19, dtype=torch.float32)),  # stays fp32: 2 MiB, oversized
            ("c", torch.zeros(1000, dtype=torch.float32)),
        ]:
            buckets.extend(planner.add(name, t))
        tail = planner.flush()
        if tail is not None:
            buckets.append(tail)
        return buckets

    p1, p2 = _plan(2), _plan(2)
    assert [[e.name for e in b.entries] for b in p1] == [[e.name for e in b.entries] for b in p2]
    assert [b.owner for b in p1] == [b.owner for b in p2]

    # 'big' evicted {a,b} and closed solo: two buckets from one add()
    assert [[e.name for e in b.entries] for b in p1] == [["a", "b"], ["big"], ["c"]]
    # greedy: bucket0 (602KB) -> rank0, big (2MiB) -> rank1, c -> rank0 (least loaded)
    assert [b.owner for b in p1] == [0, 1, 0]
    # fused-cast output sizes and alignment
    a, b = p1[0].entries
    assert a.nbytes == 2000 and a.out_dtype is torch.bfloat16
    assert b.offset % 16 == 0 and b.offset == 2000  # 2000 is already 16-aligned
    assert p1[1].entries[0].out_dtype is torch.float32  # exact-name map wins


def test_cooperative_stream_contract(monkeypatch):
    """Every rank yields every tensor in checkpoint order on comm.device with
    the fused cast applied, regardless of which rank owned/staged the bucket;
    the group stays in lockstep across many buckets and pipeline windows
    (small bucket_bytes forces multi-bucket, multi-owner, multi-window
    traffic), and both ranks observe identical bytes."""
    monkeypatch.setattr(cooperative_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))
    results = _run_ranks(
        2,
        lambda rank: _stream(n=12),
        bucket_bytes=200 << 10,  # 4 bf16 tensors per bucket -> 3 buckets
        target_dtypes={"layer.0.weight": torch.float32},
        default_dtype=torch.bfloat16,
    )
    for rank_out in results:
        assert [n for n, *_ in rank_out] == [f"layer.{i}.weight" for i in range(12)]
        for name, dtype, device, value in rank_out:
            i = int(name.split(".")[1])
            want_dtype = torch.float32 if i == 0 else torch.bfloat16
            assert dtype is want_dtype and device == "cpu"
            assert torch.equal(value, torch.full((24576,), float(i), dtype=want_dtype))
    for (_, _, _, v0), (_, _, _, v1) in zip(results[0], results[1]):
        assert torch.equal(v0, v1)

    # Tight budget forces several windows; next-window staging overlaps H2D.
    results = _run_ranks(
        2,
        lambda rank: _stream(n=20),
        bucket_bytes=200 << 10,
        max_inflight_bytes=400 << 10,
        default_dtype=torch.bfloat16,
    )
    for rank_out in results:
        _assert_complete_bf16(rank_out, 20)


def test_owner_h2d_failure_rendezvouses_before_broadcast(monkeypatch):
    """An owner-side H2D failure must be voted across the group before any
    peer enters broadcast; every rank can then replay the unyielded window
    locally instead of leaving peers blocked in the collective."""
    monkeypatch.setattr(cooperative_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))
    real_copy = cooperative_staging._copy_bucket_to_device
    tripped = threading.Event()

    def _fail_one_owner(bucket, stream):
        if bucket.owner == 0 and not tripped.is_set():
            tripped.set()
            raise RuntimeError("injected H2D failure")
        return real_copy(bucket, stream)

    monkeypatch.setattr(cooperative_staging, "_copy_bucket_to_device", _fail_one_owner)
    results = _run_ranks(
        2,
        lambda rank: _stream(n=12),
        bucket_bytes=200 << 10,
        default_dtype=torch.bfloat16,
    )

    assert tripped.is_set()
    for rank_out in results:
        _assert_complete_bf16(rank_out, 12, ordered=False)


def test_cooperative_memory_budget_and_oversized_fallback(monkeypatch):
    """The configured budget caps packed collective windows and the pinned
    free list. A single tensor larger than the budget still cooperates as a
    singleton window (fallback would make every rank stage it). Payload OOM
    still falls back before any broadcast."""
    monkeypatch.setattr(cooperative_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))

    class _BudgetComm(_FakeComm):
        max_window_bytes = 0
        broadcasts = 0

        def all_reduce_max(self, values):
            if len(values) == 8:
                type(self).max_window_bytes = max(type(self).max_window_bytes, values[3])
            return super().all_reduce_max(values)

        def broadcast(self, tensor, src):
            type(self).broadcasts += 1
            return super().broadcast(tensor, src)

    budget = 400 << 10
    results = _run_ranks(
        2,
        lambda rank: _stream(n=20),
        comm_factory=_BudgetComm,
        bucket_bytes=200 << 10,
        max_inflight_bytes=budget,
    )
    assert all(len(rank_out) == 20 for rank_out in results)
    assert _BudgetComm.max_window_bytes <= budget

    _BudgetComm.broadcasts = 0
    oversized = 200_000  # fp32 = 800 KiB, larger than the 400 KiB group budget
    results = _run_ranks(
        2,
        lambda rank: _stream(n=1, numel=oversized),
        comm_factory=_BudgetComm,
        bucket_bytes=200 << 10,
        max_inflight_bytes=budget,
    )
    assert all(len(rank_out) == 1 for rank_out in results)
    name, dtype, _, value = results[0][0]
    assert name == "layer.0.weight" and dtype is torch.float32
    assert torch.equal(value, torch.zeros(oversized, dtype=torch.float32))
    assert _BudgetComm.broadcasts >= 2

    real_alloc_payload = cooperative_staging._alloc_device_payload
    failed = threading.Event()

    def _oom_once(nbytes, device):
        if not failed.is_set():
            failed.set()
            raise torch.cuda.OutOfMemoryError("injected payload OOM")
        return real_alloc_payload(nbytes, device)

    monkeypatch.setattr(cooperative_staging, "_alloc_device_payload", _oom_once)
    _BudgetComm.broadcasts = 0
    results = _run_ranks(
        2,
        lambda rank: _stream(n=8),
        comm_factory=_BudgetComm,
        bucket_bytes=200 << 10,
        max_inflight_bytes=budget,
    )
    assert failed.is_set() and all(len(rank_out) == 8 for rank_out in results)
    assert _BudgetComm.broadcasts == 0

    pool = cooperative_staging._PinnedPool(max_cached_bytes=budget)
    pool.put(torch.empty(256 << 10, dtype=torch.uint8))
    pool.put(torch.empty(256 << 10, dtype=torch.uint8))
    assert pool.cached_bytes <= budget
    assert pool.drops == 1


def test_group_degradation_paths(monkeypatch):
    """Degradation paths preserve each local stream without deadlocking;
    coordinated source failures abort every rank instead of stranding peers
    in a collective (a barrier timeout here IS that hang):

    1. Pre-flight veto: one ineligible rank vetoes cooperation for everyone;
       the eligible rank degrades to per-rank pinned staging, the ineligible
       one to pass-through.
    2. Mid-stream stage failure on ONE rank aborts the whole group at the
       same window boundary via the error all-reduce.
    3. Rank-divergent source order (the production incident: a completion-
       order multi-thread shard iterator) is caught by the plan agreement.
    4. Equal names/bytes with divergent metadata are also rejected.
    5. Different stream lengths rendezvous on the terminal agreement.
    6. A source exception aborts every rank without hanging.
    7. Preflight, agreement, and broadcast collective failures make every
       rank fall back locally without losing the current window."""
    monkeypatch.setattr(cooperative_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))

    # 1. pre-flight veto (rank1 ineligible)
    results = _run_ranks(
        2,
        lambda rank: _stream(n=6),
        per_rank=[
            dict(local_eligible=True, default_dtype=torch.bfloat16),
            dict(local_eligible=False, default_dtype=torch.bfloat16),
        ],
    )
    _assert_complete_bf16(results[0], 6, ordered=False)  # staged: threads may reorder
    assert [n for n, *_ in results[1]] == [f"layer.{i}.weight" for i in range(6)]
    assert all(d is torch.float32 for _, d, _, _ in results[1])  # pass-through: untouched

    # 2. one rank's staging fails mid-stream (whichever rank owns the bucket
    # headed by layer.4 hits it; the other learns via the all-reduce). Use
    # enough tensors to leave a partially planned bucket beyond the pipeline
    # window; fallback must retain that bucket too.
    real_stage = cooperative_staging._stage_bucket
    tripped = threading.Event()

    def _flaky_stage(bucket, pool, cap):
        if bucket.entries[0].name == "layer.4.weight" and not tripped.is_set():
            tripped.set()
            bucket.error = RuntimeError("injected stage failure")
            return
        real_stage(bucket, pool, cap)

    monkeypatch.setattr(cooperative_staging, "_stage_bucket", _flaky_stage)
    results = _run_ranks(2, lambda rank: _stream(n=40), bucket_bytes=200 << 10, default_dtype=torch.bfloat16)
    assert tripped.is_set()
    for rank_out in results:
        _assert_complete_bf16(rank_out, 40, ordered=False)
    monkeypatch.setattr(cooperative_staging, "_stage_bucket", real_stage)

    # 3. divergent source order across ranks
    def _divergent(rank):
        items = list(_stream(n=8))
        return iter(items if rank == 0 else items[::-1])

    results = _run_ranks(2, _divergent, bucket_bytes=200 << 10, default_dtype=torch.bfloat16)
    for rank_out in results:
        _assert_complete_bf16(rank_out, 8, ordered=False)

    # 4. Same names and aggregate bytes, but different per-entry metadata.
    # The agreement signature must include shapes/dtypes/bucket layout rather
    # than accepting name+window-total equality and broadcasting corrupt views.
    def _metadata_divergent(rank):
        sizes = (1024, 2048) if rank == 0 else (2048, 1024)
        for i, size in enumerate(sizes):
            yield f"w.{i}", torch.full((size,), float(i))

    results = _run_ranks(2, _metadata_divergent, bucket_bytes=1 << 20)
    for rank, rank_out in enumerate(results):
        sizes = (1024, 2048) if rank == 0 else (2048, 1024)
        assert [(name, value.numel()) for name, _, _, value in rank_out] == [
            (f"w.{i}", size) for i, size in enumerate(sizes)
        ]
        for i, (_, _, _, value) in enumerate(rank_out):
            assert torch.equal(value, torch.full((sizes[i],), float(i)))

    # 5. Different stream lengths must rendezvous on an explicit terminal
    # window and fall back locally; the shorter rank must not leave its peer
    # blocked in the next collective.
    results = _run_ranks(2, lambda rank: _stream(n=8 if rank == 0 else 12), bucket_bytes=200 << 10)
    assert [len(rank_out) for rank_out in results] == [8, 12]

    # 6. A source exception is a real checkpoint failure, not a staging
    # fallback, but it must abort every rank rather than strand peers.
    def _source_failure(rank):
        yield from _stream(n=4)
        if rank == 0:
            raise RuntimeError("checkpoint corrupt")
        yield from _stream(n=4)

    _, errors = _run_ranks(2, _source_failure, bucket_bytes=200 << 10, return_errors=True)
    assert all(isinstance(error, RuntimeError) for error in errors)

    # 7. The fake transport reports the same collective failure to every
    # participant, as an aborted NCCL communicator does. No item in the
    # in-flight window has been yielded yet, so every rank can safely replay
    # that window and the untouched tail through local pinned staging.
    class _FailingCollectiveComm(_FakeComm):
        operation = ""

        def __init__(self, group, rank):
            super().__init__(group, rank)
            self.all_reduce_calls = 0

        def all_reduce_max(self, values):
            self.all_reduce_calls += 1
            if self.operation == "preflight" and self.all_reduce_calls == 1:
                raise RuntimeError("injected preflight failure")
            if self.operation == "all_reduce" and self.all_reduce_calls == 2:
                raise RuntimeError("injected all-reduce failure")
            return super().all_reduce_max(values)

        def broadcast(self, tensor, src):
            if self.operation == "broadcast":
                raise RuntimeError("injected broadcast failure")
            return super().broadcast(tensor, src)

    for operation in ("preflight", "all_reduce", "broadcast"):
        comm_type = type(f"Failing{operation.title()}Comm", (_FailingCollectiveComm,), {"operation": operation})
        results = _run_ranks(
            2,
            lambda rank: _stream(n=12),
            comm_factory=comm_type,
            bucket_bytes=200 << 10,
            default_dtype=torch.bfloat16,
        )
        for rank_out in results:
            _assert_complete_bf16(rank_out, 12, ordered=False)


def test_owner_bucket_is_staged_exactly_once(monkeypatch):
    """A bucket leaves ``staged`` for the forming window without being
    submitted again.

    Regression for a double submission: ``_fill_staged`` admitted a bucket
    (one ``_stage_bucket`` task), then ``_drain_window`` re-admitted the same
    bucket when relinking ``staged`` -> ``pending``. Besides staging every byte
    twice, the second submit overwrote ``bucket.future`` (so the first task's
    exception never reached the error vote) and ``bucket.pinned`` (so the
    consumer could copy a buffer the first task was still filling). The check
    below counts submissions per bucket and also holds the first task at the
    pool to make the overwrite window wide.
    """
    monkeypatch.setattr(cooperative_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))

    # Count one submission per *tensor*: a double-submitted bucket stages each
    # of its entries twice, so any value other than 1 is the regression.
    per_name: dict[str, int] = {}
    finished: set[str] = set()
    lock = threading.Lock()
    real_stage = cooperative_staging._stage_bucket

    def _counting_stage(bucket, pool, cap):
        # Hold the task before it can claim a buffer, widening the window in
        # which a second submission would overwrite `pinned`/`future`.
        time.sleep(0.002)
        with lock:
            for entry in bucket.entries:
                per_name[entry.name] = per_name.get(entry.name, 0) + 1
        real_stage(bucket, pool, cap)
        with lock:
            finished.update(entry.name for entry in bucket.entries)

    monkeypatch.setattr(cooperative_staging, "_stage_bucket", _counting_stage)

    n = 12
    bucket_bytes = 200 << 10  # two tensors per bucket -> several windows
    results = _run_ranks(2, lambda rank: _stream(n=n), bucket_bytes=bucket_bytes, default_dtype=torch.bfloat16)
    for rank_out in results:
        _assert_complete_bf16(rank_out, n, ordered=False)

    # Every tensor is owned and staged by exactly one rank, exactly once.
    assert set(per_name) == {f"layer.{i}.weight" for i in range(n)}
    assert set(per_name.values()) == {1}, per_name
    # The task that produced the yielded bytes had already completed; with a
    # single submission the yield waits on that task.
    assert len(finished) == n


def test_staged_bucket_keeps_its_first_future(monkeypatch):
    """Relinking ``staged`` -> ``pending`` must not replace ``bucket.future``.

    A replaced future would let the window loop wait on the wrong task while an
    older task still writes ``bucket.pinned``.
    """
    monkeypatch.setattr(cooperative_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))

    seen: list[tuple[str, object]] = []
    real_alloc = cooperative_staging._BucketPlanner

    class _RecordingPlanner(real_alloc):
        def add(self, name, tensor):
            buckets = super().add(name, tensor)
            for bucket in buckets:
                seen.append(("admitted", bucket))
            return buckets

    monkeypatch.setattr(cooperative_staging, "_BucketPlanner", _RecordingPlanner)

    results = _run_ranks(2, lambda rank: _stream(n=8), bucket_bytes=200 << 10, default_dtype=torch.bfloat16)
    for rank_out in results:
        _assert_complete_bf16(rank_out, 8, ordered=False)

    # One planner admission per bucket, and every admitted bucket object is
    # unique: a re-admission would have produced a second, distinct bucket for
    # the same tensors.
    admitted = [b for tag, b in seen if tag == "admitted"]
    assert len(admitted) == len({id(b) for b in admitted})


def test_preflight_votes_residency_and_eligibility_in_one_collective(monkeypatch):
    """The residency verdict rides in the eligibility all-reduce.

    Two facts need agreement before any bucket is planned: every rank can pin,
    and no rank has the checkpoint warm already. Sending them as two separately
    gated collectives is what previously stranded a rank: a resolver that
    decided locally skipped the first vote while its peer entered it. One
    collective with two slots cannot be entered by one rank alone.
    """
    monkeypatch.setattr(cooperative_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))

    cases = [
        # (per-rank (eligible, resident), expected comm, expected resident)
        ([(True, True), (True, True)], False, True),
        ([(True, False), (True, False)], True, False),
        # One warm rank does not veto cooperation: the cold rank is the critical
        # path and still benefits, while the warm rank only pays a broadcast per
        # bucket instead of skipping a collective its peers have entered.
        ([(True, True), (True, False)], True, False),
        ([(False, False), (True, False)], False, False),
    ]
    for per_rank, want_comm, want_resident in cases:
        group = _FakeGroup(len(per_rank))
        out = [None] * len(per_rank)

        def _go(rank, spec):
            monkeypatch.setattr(cooperative_staging, "_tp_comm", lambda: _FakeComm(group, rank))
            out[rank] = cooperative_staging.prepare_cooperative_staging(spec[0], local_resident=spec[1])

        threads = [threading.Thread(target=_go, args=(r, spec)) for r, spec in enumerate(per_rank)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not any(t.is_alive() for t in threads), f"preflight deadlocked for {per_rank}"
        for ctx in out:
            assert (ctx.comm is not None) is want_comm, per_rank
            assert ctx.resident is want_resident, per_rank


def test_iterator_without_a_group_never_resolves_one(monkeypatch):
    """``context``/``comm`` absent means per-rank staging, not "find a group".

    Resolving the TP group is a collective. Leaving that to this iterator is how
    a component that only some ranks load ended up in a 600 s all-reduce with no
    peer: the loader passed ``context=None``, which used to mean "go resolve it".
    """
    monkeypatch.setattr(cooperative_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))
    touched = []

    def _peek():
        touched.append(True)
        return _FakeComm(_FakeGroup(2), 0)

    monkeypatch.setattr(cooperative_staging, "_tp_comm", _peek)
    names = [name for name, _ in cooperative_staging_weights_iterator(_stream(n=4))]
    assert touched == [], "the iterator must not look up a group on its own"
    # Per-rank pinned staging may reorder (staging threads), so compare as a set.
    assert sorted(names) == sorted(f"layer.{i}.weight" for i in range(4))


def test_consumer_leaving_early_is_reported(monkeypatch, caplog):
    """Walking away mid-stream must say so: the peer only sees a bare timeout.

    An NCCL collective cannot be interrupted, so a consumer that stops early
    (exception or early break) leaves peers in their next schedule all-reduce
    until the process-group timeout, and the watchdog then aborts them
    (measured: a 20 s timeout aborted the peer after 20.9 s). This rank's log is
    the only place that can explain the peer's timeout.
    """
    monkeypatch.setattr(cooperative_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))

    # world_size=1 keeps the barrier free: we only exercise the abandonment path.
    group = _FakeGroup(1)
    comm = _FakeComm(group, 0)

    with caplog.at_level("WARNING", logger=cooperative_staging.logger.name):
        it = cooperative_staging_weights_iterator(iter(_stream(n=8)), comm=comm, bucket_bytes=200 << 10)
        next(it)
        it.close()  # what a consumer exception does to the generator
    assert any("consumer left" in rec.message for rec in caplog.records), [r.message for r in caplog.records]

    caplog.clear()
    with caplog.at_level("WARNING", logger=cooperative_staging.logger.name):
        list(cooperative_staging_weights_iterator(iter(_stream(n=8)), comm=comm, bucket_bytes=200 << 10))
    assert not any("consumer left" in rec.message for rec in caplog.records)
