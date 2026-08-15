# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cooperative weight staging across a tensor-parallel group (RFC #4896, item 3).

Under TP, every rank drains the same full checkpoint iterator and its layer
``weight_loader`` keeps only this rank's slice. Measured on Wan2.2 fp32 TP2,
per-rank pinned staging recovered only 1.05x (vs 1.43x on one GPU) because
the whole pipeline runs N times: pin+cast burn the same cores N times, N
interleaved readers break sequential readahead, and the per-rank narrow of a
staged *pinned* tensor makes the H2D source strided — off the DMA fast path.

This iterator splits the work instead of repeating it. Tensors are packed
into byte buckets; each bucket has one deterministic owner rank. The owner
stages its buckets exactly like pinned staging (mmap read, pooled pinned
buffer, fused dtype cast), pushes the packed bucket to the GPU with one
contiguous H2D copy, and broadcasts it to the group. Every rank then yields
GPU-resident tensors in the original checkpoint order, and the layer
``weight_loader`` narrows on the GPU — cheap, and the strided-pinned problem
disappears. Per-rank disk faults, pin, cast and H2D all drop to ~1/N.

Buckets are processed in pipeline windows: transfers issue back-to-back on
a dedicated CUDA stream and the consumer gates on per-bucket events, so
window N's transfers overlap window N-1's weight_loader copies and window
N+1's staging (measured: Wan2.2 cold 20.6s -> 15.4s over the serialized
form — the apparent "disk floor" had serialized transfer time hidden in it).

Lockstep: all ranks walk identical (name, shape, dtype) streams (the local
safetensors headers), so the bucket plan and ownership are computed
independently yet identically — no metadata communication. Only owners touch
tensor *data*; mmap keeps non-owner reads free.

Failure semantics: collectives make one-sided fallback a deadlock, so
degradation is group-wide. A pre-flight vote (all-reduce) enters cooperative
mode only if every rank is eligible; a per-window error+plan-agreement vote
coordinates staging failures and catches rank-divergent source streams. Those
failures, plus all-reduce or broadcast transport failures reported to the
participants, replay the unyielded window and remaining stream through plain
per-rank pinned staging. Checkpoint source errors are coordinated separately
and raised on every rank. The load never fails because of staging — same
contract as ``pinned_staging``.

Consumer contract: identical to pinned staging — a yielded tensor is valid
only until the next item is requested.
"""

import threading
import time
import zlib
from collections.abc import Generator, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import chain
from types import TracebackType

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.model_loader.pinned_staging import (
    _alloc_pinned,
    _bucket_bytes,
    _prefault,
    _resolve_out_dtype,
    pinned_staging_weights_iterator,
)

logger = init_logger(__name__)

# The bucket count IS the protocol overhead (one broadcast per bucket, one
# agreement all-reduce per window). 64 MiB packed Wan2.2's ~50 MiB tensors
# fine (444 buckets) but HunyuanImage MoE's ~31 MiB tensors barely packed:
# 4273 buckets of collectives ate the entire warm-load win. 256 MiB caps
# the count while the transient GPU/pinned footprint stays trivial.
_DEFAULT_BUCKET_BYTES = 256 << 20
# Offsets inside a bucket are aligned so a uint8 slice can be viewed as any
# dtype (largest element size is 8 bytes; 16 also keeps copies SIMD-friendly).
_ALIGN = 16
# Buckets per pipeline window; also the stagers' plan-ahead depth.
_WINDOW_BUCKETS = 4
_STAGE_WORKERS = 4  # concurrent bucket stagings per rank
_COLLECTIVE_ERRORS = (OSError, RuntimeError, TypeError, ValueError)


class _CapturedFailure:
    """Capture an operation's failure for coordinated cross-rank handling."""

    def __init__(self) -> None:
        self.error: BaseException | None = None

    def __enter__(self) -> None:
        self.error = None
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, traceback
        self.error = exc
        return exc is not None


class _TorchDistComm:
    """Collectives on a torch process group (NCCL in production)."""

    # [stage_err, source_err, buckets, total, hash,
    #  -buckets, -total, -hash]
    _VEC = 8

    def __init__(self, group: torch.distributed.ProcessGroup, device: torch.device):
        self._group = group
        self.device = device
        self.rank = torch.distributed.get_rank(group)
        self.world_size = torch.distributed.get_world_size(group)
        self._flag = torch.zeros(self._VEC, dtype=torch.int64, device=device)

    def all_reduce_max(self, values: list[int]) -> list[int]:
        self._flag.zero_()
        self._flag[: len(values)] = torch.tensor(values, dtype=torch.int64)
        torch.distributed.all_reduce(self._flag, op=torch.distributed.ReduceOp.MAX, group=self._group)
        return self._flag.tolist()[: len(values)]

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        global_src = torch.distributed.get_global_rank(self._group, src)
        torch.distributed.broadcast(tensor, src=global_src, group=self._group)


def _tp_comm() -> _TorchDistComm | None:
    """The TP group as a comm, or None when not distributed / TP degenerate."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    try:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        if group.world_size <= 1:
            return None
        device = (
            torch.device("cuda", torch.accelerator.current_device_index())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        return _TorchDistComm(group.device_group, device)
    except (ImportError, AttributeError, AssertionError, RuntimeError, ValueError):
        return None


@dataclass
class _Entry:
    name: str
    src: torch.Tensor
    out_dtype: torch.dtype
    offset: int
    nbytes: int


@dataclass
class _Bucket:
    owner: int
    entries: list[_Entry] = field(default_factory=list)
    total: int = 0  # payload bytes (last entry's offset + nbytes)
    future: Future[None] | None = None  # owner-side staging task
    stage_s: float = 0.0  # owner-side read + staging-copy time
    pinned: torch.Tensor | None = None  # owner-side staged buffer
    error: BaseException | None = None
    gpu: torch.Tensor | None = None  # broadcast payload
    h2d_start: object | None = None  # owner H2D timing event
    h2d_done: object | None = None  # owner H2D event (travels with pinned buf)
    ready: object | None = None  # transfer-stream event gating the yields


class _BucketPlanner:
    """Packs the (name, tensor) stream into owned buckets, identically on
    every rank: ownership is greedy least-loaded-by-bytes (ties to the lower
    rank), computed purely from the deterministic metadata stream."""

    def __init__(
        self,
        world_size: int,
        bucket_bytes: int,
        target_dtypes: Mapping[str, torch.dtype] | None,
        default_dtype: torch.dtype | None,
    ):
        self._world = world_size
        self._cap = bucket_bytes
        self._target_dtypes = target_dtypes
        self._default_dtype = default_dtype
        self._loads = [0] * world_size
        self._open: list[tuple[str, torch.Tensor, torch.dtype, int]] = []
        self._open_total = 0

    def _close(self) -> _Bucket | None:
        if not self._open:
            return None
        owner = min(range(self._world), key=lambda r: (self._loads[r], r))
        self._loads[owner] += self._open_total
        bucket = _Bucket(owner=owner)
        offset = 0
        for name, src, out_dtype, nbytes in self._open:
            bucket.entries.append(_Entry(name, src, out_dtype, offset, nbytes))
            offset += -(-nbytes // _ALIGN) * _ALIGN
        bucket.total = bucket.entries[-1].offset + bucket.entries[-1].nbytes
        self._open, self._open_total = [], 0
        return bucket

    def add(self, name: str, tensor: torch.Tensor) -> list[_Bucket]:
        """Add one tensor; returns the buckets this add closed (0, 1, or 2 —
        two when an oversized tensor both evicts the open bucket and closes
        solo)."""
        out_dtype = _resolve_out_dtype(name, tensor.dtype, self._target_dtypes, self._default_dtype)
        nbytes = tensor.numel() * out_dtype.itemsize
        closed = []
        if self._open and self._open_total + nbytes > self._cap:
            closed.append(self._close())
        self._open.append((name, tensor, out_dtype, nbytes))
        self._open_total += -(-nbytes // _ALIGN) * _ALIGN
        if self._open_total >= self._cap:
            closed.append(self._close())
        return closed

    def flush(self) -> _Bucket | None:
        return self._close()


class _PinnedPool:
    """Free list of pinned bucket buffers, shared between the consumer loop
    and the stager threads (hence the lock).

    A buffer may be repooled while its H2D copy is still in flight; the
    associated event is waited only when the buffer is REUSED (in get, on a
    stager thread) so the wait stays off the per-bucket consumer path."""

    def __init__(self):
        self._free: list[tuple[torch.Tensor, object | None]] = []
        self._lock = threading.Lock()

    def get(self, nbytes: int, floor: int) -> torch.Tensor:
        need = _bucket_bytes(max(nbytes, 1))
        with self._lock:
            found = None
            for i, (free, event) in enumerate(self._free):
                if free.numel() >= need:
                    found = self._free.pop(i)
                    break
        if found is not None:
            buf, event = found
            if event is not None:
                event.synchronize()
            return buf
        return _alloc_pinned(max(need, _bucket_bytes(floor)))

    def put(self, buf: torch.Tensor, event=None) -> None:
        with self._lock:
            self._free.append((buf, event))


@dataclass
class CooperativeStagingContext:
    """Result of the group preflight performed before source selection."""

    comm: object | None
    pool: _PinnedPool
    bucket_bytes: int


def prepare_cooperative_staging(
    local_eligible: bool,
    max_inflight_bytes: int = 512 << 20,
) -> CooperativeStagingContext:
    """Vote on cooperation before choosing a deterministic source iterator."""
    comm = _tp_comm()
    pool = _PinnedPool()
    bucket_bytes = min(_DEFAULT_BUCKET_BYTES, max(1 << 20, max_inflight_bytes))
    if comm is None:
        return CooperativeStagingContext(None, pool, bucket_bytes)

    eligible = 0
    if local_eligible:
        try:
            pool.put(_alloc_pinned(_bucket_bytes(bucket_bytes)))
            eligible = 1
        except (OSError, RuntimeError, TypeError, ValueError):
            eligible = 0
    try:
        if comm.all_reduce_max([1 - eligible])[0] != 0:
            comm = None
    except _COLLECTIVE_ERRORS as exc:
        logger.warning("Cooperative staging: preflight collective failed (%s); falling back locally.", exc)
        comm = None
    return CooperativeStagingContext(comm, pool, bucket_bytes)


def _stage_bucket(bucket: _Bucket, pool: _PinnedPool, cap: int) -> None:
    """Owner-side: pack all entries (fused cast) into one pinned buffer."""
    started = time.perf_counter()
    buf = pool.get(bucket.total, cap)
    bucket.pinned = buf
    for e in bucket.entries:
        view = buf[e.offset : e.offset + e.nbytes].view(e.out_dtype).view(e.src.shape)
        view.copy_(e.src)
    bucket.stage_s = time.perf_counter() - started


def cooperative_staging_weights_iterator(
    weights_iter: Iterator[tuple[str, torch.Tensor]],
    comm=None,
    bucket_bytes: int = _DEFAULT_BUCKET_BYTES,
    max_inflight_bytes: int = 512 << 20,
    target_dtypes: Mapping[str, torch.dtype] | None = None,
    default_dtype: torch.dtype | None = None,
    local_eligible: bool = True,
    context: CooperativeStagingContext | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stage the checkpoint cooperatively across ``comm``'s ranks.

    Yields ``(name, gpu_tensor)`` in the original order on every rank; the
    tensors live on ``comm.device`` and are valid until the next item is
    requested. Falls back to per-rank ``pinned_staging_weights_iterator``
    (CPU pinned tensors, unchanged semantics) when there is no usable group,
    when any rank votes not-eligible pre-flight, or group-wide after any
    rank's mid-stream failure; with ``local_eligible=False`` and no group it
    passes tensors through unstaged.

    ``local_eligible`` carries the caller's *runtime* staging conditions
    (CUDA present, pinnable memory, ...). They must be part of the vote, not
    a caller-side gate: a rank that skipped this call entirely while its
    peers entered the pre-flight collective would deadlock the group. Only
    rank-invariant (config-derived) conditions may gate the call itself.

    ``comm`` needs ``rank``/``world_size``/``device``/``all_reduce_max``/
    ``broadcast``; None resolves the vLLM TP group.
    """
    if context is not None:
        comm = context.comm
        pool = context.pool
        bucket_bytes = context.bucket_bytes
    else:
        # Backward-compatible lazy preflight for direct callers. The loader
        # prepares a context eagerly so it can retain multithread loading when
        # the group vetoes cooperation.
        if comm is None:
            context = prepare_cooperative_staging(local_eligible, bucket_bytes)
            comm = context.comm
            pool = context.pool
        else:
            pool = _PinnedPool()
            eligible = 0
            if local_eligible:
                try:
                    pool.put(_alloc_pinned(_bucket_bytes(bucket_bytes)))
                    eligible = 1
                except (OSError, RuntimeError, TypeError, ValueError):
                    eligible = 0
            try:
                if comm.all_reduce_max([1 - eligible])[0] != 0:
                    comm = None
            except _COLLECTIVE_ERRORS as exc:
                logger.warning("Cooperative staging: preflight collective failed (%s); falling back locally.", exc)
                comm = None

    src = iter(weights_iter)
    if comm is None:
        if not local_eligible:
            yield from src
            return
        yield from pinned_staging_weights_iterator(
            src,
            max_inflight_bytes=max_inflight_bytes,
            target_dtypes=target_dtypes,
            default_dtype=default_dtype,
        )
        return

    planner = _BucketPlanner(comm.world_size, bucket_bytes, target_dtypes, default_dtype)
    pending: list[_Bucket] = []
    stager = ThreadPoolExecutor(max_workers=_STAGE_WORKERS, thread_name_prefix="coop-staging")
    stats = {"own_bytes": 0, "source_s": 0.0, "stage_s": 0.0, "buckets": 0, "aborted": False}
    h2d_events: list[tuple[object, object]] = []
    exhausted = False
    source_error: BaseException | None = None

    def _admit(bucket: _Bucket) -> None:
        if bucket.owner == comm.rank:
            for e in bucket.entries:
                _prefault(e.src)
            bucket.future = stager.submit(_stage_bucket, bucket, pool, bucket_bytes)
        pending.append(bucket)

    def _local_fallback(unyielded: list[_Bucket]) -> Iterator[tuple[str, torch.Tensor]]:
        # Group-wide degrade: wait out in-flight stagings, then hand every
        # not-yet-yielded tensor plus the untouched stream to plain per-rank
        # pinned staging (correct, just not cooperative). ``planner`` may hold
        # an open bucket whose tensors have already been consumed from ``src``;
        # drain it explicitly or those weights would disappear on fallback.
        stager.shutdown(wait=True, cancel_futures=True)
        tail = planner.flush()
        if tail is not None:
            unyielded.append(tail)
        for b in unyielded:
            if b.pinned is not None:
                pool.put(b.pinned, b.h2d_done)
                b.pinned = None
        remaining = chain(((e.name, e.src) for b in unyielded for e in b.entries), src)
        return pinned_staging_weights_iterator(
            remaining,
            max_inflight_bytes=max_inflight_bytes,
            target_dtypes=target_dtypes,
            default_dtype=default_dtype,
        )

    def _fill_pending() -> None:
        # Plan ahead so owned buckets stage while transfers and consumer
        # copies of the current window are still in flight. Fill depth is a
        # pure function of the stream (never of timing): the window
        # composition must be identical on every rank.
        nonlocal exhausted, source_error
        while not exhausted and len(pending) < _WINDOW_BUCKETS:
            source_t0 = time.perf_counter()
            captured = _CapturedFailure()
            item = None
            with captured:
                item = next(src)
            stats["source_s"] += time.perf_counter() - source_t0
            if isinstance(captured.error, StopIteration):
                exhausted = True
                tail = planner.flush()
                if tail is not None:
                    _admit(tail)
                break
            if captured.error is not None:
                source_error = captured.error
                exhausted = True
                break
            assert item is not None
            for closed in planner.add(*item):
                _admit(closed)

    def _plan_signature(window: list[_Bucket]) -> tuple[int, int]:
        """Hash the exact collective schedule and tensor views for a window."""
        checksum = 0
        for bucket in window:
            checksum = zlib.crc32(f"B:{bucket.owner}:{bucket.total}\0".encode(), checksum)
            for entry in bucket.entries:
                descriptor = (
                    f"E:{entry.name}:{tuple(entry.src.shape)}:{entry.src.dtype}:"
                    f"{entry.out_dtype}:{entry.offset}:{entry.nbytes}\0"
                )
                checksum = zlib.crc32(descriptor.encode(), checksum)
        return sum(bucket.total for bucket in window), checksum

    # Transfers (owner H2D + broadcast) run on a dedicated stream; the
    # consumer's weight_loader copies stay on the default stream and only
    # wait on per-bucket events — so bucket k's transfers overlap bucket
    # k-1's consumption and bucket k+1's staging, instead of the whole
    # pipeline serializing through one stream sync per bucket.
    xfer = torch.cuda.Stream(device=comm.device) if comm.device.type == "cuda" else None

    try:
        _fill_pending()
        while True:
            # Every rank enters one agreement for every window, including an
            # empty terminal window. This keeps an early-EOF rank in lockstep
            # long enough to detect a peer with additional buckets.
            window = list(pending)
            pending.clear()
            if window:
                # Refill immediately so stagers keep working while this window
                # transfers and the previous one is consumed.
                _fill_pending()

            stage_err = 0
            for bucket in window:
                if bucket.future is not None:
                    future_error = bucket.future.exception()
                    if future_error is not None:
                        bucket.error = future_error
                    bucket.future = None
                    if bucket.error is not None:
                        logger.warning("Cooperative staging: local stage failed (%s)", bucket.error)
                        stage_err = 1
                try:
                    bucket.gpu = torch.empty(bucket.total, dtype=torch.uint8, device=comm.device)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    logger.warning("Cooperative staging: payload alloc failed (%s)", exc)
                    stage_err = 1

            # Agree on both errors and the exact collective schedule before
            # issuing any broadcast. MAX over positive and negative values
            # gives every rank the group min and max without another call.
            total, plan_hash = _plan_signature(window)
            bucket_count = len(window)
            try:
                agreed = comm.all_reduce_max(
                    [
                        stage_err,
                        int(source_error is not None),
                        bucket_count,
                        total,
                        plan_hash,
                        -bucket_count,
                        -total,
                        -plan_hash,
                    ]
                )
            except _COLLECTIVE_ERRORS as exc:
                logger.warning(
                    "Cooperative staging: schedule collective failed (%s); all ranks falling back locally.",
                    exc,
                )
                stats["aborted"] = True
                yield from _local_fallback(window + list(pending))
                return
            diverged = agreed[2] != -agreed[5] or agreed[3] != -agreed[6] or agreed[4] != -agreed[7]
            if agreed[1] != 0:
                stats["aborted"] = True
                stager.shutdown(wait=True, cancel_futures=True)
                if source_error is not None:
                    raise source_error
                raise RuntimeError("A tensor-parallel peer failed while reading the checkpoint")
            if diverged:
                logger.warning(
                    "Cooperative staging: bucket plan diverged across ranks (different source "
                    "order, metadata, or length); falling back to per-rank staging."
                )
            if agreed[0] != 0 or diverged:
                stats["aborted"] = True
                yield from _local_fallback(window + list(pending))
                return
            if not window:
                return

            # Issue the whole window's transfers back-to-back on the
            # transfer stream; consumers gate on per-bucket events only.
            collective_error = None
            with torch.cuda.stream(xfer) if xfer is not None else nullcontext():
                for bucket in window:
                    if bucket.owner == comm.rank:
                        if xfer is not None:
                            bucket.h2d_start = torch.cuda.Event(enable_timing=True)
                            bucket.h2d_start.record()
                        bucket.gpu.copy_(bucket.pinned[: bucket.total], non_blocking=True)
                        if xfer is not None:
                            bucket.h2d_done = torch.cuda.Event(enable_timing=True)
                            bucket.h2d_done.record()
                    try:
                        comm.broadcast(bucket.gpu, src=bucket.owner)
                    except _COLLECTIVE_ERRORS as exc:
                        collective_error = exc
                        break
                    if xfer is not None:
                        # The payload was allocated under the default stream
                        # but written here — tell the caching allocator
                        # before the reference can drop. (Consumer reads on
                        # the default/allocation stream are auto-tracked.)
                        bucket.gpu.record_stream(xfer)
                        bucket.ready = torch.cuda.Event()
                        bucket.ready.record()

            if collective_error is not None:
                logger.warning(
                    "Cooperative staging: broadcast collective failed (%s); all ranks falling back locally.",
                    collective_error,
                )
                stats["aborted"] = True
                yield from _local_fallback(window + list(pending))
                return

            for bucket in window:
                if bucket.ready is not None:
                    torch.cuda.current_stream().wait_event(bucket.ready)
                gpu_buf = bucket.gpu
                for e in bucket.entries:
                    yield e.name, gpu_buf[e.offset : e.offset + e.nbytes].view(e.out_dtype).view(e.src.shape)
                bucket.gpu = None
                # Repool the pinned buffer; its H2D may still be in flight —
                # the event travels with the buffer and is waited on REUSE,
                # keeping the sync off the consumer path.
                if bucket.pinned is not None:
                    stats["own_bytes"] += bucket.total
                    stats["stage_s"] += bucket.stage_s
                    if bucket.h2d_start is not None and bucket.h2d_done is not None:
                        h2d_events.append((bucket.h2d_start, bucket.h2d_done))
                    pool.put(bucket.pinned, bucket.h2d_done)
                    bucket.pinned = None
                stats["buckets"] += 1
    finally:
        stager.shutdown(wait=True, cancel_futures=True)
        if stats["buckets"] and not stats["aborted"]:
            h2d_s = 0.0
            if h2d_events:
                h2d_events[-1][1].synchronize()
                h2d_s = sum(start.elapsed_time(done) for start, done in h2d_events) / 1000
            h2d_bw = stats["own_bytes"] / (1 << 30) / h2d_s if h2d_s > 0 else float("inf")
            logger.info(
                "Cooperative staging: %d buckets, %.2f GiB staged by this rank (1/%d of the stream; "
                "source iterator %.2f s; read+stage copy %.2f s; H2D %.2f s, %.1f GiB/s).",
                stats["buckets"],
                stats["own_bytes"] / (1 << 30),
                comm.world_size,
                stats["source_s"],
                stats["stage_s"],
                h2d_s,
                h2d_bw,
            )
