# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded pinned-memory staging for diffusion weight loading (RFC #4896, item 1).

Wraps a ``(name, tensor)`` weights iterator: staging threads copy each CPU
tensor into page-locked (pinned) memory so the consumer's ``param.copy_()``
takes the fast DMA path instead of the pageable-copy path, and staging
overlaps with the H2D copies of previously yielded tensors.

When ``target_dtypes`` maps a tensor's name to a different floating-point
dtype (e.g. an fp32 checkpoint loaded into bf16 params), the dtype cast is
fused into the staging copy. This is required, not just nice: a
dtype-converting H2D ``copy_`` falls off the pinned fast path entirely
(measured as slow as a pageable copy), so without the fused cast pinned
staging buys nothing for mismatched checkpoints. Casting while staging also
halves the staged bytes and H2D traffic for fp32->bf16.

Pinned buffers come from an internal power-of-two size pool and are recycled
once the consumer moves past a tensor, so each size class pays ``cudaHostAlloc``
only a handful of times per load rather than once per tensor — host allocation
can cost >100 ms per 100 MiB, far more than the copy. The in-flight byte budget
bounds read-ahead.

Consumer contract: a yielded tensor is only valid until the next item is
requested from the iterator — its backing buffer is then recycled. This
matches ``load_weights`` implementations that copy each tensor into model
parameters before advancing (the standard vLLM weight-loader pattern).
Consumers that stash raw checkpoint tensors across iterations must not
enable pinned staging.

Falls back to passing tensors through unpinned on the first allocation
failure (e.g. ``RLIMIT_MEMLOCK`` in containers); loading then behaves as
before (pass-through tensors are never recycled).
"""

import ctypes
import os
import queue
import sys
import threading
import time
from collections.abc import Generator, Iterator, Mapping
from types import TracebackType

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_SENTINEL = object()


class _ForwardThreadFailure:
    """Forward a producer failure through its output queue.

    ``threading.Thread`` otherwise reports uncaught failures only through the
    process-wide excepthook, leaving this iterator's consumer blocked.
    """

    def __init__(self, out_q: queue.Queue[object]):
        self._out_q = out_q

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, traceback
        if exc is None:
            return False
        self._out_q.put(exc)
        return True


# madvise(MADV_WILLNEED): async readahead of the mmap'd source pages, so staging
# one tensor overlaps the disk read of the next — the cold-cache win (~2.3x; a
# no-op on a warm cache). Not MADV_POPULATE_READ, which faults synchronously and
# serializes the reads (measured slower than no staging at all on a cold cache).
_MADV_WILLNEED = 3
_PAGE = 4096
# madvise via the process's already-linked libc (dlopen(NULL)): no library
# file lookup, no ldconfig subprocess. There is no stdlib equivalent for
# foreign memory -- mmap.madvise() only covers mmap objects Python itself
# created, and the source tensors here are mmap'd inside safetensors.
_libc = None
if sys.platform == "linux":
    try:
        _libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        _libc = None


def _prefault(tensor: torch.Tensor) -> None:
    """Kick off async readahead of the tensor's backing pages (no-op off Linux)."""
    if _libc is None:
        return
    nbytes = tensor.numel() * tensor.element_size()
    addr = tensor.data_ptr()
    start = addr - (addr % _PAGE)
    _libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(nbytes + addr - start), _MADV_WILLNEED)


def _alloc_pinned(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


# Only these source dtypes are eligible for the fused cast: quantized formats
# (float8 etc.) must reach load_weights bit-exact for dequantization.
_CASTABLE = (torch.float32, torch.float16, torch.bfloat16)

# Tensors below this stage nothing: a sub-64 KiB H2D copy is latency-bound,
# so pinned DMA buys nothing, while the 1 MiB pool floor over-allocates it up
# to 256x (a 17705-tensor quant checkpoint measured ~2000 concurrent 1 MiB
# buffers ~= 2 GiB transient pinned for kilobytes of live data). Passing tiny
# tensors through also skips their fused cast; the consumer's dtype-converting
# copy_ on such tensors moves negligible bytes.
_MIN_STAGE_BYTES = 64 << 10


def _resolve_out_dtype(
    name: str,
    src_dtype: torch.dtype,
    target_dtypes: Mapping[str, torch.dtype] | None,
    default_dtype: torch.dtype | None,
) -> torch.dtype:
    """Destination dtype for the fused staging cast.

    Exact-name matches in ``target_dtypes`` win (protecting params that
    intentionally keep e.g. fp32); tensors whose checkpoint names don't match
    any param (renamed/fused weights: q/k/v -> qkv etc.) fall back to
    ``default_dtype`` — the dtype the model was constructed under.
    """
    if src_dtype not in _CASTABLE:
        return src_dtype
    want = target_dtypes.get(name) if target_dtypes is not None else None
    if want is None:
        want = default_dtype
    if want is not None and want is not src_dtype and want in _CASTABLE:
        return want
    return src_dtype


def _usable_cpu_count() -> int:
    """CPUs this process may actually use: honors cgroup/taskset affinity
    (os.cpu_count reports HOST cores and over-threads limited containers)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux
        return os.cpu_count() or 8


def _auto_staging_threads() -> int:
    """Copy-thread count for ``num_staging_threads=0``.

    ``torch.copy_`` uses intra-op workers of its own, and the source loader
    and page-cache prewarm may also be active. Cap producers at four to avoid
    the nested oversubscription observed on 32-vCPU weight-load hosts.
    """
    return max(2, min(4, _usable_cpu_count() // 4))


def _bucket_bytes(nbytes: int) -> int:
    """Round a request up to a power-of-two bucket (min 1 MiB).

    cudaHostAlloc synchronizes with in-flight device work, so an allocation
    issued while the consumer's H2D copies are running can stall for over
    100 ms regardless of size. Bucketing collapses the thousands of distinct
    tensor sizes in a checkpoint into ~a dozen size classes so the pool hit
    rate approaches 100% after the first few tensors of each class.
    """
    size = 1 << 20
    while size < nbytes:
        size <<= 1
    return size


class _PinnedBufferPool:
    """Power-of-two pinned buffers with a bounded free-list.

    In-flight buffers may temporarily exceed the cache cap, but once the
    consumer returns them only ``max_cached_bytes`` remains page-locked. This
    prevents one-off tensor size classes from accumulating for the lifetime
    of a large component load.
    """

    def __init__(self, max_cached_bytes: int):
        self._max_cached_bytes = max_cached_bytes
        self._free: dict[int, list[torch.Tensor]] = {}
        self._lock = threading.Lock()
        self.cached_bytes = 0
        self.reserved_bytes = 0
        self.peak_reserved_bytes = 0
        self.allocs = 0
        self.reuses = 0
        self.drops = 0

    def get(self, nbytes: int) -> torch.Tensor:
        bucket = _bucket_bytes(nbytes)
        with self._lock:
            free = self._free.get(bucket)
            if free:
                self.reuses += 1
                self.cached_bytes -= bucket
                return free.pop()
        buf = _alloc_pinned(bucket)
        with self._lock:
            self.allocs += 1
            self.reserved_bytes += bucket
            self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)
        return buf

    def put(self, buf: torch.Tensor) -> None:
        size = buf.numel()
        with self._lock:
            # The read-ahead budget may be smaller than the 1 MiB pool floor,
            # or a legitimate tensor may exceed the whole budget. Always allow
            # one such buffer; otherwise every item reallocates it forever.
            effective_cap = max(self._max_cached_bytes, size)
            if self.cached_bytes + size <= effective_cap:
                self._free.setdefault(size, []).append(buf)
                self.cached_bytes += size
                return
            self.reserved_bytes -= size
            self.drops += 1


def pinned_staging_weights_iterator(
    weights_iter: Iterator[tuple[str, torch.Tensor]],
    max_inflight_bytes: int = 512 << 20,
    num_staging_threads: int = 0,
    target_dtypes: Mapping[str, torch.dtype] | None = None,
    default_dtype: torch.dtype | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stage CPU tensors into pooled pinned memory on producer threads.

    Yields the same ``(name, tensor)`` pairs as ``weights_iter`` (order may
    interleave across staging threads); tensors stay on CPU so downstream
    ``load_weights`` semantics (TP narrowing, QKV fusion) are unchanged.
    See the module docstring for the buffer-recycling consumer contract.

    Mismatched float checkpoints are cast during the staging copy (see the
    module docstring): ``target_dtypes`` (param name -> dtype) by exact name,
    ``default_dtype`` for names it doesn't cover (renamed/fused weights).
    Non-float and quantized tensors always stage dtype-unchanged.

    ``max_inflight_bytes`` bounds read-ahead in checkpoint bytes; the actual
    pinned peak can reach ~2x that (power-of-two buffer buckets, 1 MiB floor).
    The 512 MiB default measured as sufficient (2 GiB gained nothing).
    """
    if num_staging_threads <= 0:  # 0 = auto; and 0 producers would strand the consumer
        num_staging_threads = _auto_staging_threads()
    out_q: queue.Queue[object] = queue.Queue()
    lock = threading.Condition()
    state = {
        "inflight": 0,
        "pin_failed": False,
        "stopped": False,
        "producers": num_staging_threads,
    }
    stats = {
        "bytes": 0,
        "source_s": 0.0,
        "stage_s": 0.0,
        "consumer_s": 0.0,
        "casts": 0,
    }
    src_lock = threading.Lock()
    src = iter(weights_iter)
    # Keep up to two read-ahead windows in the free-list. One window caused
    # recurring size classes to churn through cudaHostAlloc on Wan2.2 (60+
    # drops/component); two retained every hot class with only one drop while
    # preserving the same measured process RSS. Honor the 1 MiB bucket floor
    # when callers choose a smaller test or memory-constrained read-ahead
    # budget, otherwise two sub-MiB in-flight tensors can only cache one buffer.
    pool = _PinnedBufferPool(max(2 * max_inflight_bytes, 2 << 20))

    def _next_item():
        # Generators are not thread-safe; serialize next() across staging threads.
        with src_lock:
            t0 = time.perf_counter()
            try:
                return next(src)
            except StopIteration:
                return _SENTINEL
            finally:
                stats["source_s"] += time.perf_counter() - t0

    def _stage(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor, torch.Tensor | None]:
        """Copy ``tensor`` into a pooled pinned buffer; returns the backing buffer for recycling."""
        if (
            state["pin_failed"]
            or tensor.device.type != "cpu"
            or tensor.is_pinned()
            or tensor.numel() * tensor.element_size() < _MIN_STAGE_BYTES
        ):
            return name, tensor, None
        buf = None
        try:
            t0 = time.perf_counter()
            out_dtype = _resolve_out_dtype(name, tensor.dtype, target_dtypes, default_dtype)
            nbytes = tensor.numel() * out_dtype.itemsize
            _prefault(tensor)
            buf = pool.get(nbytes)
            staged = buf[:nbytes].view(out_dtype).view(tensor.shape)
            # torch.copy_, not a single-core memcpy: its intra-op parallelism holds
            # throughput while the consumer's concurrent H2D DMA saturates memory
            # bandwidth. When out_dtype differs it also performs the cast in the
            # same pass (the rounding param.copy_ would otherwise do later).
            staged.copy_(tensor)
            with lock:
                stats["bytes"] += nbytes
                stats["stage_s"] += time.perf_counter() - t0
                if out_dtype is not tensor.dtype:
                    stats["casts"] += 1
            return name, staged, buf
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            # Expected staging failures (pinned alloc under RLIMIT_MEMLOCK or
            # an invalid view/copy combination) degrade to pass-through and
            # never abort the load. Loading correctness does not depend on staging.
            if buf is not None:
                pool.put(buf)
            state["pin_failed"] = True
            logger.warning(
                "Pinned staging disabled: staging failed (%s); falling back to pageable copies.",
                exc,
            )
            return name, tensor, None

    def _producer():
        try:
            with _ForwardThreadFailure(out_q):
                while True:
                    if state["stopped"]:
                        break
                    item = _next_item()
                    if item is _SENTINEL:
                        break
                    name, tensor = item
                    nbytes = tensor.numel() * tensor.element_size() if tensor.device.type == "cpu" else 0
                    with lock:
                        # Admit an oversized tensor only when nothing else is in flight,
                        # so a tensor larger than the whole budget cannot deadlock.
                        while (
                            not state["stopped"]
                            and state["inflight"] > 0
                            and state["inflight"] + nbytes > max_inflight_bytes
                        ):
                            lock.wait(timeout=0.1)
                        if state["stopped"]:
                            break
                        state["inflight"] += nbytes
                    out_q.put((*_stage(name, tensor), nbytes))
        finally:
            with lock:
                state["producers"] -= 1
                if state["producers"] == 0:
                    out_q.put(_SENTINEL)

    # Tag thread names per iterator instance so overlapping loads stay distinguishable.
    uid = f"{id(out_q):x}"
    threads = [
        threading.Thread(target=_producer, name=f"pinned-staging-{uid}-{i}", daemon=True)
        for i in range(num_staging_threads)
    ]
    t_start = time.perf_counter()
    for t in threads:
        t.start()

    try:
        while True:
            item = out_q.get()
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            name, tensor, buf, nbytes = item
            consumer_t0 = time.perf_counter()
            yield name, tensor
            # Control returned: the consumer has finished with `tensor`
            # (normally its blocking H2D/weight_loader copy), so its buffer can
            # be reused. Return it before releasing the read-ahead budget: a
            # woken producer can then reuse this buffer instead of racing its
            # return and issuing another cudaHostAlloc.
            stats["consumer_s"] += time.perf_counter() - consumer_t0
            if buf is not None:
                pool.put(buf)
            with lock:
                state["inflight"] -= nbytes
                lock.notify_all()
    finally:
        with lock:
            state["stopped"] = True
            lock.notify_all()
        # Shared deadline: a producer parked inside next(src) on a slow read
        # can't observe `stopped` until the read returns, so cap total close
        # latency instead of paying the timeout once per thread.
        deadline = time.monotonic() + 5.0
        for t in threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
        if stats["bytes"]:
            wall = time.perf_counter() - t_start
            stage_bw = stats["bytes"] / (1 << 30) / stats["stage_s"] if stats["stage_s"] > 0 else float("inf")
            logger.info(
                "Pinned staging: %.2f GiB in %.2f s wall (source iterator %.2f s; read+stage copy "
                "%.2f s, %.1f GiB/s; consumer/H2D %.2f s; %d producers; %d buffer allocs, %d reuses, "
                "%d drops, %.2f GiB peak pinned, %d dtype casts)",
                stats["bytes"] / (1 << 30),
                wall,
                stats["source_s"],
                stats["stage_s"],
                stage_bw,
                stats["consumer_s"],
                num_staging_threads,
                pool.allocs,
                pool.reuses,
                pool.drops,
                pool.peak_reserved_bytes / (1 << 30),
                stats["casts"],
            )
