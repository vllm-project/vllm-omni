# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Best-effort page-cache prewarm of model weights (RFC #4896).

The multiprocessing executor starts readers in its parent before spawning
workers, overlapping otherwise-idle process imports. The first worker to reach
model initialization performs a group handoff: speculative readers stop and
join before any pipeline demand I/O begins. Child-local prewarm remains as a
fallback for executor implementations without that parent lifecycle.

Cap-then-stop: parent prewarm follows demand order (pipeline components,
then DiT shards); the child-local fallback reads DiT shards first because its
pipeline components are already being loaded concurrently. Reading STOPS once
the RAM budget is spent. Prewarming past available RAM is worse than useless —
the kernel evicts the head of the model while reading the tail. The budget
keeps a headroom margin for the page cache and for the runtime's later
non-evictable allocations.

Page-cache-resident files are detected with sampled ``mincore`` probes and
skipped without a userspace copy. The parent handoff stops before worker model
initialization; the child-local fallback stops when demand weight loading
begins. In both lifecycles, broad speculative reads cannot overlap targeted
staging faults.

Everything here is advisory: any failure degrades to no-prewarm, never to a
load failure.
"""

import ctypes
import mmap
import multiprocessing as mp
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

from vllm.logger import init_logger

logger = init_logger(__name__)

_CHUNK = 16 << 20  # sequential 16 MiB reads per stream
_DEFAULT_READERS = 2  # one buffered stream tops out ~2 GiB/s (copy + cache-insert); two saturate a local NVMe
_MIN_WORTHWHILE = 1 << 30  # below 1 GiB of budget, skip entirely
# Non-evictable anonymous memory the process grows AFTER the budget snapshot
# (python + CUDA context + pinned pools, measured ~10-15 GiB): reserve it up
# front so the prewarmed cache keeps a stable cushion once the runtime is up.
_RUNTIME_HEADROOM = 8 << 30

_libc = None
if sys.platform == "linux":
    try:
        _libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        _libc = None


def _is_file_fully_resident(path: str) -> bool:
    """Whether sampled regions of ``path`` are already in the page cache.

    A full ``mincore`` walk of a 117 GiB checkpoint added ~13 seconds to a
    cold start. Sample one page per prewarm read chunk instead: sequentially
    populated files still classify warm, while an occasional false positive
    is harmless because demand loading faults any missing pages normally.
    """
    if _libc is None:
        return False
    try:
        size = os.path.getsize(path)
        if size == 0:
            return True
        with open(path, "rb", buffering=0) as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
            probe = ctypes.c_char.from_buffer(mapping)
            address = ctypes.addressof(probe)
            residency = ctypes.c_ubyte()
            for offset in range(0, size, _CHUNK):
                if (
                    _libc.mincore(
                        ctypes.c_void_p(address + offset),
                        ctypes.c_size_t(1),
                        ctypes.byref(residency),
                    )
                    != 0
                    or not residency.value & 1
                ):
                    del probe
                    return False
            del probe
            return True
    except (OSError, ValueError, BufferError):
        return False


def _read_order_key(relpath: str) -> tuple:
    """DiT shards first, then everything else, natural-sorted within each group.

    Keyed on the path RELATIVE to the model dir (an ancestor directory named
    e.g. ``transformers-models`` must not turn every file into a DiT hit).
    load_weights consumes the transformer components (diffusers layout puts
    them under ``transformer*/``); the other components (text_encoder, vae)
    are read by ``from_pretrained`` during the model build itself — i.e.
    inside the prewarm window — so spending the window's budget on them buys
    nothing. Measured: alphabetical order (text_encoder first) gained ~0;
    DiT-first ordering is what makes the window count. Layouts without a
    ``transformer*`` component dir degrade to plain natural order.
    """
    first = relpath.split(os.sep, 1)[0]
    is_dit = 0 if first.startswith("transformer") else 1
    return (is_dit, [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", relpath)])


def _parent_read_order_key(relpath: str) -> tuple:
    """Demand-loaded components first, then DiT shards.

    Parent prewarm ends before pipeline construction, so its idle startup
    window should prepare the text encoder/VAE pages needed first. Local
    fallback prewarm uses the inverse order because those components are
    already being loaded concurrently in that older lifecycle.
    """
    first = relpath.split(os.sep, 1)[0]
    is_dit = 1 if first.startswith("transformer") else 0
    return (is_dit, [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", relpath)])


def _resolve_local_dir(
    model: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> str | None:
    """Local directory holding the requested weights, or None.

    A HF repo id resolves through the local hub cache (offline — no
    network), using the same revision and cache directory as demand loading.
    A previously downloaded model on a restarted node is exactly the
    cold-cache case prewarm exists for. An uncached repo id returns None; the
    download that follows populates the page cache by itself.
    """
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            model,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except (ImportError, OSError, ValueError):
        return None


def _host_available_bytes() -> int | None:
    """Host-view available RAM via psutil (a vllm dependency)."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        return None


# (limit_path, used_path) pairs, cgroup v2 then v1. psutil (like vllm's own
# memory helpers) reads /proc/meminfo and therefore reports the HOST in
# containers (observed: a 126 GiB slice reporting the 1 TiB host), while
# page cache is charged to the cgroup — sizing the prewarm off the host
# figure over-fills the cgroup and reclaims the very pages just warmed.
_CGROUP_MEM_PATHS = (
    ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
    ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
)
_CGROUP_UNLIMITED = 1 << 60  # v1 reports "no limit" as ~2**63


def _cgroup_available_bytes() -> int | None:
    """Memory left under the cgroup limit, or None when unlimited/absent."""
    for limit_path, used_path in _CGROUP_MEM_PATHS:
        try:
            with open(limit_path) as f:
                raw = f.read().strip()
            if raw == "max":
                return None
            limit = int(raw)
            if limit >= _CGROUP_UNLIMITED:
                return None
            with open(used_path) as f:
                used = int(f.read().strip())
            return max(0, limit - used)
        except (OSError, ValueError):
            continue
    return None


def _available_ram_bytes() -> int | None:
    host = _host_available_bytes()
    cgroup = _cgroup_available_bytes()
    if host is None:
        return cgroup
    if cgroup is None:
        return host
    return min(host, cgroup)


def _prewarm_budget_bytes(total_bytes: int) -> int:
    """RAM the prewarm may spend: what's available minus headroom."""
    available = _available_ram_bytes()
    if available is None:
        return 0
    headroom = max(int(0.15 * total_bytes), 8 << 30) + _RUNTIME_HEADROOM
    return max(0, available - headroom)


class _Budget:
    """Byte budget shared across reader threads (a static per-reader split
    strands half the budget when the DiT shards land in one partition)."""

    def __init__(self, cap: int):
        self._left = cap
        self._lock = threading.Lock()

    def take(self, nbytes: int) -> bool:
        with self._lock:
            if self._left <= 0:
                return False
            self._left -= nbytes
            return True


def _read_files(
    files: list[str],
    budget: "_Budget",
    stop_event: threading.Event | None = None,
) -> tuple[int, int]:
    """Sequentially read files until the budget is spent or loading starts."""
    done_bytes = 0
    done_files = 0
    buf = bytearray(_CHUNK)
    for path in files:
        if stop_event is not None and stop_event.is_set():
            break
        try:
            with open(path, "rb", buffering=0) as f:
                while stop_event is None or not stop_event.is_set():
                    n = f.readinto(buf)
                    if not n:
                        done_files += 1
                        break
                    done_bytes += n
                    if not budget.take(n):
                        return done_bytes, done_files
        except OSError:
            continue
    return done_bytes, done_files


def _is_prewarm_rank() -> bool:
    """Only TP rank 0 prewarms: the page cache is node-shared, so N ranks
    reading the same files buys nothing and their interleaved streams break
    each other's sequential readahead. (Diffusion TP groups are intra-node
    in practice; a multi-node TP rank skipping here loses only prewarm,
    never correctness.)"""
    try:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return True
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0
    except (ImportError, AttributeError, AssertionError, RuntimeError, ValueError):
        return True


class WeightsPrewarmHandle:
    """Cancelable prewarm task handed off when demand loading begins."""

    def __init__(self, thread: threading.Thread, stop_event: threading.Event):
        self._thread = thread
        self._stop_event = stop_event

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()


_parent_prewarm_active: ContextVar[bool] = ContextVar("parent_weights_prewarm_active", default=False)


def start_weights_prewarm(
    model: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> WeightsPrewarmHandle | None:
    """Start local prewarm unless a parent process already owns the window."""
    if _parent_prewarm_active.get():
        return None
    return _start_weights_prewarm(model, revision=revision, cache_dir=cache_dir)


def _start_weights_prewarm(
    model: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    read_order_key: Callable[[str], tuple] = _read_order_key,
) -> WeightsPrewarmHandle | None:
    """Kick off daemon threads reading the model's safetensors into the page
    cache. ``model`` may be a local directory or a HF repo id resolved from
    the requested revision and cache directory. Returns a cancelable task
    handle, or None when skipped (unresolvable model, warm files, non-Linux,
    or low budget)."""
    if not _is_prewarm_rank():
        return None
    try:
        model_dir = _resolve_local_dir(model, revision=revision, cache_dir=cache_dir)
        if model_dir is None:
            return None
        files = []
        for root, _, names in os.walk(model_dir):
            files.extend(os.path.join(root, n) for n in names if n.endswith(".safetensors"))
        files.sort(key=lambda p: read_order_key(os.path.relpath(p, model_dir)))
        total = sum(os.path.getsize(f) for f in files)
        if not total:
            return None
        resident = [f for f in files if _is_file_fully_resident(f)]
        if resident:
            resident_set = set(resident)
            files = [f for f in files if f not in resident_set]
            logger.info(
                "Weights prewarm skipped %d page-cache-resident safetensors (%.1f GiB).",
                len(resident),
                sum(os.path.getsize(f) for f in resident) / (1 << 30),
            )
        if not files:
            return None
        remaining = sum(os.path.getsize(f) for f in files)
        cap = min(remaining, _prewarm_budget_bytes(total))
        if cap <= 0 or cap < _MIN_WORTHWHILE:
            logger.debug("Weights prewarm skipped: budget %.1f GiB below threshold.", cap / (1 << 30))
            return None
    except OSError:
        return None

    stop_event = threading.Event()

    def _run():
        try:
            t0 = time.perf_counter()
            budget = _Budget(cap)
            # Round-robin the (DiT-first) file order across readers so all
            # streams work the head of the load order; the budget is shared,
            # so one partition running dry strands nothing.
            readers = _DEFAULT_READERS
            parts = [files[i::readers] for i in range(readers)]
            results = [(0, 0)] * len(parts)
            workers = []
            for i, part in enumerate(parts):

                def _one(i=i, part=part):
                    results[i] = _read_files(part, budget, stop_event)

                w = threading.Thread(target=_one, name=f"weights-prewarm-{i}", daemon=True)
                w.start()
                workers.append(w)
            for w in workers:
                w.join()
            done_bytes = sum(b for b, _ in results)
            done_files = sum(f for _, f in results)
            dt = time.perf_counter() - t0
            logger.info(
                "Weights prewarm%s: %.1f/%.1f GiB (%d/%d files) into page cache in %.1f s (%.1f GiB/s).",
                " stopped at loader handoff" if stop_event.is_set() else "",
                done_bytes / (1 << 30),
                remaining / (1 << 30),
                done_files,
                len(files),
                dt,
                done_bytes / (1 << 30) / dt if dt > 0 else float("inf"),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Weights prewarm aborted: %s", exc)

    logger.info(
        "Weights prewarm started: %d safetensors, %.1f GiB (budget %.1f GiB).",
        len(files),
        remaining / (1 << 30),
        cap / (1 << 30),
    )
    thread = threading.Thread(target=_run, name="weights-prewarm", daemon=True)
    handle = WeightsPrewarmHandle(thread, stop_event)
    thread.start()
    return handle


class _ProcessEvent(Protocol):
    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


@dataclass(frozen=True)
class WeightsPrewarmHandoff:
    """Opaque worker token for a parent-owned prewarm window."""

    _requested: _ProcessEvent | None = None
    _completed: _ProcessEvent | None = None

    def wait(self) -> None:
        if self._requested is not None and self._completed is not None:
            self._requested.set()
            self._completed.wait()


@contextmanager
def parent_weights_prewarm(
    model: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> Iterator[WeightsPrewarmHandoff]:
    """Prewarm during spawned-worker startup, then hand off before model init.

    The returned token is passed unchanged to every worker. The first worker
    reaching the handoff stops and joins the parent's speculative readers;
    all workers then cross together, with worker-local prewarm suppressed.
    """
    handle = _start_weights_prewarm(
        model,
        revision=revision,
        cache_dir=cache_dir,
        read_order_key=_parent_read_order_key,
    )
    if handle is None:
        # The parent already performed all resolution, residency, and budget
        # checks. Suppress an identical scan in rank 0 even when no reads were
        # needed or possible.
        yield WeightsPrewarmHandoff()
        return

    context = mp.get_context("spawn")
    requested = context.Event()
    completed = context.Event()
    handoff = WeightsPrewarmHandoff(requested, completed)

    def _coordinate_handoff() -> None:
        requested.wait()
        try:
            handle.stop()
            handle.join()
        finally:
            completed.set()
        logger.info("Parent weights prewarm handed off before worker model initialization.")

    coordinator = threading.Thread(
        target=_coordinate_handoff,
        name="weights-prewarm-handoff",
        daemon=True,
    )
    coordinator.start()
    try:
        yield handoff
    finally:
        requested.set()
        coordinator.join()


@contextmanager
def use_parent_weights_prewarm(handoff: WeightsPrewarmHandoff | None) -> Iterator[None]:
    """Wait for parent handoff and suppress duplicate local prewarm."""
    if handoff is None:
        yield
        return

    handoff.wait()
    token = _parent_prewarm_active.set(True)
    try:
        yield
    finally:
        _parent_prewarm_active.reset(token)
