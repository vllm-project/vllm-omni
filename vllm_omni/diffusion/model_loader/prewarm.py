# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Best-effort page-cache prewarm of model weights (RFC #4896).

The multiprocessing executor starts readers in its parent before spawning
workers, overlapping otherwise-idle process imports. Parent prewarm runs in
daemon threads and populates the page cache with chunked ``readahead`` /
``posix_fadvise(WILLNEED)`` plus an immediate copy unless ``mincore`` already
shows the chunk resident. Non-Linux copies only. Workers reaching model
initialization no longer kill the readers. Demand loading yields individual
safetensors; remaining copies pause only while the demanded file still needs
the disk — a resident component leaves the disk idle, so later shards keep
filling (hints at most otherwise). Copying resumes after that component's
iterator finishes. The parent context still stops all readers once workers
are ready. Child-local prewarm remains as a fallback for executor
implementations without that parent lifecycle.

Cap-then-stop: parent prewarm reads DiT shards first and pipeline components
last, because it now runs through pipeline construction: the components are
being opened by ``from_pretrained`` at that moment, so prefetching them would
compete with the demand I/O instead of hiding it. The child-local fallback
keeps the same order. Reading STOPS once
the RAM budget is spent. Prewarming past available RAM is worse than useless —
the kernel evicts the head of the model while reading the tail. The budget
keeps a headroom margin for the page cache and for the runtime's later
non-evictable allocations.

Page-cache-resident files are detected with sampled ``mincore`` probes and
skipped without a userspace copy. The parent context joins after workers
report ready. Demand paths yield the file they are about to open so
speculative reads do not overlap that file's staging faults.

Everything here is advisory: any failure degrades to no-prewarm, never to a
load failure.
"""

from __future__ import annotations

import ctypes
import mmap
import multiprocessing as mp
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from queue import Empty
from typing import Protocol

import regex as re
from vllm.logger import init_logger

logger = init_logger(__name__)

_CHUNK = 16 << 20  # sequential 16 MiB reads per stream
_DEFAULT_READERS = 2  # one buffered stream tops out ~2 GiB/s (copy + cache-insert); two saturate a local NVMe
_MIN_WORTHWHILE = 1 << 30  # below 1 GiB of budget, skip entirely
# Non-evictable anonymous memory the process grows AFTER the budget snapshot
# (python + CUDA context + pinned pools, measured ~10-15 GiB): reserve it up
# front so the prewarmed cache keeps a stable cushion once the runtime is up.
_RUNTIME_HEADROOM = 8 << 30
_PREWARM_STOP_TIMEOUT_S = 5.0
_POSIX_FADV_WILLNEED = 3
_MINCORE_STRIDE = 2 << 20  # sample residency every 2 MiB inside a chunk

_libc = None
if sys.platform == "linux":
    try:
        _libc = ctypes.CDLL(None, use_errno=True)
        if hasattr(_libc, "readahead"):
            _libc.readahead.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_size_t]
            _libc.readahead.restype = ctypes.c_ssize_t
        if hasattr(_libc, "posix_fadvise"):
            _libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
            _libc.posix_fadvise.restype = ctypes.c_int
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
    """DiT shards first, then everything else.

    Parent prewarm now runs *through* pipeline construction rather than
    ending before it, so filling the text encoder/VAE pages first would put
    speculative reads on the exact files ``from_pretrained`` is opening at
    that moment. Measured on Wan2.2 TP=2 cold (g7e.12xlarge): component-first
    order stretched pipeline construction to 17.8 s and finished the load in
    56.0 s, while DiT-first kept construction at 5.3 s and finished in 38.2 s.
    The DiT shards are only demanded later, by the weight loader, which is the
    overlap the spawn window exists for. Same natural sort as
    :func:`_read_order_key`; layouts without a ``transformer*`` dir degrade to
    plain natural order.
    """
    return _read_order_key(relpath)


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
        from vllm_omni.transformers_utils.repo_utils import hf_api

        return hf_api().snapshot_download(
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


def _advise_range(fd: int, offset: int, length: int) -> bool:
    """Submit kernel readahead for ``[offset, offset+length)``. Returns whether a hint was issued."""
    if _libc is None:
        return False
    try:
        if _libc.readahead(fd, offset, length) == 0:
            return True
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    try:
        if _libc.posix_fadvise(fd, offset, length, _POSIX_FADV_WILLNEED) == 0:
            return True
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return False


def _range_resident(mapping: mmap.mmap, offset: int, length: int) -> bool:
    """Whether sampled pages in ``[offset, offset+length)`` are in the page cache."""
    if _libc is None or length <= 0:
        return False
    page = mmap.PAGESIZE
    try:
        probe = ctypes.c_char.from_buffer(mapping)
        base = ctypes.addressof(probe)
        vec = ctypes.c_ubyte()
        end = offset + length
        off = offset & ~(page - 1)
        last = (end - 1) & ~(page - 1)
        while off <= last:
            if (
                _libc.mincore(ctypes.c_void_p(base + off), ctypes.c_size_t(page), ctypes.byref(vec)) != 0
                or not vec.value & 1
            ):
                del probe
                return False
            if off == last:
                break
            off = min(last, off + _MINCORE_STRIDE)
        del probe
        return True
    except (OSError, ValueError, BufferError, TypeError):
        return False


def _force_read_range(
    handle,
    offset: int,
    length: int,
    buf: bytearray,
    stop_event: _StopEvent | None,
) -> int:
    """Userspace copy that forces the range into the page cache."""
    handle.seek(offset)
    remaining = length
    copied = 0
    while remaining:
        if stop_event is not None and stop_event.is_set():
            return copied
        n = handle.readinto(memoryview(buf)[: min(len(buf), remaining)])
        if not n:
            return copied
        remaining -= n
        copied += n
    return copied


def _populate_range(
    handle,
    mapping: mmap.mmap | None,
    offset: int,
    length: int,
    buf: bytearray,
    stop_event: _StopEvent | None,
    file_size: int = 0,
) -> int:
    """Make ``[offset, offset+length)`` page-cache resident and return bytes covered.

    Issue ``readahead`` / ``posix_fadvise`` as a hint (and hint the next chunk),
    then copy immediately unless ``mincore`` already shows the range resident.
    Do not wait on mincore: a false negative burned the 7 s spawn window at
    ~0 GiB/s in measurement.
    """
    fd = handle.fileno()
    _advise_range(fd, offset, length)
    nxt = offset + length
    if file_size > nxt:
        _advise_range(fd, nxt, min(_CHUNK, file_size - nxt))
    if mapping is not None and _range_resident(mapping, offset, length):
        return length
    if _hint_only(stop_event):
        return 0
    return _force_read_range(handle, offset, length, buf, stop_event)


def _stop_all(control: object | None) -> bool:
    if control is None:
        return False
    is_set = getattr(control, "is_set", None)
    return bool(is_set()) if is_set is not None else False


def _hint_only(control: object | None) -> bool:
    if control is None:
        return False
    hint_only = getattr(control, "hint_only", None)
    return bool(hint_only()) if hint_only is not None else False


def _skip_file(control: object | None, path: str) -> bool:
    if control is None:
        return False
    should_skip = getattr(control, "should_skip", None)
    if should_skip is not None:
        return bool(should_skip(path))
    return _stop_all(control)


def _read_files(
    files: list[str],
    budget: _Budget,
    stop_event: object | None = None,
) -> tuple[int, int]:
    """Populate file pages until the budget is spent, stop-all, or each file is yielded."""
    done_bytes = 0
    done_files = 0
    buf = bytearray(_CHUNK)
    for path in files:
        if _stop_all(stop_event):
            break
        if _skip_file(stop_event, path):
            continue
        done_bytes, done_files = _read_one_file(path, budget, buf, stop_event, done_bytes, done_files)
    return done_bytes, done_files


def _read_one_file(
    path: str,
    budget: _Budget,
    buf: bytearray,
    stop_event: object | None,
    done_bytes: int,
    done_files: int,
) -> tuple[int, int]:
    """Populate one file; returns the running (bytes, files) counters."""
    try:
        size = os.path.getsize(path)
        if size == 0:
            return done_bytes, done_files + 1
        with open(path, "rb", buffering=0) as handle:
            mapping = None
            try:
                mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            except (OSError, ValueError):
                mapping = None
            try:
                offset = 0
                copied_any = False
                while offset < size:
                    if _stop_all(stop_event) or _skip_file(stop_event, path):
                        break
                    length = min(_CHUNK, size - offset)
                    covered = _populate_range(handle, mapping, offset, length, buf, stop_event, file_size=size)
                    offset += length
                    if covered:
                        copied_any = True
                        done_bytes += covered
                        if not budget.take(covered):
                            return done_bytes, done_files
                    elif _hint_only(stop_event):
                        continue
                    else:
                        return done_bytes, done_files
                else:
                    if copied_any and not _stop_all(stop_event) and not _skip_file(stop_event, path):
                        done_files += 1
            finally:
                if mapping is not None:
                    mapping.close()
    except OSError:
        pass
    return done_bytes, done_files


def checkpoint_fully_resident(files: list[str]) -> bool:
    """Whether every sampled page of every file is already in the page cache."""
    return bool(files) and all(_is_file_fully_resident(f) for f in files)


def should_pause_for_demand(path: str) -> bool:
    """Whether speculative copies must yield the disk to this file.

    A resident file stages from page cache without touching the disk, so
    other shards can keep prewarming during its staging."""
    return not _is_file_fully_resident(path)


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


class _StopEvent(Protocol):
    def set(self) -> None: ...

    def is_set(self) -> bool: ...


class _PrewarmWorker(Protocol):
    def start(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...


def _run_weights_prewarm(files: list[str], cap: int, remaining: int, control: object) -> None:
    """Populate checkpoint pages from daemon threads."""
    try:
        t0 = time.perf_counter()
        budget = _Budget(cap)
        # Round-robin the file order across readers so every stream works near
        # the head while sharing one budget.
        parts = [files[i::_DEFAULT_READERS] for i in range(_DEFAULT_READERS)]
        results = [(0, 0)] * len(parts)
        workers = []
        for i, part in enumerate(parts):

            def _one(i=i, part=part):
                results[i] = _read_files(part, budget, control)

            worker = threading.Thread(target=_one, name=f"weights-prewarm-{i}", daemon=True)
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()
        done_bytes = sum(bytes_read for bytes_read, _ in results)
        done_files = sum(file_count for _, file_count in results)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Weights prewarm%s: %.1f/%.1f GiB (%d/%d files) into page cache in %.1f s (%.1f GiB/s).",
            " stopped" if _stop_all(control) else "",
            done_bytes / (1 << 30),
            remaining / (1 << 30),
            done_files,
            len(files),
            elapsed,
            done_bytes / (1 << 30) / elapsed if elapsed > 0 else float("inf"),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Weights prewarm aborted: %s", exc)


class WeightsPrewarmHandle:
    """Cancelable prewarm task handed off when demand loading begins."""

    def __init__(self, worker: _PrewarmWorker | None, stop_event: _StopEvent) -> None:
        self._worker = worker
        self._stop_event = stop_event
        self._yielded: set[str] = set()
        self._yield_lock = threading.Lock()
        self._hint_only = False

    def stop(self) -> None:
        self._stop_event.set()

    def is_set(self) -> bool:
        return self._stop_event.is_set()

    def yield_file(self, path: str) -> None:
        """Stop speculative reads of ``path``; other files keep prewarming."""
        with self._yield_lock:
            self._yielded.add(path)
            try:
                self._yielded.add(os.path.realpath(path))
            except OSError:
                pass

    def pause_copy(self) -> None:
        """Keep fadvise hints; do not copy remaining files until resume."""
        with self._yield_lock:
            self._hint_only = True

    def resume_copy(self) -> None:
        with self._yield_lock:
            self._hint_only = False

    def hint_only(self) -> bool:
        with self._yield_lock:
            return self._hint_only

    def should_skip(self, path: str) -> bool:
        if self._stop_event.is_set():
            return True
        with self._yield_lock:
            if path in self._yielded:
                return True
            try:
                return os.path.realpath(path) in self._yielded
            except OSError:
                return False

    def join(self, timeout: float | None = None) -> None:
        if self._worker is not None:
            self._worker.join(timeout)

    def is_alive(self) -> bool:
        return bool(self._worker is not None and self._worker.is_alive())

    def terminate(self) -> bool:
        terminate = getattr(self._worker, "terminate", None)
        if terminate is None:
            return False
        terminate()
        return True


def stop_weights_prewarm(
    handle: WeightsPrewarmHandle,
    timeout: float | None = None,
) -> bool:
    """Stop a reader within ``timeout``; threads check the stop event between chunks."""
    if timeout is None:
        timeout = _PREWARM_STOP_TIMEOUT_S
    handle.stop()
    handle.join(timeout)
    if not handle.is_alive():
        return True
    if not handle.terminate():
        logger.warning("Weights prewarm reader did not stop within %.1f s; continuing without waiting.", timeout)
        return False
    handle.join(timeout)
    stopped = not handle.is_alive()
    if not stopped:
        logger.warning("Weights prewarm worker did not stop within %.1f s.", timeout)
    return stopped


_parent_prewarm_active: ContextVar[bool] = ContextVar("parent_weights_prewarm_active", default=False)
_parent_prewarm_handoff: ContextVar[WeightsPrewarmHandoff | None] = ContextVar(
    "parent_weights_prewarm_handoff", default=None
)


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
    """Kick off daemon threads that populate checkpoint pages into the cache.
    ``model`` may be a local directory or a HF repo id resolved from
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

    logger.info(
        "Weights prewarm started: %d safetensors, %.1f GiB (budget %.1f GiB).",
        len(files),
        remaining / (1 << 30),
        cap / (1 << 30),
    )
    stop_event = threading.Event()
    handle = WeightsPrewarmHandle(None, stop_event)
    worker = threading.Thread(
        target=_run_weights_prewarm,
        args=(files, cap, remaining, handle),
        name="weights-prewarm",
        daemon=True,
    )
    handle._worker = worker
    worker.start()
    return handle


@dataclass(frozen=True)
class WeightsPrewarmHandoff:
    """Worker token for a parent-owned prewarm window."""

    _commands: object | None = None
    _acks: object | None = None

    def wait(self) -> None:
        """Workers may enter model init without stopping parent readers."""
        return None

    def yield_file(self, path: str, timeout: float = 5.0) -> None:
        self._command("yield", path, timeout=timeout)

    def _command(self, op: str, payload: str | None = None, timeout: float = 5.0) -> None:
        if self._commands is None:
            return
        self._commands.put((op, payload))
        if self._acks is None:
            return
        try:
            self._acks.get(timeout=timeout)
        except (Empty, OSError, ValueError):
            return

    def request_stop(self) -> None:
        if self._commands is not None:
            self._commands.put(("stop", None))

    def pause_copy(self, timeout: float = 5.0) -> None:
        self._command("pause_copy", timeout=timeout)

    def resume_copy(self, timeout: float = 5.0) -> None:
        self._command("resume_copy", timeout=timeout)


@contextmanager
def parent_weights_prewarm(
    model: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> Iterator[WeightsPrewarmHandoff]:
    """Prewarm during spawned-worker startup and model load.

    Workers yield individual files when demand I/O starts. The parent joins
    readers after workers report ready (when this context exits).
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
    commands = context.Queue()
    acks = context.Queue()
    handoff = WeightsPrewarmHandoff(commands, acks)

    def _coordinate() -> None:
        while True:
            try:
                op, payload = commands.get()
            except (OSError, EOFError, KeyboardInterrupt):
                break
            if op == "stop":
                break
            if op == "yield" and payload:
                handle.yield_file(str(payload))
            elif op == "pause_copy":
                handle.pause_copy()
            elif op == "resume_copy":
                handle.resume_copy()
            try:
                acks.put(True)
            except (OSError, ValueError):
                pass
        stop_weights_prewarm(handle)
        logger.info("Parent weights prewarm stopped after worker ready.")

    coordinator = threading.Thread(
        target=_coordinate,
        name="weights-prewarm-handoff",
        daemon=True,
    )
    coordinator.start()
    try:
        yield handoff
    finally:
        try:
            commands.put(("stop", None))
        except (OSError, ValueError):
            handle.stop()
        coordinator.join()


def yield_parent_prewarm_file(path: str) -> None:
    """Ask the parent reader to stop speculative I/O for ``path``."""
    handoff = _parent_prewarm_handoff.get()
    if handoff is not None:
        handoff.yield_file(path)


def pause_parent_prewarm_copy() -> None:
    handoff = _parent_prewarm_handoff.get()
    if handoff is not None:
        handoff.pause_copy()


def resume_parent_prewarm_copy() -> None:
    handoff = _parent_prewarm_handoff.get()
    if handoff is not None:
        handoff.resume_copy()


@contextmanager
def use_parent_weights_prewarm(handoff: WeightsPrewarmHandoff | None) -> Iterator[None]:
    """Suppress duplicate local prewarm; parent readers keep running."""
    if handoff is None:
        yield
        return

    handoff.wait()
    token = _parent_prewarm_active.set(True)
    handoff_token = _parent_prewarm_handoff.set(handoff)
    try:
        yield
    finally:
        _parent_prewarm_handoff.reset(handoff_token)
        _parent_prewarm_active.reset(token)
