"""Per-device shared/exclusive init phase locks for parallel stage init.

When ``VllmOmniOrchestratorConfig.parallel_stage_init`` is enabled, several
stage engine-core processes may initialize on the same physical GPU at the same
time. Correctness then rests on two mechanisms (see
``rfc_parallel_stage_init.zh.md``):

  * **Admission** (``stage_admission``) proves, before any stage launches, that
    every physical device's summed budget fits — this is the hard OOM backstop.
  * **SH/EX phase locks** (this module) make each memory *measurement* quiescent:
    memory-mutating phases (weight load, KV allocation, CUDA-graph capture) take
    a per-device shared (``LOCK_SH``) lock; the profiling measurement inside
    ``determine_available_memory`` takes an exclusive (``LOCK_EX``) lock. The
    exclusive holder waits for in-flight mutations to drain and blocks new ones
    for its (seconds-long) window, so the device-level profiling number each
    stage computes stays within its admitted budget.

The lock is held by the engine-core **driver** process (one per stage / per local
DP rank) and wraps the driver-side collective calls, so a single acquisition
covers every worker rank with no per-rank locking (which would deadlock on
inverted lock order).
Acquisition is always over the driver's whole device slice in **sorted global
order** so different drivers cannot deadlock, and it is **fail-closed**: a lock
timeout raises rather than silently proceeding.

The core locking logic here intentionally depends only on ``os``/``fcntl`` so it
is importable and unit-testable without a GPU; torch/platform imports are lazy.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from collections.abc import Iterator

from vllm.logger import init_logger

from vllm_omni.engine.stage_init_utils import (
    device_init_lock_path,
    open_device_lock_file,
    record_lock_holder_pid,
)

logger = init_logger(__name__)

# Default upper bound on how long a single phase will wait for its device locks
# before failing closed. Generous: a peer's exclusive profile window is seconds,
# but a peer's whole shared load/capture span can be much longer.
_DEFAULT_LOCK_TIMEOUT_S = 900.0


class DeviceLockTimeoutError(RuntimeError):
    """Raised (fail-closed) when a phase cannot acquire its device locks in time."""


def resolve_driver_device_ids(vllm_config, local_dp_rank: int = 0) -> list[int]:
    """Physical GPU ids this engine-core driver owns, as a sorted list.

    A driver (one local vLLM DP rank) owns a contiguous slice of the visible
    devices of size ``TP*PP*PCP*SP*CFG`` (DP excluded — each DP rank is its own
    driver). Visible ids come from the device-control env var the parent set on
    the child (e.g. ``CUDA_VISIBLE_DEVICES``); when unset, all devices are visible.
    """
    from vllm_omni.platforms import current_omni_platform

    env_var = current_omni_platform.device_control_env_var
    visible = os.environ.get(env_var) if env_var else None
    physical: list[int]
    if visible:
        physical = []
        for tok in visible.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                physical.append(int(tok))
            except ValueError:
                # Non-integer visibility (UUID / MIG) can't be mapped to a lock
                # id; fall back to positional ordinals below.
                physical = []
                break
    else:
        physical = []
    if not physical:
        physical = list(range(current_omni_platform.get_device_count()))

    pc = vllm_config.parallel_config
    devices_per_dp = (
        int(getattr(pc, "tensor_parallel_size", 1))
        * int(getattr(pc, "pipeline_parallel_size", 1))
        * int(getattr(pc, "prefill_context_parallel_size", 1))
        * int(getattr(pc, "sequence_parallel_size", 1))
        * int(getattr(pc, "cfg_parallel_size", 1))
    )
    devices_per_dp = max(1, devices_per_dp)
    start = int(local_dp_rank) * devices_per_dp
    slice_ = physical[start : start + devices_per_dp]
    if not slice_:
        # Defensive: fall back to the whole visible set rather than lock nothing.
        slice_ = physical
    return sorted(slice_)


class DevicePhaseLock:
    """Sorted, fail-closed, whole-set SH/EX flock over a driver's physical devices."""

    def __init__(
        self,
        device_ids: list[int],
        *,
        timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S,
        lock_dir: str = "/tmp",
    ) -> None:
        self._device_ids = sorted(device_ids)
        self._timeout_s = timeout_s
        self._lock_dir = lock_dir

    @classmethod
    def from_child(
        cls,
        vllm_config,
        local_dp_rank: int = 0,
        *,
        timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S,
    ) -> DevicePhaseLock:
        """Build the lock for the current engine-core child from its config/env."""
        device_ids = resolve_driver_device_ids(vllm_config, local_dp_rank)
        return cls(device_ids, timeout_s=timeout_s)

    @property
    def device_ids(self) -> list[int]:
        return list(self._device_ids)

    # ---- acquisition -------------------------------------------------------

    def _acquire(self, mode: int) -> list[int]:
        """Acquire *mode* (LOCK_SH|LOCK_EX) on every device in sorted order.

        Fail-closed: on timeout, releases whatever was acquired and raises
        ``DeviceLockTimeoutError``. Sorted global order makes cross-driver deadlock
        impossible.
        """
        acquired: list[int] = []
        deadline = time.monotonic() + self._timeout_s
        try:
            for device_id in self._device_ids:
                fd, writable = open_device_lock_file(device_init_lock_path(device_id, self._lock_dir))
                while True:
                    try:
                        fcntl.flock(fd, mode | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() > deadline:
                            os.close(fd)
                            raise DeviceLockTimeoutError(
                                f"Timed out after {self._timeout_s:.0f}s acquiring "
                                f"{'EX' if mode == fcntl.LOCK_EX else 'SH'} lock on device {device_id}"
                            )
                        time.sleep(0.01)
                record_lock_holder_pid(fd, writable)
                acquired.append(fd)
            return acquired
        except BaseException:
            _release_fds(acquired)
            raise

    @contextlib.contextmanager
    def shared(self) -> Iterator[None]:
        """Hold ``LOCK_SH`` on the device slice for a memory-mutating phase."""
        if not self._device_ids:
            yield
            return
        fds = self._acquire(fcntl.LOCK_SH)
        try:
            yield
        finally:
            _release_fds(fds)

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold ``LOCK_EX`` on the device slice for the profiling measurement."""
        if not self._device_ids:
            yield
            return
        fds = self._acquire(fcntl.LOCK_EX)
        try:
            yield
        finally:
            _release_fds(fds)


def _release_fds(fds: list[int]) -> None:
    for fd in fds:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except (OSError, ValueError):
            pass


def wrap_executor_with_phase_locks(executor_class, locker: DevicePhaseLock):
    """Return a subclass of *executor_class* that brackets init phases with locks.

    * ``__init__`` (executor construction → weight load) under ``shared()``.
    * ``determine_available_memory`` (profiling) under ``exclusive()``.
    * ``initialize_from_config`` (KV cache allocation) under ``shared()``.
    * ``compile_or_warm_up_model`` (kernel warmup + CUDA-graph capture) under
      ``shared()``.

    ``compile_or_warm_up_model`` needs its own bracket: upstream used to fuse it
    into ``Executor.initialize_from_config``, but since vLLM v0.20.0 (vllm#39240)
    ``EngineCore._initialize_kv_caches`` calls the two separately. Leaving it
    unwrapped would let one stage capture graphs — mutating device memory — while
    a peer holds the exclusive lock for its profiling measurement, which is
    exactly the interference this protocol exists to prevent.

    Works uniformly for any executor (UniProc/MultiProc): the lock is taken in
    the driver process that instantiates the executor and issues its collectives.
    """

    class _PhaseLockedExecutor(executor_class):
        def __init__(self, vllm_config):
            with locker.shared():
                super().__init__(vllm_config)

        def determine_available_memory(self):
            with locker.exclusive():
                return super().determine_available_memory()

        def initialize_from_config(self, kv_cache_configs):
            with locker.shared():
                return super().initialize_from_config(kv_cache_configs)

        def compile_or_warm_up_model(self):
            with locker.shared():
                return super().compile_or_warm_up_model()

    _PhaseLockedExecutor.__name__ = f"PhaseLocked{executor_class.__name__}"
    _PhaseLockedExecutor.__qualname__ = _PhaseLockedExecutor.__name__
    return _PhaseLockedExecutor
