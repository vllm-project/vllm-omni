# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cooperative cancellation for local, full-forward diffusion workers.

The engine owns each signal until the worker has returned. Unlike an executor
RPC, writing the signal does not wait behind the forward being cancelled.
Opted-in pipelines call ``check_request_cancellation`` at safe boundaries.

Multiproc/UniProc use POSIX ``SharedMemory``. Ray uses a per-worker local flag
table updated through a dedicated actor concurrency group so cancel does not
queue behind ``execute_rpc`` and works across machines.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from multiprocessing.shared_memory import SharedMemory
from typing import Protocol

RAY_CANCEL_PREFIX = "raycancel:"


class RequestCancellationRegistry:
    """Engine-owned SHM signals; scheduler mutation remains on the engine thread."""

    def __init__(self) -> None:
        self._signals: dict[str, SharedMemory] = {}
        self._lock = threading.Lock()

    def create(self, request_id: str) -> str:
        with self._lock:
            if request_id in self._signals:
                raise ValueError(f"Duplicate cancellation request: {request_id}")
            signal = SharedMemory(create=True, size=1)
            signal.buf[0] = 0
            self._signals[request_id] = signal
            return signal.name

    def cancel(self, request_ids: Iterable[str]) -> None:
        with self._lock:
            for request_id in request_ids:
                signal = self._signals.get(request_id)
                if signal is not None:
                    signal.buf[0] = 1

    def cancel_all(self) -> None:
        with self._lock:
            for signal in self._signals.values():
                signal.buf[0] = 1

    @staticmethod
    def _dispose(signal: SharedMemory) -> None:
        try:
            signal.unlink()
        except FileNotFoundError:
            # A process-level resource tracker may already have removed it.
            pass
        finally:
            signal.close()

    def finish(self, request_id: str) -> None:
        """Release only after execution returns, including aborted execution."""
        with self._lock:
            signal = self._signals.pop(request_id, None)
            if signal is not None:
                self._dispose(signal)

    def close(self) -> None:
        """Release remaining signals after the executor has shut down."""
        with self._lock:
            for signal in self._signals.values():
                self._dispose(signal)
            self._signals.clear()


class RayRequestCancellationRegistry:
    """Cross-node cancellation: fire-and-forget flag updates on every Ray worker."""

    def __init__(self, broadcast: Callable[..., None]) -> None:
        # broadcast(request_ids, *, cancelled=False, finish=False)
        self._broadcast = broadcast
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def create(self, request_id: str) -> str:
        with self._lock:
            if request_id in self._active:
                raise ValueError(f"Duplicate cancellation request: {request_id}")
            self._active.add(request_id)
        self._broadcast([request_id], cancelled=False, finish=False)
        return f"{RAY_CANCEL_PREFIX}{request_id}"

    def cancel(self, request_ids: Iterable[str]) -> None:
        ids = list(request_ids)
        with self._lock:
            ids = [request_id for request_id in ids if request_id in self._active]
        if ids:
            self._broadcast(ids, cancelled=True, finish=False)

    def cancel_all(self) -> None:
        with self._lock:
            ids = list(self._active)
        if ids:
            self._broadcast(ids, cancelled=True, finish=False)

    def finish(self, request_id: str) -> None:
        with self._lock:
            self._active.discard(request_id)
        self._broadcast([request_id], cancelled=False, finish=True)

    def close(self) -> None:
        with self._lock:
            ids = list(self._active)
            self._active.clear()
        if ids:
            self._broadcast(ids, cancelled=False, finish=True)


class _CancellationFlag(Protocol):
    @property
    def buf(self) -> object: ...

    def close(self) -> None: ...


class _ByteFlag:
    """One-byte view compatible with the SharedMemory ``buf[0]`` check."""

    def __init__(self, reader: Callable[[], bool]) -> None:
        self._reader = reader

    def __getitem__(self, index: int) -> int:
        if index != 0:
            raise IndexError(index)
        return 1 if self._reader() else 0


class _RayLocalCancelSignal:
    def __init__(self, request_id: str, table: dict[str, threading.Event]) -> None:
        self._request_id = request_id
        self._table = table
        self.buf = _ByteFlag(self._is_set)

    def _is_set(self) -> bool:
        event = self._table.get(self._request_id)
        return bool(event is not None and event.is_set())

    def close(self) -> None:
        return None


_current_signals: ContextVar[tuple[_CancellationFlag | None, ...] | None] = ContextVar(
    "diffusion_request_cancellation", default=None
)
_ray_cancel_table: ContextVar[dict[str, threading.Event] | None] = ContextVar(
    "diffusion_ray_cancel_table", default=None
)


def bind_ray_cancel_table(table: dict[str, threading.Event] | None):
    """Worker execute path publishes the local cancel table for Ray signals."""
    return _ray_cancel_table.set(table)


def reset_ray_cancel_table(token) -> None:
    _ray_cancel_table.reset(token)


@contextmanager
def request_cancellation_scope(signal_names: Sequence[str | None], *, enabled: bool = True) -> Iterator[None]:
    """Attach worker readers without taking ownership of the engine's names."""
    signals: list[_CancellationFlag | None] = []
    try:
        for name in signal_names if enabled else ():
            if name is None:
                signals.append(None)
            elif name.startswith(RAY_CANCEL_PREFIX):
                table = _ray_cancel_table.get()
                if table is None:
                    raise RuntimeError(
                        "Ray cancellation signal requires a worker cancel table; "
                        "multi-node Ray cannot open POSIX SharedMemory by name."
                    )
                signals.append(_RayLocalCancelSignal(name[len(RAY_CANCEL_PREFIX) :], table))
            else:
                signals.append(SharedMemory(name=name))
        # An opted-in rank without a local signal must still join every
        # checkpoint collective and vote "not cancelled". None means only
        # that this execution does not use cooperative cancellation at all.
        token = _current_signals.set(tuple(signals) if enabled else None)
        try:
            yield
        finally:
            _current_signals.reset(token)
    finally:
        for signal in signals:
            if signal is not None:
                signal.close()


def check_request_cancellation(*, synchronize: bool = False) -> None:
    """Stop a cancelled execution wave without stranding a peer's collectives.

    ``synchronize=True`` drains queued device work only when cancellation has
    been requested locally, before re-reading the flags. Every abort drains queued
    device work before unwinding. Successful steps retain asynchronous execution
    when no local cancellation is observed.
    Independent requests coupled by an AllGather offload wave must all be cancelled
    before the wave can exit; cancelling one must not abort its live peers.
    """
    signals = _current_signals.get()
    if signals is None:
        return

    import torch

    locally_requested = any(signal is not None and signal.buf[0] for signal in signals)
    device_drained = False
    if synchronize and locally_requested:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()
        device_drained = True

    cancelled = bool(signals) and all(signal is not None and signal.buf[0] for signal in signals)
    if torch.distributed.is_initialized():
        from vllm_omni.diffusion.distributed.parallel_state import get_world_group

        group = get_world_group()
        if group.world_size > 1:
            # Reading a shared byte alone is insufficient: ranks can observe a
            # concurrent write on different steps. Make one collective decision.
            flag = torch.tensor([int(cancelled)], dtype=torch.int32, device="cpu")
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group)
            cancelled = bool(flag.item())

    if cancelled:
        from vllm_omni.diffusion.data import DiffusionRequestAbortedError

        if not device_drained:
            from vllm_omni.platforms import current_omni_platform

            current_omni_platform.synchronize()
        raise DiffusionRequestAbortedError("Request cancelled at a model execution boundary")
