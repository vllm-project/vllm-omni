# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cooperative cancellation for local, full-forward diffusion workers.

The engine owns each signal until the worker has returned. Unlike an executor
RPC, writing the signal does not wait behind the forward being cancelled.
Opted-in pipelines call ``check_request_cancellation`` at safe boundaries.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from multiprocessing.shared_memory import SharedMemory


class RequestCancellationRegistry:
    """Engine-owned signals; scheduler mutation remains on the engine thread."""

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


_current_signals: ContextVar[tuple[SharedMemory | None, ...] | None] = ContextVar(
    "diffusion_request_cancellation", default=None
)


@contextmanager
def request_cancellation_scope(signal_names: Sequence[str | None]) -> Iterator[None]:
    """Attach worker readers without taking ownership of the engine's names."""
    signals: list[SharedMemory | None] = []
    try:
        for name in signal_names:
            signals.append(SharedMemory(name=name) if name is not None else None)
        token = _current_signals.set(tuple(signals) if any(signals) else None)
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

    ``synchronize=True`` bounds device work queued ahead of a model-step boundary.
    Without it, a CPU could enqueue the entire denoise loop before DELETE arrives.
    Independent requests coupled by an AllGather offload wave must all be cancelled
    before the wave can exit; cancelling one must not abort its live peers.
    """
    signals = _current_signals.get()
    if signals is None:
        return

    import torch

    if synchronize:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()

    cancelled = all(signal is not None and signal.buf[0] for signal in signals)
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

        raise DiffusionRequestAbortedError("Request cancelled at a model execution boundary")
