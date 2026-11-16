# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request streaming contracts: execution-scoped emits and Engine-owned output state.

No IPC or client transport lives here. The Runner binds a request emitter from
Worker transport; the Engine serializes state transitions under its own lock.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm_omni.diffusion.data import DiffusionOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.request import OmniDiffusionRequest

EmitChunkFn = Callable[[DiffusionOutput], None]
EmitChunkFactory = Callable[[str], EmitChunkFn]
_emit_chunk: ContextVar[EmitChunkFn | None] = ContextVar("request_emit_chunk", default=None)
_emit_factory: ContextVar[EmitChunkFactory | None] = ContextVar("request_emit_factory", default=None)


def bind_emit_chunk(fn: EmitChunkFn | None) -> Token:
    """Bind a model output callback for the current invocation."""
    return _emit_chunk.set(fn)


def reset_emit_chunk(token: Token) -> None:
    """Restore the previous model output callback."""
    _emit_chunk.reset(token)


def get_emit_chunk() -> EmitChunkFn | None:
    """Return the callback; background decoders must capture it before starting."""
    return _emit_chunk.get()


def bind_emit_factory(factory: EmitChunkFactory | None) -> Token:
    """Bind the replying Worker's transport without inspecting model RPC arguments."""
    return _emit_factory.set(factory)


def reset_emit_factory(token: Token) -> None:
    """Restore transport binding after the RPC exits."""
    _emit_factory.reset(token)


@contextmanager
def request_chunk_emitter(requests: Sequence[OmniDiffusionRequest], *, enabled: bool) -> Iterator[None]:
    """Bind single-request streaming at the common Runner execution boundary.

    Multi-request forwards retain final-output behavior until their model
    adapters can emit an explicit per-member identity.
    """
    factory = _emit_factory.get()
    emit = factory(requests[0].request_id) if enabled and factory is not None and len(requests) == 1 else None
    token = bind_emit_chunk(emit)
    try:
        yield
    finally:
        reset_emit_chunk(token)


@contextmanager
def request_chunk_transport(factory: EmitChunkFactory | None) -> Iterator[None]:
    """Bind transport only while this forward can produce output.

    Close captured decoder callbacks before the execution returns, including
    failure/cancellation. The Worker then drains already submitted messages
    before replying, so a retired request id cannot receive a late chunk.
    """
    lock = threading.Lock()
    active = True

    def scoped_factory(request_id: str) -> EmitChunkFn:
        assert factory is not None
        emit = factory(request_id)

        def submit(output: DiffusionOutput) -> None:
            with lock:
                if active:
                    emit(output)

        return submit

    token = bind_emit_factory(scoped_factory if factory is not None else None)
    try:
        yield
    finally:
        with lock:
            active = False
        reset_emit_factory(token)


@dataclass
class RequestStreamState:
    """In-order request-mode chunk stream. The scheduler does not see these chunks."""

    next_chunk_index: int = 0
    in_flight_bytes: int = 0
    max_output_bytes: int = 64 * 1024 * 1024
    cancelled: bool = False
    output_ack: bool = False
    failed: bool = False
    terminal_emitted: bool = False
    error: str | None = None
    retained_by_chunk: dict[int, int] = field(default_factory=dict)

    def release_chunk(self, chunk_index: int) -> None:
        """Release retained bytes once the downstream consumer releases a chunk."""
        retained = self.retained_by_chunk.pop(chunk_index, 0)
        self.in_flight_bytes = max(0, self.in_flight_bytes - retained)

    def fail(self, message: str) -> tuple[str, str]:
        """Choose one error terminal; Engine propagates cancellation to execution."""
        self.failed = self.cancelled = self.terminal_emitted = True
        self.error = message
        return "fail", message

    def accept_chunk(
        self,
        request_id: str,
        chunk_index: int,
        output: DiffusionOutput | None,
        error: str | None,
        async_output_id: str | None,
    ) -> tuple[str, DiffusionOutput | str | None]:
        """Validate order and budget before accepting one chunk."""
        if self.cancelled or self.terminal_emitted:
            return "drop", None
        if error:
            return self.fail(error)
        if output is None or async_output_id is None:
            return self.fail("request streaming chunk missing output")
        if chunk_index < 0 or chunk_index > self.next_chunk_index:
            return self.fail(
                f"request streaming protocol error for {request_id}: "
                f"expected chunk {self.next_chunk_index}, got {chunk_index}",
            )
        if chunk_index < self.next_chunk_index:
            return "drop", None
        nbytes = output.output_nbytes()
        if nbytes > self.max_output_bytes or self.in_flight_bytes + nbytes > self.max_output_bytes:
            return self.fail(
                "Decoded streaming output exceeds streaming_max_output_bytes",
            )
        self.next_chunk_index += 1
        self.in_flight_bytes += nbytes
        self.retained_by_chunk[chunk_index] = nbytes
        return (
            "emit",
            DiffusionOutput(
                finished=False,
                chunk_index=chunk_index,
                total_chunks=output.total_chunks,
                async_output_id=async_output_id,
            ),
        )

    def finish(self, output: DiffusionOutput) -> DiffusionOutput:
        """Suppress duplicate final media after chunks.

        Callers must not invoke this after ``fail()``: ``terminal_emitted`` is
        checked first and returns ``None`` from the Engine path.
        """
        if output.error or output.aborted or self.next_chunk_index == 0:
            return output
        return DiffusionOutput(
            finished=True,
            chunk_index=self.next_chunk_index - 1,
            total_chunks=self.next_chunk_index,
        )
