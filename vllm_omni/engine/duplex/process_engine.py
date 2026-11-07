# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Process-owned native duplex runtime and its session-only engine client.

Only a privately inherited socketpair connects the frontend and worker. Typed
commands/events are encoded as MessagePack data, never executable objects.
The native engine retains its orchestrator thread and owns all stage clients.
"""

from __future__ import annotations

import asyncio
import itertools
import multiprocessing
import os
import signal
import socket
import traceback
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

import anyio
import msgpack
import msgspec
import uvloop
from anyio.abc import TaskGroup
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.config.endpoint_policy import EndpointRestriction, OmniServingCapability
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands, events
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer, DuplexOutputOverflowError
from vllm_omni.engine.duplex.events import AudioDelta, ResponseDone, SessionClosed, error_event
from vllm_omni.engine.duplex.messages import DuplexControlResultMessage, DuplexSessionError, DuplexSessionEventMessage

logger = init_logger(__name__)
_QUEUE_SIZE = 2048
_QUEUE_BYTES = 16 * 1024 * 1024
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_READ_BYTES = 64 * 1024
_COMMAND_TYPES = {
    value.__name__: value
    for value in vars(commands).values()
    if isinstance(value, type) and issubclass(value, commands.DuplexCommand)
}
_EVENT_TYPES = {
    value.__name__: value
    for value in vars(events).values()
    if isinstance(value, type) and issubclass(value, events.DuplexEvent)
}
_CONTROLS = frozenset({"open_session_async", "close_session_async", "resume_session_async", "touch_session_async"})


class WorkerRuntime(Protocol):
    """Factory result; the process module has no entrypoint/model imports."""

    model: str
    engine: Any  # Native engine or the lightweight engine used by spawn tests.
    process_warmup_audio: dict[str, object]

    async def check_health(self) -> None: ...

    def shutdown(self) -> None: ...


EngineFactory = Callable[[str, dict[str, Any]], WorkerRuntime]


@dataclass(frozen=True)
class ProcessDuplexStageMetadata:
    """Small deployment snapshot; stage clients and processors stay in the worker."""

    stage_id: int
    stage_type: str
    engine_args: dict[str, int | None]


@dataclass(frozen=True)
class ProcessDuplexMetadata:
    """Serializable startup facts needed by the frontend without stage clients."""

    model: str
    num_stages: int
    stage_configs: list[ProcessDuplexStageMetadata]
    config_path: str | None
    duplex_session_config: DuplexSessionRuntimeConfig
    duplex_capabilities: DuplexCapabilities
    frontend_group_count: int
    frontend_max_sessions: int
    warmup_audio: dict[str, object]
    endpoint_restrictions: list[tuple[str, str]]

    @classmethod
    def from_runtime(cls, runtime: WorkerRuntime) -> ProcessDuplexMetadata:
        """Snapshot deployment facts after the native runtime has initialized."""
        engine = runtime.engine
        stages = []
        for stage, vllm_config in zip(engine.stage_configs, engine.stage_vllm_configs):
            model_config = getattr(vllm_config, "model_config", None)
            max_model_len = getattr(model_config, "max_model_len", None)
            stages.append(
                ProcessDuplexStageMetadata(
                    stage_id=stage.stage_id,
                    stage_type=getattr(stage.stage_type, "value", stage.stage_type),
                    engine_args={
                        "max_model_len": max_model_len
                        if max_model_len is not None
                        else stage.engine_args.get("max_model_len")
                    },
                )
            )
        return cls(
            model=runtime.model,
            num_stages=engine.num_stages,
            stage_configs=stages,
            config_path=engine.config_path,
            duplex_session_config=engine.duplex_session_config,
            duplex_capabilities=engine.duplex_capabilities,
            frontend_group_count=engine.duplex_frontend_group_count,
            frontend_max_sessions=engine.duplex_frontend_max_sessions,
            warmup_audio=runtime.process_warmup_audio,
            endpoint_restrictions=[(r.capability.name, r.reason) for r in engine.endpoint_restrictions],
        )


class _MessageSocket:
    """Bounded FIFO with one async writer and a streaming MessagePack reader."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        unpacker: msgpack.Unpacker | None = None,
    ) -> None:
        self.reader, self.writer = reader, writer
        self.unpacker = unpacker if unpacker is not None else _new_unpacker()
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(_QUEUE_SIZE)
        self.pending_bytes = 0
        self._space = asyncio.Event()
        self._failure: str | None = None

    @staticmethod
    def entry(message: tuple[Any, ...]) -> bytes:
        """Encode one message and reject payloads exceeding the queue byte budget."""
        payload = msgspec.msgpack.encode(message)
        if len(payload) > _QUEUE_BYTES:
            raise ValueError("Duplex IPC message exceeds the 16 MiB queue budget")
        return payload

    def put_nowait(self, entry: bytes) -> None:
        """Admit encoded data without waiting; raise QueueFull or engine_dead."""
        if self._failure is not None:
            raise DuplexSessionError(self._failure, code="engine_dead")
        if self.pending_bytes + len(entry) > _QUEUE_BYTES:
            raise asyncio.QueueFull
        self.queue.put_nowait(entry)
        self.pending_bytes += len(entry)

    async def send(self, message: tuple[Any, ...], *, timeout: float | None = None) -> None:
        """Wait for queue admission, not peer delivery.

        The optional timeout bounds capacity waits and reports retryable
        backpressure. Cancellation before admission leaves no queued message.
        """
        entry = self.entry(message)
        try:
            self.put_nowait(entry)
        except asyncio.QueueFull:
            try:
                await asyncio.wait_for(self.send_entry(entry), timeout)
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise DuplexSessionError(
                    "Duplex IPC queue is full", code="engine_backpressure", retryable=True
                ) from exc

    async def send_entry(self, entry: bytes) -> None:
        """Wait for queue capacity or fail when the channel closes."""
        while True:
            try:
                self.put_nowait(entry)
                return
            except asyncio.QueueFull:
                self._space.clear()
                await self._space.wait()

    async def write(self) -> None:
        """Drain admitted messages in FIFO order; run as the sole writer task.

        Capacity is released before socket drain, so the active payload and
        transport buffers are additional to the queue byte budget.
        """
        while True:
            payload = await self.queue.get()
            self.pending_bytes -= len(payload)
            self._space.set()
            self.writer.write(payload)
            await self.writer.drain()

    async def receive(self) -> Any:
        """Decode the next value, retaining partial and trailing data across reads."""
        while True:
            for message in self.unpacker:
                return message
            chunk = await self.reader.read(_READ_BYTES)
            if not chunk:
                raise EOFError("Duplex IPC connection closed")
            self.unpacker.feed(chunk)

    def close(self, detail: str = "Duplex IPC connection closed") -> None:
        """Reject new sends, wake capacity waiters, and close the socket transport."""
        self._failure = self._failure or detail
        self._space.set()
        self.writer.close()


def _new_unpacker() -> msgpack.Unpacker:
    return msgpack.Unpacker(raw=False, strict_map_key=False, timestamp=3, max_buffer_size=_MAX_MESSAGE_BYTES)


def _error(exc: Exception) -> dict[str, Any]:
    return {
        "message": str(exc),
        "code": getattr(exc, "code", "engine_error"),
        "retryable": getattr(exc, "retryable", False),
        "session_id": getattr(exc, "session_id", None),
    }


def _worker(sock: socket.socket, model: str, options: dict[str, Any], factory: EngineFactory) -> None:
    """Create a process group for the runtime and its children, then start its loop."""
    os.setsid()
    uvloop.run(_serve_worker(sock, model, options, factory))


async def _serve_worker(sock: socket.socket, model: str, options: dict[str, Any], factory: EngineFactory) -> None:
    """Initialize the runtime, send its startup result, and serve until shutdown.

    Startup uses the blocking socket before handing it to asyncio. This scope
    owns native runtime cleanup after the worker tasks have stopped.
    """
    runtime = None
    channel = None
    ready = False
    try:
        runtime = factory(model, options)
        payload = msgspec.msgpack.encode((True, ProcessDuplexMetadata.from_runtime(runtime)))
        if len(payload) > _MAX_MESSAGE_BYTES:
            raise ValueError("Oversized duplex startup metadata")
        sock.sendall(payload)
        ready = True
        sock.setblocking(False)
        reader, writer = await asyncio.open_connection(sock=sock)
        channel = _MessageSocket(reader, writer)
        await _DuplexProcessWorker(runtime, channel).run()
    except Exception:
        logger.exception("Duplex engine process failed")
        if not ready:
            with suppress(OSError):
                sock.sendall(msgspec.msgpack.encode((False, traceback.format_exc())))
    finally:
        if runtime is not None:
            runtime.shutdown()
        if channel is not None:
            channel.close()
            with suppress(ConnectionError):
                await channel.writer.wait_closed()
        sock.close()


class _ProcessOutputBuffer(DuplexOutputBuffer):
    """Native session buffer that forwards invalidations over the process boundary."""

    def __init__(self, limits: DuplexSessionRuntimeConfig, notify: Callable[[str, int], None]) -> None:
        super().__init__(
            max_bytes=limits.max_pending_output_bytes_per_session,
            max_events=limits.max_pending_output_events_per_session,
        )
        self._notify_invalidation = notify

    def invalidate(self, response_id: str, through_epoch: int) -> None:
        super().invalidate(response_id, through_epoch)
        self._notify_invalidation(response_id, through_epoch)


class _DuplexProcessWorker:
    """Child-side session forwarding; owns buffers and supervised runtime tasks."""

    def __init__(self, runtime: WorkerRuntime, channel: _MessageSocket) -> None:
        self.runtime, self.engine, self.channel = runtime, runtime.engine, channel
        self.loop = asyncio.get_running_loop()
        self.buffers: dict[str, DuplexOutputBuffer] = {}
        self.tasks: TaskGroup

    async def run(self) -> None:
        """Supervise worker tasks, cancelling siblings on failure or disconnect."""
        async with anyio.create_task_group() as self.tasks:
            self.tasks.start_soon(self.read_requests)
            self.tasks.start_soon(self.pump_engine)
            with suppress(ConnectionError):
                await self.channel.write()
            self.tasks.cancel_scope.cancel()

    def notify_invalidation(self, session_id: str, response_id: str, through_epoch: int) -> None:
        """Schedule an invalidation from the orchestrator thread onto the IPC loop.

        Overflow closes the channel: losing an invalidation could let stale
        audio reach the frontend.
        """
        entry = self.channel.entry(("invalidate", session_id, response_id, through_epoch))

        def enqueue() -> None:
            try:
                self.channel.put_nowait(entry)
            except DuplexSessionError:
                return  # The channel closed before this callback ran.
            except asyncio.QueueFull:
                logger.error("Duplex IPC invalidation queue overflow")
                self.channel.close("Duplex IPC invalidation queue overflow")

        self.loop.call_soon_threadsafe(enqueue)

    async def send_event(self, session_id: str, event: events.DuplexEvent) -> None:
        await self.channel.send(("event", session_id, type(event).__name__, event))

    async def pump_session(self, session_id: str, buffer: DuplexOutputBuffer) -> None:
        """Forward valid buffered events until closure, then retire the buffer."""
        try:
            while (event := await buffer.get()) is not None:
                if buffer.is_valid(event):
                    await self.send_event(session_id, event)
        finally:
            self.buffers.pop(session_id, None)

    async def pump_engine(self) -> None:
        """Forward engine events, routing session closure through its output buffer."""
        while True:
            for message in await self.engine.get_outputs_async():
                if not isinstance(message, DuplexSessionEventMessage):
                    raise RuntimeError(f"Unexpected session-only engine output: {message!r}")
                buffer = self.buffers.get(message.session_id)
                if isinstance(message.event, SessionClosed) and buffer is not None:
                    buffer.close(message.event)
                else:
                    await self.send_event(message.session_id, message.event)

    async def control(self, request_id: int, operation: str, args: list[Any], kwargs: dict[str, Any]) -> None:
        """Execute a control RPC and return its result or a structured error."""
        try:
            if operation == "open_session_async":
                session_id = args[0]
                args[1] = msgspec.convert(args[1], type=DuplexSessionConfig)
                buffer = _ProcessOutputBuffer(
                    self.engine.duplex_session_config, partial(self.notify_invalidation, session_id)
                )
                self.buffers[session_id] = buffer
                self.tasks.start_soon(self.pump_session, session_id, buffer)
                try:
                    result = await self.engine.open_session_async(*args, output_buffer=buffer, **kwargs)
                except BaseException:
                    buffer.close()
                    raise
            elif operation == "check_health":
                await self.runtime.check_health()
                result = None
            else:
                result = await getattr(self.engine, operation)(*args, **kwargs)
            await self.channel.send(("result", request_id, True, result))
        except Exception as exc:
            await self.channel.send(("result", request_id, False, _error(exc)))

    async def read_requests(self) -> None:
        """Dispatch commands in receive order and run control RPCs concurrently."""
        with suppress(EOFError, ConnectionError):
            while True:
                request_id, operation, args, kwargs = await self.channel.receive()
                if operation == "shutdown":
                    break
                if operation == "submit_command_async":
                    session_id, name, data = args
                    command: commands.DuplexCommand = msgspec.convert(data, type=_COMMAND_TYPES[name])
                    try:
                        await self.engine.submit_command_async(session_id, command)
                    except Exception as exc:
                        await self.send_event(
                            session_id,
                            error_event(
                                getattr(exc, "code", "engine_error"),
                                str(exc),
                                event_id=getattr(command, "event_id", None),
                            ),
                        )
                elif operation in _CONTROLS or operation == "check_health":
                    self.tasks.start_soon(self.control, request_id, operation, args, kwargs)
                else:
                    raise ValueError(f"Unsupported duplex IPC operation: {operation}")
        self.tasks.cancel_scope.cancel()


class ProcessDuplexEngineClient:
    """Session engine client; owns IPC and local delivery buffers, never handles.

    Invalidation is applied before later events on the IPC stream. It cannot
    revoke a frontend send already committed before the notification arrives.
    """

    supported_tasks: tuple[str, ...] = ()
    input_processor = None

    def __init__(self, model: str, options: dict[str, Any], *, factory: EngineFactory) -> None:
        """Spawn the runtime and synchronously wait for startup metadata or failure."""
        from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus

        if os.name != "posix":
            raise ValueError("The process duplex backend requires a POSIX host")
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            setup_multiprocess_prometheus()
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._ids = itertools.count()
        self._buffers: dict[str, DuplexOutputBuffer] = {}
        self._invalidated: dict[str, dict[str, int]] = {}
        self._outputs: asyncio.Queue[DuplexSessionEventMessage] = asyncio.Queue(_QUEUE_SIZE)
        self._failure: str | None = None
        self._channel: _MessageSocket | None = None
        self._start_task: asyncio.Task[None] | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._cleanup_tasks: set[asyncio.Task[Any]] = set()
        self._closed = False
        self._unpacker = _new_unpacker()
        self._socket, child = socket.socketpair()
        self._socket.settimeout(float(options.get("init_timeout", 600)) + 30)
        self._process = multiprocessing.get_context("spawn").Process(
            target=_worker, args=(child, model, options, factory), name="duplex-engine"
        )
        try:
            self._process.start()
            child.close()
            ok, facts = self._read_startup()
            if not ok:
                raise RuntimeError(f"Duplex engine process startup failed:\n{facts}")
            self.metadata = msgspec.convert(facts, type=ProcessDuplexMetadata)
            self.config_path = self.metadata.config_path
            self.stage_configs = self.metadata.stage_configs
            self.endpoint_restrictions = tuple(
                EndpointRestriction(OmniServingCapability[name], reason)
                for name, reason in self.metadata.endpoint_restrictions
            )
            self.duplex_session_config = self.metadata.duplex_session_config
            self.duplex_capabilities = self.metadata.duplex_capabilities
            self.num_stages = self.metadata.num_stages
            self.duplex_frontend_group_count = self.metadata.frontend_group_count
            self.duplex_frontend_max_sessions = self.metadata.frontend_max_sessions
            self._socket.setblocking(False)
            logger.info("Duplex process ready: frontend PID=%d engine PID=%d", os.getpid(), self._process.pid)
        except BaseException:
            child.close()
            self.shutdown()
            raise

    def get_frontend_vllm_config(self) -> None:
        """The worker owns stage configurations and generation processors."""
        return None

    def get_frontend_diffusion_od_config(self) -> None:
        return None

    def _read_startup(self) -> Any:
        """Read the blocking startup reply, preserving trailing data for async reads."""
        while True:
            for message in self._unpacker:
                return message
            chunk = self._socket.recv(_READ_BYTES)
            if not chunk:
                raise RuntimeError("Duplex engine process exited during startup")
            self._unpacker.feed(chunk)

    def start(self) -> None:
        """Lazily attach IPC tasks to the caller's running event loop."""
        if self._start_task is None:
            self._start_task = asyncio.create_task(self._connect())

    async def _connect(self) -> None:
        reader, writer = await asyncio.open_connection(sock=self._socket)
        self._channel = _MessageSocket(reader, writer, self._unpacker)
        self._tasks = [asyncio.create_task(self._channel.write()), asyncio.create_task(self._read())]
        for task in self._tasks:
            task.add_done_callback(self._transport_done)

    def _transport_done(self, task: asyncio.Task[None]) -> None:
        """Fan out transport failure and cancel the other IPC task."""
        exc = None if task.cancelled() else task.exception()
        if self._closed:
            return
        self._fail(str(exc or "IPC connection closed"))
        for other in self._tasks:
            if other is not task:
                other.cancel()

    def _fail(self, detail: str) -> None:
        """Fail pending controls and close local buffers, waking their consumers."""
        self._failure = detail
        if self._channel is not None:
            with suppress(RuntimeError):
                self._channel.close(detail)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(DuplexSessionError(detail, code="engine_dead"))
        for buffer in self._buffers.values():
            buffer.close()
        self._buffers.clear()
        self._invalidated.clear()

    async def _send(self, message: tuple[Any, ...]) -> None:
        if self._closed or self._failure:
            raise DuplexSessionError(self._failure or "engine closed", code="engine_dead")
        self.start()
        assert self._start_task is not None
        await asyncio.shield(self._start_task)
        assert self._channel is not None
        await self._channel.send(message, timeout=30)

    async def _rpc(self, operation: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Correlate a control reply with a bounded wait and retire its future.

        A timeout or caller cancellation does not cancel work in the child;
        late replies are ignored after the pending entry is removed.
        """
        request_id = next(self._ids)
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send((request_id, operation, args, kwargs))
            try:
                return await asyncio.wait_for(future, kwargs.get("timeout", 10))
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise DuplexSessionError(f"{operation} IPC timeout", code="timeout", retryable=True) from exc
        finally:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()

    async def open_session_async(
        self,
        session_id: str,
        config: DuplexSessionConfig,
        *,
        output_buffer: DuplexOutputBuffer,
        timeout: float | None = 60,
    ) -> DuplexControlResultMessage:
        """Register local output delivery before asking the worker to open a session."""
        self._buffers[session_id] = output_buffer
        try:
            return await self._rpc("open_session_async", (session_id, config), {"timeout": timeout})
        except BaseException:
            self._buffers.pop(session_id, None)
            self._invalidated.pop(session_id, None)
            raise

    async def close_session_async(
        self, session_id: str, *, reason: str = "client_close", timeout: float | None = 10
    ) -> DuplexControlResultMessage:
        return await self._rpc("close_session_async", (session_id,), {"reason": reason, "timeout": timeout})

    async def resume_session_async(
        self,
        session_id: str,
        *,
        expected_lease_generation: int,
        control_id: str | None = None,
        timeout: float | None = 10,
    ) -> DuplexControlResultMessage:
        return await self._rpc(
            "resume_session_async",
            (session_id,),
            {"expected_lease_generation": expected_lease_generation, "control_id": control_id, "timeout": timeout},
        )

    async def touch_session_async(
        self,
        session_id: str,
        *,
        activity: str,
        expected_lease_generation: int | None = None,
        timeout: float | None = 10,
    ) -> DuplexControlResultMessage:
        return await self._rpc(
            "touch_session_async",
            (session_id,),
            {"activity": activity, "expected_lease_generation": expected_lease_generation, "timeout": timeout},
        )

    async def submit_command_async(self, session_id: str, command: commands.DuplexCommand) -> None:
        """Queue a command; worker-side rejection arrives as a session error event."""
        await self._send((None, "submit_command_async", (session_id, type(command).__name__, command), {}))

    async def check_health_async(self) -> None:
        """Check native runtime health over IPC, beyond child-process liveness."""
        if not self.is_alive():
            raise EngineDeadError(self._failure or "Duplex engine process is not alive")
        try:
            await self._rpc("check_health", (), {})
        except DuplexSessionError as exc:
            raise EngineDeadError(str(exc)) from exc

    def is_alive(self) -> bool:
        return self._process.is_alive() and self._failure is None and not self._closed

    async def get_outputs_async(self, timeout: float = 0.1) -> list[DuplexSessionEventMessage]:
        """Poll lifecycle events for the frontend pump; session data uses local buffers."""
        self.start()
        assert self._start_task is not None
        await asyncio.shield(self._start_task)
        if not self.is_alive():
            raise EngineDeadError(self._failure or "Duplex engine process exited")
        try:
            return [await asyncio.wait_for(self._outputs.get(), timeout)]
        except (TimeoutError, asyncio.TimeoutError):
            return []

    async def _read(self) -> None:
        """Resolve RPCs and apply invalidations and events in wire order.

        Stale audio is discarded before local delivery. Session closure goes
        through the frontend output pump; other events enter session buffers.
        """
        assert self._channel is not None
        while True:
            message = await self._channel.receive()
            kind, *body = message
            if kind == "result":
                request_id, ok, value = body
                future = self._pending.get(request_id)
                if future is not None and not future.done():
                    if not ok:
                        future.set_exception(DuplexSessionError(value.pop("message"), **value))
                    else:
                        if value is not None:
                            value = msgspec.convert(value, type=DuplexControlResultMessage)
                        future.set_result(value)
            elif kind == "invalidate":
                session_id, response_id, epoch = body
                if (buffer := self._buffers.get(session_id)) is not None:
                    epochs = self._invalidated.setdefault(session_id, {})
                    epochs[response_id] = max(epoch, epochs.get(response_id, -1))
                    buffer.invalidate(response_id, epoch)
            elif kind == "event":
                session_id, name, data = body
                event: events.DuplexEvent = msgspec.convert(data, type=_EVENT_TYPES[name])
                buffer = self._buffers.get(session_id)
                if isinstance(event, AudioDelta) and event.epoch is not None and event.response_id is not None:
                    if event.epoch <= self._invalidated.get(session_id, {}).get(event.response_id, -1):
                        continue
                if isinstance(event, SessionClosed):
                    self._buffers.pop(session_id, None)
                    self._invalidated.pop(session_id, None)
                    if buffer is not None:
                        self._outputs.put_nowait(DuplexSessionEventMessage(session_id=session_id, event=event))
                elif buffer is not None:
                    try:
                        buffer.put(event)
                    except DuplexOutputOverflowError:
                        self._overflow_session(session_id, buffer)
                    if isinstance(event, ResponseDone) and event.response_id is not None:
                        self._invalidated.get(session_id, {}).pop(event.response_id, None)
            else:
                raise RuntimeError(f"Unknown duplex IPC message: {kind}")

    def _overflow_session(self, session_id: str, buffer: DuplexOutputBuffer) -> None:
        """Close the overflowing frontend session and asynchronously release it in the worker."""
        self._buffers.pop(session_id, None)
        self._invalidated.pop(session_id, None)
        with suppress(DuplexOutputOverflowError):
            buffer.put(error_event("server_error", "Frontend pending output limit exceeded"))
        self._outputs.put_nowait(
            DuplexSessionEventMessage(
                session_id=session_id, event=SessionClosed(session_id=session_id, reason="frontend_output_overflow")
            )
        )

        async def release() -> None:
            try:
                await self.close_session_async(session_id, reason="frontend_output_overflow")
            except DuplexSessionError:
                logger.exception("Failed to release overflowed duplex session %s", session_id)

        task = asyncio.create_task(release())
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    async def aclose(self) -> None:
        """Request graceful worker cleanup for up to 30 seconds, then ensure termination."""
        if self._closed:
            return

        async def graceful() -> None:
            if self._cleanup_tasks:
                await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            await self._send((None, "shutdown", (), {}))
            while self._process.is_alive() and self._failure is None:
                await asyncio.sleep(0.02)

        try:
            with suppress(TimeoutError, asyncio.TimeoutError, DuplexSessionError):
                await asyncio.wait_for(graceful(), 30)
        finally:
            self.shutdown(timeout=0)

    def shutdown(self, timeout: float | None = None) -> None:
        """Idempotently close IPC and join the child, escalating to process-group signals."""
        if self._closed:
            return
        self._closed = True
        self._fail("shutdown")
        for task in self._tasks:
            task.cancel()
        if self._start_task is not None:
            self._start_task.cancel()
        for task in self._cleanup_tasks:
            task.cancel()
        self._socket.close()
        if self._process.pid is not None:
            self._process.join(timeout=5 if timeout is None else timeout)
            with suppress(ProcessLookupError):
                os.killpg(self._process.pid, signal.SIGTERM)
            self._process.join(timeout=5)
            if self._process.is_alive():
                # setsid may not have run if the child failed during import.
                with suppress(ProcessLookupError):
                    os.killpg(self._process.pid, signal.SIGKILL)
                self._process.kill()
                self._process.join(timeout=5)
            from prometheus_client import multiprocess

            multiprocess.mark_process_dead(self._process.pid)
