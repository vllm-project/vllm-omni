# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.experimental.fullduplex.core.adapter import DuplexAdapter, DuplexCapability, OutputChunk
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime, Emit
from vllm_omni.experimental.fullduplex.core.session import DuplexSession


@dataclass(frozen=True)
class MageVLCodecWindow:
    """One causal Mage-VL visual window.

    A window may carry decoded frames, codec-native side information, or both.
    The adapter deliberately keeps these fields opaque so traditional H.264/HEVC
    metadata and neural codec tensors can share the same full-duplex session API.
    """

    data: Any
    kind: str = "frames"
    segment_id: str | None = None
    pts_ms: int | None = None
    duration_ms: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MageVLGateDecision:
    should_respond: bool
    text: str = ""
    event_id: str | None = None
    score: float | None = None
    reason: str = ""


GateResult = MageVLGateDecision | Mapping[str, Any] | bool | str | None
GateFn = Callable[[DuplexSession, Sequence[MageVLCodecWindow]], Awaitable[GateResult] | GateResult]
GenerateResult = AsyncIterator[str | OutputChunk] | Sequence[str | OutputChunk] | str | OutputChunk | Awaitable[Any]
GenerateFn = Callable[
    [DuplexSession, Sequence[MageVLCodecWindow], str | None, MageVLGateDecision | None],
    GenerateResult,
]


class MageVLDuplexAdapter(DuplexAdapter):
    """Full-duplex session adapter for Mage-VL's proactive streaming mode.

    This adapter owns the serving/session contract around Mage-VL: bounded causal
    visual memory, rolling-window cognition-gate evaluation, event de-duplication,
    explicit user queries, and stale-output cleanup through the shared runtime.
    Model execution is injected through ``gate`` and ``generate`` callables so the
    same adapter can front the remote-code Transformers path, vLLM once native
    ``mage_vl`` support lands, or an OpenAI-compatible backend.
    """

    def __init__(
        self,
        *,
        gate: GateFn | None = None,
        generate: GenerateFn | None = None,
        window_size: int = 4,
        max_windows: int = 64,
        output_modality: str = "text",
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if max_windows <= 0:
            raise ValueError("max_windows must be positive")
        if window_size > max_windows:
            raise ValueError("window_size cannot exceed max_windows")
        self._gate = gate
        self._generate = generate
        self._window_size = window_size
        self._max_windows = max_windows
        self._output_modality = output_modality
        # Adapter instances are shared by the serving runtime. Keep mutable
        # cognition state per session so concurrent sessions cannot leak visual
        # windows, queries, or event ids into one another.
        self._session_state: dict[str, _MageVLSessionState] = {}

    def capabilities(self) -> DuplexCapability:
        return DuplexCapability(
            # The production Transformers backend currently consumes encoded
            # video windows.  Do not advertise modalities it cannot materialize.
            input_modalities=frozenset({"text", "video"}),
            output_modalities=frozenset({self._output_modality}),
            proactive=True,
        )

    async def on_input(self, session: DuplexSession, modality: str, data: Any) -> None:
        state = self._state_for(session)
        if modality == "text":
            text = str(data or "").strip()
            if text:
                state.pending_query = text
            return

        window = _coerce_window(modality, data)
        state.windows.append(window)
        if len(state.windows) > self._max_windows:
            del state.windows[: len(state.windows) - self._max_windows]
        self._schedule_gate(session)

    def should_respond(self, session: DuplexSession) -> bool:
        state = self._state_for(session)
        if state.pending_query and state.windows:
            return True
        return bool(state.pending_gate and state.pending_gate.should_respond)

    async def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]:
        state = self._state_for(session)
        windows = tuple(state.windows[-self._window_size :])
        query = state.pending_query
        gate = state.pending_gate
        state.pending_query = None
        state.pending_gate = None

        if self._generate is None:
            if gate and gate.text:
                yield OutputChunk(self._output_modality, gate.text)
            return

        async for chunk in _iter_generate_result(self._generate(session, windows, query, gate)):
            yield _as_output_chunk(chunk, self._output_modality)

    async def on_barge_in(self, session: DuplexSession) -> None:
        state = self._state_for(session)
        state.pending_query = None
        state.pending_gate = None
        for task in tuple(state.gate_tasks):
            task.cancel()
        while not state.gate_ready.empty():
            state.gate_ready.get_nowait()

    async def on_close(self, session: DuplexSession) -> None:
        state = self._session_state.get(session.session_id)
        if state is not None and state.gate_tasks:
            await asyncio.gather(*tuple(state.gate_tasks), return_exceptions=True)
        self._session_state.pop(session.session_id, None)

    async def flush(self, session: DuplexSession) -> None:
        state = self._session_state.get(session.session_id)
        if state is not None and state.gate_tasks:
            await asyncio.gather(*tuple(state.gate_tasks), return_exceptions=True)

    async def wait_for_gate(self, session: DuplexSession) -> None:
        await self._state_for(session).gate_ready.get()

    async def _evaluate_gate(self, session: DuplexSession) -> None:
        state = self._state_for(session)
        if self._gate is None or len(state.windows) < self._window_size:
            return
        decision = _coerce_gate_decision(
            await _maybe_await(self._gate(session, tuple(state.windows[-self._window_size :])))
        )
        if not decision.should_respond:
            return
        if decision.event_id and decision.event_id in state.seen_events:
            return
        if decision.event_id:
            state.seen_events.add(decision.event_id)
        state.pending_gate = decision
        state.gate_ready.put_nowait(None)

    def _schedule_gate(self, session: DuplexSession) -> None:
        state = self._state_for(session)
        if self._gate is None or len(state.windows) < self._window_size:
            return
        task = asyncio.create_task(self._evaluate_gate(session))
        state.gate_tasks.add(task)
        task.add_done_callback(state.gate_tasks.discard)

    def _state_for(self, session: DuplexSession) -> _MageVLSessionState:
        return self._session_state.setdefault(session.session_id, _MageVLSessionState())


@dataclass
class _MageVLSessionState:
    windows: list[MageVLCodecWindow] = field(default_factory=list)
    pending_query: str | None = None
    pending_gate: MageVLGateDecision | None = None
    seen_events: set[str] = field(default_factory=set)
    gate_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    gate_ready: asyncio.Queue[None] = field(default_factory=asyncio.Queue)


class MageVLDuplexRuntime(DuplexRuntime):
    """Shared turn runtime with a model-owned deferred-gate watcher."""

    adapter: MageVLDuplexAdapter

    def __init__(self, session: DuplexSession, adapter: MageVLDuplexAdapter) -> None:
        super().__init__(session, adapter)
        self._gate_watch_task: asyncio.Task[None] | None = None

    async def _on_input_waiting(self, modality: str, emit: Emit) -> None:
        if modality == "text" or self._gate_watch_task is not None:
            return
        self._gate_watch_task = asyncio.create_task(self._watch_gate(emit))

    async def _watch_gate(self, emit: Emit) -> None:
        while True:
            await self.adapter.wait_for_gate(self.session)
            if self.adapter.should_respond(self.session):
                await self._start_response(emit)
                # Let respond() consume the accepted gate before checking the
                # next notification, so stacked decisions cannot supersede it.
                await asyncio.sleep(0)

    async def _stop_background(self) -> None:
        task = self._gate_watch_task
        self._gate_watch_task = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _on_graceful_close(self, emit: Emit) -> None:
        await self.adapter.flush(self.session)
        if self.session.config.proactive and self.adapter.should_respond(self.session):
            await self._start_response(emit)


def _coerce_window(modality: str, data: Any) -> MageVLCodecWindow:
    if isinstance(data, MageVLCodecWindow):
        return data
    if isinstance(data, Mapping):
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {"metadata": metadata}
        payload = data.get("data", data.get("frames", data.get("frame", data.get("codec"))))
        return MageVLCodecWindow(
            data=payload,
            kind=str(data.get("kind", modality)),
            segment_id=_optional_str(data.get("segment_id")),
            pts_ms=_optional_int(data.get("pts_ms")),
            duration_ms=_optional_int(data.get("duration_ms")),
            metadata=metadata,
        )
    return MageVLCodecWindow(data=data, kind=modality)


def _coerce_gate_decision(result: GateResult) -> MageVLGateDecision:
    if isinstance(result, MageVLGateDecision):
        return result
    if isinstance(result, Mapping):
        return MageVLGateDecision(
            should_respond=bool(result.get("should_respond", result.get("respond", result.get("open", False)))),
            text=str(result.get("text", "")),
            event_id=_optional_str(result.get("event_id")),
            score=_optional_float(result.get("score")),
            reason=str(result.get("reason", "")),
        )
    if isinstance(result, str):
        text = result.strip()
        return MageVLGateDecision(should_respond=bool(text), text=text)
    return MageVLGateDecision(should_respond=bool(result))


async def _iter_generate_result(result: GenerateResult) -> AsyncIterator[str | OutputChunk]:
    resolved: Any = await _maybe_await(result)
    if isinstance(resolved, str | OutputChunk):
        yield resolved
        return
    if hasattr(resolved, "__aiter__"):
        async for item in resolved:
            yield item
        return
    for item in resolved:
        yield item


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _as_output_chunk(chunk: str | OutputChunk, default_modality: str) -> OutputChunk:
    if isinstance(chunk, OutputChunk):
        return chunk
    return OutputChunk(default_modality, chunk)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
