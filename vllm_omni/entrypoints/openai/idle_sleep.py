# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Automatic level-1 sleep for idle, single-worker API deployments."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from enum import Enum
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send
from vllm.logger import init_logger

from vllm_omni.config.stage_config import IdleSleepRuntimeConfig

logger = init_logger(__name__)


class IdleSleepState(str, Enum):
    AWAKE = "awake"
    SLEEPING = "sleeping"
    ASLEEP = "asleep"
    WAKING = "waking"
    SUSPENDED = "suspended"
    ERROR = "error"
    CLOSED = "closed"


class IdleSleepUnavailableError(RuntimeError):
    """Automatic lifecycle cannot serve inference in its current state."""


class IdleSleepConflictError(RuntimeError):
    """Manual lifecycle state conflicts with automatic ownership."""


_INFERENCE_HTTP_PATHS = frozenset(
    {
        "/pooling",
        "/rerank",
        "/score",
        "/v1/audio/generate",
        "/v1/audio/speech",
        "/v1/audio/speech/batch",
        "/v1/chat/completions",
        "/v1/chat/completions/batch",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/images/edits",
        "/v1/images/generations",
        "/v1/images/segmentations",
        "/v1/realtime/sessions",
        "/v1/rerank",
        "/v1/run",
        "/v1/score",
        "/v1/videos",
        "/v1/videos/depth",
        "/v1/videos/segmentations",
        "/v1/videos/sync",
        "/v2/rerank",
    }
)
_INFERENCE_WEBSOCKET_PATHS = frozenset(
    {
        "/v1/audio/speech/stream",
        "/v1/duplex",
        "/v1/realtime",
        "/v1/realtime/robot/openpi",
        "/v1/realtime/video",
        "/v1/video/chat/stream",
    }
)


def is_inference_scope(scope: Scope) -> bool:
    """Classify requests that require model admission."""
    scope_type = scope["type"]
    path = scope.get("path", "")
    if scope_type == "websocket":
        return path in _INFERENCE_WEBSOCKET_PATHS
    if scope_type != "http" or scope.get("method") != "POST":
        return False
    return path in _INFERENCE_HTTP_PATHS or path.startswith("/v1/realtime/sessions/")


def _config_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _stage_sleep_enabled(stage_config: Any) -> bool:
    if _config_value(stage_config, "enable_sleep_mode") is True:
        return True
    for owner_name in ("model_config", "engine_args", "yaml_engine_args"):
        owner = _config_value(stage_config, owner_name)
        if _config_value(owner, "enable_sleep_mode") is True:
            return True
    return False


def validate_idle_sleep_startup(engine: Any, *, api_server_count: int) -> list[int]:
    """Validate process-local and engine requirements before serving."""
    if api_server_count != 1:
        raise ValueError("idle_sleep requires exactly one API server worker")
    required = ("is_idle_for_sleep", "is_paused", "is_sleeping", "resume_generation", "sleep", "wake_up")
    missing = [name for name in required if not callable(getattr(engine, name, None))]
    if missing:
        raise ValueError(f"idle_sleep engine is missing lifecycle method(s): {', '.join(missing)}")
    stage_configs = list(getattr(engine, "stage_configs", ()) or ())
    if not stage_configs:
        raise ValueError("idle_sleep requires at least one resolved stage")
    disabled = [index for index, stage in enumerate(stage_configs) if not _stage_sleep_enabled(stage)]
    if disabled:
        raise ValueError(f"idle_sleep requires enable_sleep_mode=true for stage(s): {disabled}")
    return list(range(len(stage_configs)))


class IdleSleepController:
    """Own one idle deadline and serialize all engine lifecycle transitions."""

    def __init__(
        self,
        engine: Any,
        *,
        timeout_s: float,
        stage_ids: list[int],
        sleeping_stages: set[int],
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.engine = engine
        self.timeout_s = timeout_s
        self.stage_ids = stage_ids
        self.sleeping_stages = sleeping_stages
        self.state = IdleSleepState.AWAKE
        self.error_reason: str | None = None
        self.active_transports = 0
        self._sleep = sleep
        self._state_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._manual_lock = asyncio.Lock()
        self._timer_generation = 0
        self._timer_task: asyncio.Task[None] | None = None
        self._transition_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        async with self._state_lock:
            self._arm_timer_locked()

    async def begin_activity(self) -> None:
        async with self._state_lock:
            self.active_transports += 1
            self._disarm_timer_locked()
        try:
            await self._ensure_awake()
        except BaseException:
            await self.end_activity()
            raise

    async def end_activity(self) -> None:
        async with self._state_lock:
            if self.active_transports <= 0:
                raise RuntimeError("idle sleep activity counter underflow")
            self.active_transports -= 1
            if self.active_transports == 0 and self.state is IdleSleepState.AWAKE:
                self._arm_timer_locked()

    async def manual_sleep(self, *, stage_ids: list[int], level: int) -> list[Any]:
        async with self._manual_lock:
            await self._prepare_manual_transition()
            async with self._lifecycle_lock:
                try:
                    return await self.engine.sleep(stage_ids=stage_ids, level=level)
                finally:
                    await self._reconcile_manual_state()

    async def manual_wake(self, *, stage_ids: list[int]) -> list[Any] | None:
        async with self._manual_lock:
            previous_state = await self._prepare_manual_transition()
            auto_owned = previous_state is IdleSleepState.ASLEEP
            async with self._lifecycle_lock:
                if not auto_owned and not any(stage_id in self.sleeping_stages for stage_id in stage_ids):
                    await self._reconcile_manual_state()
                    return None
                if auto_owned and set(stage_ids) != set(self.stage_ids):
                    async with self._state_lock:
                        if self.state is IdleSleepState.SUSPENDED:
                            self.state = IdleSleepState.ASLEEP
                    raise IdleSleepConflictError("auto-slept stages must be woken together")
                try:
                    acks = await self.engine.wake_up(stage_ids=stage_ids)
                    if auto_owned:
                        await self.engine.resume_generation(stage_ids=stage_ids)
                    return acks
                finally:
                    await self._reconcile_manual_state()

    async def close(self) -> None:
        async with self._state_lock:
            if self.state is IdleSleepState.CLOSED:
                return
            self.state = IdleSleepState.CLOSED
            self._disarm_timer_locked()
            transition = self._transition_task
        if transition is not None and transition is not asyncio.current_task():
            with suppress(Exception):
                await asyncio.shield(transition)

    async def _ensure_awake(self) -> None:
        while True:
            task: asyncio.Task[None] | None = None
            async with self._state_lock:
                if self.state is IdleSleepState.AWAKE:
                    return
                if self.state is IdleSleepState.ASLEEP:
                    self.state = IdleSleepState.WAKING
                    task = asyncio.create_task(self._auto_wake())
                    self._transition_task = task
                elif self.state in (IdleSleepState.SLEEPING, IdleSleepState.WAKING):
                    task = self._transition_task
                    if task is None:
                        raise IdleSleepUnavailableError("idle sleep transition state is inconsistent")
                elif self.state is IdleSleepState.SUSPENDED:
                    raise IdleSleepConflictError("manual sleep state must be resolved before inference")
                else:
                    raise IdleSleepUnavailableError(f"idle sleep controller is {self.state.value}")
            assert task is not None
            await asyncio.shield(task)

    async def _idle_timer(self, generation: int) -> None:
        try:
            await self._sleep(self.timeout_s)
            try:
                engine_idle = await self.engine.is_idle_for_sleep()
            except Exception:
                logger.warning("Could not read engine idle state; postponing automatic sleep", exc_info=True)
                async with self._state_lock:
                    if generation == self._timer_generation and self.state is IdleSleepState.AWAKE:
                        self._timer_task = None
                        self._arm_timer_locked()
                return
            async with self._state_lock:
                if (
                    generation != self._timer_generation
                    or self.active_transports
                    or self.state is not IdleSleepState.AWAKE
                ):
                    return
                self._timer_task = None
                if not engine_idle:
                    self._arm_timer_locked()
                    return
                self.state = IdleSleepState.SLEEPING
                self._transition_task = asyncio.current_task()
            await self._auto_sleep()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.exception("Idle sleep timer failed")
            async with self._state_lock:
                if self.state is not IdleSleepState.CLOSED:
                    self.state = IdleSleepState.ERROR
                    self.error_reason = f"idle sleep timer failed: {type(exc).__name__}: {exc}"
        finally:
            async with self._state_lock:
                if self._timer_task is asyncio.current_task():
                    self._timer_task = None
                if self._transition_task is asyncio.current_task():
                    self._transition_task = None

    async def _auto_sleep(self) -> None:
        started = time.monotonic()
        logger.info(
            "Automatic idle sleep starting after %.3f idle seconds for stages %s",
            self.timeout_s,
            self.stage_ids,
        )
        try:
            async with self._lifecycle_lock:
                await self.engine.sleep(stage_ids=self.stage_ids, level=1, mode="wait")
        except Exception as exc:
            logger.exception(
                "Automatic idle sleep failed after %.3f seconds for stages %s",
                time.monotonic() - started,
                self.stage_ids,
            )
            sleeping: set[int] | None = None
            with suppress(Exception):
                sleeping = await self._read_sleeping_stages()
            async with self._state_lock:
                if sleeping is not None:
                    self.sleeping_stages.difference_update(self.stage_ids)
                    self.sleeping_stages.update(sleeping)
                if self.state is not IdleSleepState.CLOSED:
                    self.state = IdleSleepState.ERROR
                    self.error_reason = f"automatic idle sleep failed: {type(exc).__name__}: {exc}"
            return
        async with self._state_lock:
            self.sleeping_stages.update(self.stage_ids)
            if self.state is not IdleSleepState.CLOSED:
                self.state = IdleSleepState.ASLEEP
        logger.info(
            "Automatic idle sleep completed in %.3f seconds for stages %s",
            time.monotonic() - started,
            self.stage_ids,
        )

    async def _auto_wake(self) -> None:
        started = time.monotonic()
        logger.info("Automatic idle wake starting for stages %s", self.stage_ids)
        try:
            async with self._lifecycle_lock:
                await self.engine.wake_up(stage_ids=self.stage_ids)
                await self.engine.resume_generation(stage_ids=self.stage_ids)
        except Exception as exc:
            logger.exception(
                "Automatic idle wake failed after %.3f seconds for stages %s",
                time.monotonic() - started,
                self.stage_ids,
            )
            async with self._state_lock:
                if self.state is not IdleSleepState.CLOSED:
                    self.state = IdleSleepState.ERROR
                    self.error_reason = f"automatic idle wake failed: {type(exc).__name__}: {exc}"
                if self._transition_task is asyncio.current_task():
                    self._transition_task = None
            return
        async with self._state_lock:
            self.sleeping_stages.difference_update(self.stage_ids)
            if self.state is not IdleSleepState.CLOSED:
                self.state = IdleSleepState.AWAKE
            if self._transition_task is asyncio.current_task():
                self._transition_task = None
        logger.info(
            "Automatic idle wake completed in %.3f seconds for stages %s",
            time.monotonic() - started,
            self.stage_ids,
        )

    async def _prepare_manual_transition(self) -> IdleSleepState:
        while True:
            async with self._state_lock:
                self._disarm_timer_locked()
                transition = self._transition_task
                if transition is None:
                    if self.state in (IdleSleepState.CLOSED, IdleSleepState.ERROR):
                        raise IdleSleepUnavailableError(f"idle sleep controller is {self.state.value}")
                    previous_state = self.state
                    self.state = IdleSleepState.SUSPENDED
                    return previous_state
            if transition is not asyncio.current_task():
                await asyncio.shield(transition)

    async def _reconcile_manual_state(self) -> None:
        sleeping = await self._read_sleeping_stages()
        paused = await self.engine.is_paused()
        async with self._state_lock:
            self.sleeping_stages.difference_update(self.stage_ids)
            self.sleeping_stages.update(sleeping)
            if self.state in (IdleSleepState.CLOSED, IdleSleepState.ERROR):
                return
            if sleeping or paused:
                self.state = IdleSleepState.SUSPENDED
            else:
                self.state = IdleSleepState.AWAKE
                if self.active_transports == 0:
                    self._arm_timer_locked()

    async def _read_sleeping_stages(self) -> set[int]:
        return {stage_id for stage_id in self.stage_ids if await self.engine.is_sleeping(stage_ids=[stage_id])}

    def _arm_timer_locked(self) -> None:
        if self.active_transports or self.state is not IdleSleepState.AWAKE:
            return
        self._disarm_timer_locked()
        generation = self._timer_generation
        self._timer_task = asyncio.create_task(self._idle_timer(generation))

    def _disarm_timer_locked(self) -> None:
        self._timer_generation += 1
        timer = self._timer_task
        self._timer_task = None
        if timer is not None and timer is not asyncio.current_task() and not timer.done():
            timer.cancel()


def create_idle_sleep_controller(
    engine: Any,
    config: IdleSleepRuntimeConfig,
    *,
    api_server_count: int,
    sleeping_stages: set[int],
) -> IdleSleepController | None:
    if config.timeout_s is None:
        return None
    stage_ids = validate_idle_sleep_startup(engine, api_server_count=api_server_count)
    return IdleSleepController(
        engine,
        timeout_s=config.timeout_s,
        stage_ids=stage_ids,
        sleeping_stages=sleeping_stages,
    )


class IdleSleepASGI:
    """Hold an activity lease for the complete inference transport lifetime."""

    def __init__(self, inner: ASGIApp, controller: IdleSleepController) -> None:
        self._inner = inner
        self._controller = controller

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not is_inference_scope(scope):
            await self._inner(scope, receive, send)
            return
        try:
            await self._controller.begin_activity()
        except IdleSleepConflictError as exc:
            await _reject_scope(scope, send, status=409, detail=str(exc))
            return
        except IdleSleepUnavailableError as exc:
            await _reject_scope(scope, send, status=503, detail=str(exc))
            return
        try:
            await self._inner(scope, receive, send)
        finally:
            await self._controller.end_activity()


async def _reject_scope(scope: Scope, send: Send, *, status: int, detail: str) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1013, "reason": detail})
        return
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})
