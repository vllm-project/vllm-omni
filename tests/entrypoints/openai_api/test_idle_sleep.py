# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute

from vllm_omni.config.stage_config import IdleSleepRuntimeConfig
from vllm_omni.entrypoints.openai.idle_sleep import (
    IdleSleepASGI,
    IdleSleepConflictError,
    IdleSleepController,
    IdleSleepState,
    create_idle_sleep_controller,
    is_inference_scope,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class ManualSleeper:
    def __init__(self) -> None:
        self.calls: list[tuple[float, asyncio.Future[None]]] = []

    async def __call__(self, delay: float) -> None:
        future = asyncio.get_running_loop().create_future()
        self.calls.append((delay, future))
        await future

    async def wait_for_calls(self, count: int) -> None:
        async def wait() -> None:
            while len(self.calls) < count:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait(), timeout=1)

    def fire(self, index: int = -1) -> None:
        future = self.calls[index][1]
        if not future.done():
            future.set_result(None)


class FakeEngine:
    def __init__(self) -> None:
        self.stage_configs = [SimpleNamespace(model_config=SimpleNamespace(enable_sleep_mode=True))]
        self.idle = True
        self.paused = False
        self.sleeping = False
        self.request_states: dict[str, object] = {}
        self.sleep_calls: list[dict] = []
        self.wake_calls: list[list[int]] = []
        self.resume_calls: list[list[int]] = []
        self.wake_started = asyncio.Event()
        self.release_wake = asyncio.Event()
        self.release_wake.set()

    async def is_idle_for_sleep(self) -> bool:
        return self.idle and not self.request_states

    async def is_paused(self) -> bool:
        return self.paused

    async def is_sleeping(self, *, stage_ids: list[int]) -> bool:
        return self.sleeping and stage_ids == [0]

    async def sleep(self, *, stage_ids: list[int], level: int, mode: str = "abort"):
        self.sleep_calls.append({"stage_ids": stage_ids, "level": level, "mode": mode})
        self.sleeping = True
        self.paused = True
        return []

    async def wake_up(self, *, stage_ids: list[int]):
        self.wake_calls.append(stage_ids)
        self.wake_started.set()
        await self.release_wake.wait()
        self.sleeping = False
        return []

    async def resume_generation(self, *, stage_ids: list[int]) -> None:
        self.resume_calls.append(stage_ids)
        self.paused = False


def make_controller(engine: FakeEngine, sleeper: ManualSleeper) -> IdleSleepController:
    return IdleSleepController(
        engine,
        timeout_s=30,
        stage_ids=[0],
        sleeping_stages=set(),
        sleep=sleeper,
    )


async def sleep_once(controller: IdleSleepController, sleeper: ManualSleeper) -> None:
    await controller.start()
    await sleeper.wait_for_calls(1)
    sleeper.fire()

    async def wait() -> None:
        while controller.state is not IdleSleepState.ASLEEP:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)


async def send_inference_request(app: IdleSleepASGI, path: str = "/v1/chat/completions") -> list[dict]:
    messages: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    await app({"type": "http", "method": "POST", "path": path}, receive, send)
    return messages


@pytest.mark.asyncio
async def test_controller_sleeps_once_after_idle_deadline():
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)

    await sleep_once(controller, sleeper)

    assert engine.sleep_calls == [{"stage_ids": [0], "level": 1, "mode": "wait"}]
    assert controller.sleeping_stages == {0}
    await controller.close()


@pytest.mark.asyncio
async def test_activity_invalidates_deadline_and_rearms_after_completion():
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    await controller.start()
    await sleeper.wait_for_calls(1)

    await controller.begin_activity()
    assert sleeper.calls[0][1].cancelled()
    assert engine.sleep_calls == []

    await controller.end_activity()
    await sleeper.wait_for_calls(2)
    sleeper.fire()

    async def wait() -> None:
        while not engine.sleep_calls:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)
    assert len(engine.sleep_calls) == 1
    await controller.close()


@pytest.mark.asyncio
async def test_concurrent_activity_shares_one_wake_transition():
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    await sleep_once(controller, sleeper)
    engine.release_wake.clear()

    first = asyncio.create_task(controller.begin_activity())
    second = asyncio.create_task(controller.begin_activity())
    await engine.wake_started.wait()

    assert engine.wake_calls == [[0]]
    engine.release_wake.set()
    await asyncio.gather(first, second)

    assert engine.resume_calls == [[0]]
    assert controller.state is IdleSleepState.AWAKE
    await controller.end_activity()
    await controller.end_activity()
    await controller.close()


@pytest.mark.asyncio
async def test_manual_wake_of_auto_sleep_resumes_generation():
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    await sleep_once(controller, sleeper)

    acks = await controller.manual_wake(stage_ids=[0])

    assert acks == []
    assert engine.wake_calls == [[0]]
    assert engine.resume_calls == [[0]]
    assert controller.state is IdleSleepState.AWAKE
    await controller.close()


@pytest.mark.asyncio
async def test_partial_manual_wake_preserves_auto_sleep_ownership():
    engine = FakeEngine()
    controller = IdleSleepController(
        engine,
        timeout_s=30,
        stage_ids=[0, 1],
        sleeping_stages={0, 1},
    )
    controller.state = IdleSleepState.ASLEEP

    with pytest.raises(IdleSleepConflictError, match="woken together"):
        await controller.manual_wake(stage_ids=[0])

    assert controller.state is IdleSleepState.ASLEEP
    assert engine.wake_calls == []
    await controller.close()


@pytest.mark.asyncio
async def test_manual_sleep_blocks_new_inference_before_engine_transition(mocker):
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()

    async def blocking_sleep(*, stage_ids, level, mode="abort"):
        sleep_started.set()
        await release_sleep.wait()
        engine.sleeping = True
        engine.paused = True
        return []

    mocker.patch.object(engine, "sleep", side_effect=blocking_sleep)
    task = asyncio.create_task(controller.manual_sleep(stage_ids=[0], level=1))
    await sleep_started.wait()

    with pytest.raises(IdleSleepConflictError, match="manual sleep state"):
        await controller.begin_activity()

    release_sleep.set()
    await task
    assert controller.state is IdleSleepState.SUSPENDED
    await controller.close()


@pytest.mark.asyncio
async def test_busy_engine_starts_a_fresh_idle_deadline():
    engine = FakeEngine()
    engine.idle = False
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    await controller.start()
    await sleeper.wait_for_calls(1)
    sleeper.fire()
    await sleeper.wait_for_calls(2)

    assert engine.sleep_calls == []
    assert sleeper.calls[1][0] == 30
    await controller.close()


@pytest.mark.asyncio
async def test_close_cancels_pending_deadline():
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    await controller.start()
    await sleeper.wait_for_calls(1)

    await controller.close()

    assert sleeper.calls[0][1].cancelled()
    assert controller.state is IdleSleepState.CLOSED
    assert engine.sleep_calls == []


@pytest.mark.asyncio
async def test_failed_automatic_sleep_records_health_error(mocker):
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)

    async def failed_sleep(*, stage_ids, level, mode="abort"):
        raise RuntimeError("offload failed")

    mocker.patch.object(engine, "sleep", side_effect=failed_sleep)
    await controller.start()
    await sleeper.wait_for_calls(1)
    sleeper.fire()

    async def wait() -> None:
        while controller.state is not IdleSleepState.ERROR:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)
    assert controller.error_reason == "automatic idle sleep failed: RuntimeError: offload failed"
    await controller.close()


def test_startup_validation_requires_single_worker_and_sleep_enabled():
    engine = FakeEngine()

    with pytest.raises(ValueError, match="exactly one"):
        create_idle_sleep_controller(
            engine,
            IdleSleepRuntimeConfig(timeout_s=30),
            api_server_count=2,
            sleeping_stages=set(),
        )

    engine.stage_configs[0].model_config.enable_sleep_mode = False
    with pytest.raises(ValueError, match="stage\\(s\\): \\[0\\]"):
        create_idle_sleep_controller(
            engine,
            IdleSleepRuntimeConfig(timeout_s=30),
            api_server_count=1,
            sleeping_stages=set(),
        )


def test_startup_validation_accepts_legacy_stage_engine_args():
    engine = FakeEngine()
    engine.stage_configs = [SimpleNamespace(yaml_engine_args={"enable_sleep_mode": True})]

    controller = create_idle_sleep_controller(
        engine,
        IdleSleepRuntimeConfig(timeout_s=30),
        api_server_count=1,
        sleeping_stages=set(),
    )

    assert controller is not None


@pytest.mark.parametrize(
    "scope, expected",
    [
        ({"type": "http", "method": "POST", "path": "/v1/chat/completions"}, True),
        ({"type": "http", "method": "GET", "path": "/health"}, False),
        ({"type": "http", "method": "GET", "path": "/v1/videos/id"}, False),
        ({"type": "websocket", "path": "/v1/realtime"}, True),
        ({"type": "websocket", "path": "/unknown"}, False),
    ],
)
def test_inference_scope_classification(scope, expected):
    assert is_inference_scope(scope) is expected


def test_route_census_classifies_all_assembled_routes():
    from vllm_omni.entrypoints.openai.api_server import router

    assembled: set[tuple[str, str | None, str]] = set()
    for route in router.routes:
        if isinstance(route, APIRoute):
            for method in sorted(route.methods):
                assembled.add(("http", method, route.path))
        elif isinstance(route, APIWebSocketRoute):
            assembled.add(("websocket", None, route.path))

    inference = {
        ("http", "POST", "/v1/chat/completions"),
        ("http", "POST", "/v1/chat/completions/batch"),
        ("http", "POST", "/v1/audio/speech"),
        ("http", "POST", "/v1/audio/speech/batch"),
        ("http", "POST", "/v1/audio/generate"),
        ("websocket", None, "/v1/audio/speech/stream"),
        ("websocket", None, "/v1/video/chat/stream"),
        ("websocket", None, "/v1/realtime/video"),
        ("websocket", None, "/v1/realtime"),
        ("websocket", None, "/v1/duplex"),
        ("websocket", None, "/v1/realtime/robot/openpi"),
        ("http", "POST", "/v1/run"),
        ("http", "POST", "/v1/realtime/sessions"),
        ("http", "POST", "/v1/realtime/sessions/{session_id}/step"),
        ("http", "POST", "/v1/realtime/sessions/{session_id}/reset"),
        ("http", "POST", "/v1/realtime/sessions/{session_id}/close"),
        ("http", "POST", "/v1/images/generations"),
        ("http", "POST", "/v1/images/edits"),
        ("http", "POST", "/v1/images/segmentations"),
        ("http", "POST", "/v1/videos/segmentations"),
        ("http", "POST", "/v1/videos/depth"),
        ("http", "POST", "/v1/videos"),
        ("http", "POST", "/v1/videos/sync"),
    }
    lifecycle = {
        ("http", "GET", "/health"),
        ("http", "POST", "/v1/omni/sleep"),
        ("http", "POST", "/v1/omni/wakeup"),
    }
    passive = {
        ("http", "GET", "/v1/audio/voices"),
        ("http", "POST", "/v1/audio/voices"),
        ("http", "DELETE", "/v1/audio/voices/{name}"),
        ("http", "GET", "/v1/realtime/sessions/{session_id}/status"),
        ("http", "GET", "/v1/models"),
        ("http", "GET", "/v1/videos"),
        ("http", "GET", "/v1/videos/{video_id}"),
        ("http", "DELETE", "/v1/videos/{video_id}"),
        ("http", "GET", "/v1/videos/{video_id}/content"),
    }
    expected = inference | lifecycle | passive

    assert assembled == expected

    for route in inference:
        scope_type, method, path = route
        scope = {"type": scope_type, "path": path}
        if method is not None:
            scope["method"] = method
        assert is_inference_scope(scope) is True

    for route in lifecycle | passive:
        scope_type, method, path = route
        scope = {"type": scope_type, "path": path}
        if method is not None:
            scope["method"] = method
        assert is_inference_scope(scope) is False


def test_missing_lifecycle_method_fails_startup():
    for missing in ("is_idle_for_sleep", "is_paused", "is_sleeping", "resume_generation", "sleep", "wake_up"):
        engine = FakeEngine()
        setattr(engine, missing, None)
        with pytest.raises(ValueError, match=missing):
            create_idle_sleep_controller(
                engine,
                IdleSleepRuntimeConfig(timeout_s=30),
                api_server_count=1,
                sleeping_stages=set(),
            )


@pytest.mark.asyncio
async def test_asgi_holds_activity_until_http_stream_finishes():
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    await controller.start()
    stream_open = asyncio.Event()
    release_stream = asyncio.Event()

    async def inner(scope, receive, send):
        stream_open.set()
        await release_stream.wait()

    app = IdleSleepASGI(inner, controller)
    task = asyncio.create_task(
        app(
            {"type": "http", "method": "POST", "path": "/v1/chat/completions"},
            None,
            None,
        )
    )
    await stream_open.wait()

    assert controller.active_transports == 1
    release_stream.set()
    await task
    assert controller.active_transports == 0
    await controller.close()


@pytest.mark.asyncio
async def test_asgi_holds_activity_for_websocket_lifetime():
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    await controller.start()
    connected = asyncio.Event()
    disconnect = asyncio.Event()

    async def inner(scope, receive, send):
        connected.set()
        await disconnect.wait()

    app = IdleSleepASGI(inner, controller)
    task = asyncio.create_task(app({"type": "websocket", "path": "/v1/realtime"}, None, None))
    await connected.wait()

    assert controller.active_transports == 1
    disconnect.set()
    await task
    assert controller.active_transports == 0
    await controller.close()


@pytest.mark.asyncio
async def test_background_video_job_remains_busy_after_submit_response():
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    app = IdleSleepASGI(lambda scope, receive, send: asyncio.sleep(0), controller)

    engine.request_states["video-job"] = object()
    await controller.start()
    await sleeper.wait_for_calls(1)

    await app(
        {"type": "http", "method": "POST", "path": "/v1/videos"},
        None,
        None,
    )
    assert sleeper.calls[0][1].cancelled()

    await sleeper.wait_for_calls(2)
    sleeper.fire(1)
    await sleeper.wait_for_calls(3)
    assert engine.sleep_calls == []
    assert sleeper.calls[2][0] == 30

    engine.request_states.clear()
    sleeper.fire(2)

    async def wait() -> None:
        while controller.state is not IdleSleepState.ASLEEP:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)
    assert engine.sleep_calls == [{"stage_ids": [0], "level": 1, "mode": "wait"}]
    await controller.close()


@pytest.mark.asyncio
async def test_wake_failure_sets_error_and_rejects_inference(mocker):
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    await sleep_once(controller, sleeper)
    wake_up = mocker.patch.object(engine, "wake_up", side_effect=RuntimeError("wake failed"))

    async def inner(scope, receive, send):
        raise AssertionError("inference handler should not run after wake failure")

    app = IdleSleepASGI(inner, controller)
    first = await send_inference_request(app)
    second = await send_inference_request(app)

    assert first[0]["status"] == 503
    assert second[0]["status"] == 503
    assert wake_up.await_count == 1
    assert controller.state is IdleSleepState.ERROR
    assert controller.error_reason == "automatic idle wake failed: RuntimeError: wake failed"
    await controller.close()


@pytest.mark.asyncio
async def test_resume_failure_sets_error_and_rejects_inference(mocker):
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    await sleep_once(controller, sleeper)
    wake_up = mocker.patch.object(engine, "wake_up", side_effect=engine.wake_up)
    resume = mocker.patch.object(engine, "resume_generation", side_effect=RuntimeError("resume failed"))

    async def inner(scope, receive, send):
        raise AssertionError("inference handler should not run after resume failure")

    app = IdleSleepASGI(inner, controller)
    first = await send_inference_request(app)
    second = await send_inference_request(app)

    assert first[0]["status"] == 503
    assert second[0]["status"] == 503
    assert wake_up.await_count == 1
    assert resume.await_count == 1
    assert controller.state is IdleSleepState.ERROR
    assert controller.error_reason == "automatic idle wake failed: RuntimeError: resume failed"
    await controller.close()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_shared_wake():
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    await sleep_once(controller, sleeper)
    engine.release_wake.clear()

    cancelled = asyncio.create_task(controller.begin_activity())
    survivor = asyncio.create_task(controller.begin_activity())
    await engine.wake_started.wait()

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    engine.release_wake.set()
    await survivor

    assert engine.wake_calls == [[0]]
    assert engine.resume_calls == [[0]]
    assert controller.state is IdleSleepState.AWAKE
    await controller.end_activity()
    await controller.close()


@pytest.mark.asyncio
async def test_asgi_releases_activity_on_handler_error():
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    app = IdleSleepASGI(
        lambda scope, receive, send: (_ for _ in ()).throw(RuntimeError("handler failed")),
        controller,
    )

    with pytest.raises(RuntimeError, match="handler failed"):
        await app({"type": "http", "method": "POST", "path": "/v1/chat/completions"}, None, None)

    assert controller.active_transports == 0
    await controller.close()


@pytest.mark.asyncio
async def test_close_during_sleep_or_wake_starts_no_new_transition_sleep(mocker):
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()

    async def blocking_sleep(*, stage_ids, level, mode="abort"):
        sleep_started.set()
        await release_sleep.wait()
        engine.sleeping = True
        engine.paused = True
        return []

    sleep_mock = mocker.patch.object(engine, "sleep", side_effect=blocking_sleep)
    await controller.start()
    await sleeper.wait_for_calls(1)
    sleeper.fire()
    await sleep_started.wait()
    assert controller.state is IdleSleepState.SLEEPING

    close_task = asyncio.create_task(controller.close())
    await asyncio.sleep(0)
    release_sleep.set()
    await close_task

    assert controller.state is IdleSleepState.CLOSED
    assert sleep_mock.await_count == 1
    assert engine.wake_calls == []
    assert engine.resume_calls == []


@pytest.mark.asyncio
async def test_close_during_sleep_or_wake_starts_no_new_transition_wake(mocker):
    engine = FakeEngine()
    sleeper = ManualSleeper()
    controller = make_controller(engine, sleeper)
    await sleep_once(controller, sleeper)
    release_wake = asyncio.Event()

    async def blocking_wake(*, stage_ids):
        engine.wake_calls.append(stage_ids)
        engine.wake_started.set()
        await release_wake.wait()
        engine.sleeping = False
        return []

    wake_mock = mocker.patch.object(engine, "wake_up", side_effect=blocking_wake)
    activity = asyncio.create_task(controller.begin_activity())
    await engine.wake_started.wait()
    assert controller.state is IdleSleepState.WAKING

    close_task = asyncio.create_task(controller.close())
    await asyncio.sleep(0)
    release_wake.set()
    with pytest.raises(Exception, match="idle sleep controller is closed"):
        await asyncio.gather(activity, close_task)

    assert controller.state is IdleSleepState.CLOSED
    assert wake_mock.await_count == 1
    assert engine.resume_calls == [[0]]
