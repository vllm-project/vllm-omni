# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``DuplexOmniEngine``: the session message surface over the generic engine base."""

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.engine.duplex.commands import Heartbeat
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    DuplexSessionError,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
    TouchDuplexSessionMessage,
)
from vllm_omni.engine.duplex_omni_engine import DuplexOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeRpcClient:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[tuple[str, str], Any, float | None]] = []

    def execute(self, key, message, *, timeout=None, timeout_message=None):
        self.calls.append((key, message, timeout))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _FakeSyncQueue:
    def __init__(self, *, full: bool = False) -> None:
        self.items: list[Any] = []
        self.full = full

    def put(self, item: Any, timeout: float | None = None) -> None:
        if self.full:
            raise queue.Full
        self.items.append(item)


def _engine(result: Any, *, alive: bool = True, full: bool = False) -> DuplexOmniEngine:
    engine = object.__new__(DuplexOmniEngine)
    engine._correlated_rpc_client = _FakeRpcClient(result)
    engine.request_queue = SimpleNamespace(sync_q=_FakeSyncQueue(full=full))
    engine.is_alive = lambda: alive
    engine.plugin = None
    return engine


def _ok(operation: str, **fields: Any) -> DuplexControlResultMessage:
    return DuplexControlResultMessage(control_id="c", operation=operation, session_id="sid", ok=True, **fields)


@pytest.mark.asyncio
async def test_open_session_runs_the_correlated_rpc_with_the_typed_config() -> None:
    result = _ok("open", capabilities=DuplexCapabilities(), public_session={"id": "sid"}, lease_generation=0)
    engine = _engine(result)
    config = DuplexSessionConfig(model="m")

    returned = await engine.open_session_async("sid", config, timeout=3.0)

    assert returned is result
    ((key, message, timeout),) = engine.rpc_client.calls
    assert key == ("duplex", message.control_id)
    assert isinstance(message, OpenDuplexSessionMessage)
    assert message.session_id == "sid" and message.session_config is config
    assert timeout == 3.0


@pytest.mark.asyncio
async def test_close_resume_and_touch_build_their_messages_without_incarnation() -> None:
    engine = _engine(_ok("close"))
    await engine.close_session_async("sid", reason="bye")
    await engine.resume_session_async("sid", expected_lease_generation=4)
    await engine.touch_session_async("sid", activity="detach")

    close, resume, touch = (call[1] for call in engine.rpc_client.calls)
    assert isinstance(close, CloseDuplexSessionMessage) and close.reason == "bye"
    assert isinstance(resume, ResumeDuplexSessionMessage) and resume.expected_lease_generation == 4
    assert isinstance(touch, TouchDuplexSessionMessage) and touch.activity == "detach"
    for message in (close, resume, touch):
        assert message.session_id == "sid"
        assert not hasattr(message, "incarnation")


@pytest.mark.asyncio
async def test_failed_control_result_maps_to_a_typed_session_error() -> None:
    engine = _engine(
        DuplexControlResultMessage(
            control_id="c",
            operation="open",
            session_id="sid",
            ok=False,
            error_code="resource_exhausted",
            error_message="duplex_session_capacity_exhausted: limit=1",
            error_retryable=True,
        )
    )
    with pytest.raises(DuplexSessionError) as excinfo:
        await engine.open_session_async("sid", DuplexSessionConfig())
    assert excinfo.value.code == "resource_exhausted"
    assert excinfo.value.retryable is True
    assert excinfo.value.session_id == "sid"


@pytest.mark.asyncio
async def test_engine_error_and_timeout_map_to_session_errors() -> None:
    # The correlated RPC client raises RuntimeError for a closed router or a
    # terminal orchestrator error; both become a typed engine_error.
    engine = _engine(RuntimeError("orchestrator exploded"))
    with pytest.raises(DuplexSessionError, match="orchestrator exploded") as excinfo:
        await engine.close_session_async("sid")
    assert excinfo.value.code == "engine_error"

    engine = _engine(TimeoutError("duplex open timed out for session sid"))
    with pytest.raises(DuplexSessionError) as excinfo:
        await engine.open_session_async("sid", DuplexSessionConfig())
    assert excinfo.value.code == "timeout" and excinfo.value.retryable is True


@pytest.mark.asyncio
async def test_submit_command_is_one_way_on_the_request_queue() -> None:
    engine = _engine(None)
    command = Heartbeat(event_id="evt-1")

    await engine.submit_command_async("sid", command)

    (message,) = engine.request_queue.sync_q.items
    assert isinstance(message, DuplexSessionCommandMessage)
    assert message.session_id == "sid" and message.command is command
    assert engine.rpc_client.calls == []


@pytest.mark.asyncio
async def test_submit_command_rejects_dead_engine_and_full_queue() -> None:
    with pytest.raises(DuplexSessionError) as excinfo:
        await _engine(None, alive=False).submit_command_async("sid", Heartbeat())
    assert excinfo.value.code == "engine_dead"

    with pytest.raises(DuplexSessionError) as excinfo:
        await _engine(None, full=True).submit_command_async("sid", Heartbeat())
    assert excinfo.value.code == "engine_backpressure" and excinfo.value.retryable is True


def test_duplex_capabilities_come_from_the_loaded_plugin() -> None:
    engine = object.__new__(DuplexOmniEngine)
    engine.plugin = None
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.duplex_capabilities
    capabilities = DuplexCapabilities(supports_barge_in=False)
    engine.plugin = SimpleNamespace(capabilities=lambda *, max_sessions: capabilities)
    engine.duplex_session_config = SimpleNamespace(max_sessions=3)
    assert engine.duplex_capabilities is capabilities


def test_validate_deployment_requires_a_plugin_and_duplex_session_mode() -> None:
    engine = object.__new__(DuplexOmniEngine)
    engine.model = "plain-model"
    engine.pipeline_config = SimpleNamespace(duplex_plugin=None)
    engine.deploy_config = SimpleNamespace(session_mode="duplex")
    with pytest.raises(ValueError, match="not a duplex model"):
        engine._validate_deployment()

    engine.pipeline_config = SimpleNamespace(duplex_plugin="pkg.mod.Plugin")
    engine.deploy_config = SimpleNamespace(session_mode="turn")
    with pytest.raises(ValueError, match="session_mode: duplex"):
        engine._validate_deployment()

    engine.deploy_config = None
    with pytest.raises(ValueError, match="none resolved"):
        engine._validate_deployment()


@pytest.mark.asyncio
async def test_async_wrappers_run_the_blocking_calls_off_the_event_loop() -> None:
    loop = asyncio.get_running_loop()
    engine = _engine(_ok("touch"))
    seen: list[bool] = []
    original = engine.rpc_client.execute

    def execute(*args, **kwargs):
        # Called from the default executor thread, never on the event loop thread.
        try:
            seen.append(asyncio.get_running_loop() is loop)
        except RuntimeError:
            seen.append(False)
        return original(*args, **kwargs)

    engine.rpc_client.execute = execute
    await engine.touch_session_async("sid", activity="heartbeat")
    assert seen == [False]
