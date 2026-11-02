# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``DuplexOmniEngine``: the session message surface over the generic engine base."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from pytest_mock import MockerFixture

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.commands import Heartbeat
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    DuplexSessionError,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
    TouchDuplexSessionMessage,
)
from vllm_omni.engine.duplex.plugin import validate_duplex_replica_policy
from vllm_omni.engine.duplex_omni_engine import DuplexOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeRpcClient:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[tuple[str, str], Any, float | None]] = []
        #: Control RPCs must block rather than raise a raw queue.Full.
        self.block_on_submit_flags: list[bool] = []

    def execute(self, key, message, *, timeout=None, timeout_message=None, block_on_submit=False):
        self.calls.append((key, message, timeout))
        self.block_on_submit_flags.append(block_on_submit)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def execute_async(self, key, message, *, timeout=None, timeout_message=None):
        return self.execute(key, message, timeout=timeout, timeout_message=timeout_message)


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
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)

    returned = await engine.open_session_async("sid", config, output_buffer=output_buffer, timeout=3.0)

    assert returned is result
    ((key, message, timeout),) = engine.rpc_client.calls
    assert key == ("duplex", message.control_id)
    assert isinstance(message, OpenDuplexSessionMessage)
    assert message.session_id == "sid" and message.session_config is config
    assert message.output_buffer is output_buffer
    assert timeout == 3.0
    # A momentarily full request queue is backpressure, not a failed open: the
    # control RPC blocks rather than raising an untyped queue.Full at the caller.
    assert engine.rpc_client.block_on_submit_flags == [True]


@pytest.mark.asyncio
async def test_open_uses_the_cold_start_budget_and_controls_keep_the_short_one() -> None:
    # #9072: the first open on a cold engine also initializes per-session runner
    # state, which can exceed the 10 s control budget. Only open gets the longer
    # budget; control ops on a live session keep the short one.
    from vllm_omni.engine.duplex_omni_engine import _DEFAULT_CONTROL_TIMEOUT_S, _DEFAULT_OPEN_TIMEOUT_S

    assert _DEFAULT_OPEN_TIMEOUT_S > _DEFAULT_CONTROL_TIMEOUT_S
    output_buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)

    open_engine = _engine(
        _ok("open", capabilities=DuplexCapabilities(), public_session={"id": "sid"}, lease_generation=0)
    )
    await open_engine.open_session_async("sid", DuplexSessionConfig(), output_buffer=output_buffer)
    assert open_engine.rpc_client.calls[-1][2] == _DEFAULT_OPEN_TIMEOUT_S

    control_engine = _engine(_ok("close"))
    await control_engine.close_session_async("sid")
    assert control_engine.rpc_client.calls[-1][2] == _DEFAULT_CONTROL_TIMEOUT_S


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
        await engine.open_session_async(
            "sid", DuplexSessionConfig(), output_buffer=DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
        )
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
        await engine.open_session_async(
            "sid", DuplexSessionConfig(), output_buffer=DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
        )
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


@pytest.mark.parametrize(("count", "index"), [(1, 0), (4, 0), (4, 3)])
def test_orchestrator_receives_real_frontend_identity_and_exposes_warmup_budget(monkeypatch, count, index) -> None:
    engine = object.__new__(DuplexOmniEngine)
    engine.plugin = object()
    engine.duplex_session_config = DuplexSessionRuntimeConfig(max_sessions=64)
    engine.pipeline_config = SimpleNamespace(stages=[SimpleNamespace(stage_id=0, owns_tokenizer=True)])
    engine.stage_pools = []
    engine._client_config = {"client_count": count, "client_index": index, "stage_addresses": {}}
    received = {}
    orchestrator = SimpleNamespace(
        replica_placement=SimpleNamespace(owned_group_count=2),
        session_manager=SimpleNamespace(admission_limit=16),
    )

    def create(**kwargs):
        received.update(kwargs)
        return orchestrator

    monkeypatch.setattr("vllm_omni.engine.duplex_orchestrator.DuplexOrchestrator", create)
    assert engine._create_orchestrator() is orchestrator
    assert received["frontend_count"] == count
    assert received["frontend_index"] == index
    assert received["duplex_session_config"].max_sessions == 64
    assert engine._client_config["client_count"] == count and engine._client_config["client_index"] == index
    assert engine.duplex_frontend_group_count == 2
    assert engine.duplex_frontend_max_sessions == 16


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


def test_duplex_runtime_uses_tokenizer_owner_model_config(monkeypatch):
    from vllm_omni.model_executor.models.joyai_vl_interaction.pipeline import JOYAI_VL_INTERACTION_ASR_PIPELINE

    engine = object.__new__(DuplexOmniEngine)
    engine.plugin = object()
    engine.pipeline_config = JOYAI_VL_INTERACTION_ASR_PIPELINE
    engine.duplex_session_config = object()
    engine._client_config = None
    configs = [object(), object(), object(), object()]
    engine.stage_pools = [SimpleNamespace(stage_vllm_config=SimpleNamespace(model_config=config)) for config in configs]
    received = {}
    orchestrator = SimpleNamespace(replica_placement=None, session_manager=SimpleNamespace(admission_limit=1))

    def create(**kwargs):
        received.update(kwargs)
        return orchestrator

    monkeypatch.setattr("vllm_omni.engine.duplex_orchestrator.DuplexOrchestrator", create)
    assert engine._create_orchestrator() is orchestrator
    assert received["model_config"] is configs[1]
    assert received["frontend_count"] == 1
    assert received["frontend_index"] == 0


@pytest.mark.parametrize(
    ("replica_policy", "replica_cap", "single_stage_mode", "supports_sharing", "error"),
    [
        ("least_sessions", 1, True, False, "does not support distributed hub/LB mode"),
        ("least_sessions", 2, True, False, "does not support distributed hub/LB mode"),
        ("least_sessions", 2, False, False, "set max_sessions_per_replica to 1"),
        ("least_sessions", 1, False, False, None),
        ("least_sessions", 2, False, True, None),
        ("legacy", None, True, False, None),
        ("legacy", 2, False, False, None),
    ],
)
def test_validate_deployment_checks_replica_policy_before_starting_workers(
    monkeypatch,
    replica_policy: str,
    replica_cap: int | None,
    single_stage_mode: bool,
    supports_sharing: bool,
    error: str | None,
) -> None:
    engine = object.__new__(DuplexOmniEngine)
    engine.model = "fake-duplex-model"
    engine.pipeline_config = SimpleNamespace(duplex_plugin="fake.Plugin")
    runtime_config = DuplexSessionRuntimeConfig(
        max_sessions=11,
        replica_policy=replica_policy,
        max_sessions_per_replica=replica_cap,
    )
    engine.deploy_config = SimpleNamespace(session_mode="duplex", duplex_session=runtime_config)
    engine.single_stage_mode = single_stage_mode
    engine._audio_encoder = object()
    capability_calls: list[int] = []
    configure_pipeline_calls: list[object] = []

    def configure_pipeline(pipeline_config) -> None:
        configure_pipeline_calls.append(pipeline_config)

    def capabilities(*, max_sessions: int) -> DuplexCapabilities:
        capability_calls.append(max_sessions)
        return DuplexCapabilities(supports_multi_session_same_replica=supports_sharing)

    plugin = SimpleNamespace(capabilities=capabilities, configure_pipeline=configure_pipeline)

    def load_plugin(path: str, audio_encoder):
        assert path == "fake.Plugin"
        assert audio_encoder is engine._audio_encoder
        return plugin

    monkeypatch.setattr("vllm_omni.engine.duplex_omni_engine.load_duplex_plugin", load_plugin)
    validation_calls: list[bool] = []

    def validate_policy(candidate_plugin, candidate_config, *, distributed: bool) -> None:
        assert candidate_plugin is plugin
        assert candidate_config is runtime_config
        assert len(configure_pipeline_calls) == 1
        assert configure_pipeline_calls[0] is engine.pipeline_config
        validation_calls.append(distributed)
        validate_duplex_replica_policy(candidate_plugin, candidate_config, distributed=distributed)

    monkeypatch.setattr("vllm_omni.engine.duplex_omni_engine.validate_duplex_replica_policy", validate_policy)

    if error is not None:
        with pytest.raises(ValueError, match=error):
            engine._validate_deployment()
    else:
        engine._validate_deployment()
        assert engine.plugin is plugin
        assert engine.duplex_session_config is runtime_config

    # Validation uses per-replica capacity, never the global admission limit.
    # Single-session replicas and legacy deployments need no sharing support.
    expected_calls = (
        [replica_cap]
        if replica_policy == "least_sessions" and not single_stage_mode and replica_cap is not None and replica_cap > 1
        else []
    )
    assert capability_calls == expected_calls
    assert validation_calls == [single_stage_mode]
    assert len(configure_pipeline_calls) == 1
    assert configure_pipeline_calls[0] is engine.pipeline_config


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


@asynccontextmanager
async def _occupied_default_executor() -> AsyncIterator[None]:
    """Occupy this isolated loop's worker and always release/join our work."""
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    occupied = asyncio.Event()
    release = threading.Event()

    def hold_worker() -> None:
        loop.call_soon_threadsafe(occupied.set)
        if not release.wait(5):
            raise AssertionError("test did not release its own executor worker")

    blocker = loop.run_in_executor(None, hold_worker)
    try:
        await asyncio.wait_for(occupied.wait(), 2)
        yield
    finally:
        release.set()
        await asyncio.wait_for(blocker, 2)
        # A sentinel verifies that any cancelled queued drain was skipped.
        await loop.run_in_executor(None, lambda: None)


def test_drain_does_not_need_a_free_default_executor_worker() -> None:
    async def exercise() -> None:
        engine = _engine(_ok("drain_audio"))
        async with _occupied_default_executor():
            drain = asyncio.create_task(engine.drain_audio_async("sid", timeout=0.05))
            try:
                done, _ = await asyncio.wait({drain}, timeout=0.5)
                assert drain in done, "drain depended on the occupied executor"
                assert await drain is engine.rpc_client.result
            finally:
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)
        assert len(engine.rpc_client.calls) == 1
        assert engine.rpc_client.block_on_submit_flags == [False]

    asyncio.run(exercise())


def test_drain_cancelled_before_start_is_never_submitted() -> None:
    async def exercise() -> None:
        engine = _engine(_ok("drain_audio"))
        async with _occupied_default_executor():
            drain = asyncio.create_task(engine.drain_audio_async("sid", timeout=2))
            try:
                drain.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await drain
            finally:
                drain.cancel()
                await asyncio.gather(drain, return_exceptions=True)
        assert engine.rpc_client.calls == []

    asyncio.run(exercise())


@pytest.mark.asyncio
async def test_drain_forwards_only_the_remaining_budget_to_rpc_and_backend(mocker: MockerFixture) -> None:
    # Replace this module's clock reference, not the asyncio loop's real clock.
    clock = mocker.patch("vllm_omni.engine.duplex_omni_engine.time", autospec=True)
    clock.monotonic.side_effect = [100.0, 100.75]
    result = _ok("drain_audio")
    engine = _engine(result)

    assert await engine.drain_audio_async("sid", timeout=2) is result

    ((_, message, timeout),) = engine.rpc_client.calls
    assert message.timeout_s == timeout == 1.25
    assert engine.rpc_client.block_on_submit_flags == [False]


@pytest.mark.asyncio
async def test_expired_drain_budget_cannot_admit_a_control(mocker: MockerFixture) -> None:
    clock = mocker.patch("vllm_omni.engine.duplex_omni_engine.time", autospec=True)
    clock.monotonic.side_effect = [100.0, 103.0]
    engine = _engine(_ok("drain_audio"))

    with pytest.raises(DuplexSessionError) as error:
        await engine.drain_audio_async("sid", timeout=2)

    assert error.value.code == "timeout" and error.value.retryable
    assert error.value.session_id == "sid"
    assert not engine.rpc_client.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [True, False, 0, -1, float("inf"), float("-inf"), float("nan")])
async def test_drain_rejects_invalid_timeout_before_submission(timeout) -> None:
    engine = _engine(_ok("drain_audio"))
    with pytest.raises(ValueError, match="finite positive timeout"):
        await engine.drain_audio_async("sid", timeout=timeout)
    assert not engine.rpc_client.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "code", "retryable"),
    [
        (queue.Full(), "engine_backpressure", True),
        (TimeoutError("expired"), "timeout", True),
        (RuntimeError("router closed"), "engine_error", False),
        (
            OpenDuplexSessionMessage(
                control_id="c",
                session_id="sid",
                session_config=DuplexSessionConfig(),
                output_buffer=DuplexOutputBuffer(max_bytes=1024, max_events=8),
            ),
            "internal_error",
            False,
        ),
        (
            DuplexControlResultMessage(
                control_id="c",
                operation="drain_audio",
                session_id="sid",
                ok=False,
                error_code="failed_precondition",
                error_message="closed",
                error_retryable=False,
            ),
            "failed_precondition",
            False,
        ),
    ],
)
async def test_async_drain_retains_typed_control_errors(result, code, retryable) -> None:
    engine = _engine(result)
    with pytest.raises(DuplexSessionError) as error:
        await engine.drain_audio_async("sid", timeout=1)
    assert error.value.code == code
    assert error.value.retryable is retryable
    assert error.value.session_id == "sid"


def test_orchestrator_requires_validated_pipeline_configuration() -> None:
    engine = object.__new__(DuplexOmniEngine)
    engine.model = "unresolved-model"
    engine.pipeline_config = None
    with pytest.raises(ValueError, match="pipeline could not be resolved"):
        engine._validate_deployment()
    # Even a supplied plugin cannot bypass the validated-pipeline invariant.
    engine.plugin = object()
    with pytest.raises(AssertionError, match="_validate_deployment"):
        engine._create_orchestrator()
