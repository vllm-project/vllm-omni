# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check replica startup ordering without model-loading timing assumptions."""

import contextlib
import fcntl
import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NoReturn

import pytest
from pytest_mock import MockerFixture
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.utils import CoreEngineProcManager, EngineZmqAddresses

from vllm_omni.config.stage_config import StageConfig
from vllm_omni.engine import stage_init_utils, stage_runtime
from vllm_omni.engine.stage_engine_startup import StageReplicaResources
from vllm_omni.engine.stage_init_utils import LogicalStageInitPlan, ReplicaInitPlan, StageMetadata

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def launch_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mocker: MockerFixture
) -> Iterator[stage_runtime.StageRuntime]:
    runtime = stage_runtime.StageRuntime(
        stage_configs=[],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=5,
        async_chunk=False,
    )
    # These are virtual device IDs: only the real per-device file locking and
    # scheduling run. No model, CUDA context, or engine process is created.
    monkeypatch.setattr(runtime, "_resolve_replica_physical_devices", lambda _stage, cfg: cfg["devices"])
    monkeypatch.setattr(
        stage_init_utils, "device_init_lock_path", lambda device: str(tmp_path / f"device-{device}.lock")
    )
    mocker.patch.object(
        stage_runtime.StageEngineCoreClientBase,
        "make_async_mp_client",
        return_value=mocker.Mock(spec=stage_runtime.StageClient),
    )
    monkeypatch.setenv("VLLM_OMNI_TEST_LAUNCH_ENV", "parent")
    yield runtime
    if runtime._stage_init_executor is not None:
        runtime._stage_init_executor.shutdown(wait=True)


def _plan(stage_id: int, device: str) -> LogicalStageInitPlan:
    metadata = StageMetadata(
        stage_id=stage_id,
        stage_type="llm",
        engine_output_type="text",
        is_comprehension=False,
        requires_multimodal_data=False,
        engine_input_source=[],
        final_output=False,
        final_output_type=None,
        default_sampling_params=SamplingParams(),
        custom_process_input_func=None,
        model_stage="dummy",
        runtime_cfg={"devices": device, "env": {"VLLM_OMNI_TEST_LAUNCH_ENV": str(stage_id)}},
    )
    replica = ReplicaInitPlan(
        replica_id=0,
        num_replicas=1,
        launch_mode="local",
        stage_cfg=StageConfig(stage_id=stage_id, model_stage="dummy", yaml_runtime=metadata.runtime_cfg),
        metadata=metadata,
        stage_connector_spec={},
        omni_kv_connector=(None, None, None),
        stage_vllm_config=VllmConfig(),
        executor_class=object,
        engine_args_dict={},
    )
    return LogicalStageInitPlan(stage_idx=stage_id, stage_id=stage_id, replicas=[replica])


def _resources(mocker: MockerFixture, client_count: int = 1) -> StageReplicaResources:
    return StageReplicaResources(
        manager=mocker.Mock(spec=CoreEngineProcManager),
        addresses=EngineZmqAddresses(
            inputs=[f"ipc://unused-input-{index}" for index in range(client_count)],
            outputs=[f"ipc://unused-output-{index}" for index in range(client_count)],
        ),
    )


def test_distinct_devices_spawn_before_either_is_ready(
    launch_runtime: stage_runtime.StageRuntime, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    spawned = [threading.Event(), threading.Event()]

    @contextlib.contextmanager
    def launch(*, stage_id: int, stage_visible_devices: str, **_kwargs: object) -> Iterator[StageReplicaResources]:
        assert os.environ["VLLM_OMNI_TEST_LAUNCH_ENV"] == str(stage_id)
        spawned[stage_id].set()
        yield _resources(mocker)
        # The device remains exclusively locked throughout loading/profiling,
        # even though another device's replica must be able to start.
        with open(stage_init_utils.device_init_lock_path(int(stage_visible_devices)), "r+") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert spawned[1 - stage_id].wait(5), "A different device's spawn was blocked by the READY wait"

    monkeypatch.setattr(stage_runtime, "launch_stage_replica", launch)
    clients = launch_runtime._initialize_stage_replicas([_plan(0, "0"), _plan(1, "1")], 5)
    assert all(clients[stage][0] is not None for stage in (0, 1))
    assert os.environ["VLLM_OMNI_TEST_LAUNCH_ENV"] == "parent"


def test_same_device_waits_for_ready_before_next_spawn(
    launch_runtime: stage_runtime.StageRuntime, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    events: list[tuple[int, str]] = []

    @contextlib.contextmanager
    def launch(*, stage_id: int, **_kwargs: object) -> Iterator[StageReplicaResources]:
        events.append((stage_id, "spawn"))
        yield _resources(mocker)
        events.append((stage_id, "ready"))

    monkeypatch.setattr(stage_runtime, "launch_stage_replica", launch)
    launch_runtime._initialize_stage_replicas([_plan(0, "0"), _plan(1, "0")], 5)
    assert events == [(0, "spawn"), (0, "ready"), (1, "spawn"), (1, "ready")]
    assert os.environ["VLLM_OMNI_TEST_LAUNCH_ENV"] == "parent"


def test_ready_failure_cleans_up_resources_and_releases_device(
    launch_runtime: stage_runtime.StageRuntime, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    manager = mocker.Mock(spec=CoreEngineProcManager)
    resources = StageReplicaResources(manager=manager, addresses=EngineZmqAddresses(inputs=[], outputs=[]))

    @contextlib.contextmanager
    def launch(**_kwargs: object) -> Iterator[StageReplicaResources]:
        yield resources
        raise RuntimeError("READY handshake failed")

    monkeypatch.setattr(stage_runtime, "launch_stage_replica", launch)
    with pytest.raises(RuntimeError, match="READY handshake failed"):
        launch_runtime._initialize_local_llm_replica(_plan(0, "0").replicas[0], 5)

    manager.shutdown.assert_called_once_with()
    assert os.environ["VLLM_OMNI_TEST_LAUNCH_ENV"] == "parent"
    with open(stage_init_utils.device_init_lock_path(0), "r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _assert_devices_unlocked(*devices: str) -> None:
    for device in sorted({int(part) for device_list in devices for part in device_list.split(",")}):
        with open(stage_init_utils.device_init_lock_path(device), "r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def test_multi_api_disjoint_devices_complete_readiness_concurrently(
    launch_runtime: stage_runtime.StageRuntime, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    plans = [_plan(0, "0"), _plan(1, "1")]
    resources = [_resources(mocker, 2) for _ in plans]
    ready_barrier = threading.Barrier(2, timeout=5)
    events: list[tuple[int, str]] = []
    launch_kwargs: list[dict[str, object]] = []
    frontend = object()

    @contextlib.contextmanager
    def launch(*, stage_id: int, **kwargs: object) -> Iterator[StageReplicaResources]:
        assert os.environ["VLLM_OMNI_TEST_LAUNCH_ENV"] == str(stage_id)
        events.append((stage_id, "spawn"))
        launch_kwargs.append(kwargs)
        yield resources[stage_id]
        assert not launch_runtime._replica_launch_lock.locked()
        assert os.environ["VLLM_OMNI_TEST_LAUNCH_ENV"] == "parent"
        assert kwargs["watched_frontend_processes"] == [frontend]
        # Locks must outlive all READY waits, including waits on another GPU.
        for device in (0, 1):
            with open(stage_init_utils.device_init_lock_path(device), "r+") as lock:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        events.append((stage_id, "ready-start"))
        ready_barrier.wait()
        events.append((stage_id, "ready-done"))

    monkeypatch.setattr(stage_runtime, "launch_stage_replica", launch)
    with launch_runtime.launch_stage_engines(2, stage_plans=plans) as engine_launch:
        assert events == [(0, "spawn"), (1, "spawn")]
        assert engine_launch.resources == resources
        for config in engine_launch.client_configs:
            assert set(config["stage_addresses"]) == {0, 1}
        assert all(
            kwargs["watched_frontend_processes"] is engine_launch.watched_frontend_processes for kwargs in launch_kwargs
        )
        engine_launch.watched_frontend_processes.append(frontend)

    assert sorted(event for event in events if event[1] == "ready-done") == [(0, "ready-done"), (1, "ready-done")]
    for resource in resources:
        resource.manager.shutdown.assert_not_called()
    _assert_devices_unlocked("0", "1")


@pytest.mark.parametrize("overlapping_devices", [("0", "0"), ("0,1", "1"), ("0", "0,1", "1")])
def test_multi_api_overlapping_readiness_is_serial_while_other_devices_progress(
    launch_runtime: stage_runtime.StageRuntime,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    overlapping_devices: tuple[str, ...],
) -> None:
    devices = (*overlapping_devices, "2")
    plans = [_plan(stage_id, device) for stage_id, device in enumerate(devices)]
    resources = [_resources(mocker, 2) for _ in plans]
    independent_stage = len(overlapping_devices)
    concurrent_groups = threading.Barrier(2, timeout=5)
    completed: set[int] = set()
    overlap_events: list[tuple[int, str]] = []

    @contextlib.contextmanager
    def launch(*, stage_id: int, **_kwargs: object) -> Iterator[StageReplicaResources]:
        yield resources[stage_id]
        if stage_id < independent_stage:
            if stage_id > 0:
                assert stage_id - 1 in completed, "Overlapping replicas entered READY concurrently"
            overlap_events.append((stage_id, "ready-start"))
        if stage_id in (0, independent_stage):
            concurrent_groups.wait()
        completed.add(stage_id)
        if stage_id < independent_stage:
            overlap_events.append((stage_id, "ready-done"))

    monkeypatch.setattr(stage_runtime, "launch_stage_replica", launch)
    with launch_runtime.launch_stage_engines(2, stage_plans=plans):
        assert not completed

    assert overlap_events == [
        (stage_id, event) for stage_id in range(independent_stage) for event in ("ready-start", "ready-done")
    ]
    assert completed == set(range(len(plans)))
    _assert_devices_unlocked(*devices)


def test_multi_api_parallel_stage_init_keeps_same_device_readiness_parallel(
    launch_runtime: stage_runtime.StageRuntime, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    launch_runtime._parallel_stage_init = True
    plans = [_plan(0, "0"), _plan(1, "0")]
    resources = [_resources(mocker, 2) for _ in plans]
    ready_barrier = threading.Barrier(2, timeout=5)
    events: list[str] = []
    monkeypatch.setattr(launch_runtime, "_prepare_stage_plans", lambda: plans)
    monkeypatch.setattr(launch_runtime, "_reject_unguardable_executors", lambda _: events.append("guard"))
    monkeypatch.setattr(launch_runtime, "_run_stage_admission", lambda _: events.append("admission"))

    def forbidden_lock(*_args: object, **_kwargs: object) -> NoReturn:
        fail: Callable[[str], NoReturn] = pytest.fail
        fail("Parallel initialization must leave device phase locks to engine workers")

    monkeypatch.setattr(stage_runtime, "acquire_device_locks", forbidden_lock)

    @contextlib.contextmanager
    def launch(*, stage_id: int, **kwargs: object) -> Iterator[StageReplicaResources]:
        assert events[:2] == ["guard", "admission"]
        assert kwargs["omni_parallel_stage_init"] is True
        events.append(f"spawn-{stage_id}")
        yield resources[stage_id]
        assert not launch_runtime._replica_launch_lock.locked()
        ready_barrier.wait()
        events.append(f"ready-{stage_id}")

    monkeypatch.setattr(stage_runtime, "launch_stage_replica", launch)
    with launch_runtime.launch_stage_engines(2):
        assert events == ["guard", "admission", "spawn-0", "spawn-1"]

    assert sorted(events[4:]) == ["ready-0", "ready-1"]


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_multi_api_readiness_failure_stops_engines_before_joining_and_closes_unstarted_contexts(
    launch_runtime: stage_runtime.StageRuntime,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    failure_type: type[BaseException],
) -> None:
    # The first group blocks until engine shutdown. The second group fails on
    # its first replica; its next replica must be unwound without starting READY.
    devices = ("0", "1", "1")
    plans = [_plan(stage_id, device) for stage_id, device in enumerate(devices)]
    resources = [_resources(mocker, 2) for _ in plans]
    first_waiting = threading.Event()
    first_stopped = threading.Event()
    error = failure_type("READY handshake failed")
    events: list[tuple[int, str]] = []
    exceptional_exits: dict[int, BaseException] = {}

    for stage_id, resource in enumerate(resources):

        def shutdown(stage_id: int = stage_id) -> None:
            events.append((stage_id, "shutdown"))
            if stage_id == 0:
                first_stopped.set()

        resource.manager.shutdown.side_effect = shutdown

    @contextlib.contextmanager
    def launch(*, stage_id: int, **_kwargs: object) -> Iterator[StageReplicaResources]:
        try:
            yield resources[stage_id]
            events.append((stage_id, "ready-start"))
            if stage_id == 0:
                first_waiting.set()
                assert first_stopped.wait(5), "Waiting for READY joined a worker before shutting down its engine"
            elif stage_id == 1:
                assert first_waiting.wait(5)
                raise error
            else:
                pytest.fail("A later replica started READY after its group failed")
        except BaseException as exc:
            exceptional_exits[stage_id] = exc
            raise
        finally:
            events.append((stage_id, "context-exit"))

    monkeypatch.setattr(stage_runtime, "launch_stage_replica", launch)
    with pytest.raises(failure_type, match="READY handshake failed") as exc_info:
        with launch_runtime.launch_stage_engines(2, stage_plans=plans):
            pass

    assert exc_info.value is error
    assert exceptional_exits[1] is error
    assert exceptional_exits[2] is error
    assert (2, "ready-start") not in events
    assert events.index((0, "shutdown")) < events.index((0, "context-exit"))
    assert events.index((2, "shutdown")) < events.index((2, "context-exit"))
    for stage_id, resource in enumerate(resources):
        resource.manager.shutdown.assert_called_once_with()
        assert events.count((stage_id, "context-exit")) == 1
    assert os.environ["VLLM_OMNI_TEST_LAUNCH_ENV"] == "parent"
    _assert_devices_unlocked(*devices)


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_multi_api_caller_failure_unwinds_contexts_without_starting_readiness(
    launch_runtime: stage_runtime.StageRuntime,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    failure_type: type[BaseException],
) -> None:
    devices = ("0", "1")
    plans = [_plan(stage_id, device) for stage_id, device in enumerate(devices)]
    resources = [_resources(mocker, 2) for _ in plans]
    error = failure_type("Frontend launch interrupted")
    exceptional_exits: dict[int, BaseException] = {}

    @contextlib.contextmanager
    def launch(*, stage_id: int, **_kwargs: object) -> Iterator[StageReplicaResources]:
        try:
            yield resources[stage_id]
            pytest.fail("Failed frontend launch must not start engine readiness")
        except BaseException as exc:
            assert all(resource.manager.shutdown.called for resource in resources)
            exceptional_exits[stage_id] = exc
            raise

    monkeypatch.setattr(stage_runtime, "launch_stage_replica", launch)
    with pytest.raises(failure_type) as exc_info:
        with launch_runtime.launch_stage_engines(2, stage_plans=plans):
            raise error

    assert exc_info.value is error
    assert exceptional_exits == {0: error, 1: error}
    for resource in resources:
        resource.manager.shutdown.assert_called_once_with()
    assert os.environ["VLLM_OMNI_TEST_LAUNCH_ENV"] == "parent"
    _assert_devices_unlocked(*devices)
