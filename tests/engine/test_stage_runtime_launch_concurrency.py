# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check replica startup ordering without model-loading timing assumptions."""

import contextlib
import fcntl
import os
import threading
from collections.abc import Iterator
from pathlib import Path

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


def _resources(mocker: MockerFixture) -> StageReplicaResources:
    return StageReplicaResources(
        manager=mocker.Mock(spec=CoreEngineProcManager),
        addresses=EngineZmqAddresses(inputs=["unused-input"], outputs=["unused-output"]),
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
