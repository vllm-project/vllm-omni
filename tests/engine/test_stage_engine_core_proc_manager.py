# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from multiprocessing.process import BaseProcess

import pytest
from vllm.config import ParallelConfig, VllmConfig
from vllm.platforms import current_platform
from vllm.utils import numa_utils
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from vllm_omni.engine import stage_engine_core_proc_manager as module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class CustomUniProcExecutor(UniProcExecutor):
    """An executor extension that retains in-process worker execution."""


@pytest.fixture
def spawn_boundary(mocker, monkeypatch):
    """Keep NUMA argument selection real; intercept only process execution."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.delenv(numa_utils._NUMACTL_ARGS_ENV, raising=False)
    monkeypatch.delenv(numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV, raising=False)
    mocker.patch.object(numa_utils, "_get_numactl_executable", return_value=("/bin/true", "test executable"))
    mocker.patch.object(numa_utils, "_resolve_numactl_args", side_effect=lambda args: args)
    observed = []

    def make_process(**kwargs):
        process = mocker.Mock(spec=BaseProcess)
        process.exitcode = None
        process.child_kwargs = kwargs["kwargs"]
        process.start.side_effect = lambda: observed.append(
            {
                "stage": kwargs["kwargs"]["omni_stage_id"],
                "rank": kwargs["kwargs"]["local_dp_rank"],
                "binding": os.environ.get(numa_utils._NUMACTL_ARGS_ENV),
            }
        )
        return process

    context = mocker.Mock()
    context.Process.side_effect = make_process
    mocker.patch.object(module, "get_mp_context", return_value=context)
    return observed


@pytest.fixture
def auto_numa(monkeypatch):
    """Detect two GPUs per NUMA node through the visibility a stage spawns under."""
    monkeypatch.setattr(numa_utils, "_is_auto_numa_available", lambda: True)
    monkeypatch.setattr(numa_utils, "_maybe_get_pct_cpu_binding", lambda numa_nodes: None)
    monkeypatch.setattr(
        current_platform,
        "get_all_device_numa_nodes",
        lambda: [int(device) // 2 for device in os.environ["CUDA_VISIBLE_DEVICES"].split(",")],
        raising=False,
    )
    numa_utils.get_auto_numa_nodes.cache_clear()
    yield
    numa_utils.get_auto_numa_nodes.cache_clear()


def make_config(executor_class, *, cpus=None, numa_bind=True, dp_size=1, tp_size=1, auto_nodes=False):
    # No model is constructed: the manager consumes only these config fields.
    config = VllmConfig.__new__(VllmConfig)
    config.parallel_config = ParallelConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp" if executor_class is MultiprocExecutor else "uni",
        data_parallel_size=dp_size,
        numa_bind=numa_bind,
        numa_bind_nodes=[0] * max(dp_size, 2) if numa_bind and not auto_nodes else None,
        numa_bind_cpus=cpus if numa_bind else None,
    )
    config.shutdown_timeout = 5.0
    return config


def start_manager(executor_class, *, stage_id, cpus=None, numa_bind=True, dp_size=1, tp_size=1, config=None):
    if config is None:
        config = make_config(executor_class, cpus=cpus, numa_bind=numa_bind, dp_size=dp_size, tp_size=tp_size)
    manager = module.StageEngineCoreProcManager(
        local_engine_count=dp_size,
        start_index=0,
        local_start_index=0,
        vllm_config=config,
        local_client=True,
        handshake_address="ipc:///unused-test-handshake",
        executor_class=executor_class,
        log_stats=False,
        omni_stage_id=stage_id,
    )
    # Fake processes need no finalizer; all patches are fixture-scoped.
    manager._finalizer.detach()
    return manager


@pytest.mark.parametrize("executor_class", [UniProcExecutor, CustomUniProcExecutor])
@pytest.mark.parametrize("stage_id,cpus", [(0, ["2,3"]), (1, ["4-5"])])
def test_in_process_stage_honors_its_worker_cpu_list(spawn_boundary, executor_class, stage_id, cpus):
    start_manager(executor_class, stage_id=stage_id, cpus=cpus)
    assert spawn_boundary == [{"stage": stage_id, "rank": 0, "binding": f"--physcpubind={cpus[0]} --membind=0"}]


@pytest.mark.parametrize("tp_size", [1, 2])
def test_multiprocess_stage_keeps_enginecore_on_the_numa_node(spawn_boundary, tp_size):
    start_manager(MultiprocExecutor, stage_id=0, cpus=["2", "4"], tp_size=tp_size)
    assert spawn_boundary == [{"stage": 0, "rank": 0, "binding": "--cpunodebind=0 --membind=0"}]


def test_in_process_dp_replicas_use_their_local_rank_cpu_lists(spawn_boundary):
    start_manager(UniProcExecutor, stage_id=0, cpus=["2", "4"], dp_size=2)
    assert spawn_boundary == [
        {"stage": 0, "rank": 0, "binding": "--physcpubind=2 --membind=0"},
        {"stage": 0, "rank": 1, "binding": "--physcpubind=4 --membind=0"},
    ]


def test_disabled_numa_binding_leaves_spawn_unbound(spawn_boundary):
    start_manager(UniProcExecutor, stage_id=0, cpus=None, numa_bind=False)
    assert spawn_boundary == [{"stage": 0, "rank": 0, "binding": None}]


def test_in_process_binding_restores_the_parent_environment(spawn_boundary, monkeypatch):
    monkeypatch.setenv(numa_utils._NUMACTL_ARGS_ENV, "--cpunodebind=1")
    monkeypatch.setenv(numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV, "/parent/python")
    start_manager(UniProcExecutor, stage_id=0, cpus=["2"])
    assert spawn_boundary[0]["binding"] == "--physcpubind=2 --membind=0"
    assert os.environ[numa_utils._NUMACTL_ARGS_ENV] == "--cpunodebind=1"
    assert os.environ[numa_utils._NUMACTL_PYTHON_EXECUTABLE_ENV] == "/parent/python"


@pytest.mark.parametrize("executor_class", [UniProcExecutor, MultiprocExecutor])
def test_auto_numa_nodes_follow_each_stage_devices(spawn_boundary, auto_numa, monkeypatch, executor_class):
    managers = []
    for stage_id, devices in [(0, "0"), (1, "2")]:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)
        config = make_config(executor_class, auto_nodes=True)
        managers.append(start_manager(executor_class, stage_id=stage_id, config=config))

    assert [spawn["binding"] for spawn in spawn_boundary] == [
        "--cpunodebind=0 --membind=0",
        "--cpunodebind=1 --membind=1",
    ]
    # A multiprocess stage binds its workers from the nodes its EngineCore receives.
    child_configs = [manager.processes[0].child_kwargs["vllm_config"] for manager in managers]
    assert [config.parallel_config.numa_bind_nodes for config in child_configs] == [[0], [1]]


@pytest.mark.parametrize("executor_class", [UniProcExecutor, MultiprocExecutor])
def test_replicas_sharing_a_config_detect_their_own_numa_nodes(spawn_boundary, auto_numa, monkeypatch, executor_class):
    config = make_config(executor_class, auto_nodes=True)
    for devices in ["0", "2"]:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)
        start_manager(executor_class, stage_id=0, config=config)

    assert [spawn["binding"] for spawn in spawn_boundary] == [
        "--cpunodebind=0 --membind=0",
        "--cpunodebind=1 --membind=1",
    ]
    assert config.parallel_config.numa_bind_nodes is None
