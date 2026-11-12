# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib
import multiprocessing
from dataclasses import dataclass
from typing import cast

import pytest

from tests.distributed.weight_sharing.helpers import load_class_methods, load_weight_sharing

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Spawn must be able to resolve the registry's test-only module before unpickling.
load_weight_sharing()


@dataclass
class _StartupConfig:
    """Only startup fields, with the same ordinary dataclass pickle transport."""

    num_gpus: int = 2
    worker_extension_cls: str | None = None


def _startup_worker(rank, config, writer, broadcast, wake, extension, pipeline_args, *, sharing_registry=None):
    # Exercise real spawn transport and shared registration without initializing CUDA.
    assert not hasattr(config, "_sharing_worker_registry")
    if sharing_registry is not None:
        assert sharing_registry.register(rank)
    writer.send({"status": "ready", "result_handle": sharing_registry is not None})
    writer.close()
    assert wake.wait(30)


@pytest.mark.parametrize("sharing", [False, True])
def test_startup_registry_stays_out_of_request_rpc(mocker, sharing):
    registry_cls = importlib.import_module(f"{load_weight_sharing().__name__}.worker_registry").ConsumerWorkerRegistry
    ctx = multiprocessing.get_context("spawn")
    registry = registry_cls(ctx, 2) if sharing else None
    config = _StartupConfig()
    if sharing:
        setattr(config, "_sharing_worker_registry", registry)
    executor_cls = load_class_methods(
        "vllm_omni/diffusion/executor/multiproc_executor.py",
        "MultiprocDiffusionExecutor",
        ["_launch_workers"],
        dict(
            mp=ctx,
            cast=cast,
            logger=mocker.Mock(),
            set_multiprocessing_worker_envs=lambda: None,
            WorkerProc=mocker.Mock(worker_main=_startup_worker),
        ),
    )
    # Production chooses spawn; the isolated context already uses it.
    mocker.patch.object(ctx, "set_start_method")
    executor = executor_cls()
    executor.od_config = config
    wake_events = [ctx.Event(), ctx.Event()]
    processes, handles = executor._launch_workers(None, wake_events)
    rpc_reader, rpc_writer = ctx.Pipe(duplex=False)
    try:
        # Pipe.send uses ordinary multiprocessing serialization outside spawn,
        # reproducing the same shared-array restriction as RPC enqueue.
        for method, args in (
            ("execute_model", ("request", config, None)),
            ("execute_model_batch", ("scheduler_output", config)),
        ):
            rpc_writer.send({"method": method, "args": args})
            message = rpc_reader.recv()
            assert message["args"][1].num_gpus == 2
        assert handles == [sharing, sharing]
        assert not hasattr(config, "_sharing_worker_registry")
        if sharing:
            assert list(registry._data[1::2]) == [process.pid for process in processes]
    finally:
        rpc_reader.close()
        rpc_writer.close()
        for wake in wake_events:
            wake.set()
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join(10)
    assert all(process.exitcode == 0 for process in processes)


def test_worker_checks_startup_admission_before_loading_plugins(mocker):
    worker_cls = load_class_methods("vllm_omni/diffusion/worker/diffusion_worker.py", "WorkerProc", ["worker_main"], {})
    registry = mocker.Mock()
    registry.register.return_value = False
    worker_cls.worker_main(1, _StartupConfig(), None, None, None, sharing_registry=registry)
    registry.register.assert_called_once_with(1)
