# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.executor.multiproc_executor as multiproc_executor
from vllm_omni.diffusion.distributed import utils as distributed_utils

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _PipeEndpoint:
    def __init__(self, message=None):
        self.message = message

    def recv(self):
        return self.message

    def close(self):
        pass


class _Process:
    def __init__(self, *, target, args, kwargs=None, name, daemon):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.name = name
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True


def test_get_diffusion_worker_init_method_defaults_to_file_store(mocker):
    """Use the real FileStore helper unless AITER requires TCPStore."""
    mocker.patch.object(distributed_utils, "aiter_requires_tcp_store", return_value=False)

    init_method = distributed_utils.get_diffusion_worker_init_method()

    assert init_method.startswith("file://")


@pytest.mark.parametrize(
    ("use_loopback", "ip_getter", "host"),
    [(True, "get_loopback_ip", "127.0.0.1"), (False, "get_ip", "10.0.0.1")],
)
def test_get_diffusion_worker_init_method_uses_tcp_for_aiter(mocker, use_loopback, ip_getter, host):
    """AITER custom all-reduce needs an explicit TCPStore rendezvous."""
    mocker.patch.object(distributed_utils, "aiter_requires_tcp_store", return_value=True)
    mocker.patch.object(distributed_utils, ip_getter, return_value=host)
    mocker.patch.object(distributed_utils, "get_open_port", return_value=23456)

    init_method = distributed_utils.get_diffusion_worker_init_method(use_loopback=use_loopback)

    assert init_method == f"tcp://{host}:23456"


@pytest.mark.parametrize("num_workers", [1, 2])
def test_launch_workers_forwards_selected_rendezvous(mocker, num_workers: int):
    """Forward one executor-selected rendezvous to every local rank."""
    expected_init_method = "file:///tmp/vllm_dist_test"
    get_init_method = mocker.patch.object(
        multiproc_executor,
        "get_diffusion_worker_init_method",
        return_value=expected_init_method,
    )
    mocker.patch.object(multiproc_executor, "set_multiprocessing_worker_envs")
    mocker.patch.object(multiproc_executor.mp, "set_start_method")
    processes = []

    def make_process(**kwargs):
        process = _Process(**kwargs)
        processes.append(process)
        return process

    mocker.patch.object(multiproc_executor.mp, "Process", side_effect=make_process)

    result_handles = iter(object() for _ in range(num_workers))

    def make_pipe(*, duplex):
        return (
            _PipeEndpoint({"status": "ready", "result_handle": next(result_handles)}),
            _PipeEndpoint(),
        )

    mocker.patch.object(multiproc_executor.mp, "Pipe", side_effect=make_pipe)

    executor = object.__new__(multiproc_executor.MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(
        num_gpus=num_workers,
        worker_extension_cls=None,
        custom_pipeline_args=None,
    )

    launched_processes, handles = executor._launch_workers(
        broadcast_handle=object(), wake_events=[object()] * num_workers
    )

    assert launched_processes == processes
    assert len(handles) == num_workers
    assert all(process.started for process in processes)
    assert [process.kwargs["distributed_init_method"] for process in processes] == [expected_init_method] * num_workers
    get_init_method.assert_called_once_with()
