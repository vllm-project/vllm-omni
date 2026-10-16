# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.executor.multiproc_executor as multiproc_executor

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


@pytest.mark.parametrize(
    ("num_workers", "aiter_tcp_store", "expected_init_method"),
    [
        (2, False, "file:///tmp/vllm_dist_test"),
        (2, True, "env://"),
        (1, False, "file:///tmp/vllm_dist_test"),
    ],
)
def test_launch_workers_uses_expected_rendezvous_method(
    mocker,
    num_workers: int,
    aiter_tcp_store: bool,
    expected_init_method: str,
):
    """Use one FileStore rendezvous for local ranks unless TCPStore is required."""
    mocker.patch.object(multiproc_executor, "aiter_requires_tcp_store", return_value=aiter_tcp_store)
    get_file_store_init_method = mocker.patch.object(
        multiproc_executor,
        "get_file_store_init_method",
        return_value="file:///tmp/vllm_dist_test",
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
    if expected_init_method == "env://":
        get_file_store_init_method.assert_not_called()
    else:
        get_file_store_init_method.assert_called_once_with()
