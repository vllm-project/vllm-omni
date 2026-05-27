# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import os
import threading
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

ray = pytest.importorskip("ray")

from vllm_omni.diffusion.data import DiffusionOutput  # noqa: E402
from vllm_omni.diffusion.executor.ray_executor import (  # noqa: E402
    EXECUTE_MODEL_TIMEOUT,
    RayDiffusionExecutor,
    RayDiffusionWorkerWrapper,
    RayWorkerMetaData,
)
from vllm_omni.diffusion.worker.utils import RunnerOutput  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def mock_od_config(mocker: MockerFixture):
    """Create a mock OmniDiffusionConfig."""
    config = mocker.Mock()
    config.num_gpus = 2
    config.ray_address = None
    config.distributed_executor_backend = "ray"
    config.worker_extension_cls = None
    config.custom_pipeline_args = None
    return config


class TestRayDiffusionWorkerWrapper:
    """Test the Ray actor wrapper."""

    def test_init_worker(self, mocker: MockerFixture, mock_od_config):
        """init_worker should create worker via WorkerWrapperBase."""
        mocker.patch("vllm_omni.plugins.load_omni_general_plugins")
        mock_wrapper_base = mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.WorkerWrapperBase",
        )
        mock_wrapper_base.return_value.worker = mocker.Mock()

        os.environ["RANK"] = "3"
        os.environ["LOCAL_RANK"] = "0"
        try:
            wrapper = RayDiffusionWorkerWrapper(rpc_rank=0)
            wrapper.init_worker(od_config=mock_od_config)

            assert wrapper.rpc_rank == 3
            assert wrapper.worker is not None
            assert wrapper.od_config is mock_od_config
            mock_wrapper_base.assert_called_once_with(
                gpu_id=0,
                od_config=mock_od_config,
                worker_extension_cls=mock_od_config.worker_extension_cls,
                custom_pipeline_args=None,
                rank=3,
            )
        finally:
            del os.environ["RANK"]
            del os.environ["LOCAL_RANK"]

    def test_execute_raises_when_uninitialized(self):
        """execute_model and execute_method should fail before init_worker."""
        wrapper = RayDiffusionWorkerWrapper(rpc_rank=0)
        with pytest.raises(RuntimeError, match="Worker is not initialized"):
            wrapper.execute_model(object())
        with pytest.raises(RuntimeError, match="Worker is not initialized"):
            wrapper.execute_method("any_method")

    def test_get_open_port_returns_actor_local_port(self, mocker: MockerFixture):
        mock_get_open_port = mocker.patch("vllm_omni.diffusion.executor.ray_executor.get_open_port", return_value=23456)

        assert RayDiffusionWorkerWrapper(rpc_rank=0).get_open_port() == "23456"

        mock_get_open_port.assert_called_once_with()

    def test_execute_rpc_skips_non_output_rank(self, mocker: MockerFixture):
        """execute_rpc should not run single-rank RPCs on other ranks."""
        wrapper = RayDiffusionWorkerWrapper(rpc_rank=1)
        wrapper.worker = mocker.Mock()
        wrapper.worker.ping.return_value = "pong"

        assert wrapper.execute_rpc("ping", output_rank=0, exec_all_ranks=False) is None

        wrapper.worker.ping.assert_not_called()

    def test_execute_rpc_executes_all_ranks_but_only_output_rank_replies(self, mocker: MockerFixture):
        """exec_all_ranks should execute locally while suppressing non-output replies."""
        wrapper = RayDiffusionWorkerWrapper(rpc_rank=1)
        wrapper.worker = mocker.Mock()
        wrapper.worker.ping.return_value = "pong"

        assert wrapper.execute_rpc("ping", output_rank=0, exec_all_ranks=True) is None

        wrapper.worker.ping.assert_called_once_with()

    def test_execute_rpc_moves_diffusion_output_to_cpu(self, mocker: MockerFixture):
        """Ray RPC replies should not attempt to serialize GPU tensors."""
        wrapper = RayDiffusionWorkerWrapper(rpc_rank=0)
        wrapper.worker = mocker.Mock()
        output = DiffusionOutput(error="ok")
        to_cpu = mocker.spy(output, "to_cpu")
        wrapper.worker.generate.return_value = output

        result = wrapper.execute_rpc("generate", args=("req",), output_rank=0)

        assert result is output
        wrapper.worker.generate.assert_called_once_with("req")
        to_cpu.assert_called_once_with()


class TestExecutorApi:
    """Test RayDiffusionExecutor implements the current DiffusionExecutor API."""

    def test_executor_is_concrete(self):
        assert not inspect.isabstract(RayDiffusionExecutor)

    def test_execute_request_dispatches_execute_model(self, mocker: MockerFixture, mock_od_config):
        executor = object.__new__(RayDiffusionExecutor)
        executor._closed = False
        executor.od_config = mock_od_config
        result = DiffusionOutput(error="ok")
        executor.collective_rpc = mocker.Mock(return_value=result)
        request = object()
        scheduler_output = SimpleNamespace(
            num_scheduled_reqs=1,
            scheduled_new_reqs=[
                SimpleNamespace(
                    req=request,
                    sched_req_id="sched-req-1",
                )
            ],
        )

        output = RayDiffusionExecutor.execute_request(executor, scheduler_output)

        assert isinstance(output, RunnerOutput)
        assert output.req_id == "sched-req-1"
        assert output.finished is True
        assert output.result is result
        executor.collective_rpc.assert_called_once_with(
            "execute_model",
            args=(request, mock_od_config),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )

    def test_execute_request_rejects_batches(self, mocker: MockerFixture, mock_od_config):
        executor = object.__new__(RayDiffusionExecutor)
        executor._closed = False
        executor.od_config = mock_od_config
        executor.collective_rpc = mocker.Mock()
        scheduler_output = SimpleNamespace(
            num_scheduled_reqs=2,
            scheduled_new_reqs=[],
        )

        with pytest.raises(ValueError, match="batch_size=1"):
            RayDiffusionExecutor.execute_request(executor, scheduler_output)

        executor.collective_rpc.assert_not_called()

    def test_execute_step_dispatches_execute_stepwise(self, mocker: MockerFixture):
        executor = object.__new__(RayDiffusionExecutor)
        executor._closed = False
        expected = RunnerOutput(req_id="sched-step-1", step_index=1, finished=False, result=None)
        executor.collective_rpc = mocker.Mock(return_value=expected)
        scheduler_output = SimpleNamespace(scheduled_req_ids=["sched-step-1"])

        output = RayDiffusionExecutor.execute_step(executor, scheduler_output)

        assert output is expected
        executor.collective_rpc.assert_called_once_with(
            "execute_stepwise",
            args=(scheduler_output,),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )

    def test_execute_step_accepts_legacy_diffusion_output(self, mocker: MockerFixture):
        executor = object.__new__(RayDiffusionExecutor)
        executor._closed = False
        result = DiffusionOutput(error="legacy")
        executor.collective_rpc = mocker.Mock(return_value=result)
        scheduler_output = SimpleNamespace(scheduled_req_ids=["sched-step-1"])

        output = RayDiffusionExecutor.execute_step(executor, scheduler_output)

        assert isinstance(output, RunnerOutput)
        assert output.req_id == "sched-step-1"
        assert output.finished is True
        assert output.result is result

    def test_shutdown_runs_finalizer(self, mocker: MockerFixture):
        executor = object.__new__(RayDiffusionExecutor)
        executor._closed = False
        executor._finalizer = mocker.Mock()

        RayDiffusionExecutor.shutdown(executor)

        assert executor._closed is True
        executor._finalizer.assert_called_once_with()

    def test_init_workers_uses_rank0_actor_port(self, mocker: MockerFixture, mock_od_config):
        executor = object.__new__(RayDiffusionExecutor)
        executor.od_config = mock_od_config
        executor.workers = []
        executor._resources = SimpleNamespace(workers=None)
        mock_od_config.num_gpus = 2

        actors = []

        class RemoteWorkerClass:
            def remote(self, rpc_rank: int):
                actor = mocker.Mock()
                actor.get_node_ip.remote.return_value = f"ip-future-{rpc_rank}"
                actor.get_open_port.remote.return_value = f"port-future-{rpc_rank}"
                actor.update_environment_variables.remote.return_value = f"env-future-{rpc_rank}"
                actor.init_worker.remote.return_value = f"init-future-{rpc_rank}"
                actors.append(actor)
                return actor

        mocker.patch(
            "vllm_omni.diffusion.executor.ray_executor.ray.remote",
            side_effect=lambda **_: lambda _: RemoteWorkerClass(),
        )
        mocker.patch("vllm_omni.diffusion.executor.ray_executor.get_ip", return_value="10.0.0.driver")

        def fake_ray_get(refs, timeout=None):
            del timeout
            if refs == ["ip-future-0", "ip-future-1"]:
                # Neither actor is on the driver. Sorting by IP makes actor 1 rank 0.
                return ["10.0.0.2", "10.0.0.1"]
            if refs == "port-future-1":
                return "23456"
            if isinstance(refs, list):
                return [None for _ in refs]
            raise AssertionError(f"unexpected ray.get input: {refs!r}")

        mocker.patch("vllm_omni.diffusion.executor.ray_executor.ray.get", side_effect=fake_ray_get)

        RayDiffusionExecutor._init_workers_ray(executor, placement_group=mocker.Mock())

        rank0_env = actors[1].update_environment_variables.remote.call_args.args[0]
        rank1_env = actors[0].update_environment_variables.remote.call_args.args[0]
        assert rank0_env["MASTER_ADDR"] == "10.0.0.1"
        assert rank0_env["MASTER_PORT"] == "23456"
        assert rank0_env["RANK"] == "0"
        assert rank1_env["MASTER_ADDR"] == "10.0.0.1"
        assert rank1_env["MASTER_PORT"] == "23456"
        assert rank1_env["RANK"] == "1"


class TestAddReq:
    """Test add_req on RayDiffusionExecutor."""

    @pytest.fixture
    def executor(self, mocker: MockerFixture):
        ex = object.__new__(RayDiffusionExecutor)
        ex._closed = False
        ex.workers = [RayWorkerMetaData(worker=mocker.Mock(), rank=i) for i in range(3)]
        return ex

    def test_add_req_returns_rank0_and_waits_for_followers(self, mocker: MockerFixture, executor):
        request = object()
        expected = DiffusionOutput(error="rank0")
        futures = [f"future-{i}" for i in range(3)]
        for meta, future in zip(executor.workers, futures):
            meta.worker.execute_model.remote.return_value = future
        mock_ray_get = mocker.patch("ray.get", side_effect=[expected, [None, None]])

        result = RayDiffusionExecutor.add_req(executor, request)

        assert result is expected
        for meta in executor.workers:
            meta.worker.execute_model.remote.assert_called_once_with(request)
        assert mock_ray_get.call_args_list[0].args == (futures[0],)
        assert mock_ray_get.call_args_list[0].kwargs == {"timeout": EXECUTE_MODEL_TIMEOUT}
        assert mock_ray_get.call_args_list[1].args == (futures[1:],)
        assert mock_ray_get.call_args_list[1].kwargs == {"timeout": EXECUTE_MODEL_TIMEOUT}

    def test_add_req_closed_executor_raises(self, executor):
        executor._closed = True

        with pytest.raises(RuntimeError, match="closed"):
            RayDiffusionExecutor.add_req(executor, object())

    def test_add_req_rejects_unexpected_rank0_result(self, mocker: MockerFixture, executor):
        for i, meta in enumerate(executor.workers):
            meta.worker.execute_model.remote.return_value = f"future-{i}"
        mocker.patch("ray.get", side_effect=["not-output", [None, None]])

        with pytest.raises(RuntimeError, match="Unexpected response type"):
            RayDiffusionExecutor.add_req(executor, object())


class TestCollectiveRpc:
    """Test collective_rpc on RayDiffusionExecutor."""

    @pytest.fixture
    def executor(self, mocker: MockerFixture):
        ex = object.__new__(RayDiffusionExecutor)
        ex._closed = False
        ex.workers = [RayWorkerMetaData(worker=mocker.Mock(), rank=i) for i in range(3)]
        return ex

    def test_returns_all_responses(self, mocker, executor):
        """collective_rpc should aggregate responses from all workers."""
        expected = ["r0", "r1", "r2"]
        mocker.patch("ray.get", return_value=expected)
        assert executor.collective_rpc("ping") == expected
        for meta in executor.workers:
            meta.worker.execute_rpc.remote.assert_called_once_with(
                "ping",
                (),
                {},
                None,
                True,
            )

    def test_unique_reply_rank(self, mocker, executor):
        """collective_rpc with unique_reply_rank should return single response."""
        mocker.patch("ray.get", return_value=["r0", "r1", "r2"])
        assert executor.collective_rpc("ping", unique_reply_rank=2) == "r2"
        for meta in executor.workers:
            meta.worker.execute_rpc.remote.assert_called_once_with(
                "ping",
                (),
                {},
                2,
                False,
            )

    def test_unique_reply_rank_can_execute_all_ranks(self, mocker, executor):
        """collective_rpc should propagate exec_all_ranks to Ray workers."""
        mocker.patch("ray.get", return_value=["r0", None, None])

        assert executor.collective_rpc("execute_model", unique_reply_rank=0, exec_all_ranks=True) == "r0"
        for meta in executor.workers:
            meta.worker.execute_rpc.remote.assert_called_once_with(
                "execute_model",
                (),
                {},
                0,
                True,
            )

    def test_timeout_raises(self, mocker, executor):
        """collective_rpc should raise TimeoutError on timeout."""
        mock_ray_get = mocker.patch("ray.get", side_effect=ray.exceptions.GetTimeoutError("pg"))
        with pytest.raises(TimeoutError):
            executor.collective_rpc("slow", timeout=1.0)
        _, call_kwargs = mock_ray_get.call_args
        assert call_kwargs.get("timeout") == 1.0

    def test_concurrent_calls_keep_results_with_callers(self, mocker: MockerFixture, executor):
        """Concurrent collective_rpc calls should not mix caller results."""
        for meta in executor.workers:
            meta.worker.execute_rpc.remote.side_effect = (
                lambda method, args, kwargs, output_rank, exec_all_ranks, rank=meta.rank: {
                    "rank": rank,
                    "tag": args[0],
                }
            )

        def fake_ray_get(futures, timeout=None):
            del timeout
            return [f"rank{future['rank']}-{future['tag']}" for future in futures]

        mocker.patch("ray.get", side_effect=fake_ray_get)
        results: dict[str, str] = {}

        def call(tag: str) -> None:
            results[tag] = executor.collective_rpc(
                "ping",
                args=(tag,),
                unique_reply_rank=0,
            )

        threads = [threading.Thread(target=call, args=(f"call-{i}",)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results == {f"call-{i}": f"rank0-call-{i}" for i in range(8)}


class TestExecutorFactory:
    """Test executor class resolution."""

    def test_get_class_returns_ray_executor(self, mock_od_config):
        from vllm_omni.diffusion.executor.abstract import DiffusionExecutor

        cls = DiffusionExecutor.get_class(mock_od_config)
        assert cls is RayDiffusionExecutor
