# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
from pytest_mock import MockerFixture

ray = pytest.importorskip("ray")

from vllm_omni.diffusion.executor.ray_executor import (  # noqa: E402
    RayDiffusionExecutor,
    RayDiffusionWorkerWrapper,
)

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


class TestCollectiveRpc:
    """Test collective_rpc on RayDiffusionExecutor."""

    @pytest.fixture
    def executor(self, mocker: MockerFixture):
        from vllm_omni.diffusion.executor.ray_executor import RayWorkerMetaData

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
            meta.worker.execute_method.remote.assert_called_once_with("ping")

    def test_unique_reply_rank(self, mocker, executor):
        """collective_rpc with unique_reply_rank should return single response."""
        mocker.patch("ray.get", return_value=["r0", "r1", "r2"])
        assert executor.collective_rpc("ping", unique_reply_rank=2) == "r2"

    def test_timeout_raises(self, mocker, executor):
        """collective_rpc should raise TimeoutError on timeout."""
        mock_ray_get = mocker.patch("ray.get", side_effect=ray.exceptions.GetTimeoutError("pg"))
        with pytest.raises(TimeoutError):
            executor.collective_rpc("slow", timeout=1.0)
        _, call_kwargs = mock_ray_get.call_args
        assert call_kwargs.get("timeout") == 1.0


class TestExecutorFactory:
    """Test executor class resolution."""

    def test_get_class_returns_ray_executor(self, mock_od_config):
        from vllm_omni.diffusion.executor.abstract import DiffusionExecutor

        cls = DiffusionExecutor.get_class(mock_od_config)
        assert cls is RayDiffusionExecutor
