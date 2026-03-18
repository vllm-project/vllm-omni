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
    return config


class TestRayDiffusionWorkerWrapper:
    """Test the Ray actor wrapper."""

    def test_init_worker(self, mocker: MockerFixture, mock_od_config):
        """init_worker should create DiffusionWorker with env-based rank."""
        mocker.patch("vllm_omni.plugins.load_omni_general_plugins")
        mock_worker_cls = mocker.patch(
            "vllm_omni.diffusion.worker.DiffusionWorker",
        )

        os.environ["RANK"] = "3"
        os.environ["LOCAL_RANK"] = "0"
        try:
            wrapper = RayDiffusionWorkerWrapper(rpc_rank=0)
            wrapper.init_worker(od_config=mock_od_config)

            assert wrapper.rpc_rank == 3
            assert wrapper.worker is not None
            assert wrapper.od_config is mock_od_config
            mock_worker_cls.assert_called_once_with(
                local_rank=0, rank=3, od_config=mock_od_config,
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


class TestExecutorFactory:
    """Test executor class resolution."""

    def test_get_class_returns_ray_executor(self, mock_od_config):
        from vllm_omni.diffusion.executor.abstract import DiffusionExecutor

        cls = DiffusionExecutor.get_class(mock_od_config)
        assert cls is RayDiffusionExecutor
