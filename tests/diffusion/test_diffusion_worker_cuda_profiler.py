# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from pytest_mock import MockerFixture

from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def mock_od_config(mocker: MockerFixture):
    """Create a mock OmniDiffusionConfig with a CUDA profiler backend."""
    config = mocker.Mock()
    config.diffusion_weight_stack = "legacy"
    config.profiler_config = mocker.Mock()
    config.profiler_config.profiler = "cuda"
    config.diffusion_load_format = "default"
    return config


@pytest.fixture
def mock_diffusion_worker_dependencies(mocker: MockerFixture):
    """Patch heavy worker dependencies for focused profiler tests."""
    workers = []
    mocker.patch.object(DiffusionWorker, "init_device", autospec=True, side_effect=workers.append)
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.DiffusionModelRunner")
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.destroy_distributed_env")
    yield
    for worker in workers:
        worker.shutdown()


class TestDiffusionWorkerCudaProfiler:
    def test_creates_cuda_profiler_wrapper(
        self,
        mocker: MockerFixture,
        mock_od_config,
        mock_diffusion_worker_dependencies,
    ):
        fake_profiler = mocker.Mock()
        cuda_profiler = mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.CudaProfilerWrapper",
            return_value=fake_profiler,
        )
        create_omni_profiler = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.create_omni_profiler")

        worker = DiffusionWorker(local_rank=0, rank=0, od_config=mock_od_config, skip_load_model=True)

        cuda_profiler.assert_called_once_with(mock_od_config.profiler_config)
        create_omni_profiler.assert_not_called()
        assert worker.profiler is fake_profiler

    def test_profile_start_stop_delegates_to_cuda_profiler(
        self,
        mocker: MockerFixture,
        mock_od_config,
        mock_diffusion_worker_dependencies,
    ):
        fake_profiler = mocker.Mock()
        fake_profiler.start = mocker.Mock()
        fake_profiler.stop = mocker.Mock()
        mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.CudaProfilerWrapper",
            return_value=fake_profiler,
        )

        worker = DiffusionWorker(local_rank=0, rank=0, od_config=mock_od_config, skip_load_model=True)

        assert worker.profile(is_start=True) is None
        assert worker.profile(is_start=False) is None

        fake_profiler.start.assert_called_once_with()
        fake_profiler.stop.assert_called_once_with()

    def test_returns_none_when_profiler_config_is_missing(
        self,
        mocker: MockerFixture,
        mock_od_config,
        mock_diffusion_worker_dependencies,
    ):
        mock_od_config.profiler_config = None
        cuda_profiler = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.CudaProfilerWrapper")
        create_omni_profiler = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.create_omni_profiler")

        worker = DiffusionWorker(local_rank=0, rank=0, od_config=mock_od_config, skip_load_model=True)

        cuda_profiler.assert_not_called()
        create_omni_profiler.assert_not_called()
        assert worker.profiler is None

    def test_cuda_backend_does_not_use_torch_profiler_factory(
        self,
        mocker: MockerFixture,
        mock_od_config,
        mock_diffusion_worker_dependencies,
    ):
        mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.CudaProfilerWrapper",
            return_value=mocker.Mock(),
        )
        create_omni_profiler = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.create_omni_profiler")

        DiffusionWorker(local_rank=0, rank=0, od_config=mock_od_config, skip_load_model=True)

        create_omni_profiler.assert_not_called()
