# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class TestDiffusionWorkerDistributedEnv:
    """Test worker distributed environment setup."""

    def test_mp_backend_overrides_ambient_master_env(self, monkeypatch):
        monkeypatch.setenv("MASTER_ADDR", "stale-host")
        monkeypatch.setenv("MASTER_PORT", "11111")
        worker = object.__new__(DiffusionWorker)
        worker.local_rank = 2
        worker.od_config = SimpleNamespace(
            distributed_executor_backend="mp",
            master_port=12345,
        )

        worker._setup_distributed_env_vars(world_size=4, rank=2)

        assert os.environ["MASTER_ADDR"] == "localhost"
        assert os.environ["MASTER_PORT"] == "12345"
        assert os.environ["LOCAL_RANK"] == "2"
        assert os.environ["RANK"] == "2"
        assert os.environ["WORLD_SIZE"] == "4"

    def test_ray_backend_preserves_executor_supplied_master_env(self, monkeypatch):
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "23456")
        worker = object.__new__(DiffusionWorker)
        worker.local_rank = 0
        worker.od_config = SimpleNamespace(
            distributed_executor_backend="ray",
            master_port=12345,
        )

        worker._setup_distributed_env_vars(world_size=2, rank=1)

        assert os.environ["MASTER_ADDR"] == "10.0.0.1"
        assert os.environ["MASTER_PORT"] == "23456"
        assert os.environ["LOCAL_RANK"] == "0"
        assert os.environ["RANK"] == "1"
        assert os.environ["WORLD_SIZE"] == "2"
