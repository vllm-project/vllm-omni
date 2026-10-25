# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor
from vllm_omni.diffusion.models.hy_nomas import pipeline_hy_nomas

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class CustomUniExecutor(UniProcDiffusionExecutor):
    pass


@pytest.mark.parametrize(
    "backend",
    [
        None,
        "uni",
        UniProcDiffusionExecutor,
        CustomUniExecutor,
        "vllm_omni.diffusion.executor.uniproc_executor.UniProcDiffusionExecutor",
    ],
)
def test_model_rejects_in_process_execution_before_initializing_components(monkeypatch, backend):
    device = Mock(side_effect=AssertionError("Component initialization must not start"))
    monkeypatch.setattr(pipeline_hy_nomas, "get_local_device", device)
    config = OmniDiffusionConfig(dtype=torch.bfloat16, distributed_executor_backend=backend)
    with pytest.raises(ValueError, match="requires process isolation"):
        pipeline_hy_nomas.HyNomasPipeline(od_config=config)
    device.assert_not_called()


@pytest.mark.parametrize("backend", ["mp", "ray"])
def test_model_accepts_isolated_executor(backend):
    pipeline_hy_nomas.validate_config(OmniDiffusionConfig(dtype=torch.bfloat16, distributed_executor_backend=backend))
