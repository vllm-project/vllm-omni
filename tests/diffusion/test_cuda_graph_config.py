# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.cuda_graph import DiffusionCUDAGraphConfig

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def test_cuda_graph_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown diffusion cuda_graph_config field"):
        DiffusionCUDAGraphConfig.from_value({"max_graph": 8})


def test_cuda_graph_config_accepts_known_fields() -> None:
    config = DiffusionCUDAGraphConfig.from_value(
        {"max_graphs": 8, "warmup_steps": 0},
        enabled=True,
    )

    assert config.enabled
    assert config.max_graphs == 8
    assert config.warmup_steps == 0
