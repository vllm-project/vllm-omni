# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.config import CompilationConfig

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_native_cudagraph_pool_selection_accepts_diffusion_config():
    config = create_diffusion_vllm_config(torch.device("cpu"), OmniDiffusionConfig(model="test"))

    # Native attention initialization reads this property even when sleep is disabled.
    assert config.use_cumem_cudagraph_pool is False


def test_native_config_keeps_vllm_compilation_defaults():
    """Only the stage's ``backend`` applies to diffusion, in the model runner.

    A global ``--compilation-config`` also reaches diffusion stages of a mixed
    pipeline; its other settings (custom ops, IR wrapping, mode) must not change
    which kernels the diffusion stage's native layers use.
    """
    od_config = OmniDiffusionConfig(model="test", compilation_config={"backend": "eager", "custom_ops": ["all"]})

    config = create_diffusion_vllm_config(torch.device("cpu"), od_config)
    default = create_diffusion_vllm_config(torch.device("cpu"), OmniDiffusionConfig(model="test"))

    assert isinstance(od_config.compilation_config, CompilationConfig)
    assert od_config.compilation_config.backend == "eager"
    assert config.compilation_config.custom_ops == default.compilation_config.custom_ops
    assert config.compilation_config.ir_enable_torch_wrap == default.compilation_config.ir_enable_torch_wrap
