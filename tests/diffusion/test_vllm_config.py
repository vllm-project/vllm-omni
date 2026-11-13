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


def test_stage_compilation_config_reaches_the_native_config():
    od_config = OmniDiffusionConfig(model="test", compilation_config={"backend": "eager", "cache_dir": "/tmp/omni-cc"})

    config = create_diffusion_vllm_config(torch.device("cpu"), od_config)

    assert isinstance(od_config.compilation_config, CompilationConfig)
    assert config.compilation_config.cache_dir == "/tmp/omni-cc"
    # The native config gets its own copy: vLLM's platform hooks may rewrite its
    # mode and backend (the CPU platform does), and the stage's value, which the
    # model runner compiles with, must keep the user's backend.
    assert config.compilation_config is not od_config.compilation_config
    assert od_config.compilation_config.backend == "eager"
