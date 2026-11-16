# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_native_cudagraph_pool_selection_accepts_diffusion_config():
    config = create_diffusion_vllm_config(torch.device("cpu"), OmniDiffusionConfig(model="test"))

    # Native attention initialization reads this property even when sleep is disabled.
    assert config.use_cumem_cudagraph_pool is False
