# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_get_diffusion_od_config_returns_direct_config():
    diffusion = object.__new__(AsyncOmniDiffusion)
    diffusion.od_config = object()

    assert diffusion.get_diffusion_od_config() is diffusion.od_config
