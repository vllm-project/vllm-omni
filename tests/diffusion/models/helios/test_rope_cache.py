# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import gc
import weakref

import pytest

from vllm_omni.diffusion.models.helios.helios_transformer import HeliosRotaryPosEmbed

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_spatial_meshgrid_cache_does_not_outlive_module():
    rope = HeliosRotaryPosEmbed(rope_dim=(2, 2, 2), theta=10_000)
    first = rope._get_spatial_meshgrid(2, 2, "cpu")
    assert rope._get_spatial_meshgrid(2, 2, "cpu") is first
    ref = weakref.ref(rope)

    del rope
    gc.collect()

    assert ref() is None
