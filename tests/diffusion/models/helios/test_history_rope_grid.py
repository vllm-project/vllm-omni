# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Each history component must derive its own rotary grid.

The mid/long branches used to reuse ``H1``/``W1`` captured from the short
branch, so a request that supplied mid/long history without short history
raised ``UnboundLocalError`` inside the transformer forward.
"""

import inspect

import pytest

from vllm_omni.diffusion.models.helios.helios_transformer import (
    HeliosTransformer3DModel,
    history_rope_grid,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_short_grid_equals_patch_grid():
    # patch_short uses stride (1, 2, 2) and is not downsampled.
    assert history_rope_grid(48, 80) == (24, 40)


def test_mid_grid_matches_patch_output_after_downsample():
    height, width = history_rope_grid(48, 80)
    assert (height // 2, width // 2) == (12, 20)  # patch_mid stride (2, 4, 4)


def test_long_grid_matches_patch_output_after_downsample():
    height, width = history_rope_grid(48, 80)
    assert (height // 4, width // 4) == (6, 10)  # patch_long stride (4, 8, 8)


def test_padded_grid_tracks_padding():
    # pad_for_3d_conv runs before the grid is taken, so a 50-row latent padded
    # to 52 for the (2, 4, 4) mid patch yields a 26-row rotary grid.
    assert history_rope_grid(52, 80)[0] // 2 == 13


def test_forward_does_not_capture_short_branch_locals():
    source = inspect.getsource(HeliosTransformer3DModel.forward)
    assert "H1" not in source
    assert "height=short_height" in source
    assert "height=mid_height" in source
    assert "height=long_height" in source
