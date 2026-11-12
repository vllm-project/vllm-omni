# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.models.hyperflow_h3_flash.schedule import (
    DEFAULT_STEPS,
    SIGMA_GRIDS,
    SUPPORTED_STEPS,
    hyperflow_schedule,
    sigma_grid,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_default_steps_is_six():
    assert DEFAULT_STEPS == 6
    assert sigma_grid() == SIGMA_GRIDS[6]


@pytest.mark.parametrize("steps", sorted(SUPPORTED_STEPS))
def test_sigma_grid_lengths(steps: int):
    grid = sigma_grid(steps)
    assert len(grid) == steps + 1
    assert grid[0] == 1.0
    assert grid[-1] == 0.0
    assert hyperflow_schedule(steps).num_inference_steps == steps


def test_rejects_unsupported_steps():
    with pytest.raises(ValueError, match="4.*6.*8"):
        sigma_grid(5)
