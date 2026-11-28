# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the LingBot-Video DMD student scheduler (#11260)."""

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.schedulers import DMDStudentScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_sigma_grid_matches_the_distilled_recipe() -> None:
    """8 steps / shift 3 reproduce the trained grid bit-for-bit."""
    sched = DMDStudentScheduler()
    sched.set_timesteps(8)
    expected = [
        0.99999899,
        0.95459503,
        0.90011996,
        0.83355546,
        0.75037479,
        0.64346898,
        0.50099897,
        0.30167764,
        0.0,
    ]
    got = [round(float(s), 8) for s in sched.sigmas]
    assert got == expected
    # Integer-truncated timesteps from the same grid.
    assert sched.timesteps.tolist() == [
        999,
        954,
        900,
        833,
        750,
        643,
        500,
        301,
    ]


def test_low_noise_step_is_deterministic_euler() -> None:
    """Below the threshold the step is the plain Euler ODE (no RNG draw)."""
    sched = DMDStudentScheduler()
    sched.set_timesteps(8)
    # Jump to step 7 (sigma 0.30167764, below the 0.5 gate).
    sched._step_index = 7
    sigma, sigma_next = float(sched.sigmas[7]), float(sched.sigmas[8])
    v = torch.randn(2, 3, 4)
    z = torch.randn(2, 3, 4)
    out = sched.step(v, sched.timesteps[7], z, generator=torch.Generator().manual_seed(0))[0]
    assert torch.allclose(out, z + v * (sigma_next - sigma), atol=1e-6)


def test_high_noise_step_is_ddim_eta_ancestral() -> None:
    """Above the threshold the step adds exactly the DDIM ancestral std of noise."""
    sched = DMDStudentScheduler()
    sched.set_timesteps(8)
    sched._step_index = 0
    gen = torch.Generator().manual_seed(42)
    v = torch.randn(2, 3, 4, generator=gen)
    z = torch.randn(2, 3, 4, generator=gen)
    noise = torch.randn(2, 3, 4, generator=gen)
    sigma, sigma_next = float(sched.sigmas[0]), float(sched.sigmas[1])
    x0 = z - sigma * v
    x1 = z + v * (1 - sigma)
    from vllm_omni.diffusion.models.schedulers.scheduling_dmd_student import _ddim_ancestral_std

    std = float(_ddim_ancestral_std(torch.tensor(sigma), torch.tensor(sigma_next), eta=1.0))
    expected = x0 * (1 - sigma_next) + x1 * (sigma_next**2 - std**2) ** 0.5 + std * noise
    # A second, identically-seeded generator replays the same noise stream the
    # step's randn_tensor call consumes.
    replay = torch.Generator().manual_seed(42)
    _ = torch.randn(2, 3, 4, generator=replay)
    _ = torch.randn(2, 3, 4, generator=replay)
    out = sched.step(v, sched.timesteps[0], z, generator=replay)[0]
    assert torch.allclose(out, expected, atol=1e-5)


def test_custom_sigmas_and_mu_are_rejected() -> None:
    """The student was distilled on its own grid; refiners cannot override it."""
    sched = DMDStudentScheduler()
    with pytest.raises(ValueError, match="own sigma grid"):
        sched.set_timesteps(8, sigmas=np.linspace(1, 0, 9))
    with pytest.raises(ValueError, match="dynamic shifting"):
        sched.set_timesteps(8, mu=1.5)


def test_step_without_set_timesteps_raises() -> None:
    sched = DMDStudentScheduler()
    with pytest.raises(RuntimeError, match="set_timesteps"):
        sched.step(torch.zeros(1), 0, torch.zeros(1))


def test_non_flow_prediction_is_rejected() -> None:
    with pytest.raises(ValueError, match="flow_prediction"):
        DMDStudentScheduler(prediction_type="epsilon")


def test_scheduler_selection_by_config(mocker) -> None:
    """#11260: config 'dmd_student' selects the new scheduler; 'auto' and
    'flow_unipc' keep the checkpoint-declared FlowUniPC."""
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import (
        _select_lingbot_scheduler,
    )

    dmd = _select_lingbot_scheduler("/m", {"scheduler": "dmd_student"}, scheduler_subfolder="s", local_files_only=True)
    assert isinstance(dmd, DMDStudentScheduler)
    # The recipe knobs are forwarded from the config.
    assert dmd.config.flow_shift == 3.0
    assert dmd.config.high_noise_threshold == 0.5
    assert dmd.config.ddim_eta == 1.0

    unipc = mocker.patch(
        "vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video.FlowUniPCMultistepScheduler.from_pretrained"
    )
    for cfg in ({"scheduler": "flow_unipc"}, {}, {"scheduler": "auto"}):
        got = _select_lingbot_scheduler("/m", cfg, scheduler_subfolder="s", local_files_only=True)
        assert got is unipc.return_value
        unipc.assert_called_with("/m", subfolder="s", local_files_only=True)
        unipc.reset_mock()

    with pytest.raises(ValueError, match="Unknown LingBot-Video scheduler"):
        _select_lingbot_scheduler("/m", {"scheduler": "bogus"}, scheduler_subfolder="s", local_files_only=True)


def test_recipe_defaults_follow_the_scheduler() -> None:
    """#11260: unspecified steps/guidance resolve per scheduler, so the request
    path (normalize_lingbot_request) and direct _generate calls agree."""
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import (
        _lingbot_recipe_defaults,
    )

    assert _lingbot_recipe_defaults(DMDStudentScheduler()) == (8, 1.0)
    assert _lingbot_recipe_defaults(object()) == (40, 6.0)


def _warped_grid(shift: float, steps: int = 8) -> list[float]:
    grid = np.linspace(1, 1 / 1000, steps + 1)[:-1]
    grid = shift * grid / (1 + (shift - 1) * grid)
    if np.fabs(grid[0] - 1) < 1e-6:
        grid[0] -= 1e-6
    # The scheduler stores the grid as float32; match its rounding.
    grid = grid.astype(np.float32)
    grid = np.concatenate([grid, [0.0]])
    return [round(float(s), 8) for s in grid]


def test_configured_flow_shift_is_the_omitted_default() -> None:
    """#11260 review: the configured dmd_flow_shift is the one source of truth
    for an omitted request shift — the distilled grid is built from it, not
    re-warped by the generic 3.0 default."""
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import (
        _lingbot_omitted_shift,
    )

    configured = DMDStudentScheduler(flow_shift=2.0)
    configured.set_timesteps(8)
    assert [round(float(s), 8) for s in configured.sigmas] == _warped_grid(2.0)
    assert _lingbot_omitted_shift(configured) == 2.0

    default = DMDStudentScheduler()
    assert _lingbot_omitted_shift(default) == 3.0
    assert _lingbot_omitted_shift(object()) == 3.0


def test_explicit_shift_overrides_the_configured_grid() -> None:
    """An explicit request-level shift still overrides the configured warp."""
    sched = DMDStudentScheduler(flow_shift=2.0)
    sched.set_timesteps(8, shift=5.0)
    assert [round(float(s), 8) for s in sched.sigmas] == _warped_grid(5.0)
