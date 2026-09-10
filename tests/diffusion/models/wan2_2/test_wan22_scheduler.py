# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare actual Wan schedules to native Wan outputs and independent sigma arithmetic."""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    FASTWAN_DMD_SCHEDULER_SHIFT,
    build_wan_scheduler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_wan_unipc_matches_captured_native_full_schedule():
    # Captured from the native implementation in the protocol audit; metadata
    # pins the actual source, rather than importing the implementation under test.
    golden = json.loads((Path(__file__).parent / "fixtures/native_unipc_shift12.json").read_text())
    scheduler = build_wan_scheduler("unipc", golden["runtime_shift"])
    for steps in (golden["steps"], 20, golden["steps"]):
        scheduler.set_timesteps(steps, device="cpu")
        if steps == golden["steps"]:
            assert scheduler.timesteps.tolist() == golden["timesteps"]
            torch.testing.assert_close(scheduler.sigmas, torch.tensor(golden["sigmas"]), rtol=0, atol=0)
            assert (scheduler.timesteps >= golden["boundary_timestep"]).sum().item() == 26
            assert (scheduler.timesteps < golden["boundary_timestep"]).sum().item() == 14


@pytest.mark.parametrize("shift", [1.0, 3.0, 5.0])
def test_wan_unipc_shifts_unmodified_training_endpoints(shift):
    # Native Wan: float32 training sigma_max=(1-1/1000), sigma_min=0;
    # interpolate in float64, apply the runtime shift, then cast sigma to FP32.
    unshifted = np.linspace(float(np.float32(0.999)), 0.0, 41)[:-1]
    shifted = shift * unshifted / (1.0 + (shift - 1.0) * unshifted)
    expected_timesteps = (shifted * 1000).astype(np.int64)
    expected_sigmas: npt.NDArray[np.float32] = np.append(shifted, 0.0).astype(np.float32)
    scheduler = build_wan_scheduler("unipc", shift)
    scheduler.set_timesteps(40, device="cpu")
    torch.testing.assert_close(scheduler.timesteps, torch.from_numpy(expected_timesteps), rtol=0, atol=0)
    torch.testing.assert_close(scheduler.sigmas, torch.from_numpy(expected_sigmas), rtol=0, atol=0)


def test_wan_euler_keeps_dmd_shift_semantics():
    scheduler = build_wan_scheduler("euler", FASTWAN_DMD_SCHEDULER_SHIFT)
    scheduler.set_timesteps(3, device="cpu")
    # The unchanged Euler branch starts at sigma=1, with shift=8 for DMD.
    expected = torch.tensor([1.0, 16.0 / 17.0, 0.8, 0.0])
    torch.testing.assert_close(scheduler.sigmas, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(scheduler.timesteps, expected[:-1] * 1000, rtol=0, atol=1e-4)
