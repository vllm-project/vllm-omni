# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.models.lingbot_world.dmd_block import LingBotDMDBlockRunner
from vllm_omni.diffusion.models.lingbot_world.pipeline import _build_shifted_flow_schedule

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_FIXTURE = Path(__file__).with_name("fixtures") / "lingbot_world_official_dmd_schedule.json.fixture"


@pytest.mark.parametrize("shift", [1.0, 2.5, 5.0, 10.0])
def test_dmd_schedule_matches_native_reference_lattice(shift: float) -> None:
    # Frozen outputs of the official scheduler, not idealized timestep/1000.
    expected = json.loads(_FIXTURE.read_text())["schedules"][f"{shift:g}"]
    actual = _build_shifted_flow_schedule(flow_shift=shift)
    assert [step[0] for step in actual] == [step["timestep"] for step in expected]
    assert all(isinstance(step[0], int) for step in actual)
    assert [step[1] for step in actual] == [step["x0_sigma_first_match"] for step in expected]
    assert [step[2] for step in actual] == [step["renoise_sigma_second_match"] for step in expected]


def test_duplicate_integer_timestep_keeps_two_distinct_sigma_lookups() -> None:
    schedule = _build_shifted_flow_schedule(flow_shift=10.0)
    assert schedule[1][0] == 967
    assert schedule[1][1] != schedule[1][2]


def test_flow_inversion_keeps_native_fp64_rounding() -> None:
    # Values chosen so rounding the product before subtraction loses a bit.
    current = torch.tensor([1.00001, 3.1415927, 0.0001, -11.3], dtype=torch.float32)
    flow = torch.tensor([1.12345, 2.7182817, 1.00001, 20.2], dtype=torch.float32)
    sigma = json.loads(_FIXTURE.read_text())["schedules"]["10"][1]["x0_sigma_first_match"]
    expected = (current.double() - sigma * flow.double()).float()
    rounded_early = current - sigma * flow
    assert not torch.equal(expected, rounded_early)
    # This transition does not touch transformer/cache/device state.
    runner = LingBotDMDBlockRunner.__new__(LingBotDMDBlockRunner)
    result = runner.apply_transition(current, flow, sigma, next_sigma=None, generator=torch.Generator().manual_seed(42))
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
