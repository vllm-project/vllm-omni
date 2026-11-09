# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Candidate-only long sessions, memory, control and teardown checks."""

import pytest

from tests.dfx.lingbot_world.runner import run_stability
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=4)
def test_cuda_long_stability(lingbot_settings):
    assert lingbot_settings["platform"] == "cuda"
    assert run_stability(lingbot_settings)["passed"]


@hardware_test(res={"npu": "A2"}, num_cards=4)
def test_npu_long_stability(lingbot_settings):
    assert lingbot_settings["platform"] == "npu"
    assert run_stability(lingbot_settings)["passed"]
