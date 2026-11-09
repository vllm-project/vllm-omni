# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real generation only; numerical, speed and semantic judgments are separate tests."""

import pytest

from tests.dfx.lingbot_world.runner import aggregate_measurements, collect_measurements
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=4)
def test_collect_cuda_measurements(lingbot_settings):
    assert lingbot_settings["platform"] == "cuda"
    assert collect_measurements(lingbot_settings)["complete"]


@hardware_test(res={"npu": "A2"}, num_cards=4)
def test_collect_npu_measurements(lingbot_settings):
    assert lingbot_settings["platform"] == "npu"
    assert collect_measurements(lingbot_settings, lingbot_settings["measurement_repeat"])["complete"]


@pytest.mark.cpu
def test_aggregate_npu_measurements(lingbot_settings):
    assert aggregate_measurements(lingbot_settings)["complete"]
