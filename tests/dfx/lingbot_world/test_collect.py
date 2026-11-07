# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Collect one complete NPU pair; the aggregate job applies all regression gates."""

import pytest

from tests.dfx.lingbot_world.runner import generate
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


@hardware_test(res={"npu": "A2"}, num_cards=4)
def test_collect_npu_measurement_pair(lingbot_settings):
    repeat = lingbot_settings["measurement_repeat"]
    assert repeat in (0, 1, 2), "A measurement pair requires an explicit repeat index"
    report = generate(lingbot_settings, measurement_repeat=repeat)
    assert report["complete"] and report["measurement_only"]
