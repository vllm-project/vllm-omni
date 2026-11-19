# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent native vbench semantic quality test."""

import pytest

from tests.e2e.accuracy.lingbot_world.checks import run_accuracy
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_vbench_quality(lingbot_settings, lingbot_measurements):
    report = run_accuracy(lingbot_settings, lingbot_measurements, "vbench")
    assert report["passed"], f"See {lingbot_settings['output']}/vbench.json"
