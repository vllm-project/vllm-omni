# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent performance judgment over real paired measurements."""

import pytest

from tests.dfx.lingbot_world.checks import performance_report

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_performance_regression(lingbot_settings, lingbot_measurements):
    report = performance_report(lingbot_settings, lingbot_measurements)
    assert report["passed"], f"See {lingbot_settings['output']}/performance.json"
