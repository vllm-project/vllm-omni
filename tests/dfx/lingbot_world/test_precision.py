# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent precision judgment over real paired measurements."""

import pytest

from tests.dfx.lingbot_world.checks import numeric_report

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_numeric_precision(lingbot_settings, lingbot_measurements):
    report = numeric_report(lingbot_settings, lingbot_measurements)
    assert report["passed"], f"See {lingbot_settings['output']}/precision.json"
