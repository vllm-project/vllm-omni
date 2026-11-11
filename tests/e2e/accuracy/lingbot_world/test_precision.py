# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent precision judgment over real paired measurements."""

import pytest

from tests.e2e.accuracy.lingbot_world.checks import numeric_report, record_accuracy_result

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_numeric_precision(lingbot_settings, lingbot_measurements):
    report = numeric_report(lingbot_settings, lingbot_measurements)
    record_accuracy_result(lingbot_settings, lingbot_measurements, "precision", report)
    assert report["passed"], f"See {lingbot_settings['output']}/precision.json"
