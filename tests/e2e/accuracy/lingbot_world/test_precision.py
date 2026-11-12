# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Record finite candidate numerical data for a future accuracy baseline."""

import pytest

from tests.e2e.accuracy.lingbot_world.checks import run_accuracy

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_numeric_precision(lingbot_settings, lingbot_measurements):
    report = run_accuracy(lingbot_settings, lingbot_measurements, "precision")
    assert report["passed"], f"See {lingbot_settings['output']}/precision.json"
