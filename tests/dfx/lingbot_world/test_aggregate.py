# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Blocking performance/precision gate for all three independently collected pairs."""

import pytest

from tests.dfx.lingbot_world.runner import aggregate_measurements

# This gate consumes full-model artifacts and is selected explicitly by the
# aggregate launcher; generic CPU CI has neither a profile nor measurements.
pytestmark = [pytest.mark.full_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_all_npu_measurement_pairs_pass_regression(lingbot_settings):
    report = aggregate_measurements(lingbot_settings)
    assert report["passed"], f"Regression detected; see {lingbot_settings['output']}/generation.json"
