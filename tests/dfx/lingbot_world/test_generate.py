# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real model performance and VAE precision gates; no mock inference."""

import pytest

from tests.dfx.lingbot_world.runner import generate
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=4)
def test_cuda_served_regression(lingbot_settings):
    assert lingbot_settings["platform"] == "cuda"
    report = generate(lingbot_settings)
    assert report["passed"], f"Regression detected; see {lingbot_settings['output']}/generation.json"


@hardware_test(res={"npu": "A2"}, num_cards=4)
def test_npu_served_regression(lingbot_settings):
    assert lingbot_settings["platform"] == "npu"
    report = generate(lingbot_settings)
    assert report["passed"], f"Regression detected; see {lingbot_settings['output']}/generation.json"
