# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent native wbench semantic quality test."""

import pytest

from tests.dfx.lingbot_world.quality import evaluate_quality
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_wbench_quality(lingbot_settings, lingbot_measurements, monkeypatch):
    report = evaluate_quality(lingbot_settings, lingbot_measurements, "wbench", monkeypatch)
    assert report["passed"], f"See {lingbot_settings['output']}/wbench.json"
