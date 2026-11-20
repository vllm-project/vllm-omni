# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare fixed-input VAE output points with the generation platform's baseline."""

import pytest

from tests.e2e.accuracy.lingbot_world.checks import run_accuracy
from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.diffusion]


# Exactly one item is collected. With --lingbot-generation it only compares
# previously captured data on CPU (core_model, cpu); otherwise it first captures
# on two H100s (configs/cuda.json). Nightly jobs select this file by path.
def pytest_generate_tests(metafunc):
    if "suite" not in metafunc.fixturenames:
        return
    if metafunc.config.getoption("--lingbot-generation") is not None:
        param = pytest.param("precision", marks=[pytest.mark.core_model, pytest.mark.cpu], id="artifact")
    else:
        marks = [pytest.mark.full_model, *hardware_marks(res={"cuda": "H100"}, num_cards=2)]
        param = pytest.param("precision", marks=marks, id="capture")
    metafunc.parametrize("suite", [param])


def test_numeric_precision(lingbot_settings, lingbot_measurements, suite):
    report = run_accuracy(lingbot_settings, lingbot_measurements, suite)
    assert report["passed"], f"See {lingbot_settings['output']}/precision.json"
