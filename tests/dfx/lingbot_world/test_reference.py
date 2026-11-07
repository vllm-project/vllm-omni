# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference-only calibration inputs; collection never qualifies a candidate."""

import pytest

from tests.dfx.lingbot_world.runner import aggregate_reference_measurements, generate, validate_reference_collection
from tests.dfx.lingbot_world.test_quality import evaluate_quality
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


def _collect(settings):
    report = generate(settings, reference_only=True)
    validate_reference_collection(settings, settings["output"], report)
    assert report["complete"] and report["measurement_only"] and not report["passed"]


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=4)
def test_collect_cuda_reference_cohort(lingbot_settings):
    assert lingbot_settings["platform"] == "cuda"
    _collect(lingbot_settings)


@hardware_test(res={"npu": "A2"}, num_cards=4)
def test_collect_npu_reference_cohort(lingbot_settings):
    assert lingbot_settings["platform"] == "npu"
    _collect(lingbot_settings)


@hardware_test(res={"npu": "A2"}, num_cards=4)
def test_collect_npu_reference_measurement(lingbot_settings):
    repeat = lingbot_settings["measurement_repeat"]
    assert repeat in (0, 1, 2), "A reference measurement requires its explicit repeat index"
    report = generate(lingbot_settings, measurement_repeat=repeat, reference_only=True)
    validate_reference_collection(lingbot_settings, lingbot_settings["output"], report, measurement_repeat=repeat)
    assert report["complete"] and report["measurement_only"] and not report["passed"]


@pytest.mark.cpu
def test_aggregate_reference_measurements(lingbot_settings):
    report = aggregate_reference_measurements(lingbot_settings)
    assert report["complete"] and report["measurement_only"] and report["reference_only"] and not report["passed"]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_collect_reference_quality(lingbot_settings, monkeypatch):
    report = evaluate_quality(lingbot_settings, monkeypatch, reference_only=True)
    assert report["complete"] and report["measurement_only"] and not report["passed"]
