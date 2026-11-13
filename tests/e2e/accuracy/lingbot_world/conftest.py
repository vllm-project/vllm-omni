# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One data-only accuracy fixture; numerical/WBench/VBench verdicts are independent."""

import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("lingbot-world")
    group.addoption("--lingbot-profile", type=Path, help="Accuracy workload JSON")
    group.addoption("--lingbot-output", type=Path, default=Path("lingbot-artifacts/accuracy"))
    group.addoption("--lingbot-generation", type=Path, help="Previously captured accuracy data, e.g. NPU outputs")


@pytest.fixture(scope="session")
def lingbot_settings(request):
    from tests.e2e.accuracy.lingbot_world.runner import load_settings
    from vllm_omni.benchmarks.realtime_video.metrics import RegressionError

    profile = request.config.getoption("--lingbot-profile")
    if profile is None:
        if os.environ.get("BUILDKITE") == "true":
            raise RegressionError("Selected LingBot accuracy tests require --lingbot-profile")
        pytest.skip("Opt in with --lingbot-profile")
    settings = load_settings(
        profile,
        request.config.getoption("--lingbot-output"),
        generation=request.config.getoption("--lingbot-generation"),
    )
    if not all(settings.get("score_floors", {}).get(suite) for suite in ("wbench", "vbench")):
        raise RegressionError(f"{profile} must define WBench and VBench score floors")
    return settings


@pytest.fixture(scope="session")
def lingbot_measurements(lingbot_settings):
    from tests.e2e.accuracy.lingbot_world.prepare import prepare_capture
    from tests.e2e.accuracy.lingbot_world.runner import measurement_data

    if lingbot_settings["generation"] is None:
        # Capture errors are retained in the manifest, not promoted to a global
        # verdict. Each test validates only the artifacts needed by its check.
        prepare_capture(lingbot_settings)
        lingbot_settings["generation"] = lingbot_settings["output"]
    return measurement_data(lingbot_settings)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if os.environ.get("BUILDKITE") == "true" and report.skipped:
        report.outcome = "failed"
        report.longrepr = f"Selected LingBot accuracy test was skipped: {report.longrepr}"
