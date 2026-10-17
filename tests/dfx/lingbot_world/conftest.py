# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit opt-in locally; missing prerequisites fail selected CI jobs."""

import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("lingbot-world")
    group.addoption("--lingbot-profile", type=Path, help="Pinned rollout/profile JSON")
    group.addoption("--lingbot-output", type=Path, default=Path("lingbot-artifacts"))
    group.addoption("--lingbot-reference-root", type=Path, help="Clean checkout at profile.reference_revision")
    group.addoption("--lingbot-generation", type=Path, help="Downloaded generation artifacts for GPU evaluation")
    group.addoption("--lingbot-measurement-repeat", type=int, choices=(0, 1, 2))


@pytest.fixture(scope="module")
def lingbot_settings(request):
    from benchmarks.lingbot_world.regression import RegressionError, read_json

    path = request.config.getoption("--lingbot-profile")
    if path is None:
        if os.environ.get("BUILDKITE") == "true":
            raise RegressionError("Selected LingBot CI tests require --lingbot-profile; cannot skip")
        pytest.skip("Opt in with --lingbot-profile")
    settings = read_json(path)
    settings["output"] = request.config.getoption("--lingbot-output").resolve()
    settings["reference_root"] = request.config.getoption("--lingbot-reference-root")
    settings["generation"] = request.config.getoption("--lingbot-generation")
    settings["profile_path"] = path.resolve()
    settings["measurement_repeat"] = request.config.getoption("--lingbot-measurement-repeat")
    return settings


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if os.environ.get("BUILDKITE") == "true" and report.skipped:
        report.outcome = "failed"
        report.longrepr = f"Selected LingBot CI test was skipped: {report.longrepr}"
    if report.failed:
        from benchmarks.lingbot_world.regression import write_json
        from tests.dfx.lingbot_world.resources import collect_resources

        output = item.config.getoption("--lingbot-output")
        write_json(
            output / f"{item.name}-{report.when}-failure.json",
            {
                "passed": False,
                "nodeid": item.nodeid,
                "phase": report.when,
                "error": str(report.longrepr),
                "resources": collect_resources(),
            },
        )
