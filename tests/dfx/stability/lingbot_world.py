# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Instrumented recovery probes for the shared LingBot CUDA stability case."""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.e2e.online_serving import test_lingbot_world_v2_stepwise as lifecycle
from vllm_omni.benchmarks.realtime_video.eval import write_json
from vllm_omni.platforms import current_omni_platform as platform


@dataclass(frozen=True)
class _Endpoint:
    host: str
    port: int
    model: str


def run_lifecycle_checks(host: str, port: int, model: str, image: Path, output: Path, *, memory_budget: int) -> None:
    """Require usable follow-up sessions and bounded live-server memory growth."""
    if type(memory_budget) is not int or memory_budget < 0:
        raise ValueError("Lifecycle memory budget must be a nonnegative integer")
    if not image.is_file():
        raise ValueError("Lifecycle checks require a local image file")
    dies = platform.get_device_count()
    if dies < 1:
        raise ValueError("Lifecycle memory checks require visible devices")
    server = _Endpoint(host, port, model)
    cases = [
        (
            "stop_and_recover",
            lambda: lifecycle.test_lingbot_world_v2_stepwise_stop_during_paced_wait_releases_session(server),
        ),
        (
            "close_and_recover",
            lambda: lifecycle.test_lingbot_world_v2_stepwise_disconnect_releases_session(server, False),
        ),
        (
            "transport_loss_and_recover",
            lambda: lifecycle.test_lingbot_world_v2_stepwise_disconnect_releases_session(server, True),
        ),
    ]
    report: dict[str, object] = {"passed": False, "memory_growth_budget_bytes": memory_budget}
    verdicts: dict[str, dict[str, object]] = {name: {"status": "not_run"} for name, _ in cases}
    report["cases"] = verdicts

    def memory_used() -> list[int]:
        samples = []
        for device in range(dies):
            with platform.device(device):
                free, total = platform.get_device_memory()
            samples.append(total - free)
        return samples

    try:
        baseline = memory_used()
        report["baseline_bytes"] = baseline
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(lifecycle, "_IMAGE_PATH_OVERRIDE", str(image))
            for name, check in cases:
                verdicts[name] = {"status": "failed"}
                try:
                    check()
                    used = memory_used()
                    verdicts[name]["used_bytes"] = used
                    if any(after - before > memory_budget for before, after in zip(baseline, used)):
                        raise AssertionError(f"Memory retained after {name} exceeds the lifecycle budget")
                    verdicts[name]["status"] = "passed"
                finally:
                    write_json(output, report)
        report["passed"] = True
    finally:
        if error := sys.exc_info()[1]:
            report["error"] = f"{type(error).__name__}: {error}"
        write_json(output, report)
