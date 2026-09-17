# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
EVIDENCE_SCRIPT = Path(".buildkite/amd/scripts/rocm_ci_evidence.py")
JOBS = {
    "ROCm · Omni Function · Single-GPU": ("mi300_1", 130, "artifacts/rocm-omni-function-1gpu", "120m"),
    "ROCm · Omni Function · 2-GPU": ("mi300_2", 100, "artifacts/rocm-omni-function-2gpu", "90m"),
    "ROCm · Omni Documentation · 2-GPU": ("mi300_2", 100, "artifacts/rocm-omni-doc-2gpu", "90m"),
    "ROCm · Omni Performance · No Async Chunk": (
        "mi300_2",
        310,
        "artifacts/rocm-omni-perf-no-async",
        "300m",
    ),
    "ROCm · Omni Performance · Async Chunk": ("mi300_2", 190, "artifacts/rocm-omni-perf-async", "180m"),
    "ROCm · Omni Multi-Replica Startup · 4-GPU": (
        "mi300_4",
        55,
        "artifacts/rocm-omni-multi-replica",
        "45m",
    ),
}


def _find_step(label: str) -> dict:
    pipeline = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))

    def walk(steps: list[dict]) -> dict | None:
        for step in steps:
            if step.get("label") == label:
                return step
            if nested := walk(step.get("steps", [])):
                return nested
        return None

    step = walk(pipeline["steps"])
    assert step is not None, f"missing AMD pipeline step: {label}"
    return step


@pytest.mark.parametrize(("label", "contract"), JOBS.items())
def test_jobs_are_nonblocking_and_retain_evidence(label: str, contract: tuple[str, int, str, str]) -> None:
    queue, timeout, artifact_dir, pytest_timeout = contract
    step = _find_step(label)
    commands = step["commands"]
    command_text = "\n".join(commands)
    pytest_commands = [command for command in commands if "pytest -s" in command]

    assert step["agent_pool"] == queue
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == timeout
    assert step["artifact_paths"] == [f"{artifact_dir}/**/*"]
    assert len(pytest_commands) == 1
    assert f"--kill-after=1m {pytest_timeout}" in pytest_commands[0]
    assert "--collect-only" not in command_text
    assert "VLLM_CI_ALLOW_NO_TESTS" not in command_text
    for evidence in (
        "environment.txt",
        "pytest.log",
        "pytest.xml",
        "pytest-summary.txt",
        "pytest-result.txt",
        "process-cleanup.txt",
    ):
        assert evidence in command_text


@pytest.mark.parametrize(
    ("label", "needle"),
    [
        ("ROCm · Omni Function · Single-GPU", "full_model and H100 and B200 and omni and cards_1"),
        ("ROCm · Omni Function · 2-GPU", "full_model and H100 and B200 and omni and cards_2"),
        ("ROCm · Omni Documentation · 2-GPU", "full_model and omni and H100 and B200 and cards_2"),
        (
            "ROCm · Omni Performance · No Async Chunk",
            "tests/dfx/perf/tests/test_qwen3_omni_no_async_chunk.json",
        ),
        ("ROCm · Omni Performance · Async Chunk", "tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json"),
        (
            "ROCm · Omni Multi-Replica Startup · 4-GPU",
            "tests/e2e/online_serving/test_qwen3_omni_multi_replicas.py",
        ),
    ],
)
def test_jobs_preserve_current_cuda_scope(label: str, needle: str) -> None:
    assert needle in "\n".join(_find_step(label)["commands"])


def test_performance_results_are_retained_with_job_evidence() -> None:
    for label in ("ROCm · Omni Performance · No Async Chunk", "ROCm · Omni Performance · Async Chunk"):
        commands = "\n".join(_find_step(label)["commands"])
        assert 'export BENCHMARK_DIR="$$ROCM_CI_ARTIFACT_DIR/benchmark-results"' in commands


def test_evidence_helper_reports_collection_and_fails_closed(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("rocm_ci_evidence", EVIDENCE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    xml = tmp_path / "pytest.xml"
    log = tmp_path / "pytest.log"
    output = tmp_path / "pytest-result.txt"
    xml.write_text('<testsuite tests="3" failures="0" errors="0" skipped="1"/>', encoding="utf-8")
    log.write_text("2 passed, 1 skipped, 12 deselected\n", encoding="utf-8")

    module.summarize_pytest([xml], log, output)

    assert output.read_text(encoding="utf-8") == (
        "collected=15 selected=3 passed=2 failed=0 skipped=1 deselected=12 errors=0 executed=2\n"
    )
