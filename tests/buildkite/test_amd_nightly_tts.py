# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
JOBS = {
    "ROCm · TTS Function · Single-GPU": (
        "mi300_1",
        130,
        "120m",
        "artifacts/rocm-tts-function-1gpu",
        "full_model and L4 and B200 and tts and cards_1",
    ),
    "ROCm · TTS Performance · Single-GPU": (
        "mi300_1",
        190,
        "180m",
        "artifacts/rocm-tts-perf-1gpu",
        "H100 and B200 and cards_1",
    ),
    "ROCm · TTS Performance · 2-GPU": (
        "mi300_2",
        190,
        "180m",
        "artifacts/rocm-tts-perf-2gpu",
        "H100 and B200 and cards_2",
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
def test_tts_jobs_match_current_cuda_contract(label: str, contract: tuple[str, int, str, str, str]) -> None:
    queue, timeout, pytest_timeout, artifact_dir, markers = contract
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
    assert f'-m "{markers}"' in pytest_commands[0]
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


def test_tts_function_preserves_current_full_model_scope() -> None:
    commands = "\n".join(_find_step("ROCm · TTS Function · Single-GPU")["commands"])
    assert "pytest -s -v -ra tests/e2e/" in commands
    assert '--run-level "full_model"' in commands
    assert "--ignore=tests/e2e/accuracy" in commands
    assert 'export VLLM_USE_DEEP_GEMM="0"' in commands
    assert 'export VLLM_MOE_USE_DEEP_GEMM="0"' in commands


def test_tts_performance_jobs_retain_benchmark_json() -> None:
    for label in ("ROCm · TTS Performance · Single-GPU", "ROCm · TTS Performance · 2-GPU"):
        commands = "\n".join(_find_step(label)["commands"])
        assert "tests/dfx/perf/tests/test_tts.json" in commands
        assert 'export BENCHMARK_DIR="$$ROCM_CI_ARTIFACT_DIR/benchmark-results"' in commands
