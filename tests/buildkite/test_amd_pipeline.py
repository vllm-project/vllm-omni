# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_MERGE_PIPELINE = Path(".buildkite/amd/test-amd-merge.yml")
AMD_NIGHTLY_PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
AMD_READY_PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
AMD_TEMPLATE = Path(".buildkite/amd/test-template-amd-omni.j2")
AMD_DIAGNOSTIC_RUNNER = Path(
    ".buildkite/amd/scripts/run-amd-test-with-diagnostics.sh",
)


def _find_step(label: str, pipeline_path: Path = AMD_MERGE_PIPELINE) -> dict:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))

    def walk(steps: list[dict]) -> dict | None:
        for step in steps:
            if step.get("label") == label:
                return step
            if nested := walk(step.get("steps", [])):
                return nested
        return None

    step = walk(pipeline.get("steps", []))
    assert step is not None, f"missing AMD pipeline step: {label}"
    return step


def test_qwen3_tts_base_preserves_advanced_model_arguments() -> None:
    step = _find_step("Qwen3-TTS Base E2E Test")
    commands = step["commands"]

    assert all("bash -c" not in command for command in commands)
    pytest_command = next(command for command in commands if "pytest" in command)
    argv = split(pytest_command)

    marker_index = argv.index("-m")
    run_level_index = argv.index("--run-level")
    assert argv[marker_index + 1] == "advanced_model and cuda"
    assert argv[run_level_index + 1] == "advanced_model"


def test_qwen3_accuracy_defers_artifact_path_expansion() -> None:
    step = _find_step("Qwen3-Omni Accuracy", AMD_NIGHTLY_PIPELINE)
    staging_command = next(command for command in step["commands"] if "accuracy_results=" in command)

    # Dynamic pipelines are interpolated once during upload. Double dollars
    # preserve these variables for the GPU job's runtime shell.
    assert "$${BUILDKITE_BUILD_CHECKOUT_PATH:?}" in staging_command
    assert '"$$artifact_dir"' in staging_command
    assert "shopt -s nullglob" in staging_command
    assert "Accuracy test produced no JSON results" in staging_command
    assert step["artifact_paths"] == [
        "tests/e2e/accuracy/qwen3_omni/results/qwen_omni_acc/*.json",
        "qwen3-omni-ci/*.log",
    ]


def test_qwen3_nightly_stages_distinct_diagnostic_logs() -> None:
    runner = AMD_DIAGNOSTIC_RUNNER.read_text(encoding="utf-8")

    assert "*-collection.log" in runner
    assert "*-metadata.log" in runner
    assert "BUILDKITE_BUILD_CHECKOUT_PATH" in runner
    assert "pytest.log" not in runner

    for label in (
        "Qwen3-Omni Function Expansion",
        "Qwen3-Omni Accuracy",
        "Qwen3-Omni Documentation Examples",
        "Qwen3-Omni AITER-on Smoke",
    ):
        step = _find_step(label, AMD_NIGHTLY_PIPELINE)
        assert "qwen3-omni-ci/*.log" in step["artifact_paths"]
        assert any("run-amd-test-with-diagnostics.sh" in command for command in step["commands"])


def test_ready_diffusion_cpu_suite_is_sharded() -> None:
    step = _find_step("Simple · Diffusion Test · Shard %N/%t", AMD_READY_PIPELINE)
    pytest_command = next(command for command in step["commands"] if "pytest" in command)

    assert step["parallelism"] == 4
    assert step["timeout_in_minutes"] == 45
    assert "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT" in pytest_command
    assert "--shard-id=$$BUILDKITE_PARALLEL_JOB" in pytest_command


def test_cosyvoice_gpu_abort_gets_one_fresh_job_retry() -> None:
    step = _find_step("CosyVoice3-TTS E2E Test", AMD_READY_PIPELINE)

    assert step["retry"] == {"automatic": [{"exit_status": 134, "limit": 1}]}

    template = AMD_TEMPLATE.read_text(encoding="utf-8")
    # Both grouped and top-level AMD steps must preserve an explicit retry
    # policy when the source suite is rendered into the uploaded pipeline.
    assert template.count("{% if step.retry %}") == 2
    assert template.count("{% for retry_rule in step.retry.automatic %}") == 2
