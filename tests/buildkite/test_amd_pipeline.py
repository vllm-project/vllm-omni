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

MULTI_GPU_MARKER_EXCLUSION = (
    "core_model and cpu and not (cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)"
)
ULYSSES_UAA_2D_NODE = "tests/diffusion/attention/test_ulysses_uaa.py::test_ulysses_uaa_2d_mask_layout_matches_baseline"


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
    staging_command = next(command for command in step["commands"] if "artifact_dir=" in command)

    # Dynamic pipelines are interpolated once during upload. Double dollars
    # preserve these variables for the GPU job's runtime shell.
    assert '"$$PWD"' in staging_command
    assert '"$${BUILDKITE_BUILD_CHECKOUT_PATH:?}"' in staging_command
    assert '"$$artifact_dir"' in staging_command
    assert step["artifact_paths"] == ["tests/e2e/accuracy/qwen3_omni/results/qwen_omni_acc/*.json"]


def test_ready_diffusion_cpu_suite_is_sharded() -> None:
    step = _find_step("Simple · Diffusion Test · Shard %N/%t", AMD_READY_PIPELINE)
    pytest_command = next(command for command in step["commands"] if "pytest" in command)

    assert step["parallelism"] == 4
    assert step["timeout_in_minutes"] == 45
    assert "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT" in pytest_command
    assert "--shard-id=$$BUILDKITE_PARALLEL_JOB" in pytest_command


@pytest.mark.parametrize("pipeline_path", [AMD_READY_PIPELINE, AMD_MERGE_PIPELINE])
def test_diffusion_cpu_suite_excludes_multi_gpu_tests(pipeline_path: Path) -> None:
    step = _find_step("Simple · Diffusion Test · Shard %N/%t", pipeline_path)
    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    argv = split(pytest_command)

    marker_index = argv.index("-m")
    assert argv[marker_index + 1] == MULTI_GPU_MARKER_EXCLUSION


@pytest.mark.parametrize(
    ("pipeline_path", "step_label"),
    [
        (AMD_READY_PIPELINE, "Diffusion Sequence Parallelism Test"),
        (AMD_MERGE_PIPELINE, "Diffusion Tensor Parallelism Test"),
    ],
)
def test_ulysses_uaa_2d_mask_runs_on_two_gpu_lane(
    pipeline_path: Path,
    step_label: str,
) -> None:
    step = _find_step(step_label, pipeline_path)
    pytest_command = next(command for command in step["commands"] if ULYSSES_UAA_2D_NODE in command)
    argv = split(pytest_command)

    assert step["agent_pool"] == "mi300_2"
    assert ULYSSES_UAA_2D_NODE in argv
    marker_index = argv.index("-m")
    assert argv[marker_index + 1] == "core_model and cards_2"


def test_cosyvoice_ready_smoke_uses_sdpa() -> None:
    step = _find_step("CosyVoice3-TTS E2E Smoke (SDPA)", AMD_READY_PIPELINE)

    assert step["grade"] == "Blocking"
    assert step["retry"] == {"automatic": [{"exit_status": 134, "limit": 1}]}
    assert "export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA" in step["commands"]

    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    assert "tests/e2e/online_serving/test_cosyvoice3_tts_expansion.py::test_voice_clone_zh_002" in pytest_command


def test_cosyvoice_full_default_backend_suite_runs_nightly() -> None:
    step = _find_step("CosyVoice3-TTS E2E Test", AMD_NIGHTLY_PIPELINE)

    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 90
    assert step["retry"] == {"automatic": [{"exit_status": 134, "limit": 1}]}
    assert all("DIFFUSION_ATTENTION_BACKEND" not in command for command in step["commands"])

    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    assert "tests/e2e/online_serving/test_cosyvoice3_tts_expansion.py" in pytest_command
    assert "::" not in pytest_command


def test_amd_template_preserves_step_retry_policy() -> None:
    template = AMD_TEMPLATE.read_text(encoding="utf-8")
    # Both grouped and top-level AMD steps must preserve an explicit retry
    # policy when the source suite is rendered into the uploaded pipeline.
    assert template.count("{% if step.retry %}") == 2
    assert template.count("{% for retry_rule in step.retry.automatic %}") == 2
