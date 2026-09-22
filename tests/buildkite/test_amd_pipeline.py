# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from shlex import split

import pytest
import yaml
from jinja2 import Environment

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_MERGE_PIPELINE = Path(".buildkite/amd/test-amd-merge.yml")
AMD_NIGHTLY_PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
AMD_READY_PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
AMD_TEMPLATE = Path(".buildkite/amd/test-template-amd-omni.j2")


def _find_step_in_pipeline(label: str, pipeline: dict) -> dict:
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


def _find_step(label: str, pipeline_path: Path = AMD_MERGE_PIPELINE) -> dict:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    return _find_step_in_pipeline(label, pipeline)


def _render_amd_pipeline(pipeline_path: Path) -> dict:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    template = Environment(keep_trailing_newline=True).from_string(AMD_TEMPLATE.read_text(encoding="utf-8"))
    rendered = template.render(
        steps=pipeline["steps"],
        env=pipeline.get("env", {}),
        mirror_hw="amdproduction",
    )
    return yaml.safe_load(rendered)


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


def test_hunyuanimage3_nightly_selects_one_offline_accuracy_case() -> None:
    step = _find_step("HunyuanImage3 Offline Pixel Accuracy", AMD_NIGHTLY_PIPELINE)
    staging_command = next(command for command in step["commands"] if "artifact_dir=" in command)
    pytest_commands = [split(command) for command in step["commands"] if command.startswith("pytest ")]
    test_node = "tests/e2e/accuracy/test_hunyuan_image3_pixel_accuracy.py::test_hunyuan_image3_pixel_accuracy_offline"
    marker = "full_model and rocm and MI325 and cards_4"

    assert step["agent_pool"] == "mi300_4"
    assert step["depends_on"] == "amd-build"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 180
    assert step["env"] == {
        "HUNYUAN_IMAGE3_MODEL": "tencent/HunyuanImage-3.0-Instruct",
        "HUNYUAN_IMAGE3_DEVICES": "0,1,2,3",
        "DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA",
    }
    assert step["artifact_paths"] == ["artifacts/rocm-hunyuanimage3/**/*"]
    assert len(pytest_commands) == 2
    for command in pytest_commands:
        assert command.count(test_node) == 1
        assert not any(arg.startswith("tests/") and arg != test_node for arg in command)
        assert command[command.index("-m") + 1] == marker
        assert command[command.index("--run-level") + 1] == "full_model"
    assert "--collect-only" in pytest_commands[0]
    assert "--collect-only" not in pytest_commands[1]
    assert all("VLLM_CI_ALLOW_NO_TESTS" not in command for command in step["commands"])
    assert '"$${BUILDKITE_BUILD_CHECKOUT_PATH:?}' in staging_command
    assert '"$$artifact_dir"' in staging_command

    rendered = _render_amd_pipeline(AMD_NIGHTLY_PIPELINE)
    rendered_step = _find_step_in_pipeline(
        "mi300_4: HunyuanImage3 Offline Pixel Accuracy",
        rendered,
    )
    container = rendered_step["plugins"][0]["kubernetes"]["podSpecPatch"]["containers"][0]
    assert rendered_step["depends_on"] == "amd-build"
    assert rendered_step["agents"]["queue"] == "amd_mi300_4"
    assert container["resources"]["limits"]["amd.com/gpu"] == "4"
    assert container["resources"]["requests"]["amd.com/gpu"] == "4"
    assert rendered_step["command"] == "bash .buildkite/amd/scripts/run-amd-test.sh"
    assert rendered_step["soft_fail"] is True
    assert rendered_step["timeout_in_minutes"] == 180
    assert rendered_step["artifact_paths"] == ["artifacts/rocm-hunyuanimage3/**/*"]
    assert rendered_step["env"]["VLLM_CI_EXPECTED_GPU_COUNT"] == "4"
    assert rendered_step["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
    assert rendered_step["env"]["HUNYUAN_IMAGE3_DEVICES"] == "0,1,2,3"


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
