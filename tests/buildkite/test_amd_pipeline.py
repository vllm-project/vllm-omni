# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
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
COSMOS3_TEST = Path("tests/e2e/online_serving/test_cosmos3.py")


def _find_step(label: str, pipeline: dict) -> dict:
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


def _load_pipeline(pipeline_path: Path) -> dict:
    return yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))


def _render_pipeline(pipeline_path: Path) -> dict:
    template = Environment().from_string(AMD_TEMPLATE.read_text(encoding="utf-8"))
    rendered = template.render(
        **_load_pipeline(pipeline_path),
        mirror_hw="amdproduction",
    )
    return yaml.safe_load(rendered)


def _assigned_call_keyword(path: Path, target: str, keyword: str) -> object:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(name, ast.Name) and name.id == target for name in node.targets)
    ]
    assert len(assignments) == 1
    call = assignments[0].value
    assert isinstance(call, ast.Call)
    values = [item.value for item in call.keywords if item.arg == keyword]
    assert len(values) == 1
    return ast.literal_eval(values[0])


def test_qwen3_tts_base_preserves_advanced_model_arguments() -> None:
    step = _find_step(
        "Qwen3-TTS Base E2E Test",
        _load_pipeline(AMD_MERGE_PIPELINE),
    )
    commands = step["commands"]

    assert all("bash -c" not in command for command in commands)
    pytest_command = next(command for command in commands if "pytest" in command)
    argv = split(pytest_command)

    marker_index = argv.index("-m")
    run_level_index = argv.index("--run-level")
    assert argv[marker_index + 1] == "advanced_model and cuda"
    assert argv[run_level_index + 1] == "advanced_model"


def test_qwen3_accuracy_defers_artifact_path_expansion() -> None:
    step = _find_step(
        "Qwen3-Omni Accuracy",
        _load_pipeline(AMD_NIGHTLY_PIPELINE),
    )
    staging_command = next(command for command in step["commands"] if "artifact_dir=" in command)

    # Dynamic pipelines are interpolated once during upload. Double dollars
    # preserve these variables for the GPU job's runtime shell.
    assert '"$$PWD"' in staging_command
    assert '"$${BUILDKITE_BUILD_CHECKOUT_PATH:?}"' in staging_command
    assert '"$$artifact_dir"' in staging_command
    assert step["artifact_paths"] == ["tests/e2e/accuracy/qwen3_omni/results/qwen_omni_acc/*.json"]


def test_cosmos3_nightly_selects_one_mi300_t2i_item() -> None:
    step = _find_step(
        "mi300_1: Cosmos3 T2I Function",
        _render_pipeline(AMD_NIGHTLY_PIPELINE),
    )
    commands = step["env"]["TEST_COMMANDS"]
    argv = split(commands)
    container = step["plugins"][0]["kubernetes"]["podSpecPatch"]["containers"][0]
    artifact_dir = "$$BUILDKITE_BUILD_CHECKOUT_PATH/artifacts/rocm-cosmos3-nightly"

    assert step["depends_on"] == "amd-build"
    assert step["agents"] == {"queue": "amd_mi300_1"}
    assert step["soft_fail"] is True
    assert step["timeout_in_minutes"] == 60
    assert step["artifact_paths"] == ["artifacts/rocm-cosmos3-nightly/**/*"]
    assert container["image"] == "rocm/vllm-omni:$BUILDKITE_COMMIT"
    assert container["resources"]["limits"]["amd.com/gpu"] == "1"
    assert argv[argv.index("timeout") + 1 : argv.index("timeout") + 5] == [
        "--signal=TERM",
        "--kill-after=2m",
        "50m",
        "pytest",
    ]
    assert "tests/e2e/online_serving/test_cosmos3.py::test_text_to_image_001" in argv
    assert argv[argv.index("-m") + 1] == "core_model and rocm and MI325 and cards_1"
    assert argv[argv.index("--run-level") + 1] == "full_model"
    assert "--junitxml=$$COSMOS3_ARTIFACT_DIR/pytest.xml" in argv
    assert "VLLM_CI_ALLOW_NO_TESTS" not in commands
    assert f"COSMOS3_ARTIFACT_DIR={artifact_dir}" in argv
    for artifact in (
        "rocm-smi.txt",
        "pytest.xml",
        "pytest.log",
        "pytest-summary.txt",
    ):
        assert any(token.endswith(f"$$COSMOS3_ARTIFACT_DIR/{artifact}") for token in argv)

    assert _assigned_call_keyword(COSMOS3_TEST, "SINGLE_CARD_MARKS", "res") == {
        "cuda": "H100",
        "rocm": "MI325",
    }


def test_ready_diffusion_cpu_suite_is_sharded() -> None:
    step = _find_step(
        "Simple · Diffusion Test · Shard %N/%t",
        _load_pipeline(AMD_READY_PIPELINE),
    )
    pytest_command = next(command for command in step["commands"] if "pytest" in command)

    assert step["parallelism"] == 4
    assert step["timeout_in_minutes"] == 45
    assert "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT" in pytest_command
    assert "--shard-id=$$BUILDKITE_PARALLEL_JOB" in pytest_command


def test_cosyvoice_gpu_abort_gets_one_fresh_job_retry() -> None:
    step = _find_step(
        "CosyVoice3-TTS E2E Test",
        _load_pipeline(AMD_READY_PIPELINE),
    )

    assert step["retry"] == {"automatic": [{"exit_status": 134, "limit": 1}]}

    template = AMD_TEMPLATE.read_text(encoding="utf-8")
    # Both grouped and top-level AMD steps must preserve an explicit retry
    # policy when the source suite is rendered into the uploaded pipeline.
    assert template.count("{% if step.retry %}") == 2
    assert template.count("{% for retry_rule in step.retry.automatic %}") == 2
