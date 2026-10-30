# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
import re
import subprocess
import sys
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_MERGE_PIPELINE = Path(".buildkite/amd/test-amd-merge.yml")
AMD_NIGHTLY_PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
AMD_WEEKLY_PIPELINE = Path(".buildkite/amd/test-amd-weekly.yml")
AMD_READY_PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
AMD_BOOTSTRAP = Path(".buildkite/amd/scripts/bootstrap-amd-omni.sh")
AMD_COLLECTION_GUARD = Path(".buildkite/amd/scripts/assert-pytest-collection.sh")
AMD_SOURCE_DEPENDENCIES = Path(".buildkite/common/ci_source_file_dependencies.yml")
AMD_SUITE_COMBINER = Path(__file__).resolve().parents[2] / ".buildkite/amd/scripts/combine_test_suites.py"
AMD_RUNNER = Path(".buildkite/amd/scripts/run-amd-test.sh")
AMD_TEMPLATE = Path(".buildkite/amd/test-template-amd-omni.j2")
AMD_OMNI_PROCESSOR_RUNNER = Path(".buildkite/amd/scripts/run-omni-processor-test.sh")
CUDA_READY_PIPELINE = Path(".buildkite/cuda/test-ready.yml")
AR_PAGED_ATTENTION_LABEL = "ROCm · AR Diffusion Paged Attention GPU Test"
DIFFUSION_GROUP = ":card_index_dividers: Diffusion Test"
AR_PAGED_ATTENTION_MARKERS = "core_model and rocm and MI325 and cards_1"
JOY_GPU_LABEL = "Diffusion · JoyImage GPU Inference Test"
JOY_GPU_PATH = "tests/diffusion/models/joy_image/test_joy_image_gpu.py"
MODEL_EXECUTOR_CPU_MARKERS = "core_model and cpu and not omni"
PERSONAPLEX_TEMPORAL_LABEL = "Simple · PersonaPlex Temporal Streaming Test"
PERSONAPLEX_TEMPORAL_PATH = "tests/model_executor/models/personaplex/test_temporal_streaming_hoist.py"
PERSONAPLEX_TEMPORAL_STRESS_NODES = [
    f"{PERSONAPLEX_TEMPORAL_PATH}::test_mimi_streaming_step_matches_legacy_end_to_end",
]
PERSONAPLEX_TEMPORAL_ARTIFACTS = "artifacts/personaplex-temporal-streaming/**/*"
DIFFUSION_CPU_MARKERS = (
    "core_model and cpu and not (cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)"
)
DIFFUSION_CPU_ARTIFACTS = "artifacts/amd-diffusion-cpu-shards/*.xml"
DIFFUSION_SLOW_CPU_LABEL = "Simple · Diffusion Slow CPU Test"
DIFFUSION_SLOW_CPU_ARTIFACTS = "artifacts/amd-diffusion-slow-cpu/pytest.xml"
DIFFUSION_SLOW_CPU_NODES = [
    "tests/diffusion/distributed/test_wan_vae_fastpath_install.py"
    "::test_omni_wan_vae_decode_override_is_bitwise_exact[dtype0]",
    "tests/diffusion/distributed/test_wan_vae_fastpath_install.py"
    "::test_omni_wan_vae_decode_override_is_bitwise_exact[dtype1]",
    "tests/diffusion/distributed/test_wan_decoder_fast_path.py::test_sharded_fast_path_on_every_rank[True-exact]",
    "tests/diffusion/models/lingbot_world/test_pipeline_lingbot_world.py"
    "::test_condition_continues_real_causal_vae_beyond_initial_horizon[realtime]",
    "tests/diffusion/models/lingbot_world/test_pipeline_lingbot_world.py"
    "::test_condition_continues_real_causal_vae_beyond_initial_horizon[stepwise]",
    "tests/diffusion/models/lingbot_world/test_pipeline_lingbot_world.py"
    "::test_condition_encoder_stops_at_its_fixed_point_and_restarts_with_the_session",
]
OMNI_PROCESSOR_FULL_LABEL = "Simple · Omni Processor Test"
OMNI_PROCESSOR_SMOKE_LABEL = "Simple · Omni Processor Snapshot Smoke"
OMNI_PROCESSOR_TEST = "tests/model_executor/models/test_omni_processing.py"
OMNI_PROCESSOR_TIMING_ENV = "VLLM_OMNI_AMD_PROCESSOR_TIMING"
QWEN_IMAGE_SMOKE_LABEL = "Qwen-Image Official-Weight Smoke"
QWEN_IMAGE_SMOKE_NODE = "tests/e2e/online_serving/test_qwen_image.py::test_text_to_image_001[default]"
TINY_DIFFUSION_PATH = "tests/model_tests/diffusion/test_common_offline.py"
TINY_DIFFUSION_JOBS = (
    (
        "ROCm · Tiny Diffusion Multi-GPU · 2-GPU",
        "mi300_2",
        12,
        "full_model and diffusion and rocm and MI325 and cards_2 and not nvidia_only",
        "artifacts/rocm-tiny-diffusion-2gpu/**/*",
    ),
    (
        "ROCm · Tiny Diffusion Multi-GPU · 4-GPU",
        "mi300_4",
        11,
        "full_model and diffusion and rocm and MI325 and cards_4 and not nvidia_only",
        "artifacts/rocm-tiny-diffusion-4gpu/**/*",
    ),
)
ENTRYPOINTS_FULL_SELECTOR = "full_model and omni and rocm and MI325 and cards_2 and not nvidia_only"
ENTRYPOINTS_FULL_NIGHTLY_JOBS = (
    (
        "Qwen3-Omni Duplex Playback",
        "tests/entrypoints/openai_api/test_qwen3_omni_realtime_websocket.py::"
        "TestQwen3OmniDuplexPlayback::test_speech_interrupts_playback_full",
        40,
        "omni_qwen3_omni_duplex_playback_function",
        "artifacts/rocm-qwen3-omni-duplex-playback/**/*",
        "ROCM_QWEN3_OMNI_DUPLEX_ARTIFACT_DIR",
        "Expected exactly 40 executed Qwen3-Omni duplex playback tests",
    ),
    (
        "BAGEL Sleep Mode",
        "tests/entrypoints/test_omni_sleep_mode.py::TestBagelDiffusionSleepMode",
        3,
        "omni_bagel_sleep_function",
        "artifacts/rocm-bagel-sleep/**/*",
        "ROCM_BAGEL_SLEEP_ARTIFACT_DIR",
        "Expected exactly three executed BAGEL sleep-mode tests",
    ),
)

COMBINER_SPEC = spec_from_file_location("combine_test_suites", AMD_SUITE_COMBINER)
assert COMBINER_SPEC is not None and COMBINER_SPEC.loader is not None
COMBINER = module_from_spec(COMBINER_SPEC)
COMBINER_SPEC.loader.exec_module(COMBINER)


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


def _find_group(label: str, pipeline_path: Path) -> dict:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    group = next((step for step in pipeline.get("steps", []) if step.get("group") == label), None)
    assert group is not None, f"missing AMD pipeline group: {label}"
    return group


def _walk_steps(steps: list[dict]):
    for step in steps:
        yield step
        yield from _walk_steps(step.get("steps", []))


_MARK_EXPRESSION = re.compile(r"(?:^|\s)-m\s+(['\"])(.*?)\1", re.DOTALL)


def _marker_expressions(step: dict) -> list[str]:
    return [match.group(2) for command in step.get("commands", []) for match in _MARK_EXPRESSION.finditer(command)]


def _amd_marker_selector_inventory(marker: str) -> set[tuple[Path, str, str]]:
    return {
        (pipeline_path, step["label"], expression)
        for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE)
        for step in _walk_steps(yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))["steps"])
        for expression in _marker_expressions(step)
        if re.search(rf"\b{re.escape(marker)}\b", expression)
    }


def _steps_selecting_node(pipeline_path: Path, node: str) -> list[dict]:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    return [
        step
        for step in _walk_steps(pipeline["steps"])
        if any(node in split(command) for command in step.get("commands", []))
    ]


def test_qwen_image_has_one_amd_official_weight_nightly_owner() -> None:
    owners = [
        (pipeline_path, step)
        for pipeline_path in (
            AMD_READY_PIPELINE,
            AMD_MERGE_PIPELINE,
            AMD_NIGHTLY_PIPELINE,
        )
        for step in _steps_selecting_node(pipeline_path, QWEN_IMAGE_SMOKE_NODE)
    ]

    assert len(owners) == 1
    assert owners[0][0] == AMD_NIGHTLY_PIPELINE
    assert owners[0][1]["label"] == QWEN_IMAGE_SMOKE_LABEL


def test_qwen_image_official_weight_nightly_smoke_follows_burn_in_contract() -> None:
    step = _find_step(QWEN_IMAGE_SMOKE_LABEL, AMD_NIGHTLY_PIPELINE)

    assert step["agent_pool"].startswith("mi300_")
    card_count = step["agent_pool"].rsplit("_", 1)[1]
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] >= 90
    assert "depends_on" not in step
    assert "retry" not in step
    assert len(step["artifact_paths"]) == 1
    assert step["artifact_paths"][0].startswith("artifacts/rocm-")
    assert step["artifact_paths"][0].endswith("/**/*")

    commands = step["commands"]
    all_commands = "\n".join(commands)
    env = step.get("env", {})
    assert env.get("VLLM_ROCM_USE_AITER") == "1" or ("export VLLM_ROCM_USE_AITER=1" in commands)
    assert env.get("DIFFUSION_ATTENTION_BACKEND") == "FLASH_ATTN" or (
        "export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN" in commands
    )
    assert "AITER_JIT_DIR" in all_commands
    assert "TORCH_EXTENSIONS_DIR" in all_commands
    assert "|| true" not in all_commands

    pytest_commands = [command for command in commands if "pytest" in split(command)]
    collect_commands = [command for command in pytest_commands if "--collect-only" in split(command)]
    run_commands = [command for command in pytest_commands if "--collect-only" not in split(command)]
    assert len(collect_commands) == 1
    assert len(run_commands) == 1

    collect_argv = split(collect_commands[0])
    run_argv = split(run_commands[0])
    collect_nodes = [arg for arg in collect_argv if arg.startswith("tests/") and "::" in arg]
    run_nodes = [arg for arg in run_argv if arg.startswith("tests/") and "::" in arg]
    assert len(collect_nodes) == 1
    assert run_nodes == collect_nodes

    expected_markers = f"full_model and diffusion and rocm and MI325 and cards_{card_count}"
    assert collect_argv[collect_argv.index("-m") + 1] == expected_markers
    assert run_argv[run_argv.index("-m") + 1] == expected_markers
    assert collect_argv[collect_argv.index("--run-level") + 1] == "full_model"
    assert run_argv[run_argv.index("--run-level") + 1] == "full_model"
    assert "exit 5" in collect_commands[0]
    assert "index($$0, node)" in collect_commands[0]

    command_timeout = f"{step['timeout_in_minutes'] - 15}m"
    assert run_argv[:4] == [
        "timeout",
        "--signal=TERM",
        "--kill-after=2m",
        command_timeout,
    ]
    assert "--durations=0" in run_argv
    assert any(arg.startswith("--junitxml=") for arg in run_argv)
    assert "|" in run_argv
    assert "tee" in run_argv
    assert "record_cache_metadata" not in all_commands


def _leaf_steps(pipeline_path: Path) -> list[dict]:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    return [step for entry in pipeline["steps"] for step in entry.get("steps", [entry])]


def _step_signature(step: dict) -> str:
    return yaml.safe_dump(step, sort_keys=True)


def _find_steps_selecting_path(pipeline_path: Path, test_path: str) -> list[dict]:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    matches = []

    def walk(steps: list[dict]) -> None:
        for step in steps:
            if any(
                arg == test_path or arg.startswith(f"{test_path}::")
                for command in step.get("commands", [])
                for arg in split(command)
            ):
                matches.append(step)
            walk(step.get("steps", []))

    walk(pipeline.get("steps", []))
    return matches


def _find_model_executor_cpu_shard_step(pipeline_path: Path) -> dict:
    matches = []
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))

    def walk(steps: list[dict]) -> None:
        for step in steps:
            commands = step.get("commands", [])
            if any(
                split(command)[:3] == ["pytest", "-sv", "tests/model_executor"]
                and "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT" in split(command)
                for command in commands
            ):
                matches.append(step)
            walk(step.get("steps", []))

    walk(pipeline.get("steps", []))
    assert len(matches) == 1, f"expected one model executor CPU shard step in {pipeline_path}"
    return matches[0]


def _find_diffusion_cpu_shard_step(pipeline_path: Path) -> dict:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    matches = []

    def walk(steps: list[dict]) -> None:
        for step in steps:
            commands = step.get("commands", [])
            if any(
                "pytest -sv tests/diffusion" in command and "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT" in command
                for command in commands
            ):
                matches.append(step)
            walk(step.get("steps", []))

    walk(pipeline.get("steps", []))
    assert len(matches) == 1, f"expected one diffusion CPU shard step in {pipeline_path}"
    return matches[0]


def test_ready_merge_composition_preserves_the_structural_union() -> None:
    ready_steps = _leaf_steps(AMD_READY_PIPELINE)
    merge_steps = _leaf_steps(AMD_MERGE_PIPELINE)
    ready_signatures = {_step_signature(step) for step in ready_steps}
    merge_signatures = {_step_signature(step) for step in merge_steps}

    shared = ready_signatures & merge_signatures
    ready_only = ready_signatures - merge_signatures
    merge_only = merge_signatures - ready_signatures
    assert shared
    assert ready_only
    assert merge_only

    combined = COMBINER.combine_test_suites(
        (
            f"READY_TESTS:{AMD_READY_PIPELINE}",
            f"MERGE_TESTS:{AMD_MERGE_PIPELINE}",
        )
    )
    assert [group["group"] for group in combined["steps"]] == [
        "READY_TESTS",
        "MERGE_TESTS",
    ]
    combined_ready_steps, combined_merge_steps = (group["steps"] for group in combined["steps"])
    combined_steps = combined_ready_steps + combined_merge_steps
    combined_signatures = [_step_signature(step) for step in combined_steps]

    assert {_step_signature(step) for step in combined_ready_steps} == ready_signatures
    assert {_step_signature(step) for step in combined_merge_steps} == merge_only
    assert len(combined_signatures) == len(set(combined_signatures))
    assert set(combined_signatures) == ready_signatures | merge_signatures
    assert sum(step.get("parallelism", 1) for step in combined_steps) < sum(
        step.get("parallelism", 1) for step in ready_steps + merge_steps
    )


@pytest.mark.parametrize("extra_pipeline", [AMD_NIGHTLY_PIPELINE, AMD_WEEKLY_PIPELINE], ids=["nightly", "weekly"])
def test_combined_extra_tier_preserves_every_distinct_step(extra_pipeline: Path) -> None:
    pipelines = [AMD_READY_PIPELINE, AMD_MERGE_PIPELINE, extra_pipeline]
    extra_group = "NIGHTLY_TESTS" if extra_pipeline == AMD_NIGHTLY_PIPELINE else "WEEKLY_TESTS"
    groups = ["READY_TESTS", "MERGE_TESTS", extra_group]
    raw_signatures = {_step_signature(step) for path in pipelines for step in _leaf_steps(path)}

    combined = COMBINER.combine_test_suites(tuple(f"{group}:{path}" for group, path in zip(groups, pipelines)))
    combined_signatures = [_step_signature(step) for group in combined["steps"] for step in group["steps"]]

    assert len(combined_signatures) == len(set(combined_signatures))
    assert set(combined_signatures) == raw_signatures


def test_omni_processor_full_matrix_moves_to_nightly_with_blocking_rocm_smoke() -> None:
    smoke_definitions = []
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
        steps = list(_walk_steps(pipeline["steps"]))

        assert all(step.get("label") != OMNI_PROCESSOR_FULL_LABEL for step in steps)
        assert all(step.get("env", {}).get(OMNI_PROCESSOR_TIMING_ENV) != "1" for step in steps)

        smoke = _find_step(OMNI_PROCESSOR_SMOKE_LABEL, pipeline_path)
        smoke_definitions.append(smoke)
        assert smoke["agent_pool"] == "mi300_1"
        assert smoke["depends_on"] == "amd-build"
        assert smoke["mirror_hardwares"] == ["amdproduction"]
        assert smoke["grade"] == "Blocking"
        assert smoke["timeout_in_minutes"] == 10
        assert smoke["commands"][0] == "export VLLM_ROCM_USE_AITER=0"

        argv = split(smoke["commands"][1])
        assert argv[:4] == ["timeout", "--signal=TERM", "--kill-after=1m", "8m"]
        assert f"{OMNI_PROCESSOR_TEST}::test_rocm_processor_snapshot_smoke" in argv
        assert argv[argv.index("-m") + 1] == "core_model and cpu and omni"
        assert "--durations=0" in argv
        assert "-k" not in argv

        model_executor_step = next(
            step for step in steps if step.get("label", "").startswith("Simple · Model Executor Test")
        )
        pytest_command = next(command for command in model_executor_step["commands"] if "pytest" in command)
        assert "core_model and cpu and not omni" in pytest_command

    assert smoke_definitions[0] == smoke_definitions[1]

    nightly = yaml.safe_load(AMD_NIGHTLY_PIPELINE.read_text(encoding="utf-8"))
    nightly_steps = list(_walk_steps(nightly["steps"]))
    lane_definitions = [step for step in nightly_steps if step.get("label") == OMNI_PROCESSOR_FULL_LABEL]
    assert len(lane_definitions) == 1

    step = lane_definitions[0]
    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 60
    assert step["env"] == {OMNI_PROCESSOR_TIMING_ENV: "1"}
    assert step["artifact_paths"] == ["artifacts/omni-processor/**/*"]
    assert step["commands"] == [
        "export VLLM_ROCM_USE_AITER=0",
        "bash .buildkite/amd/scripts/run-omni-processor-test.sh",
    ]
    assert [
        candidate["label"]
        for candidate in nightly_steps
        if candidate.get("env", {}).get(OMNI_PROCESSOR_TIMING_ENV) == "1"
    ] == [OMNI_PROCESSOR_FULL_LABEL]

    runner = AMD_OMNI_PROCESSOR_RUNNER.read_text(encoding="utf-8")
    assert runner.count("\npytest -s -v \\\n") == 1
    assert OMNI_PROCESSOR_TEST in runner
    assert "-m 'core_model and cpu and omni'" in runner
    assert "--durations=0" in runner
    assert '--junitxml="${artifact_root}/pytest.xml"' in runner
    assert 'tee "${artifact_root}/pytest.log"' in runner
    assert "record_cpu_context before" in runner
    assert "record_cpu_context after" in runner
    assert "pytest_status=${pipeline_status[0]}" in runner
    assert "tee_status=${pipeline_status[1]}" in runner
    assert 'exit "${pytest_status}"' in runner


@pytest.mark.parametrize(
    "pipeline_path",
    [AMD_READY_PIPELINE, AMD_MERGE_PIPELINE],
    ids=["ready", "merge"],
)
def test_model_executor_cpu_shards_keep_fast_personaplex_and_exclude_slow_stress(
    pipeline_path: Path,
) -> None:
    step = _find_model_executor_cpu_shard_step(pipeline_path)
    pytest_command = next(command for command in step["commands"] if "tests/model_executor" in command)
    argv = split(pytest_command)

    assert step["parallelism"] == 3
    assert argv[:3] == ["pytest", "-sv", "tests/model_executor"]
    assert argv[argv.index("-m") + 1] == MODEL_EXECUTOR_CPU_MARKERS
    assert all(not arg.startswith("--ignore") for arg in argv)
    assert [arg.removeprefix("--deselect=") for arg in argv if arg.startswith("--deselect=")] == (
        PERSONAPLEX_TEMPORAL_STRESS_NODES
    )
    assert "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT" in argv
    assert "--shard-id=$$BUILDKITE_PARALLEL_JOB" in argv
    assert "-k" not in argv


def test_personaplex_temporal_stress_has_one_nonblocking_nightly_owner() -> None:
    steps = _find_steps_selecting_path(
        AMD_NIGHTLY_PIPELINE,
        PERSONAPLEX_TEMPORAL_PATH,
    )
    assert len(steps) == 1
    step = steps[0]

    assert step["label"] == PERSONAPLEX_TEMPORAL_LABEL
    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 120
    assert step["artifact_paths"] == [PERSONAPLEX_TEMPORAL_ARTIFACTS]
    assert "depends_on" not in step
    assert "retry" not in step

    commands = step["commands"]
    all_commands = "\n".join(commands)
    pytest_commands = [command for command in commands if "pytest" in split(command)]
    collect_commands = [command for command in pytest_commands if "--collect-only" in split(command)]
    run_commands = [command for command in pytest_commands if "--collect-only" not in split(command)]
    assert len(collect_commands) == 1
    assert len(run_commands) == 1

    collect_argv = split(collect_commands[0])
    assert collect_argv[:3] == [
        "pytest",
        "--collect-only",
        "-q",
    ]
    assert collect_argv[3:4] == PERSONAPLEX_TEMPORAL_STRESS_NODES
    assert "$$PERSONAPLEX_TEMPORAL_ARTIFACT_DIR/collection.txt" in collect_commands[0]

    run_argv = split(run_commands[0])
    assert run_argv[:5] == [
        "timeout",
        "--signal=TERM",
        "--kill-after=2m",
        "110m",
        "pytest",
    ]
    assert run_argv[5:8] == [
        "-s",
        "-v",
        "-ra",
    ]
    assert run_argv[8:9] == PERSONAPLEX_TEMPORAL_STRESS_NODES
    assert "--durations=50" in run_argv
    assert "--junitxml=$$PERSONAPLEX_TEMPORAL_ARTIFACT_DIR/pytest.xml" in run_argv
    assert "$$PERSONAPLEX_TEMPORAL_ARTIFACT_DIR/pytest.log" in run_commands[0]
    assert "-n" not in run_argv
    assert "-k" not in run_argv
    assert all(not arg.startswith("--ignore") for arg in run_argv)
    assert "|| true" not in all_commands
    assert "VLLM_CI_ALLOW_NO_TESTS" not in all_commands
    assert "$$PERSONAPLEX_TEMPORAL_ARTIFACT_DIR/pytest-summary.txt" in all_commands


def test_cuda_ready_retains_full_model_executor_coverage() -> None:
    step = _find_step("Simple · Model Executor Test", CUDA_READY_PIPELINE)
    assert "source_file_dependencies" not in step

    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    argv = split(pytest_command)
    marker_expression = argv[argv.index("-m") + 1]

    assert "tests/model_executor" in argv
    assert marker_expression == "core_model and cpu"
    assert "not omni" not in marker_expression
    assert all(not arg.startswith("--ignore") for arg in argv)
    assert all(not arg.startswith("--deselect") for arg in argv)

    test_source = Path(PERSONAPLEX_TEMPORAL_PATH).read_text(encoding="utf-8")
    assert "pytestmark = [pytest.mark.core_model, pytest.mark.cpu]" in test_source
    assert "@pytest.mark.slow" not in test_source


def test_ar_paged_attention_gpu_lane_is_blocking_and_pinned() -> None:
    lane_definitions = []
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        group = _find_group(DIFFUSION_GROUP, pipeline_path)
        step = next((item for item in group["steps"] if item.get("label") == AR_PAGED_ATTENTION_LABEL), None)
        assert step is not None, f"missing {AR_PAGED_ATTENTION_LABEL} from {DIFFUSION_GROUP} in {pipeline_path}"
        lane_definitions.append(step)

        assert step["grade"] == "Blocking"
        assert step["timeout_in_minutes"] == 25
        assert "export VLLM_OMNI_AR_FA_REQUIRED=1" in step["commands"]

        pytest_commands = [command for command in step["commands"] if "pytest" in command]
        assert len(pytest_commands) == 2

        argv = split(pytest_commands[0])
        assert argv[:4] == ["timeout", "--signal=TERM", "--kill-after=2m", "15m"]
        assert "tests/diffusion/ar_diffusion/test_paged_attention.py" in argv
        assert argv[argv.index("-m") + 1] == AR_PAGED_ATTENTION_MARKERS
        assert argv[argv.index("--run-level") + 1] == "core_model"

        transfer_argv = split(pytest_commands[1])
        assert transfer_argv[:4] == ["timeout", "--signal=TERM", "--kill-after=1m", "5m"]
        assert "tests/diffusion/utils/test_chunked_transfer.py" in transfer_argv
        assert transfer_argv[transfer_argv.index("-m") + 1] == AR_PAGED_ATTENTION_MARKERS
        assert transfer_argv[transfer_argv.index("--run-level") + 1] == "core_model"

    assert lane_definitions[0] == lane_definitions[1]


def _find_joy_gpu_steps(pipeline_path: Path) -> list[dict]:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    matches = []

    def walk(steps: list[dict]) -> None:
        for step in steps:
            if step.get("label") == JOY_GPU_LABEL or any(
                arg == JOY_GPU_PATH or arg.startswith(f"{JOY_GPU_PATH}::")
                for command in step.get("commands", [])
                for arg in split(command)
            ):
                matches.append(step)
            walk(step.get("steps", []))

    walk(pipeline.get("steps", []))
    return matches


@pytest.mark.parametrize("pipeline_path", [AMD_READY_PIPELINE, AMD_MERGE_PIPELINE], ids=["ready", "merge"])
def test_joy_gpu_lane_is_not_on_amd_ready_or_merge_critical_path(pipeline_path: Path) -> None:
    assert not _find_joy_gpu_steps(pipeline_path)


def test_joy_gpu_nightly_lane_is_unique_nonblocking_and_preserves_default_backend() -> None:
    steps = _find_joy_gpu_steps(AMD_NIGHTLY_PIPELINE)
    assert len(steps) == 1
    step = steps[0]
    assert step["label"] == JOY_GPU_LABEL
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_1"
    assert step["depends_on"] == "amd-build"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 90
    assert step["artifact_paths"] == ["artifacts/joy-image-gpu/*.xml"]
    assert 'export JOY_GPU_ARTIFACT_DIR="$$BUILDKITE_BUILD_CHECKOUT_PATH/artifacts/joy-image-gpu"' in step["commands"]
    assert 'export AITER_JIT_DIR="/tmp/vllm-omni-joy-aiter-$$BUILDKITE_JOB_ID"' in step["commands"]
    assert 'export TORCH_EXTENSIONS_DIR="/tmp/vllm-omni-joy-torch-$$BUILDKITE_JOB_ID"' in step["commands"]
    assert 'mkdir -p "$$AITER_JIT_DIR" "$$TORCH_EXTENSIONS_DIR" "$$JOY_GPU_ARTIFACT_DIR"' in step["commands"]
    assert "export VLLM_ROCM_USE_AITER=1" in step["commands"]
    assert all("DIFFUSION_ATTENTION_BACKEND" not in command for command in step["commands"])
    pytest_commands = [command for command in step["commands"] if "pytest" in split(command)]
    assert len(pytest_commands) == 1
    assert "|| true" not in pytest_commands[0]
    argv = split(pytest_commands[0])
    assert argv[:4] == ["timeout", "--signal=TERM", "--kill-after=2m", "80m"]
    assert argv.count(JOY_GPU_PATH) == 1
    assert "-k" not in argv
    assert argv[argv.index("-m") + 1] == AR_PAGED_ATTENTION_MARKERS
    assert argv[argv.index("--run-level") + 1] == "core_model"
    assert "--durations=20" in argv
    assert "--junitxml=$$JOY_GPU_ARTIFACT_DIR/pytest.xml" in argv


def test_joy_gpu_suite_remains_selected_by_cuda_model_lane() -> None:
    pipeline_path = Path(".buildkite/cuda/test-ready.yml")
    step = _find_step("Diffusion · Model Test", pipeline_path)
    argv = split(next(command for command in step["commands"] if "pytest" in command))
    assert "tests/diffusion/models/" in argv
    assert f"--ignore={JOY_GPU_PATH}" not in argv
    assert argv[argv.index("-m") + 1] == "core_model and cuda"
    # Module-level CPU marks cannot be cancelled by adding GPU marks.
    assert "pytest.mark.cpu" not in Path(JOY_GPU_PATH).read_text(encoding="utf-8")


# These jobs previously selected CUDA-marked items on AMD. Each migrated lane
# now selects its qualified ROCm variants and verifies the exact collection
# before running the GPU workload.
_ROCM_MIGRATED_AMD_GPU_JOBS = (
    (
        "Diffusion · Model Test",
        AMD_READY_PIPELINE,
        "Blocking",
        117,
        "core_model and rocm and MI325 and not nvidia_only and not "
        "(cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)",
    ),
    (
        "ROCm · Model Executor GPU Coverage (R2-01) · Shard %N/%t",
        AMD_READY_PIPELINE,
        "NonBlocking",
        1416,
        "core_model and rocm and MI325 and not nvidia_only and not "
        "(cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)",
    ),
    (
        "ROCm · Entrypoints GPU Coverage (R2-02)",
        AMD_READY_PIPELINE,
        "NonBlocking",
        3,
        "core_model and rocm and MI325 and not nvidia_only and not "
        "(cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)",
    ),
    (
        "Diffusion · LoRA Test",
        AMD_MERGE_PIPELINE,
        "Blocking",
        1,
        "advanced_model and rocm and MI325 and cards_1 and not nvidia_only",
    ),
    (
        "Diffusion · Wan22 Offline Test",
        AMD_MERGE_PIPELINE,
        "Blocking",
        2,
        "advanced_model and rocm and MI325 and cards_1 and diffusion and not nvidia_only",
    ),
    (
        "Diffusion · Wan22 Online Test",
        AMD_MERGE_PIPELINE,
        "Blocking",
        1,
        "advanced_model and rocm and MI325 and cards_1 and diffusion and not nvidia_only",
    ),
    (
        "Omni Model Test Qwen3-Omni",
        AMD_MERGE_PIPELINE,
        "Blocking",
        17,
        "advanced_model and rocm and MI325 and not nvidia_only",
    ),
    (
        "Qwen3-Omni SHM Leak Test",
        AMD_MERGE_PIPELINE,
        "NonBlocking",
        1,
        "advanced_model and rocm and MI325 and cards_2 and not nvidia_only",
    ),
    (
        "Qwen3-TTS Base E2E Test",
        AMD_MERGE_PIPELINE,
        "Blocking",
        3,
        "advanced_model and rocm and MI325 and cards_1 and not nvidia_only",
    ),
)


def test_ready_and_merge_pipelines_have_no_cuda_marker_selectors() -> None:
    assert _amd_marker_selector_inventory("cuda") == set()


def test_migrated_amd_gpu_jobs_select_exact_rocm_collections() -> None:
    for label, pipeline_path, grade, expected_count, marker_expression in _ROCM_MIGRATED_AMD_GPU_JOBS:
        step = _find_step(label, pipeline_path)
        assert step.get("grade") == grade

        collect_commands = [command for command in step["commands"] if AMD_COLLECTION_GUARD.name in command]
        run_commands = [
            command
            for command in step["commands"]
            if AMD_COLLECTION_GUARD.name not in command
            and marker_expression in _marker_expressions({"commands": [command]})
        ]
        assert len(collect_commands) == 1
        assert len(run_commands) == 1
        assert _marker_expressions({"commands": collect_commands}) == [marker_expression]
        assert _marker_expressions({"commands": run_commands}) == [marker_expression]

        collect_argv = split(collect_commands[0])
        bash_at = collect_argv.index("bash")
        assert collect_argv[bash_at : bash_at + 2] == ["bash", str(AMD_COLLECTION_GUARD)]
        assert collect_argv[bash_at + 2] == str(expected_count)
        if "Shard %N/%t" in label:
            assert '$${BUILDKITE_PARALLEL_JOB:-0}" == "0"' in collect_commands[0]


def test_amd_collection_guard_rejects_empty_and_changed_collections() -> None:
    guard = AMD_COLLECTION_GUARD.read_text(encoding="utf-8")
    assert "set -euo pipefail" in guard
    assert 'pytest --collect-only -q "$@"' in guard
    assert "^tests\\/.*::" in guard
    assert '"${actual_count}" != "${expected_count}"' in guard


@pytest.mark.parametrize(("label", "agent_pool", "expected_count", "markers", "artifacts"), TINY_DIFFUSION_JOBS)
def test_tiny_diffusion_multigpu_nightly_uses_native_rocm_collection(
    label: str,
    agent_pool: str,
    expected_count: int,
    markers: str,
    artifacts: str,
) -> None:
    step = _find_step(label, AMD_NIGHTLY_PIPELINE)

    assert step["agent_pool"] == agent_pool
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 40
    assert step["artifact_paths"] == [artifacts]
    assert "retry" not in step

    commands = step["commands"]
    collect_commands = [command for command in commands if AMD_COLLECTION_GUARD.name in command]
    run_commands = [command for command in commands if "pytest -s" in command]
    assert len(collect_commands) == 1
    assert len(run_commands) == 1

    collect_command = collect_commands[0]
    run_command = run_commands[0]
    collect_argv = split(collect_command)
    run_argv = split(run_command)
    assert collect_argv[:2] == ["bash", str(AMD_COLLECTION_GUARD)]
    assert collect_argv[2] == str(expected_count)

    for command, argv in ((collect_command, collect_argv), (run_command, run_argv)):
        assert TINY_DIFFUSION_PATH in argv
        assert _marker_expressions({"commands": [command]}) == [markers]
        assert argv[argv.index("--run-level") + 1] == "core_model"

    for cuda_marker in ("cuda", "L4", "H100", "B200"):
        assert re.search(rf"\b{cuda_marker}\b", markers) is None

    all_commands = "\n".join(commands)
    assert "collection.txt" in all_commands
    assert "pytest.xml" in all_commands
    assert "pytest.log" in all_commands
    assert "pytest-summary.txt" in all_commands


def test_wan_sdpa_alignment_cpu_case_stays_shared_and_cuda_kernel_stays_nvidia() -> None:
    cpu_case = Path("tests/diffusion/distributed/test_wan_attention_sdpa_alignment.py").read_text(encoding="utf-8")
    cuda_case = Path("tests/diffusion/distributed/test_wan_attention_sdpa_alignment_cuda.py").read_text(
        encoding="utf-8"
    )
    assert "pytest.mark.cpu" in cpu_case
    assert "rocm" not in cpu_case
    assert 'res={"cuda": "L4"}' in cuda_case
    assert "rocm" not in cuda_case


def test_joy_gpu_suite_is_not_duplicated_in_amd_model_lane() -> None:
    step = _find_step("Diffusion · Model Test", AMD_READY_PIPELINE)
    argv = split(
        next(command for command in step["commands"] if split(command)[:4] == ["timeout", "15m", "pytest", "-sv"])
    )

    assert step["grade"] == "Blocking"
    assert step["agent_pool"] == "mi300_1"
    assert argv[:4] == ["timeout", "15m", "pytest", "-sv"]
    assert f"--ignore={JOY_GPU_PATH}" in argv
    assert "--ignore=tests/diffusion/models/auk/test_auk_vae_fastpath.py" in argv
    assert "--ignore=tests/diffusion/models/mammoth_moda2/test_dit_attention_cuda.py" in argv
    assert argv[argv.index("-m") + 1] == (
        "core_model and rocm and MI325 and not nvidia_only and not "
        "(cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)"
    )
    assert argv[argv.index("--run-level") + 1] == "core_model"


def test_joy_gpu_helper_import_does_not_require_pytest_package_mode() -> None:
    source = Path(JOY_GPU_PATH).read_text(encoding="utf-8")
    assert "from . import" not in source
    assert "from tests.diffusion.models.joy_image import test_joy_image as joy_units" in source
    spec = find_spec("tests.diffusion.models.joy_image.test_joy_image")
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).resolve() == Path(JOY_GPU_PATH).with_name("test_joy_image.py").resolve()


@pytest.mark.parametrize("label", ["Diffusion · Batch Test", "Diffusion · Offloader Test", "Diffusion · Cache Test"])
def test_ready_diffusion_gpu_lanes_select_rocm_variants(label: str) -> None:
    step = _find_step(label, AMD_READY_PIPELINE)
    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    argv = split(pytest_command)

    assert argv[argv.index("-m") + 1] == "core_model and rocm and MI325 and cards_1"


@pytest.mark.parametrize(
    ("label", "pipeline", "cards"),
    [
        ("ROCm · Ulysses UAA Attention · 2-GPU", AMD_READY_PIPELINE, "cards_2"),
        ("ROCm · Ulysses UAA Attention · 4-GPU", AMD_MERGE_PIPELINE, "cards_4"),
    ],
)
def test_amd_ulysses_lanes_cover_uaa_and_attention_sp(label: str, pipeline: Path, cards: str) -> None:
    step = _find_step(label, pipeline)
    pytest_commands = [command for command in step["commands"] if "pytest" in command]

    assert step["grade"] == "Blocking"
    assert len(pytest_commands) == 1
    assert "tests/diffusion/attention/test_ulysses_uaa.py" in pytest_commands[0]
    assert "tests/diffusion/attention/test_attention_sp.py" in pytest_commands[0]
    assert all(f"core_model and rocm and MI325 and {cards}" in command for command in pytest_commands)

    argv = split(pytest_commands[0])
    assert argv[:4] == ["timeout", "--signal=TERM", "--kill-after=2m", "20m"]
    assert step["timeout_in_minutes"] == 25
    assert "--durations=20" in argv


@pytest.mark.parametrize(
    ("label", "cards"),
    [
        ("Diffusion · Distributed Test · 2-GPU", "cards_2"),
        ("Diffusion · Distributed & Attention Test · 4-GPU", "cards_4"),
    ],
)
def test_cuda_distributed_lanes_cover_both_attention_files_within_the_job_budget(label: str, cards: str) -> None:
    step = _find_step(label, CUDA_READY_PIPELINE)
    argv = [split(command) for command in step["commands"] if "pytest" in split(command)]

    paths = {arg for command in argv for arg in command if arg.startswith("tests/")}
    assert paths == {
        "tests/diffusion/distributed",
        "tests/diffusion/attention/test_ulysses_uaa.py",
        "tests/diffusion/attention/test_attention_sp.py",
    }
    assert step["mirror_hardwares"] == f"l4_{cards.removeprefix('cards_')}"
    for command in argv:
        assert command[command.index("-m") + 1] == f"core_model and cuda and L4 and {cards}"
        assert command[command.index("--run-level") + 1] == "core_model"
    command_budget = sum(int(command[1].removesuffix("m")) for command in argv)
    assert step["timeout_in_minutes"] >= command_budget + 5


def test_qwen3_tts_base_preserves_advanced_model_arguments() -> None:
    step = _find_step("Qwen3-TTS Base E2E Test")
    commands = step["commands"]

    assert all("bash -c" not in command for command in commands)
    pytest_command = next(command for command in commands if "pytest" in split(command))
    argv = split(pytest_command)

    marker_index = argv.index("-m")
    run_level_index = argv.index("--run-level")
    assert argv[marker_index + 1] == "advanced_model and rocm and MI325 and cards_1 and not nvidia_only"
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


@pytest.mark.parametrize(
    "pipeline_path",
    [AMD_READY_PIPELINE, AMD_MERGE_PIPELINE],
    ids=["ready", "merge"],
)
def test_diffusion_cpu_suite_shard_contract(pipeline_path: Path) -> None:
    step = _find_diffusion_cpu_shard_step(pipeline_path)
    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    argv = split(pytest_command)

    assert step["label"].endswith("Shard %N/%t")
    assert step["parallelism"] == 5
    assert step["timeout_in_minutes"] == 60
    assert step["artifact_paths"] == [DIFFUSION_CPU_ARTIFACTS]
    assert (
        'export DIFFUSION_CPU_ARTIFACT_DIR="$$BUILDKITE_BUILD_CHECKOUT_PATH/artifacts/amd-diffusion-cpu-shards"'
        in step["commands"]
    )
    assert 'mkdir -p "$$DIFFUSION_CPU_ARTIFACT_DIR"' in step["commands"]
    assert argv[:3] == ["pytest", "-sv", "tests/diffusion"]
    assert argv[argv.index("-m") + 1] == DIFFUSION_CPU_MARKERS
    assert "--ignore=tests/diffusion/cache/test_teacache_extractors.py" in argv
    assert "--ignore=tests/diffusion/models/pi05/test_pi05_units.py" in argv
    assert [arg.removeprefix("--deselect=") for arg in argv if arg.startswith("--deselect=")] == (
        DIFFUSION_SLOW_CPU_NODES
    )
    assert "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT" in pytest_command
    assert "--shard-id=$$BUILDKITE_PARALLEL_JOB" in pytest_command
    assert "--durations=50" in argv
    assert (
        "--junitxml=$$DIFFUSION_CPU_ARTIFACT_DIR/"
        "pytest-$${BUILDKITE_PARALLEL_JOB}-of-$${BUILDKITE_PARALLEL_JOB_COUNT}.xml" in argv
    )
    assert "|| true" not in pytest_command


@pytest.mark.parametrize(
    "pipeline_path",
    [AMD_READY_PIPELINE, AMD_MERGE_PIPELINE],
    ids=["ready", "merge"],
)
def test_diffusion_slow_cpu_cases_have_one_blocking_owner(pipeline_path: Path) -> None:
    step = _find_step(DIFFUSION_SLOW_CPU_LABEL, pipeline_path)
    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    argv = split(pytest_command)
    option_at = next(index for index, arg in enumerate(argv) if arg.startswith("-") and arg != "-sv")
    nodes = argv[2:option_at]

    assert step["agent_pool"] == "mi300_1"
    assert step["depends_on"] == "amd-build"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "Blocking"
    assert step["timeout_in_minutes"] == 40
    assert step["artifact_paths"] == [DIFFUSION_SLOW_CPU_ARTIFACTS]
    assert step["retry"] == {"automatic": [{"exit_status": 134, "limit": 1}]}
    assert "export VLLM_ROCM_USE_AITER=0" in step["commands"]
    assert (
        'export DIFFUSION_SLOW_CPU_ARTIFACT_DIR="$$BUILDKITE_BUILD_CHECKOUT_PATH/artifacts/amd-diffusion-slow-cpu"'
        in step["commands"]
    )
    assert 'mkdir -p "$$DIFFUSION_SLOW_CPU_ARTIFACT_DIR"' in step["commands"]
    assert argv[:2] == ["pytest", "-sv"]
    assert nodes == DIFFUSION_SLOW_CPU_NODES
    assert argv[argv.index("-m") + 1] == DIFFUSION_CPU_MARKERS
    assert "--durations=20" in argv
    assert "--junitxml=$$DIFFUSION_SLOW_CPU_ARTIFACT_DIR/pytest.xml" in argv
    assert "--num-shards" not in pytest_command
    assert "|| true" not in pytest_command

    shard = _find_diffusion_cpu_shard_step(pipeline_path)
    shard_argv = split(next(command for command in shard["commands"] if "pytest" in command))
    shard_nodes = [arg.removeprefix("--deselect=") for arg in shard_argv if arg.startswith("--deselect=")]
    assert shard_nodes == nodes


def test_diffusion_slow_cpu_cases_stay_off_nightly() -> None:
    pipeline = yaml.safe_load(AMD_NIGHTLY_PIPELINE.read_text(encoding="utf-8"))
    labels = [step.get("label") for step in _walk_steps(pipeline["steps"])]
    assert DIFFUSION_SLOW_CPU_LABEL not in labels


AITER_MHA_CACHE_PREPARE = 'eval "$$(python .buildkite/amd/scripts/aiter_mha_cache.py prepare)"'
AITER_MHA_CACHE_PUBLISH = "python .buildkite/amd/scripts/aiter_mha_cache.py publish"
STABLE_AUDIO_LAYERWISE_NODE = (
    "tests/diffusion/offloader/test_diffusion_layerwise_offload.py"
    "::test_layerwise_offload_diffusion_model[stabilityai/stable-audio-open-1.0]"
)


def _assert_publish_follows_successful_pytest(command: str, pytest_marker: str) -> None:
    then_at = command.index("; then\n")
    else_at = command.index("\nelse\n")
    publish_at = command.index(AITER_MHA_CACHE_PUBLISH)

    assert command.startswith("if ")
    assert pytest_marker in command[:then_at]
    assert then_at < publish_at < else_at
    assert "pytest_status=$$?" in command[else_at:]
    assert 'exit "$${pytest_status}"' in command[else_at:]
    assert "&&" not in command


def _assert_build_scoped_aiter_mha_cache(step: dict, pytest_marker: str, *, publishes: bool) -> None:
    commands = step["commands"]
    prepare_at = commands.index(AITER_MHA_CACHE_PREPARE)
    pytest_at = next(index for index, command in enumerate(commands) if pytest_marker in command)

    assert prepare_at < pytest_at
    assert not any(command.strip() == AITER_MHA_CACHE_PUBLISH for command in commands)
    if publishes:
        _assert_publish_follows_successful_pytest(commands[pytest_at], pytest_marker)
    else:
        assert AITER_MHA_CACHE_PUBLISH not in "\n".join(commands)
    assert "AITER_JIT_DIR=" not in "\n".join(commands)
    assert "HF_HOME" not in "\n".join(commands)
    assert "GPU_ARCHS=" not in "\n".join(commands)
    assert all("DIFFUSION_ATTENTION_BACKEND" not in command for command in commands)


def test_z_image_merge_timeout_covers_cold_aiter_compile() -> None:
    step = _find_step("Diffusion Model Test")
    pytest_command = next(command for command in step["commands"] if "test_z_image.py" in command)
    condition = pytest_command.split("; then", 1)[0]

    assert split(condition)[1:3] == ["timeout", "55m"]
    _assert_build_scoped_aiter_mha_cache(
        step,
        "tests/e2e/offline_inference/test_z_image.py",
        publishes=True,
    )


@pytest.mark.parametrize(
    ("label", "pytest_marker"),
    [
        ("Custom Pipeline Test", "tests/e2e/features/custom_pipeline/"),
        ("Diffusion · Batch Test", "tests/diffusion/batching"),
    ],
)
def test_ready_jobs_restore_bf16_aiter_module_without_publishing(label: str, pytest_marker: str) -> None:
    step = _find_step(label, AMD_READY_PIPELINE)

    assert step["agent_pool"] == "mi300_1"
    assert step["depends_on"] == "amd-fa-aiter"
    assert step["env"] == {"VLLM_OMNI_AITER_MHA_SAME_BUILD_STEP": "amd-fa-aiter"}
    assert step["mirror_hardwares"] == ["amdproduction"]
    _assert_build_scoped_aiter_mha_cache(step, pytest_marker, publishes=False)
    assert all("VLLM_ROCM_USE_AITER=0" not in command for command in step["commands"])


def test_ready_flashattention_publishes_bf16_aiter_module_on_success() -> None:
    step = _find_step("ROCm · FlashAttention/AITER GPU Coverage", AMD_READY_PIPELINE)
    pytest_commands = [command for command in step["commands"] if "pytest" in command]

    assert step["agent_pool"] == "mi300_1"
    assert step["key"] == "amd-fa-aiter"
    assert step["depends_on"] == "amd-build"
    assert "VLLM_OMNI_AITER_MHA_SAME_BUILD_STEP" not in step.get("env", {})
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 60
    assert "export VLLM_ROCM_USE_AITER=1" in step["commands"]
    _assert_build_scoped_aiter_mha_cache(step, "tests/diffusion/attention/", publishes=True)
    assert len(pytest_commands) == 2
    assert all("-n 1" in command for command in pytest_commands)
    assert "tests/diffusion/models/mammoth_moda2/test_dit_attention_cuda.py" in pytest_commands[1]
    assert "core_model and rocm and MI325 and cards_1" in pytest_commands[0]
    assert AITER_MHA_CACHE_PUBLISH not in pytest_commands[1]


def test_stable_audio_layerwise_reuses_the_bf16_aiter_module_cache() -> None:
    step = _find_step("Diffusion Stable Audio Layerwise Offload Test (quarantined)")

    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_1"
    _assert_build_scoped_aiter_mha_cache(step, STABLE_AUDIO_LAYERWISE_NODE, publishes=False)

    blocking = _find_step("Diffusion Model CPU offloading Test")
    assert "export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA" in blocking["commands"]
    assert AITER_MHA_CACHE_PREPARE not in blocking["commands"]


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


def test_amd_template_supplies_a_server_verified_aiter_artifact_source() -> None:
    template = AMD_TEMPLATE.read_text(encoding="utf-8")
    bootstrap = AMD_BOOTSTRAP.read_text(encoding="utf-8")

    assert template.count("VLLM_OMNI_AITER_MHA_TRUSTED_COMMIT") == 2
    assert "VLLM_AITER_MHA_TRUSTED_COMMIT=$(git cat-file -p HEAD | sed -n 's/^parent //p' | head -n 1)" in bootstrap
    assert "VLLM_AITER_MHA_TRUSTED_COMMIT=$DIFF_BASE" in bootstrap
    assert bootstrap.count("git merge-base origin/main HEAD") == 1
    assert '-D vllm_merge_base_commit="$DIFF_BASE"' in bootstrap
    assert '-D vllm_aiter_mha_trusted_commit="$VLLM_AITER_MHA_TRUSTED_COMMIT"' in bootstrap


def test_weekly_reliability_is_distinct_from_nightly() -> None:
    weekly = yaml.safe_load(AMD_WEEKLY_PIPELINE.read_text(encoding="utf-8"))
    nightly = AMD_NIGHTLY_PIPELINE.read_text(encoding="utf-8")
    group = weekly["steps"][0]
    jobs = {
        "Reliability Test · qwen3-omni": ("mi300_2", "cards_2", "tests/dfx/reliability/test_qwen3_omni.py"),
        "Reliability Test · wan22": ("mi300_1", "cards_1", "tests/dfx/reliability/test_wan22.py"),
        "Reliability Test · hunyuan-image": ("mi300_4", "cards_4", "tests/dfx/reliability/test_hunyuan_image.py"),
        "Reliability Test · Invalid parameters · MI300 · 2-GPU": (
            "mi300_2",
            "cards_2",
            "test_invalid_omni_chat.py",
        ),
    }

    assert group["group"] == "Reliability Test"
    assert "if" not in group
    assert [step["label"] for step in group["steps"]] == list(jobs)
    for step in group["steps"]:
        pool, cards, filename = jobs[step["label"]]
        command = "\n".join(step["commands"])
        assert "if" not in step
        assert step["agent_pool"] == pool
        assert step["grade"] == "NonBlocking"
        assert step["depends_on"] == "amd-build"
        assert "slow and rocm and MI325 and " + cards in command
        assert filename in command
        assert filename not in nightly
    assert "tests/dfx/reliability/" not in nightly
    assert "Qwen3-Omni Function Expansion" not in AMD_WEEKLY_PIPELINE.read_text(encoding="utf-8")
    hunyuan_step = _find_step("Reliability Test · hunyuan-image", AMD_WEEKLY_PIPELINE)
    hunyuan_command = "\n".join(hunyuan_step["commands"])
    assert "test_hunyuan_image_ep_startup.py" not in hunyuan_command


def test_hunyuan_rocm_reliability_disables_expert_parallel() -> None:
    from vllm_omni.config.stage_config import resolve_deploy_yaml

    base = resolve_deploy_yaml("vllm_omni/deploy/hunyuan_image3_dit.yaml")
    base_parallel = base["stages"][0]["parallel_config"]
    reliability = Path("tests/dfx/reliability/test_hunyuan_image.py").read_text(encoding="utf-8")
    startup = Path("tests/dfx/reliability/test_hunyuan_image_ep_startup.py").read_text(encoding="utf-8")
    accuracy = Path("tests/e2e/accuracy/test_hunyuan_image3_pixel_accuracy.py").read_text(encoding="utf-8")
    step = _find_step("HunyuanImage Expert Parallel Startup", AMD_WEEKLY_PIPELINE)
    command = "\n".join(step["commands"])
    reliability_step = _find_step("Reliability Test · hunyuan-image", AMD_WEEKLY_PIPELINE)
    reliability_command = "\n".join(reliability_step["commands"])
    nightly_accuracy = _find_step("HunyuanImage3 Offline Pixel Accuracy", AMD_NIGHTLY_PIPELINE)
    dependencies = yaml.safe_load(AMD_SOURCE_DEPENDENCIES.read_text(encoding="utf-8"))["source_file_dependencies"][
        "diffusion_hunyuan_image3_reliability"
    ]

    assert base_parallel["enable_expert_parallel"] is True
    assert base_parallel["tensor_parallel_size"] == 4
    assert "current_omni_platform.is_rocm()" in reliability
    assert "hunyuan_image3_dit.yaml" in reliability
    assert '"parallel_config.enable_expert_parallel": False' in reliability
    assert 'params["update"]' in reliability
    assert "hunyuan_image3_dit_rocm_reliability.yaml" not in reliability
    assert "hunyuan_image3_dit.yaml" in startup
    assert "send_health_http_request" in startup
    assert "send_images_generations_http_request" not in startup
    assert "hunyuan_image3_dit_rocm_reliability.yaml" not in startup
    assert '"enable_expert_parallel": True' in accuracy
    assert step["agent_pool"] == "mi300_4"
    assert step["grade"] == "NonBlocking"
    assert step["depends_on"] == "amd-build"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 45
    assert step["source_file_dependencies"] == "diffusion_hunyuan_image3_reliability"
    assert step["artifact_paths"] == ["artifacts/rocm-hunyuan-image-ep-startup/**/*"]
    assert "full_model and slow and diffusion and rocm and MI325 and cards_4" in command
    assert "test_hunyuan_image_ep_startup.py" in command
    assert "Expected exactly one HunyuanImage expert-parallel startup node" in command
    assert command.count("--shard-id=0 --num-shards=1") == 2
    assert "timeout --signal=TERM --kill-after=2m 30m" in command
    assert "Expected exactly one executed HunyuanImage expert-parallel startup test" in command
    assert "skipped != 0" in command
    assert reliability_step["source_file_dependencies"] == "diffusion_hunyuan_image3_reliability"
    assert reliability_step["env"] == {
        "VLLM_ROCM_USE_AITER": "0",
        "DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA",
    }
    assert reliability_step["artifact_paths"] == ["artifacts/rocm-hunyuan-image-reliability/**/*"]
    assert reliability_command.count("--shard-id=0 --num-shards=1") == 2
    assert "Expected exactly 16 HunyuanImage reliability nodes" in reliability_command
    assert "Expected exactly 16 executed HunyuanImage reliability tests" in reliability_command
    assert "skipped != 0" in reliability_command
    assert "vllm_omni/deploy/hunyuan_image3_dit_rocm_reliability.yaml" not in dependencies
    assert "tests/dfx/reliability/test_hunyuan_image_ep_startup.py" in dependencies
    assert "env" not in step
    assert nightly_accuracy["grade"] == "NonBlocking"
    assert nightly_accuracy["agent_pool"] == "mi300_4"
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE, AMD_NIGHTLY_PIPELINE):
        assert "HunyuanImage Expert Parallel Startup" not in pipeline_path.read_text(encoding="utf-8")


def test_weekly_omni_e2e_is_full_weight_and_weekly_only() -> None:
    weekly_text = AMD_WEEKLY_PIPELINE.read_text(encoding="utf-8")
    group = _find_group("Omni E2E Weekly", AMD_WEEKLY_PIPELINE)
    jobs = {
        "MiMo-Audio · 1-GPU": (
            "mi300_1",
            120,
            "cards_1",
            "tests/e2e/online_serving/test_mimo_audio_expansion.py",
            ("test_text_to_text_001",),
            "artifacts/rocm-mimo-audio-weekly/**/*",
            "Expected exactly one MiMo-Audio ROCm node",
            "omni_mimo_audio_function",
            "Expected exactly one executed MiMo-Audio test",
        ),
        "Ming-flash-omni offline · 4-GPU": (
            "mi300_4",
            180,
            "cards_4",
            "tests/e2e/offline_inference/test_ming_flash_omni_expansion.py",
            (
                "test_text_to_text",
                "test_image_to_text",
                "test_audio_to_text",
                "test_video_to_text",
                "test_mixed_to_text",
                "test_text_to_audio",
                "test_image_to_audio",
            ),
            "artifacts/rocm-ming-flash-omni-weekly/**/*",
            "Expected exactly $${#nodes[@]} Ming-flash-omni offline ROCm nodes",
            "omni_ming_flash_omni_function",
            "Expected exactly seven executed Ming-flash-omni offline tests",
        ),
    }

    assert group["group"] == "Omni E2E Weekly"
    assert "if" not in group
    assert [step["label"] for step in group["steps"]] == list(jobs)
    mimo_source = Path("tests/e2e/online_serving/test_mimo_audio_expansion.py").read_text(encoding="utf-8")
    assert "CI failed 8571" in mimo_source
    dependencies = yaml.safe_load(
        Path(".buildkite/common/ci_source_file_dependencies.yml").read_text(encoding="utf-8")
    )["source_file_dependencies"]
    for step in group["steps"]:
        pool, timeout, cards, filename, nodes, artifact, expected, dependency, execution = jobs[step["label"]]
        command = "\n".join(step["commands"])
        assert "if" not in step
        assert step["agent_pool"] == pool
        assert step["grade"] == "NonBlocking"
        assert step["depends_on"] == "amd-build"
        assert step["mirror_hardwares"] == ["amdproduction"]
        assert step["timeout_in_minutes"] == timeout
        assert step["source_file_dependencies"] == dependency
        assert step["artifact_paths"] == [artifact]
        assert f"slow and rocm and MI325 and omni and {cards}" in command
        assert "--run-level full_model" in command
        assert filename in command
        assert expected in command
        assert execution in command
        assert command.count("--shard-id=0 --num-shards=1") == 2
        assert "skipped != 0" in command
        assert "BUILDKITE_BUILD_CHECKOUT_PATH" in command
        for node in nodes:
            assert node in command
        assert filename in dependencies[dependency]
    mimo_command = "\n".join(_find_step("MiMo-Audio · 1-GPU", AMD_WEEKLY_PIPELINE)["commands"])
    assert "test_audio_to_text_audio_001" not in mimo_command
    assert "tests/e2e/online_serving/test_ming_flash_omni_expansion.py" not in weekly_text
    assert "test_covo_audio_expansion.py" not in weekly_text
    assert "test_qwen2_5_omni_expansion.py" not in weekly_text
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE, AMD_NIGHTLY_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        for label in jobs:
            assert label not in text


def test_cosyvoice_offline_zero_shot_runs_nightly_on_rocm() -> None:
    path = "tests/e2e/offline_inference/test_cosyvoice3_expansion.py"
    markers = "slow and tts and rocm and MI325 and cards_1"
    step = _find_step("CosyVoice3 Offline Zero-shot Smoke", AMD_NIGHTLY_PIPELINE)

    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 90
    assert step["retry"] == {"automatic": [{"exit_status": 134, "limit": 1}]}
    assert step["artifact_paths"] == ["artifacts/rocm-cosyvoice3-offline-nightly/**/*"]
    assert "env" not in step

    pytest_commands = [command for command in step["commands"] if "pytest" in split(command)]
    assert len(pytest_commands) == 2
    collect_command = next(command for command in pytest_commands if "--collect-only" in split(command))
    run_command = next(command for command in pytest_commands if "--collect-only" not in split(command))
    for command in (collect_command, run_command):
        argv = split(command)
        assert path in argv
        assert argv[argv.index("-m") + 1] == markers
        assert argv[argv.index("--run-level") + 1] == "full_model"
    assert "--junitxml=/tmp/cosyvoice3-offline-pytest.xml" in split(run_command)

    commands = "\n".join(step["commands"])
    assert "DIFFUSION_ATTENTION_BACKEND" not in commands
    assert "VLLM_ROCM_USE_AITER" not in commands
    assert "cp -v /tmp/cosyvoice3-offline-pytest.xml" in commands

    source = Path(path).read_text(encoding="utf-8")
    decorator = source.rsplit("@hardware_test", 1)[1].split("def test_cosyvoice3_offline_reference_zero_shot", 1)[0]
    assert '"cuda": "L4"' in decorator
    assert '"rocm": "MI325"' in decorator
    assert 'id="sync"' in source
    assert 'id="async_chunk"' in source


def test_stable_audio_cpu_offload_has_one_nonblocking_nightly_owner() -> None:
    label = "Stable Audio CPU Offload"
    step = _find_step(label, AMD_NIGHTLY_PIPELINE)
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 60
    assert step["artifact_paths"] == ["artifacts/rocm-stable-audio-cpu-offload/**/*"]
    assert "retry" not in step
    assert "env" not in step
    _assert_build_scoped_aiter_mha_cache(
        step,
        "tests/diffusion/offloader/test_diffusion_cpu_offload.py",
        publishes=False,
    )

    commands = "\n".join(step["commands"])
    assert "DIFFUSION_ATTENTION_BACKEND" not in commands
    assert "VLLM_ROCM_USE_AITER" not in commands
    assert "test_diffusion_layerwise_offload.py" not in commands
    assert "tests/diffusion/offloader/test_diffusion_cpu_offload.py" in commands
    assert "full_model and rocm and MI325 and cards_1" in commands
    assert "--run-level full_model" in commands
    assert "--collect-only" in commands
    assert "exit 5" in commands
    assert "pytest.xml" in commands
    assert "pytest.log" in commands

    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert label not in text
        assert "full_model and rocm and MI325 and cards_1" not in text


def test_amd_runner_preserves_piped_failures_and_empty_collection() -> None:
    runner = AMD_RUNNER.read_text(encoding="utf-8")
    template = AMD_TEMPLATE.read_text(encoding="utf-8")

    assert "set -euo pipefail" in runner
    assert "if /bin/bash -o pipefail -c" in runner
    assert 'exit "${test_status}"' in runner
    assert 'exit "${exit_code}"' in runner
    assert 'VLLM_CI_ALLOW_NO_TESTS:-0}" == "1"' in runner
    assert template.count('{% if step.grade and step.grade == "Blocking" %}') == 2
    assert template.count("soft_fail: true") == 2


def test_qwen_image_diffusers_accuracy_is_one_nonblocking_nightly_job() -> None:
    step = _find_step("Qwen-Image Diffusers Accuracy", AMD_NIGHTLY_PIPELINE)
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 180
    commands = "\n".join(step["commands"])
    assert "full_model and diffusion and rocm and MI325 and cards_1" in commands
    assert "--run-level full_model" in commands
    assert "--collect-only" in commands
    nodes = (
        "tests/e2e/accuracy/test_qwen_image.py::test_qwen_image_matches_diffusers",
        "tests/e2e/accuracy/test_qwen_image.py::test_qwen_image_matches_diffusers_pixelwise",
    )
    pytest_command = commands[commands.index("pytest -s -v") :]
    offsets = [pytest_command.index(node) for node in nodes]
    assert offsets == sorted(offsets)
    assert all(commands.count(f'"{node}"') == 2 for node in nodes)
    assert "Expected exactly two Qwen-Image accuracy nodes" in commands
    assert "exit 5" in commands
    assert "pytest.xml" in commands
    assert step["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
    assert "tests/e2e/accuracy/test_qwen_image.py" in commands
    assert "test_diffusers_backend_similarity.py" not in commands
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "Qwen-Image Diffusers Accuracy" not in text


WAN22_SIMILARITY = "tests/e2e/accuracy/wan22_i2v/test_wan22_i2v_video_similarity.py"
WAN22_2GPU_NODES = (
    f"{WAN22_SIMILARITY}::test_wan22_i2v_diffusers_offline_generates_video",
    f"{WAN22_SIMILARITY}::test_wan22_i2v_online_serving_generates_video",
    f"{WAN22_SIMILARITY}::test_wan22_i2v_serving_matches_diffusers_video_similarity",
)


def test_wan22_accuracy_has_one_nonblocking_comparison_job() -> None:
    two = _find_step("Wan2.2 I2V Accuracy 2 GPU", AMD_NIGHTLY_PIPELINE)
    group = _find_group(":full_moon: Wan2.2 Accuracy MI300 Nightly", AMD_NIGHTLY_PIPELINE)
    assert [step["label"] for step in group["steps"]] == ["Wan2.2 I2V Accuracy 2 GPU"]
    assert two["agent_pool"] == "mi300_2"
    two_commands = "\n".join(two["commands"])
    assert two["grade"] == "NonBlocking"
    assert two["timeout_in_minutes"] == 180
    assert two["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
    two_marker = "full_model and diffusion and rocm and MI325 and (cards_1 or cards_2)"
    assert two_marker in two_commands
    assert 'and cards_2"' not in two_commands
    pytest_command = two_commands[two_commands.index("pytest -s -v") :]
    offsets = [pytest_command.index(node) for node in WAN22_2GPU_NODES]
    assert offsets == sorted(offsets)
    assert two_commands.count(WAN22_2GPU_NODES[0]) == 2
    assert "tests/e2e/accuracy/wan22_i2v/result" in two_commands
    assert "test_diffusers_backend_similarity.py" not in two_commands
    assert "--collect-only" in two_commands
    assert "Expected exactly three Wan2.2 accuracy nodes" in two_commands
    assert "exit 5" in two_commands
    similarity = Path(WAN22_SIMILARITY).read_text(encoding="utf-8")
    assert "require_similarity_artifacts" in similarity
    assert "similarity.json" in similarity
    assert "Missing offline artifacts" not in similarity
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "Wan2.2 I2V Accuracy" not in text


def test_wan22_similarity_fails_when_offline_baseline_is_removed(tmp_path: Path) -> None:
    from tests.e2e.accuracy.wan22_i2v.prerequisite import require_similarity_artifacts

    online = tmp_path / "online.mp4"
    online.write_bytes(b"online")
    with pytest.raises(AssertionError, match="offline.mp4"):
        require_similarity_artifacts(
            [
                online,
                tmp_path / "offline.mp4",
                tmp_path / "offline_metadata.json",
            ]
        )


def test_wan22_similarity_accepts_complete_artifacts(tmp_path: Path) -> None:
    from tests.e2e.accuracy.wan22_i2v.prerequisite import require_similarity_artifacts

    paths = []
    for name in ("online.mp4", "offline.mp4", "offline_metadata.json"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths.append(path)
    require_similarity_artifacts(paths)


def test_hunyuan_image3_lora_accuracy_is_one_nonblocking_nightly_job() -> None:
    step = _find_step("HunyuanImage3 LoRA Accuracy", AMD_NIGHTLY_PIPELINE)
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_4"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 180
    commands = "\n".join(step["commands"])
    assert "full_model and diffusion and rocm and MI325 and cards_4" in commands
    assert "--run-level full_model" in commands
    assert "--collect-only" in commands
    assert "exit 5" in commands
    assert "pytest.xml" in commands
    assert "test_hunyuan_image3_lora.py::test_hunyuan_image3_dit_lora_generation" in commands
    assert "Expected exactly one HunyuanImage3 LoRA accuracy node" in commands
    assert step["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
    assert step["env"]["HUNYUAN_IMAGE3_DEVICES"] == "0,1,2,3"
    assert "test_hunyuan_image3_lora.py" in commands
    assert "test_hunyuan_image3_pixel_accuracy.py" not in commands
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "HunyuanImage3 LoRA Accuracy" not in text


def test_hunyuanvideo15_accuracy_is_one_nonblocking_nightly_job() -> None:
    step = _find_step("HunyuanVideo-1.5 Video Similarity", AMD_NIGHTLY_PIPELINE)
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 180
    commands = "\n".join(step["commands"])
    assert "full_model and diffusion and rocm and MI325 and cards_1" in commands
    assert "--run-level full_model" in commands
    assert "--collect-only" in commands
    assert "exit 5" in commands
    assert "pytest.xml" in commands
    assert step["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
    assert "hunyuanvideo15_t2v" in commands
    assert "hunyuanvideo15_i2v" in commands
    assert "test_hunyuanvideo15_t2v.py" not in commands
    expected_nodes = (
        "test_hunyuanvideo15_t2v_diffusers_offline_generates_video",
        "test_hunyuanvideo15_t2v_online_serving_generates_video",
        "test_hunyuanvideo15_t2v_serving_matches_offline_video_similarity",
        "test_hunyuanvideo15_i2v_diffusers_offline_generates_video",
        "test_hunyuanvideo15_i2v_online_serving_generates_video",
        "test_hunyuanvideo15_i2v_serving_matches_offline_video_similarity",
    )
    for node in expected_nodes:
        assert node in commands
    assert "Expected exactly $${#nodes[@]} HunyuanVideo-1.5 accuracy nodes" in commands
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "HunyuanVideo-1.5 Video Similarity" not in text


def test_minimax_h3_accuracy_has_two_nonblocking_nightly_jobs() -> None:
    timeline = _find_step("MiniMax-H3 Timeline Guide Accuracy", AMD_NIGHTLY_PIPELINE)
    i2va = _find_step("MiniMax-H3 I2VA Ref2VA Accuracy", AMD_NIGHTLY_PIPELINE)
    assert timeline["agent_pool"] == "mi300_2"
    assert i2va["agent_pool"] == "mi300_4"
    pairs = (
        (
            timeline,
            "full_model and diffusion and rocm and MI325 and cards_2",
            "tests/e2e/accuracy/minimax_h3/test_minimax_h3_timeline_guides.py",
        ),
        (
            i2va,
            "full_model and diffusion and rocm and MI325 and cards_4",
            "tests/e2e/accuracy/minimax_h3/test_minimax_h3_i2va_ref2va_similarity.py",
        ),
    )
    for step, marker, path in pairs:
        assert step["grade"] == "NonBlocking"
        assert step["timeout_in_minutes"] == 180
        assert step["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
        commands = "\n".join(step["commands"])
        assert marker in commands
        assert path in commands
        assert "--collect-only" in commands
        assert "exit 5" in commands
    timeline_commands = "\n".join(timeline["commands"])
    assert "test_minimax_h3_timeline_guides_condition_frames_and_audio" in timeline_commands
    assert "Expected exactly one MiniMax-H3 timeline accuracy node" in timeline_commands
    i2va_commands = "\n".join(i2va["commands"])
    assert "test_minimax_h3_i2va_matches_official_reference" in i2va_commands
    assert "test_minimax_h3_ref2va_matches_official_reference" in i2va_commands
    assert "Expected exactly $${#nodes[@]} MiniMax-H3 I2VA/Ref2VA nodes" in i2va_commands
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "MiniMax-H3 Timeline Guide Accuracy" not in text
        assert "MiniMax-H3 I2VA Ref2VA Accuracy" not in text


def test_voxtral_tts_perf_is_one_nonblocking_nightly_job() -> None:
    step = _find_step("Voxtral TTS Perf", AMD_NIGHTLY_PIPELINE)
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_1"
    assert step["depends_on"] == "amd-build"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 180
    assert step["source_file_dependencies"] == "tts_voxtral_tts_perf"
    assert step["env"]["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"
    assert "VLLM_ROCM_USE_AITER" not in step["env"]
    commands = "\n".join(step["commands"])
    marker = "full_model and tts and rocm and MI325 and cards_1"
    assert marker in commands
    assert "--run-level full_model" in commands
    assert "--collect-only" in commands
    assert "exit 5" in commands
    assert "pytest.xml" in commands
    assert "tests/dfx/perf/tests/test_tts.json" in commands
    assert "Expected exactly $${#nodes[@]} Voxtral TTS perf nodes" in commands
    assert commands.count("--shard-id=0 --num-shards=1") == 2
    assert "Expected exactly three executed Voxtral TTS perf tests" in commands
    assert "skipped != 0" in commands
    assert "Expected exactly 3 Voxtral TTS result JSON files" in commands
    assert "result_test_voxtral_tts_*.json" in commands
    assert 'rm -rf "$$ROCM_VOXTRAL_TTS_PERF_ARTIFACT_DIR"' in commands
    assert "python3 -c 'import sys, xml.etree.ElementTree as ET" in commands
    nodes = (
        "test_voxtral_tts-default_voice_latency_seed-tts-text",
        "test_voxtral_tts-default_voice_throughput_seed-tts-text",
        "test_voxtral_tts-default_voice_stress_seed-tts-text",
    )
    for node in nodes:
        assert commands.count(node) == 2
    assert "test_qwen3_tts" not in commands
    assert "cards_2" not in commands
    nightly = AMD_NIGHTLY_PIPELINE.read_text(encoding="utf-8")
    assert nightly.count('label: "Voxtral TTS Perf"') == 1
    config = json.loads(Path("tests/dfx/perf/tests/test_tts.json").read_text(encoding="utf-8"))
    voxtral = next(case for case in config if case["test_name"] == "test_voxtral_tts")
    res = voxtral["mark"][0]["hardware_marks"]["res"]
    assert res["rocm"] == "MI325"
    assert res["cuda"] == ["H100", "B200"]
    assert voxtral["mark"][0]["hardware_marks"]["num_cards"] == 1
    for case in config:
        for phase in case["benchmark_params"]:
            baseline = phase.get("baseline")
            if isinstance(baseline, dict):
                assert "MI325" not in baseline
    dependencies = yaml.safe_load(
        Path(".buildkite/common/ci_source_file_dependencies.yml").read_text(encoding="utf-8")
    )["source_file_dependencies"]["tts_voxtral_tts_perf"]
    assert "tests/dfx/perf/scripts/run_benchmark.py" in dependencies
    assert "tests/dfx/perf/tests/test_tts.json" in dependencies
    assert "vllm_omni/deploy/voxtral_tts.yaml" in dependencies
    assert "vllm_omni/entrypoints/openai/serving_speech.py" in dependencies
    assert "vllm_omni/entrypoints/openai/tts_adapters/" in dependencies
    assert "vllm_omni/model_executor/models/voxtral_tts/" in dependencies
    assert "vllm_omni/model_executor/stage_input_processors/voxtral_tts.py" in dependencies
    assert "vllm_omni/transformers_utils/configs/voxtral_tts.py" in dependencies
    assert "vllm_omni/transformers_utils/parsers/voxtral_tts.py" in dependencies
    assert "vllm_omni/engine/stage_runtime.py" in dependencies
    assert "vllm_omni/distributed/omni_connectors/" in dependencies
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "Voxtral TTS Perf" not in text


def test_qwen3_tts_perf_is_one_nonblocking_nightly_job() -> None:
    step = _find_step("Qwen3-TTS Perf", AMD_NIGHTLY_PIPELINE)
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_2"
    assert step["depends_on"] == "amd-build"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 180
    assert step["source_file_dependencies"] == "tts_qwen3_tts_perf"
    assert step["env"]["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"
    assert "VLLM_ROCM_USE_AITER" not in step["env"]
    assert step["artifact_paths"] == ["artifacts/rocm-qwen3-tts-perf/**/*"]
    commands = "\n".join(step["commands"])
    marker = "full_model and tts and rocm and MI325 and cards_2"
    assert marker in commands
    assert "--run-level full_model" in commands
    assert "--collect-only" in commands
    assert "exit 5" in commands
    assert "pytest.xml" in commands
    assert "tests/dfx/perf/tests/test_tts.json" in commands
    assert "Expected exactly $${#nodes[@]} Qwen3-TTS perf nodes" in commands
    assert commands.count("--shard-id=0 --num-shards=1") == 2
    assert "Expected exactly eight executed Qwen3-TTS perf tests" in commands
    assert "skipped != 0" in commands
    assert "Expected exactly 14 Qwen3-TTS result JSON files" in commands
    assert '"test_qwen3_tts_base": 4' in commands
    assert '"test_qwen3_tts_customvoice": 5' in commands
    assert '"test_qwen3_tts_voicedesign": 5' in commands
    assert "Unreadable Qwen3-TTS result JSON" in commands
    assert "result_test_qwen3_tts_*.json" in commands
    assert 'rm -rf "$$ROCM_QWEN3_TTS_PERF_ARTIFACT_DIR"' in commands
    assert "python3 -c 'import sys, xml.etree.ElementTree as ET" in commands
    nodes = (
        "test_qwen3_tts_base-voice_clone_latency_seed-tts",
        "test_qwen3_tts_base-voice_clone_throughput_seed-tts",
        "test_qwen3_tts_customvoice-default_voice_latency_seed-tts-text",
        "test_qwen3_tts_customvoice-default_voice_throughput_seed-tts-text",
        "test_qwen3_tts_customvoice-default_voice_stress_seed-tts-text",
        "test_qwen3_tts_voicedesign-voice_design_latency_seed-tts-design",
        "test_qwen3_tts_voicedesign-voice_design_throughput_seed-tts-design",
        "test_qwen3_tts_voicedesign-voice_design_stress_seed-tts-design",
    )
    for node in nodes:
        assert commands.count(node) == 2
    assert "test_voxtral_tts" not in commands
    assert "cards_1" not in commands
    assert "quality" not in commands
    nightly = AMD_NIGHTLY_PIPELINE.read_text(encoding="utf-8")
    assert nightly.count('label: "Qwen3-TTS Perf"') == 1
    config = json.loads(Path("tests/dfx/perf/tests/test_tts.json").read_text(encoding="utf-8"))
    selected = {
        "test_qwen3_tts_base",
        "test_qwen3_tts_customvoice",
        "test_qwen3_tts_voicedesign",
    }
    seen: set[str] = set()
    for case in config:
        name = case["test_name"]
        res = case["mark"][0]["hardware_marks"]["res"]
        if name in selected:
            seen.add(name)
            assert res["rocm"] == "MI325"
            assert case["mark"][0]["hardware_marks"]["num_cards"] == 2
        for phase in case["benchmark_params"]:
            if phase.get("eval_phase") == "quality":
                assert phase.get("enabled") is False
            baseline = phase.get("baseline")
            if isinstance(baseline, dict):
                assert "MI325" not in baseline
    assert seen == selected
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "Qwen3-TTS Perf" not in text


def test_minicpmo_accuracy_is_one_nonblocking_nightly_job() -> None:
    step = _find_step("MiniCPM-o 4.5 Accuracy", AMD_NIGHTLY_PIPELINE)
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 180
    commands = "\n".join(step["commands"])
    assert "full_model and omni and rocm and MI325 and cards_1" in commands
    assert "--run-level full_model" in commands
    assert "--collect-only" in commands
    assert "exit 5" in commands
    assert "pytest.xml" in commands
    assert step["env"]["SEED_TTS_WER_EVAL"] == "1"
    assert step["env"]["SEED_TTS_SIM_EVAL"] == "1"
    assert "SEED_TTS_EVAL_DEVICE" not in commands
    assert "SEED_TTS_UTMOS_EVAL" not in commands
    assert "Expected exactly $${#nodes[@]} MiniCPM-o 4.5 accuracy nodes" in commands
    for node in (
        "test_minicpmo_4_5_daily_omni_accuracy_bench",
        "test_minicpmo_4_5_seed_tts_wer_bench",
    ):
        assert node in commands
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "MiniCPM-o 4.5 Accuracy" not in text


def test_qwen_image_21_accuracy_is_one_nonblocking_nightly_job() -> None:
    step = _find_step("Qwen-Image 2.1 Accuracy", AMD_NIGHTLY_PIPELINE)
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 180
    commands = "\n".join(step["commands"])
    assert "full_model and diffusion and rocm and MI325 and cards_1" in commands
    assert "--run-level full_model" in commands
    assert "--collect-only" in commands
    assert "exit 5" in commands
    assert "pytest.xml" in commands
    assert step["env"]["QWEN_IMAGE_21_CALIBRATE"] == "0"
    assert step["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
    assert "setup_reference.sh" in commands
    assert "test_qwen_image_21_matches_reference[it2i]" in commands
    assert "test_qwen_image_21_matches_reference[t2i]" in commands
    assert "Expected exactly $${#nodes[@]} Qwen-Image 2.1 accuracy nodes" in commands
    accuracy_test = Path("tests/e2e/accuracy/qwen_image_21/test_qwen_image_21.py").read_text(encoding="utf-8")
    assert 'os.environ.get("BUILDKITE_COMMIT")' in accuracy_test
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "Qwen-Image 2.1 Accuracy" not in text


def test_minicpmo_4_5_perf_is_one_nonblocking_nightly_job() -> None:
    step = _find_step("MiniCPM-o 4.5 Perf", AMD_NIGHTLY_PIPELINE)
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_1"
    assert step["depends_on"] == "amd-build"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 180
    assert step["source_file_dependencies"] == "omni_minicpmo_4_5_perf"
    assert step["artifact_paths"] == ["artifacts/rocm-minicpmo-4-5-perf/**/*"]
    commands = "\n".join(step["commands"])
    assert "full_model and omni and rocm and MI325 and cards_1" in commands
    assert " and cuda" not in commands
    assert "--run-level full_model" in commands
    assert "--collect-only" in commands
    assert "exit 5" in commands
    assert "pytest.xml" in commands
    assert "--assert-baseline" not in commands
    assert "test_minicpmo_4_5.json" in commands
    assert "test_performance_benchmark[test_minicpmo_4_5-seed-tts]" in commands
    assert "test_minicpmo_4_5_a2" in commands
    assert "Expected exactly one MiniCPM-o 4.5 perf node" in commands
    assert commands.count("--shard-id=0 --num-shards=1") == 2
    assert "Expected exactly one executed MiniCPM-o 4.5 perf test" in commands
    assert "skipped != 0" in commands
    assert "Expected exactly 5 MiniCPM-o 4.5 result JSON files" in commands
    assert "Unreadable MiniCPM-o 4.5 result JSON" in commands
    assert "A2 MiniCPM-o result must not be produced" in commands
    assert "result_test_minicpmo_4_5_*.json" in commands
    assert 'rm -rf "$$ROCM_MINICPMO_PERF_ARTIFACT_DIR"' in commands
    assert "python3 -c 'import sys, xml.etree.ElementTree as ET" in commands
    config = json.loads(Path("tests/dfx/perf/tests/test_minicpmo_4_5.json").read_text(encoding="utf-8"))
    selected = next(case for case in config if case["test_name"] == "test_minicpmo_4_5")
    selected_res = selected["mark"][0]["hardware_marks"]["res"]
    assert selected_res["rocm"] == "MI325"
    assert selected_res["cuda"] == ["H100", "B200"]
    assert selected_res["npu"] == "A3"
    assert selected["mark"][0]["hardware_marks"]["num_cards"] == 1
    assert len(selected["benchmark_params"][0]["max_concurrency"]) == 5
    assert "MI325" not in selected["benchmark_params"][0]["baseline"]
    a2 = next(case for case in config if case["test_name"] == "test_minicpmo_4_5_a2")
    assert "rocm" not in a2["mark"][0]["hardware_marks"]["res"]
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "MiniCPM-o 4.5 Perf" not in text


def test_omni_duplex_eval_is_one_nonblocking_nightly_job() -> None:
    step = _find_step("MiniCPM-o Omni-DuplexEval CI Guard", AMD_NIGHTLY_PIPELINE)
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_2"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 240
    commands = "\n".join(step["commands"])
    assert "full_model and omni and rocm and MI325 and cards_2" in commands
    assert "--run-level full_model" in commands
    assert "--collect-only" in commands
    assert "exit 5" in commands
    assert step["artifact_paths"] == [
        "artifacts/rocm-omni-duplex-eval/collection.txt",
        "tests/dfx/perf/results/*.json",
        "tests/e2e/accuracy/omni_duplex_eval/results/*.json",
    ]
    assert "$$ROCM_OMNI_DUPLEX_EVAL_ARTIFACT_DIR/collection.txt" in commands
    assert "pytest.xml" not in commands
    assert "pytest.log" not in commands
    assert "timeout --signal=INT --kill-after=2m 220m" in commands
    assert step["env"]["BENCHMARK_DIR"] == "tests/dfx/perf/results"
    assert step["env"]["ACC_BENCH_RESULT_DIR"] == "tests/e2e/accuracy/omni_duplex_eval/results"
    assert "test_minicpmo_4_5_omni_duplex_eval.json" in commands
    assert "Expected exactly one MiniCPM-o Omni-DuplexEval node" in commands
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "MiniCPM-o Omni-DuplexEval CI Guard" not in text


def test_qwen_image_backend_similarity_is_one_nonblocking_nightly_job() -> None:
    step = _find_step("Qwen-Image Diffusers Backend Similarity", AMD_NIGHTLY_PIPELINE)
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 180
    assert step["source_file_dependencies"] == "diffusion_qwen_image_accuracy"
    assert step["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
    assert step["artifact_paths"] == ["artifacts/rocm-qwen-image-backend-similarity/**/*"]
    commands = "\n".join(step["commands"])
    assert "full_model and diffusion and rocm and MI325 and cards_1" in commands
    assert '-k "2i_matches_diffusers"' in commands
    assert "test_diffusers_backend_t2i_matches_diffusers" in commands
    assert "Expected exactly one Qwen-Image backend similarity node" in commands
    assert "--collect-only" in commands
    assert "exit 5" in commands
    assert "pytest.xml" in commands
    assert "Expected exactly one executed Qwen-Image backend similarity test" in commands
    accuracy_test = Path("tests/e2e/accuracy/test_diffusers_backend_similarity.py").read_text(encoding="utf-8")
    assert "torch.version.hip is not None" in accuracy_test
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "Qwen-Image Diffusers Backend Similarity" not in text


def _assert_nightly_junit_guard(
    tmp_path: Path,
    *,
    pipeline_path: Path = AMD_NIGHTLY_PIPELINE,
    step_label: str,
    command_fragment: str,
    expected_message: str,
    artifact_env: str,
    job_id_prefix: str,
    pytest_status: int,
    tests: int,
    skipped: int,
    timeout_status: int | None,
    expected_status: int,
    extra_env: dict[str, str] | None = None,
) -> None:
    step = _find_step(step_label, pipeline_path)
    run_command = next(command for command in step["commands"] if command_fragment in command)
    assert expected_message in run_command

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    def write_executable(name: str, body: str) -> None:
        path = fake_bin / name
        path.write_text(f"#!/bin/bash\nset -euo pipefail\n{body}", encoding="utf-8")
        path.chmod(0o755)

    write_executable("rocminfo", "exit 0\n")
    write_executable(
        "python3",
        'if [[ "${1:-}" == "-" ]]; then cat >/dev/null; exit 0; fi\nexec "$REAL_PYTHON" "$@"\n',
    )
    write_executable("python", 'exec "$REAL_PYTHON" "$@"\n')
    write_executable(
        "timeout",
        'if [[ -n "${FAKE_TIMEOUT_STATUS:-}" ]]; then exit "$FAKE_TIMEOUT_STATUS"; fi\n'
        'while [[ "$1" != "pytest" ]]; do shift; done\nexec "$@"\n',
    )
    write_executable(
        "pytest",
        'xml_path=""\n'
        'for arg in "$@"; do\n'
        '  case "$arg" in --junitxml=*) xml_path="${arg#*=}" ;; esac\n'
        "done\n"
        'mkdir -p "$(dirname "$xml_path")"\n'
        'printf \'<testsuite tests="%s" skipped="%s" failures="0" errors="0"/>\\n\' \\\n'
        '  "$FAKE_JUNIT_TESTS" "$FAKE_JUNIT_SKIPPED" > "$xml_path"\n'
        'echo "fake pytest status=$FAKE_PYTEST_STATUS"\n'
        'exit "$FAKE_PYTEST_STATUS"\n',
    )

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "REAL_PYTHON": sys.executable,
            "VLLM_CI_DOCKER_DISABLED": "1",
            "VLLM_CI_REQUIRE_PERSISTENT_HF_CACHE": "0",
            "BUILDKITE_JOB_ID": f"{job_id_prefix}-{tmp_path.name}",
            "HF_HOME": str(tmp_path / "hf"),
            "FAKE_PYTEST_STATUS": str(pytest_status),
            "FAKE_JUNIT_TESTS": str(tests),
            "FAKE_JUNIT_SKIPPED": str(skipped),
            "FAKE_TIMEOUT_STATUS": "" if timeout_status is None else str(timeout_status),
            "TEST_COMMANDS": run_command.replace("$$", "$"),
        }
    )
    env[artifact_env] = str(artifact_dir)
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        ["bash", str(AMD_RUNNER)],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected_status, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("pytest_status", "tests", "skipped", "timeout_status", "result_files", "expected_status"),
    [
        (0, 3, 0, None, 3, 0),
        (0, 3, 1, None, 3, 5),
        (0, 0, 0, None, 3, 5),
        (1, 3, 0, None, 3, 1),
        (0, 3, 0, 124, 3, 124),
        (0, 3, 0, None, 2, 5),
    ],
    ids=("pass", "skipped", "zero-tests", "pytest-failure", "timeout", "missing-result"),
)
def test_voxtral_tts_perf_executes_evidence_guards_before_exit(
    tmp_path: Path,
    pytest_status: int,
    tests: int,
    skipped: int,
    timeout_status: int | None,
    result_files: int,
    expected_status: int,
) -> None:
    benchmark_dir = tmp_path / "benchmark-results"
    benchmark_dir.mkdir()
    for index in range(result_files):
        (benchmark_dir / f"result_test_voxtral_tts_{index}.json").write_text("{}\n", encoding="utf-8")

    _assert_nightly_junit_guard(
        tmp_path,
        step_label="Voxtral TTS Perf",
        command_fragment='pytest -s -v -ra "$${nodes[@]}"',
        expected_message="Expected exactly three executed Voxtral TTS perf tests",
        artifact_env="ROCM_VOXTRAL_TTS_PERF_ARTIFACT_DIR",
        job_id_prefix="voxtral-tts-perf-guard",
        pytest_status=pytest_status,
        tests=tests,
        skipped=skipped,
        timeout_status=timeout_status,
        expected_status=expected_status,
        extra_env={"BENCHMARK_DIR": str(benchmark_dir)},
    )


@pytest.mark.parametrize(
    (
        "pytest_status",
        "tests",
        "skipped",
        "timeout_status",
        "base_files",
        "customvoice_files",
        "voicedesign_files",
        "valid_json",
        "expected_status",
    ),
    [
        (0, 8, 0, None, 4, 5, 5, True, 0),
        (0, 8, 1, None, 4, 5, 5, True, 5),
        (0, 0, 0, None, 4, 5, 5, True, 5),
        (1, 8, 0, None, 4, 5, 5, True, 1),
        (0, 8, 0, 124, 4, 5, 5, True, 124),
        (0, 8, 0, None, 3, 5, 5, True, 5),
        (0, 8, 0, None, 5, 4, 5, True, 5),
        (0, 8, 0, None, 4, 5, 5, False, 5),
    ],
    ids=(
        "pass",
        "skipped",
        "zero-tests",
        "pytest-failure",
        "timeout",
        "missing-result",
        "wrong-result-split",
        "invalid-json",
    ),
)
def test_qwen3_tts_perf_executes_evidence_guards_before_exit(
    tmp_path: Path,
    pytest_status: int,
    tests: int,
    skipped: int,
    timeout_status: int | None,
    base_files: int,
    customvoice_files: int,
    voicedesign_files: int,
    valid_json: bool,
    expected_status: int,
) -> None:
    benchmark_dir = tmp_path / "benchmark-results"
    benchmark_dir.mkdir()
    counts = {
        "test_qwen3_tts_base": base_files,
        "test_qwen3_tts_customvoice": customvoice_files,
        "test_qwen3_tts_voicedesign": voicedesign_files,
    }
    result_paths: list[Path] = []
    for test_name, count in counts.items():
        for index in range(count):
            result_path = benchmark_dir / f"result_{test_name}_{index}.json"
            result_path.write_text("{}\n", encoding="utf-8")
            result_paths.append(result_path)
    if not valid_json:
        result_paths[0].write_text("not-json\n", encoding="utf-8")

    _assert_nightly_junit_guard(
        tmp_path,
        step_label="Qwen3-TTS Perf",
        command_fragment='pytest -s -v -ra "$${nodes[@]}"',
        expected_message="Expected exactly eight executed Qwen3-TTS perf tests",
        artifact_env="ROCM_QWEN3_TTS_PERF_ARTIFACT_DIR",
        job_id_prefix="qwen3-tts-perf-guard",
        pytest_status=pytest_status,
        tests=tests,
        skipped=skipped,
        timeout_status=timeout_status,
        expected_status=expected_status,
        extra_env={"BENCHMARK_DIR": str(benchmark_dir)},
    )


@pytest.mark.parametrize(
    (
        "pytest_status",
        "tests",
        "skipped",
        "timeout_status",
        "result_files",
        "valid_json",
        "include_a2",
        "expected_status",
    ),
    [
        (0, 1, 0, None, 5, True, False, 0),
        (0, 1, 1, None, 5, True, False, 5),
        (0, 0, 0, None, 5, True, False, 5),
        (1, 1, 0, None, 5, True, False, 1),
        (0, 1, 0, 124, 5, True, False, 124),
        (0, 1, 0, None, 4, True, False, 5),
        (0, 1, 0, None, 5, False, False, 5),
        (0, 1, 0, None, 4, True, True, 5),
    ],
    ids=("pass", "skipped", "zero-tests", "pytest-failure", "timeout", "missing-result", "invalid-json", "a2-result"),
)
def test_minicpmo_4_5_perf_executes_evidence_guards_before_exit(
    tmp_path: Path,
    pytest_status: int,
    tests: int,
    skipped: int,
    timeout_status: int | None,
    result_files: int,
    valid_json: bool,
    include_a2: bool,
    expected_status: int,
) -> None:
    benchmark_dir = tmp_path / "benchmark-results"
    benchmark_dir.mkdir()
    result_paths: list[Path] = []
    for index in range(result_files):
        result_path = benchmark_dir / f"result_test_minicpmo_4_5_{index}.json"
        result_path.write_text("{}\n", encoding="utf-8")
        result_paths.append(result_path)
    if include_a2:
        a2_result = benchmark_dir / "result_test_minicpmo_4_5_a2_0.json"
        a2_result.write_text("{}\n", encoding="utf-8")
        result_paths.append(a2_result)
    if not valid_json:
        result_paths[0].write_text("not-json\n", encoding="utf-8")

    _assert_nightly_junit_guard(
        tmp_path,
        step_label="MiniCPM-o 4.5 Perf",
        command_fragment='pytest -s -v -ra "$$node"',
        expected_message="Expected exactly one executed MiniCPM-o 4.5 perf test",
        artifact_env="ROCM_MINICPMO_PERF_ARTIFACT_DIR",
        job_id_prefix="minicpmo-4-5-perf-guard",
        pytest_status=pytest_status,
        tests=tests,
        skipped=skipped,
        timeout_status=timeout_status,
        expected_status=expected_status,
        extra_env={"BENCHMARK_DIR": str(benchmark_dir)},
    )


@pytest.mark.parametrize(
    ("pytest_status", "tests", "skipped", "timeout_status", "expected_status"),
    [
        (0, 1, 0, None, 0),
        (0, 1, 1, None, 5),
        (0, 0, 0, None, 5),
        (1, 1, 0, None, 1),
        (0, 1, 0, 124, 124),
    ],
    ids=("pass", "skipped", "zero-tests", "pytest-failure", "timeout"),
)
def test_qwen_image_backend_similarity_executes_junit_guard_before_exit(
    tmp_path: Path,
    pytest_status: int,
    tests: int,
    skipped: int,
    timeout_status: int | None,
    expected_status: int,
) -> None:
    _assert_nightly_junit_guard(
        tmp_path,
        step_label="Qwen-Image Diffusers Backend Similarity",
        command_fragment="pytest -s -v -ra",
        expected_message="Expected exactly one executed Qwen-Image backend similarity test",
        artifact_env="ROCM_QWEN_IMAGE_BACKEND_ARTIFACT_DIR",
        job_id_prefix="qwen-image-guard",
        pytest_status=pytest_status,
        tests=tests,
        skipped=skipped,
        timeout_status=timeout_status,
        expected_status=expected_status,
    )


def test_minicpmo_duplex_function_is_one_nonblocking_nightly_job() -> None:
    step = _find_step("MiniCPM-o 4.5 Duplex Function", AMD_NIGHTLY_PIPELINE)
    assert step["grade"] == "NonBlocking"
    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 180
    commands = "\n".join(step["commands"])
    marker = "full_model and omni and rocm and MI325 and cards_1"
    assert marker in commands
    assert "--run-level full_model" in commands
    assert "--collect-only" in commands
    assert "exit 5" in commands
    assert "pytest.xml" in commands
    nodes = (
        "test_duplex_admission_and_expiry_reaper",
        "test_duplex_soft_interrupt",
        "test_duplex_server_vad_hard_interrupt",
        "test_window_continuous_input[basic-False-three-stage-single-gpu]",
        "test_window_continuous_input[context-False-three-stage-single-gpu]",
        "test_window_continuous_input[context-True-three-stage-single-gpu]",
    )
    for node in nodes:
        assert commands.count(node) == 2
    assert commands.count("--shard-id=0") == 2
    assert commands.count("--num-shards=1") == 2
    assert "Expected exactly $${#nodes[@]} MiniCPM-o duplex nodes" in commands
    assert "Expected exactly six executed MiniCPM-o duplex tests" in commands
    nightly = AMD_NIGHTLY_PIPELINE.read_text(encoding="utf-8")
    assert nightly.count('label: "MiniCPM-o 4.5 Duplex Function"') == 1
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "MiniCPM-o 4.5 Duplex Function" not in text


@pytest.mark.parametrize(
    ("pytest_status", "tests", "skipped", "timeout_status", "expected_status"),
    [
        (0, 6, 0, None, 0),
        (0, 6, 1, None, 5),
        (0, 0, 0, None, 5),
        (1, 6, 0, None, 1),
        (0, 6, 0, 124, 124),
    ],
    ids=("pass", "skipped", "zero-tests", "pytest-failure", "timeout"),
)
def test_minicpmo_duplex_executes_junit_guard_before_exit(
    tmp_path: Path,
    pytest_status: int,
    tests: int,
    skipped: int,
    timeout_status: int | None,
    expected_status: int,
) -> None:
    _assert_nightly_junit_guard(
        tmp_path,
        step_label="MiniCPM-o 4.5 Duplex Function",
        command_fragment='pytest -s -v "$${nodes[@]}"',
        expected_message="Expected exactly six executed MiniCPM-o duplex tests",
        artifact_env="ROCM_MINICPMO_DUPLEX_ARTIFACT_DIR",
        job_id_prefix="minicpmo-duplex-guard",
        pytest_status=pytest_status,
        tests=tests,
        skipped=skipped,
        timeout_status=timeout_status,
        expected_status=expected_status,
    )


def test_qwen_image_expansion_has_separate_nonblocking_card_jobs() -> None:
    one = _find_step("Qwen-Image Expansion 1 GPU", AMD_NIGHTLY_PIPELINE)
    two = _find_step("Qwen-Image Expansion 2 GPU", AMD_NIGHTLY_PIPELINE)
    cases = (
        (
            one,
            "mi300_1",
            "full_model and diffusion and rocm and MI325 and cards_1",
            "artifacts/rocm-qwen-image-expansion-1gpu/**/*",
            (
                "cpu_offload",
                "step_execution",
                "cache_tea_cache",
                "layerwise_offload",
            ),
            "Expected exactly four executed Qwen-Image expansion 1-GPU tests",
        ),
        (
            two,
            "mi300_2",
            "full_model and diffusion and rocm and MI325 and cards_2",
            "artifacts/rocm-qwen-image-expansion-2gpu/**/*",
            (
                "ulysses_2",
                "ring_2",
                "cfg_parallel_2",
                "vae_patch_parallel_2",
                "parallel_hsdp",
            ),
            "Expected exactly five executed Qwen-Image expansion 2-GPU tests",
        ),
    )
    for step, agent_pool, marker, artifact_path, nodes, expected_message in cases:
        assert step["grade"] == "NonBlocking"
        assert step["agent_pool"] == agent_pool
        assert step["mirror_hardwares"] == ["amdproduction"]
        assert step["timeout_in_minutes"] == 180
        assert step["source_file_dependencies"] == "diffusion_qwen_image_function"
        assert step["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
        assert step["artifact_paths"] == [artifact_path]
        commands = "\n".join(step["commands"])
        assert marker in commands
        assert "--run-level full_model" in commands
        assert "--collect-only" in commands
        assert "exit 5" in commands
        assert "pytest.xml" in commands
        for node in nodes:
            assert commands.count(f"test_qwen_image[{node}]") == 2
        assert commands.count("--shard-id=0") == 2
        assert commands.count("--num-shards=1") == 2
        assert "Expected exactly $${#nodes[@]} Qwen-Image expansion" in commands
        assert expected_message in commands
    nightly = AMD_NIGHTLY_PIPELINE.read_text(encoding="utf-8")
    assert nightly.count('label: "Qwen-Image Expansion 1 GPU"') == 1
    assert nightly.count('label: "Qwen-Image Expansion 2 GPU"') == 1
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "Qwen-Image Expansion 1 GPU" not in text
        assert "Qwen-Image Expansion 2 GPU" not in text


@pytest.mark.parametrize(
    (
        "step_label",
        "expected_tests",
        "expected_message",
        "artifact_env",
        "job_id_prefix",
    ),
    [
        (
            "Qwen-Image Expansion 1 GPU",
            4,
            "Expected exactly four executed Qwen-Image expansion 1-GPU tests",
            "ROCM_QWEN_IMAGE_EXPANSION_1GPU_ARTIFACT_DIR",
            "qwen-image-expansion-1gpu-guard",
        ),
        (
            "Qwen-Image Expansion 2 GPU",
            5,
            "Expected exactly five executed Qwen-Image expansion 2-GPU tests",
            "ROCM_QWEN_IMAGE_EXPANSION_2GPU_ARTIFACT_DIR",
            "qwen-image-expansion-2gpu-guard",
        ),
    ],
    ids=("1gpu", "2gpu"),
)
@pytest.mark.parametrize(
    (
        "pytest_status",
        "tests_override",
        "skipped",
        "timeout_status",
        "expected_status",
    ),
    [
        (0, None, 0, None, 0),
        (0, None, 1, None, 5),
        (0, 0, 0, None, 5),
        (1, None, 0, None, 1),
        (0, None, 0, 124, 124),
    ],
    ids=("pass", "skipped", "zero-tests", "pytest-failure", "timeout"),
)
def test_qwen_image_expansion_executes_junit_guard_before_exit(
    tmp_path: Path,
    step_label: str,
    expected_tests: int,
    expected_message: str,
    artifact_env: str,
    job_id_prefix: str,
    pytest_status: int,
    tests_override: int | None,
    skipped: int,
    timeout_status: int | None,
    expected_status: int,
) -> None:
    _assert_nightly_junit_guard(
        tmp_path,
        step_label=step_label,
        command_fragment='pytest -s -v "$${nodes[@]}"',
        expected_message=expected_message,
        artifact_env=artifact_env,
        job_id_prefix=job_id_prefix,
        pytest_status=pytest_status,
        tests=expected_tests if tests_override is None else tests_override,
        skipped=skipped,
        timeout_status=timeout_status,
        expected_status=expected_status,
    )


_QWEN25_OMNI_MERGE_NODES = (
    "tests/e2e/online_serving/test_qwen2_5_omni_expansion.py::test_mix_to_text_audio_001[omni_server0]",
    "tests/e2e/online_serving/test_qwen2_5_omni_expansion.py::test_audio_in_video_boundaries[False-omni_server0]",
    "tests/e2e/online_serving/test_qwen2_5_omni_expansion.py::test_audio_in_video_boundaries[True-omni_server0]",
    "tests/e2e/online_serving/test_qwen2_5_omni_expansion.py::test_text_to_text_001[omni_server0]",
    "tests/e2e/online_serving/test_qwen2_5_omni_expansion.py::test_speaker_001[Chelsie-omni_server0]",
    "tests/e2e/online_serving/test_qwen2_5_omni_expansion.py::test_speaker_001[ETHAN-omni_server0]",
    "tests/e2e/offline_inference/test_qwen2_5_omni_expansion.py::test_mix_to_audio[omni_runner0-rocm]",
    "tests/e2e/offline_inference/test_qwen2_5_omni_expansion.py::test_text_to_text[omni_runner0-rocm]",
)


def test_qwen25_omni_merge_selects_eight_marked_nodes() -> None:
    """MERGE runs exactly the eight marked two-card ROCm cases."""
    step = _find_step("Omni Model Test Qwen2-5-Omni", AMD_MERGE_PIPELINE)
    command = "\n".join(step["commands"])

    assert step["agent_pool"] == "mi300_2"
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 180
    assert step["artifact_paths"] == ["artifacts/rocm-qwen25-omni-merge/**/*"]
    assert command.count("slow and rocm and MI325 and cards_2") == 2
    assert "tests/e2e/*/" not in command
    assert "timeout --signal=TERM --kill-after=2m 165m" in command
    assert "--shard-id=0 --num-shards=1" in command
    assert "PIPESTATUS[0]" in command
    assert "collection.log" in command
    assert "pytest.xml" in command
    assert "pytest.log" in command
    assert "pytest-summary.txt" in command
    assert "Expected exactly eight executed Qwen2.5-Omni MERGE tests" in command
    for node in _QWEN25_OMNI_MERGE_NODES:
        assert node in command
    assert command.count("test_qwen2_5_omni_expansion.py::") == len(_QWEN25_OMNI_MERGE_NODES)

    source = Path("tests/e2e/online_serving/test_qwen2_5_omni_expansion.py").read_text(encoding="utf-8")
    text_test_marks = source.split("def test_text_to_text_001", maxsplit=1)[0].rsplit("@pytest.mark.slow", maxsplit=1)[
        1
    ]
    assert '@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=2)' in text_test_marks


@pytest.mark.parametrize(
    ("pytest_status", "tests", "skipped", "timeout_status", "expected_status"),
    [
        (0, 8, 0, None, 0),
        (0, 8, 1, None, 5),
        (0, 0, 0, None, 5),
        (1, 8, 0, None, 1),
        (0, 8, 0, 124, 124),
    ],
    ids=("pass", "skipped", "zero-tests", "pytest-failure", "timeout"),
)
def test_qwen25_omni_merge_executes_junit_guard_before_exit(
    tmp_path: Path,
    pytest_status: int,
    tests: int,
    skipped: int,
    timeout_status: int | None,
    expected_status: int,
) -> None:
    _assert_nightly_junit_guard(
        tmp_path,
        pipeline_path=AMD_MERGE_PIPELINE,
        step_label="Omni Model Test Qwen2-5-Omni",
        command_fragment="pytest -s -v -ra",
        expected_message="Expected exactly eight executed Qwen2.5-Omni MERGE tests",
        artifact_env="ROCM_QWEN25_OMNI_MERGE_ARTIFACT_DIR",
        job_id_prefix="qwen25-omni-merge-guard",
        pytest_status=pytest_status,
        tests=tests,
        skipped=skipped,
        timeout_status=timeout_status,
        expected_status=expected_status,
    )


def test_wan22_function_has_separate_nonblocking_card_jobs() -> None:
    one = _find_step("Wan2.2 Function 1 GPU", AMD_NIGHTLY_PIPELINE)
    two = _find_step("Wan2.2 Function 2 GPU", AMD_NIGHTLY_PIPELINE)
    cases = (
        (
            one,
            "mi300_1",
            "full_model and diffusion and rocm and MI325 and cards_1",
            "artifacts/rocm-wan22-function-1gpu/**/*",
            (
                "t2v_cpu_offload",
                "i2v_cpu_offload",
                "ti2v_cpu_offload",
                "cuda_t2v_cache_dit",
                "cuda_i2v_cache_dit",
                "cuda_ti2v_cache_dit",
            ),
            "Expected exactly six executed Wan2.2 function 1-GPU tests",
        ),
        (
            two,
            "mi300_2",
            "full_model and diffusion and rocm and MI325 and cards_2",
            "artifacts/rocm-wan22-function-2gpu/**/*",
            (
                "cuda_t2v_cfg_parallel",
                "cuda_t2v_ulysses_sp",
                "cuda_t2v_tp_vae_patch",
                "cuda_t2v_hsdp",
                "cuda_t2v_ring_atten",
                "cuda_i2v_cfg_parallel",
                "cuda_i2v_ulysses_sp",
                "cuda_i2v_tp_vae_patch",
                "cuda_i2v_hsdp",
                "cuda_i2v_ring_atten",
                "cuda_ti2v_cfg_parallel",
                "cuda_ti2v_ulysses_sp",
                "cuda_ti2v_tp_vae_patch",
                "cuda_ti2v_hsdp",
                "cuda_ti2v_ring_atten",
            ),
            "Expected exactly fifteen executed Wan2.2 function 2-GPU tests",
        ),
    )
    for step, agent_pool, marker, artifact_path, nodes, expected_message in cases:
        assert step["grade"] == "NonBlocking"
        assert step["agent_pool"] == agent_pool
        assert step["mirror_hardwares"] == ["amdproduction"]
        assert step["timeout_in_minutes"] == 180
        assert step["source_file_dependencies"] == "diffusion_wan22_function"
        assert step["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
        assert step["artifact_paths"] == [artifact_path]
        commands = "\n".join(step["commands"])
        assert marker in commands
        assert "--run-level full_model" in commands
        assert "--collect-only" in commands
        assert "exit 5" in commands
        assert "pytest.xml" in commands
        for node in nodes:
            assert commands.count(f"test_wan22_diffusion_features[{node}]") == 2
        assert commands.count("--shard-id=0") == 2
        assert commands.count("--num-shards=1") == 2
        assert "Expected exactly $${#nodes[@]} Wan2.2 function" in commands
        assert expected_message in commands
    nightly = AMD_NIGHTLY_PIPELINE.read_text(encoding="utf-8")
    assert nightly.count('label: "Wan2.2 Function 1 GPU"') == 1
    assert nightly.count('label: "Wan2.2 Function 2 GPU"') == 1
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        assert "Wan2.2 Function 1 GPU" not in text
        assert "Wan2.2 Function 2 GPU" not in text


@pytest.mark.parametrize(
    (
        "step_label",
        "expected_tests",
        "expected_message",
        "job_id_prefix",
    ),
    [
        (
            "Wan2.2 Function 1 GPU",
            6,
            "Expected exactly six executed Wan2.2 function 1-GPU tests",
            "wan22-function-1gpu-guard",
        ),
        (
            "Wan2.2 Function 2 GPU",
            15,
            "Expected exactly fifteen executed Wan2.2 function 2-GPU tests",
            "wan22-function-2gpu-guard",
        ),
    ],
    ids=("1gpu", "2gpu"),
)
@pytest.mark.parametrize(
    (
        "pytest_status",
        "tests_override",
        "skipped",
        "timeout_status",
        "expected_status",
    ),
    [
        (0, None, 0, None, 0),
        (0, None, 1, None, 5),
        (0, 0, 0, None, 5),
        (1, None, 0, None, 1),
        (0, None, 0, 124, 124),
    ],
    ids=("pass", "skipped", "zero-tests", "pytest-failure", "timeout"),
)
def test_wan22_function_executes_junit_guard_before_exit(
    tmp_path: Path,
    step_label: str,
    expected_tests: int,
    expected_message: str,
    job_id_prefix: str,
    pytest_status: int,
    tests_override: int | None,
    skipped: int,
    timeout_status: int | None,
    expected_status: int,
) -> None:
    _assert_nightly_junit_guard(
        tmp_path,
        step_label=step_label,
        command_fragment='pytest -s -v "$${nodes[@]}"',
        expected_message=expected_message,
        artifact_env="ROCM_WAN22_FN_ARTIFACT_DIR",
        job_id_prefix=job_id_prefix,
        pytest_status=pytest_status,
        tests=expected_tests if tests_override is None else tests_override,
        skipped=skipped,
        timeout_status=timeout_status,
        expected_status=expected_status,
    )


def test_qwen3_omni_shm_leak_is_a_separate_nonblocking_merge_job() -> None:
    blocking_step = _find_step("Omni Model Test Qwen3-Omni", AMD_MERGE_PIPELINE)
    shm_path = "tests/e2e/online_serving/test_qwen3_omni_shm_leak.py"
    blocking_commands = "\n".join(blocking_step["commands"])

    assert blocking_step["agent_pool"] == "mi300_2"
    assert blocking_step["grade"] == "Blocking"
    assert "assert-pytest-collection.sh 17" in blocking_commands
    assert shm_path not in blocking_commands

    step = _find_step("Qwen3-Omni SHM Leak Test", AMD_MERGE_PIPELINE)
    commands = "\n".join(step["commands"])

    assert step["agent_pool"] == "mi300_2"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 50
    assert step["artifact_paths"] == ["artifacts/rocm-qwen3-omni-shm-leak/**/*"]
    assert commands.count(shm_path) == 2
    assert commands.count("--shard-id=0 --num-shards=1") == 2
    assert "assert-pytest-collection.sh 1" in commands
    assert "timeout --signal=TERM --kill-after=2m 45m" in commands
    assert "PIPESTATUS[0]" in commands
    assert "pytest.xml" in commands
    assert "pytest.log" in commands
    assert "pytest-summary.txt" in commands
    assert "Expected exactly one executed Qwen3-Omni SHM leak test" in commands


@pytest.mark.parametrize(
    ("pytest_status", "tests", "skipped", "timeout_status", "expected_status"),
    [
        (0, 1, 0, None, 0),
        (0, 1, 1, None, 5),
        (0, 0, 0, None, 5),
        (1, 1, 0, None, 1),
        (0, 1, 0, 124, 124),
    ],
    ids=("pass", "skipped", "zero-tests", "pytest-failure", "timeout"),
)
def test_qwen3_omni_shm_leak_executes_junit_guard_before_exit(
    tmp_path: Path,
    pytest_status: int,
    tests: int,
    skipped: int,
    timeout_status: int | None,
    expected_status: int,
) -> None:
    _assert_nightly_junit_guard(
        tmp_path,
        pipeline_path=AMD_MERGE_PIPELINE,
        step_label="Qwen3-Omni SHM Leak Test",
        command_fragment="pytest -s -v -ra",
        expected_message="Expected exactly one executed Qwen3-Omni SHM leak test",
        artifact_env="ROCM_QWEN3_OMNI_SHM_ARTIFACT_DIR",
        job_id_prefix="qwen3-omni-shm-guard",
        pytest_status=pytest_status,
        tests=tests,
        skipped=skipped,
        timeout_status=timeout_status,
        expected_status=expected_status,
    )


_QWEN3_OMNI_READY_NODES = (
    "tests/e2e/online_serving/test_qwen3_omni.py::test_mix_to_text_audio_001[default-rocm]",
    "tests/e2e/online_serving/test_qwen3_omni.py::test_text_to_text_001[default-rocm]",
    "tests/e2e/online_serving/test_qwen3_omni.py::test_thinker_prefix_caching_text_output[default]",
    "tests/e2e/online_serving/test_qwen3_omni.py::test_thinker_prefix_caching_audio_output[default]",
    "tests/e2e/online_serving/test_qwen3_omni.py::"
    "test_thinker_prefix_caching_async_chunk_identical[prefix_cache_async_chunk]",
    "tests/e2e/online_serving/test_qwen3_omni.py::"
    "test_thinker_prefix_caching_async_chunk_shared_image_prefix[prefix_cache_async_chunk]",
    "tests/e2e/online_serving/test_qwen3_omni.py::test_completions_rejected_for_thinker_talker[default-rocm]",
    "tests/e2e/online_serving/test_qwen3_omni.py::test_batched_completions_text[default-rocm]",
    "tests/e2e/online_serving/test_qwen3_omni.py::test_batched_completions_audio_out[default-rocm]",
    "tests/e2e/online_serving/test_qwen3_omni.py::test_batched_completions_with_bad_values[sad_opts0-default-rocm]",
    "tests/e2e/online_serving/test_qwen3_omni.py::test_batched_completions_with_bad_values[sad_opts1-default-rocm]",
)


def test_qwen3_omni_ready_selects_eleven_core_model_nodes() -> None:
    """READY runs the online core_model cases only, and only on two MI325 cards."""
    step = _find_step("Omni Model Test Qwen3-Omni", AMD_READY_PIPELINE)
    command = "\n".join(step["commands"])

    assert step["agent_pool"] == "mi300_2"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 25
    assert step["artifact_paths"] == ["artifacts/rocm-qwen3-omni-ready/**/*"]
    assert "test_qwen3_omni_thinking.py" not in command
    assert "offline_inference" not in command
    selector = "core_model and rocm and MI325 and cards_2 and not nvidia_only"
    assert command.count(selector) == 2
    assert command.count("--shard-id=0 --num-shards=1") == 2
    assert "timeout --signal=TERM --kill-after=2m 20m" in command
    assert "PIPESTATUS[0]" in command
    assert "pytest.xml" in command
    assert "pytest.log" in command
    assert "pytest-summary.txt" in command
    assert "Expected exactly 11 executed Qwen3-Omni READY tests" in command
    for node in _QWEN3_OMNI_READY_NODES:
        assert node in command
    assert command.count("tests/e2e/online_serving/test_qwen3_omni.py::") == len(_QWEN3_OMNI_READY_NODES)


@pytest.mark.parametrize(
    ("pytest_status", "tests", "skipped", "timeout_status", "expected_status"),
    [
        (0, 11, 0, None, 0),
        (0, 11, 1, None, 5),
        (0, 0, 0, None, 5),
        (1, 11, 0, None, 1),
        (0, 11, 0, 124, 124),
    ],
    ids=("pass", "skipped", "zero-tests", "pytest-failure", "timeout"),
)
def test_qwen3_omni_ready_executes_junit_guard_before_exit(
    tmp_path: Path,
    pytest_status: int,
    tests: int,
    skipped: int,
    timeout_status: int | None,
    expected_status: int,
) -> None:
    _assert_nightly_junit_guard(
        tmp_path,
        pipeline_path=AMD_READY_PIPELINE,
        step_label="Omni Model Test Qwen3-Omni",
        command_fragment="pytest -s -v -ra",
        expected_message="Expected exactly 11 executed Qwen3-Omni READY tests",
        artifact_env="ROCM_QWEN3_OMNI_READY_ARTIFACT_DIR",
        job_id_prefix="qwen3-omni-ready-guard",
        pytest_status=pytest_status,
        tests=tests,
        skipped=skipped,
        timeout_status=timeout_status,
        expected_status=expected_status,
    )


def test_entrypoints_full_nightly_splits_qwen_and_bagel_without_skips() -> None:
    group = _find_group(":full_moon: Entrypoints Full MI300 Nightly", AMD_NIGHTLY_PIPELINE)
    assert [step["label"] for step in group["steps"]] == [
        "Qwen3-Omni Duplex Playback",
        "BAGEL Sleep Mode",
    ]

    for (
        label,
        node,
        expected_count,
        dependency_key,
        artifacts,
        _artifact_env,
        expected_message,
    ) in ENTRYPOINTS_FULL_NIGHTLY_JOBS:
        step = _find_step(label, AMD_NIGHTLY_PIPELINE)
        assert step["agent_pool"] == "mi300_2"
        assert step["mirror_hardwares"] == ["amdproduction"]
        assert step["grade"] == "NonBlocking"
        assert step["timeout_in_minutes"] == 180
        assert step["source_file_dependencies"] == dependency_key
        assert step["artifact_paths"] == [artifacts]
        assert "retry" not in step

        commands = "\n".join(step["commands"])
        assert commands.count(node) == 2
        assert commands.count(ENTRYPOINTS_FULL_SELECTOR) == 2
        assert commands.count("--run-level full_model") == 2
        assert commands.count("--shard-id=0 --num-shards=1") == 2
        assert f"assert-pytest-collection.sh {expected_count}" in commands
        assert "timeout --signal=TERM --kill-after=2m 165m" in commands
        assert "PIPESTATUS[0]" in commands
        assert expected_message in commands
        for artifact_name in ("collection.log", "pytest.xml", "pytest.log", "pytest-summary.txt"):
            assert artifact_name in commands
        for selector in _marker_expressions(step):
            assert selector == ENTRYPOINTS_FULL_SELECTOR
            for cuda_marker in ("cuda", "L4", "H100", "B200"):
                assert re.search(rf"\b{cuda_marker}\b", selector) is None

    combined_commands = "\n".join(command for step in group["steps"] for command in step["commands"])
    assert "TestBagelCoordinatedSleepMode" not in combined_commands
    assert "test_coordinated_cross_device" not in combined_commands
    for pipeline_path in (AMD_READY_PIPELINE, AMD_MERGE_PIPELINE):
        text = pipeline_path.read_text(encoding="utf-8")
        for label, *_rest in ENTRYPOINTS_FULL_NIGHTLY_JOBS:
            assert label not in text


def test_entrypoints_full_nightly_uses_focused_source_dependencies() -> None:
    registry = yaml.safe_load(AMD_SOURCE_DEPENDENCIES.read_text(encoding="utf-8"))["source_file_dependencies"]

    def flatten_paths(items: list) -> set[str]:
        paths: set[str] = set()
        for item in items:
            if isinstance(item, list):
                paths.update(flatten_paths(item))
            else:
                paths.add(item)
        return paths

    qwen_paths = flatten_paths(registry["omni_qwen3_omni_duplex_playback_function"])
    assert {
        "tests/entrypoints/openai_api/test_qwen3_omni_realtime_websocket.py",
        "tests/e2e/online_serving/helpers/minicpmo_4_5_duplex.py",
        "tests/assets/minicpmo_4_5/",
        "vllm_omni/deploy/qwen3_omni_duplex.yaml",
        "vllm_omni/engine/duplex/",
        "vllm_omni/model_executor/models/qwen3_omni/",
    } <= qwen_paths

    bagel_paths = flatten_paths(registry["omni_bagel_sleep_function"])
    assert {
        "tests/entrypoints/test_omni_sleep_mode.py",
        "vllm_omni/deploy/bagel_single_stage.yaml",
        "vllm_omni/diffusion/models/bagel/",
        "vllm_omni/diffusion/worker/diffusion_worker.py",
        "vllm_omni/entrypoints/async_omni.py",
        "vllm_omni/model_executor/models/bagel/",
        "vllm_omni/model_executor/stage_input_processors/bagel.py",
    } <= bagel_paths

    for paths in (qwen_paths, bagel_paths):
        assert "tests/entrypoints/" not in paths
        assert "vllm_omni/entrypoints/" not in paths


def test_qwen_duplex_rocm_mark_is_limited_to_the_full_model_method() -> None:
    source = Path("tests/entrypoints/openai_api/test_qwen3_omni_realtime_websocket.py").read_text(encoding="utf-8")
    assert ('@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)\nclass TestQwen3OmniDuplexPlayback:') in source
    assert source.count('@hardware_test(res={"rocm": "MI325"}, num_cards=2)') == 1
    full_method_prefix = source.split("def test_speech_interrupts_playback_full", maxsplit=1)[0].rsplit(
        "@pytest.mark.full_model", maxsplit=1
    )[1]
    assert '@hardware_test(res={"rocm": "MI325"}, num_cards=2)' in full_method_prefix


def test_bagel_rocm_nightly_allows_cold_aiter_startup() -> None:
    source = Path("tests/entrypoints/test_omni_sleep_mode.py").read_text(encoding="utf-8")
    fixture = source.split("async def bagel_diffusion_engine", maxsplit=1)[1].split(
        "@pytest.mark.full_model", maxsplit=1
    )[0]
    assert "init_timeout=1200" in fixture


@pytest.mark.parametrize(
    ("pytest_status", "tests", "skipped", "timeout_status", "expected_status"),
    [
        (0, None, 0, None, 0),
        (0, None, 1, None, 5),
        (0, 0, 0, None, 5),
        (1, None, 0, None, 1),
        (0, None, 0, 124, 124),
    ],
    ids=("pass", "skipped", "zero-tests", "pytest-failure", "timeout"),
)
@pytest.mark.parametrize(
    ("step_label", "_node", "expected_tests", "_dependency_key", "_artifacts", "artifact_env", "expected_message"),
    ENTRYPOINTS_FULL_NIGHTLY_JOBS,
    ids=("qwen3-omni-duplex", "bagel-sleep"),
)
def test_entrypoints_full_nightly_executes_junit_guard_before_exit(
    tmp_path: Path,
    step_label: str,
    _node: str,
    expected_tests: int,
    _dependency_key: str,
    _artifacts: str,
    artifact_env: str,
    expected_message: str,
    pytest_status: int,
    tests: int | None,
    skipped: int,
    timeout_status: int | None,
    expected_status: int,
) -> None:
    _assert_nightly_junit_guard(
        tmp_path,
        step_label=step_label,
        command_fragment="pytest -s -v -ra",
        expected_message=expected_message,
        artifact_env=artifact_env,
        job_id_prefix=step_label.lower().replace(" ", "-"),
        pytest_status=pytest_status,
        tests=expected_tests if tests is None else tests,
        skipped=skipped,
        timeout_status=timeout_status,
        expected_status=expected_status,
    )


_MINICPMO_READY_MARKER = "core_model and omni and rocm and MI325 and cards_1"
_MINICPMO_READY_JOBS = (
    (
        "ROCm · MiniCPM-o 4.5 Offline Coverage",
        30,
        "20m",
        "artifacts/rocm-minicpmo-45-offline",
        (
            "tests/e2e/offline_inference/test_minicpmo_4_5.py::test_text_to_text[omni_runner0]",
            "tests/e2e/offline_inference/test_minicpmo_4_5.py::test_mix_to_audio[omni_runner0]",
        ),
        "exactly two executed MiniCPM-o offline tests",
    ),
    (
        "ROCm · MiniCPM-o 4.5 Online Coverage",
        30,
        "20m",
        "artifacts/rocm-minicpmo-45-online",
        (
            "tests/e2e/online_serving/test_minicpmo_4_5.py::test_text_to_text_001[async_chunk]",
            "tests/e2e/online_serving/test_minicpmo_4_5.py::test_text_to_text_001[turn_async_chunk]",
            "tests/e2e/online_serving/test_minicpmo_4_5.py::test_mix_to_text_audio_001[async_chunk]",
        ),
        "exactly three executed MiniCPM-o online tests",
    ),
    (
        "ROCm · MiniCPM-o 4.5 Duplex Coverage",
        45,
        "35m",
        "artifacts/rocm-minicpmo-45-duplex",
        (
            "tests/e2e/online_serving/test_minicpmo_4_5_duplex.py::"
            "test_duplex_websocket_protocol_smoke[three-stage-single-gpu]",
            "tests/e2e/online_serving/test_duplex_client_live.py::"
            "test_duplex_client_live_session[three-stage-single-gpu]",
        ),
        "exactly two executed MiniCPM-o duplex tests",
    ),
)


def test_minicpmo_ready_jobs_select_marked_core_model_nodes() -> None:
    """READY owns the CUDA core_model cases, selected by an explicit ROCm mark."""
    for label, timeout, command_timeout, artifact, nodes, junit_message in _MINICPMO_READY_JOBS:
        step = _find_step(label, AMD_READY_PIPELINE)
        command = "\n".join(step["commands"])

        assert step["agent_pool"] == "mi300_1"
        assert step["grade"] == "NonBlocking"
        assert step["timeout_in_minutes"] == timeout
        assert step["artifact_paths"] == [f"{artifact}/**/*"]
        assert f"--kill-after=1m {command_timeout}" in command
        assert _MINICPMO_READY_MARKER in command
        assert "--run-level core_model" in command
        assert command.count("--shard-id=0 --num-shards=1") == 2
        assert "test_minicpmo_4_5_duplex_mrv2.py" not in command
        assert "test_minicpmo_4_5_ready.json" not in command
        assert "test_minicpmo_4_5_duplex_expansion.py" not in command
        assert junit_message in command
        assert "--junitxml=" in command
        assert "pytest.log" not in command
        assert "pytest-summary.txt" not in command
        assert "PIPESTATUS" not in command
        assert "tee " not in command
        for node in nodes:
            assert command.count(node) == 2


def test_minicpmo_ready_core_model_cases_declare_mi325() -> None:
    """The selected cases carry MI325 so a one-card CUDA mark is not the selector."""
    cases = {
        "tests/e2e/offline_inference/test_minicpmo_4_5.py": (
            "def test_text_to_text(",
            "def test_mix_to_audio(",
        ),
        "tests/e2e/online_serving/test_minicpmo_4_5.py": (
            "def test_text_to_text_001(",
            "def test_mix_to_text_audio_001(",
        ),
        "tests/e2e/online_serving/test_minicpmo_4_5_duplex.py": ("def test_duplex_websocket_protocol_smoke(",),
        "tests/e2e/online_serving/test_duplex_client_live.py": ("def test_duplex_client_live_session(",),
    }
    for path, functions in cases.items():
        source = Path(path).read_text(encoding="utf-8")
        for function_name in functions:
            head, _, _ = source.partition(function_name)
            assert '"rocm": "MI325"' in head[-300:]


@pytest.mark.parametrize(
    (
        "step_label",
        "expected_tests",
        "expected_message",
        "artifact_env",
        "job_id_prefix",
    ),
    [
        (
            "ROCm · MiniCPM-o 4.5 Offline Coverage",
            2,
            "Expected exactly two executed MiniCPM-o offline tests",
            "ROCM_MINICPMO_OFFLINE_ARTIFACT_DIR",
            "minicpmo-ready-offline-guard",
        ),
        (
            "ROCm · MiniCPM-o 4.5 Online Coverage",
            3,
            "Expected exactly three executed MiniCPM-o online tests",
            "ROCM_MINICPMO_ONLINE_ARTIFACT_DIR",
            "minicpmo-ready-online-guard",
        ),
        (
            "ROCm · MiniCPM-o 4.5 Duplex Coverage",
            2,
            "Expected exactly two executed MiniCPM-o duplex tests",
            "ROCM_MINICPMO_DUPLEX_ARTIFACT_DIR",
            "minicpmo-ready-duplex-guard",
        ),
    ],
    ids=("offline", "online", "duplex"),
)
@pytest.mark.parametrize(
    (
        "pytest_status",
        "tests_override",
        "skipped",
        "timeout_status",
        "expected_status",
    ),
    [
        (0, None, 0, None, 0),
        (0, None, 1, None, 5),
        (0, 0, 0, None, 5),
        (1, None, 0, None, 1),
        (0, None, 0, 124, 124),
    ],
    ids=("pass", "skipped", "zero-tests", "pytest-failure", "timeout"),
)
def test_minicpmo_ready_executes_junit_guard_before_exit(
    tmp_path: Path,
    step_label: str,
    expected_tests: int,
    expected_message: str,
    artifact_env: str,
    job_id_prefix: str,
    pytest_status: int,
    tests_override: int | None,
    skipped: int,
    timeout_status: int | None,
    expected_status: int,
) -> None:
    """MiniCPM READY jobs fail closed on the JUnit count before the step exits."""
    _assert_nightly_junit_guard(
        tmp_path,
        pipeline_path=AMD_READY_PIPELINE,
        step_label=step_label,
        command_fragment='pytest -s -v "$${nodes[@]}"',
        expected_message=expected_message,
        artifact_env=artifact_env,
        job_id_prefix=job_id_prefix,
        pytest_status=pytest_status,
        tests=expected_tests if tests_override is None else tests_override,
        skipped=skipped,
        timeout_status=timeout_status,
        expected_status=expected_status,
    )


_WAN22_READY_MARKER = "core_model and diffusion and rocm and MI325 and cards_1"
_WAN22_READY_NODE = "tests/e2e/online_serving/test_wan22_t2v.py::test_text_to_video_001[default0]"


def test_wan22_ready_fail_closes_on_the_one_card_rocm_node() -> None:
    """READY keeps the blocking 40-minute Wan2.2 job and rejects the NPU param.

    Both params use id ``default``. Pytest 9 names them ``default0`` (CUDA/ROCm,
    one card) and ``default1`` (NPU, two cards) in source order.
    """
    step = _find_step("Diffusion · Wan22 Test", AMD_READY_PIPELINE)
    command = "\n".join(step["commands"])

    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "Blocking"
    assert "timeout_in_minutes" not in step
    assert "timeout 40m" in command
    assert _marker_expressions(step) == [_WAN22_READY_MARKER, _WAN22_READY_MARKER]
    assert command.count('--run-level "core_model"') == 2
    assert "advanced_model" not in command
    assert command.count(_WAN22_READY_NODE) == 1
    assert "test_text_to_video_001[default1]" not in command
    assert "default1" not in command
    assert "diff -u /tmp/wan22-ready-expected.txt /tmp/wan22-ready-collected.txt" in command

    merge_online = _find_step("Diffusion · Wan22 Online Test", AMD_MERGE_PIPELINE)
    merge_markers = _marker_expressions(merge_online)
    assert merge_markers
    assert all("advanced_model" in marker and "core_model" not in marker for marker in merge_markers)
    assert "tests/e2e/online_serving/test_wan22_t2v.py" in "\n".join(merge_online["commands"])


_ENGINE_ABORT_READY_MARKER = "core_model and omni and rocm and MI325 and cards_1"
_ENGINE_ABORT_READY_NODE = "tests/engine/test_async_omni_engine_abort.py::test_abort"


def test_engine_abort_ready_pins_the_mi325_node() -> None:
    """READY fail-closes on the existing one-card ROCm abort case."""
    step = _find_step("Engine Test", AMD_READY_PIPELINE)
    command = "\n".join(step["commands"])

    assert step["grade"] == "Blocking"
    assert "timeout 15m" in command
    assert _marker_expressions(step) == [
        _ENGINE_ABORT_READY_MARKER,
        _ENGINE_ABORT_READY_MARKER,
    ]
    assert command.count(_ENGINE_ABORT_READY_NODE) == 1
    assert "diff -u /tmp/engine-abort-ready-expected.txt" in command


_QWEN3_TTS_READY_MARKER = "core_model and tts and rocm and MI325 and cards_1"
_QWEN3_TTS_READY_NODES = (
    "tests/e2e/online_serving/test_qwen3_tts_customvoice.py::test_cached_predictor_streaming_audio[omni_server0]",
    "tests/e2e/online_serving/test_qwen3_tts_customvoice.py::test_default_cuda_graph_startup[async_chunk]",
    "tests/e2e/online_serving/test_qwen3_tts_customvoice.py::test_text_to_audio_001[async_chunk]",
    "tests/e2e/online_serving/test_qwen3_tts_customvoice.py::test_text_to_audio_002[async_chunk]",
)
_QWEN3_TTS_CUSTOMVOICE = Path("tests/e2e/online_serving/test_qwen3_tts_customvoice.py")


def test_qwen3_tts_ready_pins_selected_customvoice_core_nodes() -> None:
    """READY fail-closes on the existing one-card CustomVoice core cases."""
    step = _find_step("Qwen3-TTS E2E Test", AMD_READY_PIPELINE)
    command = "\n".join(step["commands"])
    source = _QWEN3_TTS_CUSTOMVOICE.read_text(encoding="utf-8")

    assert step["grade"] == "Blocking"
    assert "timeout 30m" in command
    assert _marker_expressions(step) == [_QWEN3_TTS_READY_MARKER, _QWEN3_TTS_READY_MARKER]
    for node in _QWEN3_TTS_READY_NODES:
        assert command.count(node) == 1
    assert "diff -u /tmp/qwen3-tts-ready-expected.txt" in command
    assert source.count('"rocm": "MI325"') == 4


_WORKER_ONE_CARD_MARKER = "core_model and rocm and MI325 and cards_1"
_WORKER_ONE_CARD_FILES = (
    "tests/worker/test_batched_omni_output.py",
    "tests/worker/test_gpu_generation_async_output.py",
    "tests/worker_v2/test_eager_frame_kernels.py",
)
_WORKER_ONE_CARD_EXCLUDED = (
    "tests/worker_v2/test_capture_query_bound_gpu.py",
    "tests/worker_v2/test_decode_query_bound_gpu.py",
    "tests/worker/test_gpu_ar_model_runner_cuda.py",
)
_WORKER_ONE_CARD_NODES = (
    "tests/worker/test_batched_omni_output.py::test_async_packed_snapshot_survives_source_reuse[1]",
    "tests/worker/test_batched_omni_output.py::test_async_packed_snapshot_survives_source_reuse[3]",
    "tests/worker/test_batched_omni_output.py::test_mtp_preserves_rows_and_snapshot_without_cuda_sync[offsets0]",
    "tests/worker/test_batched_omni_output.py::test_mtp_preserves_rows_and_snapshot_without_cuda_sync[offsets1]",
    "tests/worker/test_batched_omni_output.py::test_mixed_stop_flags_do_not_synchronize_cuda",
    "tests/worker/test_gpu_generation_async_output.py::test_async_generation_owns_output_and_request_mapping[tensor]",
    "tests/worker/test_gpu_generation_async_output.py::test_async_generation_owns_output_and_request_mapping[list]",
    "tests/worker/test_gpu_generation_async_output.py::test_async_generation_owns_output_and_request_mapping[mapping]",
    "tests/worker/test_gpu_generation_async_output.py::test_async_generation_preserves_none_rows",
    "tests/worker/test_gpu_generation_async_output.py::"
    "test_full_payload_accumulation_keeps_synchronous_host_copy[False]",
    "tests/worker/test_gpu_generation_async_output.py::"
    "test_full_payload_accumulation_keeps_synchronous_host_copy[True]",
    "tests/worker/test_gpu_generation_async_output.py::"
    "test_owned_generation_output_skips_device_snapshot_and_retains_sources[tensor]",
    "tests/worker/test_gpu_generation_async_output.py::"
    "test_owned_generation_output_skips_device_snapshot_and_retains_sources[list]",
    "tests/worker/test_gpu_generation_async_output.py::"
    "test_owned_generation_output_skips_device_snapshot_and_retains_sources[mapping]",
    "tests/worker/test_gpu_generation_async_output.py::test_owned_generation_output_preserves_none_rows",
    "tests/worker/test_gpu_generation_async_output.py::test_outer_graph_execution_keeps_snapshot_even_for_owned_model",
    "tests/worker/test_gpu_generation_async_output.py::test_discarded_output_records_copy_stream[False]",
    "tests/worker/test_gpu_generation_async_output.py::test_discarded_output_records_copy_stream[True]",
    "tests/worker/test_gpu_generation_async_output.py::test_dense_waveforms_share_one_host_allocation[False-False]",
    "tests/worker/test_gpu_generation_async_output.py::test_dense_waveforms_share_one_host_allocation[False-True]",
    "tests/worker/test_gpu_generation_async_output.py::test_dense_waveforms_share_one_host_allocation[True-False]",
    "tests/worker/test_gpu_generation_async_output.py::test_dense_waveforms_share_one_host_allocation[True-True]",
    "tests/worker_v2/test_eager_frame_kernels.py::test_settled_frames_match_torch[step_dtype0]",
    "tests/worker_v2/test_eager_frame_kernels.py::test_settled_frames_match_torch[step_dtype1]",
    "tests/worker_v2/test_eager_frame_kernels.py::test_codec_graph_inputs_match_the_copies[3-8-1]",
    "tests/worker_v2/test_eager_frame_kernels.py::test_codec_graph_inputs_match_the_copies[8-8-1]",
    "tests/worker_v2/test_eager_frame_kernels.py::test_codec_graph_inputs_match_the_copies[2-4-25]",
    "tests/worker_v2/test_eager_frame_kernels.py::test_eager_post_fills_codec_graph_inputs_and_scratch[3-8]",
    "tests/worker_v2/test_eager_frame_kernels.py::test_eager_post_fills_codec_graph_inputs_and_scratch[8-8]",
    "tests/worker_v2/test_eager_frame_kernels.py::test_history_scatter_preserves_independent_storage_and_offsets",
    "tests/worker_v2/test_eager_frame_kernels.py::test_codec_direct_inputs_wait_for_slow_side_stream",
    "tests/worker_v2/test_eager_frame_kernels.py::test_batched_history_recording_preserves_ragged_reordered_requests",
)


def test_worker_one_card_ready_stays_outside_r2_01() -> None:
    """READY adds one NonBlocking MI300 job for the collected one-card nodes.

    Local collection of the three files with the MI325 marker returned these
    32 ids. R2-01 still collects 1416 model-executor nodes and does not run them.
    """
    step = _find_step("ROCm · One-Card Worker GPU Test", AMD_READY_PIPELINE)
    command = "\n".join(step["commands"])
    body = command.split("nodes=(", 1)[1].split(")", 1)[0]
    quoted = tuple(re.findall(r'"(tests/[^"]+)"', body))

    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 30
    assert step["artifact_paths"] == ["artifacts/rocm-worker-one-card/**/*"]
    assert "timeout --signal=TERM --kill-after=1m 25m" in command
    assert "timeout 30m" not in command
    assert _marker_expressions(step) == [
        _WORKER_ONE_CARD_MARKER,
        _WORKER_ONE_CARD_MARKER,
    ]
    assert command.count('--run-level "core_model"') == 2
    assert quoted == _WORKER_ONE_CARD_NODES
    assert len(_WORKER_ONE_CARD_NODES) == 32
    for path in _WORKER_ONE_CARD_FILES:
        assert sum(node.startswith(path + "::") for node in quoted) >= 1
        assert command.count(f"\n  {path} \\\n") == 2
    for path in _WORKER_ONE_CARD_EXCLUDED:
        assert path not in command
        assert "MI325" not in Path(path).read_text(encoding="utf-8")
    assert "diff -u /tmp/worker-ready-expected.txt" in command
    assert "--junitxml=" in command
    assert "pytest.log" not in command
    assert "pytest-summary.txt" not in command
    assert "PIPESTATUS" not in command
    assert "tee " not in command

    for path in _WORKER_ONE_CARD_FILES[:2]:
        source = Path(path).read_text(encoding="utf-8")
        assert source.count('res={"cuda": "L4", "rocm": "MI325"}, num_cards=1') == 1
    eager = Path(_WORKER_ONE_CARD_FILES[2]).read_text(encoding="utf-8")
    assert eager.count('res={"cuda": "L4", "rocm": "MI325"}, num_cards=1') == 6

    r2_01 = _find_step(
        "ROCm · Model Executor GPU Coverage (R2-01) · Shard %N/%t",
        AMD_READY_PIPELINE,
    )
    r2_01_command = "\n".join(r2_01["commands"])
    assert "assert-pytest-collection.sh 1416" in r2_01_command
    assert "tests/model_executor/" in r2_01_command
    for path in _WORKER_ONE_CARD_FILES:
        assert path not in r2_01_command


_DRAMA3_READY_MARKER = "core_model and tts and rocm and MI325 and cards_1"
_DRAMA3_READY_NODE = "tests/e2e/online_serving/test_drama_3.py::test_text_to_audio_001[drama_3-rocm]"
_DRAMA3_EXCLUDED = (
    "test_voice_clone_streaming_002",
    "test_voice_clone_concurrent_003",
    "test_voice_clone_requires_ref_text_004",
)


def _hardware_decorator(source: str, function_name: str) -> str:
    head = source.split(f"def {function_name}", 1)[0]
    return head.rsplit("@hardware_test", 1)[1]


def test_drama3_ready_adds_one_nonblocking_text_to_audio_node() -> None:
    """READY adds one NonBlocking MI300 job for the one-card ROCm smoke.

    CUDA uses one card and NPU uses two, so ``hardware_test`` keeps a separate
    ROCm item. The server param id is ``drama_3`` and the platform id is ``rocm``.
    """
    step = _find_step("Drama 3 E2E Test", AMD_READY_PIPELINE)
    command = "\n".join(step["commands"])
    source = Path("tests/e2e/online_serving/test_drama_3.py").read_text(encoding="utf-8")

    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 40
    assert step["artifact_paths"] == ["artifacts/rocm-drama3-ready/**/*"]
    assert "timeout 30m" in command
    assert "timeout 40m" not in command
    assert _marker_expressions(step) == [_DRAMA3_READY_MARKER, _DRAMA3_READY_MARKER]
    assert command.count('--run-level "core_model"') == 2
    assert "full_model" not in command
    assert command.count(_DRAMA3_READY_NODE) == 1
    for name in _DRAMA3_EXCLUDED:
        assert name not in command
    assert "diff -u /tmp/drama3-ready-expected.txt" in command
    assert "--junitxml=" in command
    assert "pytest.log" not in command
    assert "pytest-summary.txt" not in command
    assert "PIPESTATUS" not in command
    assert "tee " not in command
    assert '"rocm": "MI325"' in _hardware_decorator(source, "test_text_to_audio_001")
    assert '"rocm": 1' in _hardware_decorator(source, "test_text_to_audio_001")
    for name in _DRAMA3_EXCLUDED:
        assert "rocm" not in _hardware_decorator(source, name)
    assert source.count('"rocm": "MI325"') == 1


_DEVICE_REDUCTION_READY_MARKER = "core_model and diffusion and rocm and MI325 and cards_1"
_DEVICE_REDUCTION_READY_NODES = (
    "tests/diffusion/test_device_reduction.py::test_device_reduction_matches_a_float32_reference[dtype0]",
    "tests/diffusion/test_device_reduction.py::test_device_reduction_matches_a_float32_reference[dtype1]",
    "tests/diffusion/test_device_reduction.py::test_device_reduction_matches_a_float32_reference[dtype2]",
    "tests/diffusion/test_device_reduction.py::"
    "test_device_reduction_deviates_from_a_narrow_postprocess_by_at_most_one[dtype0]",
    "tests/diffusion/test_device_reduction.py::"
    "test_device_reduction_deviates_from_a_narrow_postprocess_by_at_most_one[dtype1]",
    "tests/diffusion/test_device_reduction.py::test_device_reduction_shrinks_hop1_payload_4x",
    "tests/diffusion/test_device_reduction.py::test_device_reduction_stays_on_device",
    "tests/diffusion/test_device_reduction.py::test_device_reduction_without_denormalize_matches_already_unit_range",
    "tests/diffusion/test_diffusion_engine.py::test_move_tensor_tree_moves_nested_cuda_tensors_to_cpu",
    "tests/diffusion/test_wan_device_reduction.py::"
    "test_wan_media_reduction_matches_float32_reference_and_bounds_native_path",
    "tests/diffusion/test_wan_device_reduction.py::test_wan_uint8_media_finalization_does_not_rescale",
    "tests/diffusion/test_wan_device_reduction.py::test_diffusion_output_to_cpu_moves_wan_media_off_device",
    "tests/diffusion/test_wan_device_reduction.py::test_small_typed_media_is_on_cpu_before_ipc",
    "tests/diffusion/test_wan_device_reduction.py::test_wan_float_media_finalization_matches_the_legacy_path",
    "tests/diffusion/test_wan_device_reduction.py::test_oom_fallback_media_packs_without_device_widening",
    "tests/diffusion/test_wan_device_reduction.py::test_oom_fallback_recovers_the_allocator",
    "tests/diffusion/test_wan_device_reduction.py::test_oom_fallback_releases_late_uint8_traceback_before_clone",
)


def test_device_reduction_ready_pins_collected_gpu_nodes() -> None:
    """READY fail-closes on the one-card device-reduction GPU nodes.

    Local collection returned the dtype ids for test_device_reduction.py and
    the engine case. test_wan_device_reduction.py did not import
    (vllm.renderers.chat_utils). Its nodes are the unparametrized GPU
    functions. The non-5D CPU case stays off this marker.
    """
    step = _find_step("ROCm · Diffusion Device-Reduction GPU Test", AMD_READY_PIPELINE)
    command = "\n".join(step["commands"])
    body = command.split("nodes=(", 1)[1].split(")", 1)[0]
    quoted = tuple(re.findall(r'"(tests/[^"]+)"', body))
    reduction = Path("tests/diffusion/test_device_reduction.py").read_text(encoding="utf-8")
    wan = Path("tests/diffusion/test_wan_device_reduction.py").read_text(encoding="utf-8")
    engine = Path("tests/diffusion/test_diffusion_engine.py").read_text(encoding="utf-8")

    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 20
    assert "timeout --signal=TERM --kill-after=1m 15m" in command
    assert _marker_expressions(step) == [
        _DEVICE_REDUCTION_READY_MARKER,
        _DEVICE_REDUCTION_READY_MARKER,
    ]
    assert command.count('--run-level "core_model"') == 2
    assert quoted == _DEVICE_REDUCTION_READY_NODES
    assert len(quoted) == 17
    assert "test_device_reduction_rejects_non_5d" not in command
    assert "test_mossimage_ops.py" not in command
    assert "test_diffusion_worker.py" not in command
    assert "test_wan22" not in command
    assert "diff -u /tmp/device-reduction-ready-expected.txt" in command
    assert reduction.count("@_also_rocm_mi325") == 5
    assert 'res={"cuda": "L4", "rocm": "MI325"}, num_cards=1' in wan
    assert engine.count('res={"cuda": "L4", "rocm": "MI325"}, num_cards=1') == 1


_WORKER_COPY_MARKER = "core_model and rocm and MI325 and cards_1"
_WORKER_COPY_FILES = (
    "tests/worker/test_device_copy.py",
    "tests/worker_v2/test_intermediate_buffer_cuda_residency.py",
    "tests/worker_v2/test_omni_ar_model_runner_cuda.py",
)
_WORKER_COPY_OWNED_BY_9552 = _WORKER_ONE_CARD_FILES
_WORKER_COPY_EXCLUDED = (
    "tests/worker/test_index_staging.py",
    "tests/worker_v2/test_index_staging.py",
    "tests/worker_v2/test_shared_stage_cuda.py",
    "tests/worker_v2/test_staged_writes.py",
    "tests/worker/test_gpu_ar_model_runner_cuda.py",
    "tests/worker_v2/test_capture_query_bound_gpu.py",
    "tests/worker_v2/test_decode_query_bound_gpu.py",
)
_WORKER_COPY_NODES = (
    "tests/worker/test_device_copy.py::test_tensor_copy_cuda_handles_pageable_and_pinned_sources",
    "tests/worker/test_device_copy.py::test_device_stager_views_survive_until_their_readers_finish",
    "tests/worker/test_device_copy.py::test_device_stager_empty_input_does_not_consume_a_ring_slot",
    "tests/worker_v2/test_intermediate_buffer_cuda_residency.py::"
    "test_gpu_resident_values_stay_on_device_after_update[key0-dtype0]",
    "tests/worker_v2/test_intermediate_buffer_cuda_residency.py::"
    "test_gpu_resident_values_stay_on_device_after_update[key1-dtype1]",
    "tests/worker_v2/test_omni_ar_model_runner_cuda.py::test_chunked_fixed_token_scores_are_copied_from_cuda",
)


def test_worker_copy_ready_pins_remaining_one_card_nodes() -> None:
    """READY adds one NonBlocking MI300 job for the remaining copy cases.

    The marker selects both intermediate-buffer residency rows and the three
    device-copy GPU tests. #9552's worker files stay on the other job. R2-01
    still collects 1416 model-executor nodes.
    """
    step = _find_step("ROCm · One-Card Worker Copy Test", AMD_READY_PIPELINE)
    command = "\n".join(step["commands"])
    body = command.split("nodes=(", 1)[1].split(")", 1)[0]
    quoted = tuple(re.findall(r'"(tests/[^"]+)"', body))

    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 25
    assert step["artifact_paths"] == ["artifacts/rocm-worker-copy/**/*"]
    assert "timeout --signal=TERM --kill-after=1m 20m" in command
    assert "timeout 25m" not in command
    assert _marker_expressions(step) == [_WORKER_COPY_MARKER, _WORKER_COPY_MARKER]
    assert command.count('--run-level "core_model"') == 2
    assert quoted == _WORKER_COPY_NODES
    assert len(_WORKER_COPY_NODES) == 6
    assert set(quoted).isdisjoint(_WORKER_ONE_CARD_NODES)
    residency = [node for node in quoted if "test_intermediate_buffer_cuda_residency.py::" in node]
    assert residency == [
        "tests/worker_v2/test_intermediate_buffer_cuda_residency.py::"
        "test_gpu_resident_values_stay_on_device_after_update[key0-dtype0]",
        "tests/worker_v2/test_intermediate_buffer_cuda_residency.py::"
        "test_gpu_resident_values_stay_on_device_after_update[key1-dtype1]",
    ]
    for path in _WORKER_COPY_FILES:
        assert sum(node.startswith(path + "::") for node in quoted) >= 1
        assert command.count(f"\n  {path} \\\n") == 2
    for path in _WORKER_COPY_OWNED_BY_9552:
        assert path not in command
    for path in _WORKER_COPY_EXCLUDED:
        assert path not in command
    assert "test_tensor_copy_cpu_preserves_view_and_dtype" not in command
    assert "test_device_stager_cpu_falls_back_to_a_copy" not in command
    assert "diff -u /tmp/worker-copy-ready-expected.txt" in command
    assert "--junitxml=" in command

    device_copy = Path(_WORKER_COPY_FILES[0]).read_text(encoding="utf-8")
    assert device_copy.count('res={"cuda": "L4", "rocm": "MI325"}, num_cards=1') == 3
    assert device_copy.count("@pytest.mark.cpu") == 2
    residency_source = Path(_WORKER_COPY_FILES[1]).read_text(encoding="utf-8")
    assert residency_source.count('res={"cuda": "L4", "rocm": "MI325"}, num_cards=1') == 1
    runner = Path(_WORKER_COPY_FILES[2]).read_text(encoding="utf-8")
    assert runner.count('res={"cuda": "L4", "rocm": "MI325"}, num_cards=1') == 1

    r2_01 = _find_step(
        "ROCm · Model Executor GPU Coverage (R2-01) · Shard %N/%t",
        AMD_READY_PIPELINE,
    )
    r2_01_command = "\n".join(r2_01["commands"])
    assert "assert-pytest-collection.sh 1416" in r2_01_command
    for path in _WORKER_COPY_FILES:
        assert path not in r2_01_command


_PAUSE_KEEP_READY_MARKER = "core_model and diffusion and rocm and MI325 and cards_2"
_PAUSE_KEEP_READY_NODE = (
    "tests/diffusion/distributed/test_pause_keep_two_ranks.py::test_keep_pause_two_ranks_ack_and_rank_failure"
)


def test_pause_keep_two_ranks_ready_pins_the_mi325_node() -> None:
    """READY fail-closes on the two-rank pause-keep case."""
    step = _find_step("ROCm · Pause-Keep Two-Rank GPU Test", AMD_READY_PIPELINE)
    command = "\n".join(step["commands"])
    source = Path("tests/diffusion/distributed/test_pause_keep_two_ranks.py").read_text(encoding="utf-8")

    assert step["agent_pool"] == "mi300_2"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 40
    assert "timeout --signal=TERM --kill-after=1m 30m" in command
    assert "export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=900" in command
    assert "VLLM_ROCM_USE_AITER=0" not in command
    assert _marker_expressions(step) == [
        _PAUSE_KEEP_READY_MARKER,
        _PAUSE_KEEP_READY_MARKER,
    ]
    assert command.count('--run-level "core_model"') == 2
    assert command.count(_PAUSE_KEEP_READY_NODE) == 1
    assert "diff -u /tmp/pause-keep-ready-expected.txt" in command
    assert "cards_3" not in command
    assert "cards_4" not in command
    assert "test_diffusion_worker.py" not in command
    assert 'res={"cuda": "L4", "rocm": "MI325"}, num_cards=2' in source


_VERL_GENERATE_READY_MARKER = "core_model and diffusion and rocm and MI325 and cards_1"
_VERL_GENERATE_READY_NODE = "tests/e2e/features/rlhf_test/test_verl_omni_e2e.py::test_generate"


def test_verl_generate_ready_pins_the_mi325_node() -> None:
    """READY fail-closes on the one-card VeRL-Omni generate case."""
    step = _find_step("ROCm · VeRL-Omni Generate GPU Test", AMD_READY_PIPELINE)
    command = "\n".join(step["commands"])
    source = Path("tests/e2e/features/rlhf_test/test_verl_omni_e2e.py").read_text(encoding="utf-8")
    server = Path("tests/e2e/features/helpers/verl_omni_server.py").read_text(encoding="utf-8")

    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 40
    assert "timeout --signal=TERM --kill-after=1m 30m" in command
    assert _marker_expressions(step) == [
        _VERL_GENERATE_READY_MARKER,
        _VERL_GENERATE_READY_MARKER,
    ]
    assert command.count('--run-level "core_model"') == 2
    assert command.count(_VERL_GENERATE_READY_NODE) == 1
    assert "diff -u /tmp/verl-generate-ready-expected.txt" in command
    assert "export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA" in command
    assert "VLLM_OMNI_DLO_DP_WAVE_TIMEOUT" not in command
    assert "test_weight_transfer" not in command
    assert "ray==" not in command
    assert "omegaconf==" not in command
    assert 'res={"cuda": "L4", "rocm": "MI325"}, num_cards=1' in source
    assert 'tempfile.mkdtemp(prefix="ray-", dir="/tmp")' in source
    assert "_temp_dir=ray_temp_dir" in source
    assert "ray.get(refs, timeout=600)" in source
    assert "server.shutdown.remote()" in source
    assert source.index("server.shutdown.remote()") < source.index("ray.shutdown()")
    shutdown = server.split("async def shutdown", 1)[1].split("async def generate", 1)[0]
    assert "workers = _snapshot_diffusion_workers()" in shutdown
    assert shutdown.index("workers = _snapshot_diffusion_workers()") < shutdown.index("self.engine.shutdown()")
    assert "children(recursive=True)" in server
    assert "psutil.wait_procs" in shutdown
    assert "multiprocessing.active_children" not in server


_TINY_DIFFUSION_BASE_MARKER = "core_model and diffusion and rocm and MI325 and cards_1"
_TINY_DIFFUSION_BASE_NODES = (
    "tests/model_tests/diffusion/test_common_offline.py::test_pipeline_on_supported_tasks[AnimaPipeline]",
    "tests/model_tests/diffusion/test_common_offline.py::test_pipeline_on_supported_tasks[Flux2KleinPipeline]",
    "tests/model_tests/diffusion/test_common_offline.py::test_pipeline_on_supported_tasks[SanaVideoPipeline]",
    "tests/model_tests/diffusion/test_common_offline.py::"
    "test_pipeline_on_supported_tasks[SanaVideoPipeline[cache_dit]]",
    "tests/model_tests/diffusion/test_common_offline.py::"
    "test_pipeline_on_supported_tasks[SanaVideoPipeline[cpu_offload]]",
    "tests/model_tests/diffusion/test_common_offline.py::"
    "test_pipeline_on_supported_tasks[SanaVideoPipeline[layerwise_offload]]",
    "tests/model_tests/diffusion/test_common_offline.py::"
    "test_pipeline_on_supported_tasks[SanaVideoPipeline[cache_dit+cpu_offload]]",
    "tests/model_tests/diffusion/test_common_offline.py::"
    "test_pipeline_on_supported_tasks[SanaVideoPipeline[cache_dit+layerwise_offload]]",
    "tests/model_tests/diffusion/test_common_offline.py::test_pipeline_on_supported_tasks[SanaImageToVideoPipeline]",
    "tests/model_tests/diffusion/test_common_offline.py::test_pipeline_on_supported_tasks[QwenImagePipeline]",
    "tests/model_tests/diffusion/test_common_offline.py::"
    "test_pipeline_on_supported_tasks[LongCatImagePipeline[tea_cache]]",
    "tests/model_tests/diffusion/test_common_offline.py::"
    "test_pipeline_on_supported_tasks[LongCatImageEditPipeline[tea_cache]]",
    "tests/model_tests/diffusion/test_common_offline.py::"
    "test_pipeline_on_supported_tasks[Flux2Pipeline[cache_dit+layerwise_offload]]",
    "tests/model_tests/diffusion/test_common_offline.py::test_pipeline_on_supported_tasks[QwenImageEditPipeline]",
    "tests/model_tests/diffusion/test_common_offline.py::test_pipeline_on_supported_tasks[QwenImageEditPlusPipeline]",
    "tests/model_tests/diffusion/test_common_offline.py::"
    "test_pipeline_on_supported_tasks["
    "StableDiffusion3Pipeline[cache_dit+layerwise_offload]]",
    "tests/model_tests/diffusion/test_common_offline.py::"
    "test_pipeline_on_supported_tasks[StableDiffusionXLPipeline[cpu_offload]]",
    "tests/model_tests/diffusion/test_common_online.py::test_online_on_supported_tasks[AnimaPipeline]",
    "tests/model_tests/diffusion/test_common_online.py::test_online_on_supported_tasks[Flux2KleinPipeline]",
    "tests/model_tests/diffusion/test_common_online.py::test_online_on_supported_tasks[QwenImagePipeline]",
    "tests/model_tests/diffusion/test_common_online.py::test_online_on_supported_tasks[QwenImageEditPipeline]",
    "tests/model_tests/diffusion/test_common_online.py::test_online_on_supported_tasks[QwenImageEditPlusPipeline]",
)


def test_tiny_diffusion_base_ready_pins_one_card_nodes() -> None:
    """READY fail-closes on the one-card tiny-diffusion node ids.

    The ids are the pytest.param ids from the single-device branch. Local
    collection did not import the real modules (vllm RequestError / vllm._C).
    LTX2 stays unmarked. One-card acceleration groups on that branch are
    included; multi-GPU full_model groups are not.
    """
    step = _find_step("ROCm · Tiny Diffusion Base GPU Test", AMD_READY_PIPELINE)
    command = "\n".join(step["commands"])
    body = command.split("nodes=(", 1)[1].split(")", 1)[0]
    quoted = tuple(re.findall(r'"(tests/[^"]+)"', body))

    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
    assert step["timeout_in_minutes"] == 60
    assert step["artifact_paths"] == ["artifacts/rocm-tiny-diffusion-base/**/*"]
    assert "timeout --signal=TERM --kill-after=1m 50m" in command
    assert "timeout 60m" not in command
    assert _marker_expressions(step) == [
        _TINY_DIFFUSION_BASE_MARKER,
        _TINY_DIFFUSION_BASE_MARKER,
    ]
    assert command.count('--run-level "core_model"') == 2
    assert command.count('"$${nodes[@]}"') == 3
    assert "pytest -n" not in command
    assert "-n 4" not in command
    assert "VLLM_ROCM_USE_AITER=0" not in command
    assert "LTX2" not in command
    assert "full_model" not in "\n".join(_marker_expressions(step))
    assert "tests/distributed" not in command
    assert quoted == _TINY_DIFFUSION_BASE_NODES
    assert len(quoted) == 22
    offline = [node for node in quoted if "test_common_offline.py::" in node]
    online = [node for node in quoted if "test_common_online.py::" in node]
    assert len(offline) == 17
    assert len(online) == 5
    assert "diff -u /tmp/tiny-diffusion-base-ready-expected.txt" in command
    assert "--junitxml=" in command

    nightly = _find_step(
        "ROCm · Tiny Diffusion Multi-GPU · 2-GPU",
        AMD_NIGHTLY_PIPELINE,
    )
    assert all("cards_1" not in marker for marker in _marker_expressions(nightly))
