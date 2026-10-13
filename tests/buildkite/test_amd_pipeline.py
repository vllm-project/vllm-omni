# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from importlib.util import find_spec
from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_MERGE_PIPELINE = Path(".buildkite/amd/test-amd-merge.yml")
AMD_NIGHTLY_PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
AMD_READY_PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
AMD_RUNNER = Path(".buildkite/amd/scripts/run-amd-test.sh")
AMD_TEMPLATE = Path(".buildkite/amd/test-template-amd-omni.j2")
AMD_OMNI_PROCESSOR_RUNNER = Path(".buildkite/amd/scripts/run-omni-processor-test.sh")
CUDA_MERGE_PIPELINE = Path(".buildkite/cuda/test-merge.yml")
CUDA_READY_PIPELINE = Path(".buildkite/cuda/test-ready.yml")
AR_PAGED_ATTENTION_LABEL = "ROCm · AR Diffusion Paged Attention GPU Test"
FLASH_AITER_CORE_LABEL = "ROCm · FlashAttention/AITER Core Prewarm"
FLASH_AITER_DENSE_LABEL = "ROCm · FlashAttention/AITER Dense Prewarm"
FLASH_AITER_VARLEN_LABEL = "ROCm · FlashAttention/AITER Varlen Prewarm"
FLASH_AITER_VALIDATION_LABEL = "ROCm · FlashAttention/AITER GPU Coverage"
FLASH_AITER_CORE_KEY = "rocm-flash-aiter-core-prewarm"
FLASH_AITER_DENSE_KEY = "rocm-flash-aiter-dense-prewarm"
FLASH_AITER_VARLEN_KEY = "rocm-flash-aiter-varlen-prewarm"
FLASH_AITER_VALIDATION_KEY = "rocm-flash-aiter-validation"
FLASH_AITER_CORE_MODULE = "module_aiter_core.so"
FLASH_AITER_DENSE_MODULE = "mha_fwd_bf16_nbias_nmask_nlse_ndropout_nqscale.so"
FLASH_AITER_VARLEN_MODULE = "mha_varlen_fwd_bf16_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale.so"
FLASH_AITER_MARKERS = "core_model and rocm and MI325 and cards_1"
FLASH_AITER_ATTENTION_PATH = "tests/diffusion/attention/"
FLASH_AITER_MAMMOTH_PATH = "tests/diffusion/models/mammoth_moda2/test_dit_attention_cuda.py"
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


@pytest.mark.parametrize(
    "pipeline_path",
    [CUDA_READY_PIPELINE, CUDA_MERGE_PIPELINE],
    ids=["ready", "merge"],
)
def test_cuda_pr_gates_retain_full_model_executor_coverage(pipeline_path: Path) -> None:
    step = _find_step("Simple · Model Executor Test", pipeline_path)
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
        assert step["timeout_in_minutes"] == 20
        assert "export VLLM_OMNI_AR_FA_REQUIRED=1" in step["commands"]

        pytest_command = next(command for command in step["commands"] if "pytest" in command)
        argv = split(pytest_command)
        assert argv[:4] == ["timeout", "--signal=TERM", "--kill-after=2m", "15m"]
        assert "tests/diffusion/ar_diffusion/test_paged_attention.py" in argv
        assert argv[argv.index("-m") + 1] == AR_PAGED_ATTENTION_MARKERS
        assert argv[argv.index("--run-level") + 1] == "core_model"

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


def test_ready_flash_aiter_prewarm_shards_are_staged_nonblocking_cache_producers() -> None:
    core = _find_step(FLASH_AITER_CORE_LABEL, AMD_READY_PIPELINE)
    dense = _find_step(FLASH_AITER_DENSE_LABEL, AMD_READY_PIPELINE)
    varlen = _find_step(FLASH_AITER_VARLEN_LABEL, AMD_READY_PIPELINE)

    expected = (
        (core, FLASH_AITER_CORE_KEY, "core", FLASH_AITER_CORE_MODULE, "amd-build"),
        (dense, FLASH_AITER_DENSE_KEY, "dense", FLASH_AITER_DENSE_MODULE, FLASH_AITER_CORE_KEY),
        (varlen, FLASH_AITER_VARLEN_KEY, "varlen", FLASH_AITER_VARLEN_MODULE, FLASH_AITER_CORE_KEY),
    )
    for step, key, variant, module, dependency in expected:
        assert step["key"] == key
        assert step["agent_pool"] == "mi300_1"
        assert step["depends_on"] == dependency
        assert step["mirror_hardwares"] == ["amdproduction"]
        assert step["grade"] == "NonBlocking"
        assert "parallelism" not in step
        assert "export VLLM_ROCM_USE_AITER=1" in step["commands"]
        assert 'export AITER_JIT_DIR="/tmp/vllm-omni-aiter-$$BUILDKITE_JOB_ID"' in step["commands"]
        assert 'export TORCH_EXTENSIONS_DIR="/tmp/vllm-omni-torch-extensions-$$BUILDKITE_JOB_ID"' in step["commands"]
        prewarm = next(command for command in step["commands"] if "prewarm_flash_aiter.py" in command)
        assert f"--variant {variant}" in prewarm
        assert any(module in command and command.lstrip().startswith("cp -v") for command in step["commands"])

    assert core["timeout_in_minutes"] == 10
    assert dense["timeout_in_minutes"] == 30
    assert varlen["timeout_in_minutes"] == 15
    assert core["artifact_paths"] == ["artifacts/rocm-flash-aiter-cache/core/*.so"]
    assert dense["artifact_paths"] == ["artifacts/rocm-flash-aiter-cache/dense/*.so"]
    assert varlen["artifact_paths"] == ["artifacts/rocm-flash-aiter-cache/varlen/*.so"]
    assert "5m" in next(command for command in core["commands"] if "prewarm_flash_aiter.py" in command)
    assert "24m" in next(command for command in dense["commands"] if "prewarm_flash_aiter.py" in command)
    assert "10m" in next(command for command in varlen["commands"] if "prewarm_flash_aiter.py" in command)

    for step in (dense, varlen):
        downloads = [command for command in step["commands"] if "buildkite-agent artifact download" in command]
        assert len(downloads) == 1
        assert FLASH_AITER_CORE_KEY in downloads[0]
        assert FLASH_AITER_CORE_MODULE in downloads[0]


def test_ready_flash_aiter_validation_preserves_selectors_and_result_artifacts() -> None:
    step = _find_step(FLASH_AITER_VALIDATION_LABEL, AMD_READY_PIPELINE)

    assert step["key"] == FLASH_AITER_VALIDATION_KEY
    assert step["agent_pool"] == "mi300_1"
    assert step["depends_on"] == [FLASH_AITER_DENSE_KEY, FLASH_AITER_VARLEN_KEY]
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 60
    assert "artifact_paths" not in step
    assert "retry" not in step
    assert "export VLLM_ROCM_USE_AITER=1" in step["commands"]
    assert 'export AITER_JIT_DIR="/tmp/vllm-omni-aiter-$$BUILDKITE_JOB_ID"' in step["commands"]
    assert 'export TORCH_EXTENSIONS_DIR="/tmp/vllm-omni-torch-extensions-$$BUILDKITE_JOB_ID"' in step["commands"]

    downloads = [command for command in step["commands"] if "buildkite-agent artifact download" in command]
    assert len(downloads) == 3
    assert any(FLASH_AITER_CORE_KEY in command and FLASH_AITER_CORE_MODULE in command for command in downloads)
    assert any(FLASH_AITER_DENSE_KEY in command and FLASH_AITER_DENSE_MODULE in command for command in downloads)
    assert any(FLASH_AITER_VARLEN_KEY in command and FLASH_AITER_VARLEN_MODULE in command for command in downloads)

    pytest_commands = [command for command in step["commands"] if "pytest" in split(command)]
    assert len(pytest_commands) == 2
    attention = next(command for command in pytest_commands if FLASH_AITER_ATTENTION_PATH in split(command))
    mammoth = next(command for command in pytest_commands if FLASH_AITER_MAMMOTH_PATH in split(command))
    attention_argv = split(attention)
    mammoth_argv = split(mammoth)
    assert attention_argv[:6] == ["timeout", "--signal=TERM", "--kill-after=2m", "55m", "pytest", "-s"]
    assert attention_argv.count(FLASH_AITER_ATTENTION_PATH) == 1
    assert attention_argv[attention_argv.index("-m") + 1] == FLASH_AITER_MARKERS
    assert attention_argv[attention_argv.index("--run-level") + 1] == "core_model"
    assert "-n" in attention_argv and attention_argv[attention_argv.index("-n") + 1] == "1"
    assert "--durations=0" in attention_argv
    assert "-k" not in attention_argv
    assert mammoth_argv[:6] == ["timeout", "--signal=TERM", "--kill-after=2m", "10m", "pytest", "-s"]
    assert mammoth_argv.count(FLASH_AITER_MAMMOTH_PATH) == 1
    assert mammoth_argv[mammoth_argv.index("--run-level") + 1] == "core_model"
    assert "-n" in mammoth_argv and mammoth_argv[mammoth_argv.index("-n") + 1] == "1"
    assert "--durations=0" in mammoth_argv
    assert "-k" not in mammoth_argv

    assert _find_steps_selecting_path(AMD_READY_PIPELINE, FLASH_AITER_ATTENTION_PATH) == [step]
    mammoth_steps = _find_steps_selecting_path(AMD_READY_PIPELINE, FLASH_AITER_MAMMOTH_PATH)
    assert step in mammoth_steps
    assert len(mammoth_steps) == 2
    sdpa_step = next(candidate for candidate in mammoth_steps if candidate != step)
    assert sdpa_step["label"] == "Diffusion · Model Test"
    assert any(
        "env DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA" in command and FLASH_AITER_MAMMOTH_PATH in command
        for command in sdpa_step["commands"]
    )


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


def test_joy_gpu_suite_is_not_duplicated_in_amd_model_lane() -> None:
    step = _find_step("Diffusion · Model Test", AMD_READY_PIPELINE)
    argv = split(next(command for command in step["commands"] if "tests/diffusion/models/" in split(command)))

    assert step["grade"] == "Blocking"
    assert step["agent_pool"] == "mi300_1"
    assert argv[:4] == ["timeout", "15m", "pytest", "-sv"]
    assert f"--ignore={JOY_GPU_PATH}" in argv
    assert "--ignore=tests/diffusion/models/auk/test_auk_vae_fastpath.py" in argv
    assert "--ignore=tests/diffusion/models/mammoth_moda2/test_dit_attention_cuda.py" in argv
    assert argv[argv.index("-m") + 1] == (
        "core_model and cuda and not (cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)"
    )
    assert argv[argv.index("--run-level") + 1] == "core_model"


def test_joy_gpu_helper_import_does_not_require_pytest_package_mode() -> None:
    source = Path(JOY_GPU_PATH).read_text(encoding="utf-8")
    assert "from . import" not in source
    assert "from tests.diffusion.models.joy_image import test_joy_image as joy_units" in source
    spec = find_spec("tests.diffusion.models.joy_image.test_joy_image")
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).resolve() == Path(JOY_GPU_PATH).with_name("test_joy_image.py").resolve()


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


def test_z_image_merge_timeout_covers_cold_aiter_compile() -> None:
    step = _find_step("Diffusion Model Test")
    pytest_command = next(command for command in step["commands"] if "test_z_image.py" in command)

    assert split(pytest_command)[:2] == ["timeout", "55m"]


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


def test_amd_template_preserves_step_keys_and_custom_dependencies() -> None:
    template = AMD_TEMPLATE.read_text(encoding="utf-8")
    assert template.count("{% if step.key %}") == 2
    assert template.count('key: "{{ step.key }}"') == 2
    assert template.count("{% if step.depends_on %}") == 2
    assert template.count("depends_on: {{ step.depends_on }}") == 2


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
