# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import shlex
import stat
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NIGHTLY_PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
READY_PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
MERGE_PIPELINE = Path(".buildkite/amd/test-amd-merge.yml")
SOURCE_DEPENDENCIES = Path(".buildkite/common/ci_source_file_dependencies.yml")
COLLECTION_GUARD = Path(".buildkite/amd/scripts/assert-pytest-collection.sh")
RUNNER = Path(".buildkite/amd/scripts/run-qwen3-omni-rocm-nightly.sh")
CORRECTNESS_TEST = Path("tests/e2e/online_serving/test_qwen3_omni_multi_replicas.py")
ACCURACY_TEST = Path("tests/e2e/accuracy/qwen3_omni/test_qwen3_omni.py")
PERF_CONFIGS = {
    "Qwen3-Omni No-Async-Chunk Perf": Path("tests/dfx/perf/tests/test_qwen3_omni_no_async_chunk.json"),
    "Qwen3-Omni Async-Chunk Perf": Path("tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json"),
}
RUNNER_COMMAND = "bash .buildkite/amd/scripts/run-qwen3-omni-rocm-nightly.sh"

JOB_SPECS = {
    "Qwen3-Omni Function Expansion": {
        "pool": "mi300_2",
        "timeout": 180,
        "source_group": "omni_qwen3_omni_function",
        "artifact": "artifacts/rocm-qwen3-omni-function-expansion/**/*",
        "gpus": "2",
        "tests": "28",
        "results": "0",
        "marks": "full_model and rocm and MI325 and cards_2",
        "target": "tests/e2e/online_serving/test_qwen3_omni_expansion.py",
        "inner_timeout": "165m",
        "config": None,
    },
    "Qwen3-Omni Multi-Replica Correctness": {
        "pool": "mi300_4",
        "timeout": 180,
        "source_group": "omni_qwen3_omni_function",
        "artifact": "artifacts/rocm-qwen3-omni-multi-replica-correctness/**/*",
        "gpus": "4",
        "tests": "3",
        "results": "0",
        "marks": "full_model and omni and rocm and MI325 and cards_4",
        "target": str(CORRECTNESS_TEST),
        "inner_timeout": "165m",
        "config": None,
    },
    "Qwen3-Omni No-Async-Chunk Perf": {
        "pool": "mi300_2",
        "timeout": 300,
        "source_group": ["omni_qwen3_omni_perf", "ci_rocm"],
        "artifact": "artifacts/rocm-qwen3-omni-no-async-chunk-perf/**/*",
        "gpus": "2",
        "tests": "4",
        "results": "8",
        "marks": "full_model and omni and rocm and MI325 and cards_2",
        "target": "tests/dfx/perf/scripts/run_benchmark.py",
        "inner_timeout": "285m",
        "config": str(PERF_CONFIGS["Qwen3-Omni No-Async-Chunk Perf"]),
    },
    "Qwen3-Omni Async-Chunk Perf": {
        "pool": "mi300_2",
        "timeout": 180,
        "source_group": ["omni_qwen3_omni_perf", "ci_rocm"],
        "artifact": "artifacts/rocm-qwen3-omni-async-chunk-perf/**/*",
        "gpus": "2",
        "tests": "6",
        "results": "6",
        "marks": "full_model and omni and rocm and MI325 and cards_2",
        "target": "tests/dfx/perf/scripts/run_benchmark.py",
        "inner_timeout": "165m",
        "config": str(PERF_CONFIGS["Qwen3-Omni Async-Chunk Perf"]),
    },
}


def _walk_steps(steps: list[dict]):
    for step in steps:
        yield step
        yield from _walk_steps(step.get("steps", []))


def _pipeline_steps(path: Path) -> dict[str, dict]:
    pipeline = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {step["label"]: step for step in _walk_steps(pipeline["steps"]) if "label" in step}


def _sweep_count(params: dict) -> int:
    request_rate = params.get("request_rate", [])
    concurrency = params.get("max_concurrency", [])
    if not isinstance(request_rate, list):
        request_rate = [request_rate]
    if not isinstance(concurrency, list):
        concurrency = [concurrency]
    return len(request_rate) + len(concurrency)


def test_qwen3_omni_nightly_job_contracts() -> None:
    nightly = _pipeline_steps(NIGHTLY_PIPELINE)
    for label, expected in JOB_SPECS.items():
        step = nightly[label]
        env = step["env"]
        artifact = expected["artifact"]
        assert isinstance(artifact, str)
        assert step["agent_pool"] == expected["pool"]
        assert step["depends_on"] == "amd-build"
        assert step["mirror_hardwares"] == ["amdproduction"]
        assert step["grade"] == ("Blocking" if label == "Qwen3-Omni Multi-Replica Correctness" else "NonBlocking")
        assert step["timeout_in_minutes"] == expected["timeout"]
        assert step["source_file_dependencies"] == expected["source_group"]
        assert step["artifact_paths"] == [artifact]
        assert step["commands"] == [RUNNER_COMMAND]
        assert env["QWEN3_OMNI_ARTIFACT_DIR"].endswith(artifact.removesuffix("/**/*"))
        assert env["QWEN3_OMNI_EXPECTED_GPUS"] == expected["gpus"]
        assert env["QWEN3_OMNI_EXPECTED_TESTS"] == expected["tests"]
        assert env["QWEN3_OMNI_EXPECTED_RESULTS"] == expected["results"]
        assert env["QWEN3_OMNI_MARK_EXPRESSION"] == expected["marks"]
        assert env["QWEN3_OMNI_TEST_TARGET"] == expected["target"]
        assert env["QWEN3_OMNI_TIMEOUT"] == expected["inner_timeout"]
        if expected["config"] is None:
            assert "QWEN3_OMNI_CONFIG_FILE" not in env
        else:
            assert env["QWEN3_OMNI_CONFIG_FILE"] == expected["config"]

    for path in (READY_PIPELINE, MERGE_PIPELINE):
        labels = _pipeline_steps(path)
        assert JOB_SPECS.keys().isdisjoint(labels)


def test_qwen3_omni_perf_bounds_rocm_runtime_resources() -> None:
    nightly = _pipeline_steps(NIGHTLY_PIPELINE)
    expected = {
        "MIOPEN_FIND_MODE": "FAST",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }

    for label in ("Qwen3-Omni No-Async-Chunk Perf", "Qwen3-Omni Async-Chunk Perf"):
        env = nightly[label]["env"]
        assert {name: env[name] for name in expected} == expected

    correctness_env = nightly["Qwen3-Omni Multi-Replica Correctness"]["env"]
    assert expected.keys().isdisjoint(correctness_env)


def test_runner_requires_complete_artifacts_and_clean_teardown() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert RUNNER.stat().st_mode & stat.S_IXUSR
    for required in (
        "set -euo pipefail",
        "QWEN3_OMNI_EXPECTED_GPUS",
        "assert-pytest-collection.sh",
        "--junitxml=",
        "pytest-result.txt",
        "result-validation.txt",
        "dependencies.txt",
        "runtime-inventory.txt",
        "gpu-holders-before.txt",
        "gpu-holders-after.txt",
        "process-cleanup.txt",
        'cleanup_pipeline_status=("${PIPESTATUS[@]}")',
        'test_pipeline_status=("${PIPESTATUS[@]}")',
    ):
        assert required in source
    assert "--assert-baseline" not in source


def test_aiter_smoke_requires_three_distinct_server_configurations() -> None:
    step = _pipeline_steps(NIGHTLY_PIPELINE)["Qwen3-Omni AITER-on Smoke"]
    assert step["grade"] == "Blocking"
    assert step["agent_pool"] == "mi300_2"
    assert step["env"]["QWEN3_OMNI_EXPECT_AITER_MM_ENCODER"] == "1"
    enable, run = step["commands"]
    assert enable == "export VLLM_ROCM_USE_AITER=1"
    command, runner, cards, count, timeout, artifact, *nodes, flag, marks = shlex.split(run)
    assert (command, runner) == ("bash", ".buildkite/amd/scripts/run-rocm-diffusion-nightly.sh")
    assert (cards, count, timeout) == ("2", "3", "75m")
    assert step["artifact_paths"] == [f"artifacts/{artifact}/**/*"]
    target = "tests/e2e/online_serving/test_qwen3_omni_expansion.py::test_text_to_text_audio_001"
    assert nodes == [f"{target}[{param}]" for param in ("default", "async_chunk", "batch_token_2048")]
    assert (flag, marks) == ("-m", "full_model and rocm and MI325 and cards_2")


def test_runner_is_a_dependency_of_function_and_perf_jobs() -> None:
    dependencies = yaml.safe_load(SOURCE_DEPENDENCIES.read_text(encoding="utf-8"))["source_file_dependencies"]
    runner = str(RUNNER)
    collection_guard = str(COLLECTION_GUARD)
    for group in ("omni_qwen3_omni_function", "omni_qwen3_omni_perf"):
        assert collection_guard in dependencies[group]
        assert runner in dependencies[group]


def test_qwen3_omni_correctness_enables_only_the_new_rocm_resource() -> None:
    source = CORRECTNESS_TEST.read_text(encoding="utf-8")
    decorator = '@hardware_test(res={"cuda": ["H100", "B200"], "rocm": "MI325"}, num_cards=4)'
    assert source.count(decorator) == 3
    assert "QWEN3_OMNI_COLD_STAGE_INIT_TIMEOUT_S" in source
    assert "QWEN3_OMNI_COLD_INIT_TIMEOUT_S" in source
    assert source.count("current_omni_platform.is_rocm()") == 2
    assert "stage_init_timeout=(QWEN3_OMNI_COLD_STAGE_INIT_TIMEOUT_S" in source
    assert "init_timeout=QWEN3_OMNI_COLD_INIT_TIMEOUT_S" in source


def test_qwen3_omni_accuracy_uses_the_rocm_cold_start_timeout() -> None:
    source = ACCURACY_TEST.read_text(encoding="utf-8")
    assert "QWEN3_OMNI_COLD_STAGE_INIT_TIMEOUT_S" in source
    assert "stage_init_timeout=(QWEN3_OMNI_COLD_STAGE_INIT_TIMEOUT_S" in source
    assert "if current_omni_platform.is_rocm() else None" in source


def test_qwen3_omni_perf_counts_and_existing_baselines_are_preserved() -> None:
    expected_resources = {
        "Qwen3-Omni No-Async-Chunk Perf": [{"cuda": ["H100", "B200"], "npu": "A3", "rocm": "MI325"}],
        "Qwen3-Omni Async-Chunk Perf": [
            {"cuda": ["H100", "B200"], "npu": "A3", "rocm": "MI325"},
        ],
    }
    expected_baseline_keys = {
        "Qwen3-Omni No-Async-Chunk Perf": [{"A3", "H100", "MI300"}] * 4,
        "Qwen3-Omni Async-Chunk Perf": [{"A3", "H100", "MI300"}] * 6,
    }

    for label, path in PERF_CONFIGS.items():
        configs = json.loads(path.read_text(encoding="utf-8"))
        assert [config["mark"][0]["hardware_marks"]["res"] for config in configs] == expected_resources[label]
        rocm_configs = [config for config in configs if config["mark"][0]["hardware_marks"]["res"].get("rocm")]
        test_count = sum(len(config["benchmark_params"]) for config in rocm_configs)
        result_count = sum(_sweep_count(params) for config in rocm_configs for params in config["benchmark_params"])
        assert str(test_count) == JOB_SPECS[label]["tests"]
        assert str(result_count) == JOB_SPECS[label]["results"]
        baseline_keys = [set(params.get("baseline", {})) for config in configs for params in config["benchmark_params"]]
        assert baseline_keys == expected_baseline_keys[label]
        assert all("MI325" not in keys for keys in baseline_keys)


def test_qwen3_omni_perf_uses_default_v2_profile_and_cold_start_timeout() -> None:
    async_config = json.loads(PERF_CONFIGS["Qwen3-Omni Async-Chunk Perf"].read_text(encoding="utf-8"))[0]
    async_server = async_config["server_params"]
    assert "stage_config_name" not in async_server
    assert async_server["extra_cli_args"] == [
        "--async-chunk",
        "--enable-orch-monitor",
        "--stage-init-timeout",
        "900",
    ]

    no_async_config = json.loads(PERF_CONFIGS["Qwen3-Omni No-Async-Chunk Perf"].read_text(encoding="utf-8"))[0]
    no_async_server = no_async_config["server_params"]
    assert "stage_config_name" not in no_async_server
    assert no_async_server["extra_cli_args"] == ["--no-async-chunk", "--stage-init-timeout", "900"]
