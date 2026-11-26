# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for Qwen3-Omni dataset benchmarks, without loading models."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml
from _pytest.mark.expression import Expression

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]
ROOT = Path(__file__).resolve().parents[2]
CUDA_COMBINED_JOB_KEYS = (
    "nightly-omni-performance-no-async-chunk",
    "nightly-omni-performance-async-chunk",
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_non_duplex_dataset_matrix(platform):
    configs = json.loads((ROOT / "tests/dfx/perf/tests/test_qwen3_omni_non_full_duplex.json").read_text())
    params = [p for c in configs for p in c["benchmark_params"] if platform in c["mark"][0]["hardware_marks"]["res"]]
    expected = (
        {
            ("default", "seed-tts", 64, True),
            ("async-prefix", "daily-omni", 8, True),
            *{("async-prefix", "seed-tts", c, c == 64) for c in (1, 16, 32, 64)},
        }
        if platform == "cuda"
        else {
            ("default", "seed-tts", 32, True),
            *{("async-prefix", "seed-tts", c, c == 32) for c in (1, 16, 32)},
        }
    )
    assert {
        (
            p["result_metadata"]["configuration"],
            p["dataset_name"],
            p["max_concurrency"][0],
            p["result_validation"]["accuracy"],
        )
        for p in params
    } == expected
    assert len(params) == len(expected)
    for p in params:
        full_count = {"daily-omni": 1197, "seed-tts": 1088}[p["dataset_name"]]
        assert p["dataset_source"]["count"] == full_count
        count = full_count if p["result_validation"]["accuracy"] else {1: 32, 16: 256, 32: 512}[p["max_concurrency"][0]]
        assert p["num_prompts"] == [count]


def test_dataset_server_groups_do_not_share_a_failed_media_engine():
    configs = json.loads((ROOT / "tests/dfx/perf/tests/test_qwen3_omni_non_full_duplex.json").read_text())
    assert len({cfg["test_name"] for cfg in configs}) == len(configs)
    # DFX's active server key includes test_name. Distinct dataset groups force
    # a new engine, while Seed's concurrency sweep keeps one server lifetime.
    assert all(len({p["dataset_name"] for p in cfg["benchmark_params"]}) == 1 for cfg in configs)
    seed = next(cfg for cfg in configs if cfg["test_name"] == "test_qwen3_omni_datasets_cuda_async_seed")
    assert len(seed["benchmark_params"]) == 4


@pytest.mark.parametrize(
    "config_file,test_name,prefix",
    [
        ("test_qwen3_omni_non_full_duplex.json", "test_qwen3_omni_datasets_npu_default", False),
        ("test_qwen3_omni_non_full_duplex.json", "test_qwen3_omni_datasets_npu_async_prefix", True),
        ("test_qwen3_omni_duplex_omni_duplex_eval.json", "test_qwen3_omni_duplex_omni_duplex_eval_npu", True),
    ],
)
def test_npu_matrix_preserves_platform_runtime_guards(config_file, test_name, prefix):
    from dataclasses import asdict

    from tests.helpers.stage_config import modify_stage_config
    from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config

    configs = json.loads((ROOT / "tests/dfx/perf/tests" / config_file).read_text())
    config = next(config for config in configs if config["test_name"] == test_name)
    params = config["server_params"]
    args = params["extra_cli_args"]
    source = str(ROOT / args[args.index("--deploy-config") + 1])
    original = _apply_platform_overrides(load_deploy_config(source), "npu")
    modified = modify_stage_config(source, updates=params["update"])
    actual = _apply_platform_overrides(load_deploy_config(modified), "npu")
    stages = {stage.stage_id: stage for stage in actual.stages}

    assert actual.model_runner == "v2"
    assert {stage_id: stage.devices for stage_id, stage in stages.items()} == {0: "0,1", 1: "2", 2: "2"}
    assert stages[0].tensor_parallel_size == 2
    assert config["mark"][0]["hardware_marks"]["num_cards"] == 4
    if "judge_server_params" in config:
        assert config["judge_server_params"]["device_index"] == "3"
    assert stages[0].engine_extras["enable_prefix_caching"] is prefix
    devices = {device for stage in actual.stages for device in stage.devices.split(",")}
    for benchmark in config["benchmark_params"]:
        if "result_metadata" in benchmark:
            metadata = benchmark["result_metadata"]
            assert metadata["inference_devices"] == len(devices)
            assert metadata["model_runner"] == actual.model_runner
            assert metadata["thinker_prefix_caching"] is prefix
    for stage in original.stages:
        expected = asdict(stage)
        effective = asdict(stages[stage.stage_id])
        assert expected["compilation_config"]["cudagraph_mode"] == "PIECEWISE"
        if stage.stage_id == 0:
            expected["engine_extras"].pop("enable_prefix_caching", None)
            effective["engine_extras"].pop("enable_prefix_caching", None)
        assert effective == expected, f"{test_name}: stage {stage.stage_id} changed unrelated platform settings"


def test_cuda_pipeline_uses_json_runner_and_function_regression():
    pipeline = yaml.safe_load((ROOT / ".buildkite/cuda/test-nightly.yml").read_text())
    steps = [s for group in pipeline["steps"] if "steps" in group for s in group["steps"]]
    combined = [s for s in steps if s.get("key") in CUDA_COMBINED_JOB_KEYS]
    assert len(combined) == 2
    for step in combined:
        assert 0 < step["timeout_in_minutes"] <= 120
        commands = "\n".join(step["commands"])
        assert "run_benchmark.py" in commands and "test_qwen3_omni_non_full_duplex.json" in commands
    expansion = (ROOT / "tests/e2e/online_serving/test_qwen3_omni_expansion.py").read_text(encoding="utf-8")
    assert "def test_qwen3_omni_seed_tts_reads_questions_verbatim" in expansion


@pytest.fixture
def nightly_generated_jobs(tmp_path):
    # Run the real generator's main(), stopping before the shell executes jobs.
    runner = (ROOT / "tools/nightly/run_nightly_jobs.sh").read_text(encoding="utf-8")
    generator = runner.split("python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]

    def generate(test_type: str) -> Path:
        log_dir = tmp_path / test_type.replace(",", "-")
        env = {
            **os.environ,
            "REPO_ROOT": str(ROOT),
            "LOG_DIR": str(log_dir),
            "YML": str(ROOT / ".buildkite/cuda/test-nightly.yml"),
            "LABEL_SUBSTR": "",
            "TEST_TYPE": test_type,
            "MODEL_TYPE": "omni",
            "PYTEST_MARK": "",
            "DRY_RUN": "0",
            "PYTHONUTF8": "1",
        }
        subprocess.run(
            [sys.executable, "-"],
            input=generator,
            env=env,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
            timeout=30,
        )
        return log_dir / "jobs"

    return generate


@pytest.mark.parametrize("test_type", ["perf", "acc", "perf,acc", "all"])
def test_qwen3_combined_jobs_enter_requested_filters_and_perf_manifest(nightly_generated_jobs, test_type):
    jobs = nightly_generated_jobs(test_type)
    perf_keys = (jobs / ".perf_job_keys").read_text(encoding="utf-8").splitlines()
    timeouts = dict(line.split("=", 1) for line in (jobs / ".job_timeouts").read_text(encoding="utf-8").splitlines())
    for key in CUDA_COMBINED_JOB_KEYS:
        assert perf_keys.count(key) == 1
        script = (jobs / f"{key}.sh").read_text(encoding="utf-8")
        assert "--test-config-file tests/dfx/perf/tests/test_qwen3_omni_non_full_duplex.json" in script
        assert 0 < int(timeouts[key]) <= 120
        assert f"{timeouts[key]}m pytest " in script


def test_qwen3_combined_jobs_do_not_enter_function_filter(nightly_generated_jobs):
    jobs = nightly_generated_jobs("function")
    assert any("Function Test" in path.read_text(encoding="utf-8") for path in jobs.glob("*.sh"))
    assert all(not (jobs / f"{key}.sh").exists() for key in CUDA_COMBINED_JOB_KEYS)


def test_cuda_pipeline_selects_each_matrix_case_once():
    pipeline = yaml.safe_load((ROOT / ".buildkite/cuda/test-nightly.yml").read_text(encoding="utf-8"))
    steps = [s for group in pipeline["steps"] if "steps" in group for s in group["steps"]]
    configs = {
        f"tests/dfx/perf/tests/{name}.json"
        for name in (
            "test_qwen3_omni_non_full_duplex",
            "test_qwen3_omni_duplex_omni_duplex_eval",
            "test_qwen3_omni_duplex_omniinteract",
        )
    }
    selected: Counter[tuple[str, str]] = Counter()
    for step in steps:
        for command in step.get("commands", []):
            args = shlex.split(command)
            if "--test-config-file" not in args:
                continue
            path = args[args.index("--test-config-file") + 1]
            if path not in configs:
                continue
            mark_filter = Expression.compile(args[args.index("-m") + 1])
            keyword_filter = Expression.compile(args[args.index("-k") + 1]) if "-k" in args else None
            for config in json.loads((ROOT / path).read_text(encoding="utf-8")):
                hardware = config["mark"][0]["hardware_marks"]
                marks = {mark for mark in config["mark"] if isinstance(mark, str)}
                marks.add(f"cards_{hardware['num_cards']}")
                for platform, names in hardware["res"].items():
                    marks.add(platform)
                    marks.update([names] if isinstance(names, str) else names)
                if not mark_filter.evaluate(lambda name, **_: name in marks):
                    continue
                for params in config["benchmark_params"]:
                    assert len(params["num_prompts"]) == len(params["max_concurrency"]) == 1
                    node = f"test_performance_benchmark[{config['test_name']}-{params['name']}]"
                    if keyword_filter is None or keyword_filter.evaluate(
                        lambda name, **_: name.lower() in node.lower()
                    ):
                        selected[(config["test_name"], params["name"])] += 1
    expected = {
        ("test_qwen3_omni_datasets_cuda_default", "default-seed-tts-c64"),
        ("test_qwen3_omni_datasets_cuda_async_daily", "async-prefix-daily-omni-c8"),
        *{("test_qwen3_omni_datasets_cuda_async_seed", f"async-prefix-seed-tts-c{c}") for c in (1, 16, 32, 64)},
        ("test_qwen3_omni_duplex_omni_duplex_eval", "c8_accuracy_response_stats"),
        ("test_qwen3_omni_duplex_omniinteract", "c8_accuracy_response_stats"),
    }
    assert selected == Counter({case: 1 for case in expected})


def test_a3_nightly_runs_qwen3_accuracy_and_perf_jobs():
    pipeline = yaml.safe_load((ROOT / ".buildkite/npu/test-npu-nightly.yml").read_text())
    steps = [step for group in pipeline["steps"] if "steps" in group for step in group["steps"]]
    commands = "\n".join(command for step in steps for command in step.get("commands", []))
    assert "test_qwen3_omni.py -m" in commands
    assert "test_qwen3_omni_no_async_chunk.json" in commands
    assert "test_qwen3_omni_async_chunk.json" in commands


def test_combined_artifact_enters_performance_report(tmp_path):
    report = _load("nightly_html_contract", "tools/nightly/generate_nightly_perf_html.py")
    combined = {
        "benchmark_suite": "qwen3-omni-datasets",
        "model_id": "Qwen",
        "name": "async-prefix-daily-omni-c8",
        "dataset_name": "daily-omni",
        "max_concurrency": 8,
        "mean_ttft_ms": 10,
        "daily_omni_accuracy": 0.75,
        "date": "20261110-000000",
    }
    (tmp_path / "result_test_combined.json").write_text(json.dumps(combined))
    (tmp_path / "result_acc_unrelated.json").write_text(json.dumps({"ssim": 0.9}))
    rows = list(report._iter_omni_json_records(str(tmp_path)))
    assert len(rows) == 1
    assert rows[0]["daily_omni_accuracy"] == 0.75
    assert rows[0]["mean_ttft_ms"] == 10
