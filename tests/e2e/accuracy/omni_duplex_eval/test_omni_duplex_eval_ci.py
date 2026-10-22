# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for the accuracy checks invoked by the DFX nightly."""

import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.e2e.accuracy.omni_duplex_eval.validation import assert_accuracy_result, load_accuracy_config
from vllm_omni.benchmarks.duplex.omni_duplex_eval_eval import summarize_scores
from vllm_omni.benchmarks.duplex.omni_duplex_eval_metrics import PROTOCOL_PIN

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
_CONFIG_PATH = Path(__file__).with_name("omni_duplex_eval_ci_config.json")


@pytest.fixture
def config():
    return json.loads(_CONFIG_PATH.read_text())


@pytest.fixture
def result(tmp_path, config):
    """Build real score files and summarize them through the production scorer."""
    score_root = tmp_path / "evaluation"
    for split, ids in config["sample_ids_per_split"].items():
        for sid in ids:
            row = {"id": sid, "protocol_pin": PROTOCOL_PIN, "response_meta": {"mix": config["mix"]}}
            if split.startswith("RTD_"):
                row.update(
                    family="rtd",
                    task_type=split.removeprefix("RTD_").lower(),
                    content={"content_score": 2.8},
                    temporal={"summary": {"avg_temporal_score": 3.0}},
                )
            else:
                task = {
                    "PR_correction": "correction",
                    "PR_event_reminder": "proactive_reminder",
                    "PR_post_event_reminder": "post_event_reminder",
                }[split]
                row.update(family="pr", task_type=task, pr={"all_success": True, "total_score": 1})
            path = score_root / split / f"{sid}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row))
    summary = summarize_scores(score_root)
    expected = sum(map(len, config["sample_ids_per_split"].values()))
    summary.update(
        status="completed",
        total=expected,
        evaluated=expected,
        failed=0,
        skipped=0,
        errors=[],
        score_root=str(score_root),
    )
    return {
        "date": "20261020-065052",
        "completed": expected,
        "failed": 0,
        "duplex_session_metrics": [
            {
                "client_ttft_ms": {"count": 1, "mean": 3.0},
                "client_ttfp_ms": {"count": 1, "mean": 4.0},
            }
            for _ in range(expected)
        ],
        "omni_duplex_eval": {
            "published": expected,
            "artifact_errors": [],
            "published_with_unfinished_responses": 0,
            "accuracy": summary,
        },
    }


def test_complete_scores_pass(result, config):
    assert_accuracy_result(result, config)


@pytest.mark.parametrize(
    "section,field,value,passes",
    [
        ("rtd", "mean_content_score", 1.8, True),
        ("rtd", "mean_content_score", 1.7999, False),
        ("rtd", "mean_avg_temporal_score", 1.4, True),
        ("rtd", "mean_avg_temporal_score", 1.3999, False),
        ("pr", "mean_all_success", 0.28, True),
        ("pr", "mean_all_success", 0.2799, False),
        ("pr", "mean_all_success", 11 / 36, True),
        ("pr", "mean_all_success", 10 / 36, False),
    ],
)
def test_fixed_quality_thresholds(result, config, section, field, value, passes):
    result["omni_duplex_eval"]["accuracy"][section][field] = value
    if passes:
        assert_accuracy_result(result, config)
    else:
        with pytest.raises(AssertionError, match="outside"):
            assert_accuracy_result(result, config)


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("benchmark", "completed", 65, "completed requests"),
        ("benchmark", "failed", 1, "failed benchmark requests"),
        ("duplex", "published", 65, "published samples"),
        ("duplex", "artifact_errors", ["disk full"], "artifact errors"),
        ("accuracy", "status", "partial", "evaluation did not complete"),
        ("accuracy", "total", 3, "total samples"),
        ("accuracy", "evaluated", 65, "evaluated samples"),
        ("accuracy", "failed", 1, "failed evaluations"),
        ("accuracy", "skipped", 1, "skipped evaluations"),
        ("accuracy", "errors", ["judge failed"], "evaluation errors"),
        ("accuracy", "samples", 65, "samples"),
        ("accuracy", "protocol_pin", "old", "summary protocol_pin"),
        ("accuracy", "rtd", {}, "RTD by_task"),
        ("accuracy", "pr", {}, "PR by_task"),
    ],
)
def test_incomplete_result_fails(result, config, section, field, value, message):
    duplex = result["omni_duplex_eval"]
    {"benchmark": result, "duplex": duplex, "accuracy": duplex["accuracy"]}[section][field] = value
    with pytest.raises(AssertionError, match=message):
        assert_accuracy_result(result, config)


@pytest.mark.parametrize(
    "section,field",
    [("rtd", "mean_content_score"), ("rtd", "mean_avg_temporal_score"), ("pr", "mean_all_success")],
)
@pytest.mark.parametrize("value", [0.0, None, True, float("nan"), float("inf"), 4.0])
def test_invalid_quality_fails(result, config, section, field, value):
    result["omni_duplex_eval"]["accuracy"][section][field] = value
    with pytest.raises(AssertionError, match="outside"):
        assert_accuracy_result(result, config)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("missing", "score files"),
        ("id", "sample IDs"),
        ("pin", "protocol_pin"),
        ("mix", "mix 'question'"),
        ("no_mix", "mix 'question'"),
    ],
)
def test_score_file_integrity(result, config, mutation, message):
    path = Path(result["omni_duplex_eval"]["accuracy"]["score_root"]) / "RTD_world_knowledge/360.json"
    if mutation == "missing":
        path.unlink()
    elif mutation == "id":
        path.rename(path.with_name("wrong.json"))
    else:
        row = json.loads(path.read_text())
        if mutation == "pin":
            row["protocol_pin"] = "old"
        elif mutation == "mix":
            row["response_meta"]["mix"] = "question"
        else:
            # Metadata written before mix was recorded counts as question.
            del row["response_meta"]["mix"]
        path.write_text(json.dumps(row))
    with pytest.raises(AssertionError, match=message):
        assert_accuracy_result(result, config)


def test_config_pin_must_match_runtime(result, config):
    config["protocol_pin"] = "old"
    with pytest.raises(AssertionError, match="source of truth"):
        assert_accuracy_result(result, config)


def test_zero_threshold_is_rejected(result, config):
    config["baseline"]["min_pr_mean_all_success"] = 0
    with pytest.raises(AssertionError, match="invalid PR success rate threshold"):
        assert_accuracy_result(result, config)


def test_dfx_invokes_full_accuracy_contract(result):
    from tests.dfx.perf.scripts.run_benchmark import assert_result

    test_root = Path(__file__).resolve().parents[3]
    dfx = json.loads((test_root / "dfx/perf/tests/test_minicpmo_4_5_omni_duplex_eval.json").read_text())[0]
    params = dfx["benchmark_params"][0]
    assert_result(result, params, 66)
    result["omni_duplex_eval"]["accuracy"]["pr"]["mean_all_success"] = 0
    with pytest.raises(AssertionError, match="PR success rate"):
        assert_result(result, params, 66)


@pytest.mark.parametrize("audio_sessions", [0, 1, 65, 66])
def test_dfx_accuracy_requires_batch_audio(result, audio_sessions):
    from tests.dfx.perf.scripts.run_benchmark import assert_result

    test_root = Path(__file__).resolve().parents[3]
    dfx = json.loads((test_root / "dfx/perf/tests/test_minicpmo_4_5_omni_duplex_eval.json").read_text())[0]
    for session in result["duplex_session_metrics"][audio_sessions:]:
        session["client_ttfp_ms"] = {"count": 0, "mean": None}
    if audio_sessions:
        assert_result(result, dfx["benchmark_params"][0], 66)
    else:
        with pytest.raises(AssertionError, match="no measured client audio"):
            assert_result(result, dfx["benchmark_params"][0], 66)


def test_nightly_workload_matches_ci_config():
    test_root = Path(__file__).resolve().parents[3]
    dfx = json.loads((test_root / "dfx/perf/tests/test_minicpmo_4_5_omni_duplex_eval.json").read_text())[0]
    params = dfx["benchmark_params"][0]
    assert params["duplex_eval_mix"] == "soundtrack"
    assert load_accuracy_config(str(_CONFIG_PATH), params)["mix"] == "soundtrack"


@pytest.mark.parametrize(
    "field,value",
    [
        ("duplex_eval_ids", ["0"]),
        ("num_prompts", [3]),
        ("duplex_eval_evaluate", False),
        ("duplex_eval_mix", "question"),
        ("duplex_eval_mix", None),
    ],
)
def test_wrong_workload_is_rejected(field, value):
    test_root = Path(__file__).resolve().parents[3]
    dfx = json.loads((test_root / "dfx/perf/tests/test_minicpmo_4_5_omni_duplex_eval.json").read_text())[0]
    params = dfx["benchmark_params"][0]
    if value is None:
        del params[field]
    else:
        params[field] = value
    with pytest.raises(AssertionError):
        load_accuracy_config(str(_CONFIG_PATH), params)


def test_unsupported_dataset_revision_is_rejected(tmp_path, config):
    config["dataset_revision"] = "unavailable"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(NotImplementedError, match="dataset_revision"):
        load_accuracy_config(str(path), {})


def test_empty_score_directory(tmp_path):
    assert summarize_scores(tmp_path) == {"protocol_pin": PROTOCOL_PIN, "samples": 0}


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "quality",
        "judge",
        "publication",
        "cli",
        "cli_after_result",
        "cuda_abort",
        "npu_exit",
        "silence",
        "audio_outage",
        "boundary",
    ],
)
@pytest.mark.parametrize("accuracy_dir_override", [False, True])
@pytest.mark.parametrize("resource_label", ["H100", "H800", "A3"])
@pytest.mark.parametrize("benchmark_params_name", [None, "rtd-pr-c8"])
def test_dfx_writes_both_artifacts_before_quality_gate(
    tmp_path, monkeypatch, result, config, failure, accuracy_dir_override, resource_label, benchmark_params_name
):
    from tests.dfx import conftest as dfx
    from tests.dfx.perf.scripts import run_benchmark as runner
    from vllm_omni.platforms import current_omni_platform

    test_root = Path(__file__).resolve().parents[3]
    dfx_config = json.loads((test_root / "dfx/perf/tests/test_minicpmo_4_5_omni_duplex_eval.json").read_text())[0]
    dfx_config["benchmark_params"][0]["duplex_eval_accuracy_config"] = str(_CONFIG_PATH.resolve())
    if benchmark_params_name is not None:
        dfx_config["benchmark_params"][0]["name"] = benchmark_params_name
    result["model_id"] = "/models/MiniCPM-o-4_5"
    monkeypatch.setattr(runner, "BENCHMARK_CONFIGS", [dfx_config])
    monkeypatch.setattr(runner, "get_runtime_resource_label", lambda: resource_label)
    monkeypatch.chdir(tmp_path)
    perf_dir = tmp_path / "perf-results"
    perf_dir.mkdir()
    monkeypatch.setenv("BENCHMARK_DIR", str(perf_dir))
    if accuracy_dir_override:
        accuracy_dir = tmp_path / "accuracy-results"
        monkeypatch.setenv("ACC_BENCH_RESULT_DIR", str(accuracy_dir))
    else:
        accuracy_dir = tmp_path / "tests/e2e/accuracy/omni_duplex_eval/results"
        monkeypatch.delenv("ACC_BENCH_RESULT_DIR", raising=False)
    monkeypatch.setenv("BUILDKITE_BRANCH", "test-branch")
    if failure == "quality":
        result["omni_duplex_eval"]["accuracy"]["pr"]["mean_all_success"] = 0.0
    elif failure == "judge":
        result["omni_duplex_eval"]["accuracy"].update(status="partial", failed=1, errors=["judge failed"])
    elif failure == "publication":
        result["omni_duplex_eval"]["artifact_errors"] = ["Cannot publish duplex metrics: disk full"]
    elif failure == "silence":
        result["duplex_session_metrics"][0] = {
            "client_ttft_ms": {"count": 0, "mean": None},
            "client_ttfp_ms": {"count": 0, "mean": None},
        }
    elif failure == "audio_outage":
        for session in result["duplex_session_metrics"]:
            session["client_ttfp_ms"] = {"count": 0, "mean": None}
    elif failure == "boundary":
        result["omni_duplex_eval"]["published_with_unfinished_responses"] = 1
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: failure == "npu_exit")
    returncode = -6 if failure in ("cuda_abort", "npu_exit") else 1 if failure in ("cli", "cli_after_result") else 0

    commands = []

    class BenchmarkProcess:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            if failure != "cli":
                path = (
                    Path(command[command.index("--result-dir") + 1]) / command[command.index("--result-filename") + 1]
                )
                path.write_text(json.dumps(result))

        def wait(self):
            return returncode

    @dataclass
    class ServerEndpoint:
        host: str = "localhost"
        port: int = 8000
        model: str = "openbmb/MiniCPM-o-4_5"
        judge_base_url: str = "http://localhost:8001"

    monkeypatch.setattr(dfx.subprocess, "Popen", BenchmarkProcess)
    case = {"test_name": dfx_config["test_name"], "params": dfx_config["benchmark_params"][0]}
    if failure and failure not in ("npu_exit", "silence", "boundary"):
        message = "benchmark exited with unexpected code" if returncode else ""
        with pytest.raises(AssertionError, match=message):
            runner.test_performance_benchmark(ServerEndpoint(), case)
    else:
        runner.test_performance_benchmark(ServerEndpoint(), case)

    assert len(commands) == 1
    accuracy_paths = list(accuracy_dir.glob("result_acc_*.json"))
    perf_paths = list(perf_dir.glob("result_*.json"))
    assert len(accuracy_paths) == len(perf_paths) == 1
    assert not list(perf_dir.glob("result_acc_*.json"))
    timestamp = perf_paths[0].stem.rsplit("_", 1)[-1]
    assert accuracy_paths[0].name == f"result_acc_MiniCPM-o-4_5_omni_duplex_eval_{timestamp.replace('-', '_')}.json"
    saved_perf = json.loads(perf_paths[0].read_text())
    saved = json.loads(accuracy_paths[0].read_text())
    assert saved["Hardware"] == saved_perf["Hardware"] == resource_label
    assert saved["model_id"] == (saved_perf.get("model_id") or ServerEndpoint.model)
    assert "model" not in saved
    assert saved["dataset"] == "omni_duplex_eval"
    assert saved["dataset_path"] == config["dataset"]
    assert saved["date"] == (saved_perf.get("date") or timestamp)
    assert saved["test_name"] == dfx_config["test_name"]
    assert saved["name"] == saved_perf.get("name") == benchmark_params_name
    assert saved["num_prompts"] == dfx_config["benchmark_params"][0]["num_prompts"][0] == 66
    assert saved["branch"] == "test-branch"
    assert saved["max_concurrency"] == 8
    assert saved["baseline"] == config["baseline"]
    assert saved["benchmark_exit_code"] == returncode
    if failure == "cli":
        assert saved["accuracy"]["status"] == "failed"
    else:
        assert saved["accuracy"] == result["omni_duplex_eval"]["accuracy"]
        assert saved["generation"] == {
            "completed": result["completed"],
            "failed": result["failed"],
            **{k: v for k, v in result["omni_duplex_eval"].items() if k != "accuracy"},
        }


@pytest.mark.parametrize("returncode", [0, 1, -6])
def test_qwen3_eval_keeps_existing_writer_behavior(tmp_path, monkeypatch, returncode):
    from tests.dfx import conftest as dfx

    test_root = Path(__file__).resolve().parents[3]
    config = json.loads((test_root / "dfx/perf/tests/test_qwen3_omni_duplex_omni_duplex_eval.json").read_text())[0]
    params = config["benchmark_params"][0]
    assert params["duplex_eval_evaluate"] is True
    assert "duplex_eval_accuracy_config" not in params
    monkeypatch.setenv("BENCHMARK_DIR", str(tmp_path))
    accuracy_dir = tmp_path / "accuracy"
    monkeypatch.setenv("ACC_BENCH_RESULT_DIR", str(accuracy_dir))

    class BenchmarkProcess:
        def __init__(self, command, **kwargs):
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            path = Path(command[command.index("--result-dir") + 1]) / command[command.index("--result-filename") + 1]
            path.write_text(json.dumps({"completed": params["num_prompts"][0]}))

        def wait(self):
            return returncode

    monkeypatch.setattr(dfx.subprocess, "Popen", BenchmarkProcess)
    saved = dfx.run_benchmark(
        [
            "--model",
            config["server_params"]["model"],
            "--dataset-path",
            params["dataset_path"],
            "--duplex-eval-evaluate",
        ],
        test_name=config["test_name"],
        flow=params["max_concurrency"][0],
        dataset_name=params["dataset_name"],
        num_prompt=params["num_prompts"][0],
        resource_label="H100",
    )
    assert saved["completed"] == params["num_prompts"][0]
    assert len(list(tmp_path.glob("result_*.json"))) == 1
    assert not accuracy_dir.exists()
