# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for Qwen3-Omni dataset benchmarks, without loading models."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import yaml
from _pytest.mark.expression import Expression

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]
ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nightly = _load("qwen_nightly_contract", "tests/dfx/perf/qwen3_omni_workload.py")
validation = _load("benchmark_coverage_contract", "vllm_omni/benchmarks/validation.py")


@pytest.mark.parametrize("platform", ["cuda", "npu"])
def test_json_matrix_matches_scenario_v2(platform):
    configs = json.loads((ROOT / "tests/dfx/perf/tests/test_qwen3_omni_datasets.json").read_text())
    cases = [
        nightly.case_from_params(p)[0]
        for c in configs
        for p in c["benchmark_params"]
        if p["qwen3_omni_workload"]["platform"] == platform
    ]
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
    assert {(c.config, c.dataset, c.concurrency, c.accuracy) for c in cases} == expected
    assert len(cases) == len(expected)
    assert all(c.count == nightly.DATASETS[c.dataset]["count"] for c in cases)


def test_dataset_server_groups_do_not_share_a_failed_media_engine():
    configs = json.loads((ROOT / "tests/dfx/perf/tests/test_qwen3_omni_datasets.json").read_text())
    assert len({cfg["test_name"] for cfg in configs}) == len(configs)
    # DFX's active server key includes test_name. Distinct dataset groups force
    # a new engine, while Seed's concurrency sweep keeps one server lifetime.
    assert all(len({p["dataset_name"] for p in cfg["benchmark_params"]}) == 1 for cfg in configs)
    seed = next(cfg for cfg in configs if cfg["test_name"] == "test_qwen3_omni_datasets_cuda_async_seed")
    assert len(seed["benchmark_params"]) == 4


def test_coverage_rejects_partial_success_even_with_high_accuracy():
    result = {"request_count": 1197, "completed": 1000, "successful_audio_requests": 1000, "daily_omni_accuracy": 0.70}
    errors = validation.coverage_errors(result, 1197, require_audio=True)
    assert any("completed" in error for error in errors)
    assert any("successful_audio_requests" in error for error in errors)
    case = nightly.WorkloadCase("default", "daily-omni", 64, 1197, accuracy=True)
    assert nightly.accuracy_coverage_errors({"daily_omni_evaluated_ok": 1000}, case, "cuda")


@pytest.mark.parametrize(
    "platform,dataset,concurrency,count", [("cuda", "daily-omni", 8, 1197), ("npu", "seed-tts", 32, 1088)]
)
def test_accuracy_does_not_depend_on_c64(platform, dataset, concurrency, count):
    params = {
        "dataset_name": dataset,
        "num_prompts": [count],
        "max_concurrency": [concurrency],
        "qwen3_omni_workload": {"platform": platform, "configuration": "async-prefix", "accuracy": True},
    }
    case, actual_platform, smoke = nightly.case_from_params(params)
    assert case.accuracy and actual_platform == platform and not smoke
    assert nightly.accuracy_coverage_errors({}, case, platform)
    params["num_prompts"] = [2]
    with pytest.raises(ValueError, match="complete pinned dataset"):
        nightly.case_from_params(params)
    params["qwen3_omni_workload"]["smoke"] = True
    assert nightly.case_from_params(params)[2]


def test_scoring_switches_are_scoped_and_restored(monkeypatch):
    monkeypatch.setenv("SEED_TTS_UTMOS_EVAL", "old")
    monkeypatch.delenv("SEED_TTS_WER_EVAL", raising=False)
    case = nightly.WorkloadCase("async-prefix", "seed-tts", 32, 1088, accuracy=True)
    with nightly.scoring_environment(case, "npu"):
        assert nightly.os.environ["SEED_TTS_WER_EVAL"] == "1"
        assert nightly.os.environ["SEED_TTS_UTMOS_EVAL"] == "0"
    assert "SEED_TTS_WER_EVAL" not in nightly.os.environ
    assert nightly.os.environ["SEED_TTS_UTMOS_EVAL"] == "old"


@pytest.mark.parametrize("indices", [[0, 0], [0, 1197], [0, True], [0], "0,647"])
def test_daily_smoke_rejects_ambiguous_sample_selection(indices):
    params = {
        "dataset_name": "daily-omni",
        "num_prompts": [2],
        "max_concurrency": [8],
        "qwen3_omni_workload": {
            "platform": "cuda",
            "configuration": "async-prefix",
            "accuracy": True,
            "smoke": True,
            "smoke_indices": indices,
        },
    }
    with pytest.raises(ValueError, match="matching Daily-Omni smoke subset"):
        nightly.case_from_params(params)


def test_source_rows_cannot_disappear_before_submission():
    result = {"request_count": 1000, "completed": 1000, "successful_audio_requests": 1000}
    assert any("request_count" in e for e in validation.coverage_errors(result, 1197, require_audio=True))


def test_invalid_answers_are_not_confused_with_missing_requests():
    case = nightly.WorkloadCase("default", "daily-omni", 64, 1197, accuracy=True)
    result = {
        "request_count": 1197,
        "completed": 1197,
        "successful_audio_requests": 1197,
        "daily_omni_evaluated_ok": 1197,
        "daily_omni_parse_failed": 297,
        "daily_omni_accuracy": 900 / 1197,
    }
    assert validation.coverage_errors(result, 1197, require_audio=True) == []
    assert nightly.accuracy_coverage_errors(result, case, "cuda") == []


@dataclass
class _TextOutput:
    success: bool


@dataclass
class _AudioOutput(_TextOutput):
    audio_duration: float
    audio_ttfp: float
    audio_rtf: float


def test_valid_audio_requires_decodable_duration_and_measurable_latency():
    good = _AudioOutput(success=True, audio_duration=1.0, audio_ttfp=0.2, audio_rtf=0.5)
    outputs: list[_TextOutput] = [
        good,
        replace(good, audio_duration=0.0),
        replace(good, audio_duration=float("nan")),
        replace(good, success=False),
        replace(good, audio_ttfp=0.0),
        _TextOutput(success=True),  # Upstream text-only output has no audio fields.
    ]
    assert validation.request_coverage(outputs) == {"request_count": 6, "successful_audio_requests": 1}


def test_seed_requires_all_scores_not_only_high_means():
    case = nightly.WorkloadCase("async-prefix", "seed-tts", 64, 1088, accuracy=True)
    data = {**_full_quality_result("seed-tts"), "seed_tts_sim_evaluated": 1087}
    assert nightly.accuracy_coverage_errors(data, case, "cuda") == ["seed_tts_sim_evaluated=1087, expected 1088"]
    data["seed_tts_sim_evaluated"] = 1088
    data.pop("seed_tts_utmos_evaluated")
    assert nightly.accuracy_coverage_errors(data, case, "npu") == []
    assert nightly.accuracy_coverage_errors(data, case, "cuda")


def _full_quality_result(dataset):
    if dataset == "daily-omni":
        return {"daily_omni_evaluated_ok": 1197, "daily_omni_accuracy": 0.75}
    return {
        "seed_tts_content_evaluated": 1088,
        "seed_tts_content_error_mean": 0.02,
        "seed_tts_sim_evaluated": 1088,
        "seed_tts_sim_mean": 0.85,
        "seed_tts_utmos_evaluated": 1088,
        "seed_tts_utmos_mean": 3.4,
    }


@pytest.mark.parametrize(
    "dataset,failure_key",
    [
        ("daily-omni", "daily_omni_request_failed"),
        ("daily-omni", "daily_omni_no_gold"),
        ("daily-omni", "timed_out_requests"),
        ("daily-omni", "cancelled_after_timeout_requests"),
        ("seed-tts", "seed_tts_request_failed"),
        ("seed-tts", "seed_tts_no_pcm"),
        ("seed-tts", "seed_tts_asr_failed"),
        ("seed-tts", "seed_tts_empty_audio"),
        ("seed-tts", "seed_tts_sim_failed"),
        ("seed-tts", "seed_tts_sim_skipped_no_ref"),
        ("seed-tts", "seed_tts_utmos_failed"),
    ],
)
def test_nightly_rejects_reported_failures_even_with_complete_score_counts(dataset, failure_key):
    case = nightly.WorkloadCase("default", dataset, 64, nightly.DATASETS[dataset]["count"], accuracy=True)
    result = {**_full_quality_result(dataset), failure_key: 1}
    errors = nightly.accuracy_coverage_errors(result, case, "cuda")
    assert any(failure_key in error for error in errors)


@pytest.mark.parametrize(
    "dataset,metric",
    [
        ("daily-omni", "daily_omni_accuracy"),
        ("seed-tts", "seed_tts_content_error_mean"),
        ("seed-tts", "seed_tts_sim_mean"),
        ("seed-tts", "seed_tts_utmos_mean"),
    ],
)
def test_nightly_requires_finite_scores(dataset, metric):
    case = nightly.WorkloadCase("default", dataset, 64, nightly.DATASETS[dataset]["count"], accuracy=True)
    result = {**_full_quality_result(dataset), metric: float("nan")}
    assert any(metric in error for error in nightly.accuracy_coverage_errors(result, case, "cuda"))


def test_npu_accepts_zero_wer_without_disabled_utmos_scores():
    case = nightly.WorkloadCase("default", "seed-tts", 64, 1088, accuracy=True)
    result = {**_full_quality_result("seed-tts"), "seed_tts_content_error_mean": 0.0}
    result.pop("seed_tts_utmos_evaluated")
    result.pop("seed_tts_utmos_mean")
    assert nightly.accuracy_coverage_errors(result, case, "npu") == []


def test_source_checkout_version_metadata_remains_available(monkeypatch):
    def version(package):
        if package == "vllm-omni":
            raise nightly.PackageNotFoundError(package)
        return {"vllm": "0.33.0", "torch": "2.10.0"}[package]

    monkeypatch.setattr(nightly, "version", version, raising=False)
    assert nightly.software_versions() == {"vllm": "0.33.0", "vllm-omni": "not-installed", "torch": "2.10.0"}


def _reference(case):
    return {
        "schema_version": 1,
        "references": {
            "H100": {
                case.id: {
                    "dataset_revision": nightly.DATASETS[case.dataset]["revision"],
                    "num_prompts": case.count,
                    "reference_commit": "frozen-reference",
                    "repetitions": 3,
                    "metrics": {metric: {"value": 10.0, "relative_tolerance": 0.1} for metric in case.metrics},
                }
            }
        },
    }


def _valid_performance(case, value=10.0):
    return {
        **dict.fromkeys(case.metrics, value),
        **dict.fromkeys(("num_ttft_samples", "num_audio_rtf_samples", "num_audio_ttfp_samples"), case.count),
    }


@pytest.mark.parametrize("mode", ["collect", "enforce"])
def test_performance_gate_returns_metadata_without_mutating_input(mode):
    case = nightly.WorkloadCase("default", "seed-tts", 64, 1088, accuracy=True)
    result = _valid_performance(case)
    original = dict(result)
    outcome = nightly.performance_gate(result, case, "H100", _reference(case), mode)
    assert result == original
    errors, annotated = outcome
    assert errors == []
    assert annotated == {
        **original,
        "performance_baseline_status": "collecting" if mode == "collect" else "enforced",
        "performance_baseline": _reference(case)["references"]["H100"][case.id],
    }


def test_enforced_regression_uses_its_own_hardware_and_workload():
    case = nightly.WorkloadCase("async-prefix", "daily-omni", 16, 256)
    reference = _reference(case)
    result = _valid_performance(case, 10.5)
    assert nightly.performance_gate(result, case, "H100", reference, "enforce")[0] == []
    result["mean_ttft_ms"] = 11.1
    errors, _ = nightly.performance_gate(result, case, "H100", reference, "enforce")
    assert any("mean_ttft_ms" in e for e in errors)
    assert nightly.performance_gate(result, case, "B200", reference, "enforce")[0]
    reference["references"]["H100"][case.id]["num_prompts"] = 32
    errors, _ = nightly.performance_gate(result, case, "H100", reference, "enforce")
    assert any("workload" in e for e in errors)


@pytest.mark.parametrize("bad", [None, 0, float("nan"), float("inf"), True])
def test_collection_mode_still_rejects_unmeasured_metrics(bad):
    case = nightly.WorkloadCase("default", "seed-tts", 64, 1088, accuracy=True)
    result = _valid_performance(case)
    result["mean_audio_rtf"] = bad
    errors, annotated = nightly.performance_gate(result, case, "H100", _reference(case), "collect")
    assert errors
    assert annotated["performance_baseline_status"] == "collecting"


def test_missing_reference_cannot_silently_pass_enforce():
    case = nightly.WorkloadCase("default", "daily-omni", 64, 1197, accuracy=True)
    result = _valid_performance(case)
    manifest = {"schema_version": 1, "references": {}}
    assert nightly.performance_gate(result, case, "H100", manifest, "collect")[0] == []
    assert nightly.performance_gate(result, case, "H100", manifest, "enforce")[0]


def test_fast_mean_cannot_hide_missing_latency_samples():
    case = nightly.WorkloadCase("default", "daily-omni", 64, 1197, accuracy=True)
    result = _valid_performance(case)
    result["num_ttft_samples"] = 1000
    errors, _ = nightly.performance_gate(result, case, "H100", _reference(case), "collect")
    assert any("num_ttft_samples" in e for e in errors)


def test_offline_mirror_requires_pinned_manifest(tmp_path):
    (tmp_path / "qa.json").write_text("[]")
    with pytest.raises(ValueError, match="manifest differs"):
        nightly.verify_manifest(tmp_path, "daily-omni")


def test_seed_subset_is_trimmed_before_loader_can_skip_rows(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "en/prompt_wavs").mkdir(parents=True)
    (root / "en/prompt_wavs/first.wav").write_bytes(b"wav")
    lines = [f"id{i}|reference|prompt_wavs/first.wav|target{i}" for i in range(1088)]
    (root / "en/meta.lst").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(nightly, "verify_manifest", lambda *_: None)
    case = nightly.WorkloadCase("async-prefix", "seed-tts", 1, 32)
    work = tmp_path / "subset"
    args = nightly.dataset_args(case, root, work)
    assert "--seed-tts-root" in args
    assert (work / "en/meta.lst").read_text().splitlines() == lines[:32]
    assert (work / "en/prompt_wavs/first.wav").read_bytes() == b"wav"


def test_cuda_pipeline_uses_json_runner_and_function_regression():
    pipeline = yaml.safe_load((ROOT / ".buildkite/cuda/test-nightly.yml").read_text())
    steps = [s for group in pipeline["steps"] if "steps" in group for s in group["steps"]]
    combined = [s for s in steps if "Qwen3-Omni Accuracy + Perf" in s.get("label", "")]
    assert len(combined) == 2
    for step in combined:
        commands = "\n".join(step["commands"])
        assert "run_benchmark.py" in commands and "test_qwen3_omni_datasets.json" in commands
    assert not any("Verbatim Reading Regression" in s.get("label", "") for s in steps)
    expansion = (ROOT / "tests/e2e/online_serving/test_qwen3_omni_expansion.py").read_text(encoding="utf-8")
    assert "def test_qwen3_omni_seed_tts_reads_questions_verbatim" in expansion


def test_cuda_pipeline_selects_each_matrix_case_once():
    pipeline = yaml.safe_load((ROOT / ".buildkite/cuda/test-nightly.yml").read_text(encoding="utf-8"))
    steps = [s for group in pipeline["steps"] if "steps" in group for s in group["steps"]]
    configs = {
        f"tests/dfx/perf/tests/{name}.json"
        for name in (
            "test_qwen3_omni_datasets",
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
        ("test_qwen3_omni_duplex_omni_duplex_eval", "c8_accuracy_response_stats_v5"),
        ("test_qwen3_omni_duplex_omniinteract", "c8_accuracy_response_stats_v5"),
    }
    assert selected == Counter({case: 1 for case in expected})


def test_a3_keeps_existing_jobs_until_full_matrix_qualification():
    pipeline = yaml.safe_load((ROOT / ".buildkite/npu/test-npu-nightly.yml").read_text())
    steps = [step for group in pipeline["steps"] if "steps" in group for step in group["steps"]]
    commands = "\n".join(command for step in steps for command in step.get("commands", []))
    assert "test_qwen3_omni_nightly.py" not in commands
    assert "test_qwen3_omni.py -m" in commands
    assert "test_qwen3_omni_no_async_chunk.json" in commands
    assert "test_qwen3_omni_async_chunk.json" in commands


def test_combined_artifact_enters_performance_report(tmp_path):
    report = _load("nightly_html_contract", "tools/nightly/generate_nightly_perf_html.py")
    combined = {
        "benchmark_suite": "qwen3-omni-datasets",
        "model_id": "Qwen",
        "name": "default-daily-c64",
        "dataset_name": "daily-omni",
        "max_concurrency": 64,
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
