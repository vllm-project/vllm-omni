# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the combined nightly, without loading model dependencies."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]
ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nightly = _load("qwen_nightly_contract", "tests/e2e/accuracy/qwen3_omni/nightly.py")
validation = _load("benchmark_coverage_contract", "vllm_omni/benchmarks/validation.py")


@pytest.mark.parametrize(
    "platform,sweep", [("cuda", [(1, 32), (16, 256), (32, 512)]), ("npu", [(1, 32), (4, 64), (8, 128)])]
)
def test_matrix_has_four_shared_full_runs_and_six_performance_runs(platform, sweep):
    cases = nightly.nightly_cases(platform)
    assert len(cases) == 10 and len({case.id for case in cases}) == 10
    full = [c for c in cases if c.accuracy]
    assert {(c.config, c.dataset, c.count) for c in full} == {
        (config, dataset, count)
        for config in ("default", "async-prefix")
        for dataset, count in (("daily-omni", 1197), ("seed-tts", 1088))
    }
    for dataset in ("daily-omni", "seed-tts"):
        assert [(c.concurrency, c.count) for c in cases if c.dataset == dataset and not c.accuracy] == sweep
    assert all(c.config == "async-prefix" for c in cases if not c.accuracy)


def test_coverage_rejects_partial_success_even_with_high_accuracy():
    result = {"request_count": 1197, "completed": 1000, "successful_audio_requests": 1000, "daily_omni_accuracy": 0.70}
    errors = validation.coverage_errors(result, 1197, require_audio=True)
    assert any("completed" in error for error in errors)
    assert any("successful_audio_requests" in error for error in errors)
    case = nightly.NightlyCase("default", "daily-omni", 64, 1197)
    assert nightly.accuracy_coverage_errors({"daily_omni_evaluated_ok": 1000}, case, "cuda")


def test_source_rows_cannot_disappear_before_submission():
    result = {"request_count": 1000, "completed": 1000, "successful_audio_requests": 1000}
    assert any("request_count" in e for e in validation.coverage_errors(result, 1197, require_audio=True))


def test_invalid_answers_are_not_confused_with_missing_requests():
    case = nightly.NightlyCase("default", "daily-omni", 64, 1197)
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
    case = nightly.NightlyCase("async-prefix", "seed-tts", 64, 1088)
    data = {"seed_tts_content_evaluated": 1088, "seed_tts_sim_evaluated": 1087, "seed_tts_utmos_evaluated": 1088}
    assert nightly.accuracy_coverage_errors(data, case, "cuda") == ["seed_tts_sim_evaluated=1087, expected 1088"]
    data["seed_tts_sim_evaluated"] = 1088
    data.pop("seed_tts_utmos_evaluated")
    assert nightly.accuracy_coverage_errors(data, case, "npu") == []
    assert nightly.accuracy_coverage_errors(data, case, "cuda")


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


def test_enforced_regression_uses_its_own_hardware_and_workload():
    case = nightly.NightlyCase("async-prefix", "daily-omni", 16, 256)
    reference = _reference(case)
    result = _valid_performance(case, 10.5)
    assert nightly.performance_gate(result, case, "H100", reference, "enforce") == []
    result["mean_ttft_ms"] = 11.1
    assert any("mean_ttft_ms" in e for e in nightly.performance_gate(result, case, "H100", reference, "enforce"))
    assert nightly.performance_gate(result, case, "B200", reference, "enforce")
    reference["references"]["H100"][case.id]["num_prompts"] = 32
    assert any("workload" in e for e in nightly.performance_gate(result, case, "H100", reference, "enforce"))


@pytest.mark.parametrize("bad", [None, 0, float("nan"), float("inf"), True])
def test_collection_mode_still_rejects_unmeasured_metrics(bad):
    case = nightly.NightlyCase("default", "seed-tts", 64, 1088)
    result = _valid_performance(case)
    result["mean_audio_rtf"] = bad
    assert nightly.performance_gate(result, case, "H100", _reference(case), "collect")
    assert result["performance_baseline_status"] == "collecting"


def test_missing_reference_cannot_silently_pass_enforce():
    case = nightly.NightlyCase("default", "daily-omni", 64, 1197)
    result = _valid_performance(case)
    manifest = {"schema_version": 1, "references": {}}
    assert nightly.performance_gate(result, case, "H100", manifest, "collect") == []
    assert nightly.performance_gate(result, case, "H100", manifest, "enforce")


def test_fast_mean_cannot_hide_missing_latency_samples():
    case = nightly.NightlyCase("default", "daily-omni", 64, 1197)
    result = _valid_performance(case)
    result["num_ttft_samples"] = 1000
    errors = nightly.performance_gate(result, case, "H100", _reference(case), "collect")
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
    case = nightly.NightlyCase("async-prefix", "seed-tts", 1, 32)
    work = tmp_path / "subset"
    args = nightly.dataset_args(case, root, work)
    assert "--seed-tts-root" in args
    assert (work / "en/meta.lst").read_text().splitlines() == lines[:32]
    assert (work / "en/prompt_wavs/first.wav").read_bytes() == b"wav"


def test_nightly_wiring_reuses_c64_and_preserves_other_pipelines():
    for platform, name in (("cuda", "test-nightly.yml"), ("npu", "test-npu-nightly.yml")):
        pipeline = yaml.safe_load((ROOT / ".buildkite" / platform / name).read_text())
        steps = [step for group in pipeline["steps"] if "steps" in group for step in group["steps"]]
        combined = [s for s in steps if "Qwen3-Omni Accuracy + Perf" in s.get("label", "")]
        assert len(combined) == 2
        commands = "\n".join(command for s in steps for command in s.get("commands", []))
        assert "test_qwen3_omni_no_async_chunk.json" not in commands
        assert "test_qwen3_omni_async_chunk.json" not in commands
        assert "test_qwen3_omni.py -m" not in commands
        assert all("test_qwen3_omni_nightly.py" in "\n".join(s["commands"]) for s in combined)
    cuda = (ROOT / ".buildkite/cuda/test-nightly.yml").read_text()
    assert "test_qwen3_omni_seed_tts_reads_questions_verbatim" in cuda


def test_combined_artifact_enters_performance_report(tmp_path):
    report = _load("nightly_html_contract", "tools/nightly/generate_nightly_perf_html.py")
    combined = {
        "nightly_suite": "qwen3-omni-non-duplex",
        "model_id": "Qwen",
        "name": "default-daily-c64",
        "dataset_name": "daily-omni",
        "max_concurrency": 64,
        "mean_ttft_ms": 10,
        "daily_omni_accuracy": 0.75,
        "date": "20261110-000000",
    }
    (tmp_path / "result_acc_combined.json").write_text(json.dumps(combined))
    (tmp_path / "result_acc_unrelated.json").write_text(json.dumps({"ssim": 0.9}))
    rows = list(report._iter_omni_json_records(str(tmp_path)))
    assert len(rows) == 1
    assert rows[0]["daily_omni_accuracy"] == 0.75
    assert rows[0]["mean_ttft_ms"] == 10
