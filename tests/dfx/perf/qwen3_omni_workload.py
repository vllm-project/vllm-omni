# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reproducible, combined accuracy/performance workloads for Qwen3-Omni."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict


class DatasetSpec(TypedDict):
    repo: str
    revision: str
    manifest: str
    sha256: str
    count: int


MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MANIFEST = Path(__file__).with_name("baselines") / "qwen3_omni.json"
DATASETS: dict[str, DatasetSpec] = {
    "daily-omni": {
        "repo": "liarliar/Daily-Omni",
        "revision": "bf5a6ee4c829510b7a14869e3104efeafc330b07",
        "manifest": "qa.json",
        "sha256": "3210a45d42424c7d57c1b40a0b9aa2708fc02fab2364bf01fd7d16e1242e146b",
        "count": 1197,
    },
    "seed-tts": {
        "repo": "zhaochenyang20/seed-tts-eval",
        "revision": "8f5e1aa2a35d42f42e940074c1983358b9491f89",
        "manifest": "en/meta.lst",
        "sha256": "7cd29f701743e6c7033edbca764ddc0b979de8c4f4907eff5e5454682114661f",
        "count": 1088,
    },
}


@dataclass(frozen=True)
class WorkloadCase:
    config: str
    dataset: str
    concurrency: int
    count: int

    accuracy: bool = False

    @property
    def id(self) -> str:
        return f"{self.config}-{self.dataset}-c{self.concurrency}"

    @property
    def metrics(self) -> tuple[str, ...]:
        audio = ("mean_audio_rtf", "mean_audio_ttfp_ms")
        return ("mean_ttft_ms", *audio) if self.dataset == "daily-omni" else audio


def verify_manifest(root: Path, dataset: str) -> None:
    source = DATASETS[dataset]
    path = root / source["manifest"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != source["sha256"]:
        raise ValueError(f"{path}: manifest differs from pinned {source['revision']}; got SHA256 {actual}")


def resolve_dataset(dataset: str) -> Path:
    """Use a verified offline mirror or download the exact snapshot, never main."""
    source = DATASETS[dataset]
    env_key = "QWEN3_OMNI_DAILY_ROOT" if dataset == "daily-omni" else "QWEN3_OMNI_SEED_ROOT"
    local = os.environ.get(env_key)
    if local:
        root = Path(local).expanduser().resolve()
    else:
        from vllm_omni.transformers_utils.repo_utils import hf_api

        patterns = ["qa.json", "Videos.tar"] if dataset == "daily-omni" else ["en/**"]
        root = Path(
            hf_api().snapshot_download(
                repo_id=source["repo"], repo_type="dataset", revision=source["revision"], allow_patterns=patterns
            )
        )
    verify_manifest(root, dataset)
    if dataset == "daily-omni" and not (root / "Videos").is_dir():
        from vllm_omni.benchmarks.data_modules.daily_omni_dataset import _extract_daily_omni_videos_tar

        _extract_daily_omni_videos_tar(root / "Videos.tar", source["revision"])
    return root


def dataset_args(case: WorkloadCase, root: Path, work: Path, *, indices: list[int] | None = None) -> list[str]:
    """Truncate before loading: a skipped early row cannot be replaced by a later one."""
    verify_manifest(root, case.dataset)
    work.mkdir(parents=True, exist_ok=True)
    if case.dataset == "daily-omni":
        from vllm_omni.benchmarks.data_modules.daily_omni_dataset import _extract_daily_omni_videos_tar

        videos = root / "Videos"
        if not videos.is_dir():
            videos = _extract_daily_omni_videos_tar(root / "Videos.tar", DATASETS[case.dataset]["revision"])
        rows = json.loads((root / "qa.json").read_text())
        if len(rows) != DATASETS[case.dataset]["count"]:
            raise ValueError("Pinned Daily-Omni question count changed")
        qa = work / "qa.json"
        selected = rows[: case.count] if indices is None else [rows[index] for index in indices]
        qa.write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")
        return [
            "--dataset-name",
            "daily-omni",
            "--daily-omni-qa-json",
            str(qa),
            "--daily-omni-video-dir",
            str(videos),
            "--daily-omni-inline-local-video",
        ]
    lines = [
        line for line in (root / "en/meta.lst").read_text().splitlines() if line.strip() and not line.startswith("#")
    ]
    if len(lines) != DATASETS[case.dataset]["count"]:
        raise ValueError("Pinned Seed-TTS utterance count changed")
    locale = work / "en"
    locale.mkdir()
    for child in (root / "en").iterdir():
        if child.name != "meta.lst":
            (locale / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
    (locale / "meta.lst").write_text("\n".join(lines[: case.count]) + "\n", encoding="utf-8")
    return [
        "--dataset-name",
        "seed-tts",
        "--dataset-path",
        str(work),
        "--seed-tts-root",
        str(work),
        "--seed-tts-locale",
        "en",
    ]


def quality_baseline(case: WorkloadCase, platform: str) -> dict[str, float]:
    if not case.accuracy:
        return {}
    if case.dataset == "daily-omni":
        return {"min_daily_omni_accuracy": 0.69}
    limits = {"max_seed_tts_mean_wer": 0.05, "min_seed_tts_mean_sim": 0.75}
    if platform == "cuda":
        limits["min_seed_tts_mean_utmos"] = 3.10
    return limits


def software_versions() -> dict[str, str]:
    """Record package metadata without losing diagnostics for source-only installs."""
    versions = {}
    for package in ("vllm", "vllm-omni", "torch"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def performance_gate(
    result: dict, case: WorkloadCase, hardware: str, manifest: dict, mode: str
) -> tuple[list[str], dict]:
    """Return validation errors and an annotated copy, preserving the input artifact."""
    if mode not in ("collect", "enforce"):
        raise ValueError(f"Unknown baseline mode: {mode}")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported workload baseline schema")
    annotated = dict(result)
    errors = []
    sample_counts = {
        "mean_ttft_ms": "num_ttft_samples",
        "mean_audio_rtf": "num_audio_rtf_samples",
        "mean_audio_ttfp_ms": "num_audio_ttfp_samples",
    }
    for metric in case.metrics:
        value = result.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            errors.append(f"{metric} is missing, non-finite or non-positive: {value!r}")
        count_key = sample_counts[metric]
        if result.get(count_key) != case.count:
            errors.append(f"{count_key}={result.get(count_key)!r}, expected {case.count}")
    entry = manifest.get("references", {}).get(hardware, {}).get(case.id)
    annotated["performance_baseline_status"] = "collecting" if mode == "collect" else "enforced"
    annotated["performance_baseline"] = entry
    if mode == "collect":
        return errors, annotated
    if not isinstance(entry, dict) or not entry:
        return errors + [f"No calibrated baseline for {hardware}/{case.id}"], annotated
    if entry.get("dataset_revision") != DATASETS[case.dataset]["revision"] or entry.get("num_prompts") != case.count:
        return errors + ["Performance baseline workload does not match the pinned dataset/count"], annotated
    repetitions = entry.get("repetitions")
    if not entry.get("reference_commit") or not isinstance(repetitions, int) or repetitions < 2:
        return errors + ["Baseline must identify its reference commit and repeated measurements"], annotated
    for metric in case.metrics:
        spec = entry.get("metrics", {}).get(metric, {})
        if not isinstance(spec, dict):
            errors.append(f"Invalid baseline for {metric}")
            continue
        reference, tolerance = spec.get("value"), spec.get("relative_tolerance")
        if (
            not isinstance(reference, (int, float))
            or isinstance(reference, bool)
            or not math.isfinite(reference)
            or reference <= 0
            or not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            errors.append(f"Invalid baseline/tolerance for {metric}")
            continue
        value = result.get(metric)
        if isinstance(value, (int, float)) and value > reference * (1 + tolerance):
            errors.append(f"{metric}={value} exceeds {reference} * (1 + {tolerance})")
    return errors, annotated


def accuracy_coverage_errors(result: dict, case: WorkloadCase, platform: str) -> list[str]:
    """Require scoring for every submitted row in an explicit accuracy case."""
    if not case.accuracy:
        return []
    counts = (
        ["daily_omni_evaluated_ok"]
        if case.dataset == "daily-omni"
        else [
            "seed_tts_content_evaluated",
            "seed_tts_sim_evaluated",
        ]
    )
    if case.dataset == "seed-tts" and platform == "cuda":
        counts.append("seed_tts_utmos_evaluated")
    errors = [f"{key}={result.get(key)!r}, expected {case.count}" for key in counts if result.get(key) != case.count]
    failure_keys = ["timed_out_requests", "cancelled_after_timeout_requests"]
    if case.dataset == "daily-omni":
        failure_keys += ["daily_omni_request_failed", "daily_omni_no_gold"]
        score_keys = ["daily_omni_accuracy"]
    else:
        failure_keys += [
            "seed_tts_request_failed",
            "seed_tts_no_pcm",
            "seed_tts_asr_failed",
            "seed_tts_empty_audio",
            "seed_tts_sim_failed",
            "seed_tts_sim_skipped_no_ref",
        ]
        score_keys = ["seed_tts_content_error_mean", "seed_tts_sim_mean"]
        if platform == "cuda":
            failure_keys.append("seed_tts_utmos_failed")
            score_keys.append("seed_tts_utmos_mean")
    errors += [f"{key}={result[key]}; workload coverage is incomplete" for key in failure_keys if result.get(key, 0)]
    for key in score_keys:
        value = result.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            errors.append(f"{key} must be a finite score: {value!r}")
    return errors


@contextmanager
def scoring_environment(case: WorkloadCase, platform: str):
    """Limit evaluator switches to this benchmark subprocess and restore them."""
    enabled = case.accuracy and case.dataset == "seed-tts"
    values = {
        "SEED_TTS_WER_EVAL": str(int(enabled)),
        "SEED_TTS_SIM_EVAL": str(int(enabled)),
        "SEED_TTS_UTMOS_EVAL": str(int(enabled and platform == "cuda")),
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def case_from_params(params: dict) -> tuple[WorkloadCase, str, bool]:
    """Accuracy is explicit, independent of concurrency or smoke sample count."""
    spec = params["qwen3_omni_workload"]
    platform = spec["platform"]
    if platform not in ("cuda", "npu"):
        raise ValueError(f"Unsupported workload platform: {platform}")
    if not isinstance(spec.get("accuracy"), bool) or not isinstance(spec.get("smoke", False), bool):
        raise ValueError("Workload accuracy and smoke must be explicit booleans")
    counts, concurrencies = params["num_prompts"], params["max_concurrency"]
    if len(counts) != 1 or len(concurrencies) != 1:
        raise ValueError("Each dataset workload entry must identify one count/concurrency")
    case = WorkloadCase(spec["configuration"], params["dataset_name"], concurrencies[0], counts[0], spec["accuracy"])
    if case.config not in ("default", "async-prefix") or case.dataset not in DATASETS:
        raise ValueError("Unknown deployment or pinned dataset")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (case.count, case.concurrency)
    ):
        raise ValueError("Request count and concurrency must be positive integers")
    smoke = spec.get("smoke", False)
    indices = spec.get("smoke_indices")
    if indices is not None and (
        not smoke
        or case.dataset != "daily-omni"
        or not isinstance(indices, list)
        or len(indices) != case.count
        or any(type(index) is not int or not 0 <= index < DATASETS[case.dataset]["count"] for index in indices)
        or len(set(indices)) != case.count
    ):
        raise ValueError("Explicit sample indices require a matching Daily-Omni smoke subset")
    if case.count > DATASETS[case.dataset]["count"] or (not smoke and case.count != DATASETS[case.dataset]["count"]):
        raise ValueError("Qualification requires the complete pinned dataset; mark subsets as smoke")
    return case, platform, smoke


def annotate_result(
    result: dict,
    case: WorkloadCase,
    platform: str,
    hardware: str,
    *,
    smoke: bool,
    exit_code: int,
    enforce_baseline: bool = False,
) -> dict:
    from tests.e2e.accuracy import helpers
    from vllm_omni.benchmarks.validation import coverage_errors

    errors = coverage_errors(result, case.count, require_audio=True)
    errors += accuracy_coverage_errors(result, case, platform)
    limits = quality_baseline(case, platform)
    if case.accuracy and not smoke:
        if case.dataset == "daily-omni":
            errors += helpers._validate_daily_omni(result, min_accuracy=limits["min_daily_omni_accuracy"])
        else:
            errors += helpers._validate_seed_tts(
                result,
                max_mean_wer=limits["max_seed_tts_mean_wer"],
                min_mean_sim=limits["min_seed_tts_mean_sim"],
                min_mean_utmos=limits.get("min_seed_tts_mean_utmos"),
            )
    if exit_code != 0:
        errors.append(f"Benchmark subprocess exited with {exit_code}")
    manifest = json.loads(MANIFEST.read_text())
    mode = (
        "enforce"
        if enforce_baseline
        else ("collect" if smoke else os.environ.get("QWEN3_OMNI_BASELINE_MODE", manifest["mode"]))
    )
    perf_errors, annotated = performance_gate(result, case, hardware, manifest, mode)
    errors += perf_errors
    annotated.update(
        {
            "benchmark_suite": "qwen3-omni-datasets",
            "configuration": case.config,
            "accuracy_evaluated": case.accuracy,
            "qualification_scope": "smoke" if smoke else "full-dataset",
            "async_chunk": case.config == "async-prefix",
            "thinker_prefix_caching": case.config == "async-prefix",
            "model_runner": "v2",
            "inference_devices": 2,
            "dataset": case.dataset.replace("-", "_"),
            "dataset_revision": DATASETS[case.dataset]["revision"],
            "dataset_manifest_sha256": DATASETS[case.dataset]["sha256"],
            "modalities": ["text", "audio"],
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "software_versions": software_versions(),
            "quality_thresholds": limits,
            "benchmark_exit_code": exit_code,
            "validation_errors": errors,
        }
    )
    if annotated.get("num_tpot_samples") == 0:
        for key in annotated:
            if key.endswith("_tpot_ms"):
                annotated[key] = None
    return helpers._json_safe_metric_value(annotated)


def run_workload(server, params: dict, test_name: str, hardware: str, *, enforce_baseline: bool = False) -> dict:
    """Execute JSON-defined dataset cases through the shared DFX bench runner."""
    from tests.dfx.conftest import run_benchmark

    case, platform, smoke = case_from_params(params)
    if smoke and enforce_baseline:
        raise ValueError("Smoke cannot qualify a full-dataset performance baseline")
    root = resolve_dataset(case.dataset)
    with TemporaryDirectory(prefix="qwen3-omni-dataset-") as work, scoring_environment(case, platform):
        args = [
            "--host",
            server.host,
            "--port",
            str(server.port),
            "--model",
            server.model,
            "--backend",
            "openai-chat-omni",
            "--endpoint",
            "/v1/chat/completions",
            "--max-concurrency",
            str(case.concurrency),
            "--num-prompts",
            str(case.count),
            "--request-rate",
            "inf",
            "--temperature",
            "0",
            "--no-oversample",
            "--disable-shuffle",
            "--omni-isolate-prefix-cache",
            "--omni-fail-fast-on-timeout",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el,audio_ttfp,audio_rtf,audio_duration",
            "--extra-body",
            json.dumps({"modalities": ["text", "audio"]}),
        ]
        args += dataset_args(case, root, Path(work), indices=params["qwen3_omni_workload"].get("smoke_indices"))
        if case.accuracy and case.dataset == "seed-tts":
            args.append("--seed-tts-wer-eval")
        result = run_benchmark(
            args,
            test_name,
            case.concurrency,
            case.dataset,
            case.count,
            resource_label=hardware,
            num_warmups=params.get("num_warmups", max(2, case.concurrency)),
            benchmark_params_name=case.id,
            result_transform=lambda result, code: annotate_result(
                result, case, platform, hardware, smoke=smoke, exit_code=code, enforce_baseline=enforce_baseline
            ),
        )
    assert not result["validation_errors"], "\n".join(result["validation_errors"])
    return result
