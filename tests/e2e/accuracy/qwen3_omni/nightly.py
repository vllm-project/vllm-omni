# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reproducible, combined accuracy/performance workloads for Qwen3-Omni nightly."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


class DatasetSpec(TypedDict):
    repo: str
    revision: str
    manifest: str
    sha256: str
    count: int


MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MANIFEST = Path(__file__).with_name("nightly_baselines.json")
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
class NightlyCase:
    config: str
    dataset: str
    concurrency: int
    count: int

    @property
    def accuracy(self) -> bool:
        return self.concurrency == 64

    @property
    def id(self) -> str:
        return f"{self.config}-{self.dataset}-c{self.concurrency}"

    @property
    def metrics(self) -> tuple[str, ...]:
        audio = ("mean_audio_rtf", "mean_audio_ttfp_ms")
        return ("mean_ttft_ms", *audio) if self.dataset == "daily-omni" else audio


def nightly_cases(platform: str) -> list[NightlyCase]:
    if platform not in ("cuda", "npu"):
        raise ValueError(f"Unsupported nightly platform: {platform}")
    sweep = [(1, 32), (16, 256), (32, 512)] if platform == "cuda" else [(1, 32), (4, 64), (8, 128)]
    return [
        NightlyCase(config, dataset, concurrency, count)
        for config in ("default", "async-prefix")
        for dataset, source in DATASETS.items()
        for concurrency, count in ([(64, source["count"])] + (sweep if config == "async-prefix" else []))
    ]


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


def dataset_args(case: NightlyCase, root: Path, work: Path) -> list[str]:
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
        qa.write_text(json.dumps(rows[: case.count], ensure_ascii=False), encoding="utf-8")
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


def quality_baseline(case: NightlyCase, platform: str) -> dict[str, float]:
    if not case.accuracy:
        return {}
    if case.dataset == "daily-omni":
        return {"min_daily_omni_accuracy": 0.69}
    limits = {"max_seed_tts_mean_wer": 0.05, "min_seed_tts_mean_sim": 0.75}
    if platform == "cuda":
        limits["min_seed_tts_mean_utmos"] = 3.10
    return limits


def performance_gate(result: dict, case: NightlyCase, hardware: str, manifest: dict, mode: str) -> list[str]:
    """Collect until calibrated; enforce rejects missing or incompatible references."""
    if mode not in ("collect", "enforce"):
        raise ValueError(f"Unknown baseline mode: {mode}")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported nightly baseline schema")
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
    result["performance_baseline_status"] = "collecting" if mode == "collect" else "enforced"
    result["performance_baseline"] = entry
    if mode == "collect":
        return errors
    if not isinstance(entry, dict) or not entry:
        return errors + [f"No calibrated baseline for {hardware}/{case.id}"]
    if entry.get("dataset_revision") != DATASETS[case.dataset]["revision"] or entry.get("num_prompts") != case.count:
        return errors + ["Performance baseline workload does not match the pinned dataset/count"]
    repetitions = entry.get("repetitions")
    if not entry.get("reference_commit") or not isinstance(repetitions, int) or repetitions < 2:
        return errors + ["Baseline must identify its reference commit and repeated measurements"]
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
    return errors


def accuracy_coverage_errors(result: dict, case: NightlyCase, platform: str) -> list[str]:
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
    return [f"{key}={result.get(key)!r}, expected {case.count}" for key in counts if result.get(key) != case.count]
