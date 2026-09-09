# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Evaluate Qwen3-Omni on the SocialOmni paper protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from benchmarks.socialomni.client import RequestResult, request_metrics, run_phase
from benchmarks.socialomni.dataset import (
    SOCIALOMNI_DATASET_ID,
    SOCIALOMNI_DATASET_REVISION,
    SOCIALOMNI_PAPER_CORE_SIZE,
    inspect_socialomni_dataset,
    load_socialomni_level1_samples,
    load_socialomni_level2_samples,
)
from benchmarks.socialomni.metrics import (
    SOCIALOMNI_JUDGE_NAMES,
    JudgeCompletenessError,
    compute_socialomni_level1_metrics,
    compute_socialomni_level2_metrics,
    compute_socialomni_when_metrics,
    validate_judge_scores,
)
from benchmarks.socialomni.protocol import (
    JUDGE_MAX_TOKENS,
    LEVEL1_MAX_TOKENS,
    LEVEL2_RESPONSE_MAX_TOKENS,
    LEVEL2_WHEN_MAX_TOKENS,
    build_level1_result_records,
    load_judge_config,
    make_level1_send_fn,
    run_judges,
    run_level2_model,
    validate_judge_credentials,
)


@dataclass(frozen=True)
class SocialOmniEvalConfig:
    dataset_root: str
    model: str
    base_url: str
    level: Literal["level1", "level2", "both"]
    judge_config: str | None
    prefix_cache_dir: str
    mini: bool
    max_samples: int | None
    max_concurrency: int
    timeout_s: int
    output_dir: str
    warmup: int | None = None


def _request_failure(result: RequestResult, phase: str) -> dict[str, str] | None:
    if result.is_success:
        return None
    return {
        "phase": phase,
        "request_id": result.request_id,
        "error": result.error,
    }


def _judges_complete(records: list[dict[str, Any]], configured: bool) -> bool:
    if not configured:
        return False
    try:
        for record in records:
            if (
                record["gold_when"] == "YES"
                and record["gold_response_success"]
                and str(record["gold_response"]).strip()
            ):
                validate_judge_scores(record["gold_judge_scores"], str(record.get("sample_id", "")))
    except JudgeCompletenessError:
        return False
    return True


async def run_socialomni(config: SocialOmniEvalConfig) -> dict[str, Any]:
    if config.max_concurrency < 1 or config.timeout_s <= 0:
        raise ValueError("max_concurrency and timeout_s must be positive")
    if config.warmup is not None and config.warmup < 0:
        raise ValueError("warmup must be >= 0")
    levels = ("level1", "level2") if config.level == "both" else (config.level,)
    judges = load_judge_config(config.judge_config) if "level2" in levels and config.judge_config else []
    validate_judge_credentials(judges)
    warmup = config.max_concurrency if config.warmup is None else config.warmup
    dataset_identity = inspect_socialomni_dataset(config.dataset_root, levels)
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False
    ).stdout.strip()
    versions = {}
    for package in ("vllm", "vllm-omni", "aiohttp"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    provenance = {"commit": commit or None, "client_versions": versions}
    output: dict[str, Any] = {
        "config": {
            **asdict(config),
            "dataset_root": str(Path(config.dataset_root).resolve()),
            "judge_config": None,
            "dataset_id": SOCIALOMNI_DATASET_ID,
            "expected_dataset_revision": SOCIALOMNI_DATASET_REVISION,
            "warmup_per_nonempty_model_phase": warmup,
            "judge_warmup": 0,
            "generation": {
                "temperature": 0.0,
                "stream": False,
                "level1_max_tokens": LEVEL1_MAX_TOKENS,
                "level2_when_max_tokens": LEVEL2_WHEN_MAX_TOKENS,
                "level2_response_max_tokens": LEVEL2_RESPONSE_MAX_TOKENS,
                "judge_max_tokens": JUDGE_MAX_TOKENS,
            },
        },
        "dataset": dataset_identity,
        "provenance": provenance,
        "summary": {},
        "per_sample": {},
        "failures": [],
    }

    if "level1" in levels:
        samples = load_socialomni_level1_samples(
            config.dataset_root,
            mini=config.mini,
            max_samples=config.max_samples,
        )
        request_results, wall_s = await run_phase(
            samples,
            make_level1_send_fn(config.model, config.base_url),
            max_concurrency=config.max_concurrency,
            timeout_s=config.timeout_s,
            warmup=config.warmup,
        )
        records = build_level1_result_records(samples, request_results)
        for record, result in zip(records, request_results, strict=True):
            failure = _request_failure(result, "level1")
            if failure:
                output["failures"].append(failure)
            elif not record["predicted_answer"]:
                output["failures"].append(
                    {
                        "phase": "level1_parse",
                        "request_id": result.request_id,
                        "error": f"unparsable response: {result.text!r}",
                    }
                )
        output["per_sample"]["level1"] = records
        output["summary"]["level1"] = {
            "metrics": compute_socialomni_level1_metrics(records),
            "speed": request_metrics(request_results, wall_clock_s=wall_s),
        }

    if "level2" in levels:
        samples = load_socialomni_level2_samples(
            config.dataset_root,
            mini=config.mini,
            max_samples=config.max_samples,
        )
        records, model_requests, model_wall_s = await run_level2_model(
            samples,
            model=config.model,
            base_url=config.base_url,
            prefix_cache_dir=config.prefix_cache_dir,
            max_concurrency=config.max_concurrency,
            timeout_s=config.timeout_s,
            warmup=config.warmup,
        )
        for result in model_requests:
            failure = _request_failure(result, "level2_model")
            if failure:
                output["failures"].append(failure)

        output["config"]["judges"] = [asdict(judge) for judge in judges]
        judge_requests = []
        judge_failures: list[dict[str, str]] = []
        judge_wall_s = 0.0
        if judges:
            judge_started = time.perf_counter()
            judge_requests, judge_failures = await run_judges(
                samples,
                records,
                judges,
                timeout_s=config.timeout_s,
            )
            judge_wall_s = time.perf_counter() - judge_started
            output["failures"].extend(judge_failures)
        for record in records:
            if record["when_success"] and not record["predicted_when"]:
                output["failures"].append(
                    {
                        "phase": "level2_parse",
                        "request_id": f"{record['sample_id']}:when",
                        "error": f"unparsable response: {record['when_raw_response']!r}",
                    }
                )

        required_judgments = sum(
            1
            for record in records
            if record["gold_when"] == "YES" and record["gold_response_success"] and str(record["gold_response"]).strip()
        )
        judges_complete = _judges_complete(records, bool(judges))
        complete_metrics = compute_socialomni_level2_metrics(records) if judges_complete else None
        selected = {
            "when": (complete_metrics["when"] if complete_metrics else compute_socialomni_when_metrics(records)),
            "quality": None,
            "judge_status": {
                "complete": judges_complete,
                "eligible_responses": required_judgments,
                "completed_scores": sum(len(record["gold_judge_scores"]) for record in records),
                "required_scores": required_judgments * len(SOCIALOMNI_JUDGE_NAMES),
            },
        }
        if complete_metrics:
            selected["quality"] = complete_metrics["quality"]
            selected["bootstrap"] = complete_metrics["bootstrap"]
            selected["judge_names"] = complete_metrics["judge_names"]
        output["summary"]["level2"] = {
            "metrics": selected,
            "speed": {
                "model": request_metrics(model_requests, wall_clock_s=model_wall_s),
                "judges": request_metrics(judge_requests, wall_clock_s=judge_wall_s),
            },
        }
        if len(records) >= SOCIALOMNI_PAPER_CORE_SIZE:
            core = records[:SOCIALOMNI_PAPER_CORE_SIZE]
            core_complete = _judges_complete(core, bool(judges))
            core_metrics = compute_socialomni_level2_metrics(core) if core_complete else None
            core_summary: dict[str, Any] = {
                "sample_count": len(core),
                "when": (core_metrics["when"] if core_metrics else compute_socialomni_when_metrics(core)),
                "quality": None,
                "judges_complete": core_complete,
            }
            if core_metrics:
                core_summary["quality"] = core_metrics["quality"]
            output["paper_core_200"] = core_summary
        output["per_sample"]["level2"] = records

    level2_complete = "level2" not in levels or (output["summary"]["level2"]["metrics"]["judge_status"]["complete"])
    output["summary"]["status"] = "complete" if level2_complete and not output["failures"] else "incomplete"
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--level", choices=("level1", "level2", "both"), default="both")
    parser.add_argument("--judge-config")
    parser.add_argument("--prefix-cache-dir", default="benchmarks/cache/socialomni-prefixes")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--output-dir", default="benchmarks/results/socialomni")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = SocialOmniEvalConfig(**vars(args))
    output = asyncio.run(run_socialomni(config))
    commit = output["provenance"]["commit"] or "unknown"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    directory = Path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"socialomni-{commit[:12]}-{stamp}-{uuid.uuid4().hex[:8]}.json"
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, allow_nan=False)
        output_file.write("\n")
    print(
        json.dumps(
            {
                "status": output["summary"]["status"],
                "result": str(path),
            }
        )
    )
    if output["summary"]["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
