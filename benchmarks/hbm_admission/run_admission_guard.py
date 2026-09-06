#!/usr/bin/env python3
"""A/B stress test for the per-stage HBM admission guard on one GPU.

Both arms use the same hard KV-cache budget.  A first wave occupies the KV
blocks and a delayed second wave supplies waiting work while the pool is full.
The only changed setting is ``hbm_admission_guard``.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import yaml
from run_matrix import (
    DEFAULT_DEPLOY,
    DEFAULT_MODEL,
    parse_gpu_metrics,
    start_gpu_monitor,
    start_server,
    stop_process_group,
    wait_for_gpu_release,
    write_json,
)

PAUSE_RE = re.compile(
    r"\[HBMAdmission\] paused free_blocks=(\d+) total_blocks=(\S+) "
    r"waiting=(\d+) running=(\d+) deferred_steps=(\d+)"
)
RESUME_RE = re.compile(
    r"\[HBMAdmission\] resumed free_blocks=(\d+) waiting=(\d+) running=(\d+) "
    r"deferred_steps=(\d+) deferred_requests=(\d+) resume_count=(\d+)"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def parse_stage_budgets(values: list[str]) -> dict[int, float]:
    result: dict[int, float] = {}
    for value in values:
        try:
            stage_text, budget_text = value.split("=", 1)
            stage_id, budget = int(stage_text), float(budget_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid stage budget {value!r}; expected STAGE_ID=GB") from exc
        if budget <= 0:
            raise argparse.ArgumentTypeError(f"stage budget must be positive: {value!r}")
        result[stage_id] = budget
    return result


def prepare_config(
    output_dir: Path,
    source_deploy: Path,
    stage_ids: list[int] | None,
    target_stage: int,
    other_stage_budgets: dict[int, float],
    max_model_len: int,
    max_output_tokens: int,
    max_num_seqs: int | None,
    disable_prefix_caching: bool,
    budget_gb: float,
    guard: bool,
) -> Path:
    config = yaml.safe_load(source_deploy.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("stages"), list):
        raise ValueError(f"deploy config has no stages list: {source_deploy}")
    config = deepcopy(config)
    if stage_ids is not None:
        selected = set(stage_ids)
        config["stages"] = [stage for stage in config["stages"] if stage.get("stage_id") in selected]
        missing = selected - {stage["stage_id"] for stage in config["stages"]}
        if missing:
            raise ValueError(f"selected stages are absent from {source_deploy}: {sorted(missing)}")
        if len(config["stages"]) == 1:
            config["async_chunk"] = False
        # Keep platform-specific overrides consistent with the top-level stage
        # selection. This lets one benchmark definition target a full pipeline
        # or a supported sub-pipeline without retaining stale stage overrides.
        for platform in (config.get("platforms") or {}).values():
            if isinstance(platform, dict) and isinstance(platform.get("stages"), list):
                platform["stages"] = [stage for stage in platform["stages"] if stage.get("stage_id") in selected]
    for stage in config["stages"]:
        stage.pop("hbm_limit_gb", None)
        stage.pop("hbm_admission_guard", None)
    by_id = {stage["stage_id"]: stage for stage in config["stages"]}
    if target_stage not in by_id:
        raise ValueError(f"target stage {target_stage} is absent from {source_deploy}")
    target = by_id[target_stage]
    target["hbm_limit_gb"] = budget_gb
    target["hbm_admission_guard"] = guard
    # A tiny budget cannot satisfy vLLM's startup guarantee for the production
    # 4096-token maximum.  A 512-token experiment maximum keeps the startup
    # check meaningful while letting a modest request wave fill the pool.
    target["max_model_len"] = max_model_len
    if max_num_seqs is not None:
        target["max_num_seqs"] = max_num_seqs
    if disable_prefix_caching:
        target["enable_prefix_caching"] = False
    sampling = target.get("default_sampling_params")
    if isinstance(sampling, dict):
        sampling["max_tokens"] = max_output_tokens
    for stage_id, stage_budget in other_stage_budgets.items():
        if stage_id == target_stage:
            raise ValueError("other-stage-budget cannot refer to the target stage")
        if stage_id not in by_id:
            raise ValueError(f"secondary stage {stage_id} is absent from {source_deploy}")
        by_id[stage_id]["hbm_limit_gb"] = stage_budget
    path = output_dir / "configs" / f"budget_{budget_gb:g}_guard_{str(guard).lower()}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    request_id: str,
    wave: str,
    text: str,
    release_at: float,
    model: str,
    request_profile: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    await asyncio.sleep(max(0.0, release_at - time.monotonic()))
    started = time.monotonic()
    result: dict[str, Any] = {
        "request_id": request_id,
        "wave": wave,
        "started_offset_s": started - release_at,
        "input_chars": len(text),
    }
    if request_profile == "covo-chat":
        from vllm_omni.model_executor.models.covo_audio.prompt_utils import (
            COVO_AUDIO_SYSTEM_PROMPT,
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": COVO_AUDIO_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "sampling_params_list": [
                {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": max_output_tokens,
                    "seed": 42,
                    "detokenize": True,
                    "repetition_penalty": 1.05,
                    "stop_token_ids": [151645],
                    "ignore_eos": True,
                },
                {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 2048,
                    "seed": 42,
                    "detokenize": False,
                    "repetition_penalty": 1.05,
                },
            ],
        }
    elif request_profile == "qwen3-omni-chat":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "modalities": ["text"],
            "stream": False,
            "temperature": 0.0,
            "max_tokens": max_output_tokens,
            "ignore_eos": True,
        }
    else:
        payload = {
            "model": model,
            "input": text,
            "voice": "Vivian",
            "language": "English",
            "task_type": "CustomVoice",
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
        }
    first_byte: float | None = None
    received = 0
    try:
        async with session.post(url, json=payload) as response:
            result["http_status"] = response.status
            async for chunk in response.content.iter_chunked(64 * 1024):
                if chunk and first_byte is None:
                    first_byte = time.monotonic()
                received += len(chunk)
            if response.status >= 400:
                result["error"] = f"HTTP {response.status}"
    except Exception as exc:
        result["http_status"] = None
        result["error"] = f"{type(exc).__name__}: {exc}"
    ended = time.monotonic()
    result.update(
        {
            "first_byte_latency_s": None if first_byte is None else first_byte - started,
            "end_to_end_latency_s": ended - started,
            "response_bytes": received,
            "success": result.get("http_status") == 200 and received > 0,
        }
    )
    return result


async def run_two_waves(
    port: int,
    saturation_requests: int,
    probe_requests: int,
    probe_delay_s: float,
    timeout_s: float,
    model: str,
    request_profile: str,
    max_output_tokens: int,
) -> tuple[list[dict[str, Any]], float]:
    long_text = (
        "The admission controller experiment deliberately keeps this generation "
        "request active for long enough to occupy key value cache blocks. "
    ) * 8
    short_text = "This delayed probe measures admission while the cache is full."
    start = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        endpoint = "/v1/chat/completions" if request_profile in {"covo-chat", "qwen3-omni-chat"} else "/v1/audio/speech"
        jobs = [
            one_request(
                session,
                f"http://127.0.0.1:{port}{endpoint}",
                f"saturation-{index:03d}",
                "saturation",
                long_text,
                start,
                model,
                request_profile,
                max_output_tokens,
            )
            for index in range(saturation_requests)
        ]
        jobs += [
            one_request(
                session,
                f"http://127.0.0.1:{port}{endpoint}",
                f"probe-{index:03d}",
                "probe",
                short_text,
                start + probe_delay_s,
                model,
                request_profile,
                max_output_tokens,
            )
            for index in range(probe_requests)
        ]
        results = await asyncio.gather(*jobs)
    return results, time.monotonic() - start


def parse_scheduler_events(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(errors="replace")
    pauses = [
        {
            "free_blocks": int(m.group(1)),
            "total_blocks": m.group(2),
            "waiting": int(m.group(3)),
            "running": int(m.group(4)),
            "deferred_steps": int(m.group(5)),
        }
        for m in PAUSE_RE.finditer(text)
    ]
    resumes = [
        {
            "free_blocks": int(m.group(1)),
            "waiting": int(m.group(2)),
            "running": int(m.group(3)),
            "deferred_steps": int(m.group(4)),
            "deferred_requests": int(m.group(5)),
            "resume_count": int(m.group(6)),
        }
        for m in RESUME_RE.finditer(text)
    ]
    lowered = text.lower()
    return {
        "pause_count": len(pauses),
        "resume_count": len(resumes),
        "pauses": pauses,
        "resumes": resumes,
        "final_deferred_steps": resumes[-1]["deferred_steps"] if resumes else 0,
        "final_deferred_requests": resumes[-1]["deferred_requests"] if resumes else 0,
        "preemption_log_count": lowered.count("preempt"),
        "oom_log_count": lowered.count("out of memory") + lowered.count("cuda oom"),
    }


def summarize_requests(results: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    def metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
        e2e = [r["end_to_end_latency_s"] for r in selected if r["success"]]
        ttft = [r["first_byte_latency_s"] for r in selected if r["success"]]
        return {
            "requests": len(selected),
            "successful": sum(r["success"] for r in selected),
            "failed": sum(not r["success"] for r in selected),
            "e2e_s": {
                "mean": statistics.fmean(e2e) if e2e else None,
                "p50": percentile(e2e, 0.50),
                "p95": percentile(e2e, 0.95),
                "max": max(e2e) if e2e else None,
            },
            "first_byte_s": {
                "mean": statistics.fmean(ttft) if ttft else None,
                "p50": percentile(ttft, 0.50),
                "p95": percentile(ttft, 0.95),
                "max": max(ttft) if ttft else None,
            },
        }

    return {
        "duration_s": duration_s,
        "throughput_requests_per_s": sum(r["success"] for r in results) / duration_s,
        "all": metrics(results),
        "saturation": metrics([r for r in results if r["wave"] == "saturation"]),
        "probe": metrics([r for r in results if r["wave"] == "probe"]),
    }


def run_arm(args: argparse.Namespace, output_dir: Path, budget: float, guard: bool) -> dict[str, Any]:
    arm_name = f"budget_{budget:g}_guard_{str(guard).lower()}"
    arm_dir = output_dir / "arms" / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    config = prepare_config(
        output_dir,
        args.deploy_config,
        args.stage_ids,
        args.target_stage,
        args.other_stage_budgets,
        args.max_model_len,
        args.max_output_tokens,
        args.max_num_seqs,
        args.disable_prefix_caching,
        budget,
        guard,
    )
    server_log = arm_dir / "server.log"
    gpu_csv = arm_dir / "gpu.csv"
    proc = log_file = monitor = monitor_file = None
    started_at = utc_now()
    try:
        wait_for_gpu_release()
        proc, log_file = start_server(
            Path(__file__).resolve().parents[2],
            args.model,
            config,
            server_log,
            args.startup_timeout,
            args.safetensors_load_strategy,
        )
        monitor, monitor_file = start_gpu_monitor(gpu_csv)
        results, duration = asyncio.run(
            run_two_waves(
                8000,
                args.saturation_requests,
                args.probe_requests,
                args.probe_delay,
                args.request_timeout,
                args.model,
                args.request_profile,
                args.max_output_tokens,
            )
        )
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        if monitor_file is not None:
            monitor_file.close()
        stop_process_group(proc)
        if log_file is not None:
            log_file.close()
        wait_for_gpu_release()

    write_json(arm_dir / "requests.json", results)
    events = parse_scheduler_events(server_log)
    write_json(arm_dir / "scheduler_events.json", events)
    summary = {
        "arm": arm_name,
        "budget_gb": budget,
        "guard_enabled": guard,
        "started_at": started_at,
        "finished_at": utc_now(),
        "requests": summarize_requests(results, duration),
        "scheduler": events,
        "gpu": parse_gpu_metrics(gpu_csv),
        "artifacts": {
            "config": str(config),
            "server_log": str(server_log),
            "gpu_csv": str(gpu_csv),
            "requests_json": str(arm_dir / "requests.json"),
        },
    }
    write_json(arm_dir / "summary.json", summary)
    return summary


def comparison(off: dict[str, Any], on: dict[str, Any]) -> dict[str, Any]:
    def value(item: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            item = item[key]
        return item

    return {
        "guard_observed": value(on, "scheduler", "pause_count") > 0,
        "pause_count_delta": value(on, "scheduler", "pause_count") - value(off, "scheduler", "pause_count"),
        "preemption_log_count_delta": value(on, "scheduler", "preemption_log_count")
        - value(off, "scheduler", "preemption_log_count"),
        "probe_p95_first_byte_delta_s": value(on, "requests", "probe", "first_byte_s", "p95")
        - value(off, "requests", "probe", "first_byte_s", "p95"),
        "probe_p95_e2e_delta_s": value(on, "requests", "probe", "e2e_s", "p95")
        - value(off, "requests", "probe", "e2e_s", "p95"),
        "throughput_delta_requests_per_s": value(on, "requests", "throughput_requests_per_s")
        - value(off, "requests", "throughput_requests_per_s"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument(
        "--request-profile",
        choices=("qwen3-tts", "covo-chat", "qwen3-omni-chat"),
        default="qwen3-tts",
        help="Select the endpoint and request schema used by the load generator.",
    )
    parser.add_argument(
        "--stage-ids",
        type=int,
        nargs="+",
        help="Only launch these deploy stages (for example, 0 for Thinker-only).",
    )
    parser.add_argument("--target-stage", type=int, default=0)
    parser.add_argument(
        "--other-stage-budget",
        action="append",
        default=[],
        metavar="STAGE_ID=GB",
        help="Optional fixed KV budget for a non-target stage; repeat as needed.",
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        help="Override target-stage concurrency so the stress wave can fill its KV pool.",
    )
    parser.add_argument(
        "--disable-prefix-caching",
        action="store_true",
        help="Prevent shared prompts from reducing the independent KV pressure.",
    )
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.125, 0.0625])
    parser.add_argument("--saturation-requests", type=int, default=16)
    parser.add_argument("--probe-requests", type=int, default=32)
    parser.add_argument("--probe-delay", type=float, default=0.5)
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument(
        "--safetensors-load-strategy",
        choices=("lazy", "eager", "prefetch", "torchao"),
        default="prefetch",
        help=(
            "Checkpoint loading strategy. Use lazy when the checkpoint is larger "
            "than the container memory limit to avoid prefetch cache thrashing."
        ),
    )
    args = parser.parse_args()
    args.other_stage_budgets = parse_stage_budgets(args.other_stage_budget)
    if (
        args.max_model_len <= 0
        or args.max_output_tokens <= 0
        or (args.max_num_seqs is not None and args.max_num_seqs <= 0)
    ):
        parser.error("length and max-num-seqs values must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "started_at": utc_now(),
        "model": args.model,
        "deploy_config": str(args.deploy_config.resolve()),
        "request_profile": args.request_profile,
        "stage_ids": args.stage_ids,
        "target_stage": args.target_stage,
        "other_stage_budgets_gb": args.other_stage_budgets,
        "max_model_len": args.max_model_len,
        "max_output_tokens": args.max_output_tokens,
        "max_num_seqs": args.max_num_seqs,
        "prefix_caching_disabled": args.disable_prefix_caching,
        "budgets_gb": args.budgets,
        "saturation_requests": args.saturation_requests,
        "probe_requests": args.probe_requests,
        "probe_delay_s": args.probe_delay,
        "safetensors_load_strategy": args.safetensors_load_strategy,
        "method": "same hard KV budget; guard off/on; delayed two-wave load",
    }
    write_json(args.output_dir / "metadata.json", metadata)
    attempts: list[dict[str, Any]] = []
    for budget in args.budgets:
        off = run_arm(args, args.output_dir, budget, False)
        on = run_arm(args, args.output_dir, budget, True)
        item = {"budget_gb": budget, "guard_off": off, "guard_on": on}
        item["comparison"] = comparison(off, on)
        attempts.append(item)
        write_json(args.output_dir / "summary.json", {"metadata": metadata, "attempts": attempts})
        if item["comparison"]["guard_observed"]:
            break
    metadata["finished_at"] = utc_now()
    metadata["guard_observed"] = any(a["comparison"]["guard_observed"] for a in attempts)
    write_json(args.output_dir / "summary.json", {"metadata": metadata, "attempts": attempts})
    if not metadata["guard_observed"]:
        raise SystemExit("guard did not trigger at the tested budgets; inspect summary.json")


if __name__ == "__main__":
    main()
