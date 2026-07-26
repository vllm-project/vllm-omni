#!/usr/bin/env python3
"""Run warmup and concurrent MiniCPM-o streaming proxy benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .client import build_payload, metric_summary, run_stream_request


def _host_memory() -> dict[str, int]:
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {}
    return values


async def _npu_snapshot() -> dict[str, Any]:
    if shutil.which("npu-smi") is None:
        return {"error": "npu-smi not found"}
    try:
        process = await asyncio.create_subprocess_exec(
            "npu-smi",
            "info",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
    except (OSError, asyncio.TimeoutError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "returncode": process.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }


async def _monitor_resources(stop: asyncio.Event, output: Path, interval: float) -> None:
    samples = []
    while True:
        samples.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "host_memory_bytes": _host_memory(),
                "npu_smi": await _npu_snapshot(),
            }
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass
    output.write_text(json.dumps({"samples": samples}, indent=2) + "\n", encoding="utf-8")


async def _run_configuration(
    args: argparse.Namespace,
    *,
    mode: str,
    concurrency: int,
    output_dir: Path,
) -> dict[str, Any]:
    with_audio = mode == "text_audio"
    payload = build_payload(
        model=args.model,
        prompt=args.prompt,
        input_modality=args.input_modality,
        media=args.media,
        with_audio=with_audio,
        seed=args.seed,
        thinker_max_tokens=args.thinker_max_tokens,
        talker_max_tokens=args.talker_max_tokens,
    )
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout, headers={"Authorization": "Bearer EMPTY"}) as client:
        warmups = []
        for index in range(args.warmups):
            warmups.append(
                await run_stream_request(
                    client,
                    endpoint=f"{args.base_url.rstrip('/')}/chat/completions",
                    payload=payload,
                    request_name=f"warmup-{mode}-{index}",
                    input_modality=args.input_modality,
                    with_audio=with_audio,
                )
            )
        if any(not record["success"] for record in warmups):
            return {"mode": mode, "concurrency": concurrency, "warmups": warmups, "records": [], "aborted": True}

        semaphore = asyncio.Semaphore(concurrency)

        async def one(index: int) -> dict[str, Any]:
            async with semaphore:
                return await run_stream_request(
                    client,
                    endpoint=f"{args.base_url.rstrip('/')}/chat/completions",
                    payload=payload,
                    request_name=f"measure-{mode}-c{concurrency}-{index}",
                    input_modality=args.input_modality,
                    with_audio=with_audio,
                    output_wav=output_dir / "audio" / f"request-{index:04d}.wav" if with_audio else None,
                )

        stop = asyncio.Event()
        monitor = asyncio.create_task(_monitor_resources(stop, output_dir / "resources.json", args.resource_interval))
        started = time.perf_counter()
        try:
            records = await asyncio.gather(*(one(index) for index in range(args.num_requests)))
        finally:
            duration = time.perf_counter() - started
            stop.set()
            await monitor

    summary = metric_summary(records)
    successful = [record for record in records if record["success"]]
    summary["wall_time_s"] = duration
    summary["request_throughput_per_s"] = len(successful) / duration if duration else None
    audio_seconds = sum(
        record["audio"]["pcm_bytes"]
        / (record["audio"]["sample_rate_hz"] * record["audio"]["channels"] * record["audio"]["sample_width_bytes"])
        for record in successful
        if record["audio"]["pcm_bytes"] and record["audio"]["sample_rate_hz"]
    )
    summary["generated_audio_seconds"] = audio_seconds
    summary["audio_seconds_throughput"] = audio_seconds / duration if duration else None
    return {
        "mode": mode,
        "concurrency": concurrency,
        "warmups": warmups,
        "records": records,
        "summary": summary,
        "aborted": False,
    }


async def _main(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configurations = []
    for mode in args.modes:
        for concurrency in args.concurrency:
            current_dir = args.output_dir / f"{mode}_c{concurrency}"
            current_dir.mkdir(parents=True, exist_ok=True)
            result = await _run_configuration(
                args,
                mode=mode,
                concurrency=concurrency,
                output_dir=current_dir,
            )
            configurations.append(result)
            (current_dir / "raw_results.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            status = "ABORT" if result["aborted"] else ("PASS" if result["summary"]["failed_requests"] == 0 else "FAIL")
            print(f"{mode} concurrency={concurrency}: {status}")

    result = {
        "schema_version": 1,
        "metric_scope": "local_proxy",
        "formal_score": None,
        "official_metric_definitions": "UNRESOLVED",
        "profiler_run": False,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "model_revision": os.environ.get("MINICPMO_MODEL_REVISION", "UNRESOLVED"),
        },
        "configurations": configurations,
    }
    result_path = args.output_dir / "benchmark_results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result_path)
    failed = any(
        config["aborted"] or config.get("summary", {}).get("failed_requests", 1) > 0 for config in configurations
    )
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8099/v1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--input-modality", choices=["text", "image", "audio", "video"], default="text")
    parser.add_argument("--media", help="Required for non-text input modalities")
    parser.add_argument("--prompt", default="Introduce vLLM-Omni in one short sentence.")
    parser.add_argument("--modes", nargs="+", choices=["text", "text_audio"], default=["text", "text_audio"])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--resource-interval", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinker-max-tokens", type=int, default=256)
    parser.add_argument("--talker-max-tokens", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(value < 1 for value in args.concurrency):
        parser.error("all concurrency values must be >= 1")
    if args.num_requests < 1 or args.warmups < 0:
        parser.error("num-requests must be >= 1 and warmups must be >= 0")
    if args.input_modality != "text" and not args.media:
        parser.error("--media is required for non-text input modalities")
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
