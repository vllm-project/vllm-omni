# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sequential image-serving benchmark using the existing chat request backend.

Stage names are explicit arguments: no model-specific timing or request logic.
Run with ``python -m benchmarks.diffusion.bench_stage_latency --help``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from benchmarks.diffusion.backends import RequestFuncOutput


class LatencyRecord(TypedDict):
    request: int
    text_seconds: float
    generation_seconds: float
    end_to_end_seconds: float
    stage_durations_seconds: dict[str, float]


def json_object(value: str) -> dict[str, object]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return parsed


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=positive_int, required=True)
    parser.add_argument("--height", type=positive_int, required=True)
    parser.add_argument("--num-inference-steps", type=positive_int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--timeout", type=positive_int, default=1800)
    parser.add_argument("--text-stage", required=True, help="Exact server stage key; duration must be in seconds")
    parser.add_argument("--generation-stage", required=True, help="Exact server stage key; duration must be in seconds")
    parser.add_argument("--extra-body", type=json_object, default={})
    parser.add_argument(
        "--metadata", type=json_object, default={}, help="Declared server hardware/software/run metadata"
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser


def make_record(index: int, output: RequestFuncOutput, text_stage: str, generation_stage: str) -> LatencyRecord:
    if not output.success:
        raise RuntimeError(f"Request {index} failed: {output.error}")
    values = {}
    for label, key in (("text_seconds", text_stage), ("generation_seconds", generation_stage)):
        if key not in output.stage_durations:
            raise ValueError(f"Missing stage {key!r}; enable pipeline profiling on the server")
        duration = float(output.stage_durations[key])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid duration for {key!r}: {duration}")
        values[label] = duration
    if not math.isfinite(output.latency) or output.latency <= 0:
        raise ValueError(f"Invalid end-to-end latency: {output.latency}")
    return {
        "request": index,
        "text_seconds": values["text_seconds"],
        "generation_seconds": values["generation_seconds"],
        "end_to_end_seconds": output.latency,
        "stage_durations_seconds": dict(output.stage_durations),
    }


def save_results(prefix: Path, config: dict[str, object], records: list[LatencyRecord]) -> None:
    fields: tuple[Literal["text_seconds", "generation_seconds", "end_to_end_seconds"], ...] = (
        "text_seconds",
        "generation_seconds",
        "end_to_end_seconds",
    )
    summary = (
        {
            field: {
                "mean": statistics.mean(row[field] for row in records),
                "median": statistics.median(row[field] for row in records),
            }
            for field in fields
        }
        if records
        else {}
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps({"configuration": config, "records": records, "summary_seconds": summary}, indent=2) + "\n"
    )
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["request", *fields])
        writer.writeheader()
        for row in records:
            writer.writerow({"request": row["request"], **{key: row[key] for key in fields}})


async def run(args: argparse.Namespace) -> None:
    import aiohttp

    from benchmarks.diffusion.backends import RequestFuncInput, async_request_chat_completions

    if args.extra_body.get("num_outputs_per_prompt", 1) != 1:
        raise ValueError("This benchmark requires num_outputs_per_prompt=1")
    extra_body = dict(args.extra_body)
    extra_body.update(return_stage_metrics=True, num_outputs_per_prompt=1)
    # The existing request backend uses setdefault for these keys. Reject
    # conflicting overrides rather than silently benchmark another workload.
    for key in ("width", "height", "num_inference_steps", "seed"):
        if key in extra_body and extra_body[key] != getattr(args, key):
            raise ValueError(f"--extra-body {key} conflicts with its explicit CLI argument")
    request = RequestFuncInput(
        prompt=args.prompt,
        api_url=args.base_url.rstrip("/") + "/v1/chat/completions",
        model=args.model,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        extra_body=extra_body,
    )
    config = {**vars(args), "output_prefix": str(args.output_prefix), "batch_size": 1, "concurrency": 1}
    config["extra_body"] = extra_body
    records: list[LatencyRecord] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        for index in range(-args.warmup, args.repeats):
            output = await async_request_chat_completions(request, session)
            row = make_record(index, output, args.text_stage, args.generation_stage)
            print(f"{'warmup' if index < 0 else 'measured'} {index}: {row}", flush=True)
            if index >= 0:
                records.append(row)
                # Preserve completed measurements if a subsequent request fails.
                save_results(args.output_prefix, config, records)


if __name__ == "__main__":
    asyncio.run(run(build_parser().parse_args()))
