# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark event-loop responsiveness for inline and offloaded MP4 encoding."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from benchmark_video_response_encoding import _build_inputs, _positive_int
from vllm.engine.protocol import EngineClient

from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes


@dataclass(frozen=True)
class SchedulingVariant:
    label: str
    offload: bool


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


async def _measure(
    variant: SchedulingVariant,
    *,
    handler: OmniOpenAIServingVideo,
    video: np.ndarray,
    audio: np.ndarray,
    fps: int,
    audio_sample_rate: int,
    heartbeat_interval_s: float,
) -> tuple[bytes, dict[str, object]]:
    heartbeat_times: list[float] = []
    stop_heartbeat = asyncio.Event()

    async def heartbeat() -> None:
        heartbeat_times.append(time.perf_counter())
        while not stop_heartbeat.is_set():
            await asyncio.sleep(heartbeat_interval_s)
            heartbeat_times.append(time.perf_counter())

    def encode() -> bytes:
        return _encode_video_bytes(
            video,
            fps=fps,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            video_codec_options={"preset": "ultrafast", "threads": "0"},
        )

    heartbeat_task = asyncio.create_task(heartbeat())
    await asyncio.sleep(2 * heartbeat_interval_s)
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    try:
        output = await handler._run_video_response_encoding(encode) if variant.offload else encode()
        wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000
        process_cpu_ms = (time.process_time_ns() - cpu_start) / 1_000_000
    finally:
        await asyncio.sleep(2 * heartbeat_interval_s)
        stop_heartbeat.set()
        await heartbeat_task

    heartbeat_gaps_ms = [(current - previous) * 1000 for previous, current in zip(heartbeat_times, heartbeat_times[1:])]
    return output, {
        "label": variant.label,
        "wall_ms": wall_ms,
        "process_cpu_ms": process_cpu_ms,
        "max_heartbeat_gap_ms": max(heartbeat_gaps_ms),
        "p99_heartbeat_gap_ms": float(np.percentile(heartbeat_gaps_ms, 99)),
        "heartbeat_samples": len(heartbeat_times),
        "output_bytes": len(output),
        "output_sha256": hashlib.sha256(output).hexdigest(),
    }


def _summarize(records: list[dict[str, object]], label: str) -> dict[str, object]:
    selected = [record for record in records if record["label"] == label]

    def distribution(field: str) -> dict[str, float]:
        values = [float(cast(int | float, record[field])) for record in selected]
        return {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "runs": len(selected),
        "wall_ms": distribution("wall_ms"),
        "process_cpu_ms": distribution("process_cpu_ms"),
        "max_heartbeat_gap_ms": distribution("max_heartbeat_gap_ms"),
        "p99_heartbeat_gap_ms": distribution("p99_heartbeat_gap_ms"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=_positive_int, default=12)
    parser.add_argument("--height", type=_positive_int, default=96)
    parser.add_argument("--width", type=_positive_int, default=160)
    parser.add_argument("--fps", type=_positive_int, default=24)
    parser.add_argument("--audio-sample-rate", type=_positive_int, default=32000)
    parser.add_argument("--heartbeat-interval-ms", type=_positive_float, default=5.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=_positive_int, default=5)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


async def _run(args: argparse.Namespace) -> dict[str, object]:
    video, audio = _build_inputs(
        frames=args.frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        audio_sample_rate=args.audio_sample_rate,
        seed=args.seed,
    )
    inline = SchedulingVariant("inline", offload=False)
    executor = SchedulingVariant("dedicated_executor", offload=True)
    heartbeat_interval_s = args.heartbeat_interval_ms / 1000
    handler = OmniOpenAIServingVideo(cast(EngineClient, object()), model_name="benchmark")
    try:
        for _ in range(args.warmup):
            for variant in (inline, executor):
                await _measure(
                    variant,
                    handler=handler,
                    video=video,
                    audio=audio,
                    fps=args.fps,
                    audio_sample_rate=args.audio_sample_rate,
                    heartbeat_interval_s=heartbeat_interval_s,
                )

        records: list[dict[str, object]] = []
        for round_index in range(args.rounds):
            order = (inline, executor) if round_index % 2 == 0 else (executor, inline)
            for variant in order:
                _, record = await _measure(
                    variant,
                    handler=handler,
                    video=video,
                    audio=audio,
                    fps=args.fps,
                    audio_sample_rate=args.audio_sample_rate,
                    heartbeat_interval_s=heartbeat_interval_s,
                )
                record["round"] = round_index + 1
                records.append(record)
    finally:
        handler.shutdown()

    output_hashes = {str(record["output_sha256"]) for record in records}
    if len(output_hashes) != 1:
        raise RuntimeError(f"benchmark variants produced different outputs: {sorted(output_hashes)}")
    return {
        "config": {
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "audio_sample_rate": args.audio_sample_rate,
            "heartbeat_interval_ms": args.heartbeat_interval_ms,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "seed": args.seed,
            "video_shape": list(video.shape),
            "video_strides": list(video.strides),
            "codec_options": {"preset": "ultrafast", "threads": "0"},
        },
        "records": records,
        "summary": {
            inline.label: _summarize(records, inline.label),
            executor.label: _summarize(records, executor.label),
            "output_sha256": output_hashes.pop(),
        },
    }


def main() -> None:
    args = _parse_args()
    output_json = json.dumps(asyncio.run(_run(args)), indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(output_json + "\n", encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()
