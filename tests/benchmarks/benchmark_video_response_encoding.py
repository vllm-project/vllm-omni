# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark legacy and automatic MP4 response encoding on synthetic input."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vllm_omni.entrypoints.openai.video_api_utils import (
    _encode_video_bytes,
    _encode_video_bytes_legacy,
)


@dataclass(frozen=True)
class BenchmarkVariant:
    label: str
    encoder: Callable[..., bytes]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _build_inputs(
    *,
    frames: int,
    height: int,
    width: int,
    fps: int,
    audio_sample_rate: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    planar = rng.random((3, frames, height, width), dtype=np.float32)
    video = planar.transpose(1, 2, 3, 0)
    if not all(video[..., channel].flags.c_contiguous for channel in range(3)):
        raise RuntimeError("synthetic video does not expose contiguous channel planes")

    sample_count = max(1, round(frames / fps * audio_sample_rate))
    timeline = np.arange(sample_count, dtype=np.float32) / audio_sample_rate
    left = np.sin(2 * np.pi * 220 * timeline).astype(np.float32, copy=False)
    right = np.sin(2 * np.pi * 330 * timeline).astype(np.float32, copy=False)
    return video, np.stack((left, right))


def _measure(
    variant: BenchmarkVariant,
    *,
    video: np.ndarray,
    audio: np.ndarray,
    fps: int,
    audio_sample_rate: int,
) -> tuple[bytes, dict[str, object]]:
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    output = variant.encoder(
        video,
        fps=fps,
        audio=audio,
        audio_sample_rate=audio_sample_rate,
        video_codec_options={"preset": "ultrafast", "threads": "0"},
    )
    wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000
    process_cpu_ms = (time.process_time_ns() - cpu_start) / 1_000_000
    return output, {
        "label": variant.label,
        "wall_ms": wall_ms,
        "process_cpu_ms": process_cpu_ms,
        "output_bytes": len(output),
        "output_sha256": hashlib.sha256(output).hexdigest(),
    }


def _summarize(records: list[dict[str, object]], label: str) -> dict[str, object]:
    selected = [record for record in records if record["label"] == label]
    wall_values = [float(record["wall_ms"]) for record in selected]
    cpu_values = [float(record["process_cpu_ms"]) for record in selected]
    return {
        "runs": len(selected),
        "wall_ms": {
            "median": statistics.median(wall_values),
            "min": min(wall_values),
            "max": max(wall_values),
        },
        "process_cpu_ms": {
            "median": statistics.median(cpu_values),
            "min": min(cpu_values),
            "max": max(cpu_values),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=_positive_int, default=12)
    parser.add_argument("--height", type=_positive_int, default=96)
    parser.add_argument("--width", type=_positive_int, default=160)
    parser.add_argument("--fps", type=_positive_int, default=24)
    parser.add_argument("--audio-sample-rate", type=_positive_int, default=32000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    video, audio = _build_inputs(
        frames=args.frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        audio_sample_rate=args.audio_sample_rate,
        seed=args.seed,
    )
    baseline = BenchmarkVariant("legacy", _encode_video_bytes_legacy)
    candidate = BenchmarkVariant("automatic", _encode_video_bytes)

    for _ in range(args.warmup):
        for variant in (baseline, candidate):
            _measure(
                variant,
                video=video,
                audio=audio,
                fps=args.fps,
                audio_sample_rate=args.audio_sample_rate,
            )

    records: list[dict[str, object]] = []
    for round_index in range(args.rounds):
        order = (baseline, candidate) if round_index % 2 == 0 else (candidate, baseline)
        for variant in order:
            _, record = _measure(
                variant,
                video=video,
                audio=audio,
                fps=args.fps,
                audio_sample_rate=args.audio_sample_rate,
            )
            record["round"] = round_index + 1
            records.append(record)

    output_hashes = {str(record["output_sha256"]) for record in records}
    if len(output_hashes) != 1:
        raise RuntimeError(f"benchmark variants produced different outputs: {sorted(output_hashes)}")

    paired_savings_ms = []
    paired_savings_percent = []
    for round_index in range(args.rounds):
        pair = [record for record in records if record["round"] == round_index + 1]
        baseline_ms = float(next(record["wall_ms"] for record in pair if record["label"] == baseline.label))
        candidate_ms = float(next(record["wall_ms"] for record in pair if record["label"] == candidate.label))
        paired_savings_ms.append(baseline_ms - candidate_ms)
        paired_savings_percent.append((baseline_ms - candidate_ms) / baseline_ms * 100)

    result = {
        "config": {
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "audio_sample_rate": args.audio_sample_rate,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "seed": args.seed,
            "video_shape": list(video.shape),
            "video_strides": list(video.strides),
            "codec_options": {"preset": "ultrafast", "threads": "0"},
        },
        "records": records,
        "summary": {
            baseline.label: _summarize(records, baseline.label),
            candidate.label: _summarize(records, candidate.label),
            "paired_wall_saving_ms": {
                "median": statistics.median(paired_savings_ms),
                "min": min(paired_savings_ms),
                "max": max(paired_savings_ms),
            },
            "paired_wall_saving_percent": {
                "median": statistics.median(paired_savings_percent),
                "min": min(paired_savings_percent),
                "max": max(paired_savings_percent),
            },
            "output_sha256": output_hashes.pop(),
        },
    }
    output_json = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(output_json + "\n", encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()
