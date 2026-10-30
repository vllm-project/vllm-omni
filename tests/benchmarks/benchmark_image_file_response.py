# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark newline-split and fixed-size chunking of image file responses.

Both variants stream the exact payload `ImageGenerationResponse.stream_response()`
produces, so the measurement isolates the response layer: PNG encoding and ZIP
building are shared by the variants and excluded from the timings.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import statistics
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image
from starlette.responses import StreamingResponse

from vllm_omni.entrypoints.openai.protocol.images import (
    _FILE_RESPONSE_CHUNK_SIZE,
    ImageData,
    ImageGenerationResponse,
    _iter_file_chunks,
)


@dataclass(frozen=True)
class BenchmarkVariant:
    label: str
    content: Callable[[bytes], Iterable[bytes] | AsyncIterator[memoryview]]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _encode_png(*, width: int, height: int, seed: int) -> bytes:
    """Encode incompressible noise, so the payload looks like a real photo."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="png")
    return buffer.getvalue()


async def _drain(response: StreamingResponse) -> tuple[bytes, int]:
    """Run a response through ASGI, returning its body and non-empty frame count."""
    body = bytearray()
    frames = 0

    async def send(message: dict[str, object]) -> None:
        nonlocal frames
        if message["type"] == "http.response.body" and message["body"]:
            frames += 1
            body.extend(cast(bytes, message["body"]))

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    scope = {"type": "http", "method": "GET", "asgi": {"spec_version": "2.4"}}
    await response(scope, receive, send)
    return bytes(body), frames


async def _build_payload(*, images: int, width: int, height: int, seed: int) -> bytes:
    png = _encode_png(width=width, height=height, seed=seed)
    response = ImageGenerationResponse(
        created=0,
        data=[ImageData(b64_json=base64.b64encode(png).decode()) for _ in range(images)],
        output_format="png",
        size=f"{width}x{height}",
    )
    payload, _ = await _drain(response.stream_response())
    return payload


async def _measure(variant: BenchmarkVariant, payload: bytes) -> dict[str, object]:
    response = StreamingResponse(variant.content(payload), media_type="application/octet-stream")
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    body, frames = await _drain(response)
    wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000
    process_cpu_ms = (time.process_time_ns() - cpu_start) / 1_000_000
    return {
        "label": variant.label,
        "wall_ms": wall_ms,
        "process_cpu_ms": process_cpu_ms,
        "body_frames": frames,
        "output_bytes": len(body),
        "output_sha256": hashlib.sha256(body).hexdigest(),
    }


def _summarize(records: list[dict[str, object]], label: str) -> dict[str, object]:
    selected = [record for record in records if record["label"] == label]
    wall_values = [cast(float, record["wall_ms"]) for record in selected]
    cpu_values = [cast(float, record["process_cpu_ms"]) for record in selected]
    return {
        "runs": len(selected),
        "body_frames": cast(int, selected[0]["body_frames"]),
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
    parser.add_argument("--images", type=_positive_int, default=1, help="2 or more streams a ZIP archive")
    parser.add_argument("--width", type=_positive_int, default=1024)
    parser.add_argument("--height", type=_positive_int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


async def _run(args: argparse.Namespace) -> dict[str, object]:
    payload = await _build_payload(images=args.images, width=args.width, height=args.height, seed=args.seed)
    baseline = BenchmarkVariant("newline_split", io.BytesIO)
    candidate = BenchmarkVariant("fixed_size_chunks", _iter_file_chunks)

    for _ in range(args.warmup):
        for variant in (baseline, candidate):
            await _measure(variant, payload)

    records: list[dict[str, object]] = []
    for round_index in range(args.rounds):
        order = (baseline, candidate) if round_index % 2 == 0 else (candidate, baseline)
        for variant in order:
            record = await _measure(variant, payload)
            record["round"] = round_index + 1
            records.append(record)

    output_hashes = {str(record["output_sha256"]) for record in records}
    if len(output_hashes) != 1:
        raise RuntimeError(f"benchmark variants produced different outputs: {sorted(output_hashes)}")

    baseline_summary = _summarize(records, baseline.label)
    candidate_summary = _summarize(records, candidate.label)
    baseline_ms = cast(float, cast(dict[str, object], baseline_summary["wall_ms"])["median"])
    candidate_ms = cast(float, cast(dict[str, object], candidate_summary["wall_ms"])["median"])
    return {
        "config": {
            "images": args.images,
            "width": args.width,
            "height": args.height,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "seed": args.seed,
            "chunk_size": _FILE_RESPONSE_CHUNK_SIZE,
            "payload_bytes": len(payload),
            "payload_newline_bytes": payload.count(b"\n"),
        },
        "records": records,
        "summary": {
            baseline.label: baseline_summary,
            candidate.label: candidate_summary,
            "median_wall_speedup": baseline_ms / candidate_ms,
            "body_frame_ratio": (
                cast(int, baseline_summary["body_frames"]) / cast(int, candidate_summary["body_frames"])
            ),
            "output_sha256": output_hashes.pop(),
        },
    }


def main() -> None:
    args = _parse_args()
    result = asyncio.run(_run(args))
    output_json = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(output_json + "\n", encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()
