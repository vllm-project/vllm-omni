"""Measure MiniCPM-o 4.5 multimodal TTFT and server GPU memory.

The media file is encoded before timing. TTFT is the time from locally posting a
streaming chat-completions request to its first non-terminal SSE data event.
The benchmark sends requests sequentially because the default MiniCPM-o 4.5
two-GPU deployment has ``max_num_seqs: 1`` for both stages.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import requests


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile for no values.")
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _data_url(path: Path, media_type: str) -> str:
    mime_type = {
        "image": "image/png" if path.suffix.lower() == ".png" else "image/jpeg",
        "video": "video/mp4",
    }[media_type]
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


class GpuMemorySampler:
    def __init__(self, interval_seconds: float) -> None:
        self._interval_seconds = interval_seconds
        self._samples: list[list[int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> GpuMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def samples(self) -> list[list[int]]:
        return self._samples

    def _sample(self) -> None:
        while not self._stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self._samples.append([int(line.strip()) for line in result.stdout.splitlines() if line.strip()])
            self._stop.wait(self._interval_seconds)


def _stream_request(
    base_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[float, float]:
    start = time.perf_counter()
    first_event_seconds: float | None = None
    with requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout_seconds,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            event = line.removeprefix("data:").strip()
            if event == "[DONE]":
                break
            parsed = json.loads(event)
            if first_event_seconds is None and parsed.get("choices"):
                first_event_seconds = time.perf_counter() - start
    if first_event_seconds is None:
        raise RuntimeError("The server completed without a streaming SSE data event.")
    return first_event_seconds, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8099/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--media-path", type=Path, required=True)
    parser.add_argument("--media-type", choices=("image", "video"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--sample-interval-ms", type=float, default=50.0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    media_url = _data_url(args.media_path, args.media_type)
    content = [
        {f"{args.media_type}_url": {"url": media_url}, "type": f"{args.media_type}_url"},
        {"text": "Describe the input in one short sentence.", "type": "text"},
    ]
    payload: dict[str, Any] = {
        "max_tokens": args.max_tokens,
        "messages": [{"role": "user", "content": content}],
        "model": args.model,
        "stream": True,
        "temperature": 0.0,
    }

    with GpuMemorySampler(args.sample_interval_ms / 1000) as sampler:
        for _ in range(args.warmups):
            _stream_request(args.base_url, payload, args.timeout_seconds)
        ttft_seconds: list[float] = []
        total_seconds: list[float] = []
        for _ in range(args.runs):
            ttft, total = _stream_request(args.base_url, payload, args.timeout_seconds)
            ttft_seconds.append(ttft)
            total_seconds.append(total)

    memory_samples = sampler.samples
    if not memory_samples:
        raise RuntimeError("No GPU memory samples were collected.")
    gpu_count = len(memory_samples[0])
    report = {
        "label": args.label,
        "media_path": str(args.media_path),
        "media_type": args.media_type,
        "max_tokens": args.max_tokens,
        "runs": args.runs,
        "warmups": args.warmups,
        "ttft_ms": [round(value * 1000, 3) for value in ttft_seconds],
        "ttft_p50_ms": round(_percentile(ttft_seconds, 0.50) * 1000, 3),
        "ttft_p95_ms": round(_percentile(ttft_seconds, 0.95) * 1000, 3),
        "total_p50_ms": round(_percentile(total_seconds, 0.50) * 1000, 3),
        "gpu_memory_mib_start": memory_samples[0],
        "gpu_memory_mib_peak": [max(sample[gpu_index] for sample in memory_samples) for gpu_index in range(gpu_count)],
        "gpu_sample_interval_ms": args.sample_interval_ms,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
