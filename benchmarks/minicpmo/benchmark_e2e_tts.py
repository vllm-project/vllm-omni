# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-o 4.5 OpenAI-compatible text+audio E2E benchmark."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf

SAMPLE_RATE = 24_000
DEFAULT_PROMPT = "Say hello, then introduce vLLM in one sentence."


@dataclass
class RequestResult:
    request_id: int
    ok: bool
    status_code: int | None
    latency_s: float
    audio_present: bool
    text_present: bool
    audio_bytes: int
    waveform_samples: int
    waveform_duration_s: float
    rtf: float
    sample_rate: int | None
    finite: bool
    nan_count: int
    inf_count: int
    clipping_ratio: float
    rms: float
    peak_abs: float
    text_chars: int
    completion_tokens: int | None
    prompt_tokens: int | None
    total_tokens: int | None
    error: str | None


class GpuMemorySampler:
    def __init__(self, gpu_indexes: list[int], interval_s: float) -> None:
        self.gpu_indexes = gpu_indexes
        self.interval_s = interval_s
        self.samples: list[dict[int, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuMemorySampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                row: dict[int, int] = {}
                for line in proc.stdout.splitlines():
                    if not line.strip():
                        continue
                    gpu_s, mem_s = [part.strip() for part in line.split(",", 1)]
                    gpu = int(gpu_s)
                    if gpu in self.gpu_indexes:
                        row[gpu] = int(mem_s)
                if row:
                    self.samples.append(row)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def peak_by_gpu(self) -> dict[str, int]:
        peaks = {gpu: 0 for gpu in self.gpu_indexes}
        for sample in self.samples:
            for gpu, mem in sample.items():
                peaks[gpu] = max(peaks.get(gpu, 0), mem)
        return {str(gpu): mem for gpu, mem in peaks.items()}


def _git_commit() -> str | None:
    if commit := os.getenv("VLLM_OMNI_COMMIT"):
        return commit
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _extract_response_payload(response: dict[str, Any]) -> tuple[str, bytes | None]:
    text_parts: list[str] = []
    audio_b64: str | None = None

    for choice in response.get("choices", []):
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                audio = item.get("audio")
                if isinstance(audio, dict) and isinstance(audio.get("data"), str):
                    audio_b64 = audio["data"]

        audio = message.get("audio")
        if isinstance(audio, dict) and isinstance(audio.get("data"), str):
            audio_b64 = audio["data"]

    if audio_b64 and "," in audio_b64 and audio_b64.startswith("data:"):
        audio_b64 = audio_b64.split(",", 1)[1]
    audio_bytes = base64.b64decode(audio_b64) if audio_b64 else None
    return "\n".join(text_parts), audio_bytes


def _audio_stats(audio_bytes: bytes | None, latency_s: float) -> dict[str, Any]:
    if not audio_bytes:
        return {
            "audio_present": False,
            "audio_bytes": 0,
            "waveform_samples": 0,
            "waveform_duration_s": 0.0,
            "rtf": math.inf,
            "sample_rate": None,
            "finite": False,
            "nan_count": 0,
            "inf_count": 0,
            "clipping_ratio": math.nan,
            "rms": math.nan,
            "peak_abs": math.nan,
        }

    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(wav)
    finite_values = wav[finite_mask]
    abs_values = np.abs(finite_values) if finite_values.size else np.asarray([], dtype=np.float32)
    samples = int(wav.shape[0])
    duration_s = samples / sample_rate if sample_rate else 0.0
    return {
        "audio_present": True,
        "audio_bytes": len(audio_bytes),
        "waveform_samples": samples,
        "waveform_duration_s": float(duration_s),
        "rtf": latency_s / duration_s if duration_s else math.inf,
        "sample_rate": int(sample_rate),
        "finite": bool(finite_mask.all()),
        "nan_count": int(np.isnan(wav).sum()),
        "inf_count": int(np.isinf(wav).sum()),
        "clipping_ratio": float((abs_values >= 0.999).mean()) if abs_values.size else math.nan,
        "rms": float(np.sqrt(np.mean(np.square(finite_values)))) if finite_values.size else math.nan,
        "peak_abs": float(abs_values.max()) if abs_values.size else math.nan,
    }


def _usage_field(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return int(value) if isinstance(value, int | float) else None


def _send_request(args: argparse.Namespace, request_id: int) -> RequestResult:
    prompt = args.prompt
    if args.unique_suffix:
        prompt = f"{prompt}\nRequest id: {request_id}."

    body: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "chat_template_kwargs": {"use_tts_template": True},
        "temperature": args.temperature,
        "seed": args.seed,
    }
    if args.max_tokens is not None:
        body["max_tokens"] = args.max_tokens

    url = args.base_url.rstrip("/") + "/chat/completions"
    start = time.perf_counter()
    status_code = None
    error = None
    response_json: dict[str, Any] = {}
    try:
        response = requests.post(url, json=body, timeout=args.timeout_s)
        status_code = response.status_code
        response.raise_for_status()
        response_json = response.json()
    except Exception as exc:
        error = repr(exc)
    latency_s = time.perf_counter() - start

    text = ""
    audio_bytes = None
    if response_json:
        try:
            text, audio_bytes = _extract_response_payload(response_json)
        except Exception as exc:
            error = repr(exc)

    try:
        stats = _audio_stats(audio_bytes, latency_s)
    except Exception as exc:
        error = repr(exc)
        stats = _audio_stats(None, latency_s)

    usage = response_json.get("usage", {}) if response_json else {}
    text_present = bool(text.strip())
    ok = (
        error is None
        and status_code == 200
        and text_present
        and bool(stats["audio_present"])
        and stats["finite"]
        and stats["sample_rate"] == SAMPLE_RATE
    )
    return RequestResult(
        request_id=request_id,
        ok=ok,
        status_code=status_code,
        latency_s=latency_s,
        text_present=text_present,
        text_chars=len(text),
        completion_tokens=_usage_field(usage, "completion_tokens"),
        prompt_tokens=_usage_field(usage, "prompt_tokens"),
        total_tokens=_usage_field(usage, "total_tokens"),
        error=error,
        **stats,
    )


def _summarize(args: argparse.Namespace, results: list[RequestResult], peak_by_gpu: dict[str, int]) -> dict[str, Any]:
    successful = [result for result in results if result.ok]
    latencies = [result.latency_s for result in successful]
    rtfs = [result.rtf for result in successful]
    durations = [result.waveform_duration_s for result in successful]
    return {
        "model": args.model,
        "model_revision": args.model_revision,
        "vllm_omni_commit": _git_commit(),
        "serve_config": args.serve_config,
        "base_url": args.base_url,
        "prompt": args.prompt,
        "prompt_chars": len(args.prompt),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "requests": args.requests,
        "timeout_s": args.timeout_s,
        "successes": len(successful),
        "failures": len(results) - len(successful),
        "failure_rate": (len(results) - len(successful)) / len(results) if results else math.nan,
        "latency_p50_s": statistics.median(latencies) if latencies else math.nan,
        "latency_p95_s": _percentile(latencies, 0.95),
        "latency_mean_s": statistics.mean(latencies) if latencies else math.nan,
        "rtf_p50": statistics.median(rtfs) if rtfs else math.nan,
        "rtf_p95": _percentile(rtfs, 0.95),
        "waveform_duration_s_median": statistics.median(durations) if durations else math.nan,
        "audio_finite_all": all(result.finite for result in successful) if successful else False,
        "nan_count_total": sum(result.nan_count for result in results),
        "inf_count_total": sum(result.inf_count for result in results),
        "clipping_ratio_max": max((result.clipping_ratio for result in successful), default=math.nan),
        "rms_median": statistics.median(result.rms for result in successful) if successful else math.nan,
        "peak_abs_max": max((result.peak_abs for result in successful), default=math.nan),
        "peak_gpu_memory_mib_by_gpu": peak_by_gpu,
        "stage_timing_s": {"thinker": None, "talker": None, "token2wav": None, "packaging": None},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for warmup_id in range(args.warmup):
        _send_request(args, -warmup_id - 1)

    results: list[RequestResult] = []
    gpu_indexes = [int(item) for item in args.gpu_indexes.split(",") if item.strip()]
    with GpuMemorySampler(gpu_indexes, args.gpu_sample_interval_s) as sampler:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(_send_request, args, i) for i in range(args.requests)]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: item.request_id)

    rows_path = output_dir / "requests.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), sort_keys=True) + "\n")

    summary = _summarize(args, results, sampler.peak_by_gpu())
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "rows_path": str(rows_path),
        "csv_path": str(csv_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8099/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--serve-config", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--gpu-indexes", default="0,1")
    parser.add_argument("--gpu-sample-interval-s", type=float, default=0.25)
    parser.add_argument("--unique-suffix", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
