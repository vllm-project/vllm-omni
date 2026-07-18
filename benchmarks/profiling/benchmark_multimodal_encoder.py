#!/usr/bin/env python3
"""Profile multimodal TTFT and GPU memory against a running Omni server.

This benchmark deliberately runs outside the vLLM worker processes. It keeps
the baseline independent from model implementation changes while the server's
torch profiler supplies operator- and allocator-level details.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import mimetypes
import os
import socket
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class RequestResult:
    index: int
    success: bool
    ttft_ms: float | None
    e2e_latency_ms: float
    completion_tokens: int | None
    response_text: str
    response_sha256: str
    token_ids: list[int]
    stage_metrics: dict[str, dict[str, Any]]
    error: str | None


@dataclass
class MemorySample:
    monotonic_s: float
    gpu_memory_mib: dict[str, float]
    process_memory_mib: dict[str, dict[str, float]]


class NvmlMemorySampler:
    """Sample total and per-process GPU memory with low-overhead NVML calls."""

    def __init__(self, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError("sample interval must be positive")
        self.interval_s = interval_s
        self.samples: list[MemorySample] = []
        self.gpus: dict[str, dict[str, Any]] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml: Any = None
        self._handles: list[Any] = []

    def start(self) -> None:
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError("NVML sampling requires the nvidia-ml-py package") from exc

        self._pynvml = pynvml
        pynvml.nvmlInit()
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = _decode_nvml_text(pynvml.nvmlDeviceGetUUID(handle))
            name = _decode_nvml_text(pynvml.nvmlDeviceGetName(handle))
            total_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            self._handles.append(handle)
            self.gpus[uuid] = {
                "index": index,
                "name": name,
                "total_memory_mib": _bytes_to_mib(total_bytes),
            }

        self._thread = threading.Thread(target=self._run, name="nvml-memory-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 4))
        if self._pynvml is not None:
            self._pynvml.nvmlShutdown()

    def _run(self) -> None:
        assert self._pynvml is not None
        while not self._stop_event.is_set():
            started = time.perf_counter()
            gpu_memory: dict[str, float] = {}
            process_memory: dict[str, dict[str, float]] = {}
            for handle, uuid in zip(self._handles, self.gpus, strict=True):
                gpu_memory[uuid] = _bytes_to_mib(self._pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                per_pid: dict[str, float] = {}
                try:
                    processes = self._pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                except self._pynvml.NVMLError:
                    processes = []
                for process in processes:
                    used_bytes = getattr(process, "usedGpuMemory", None)
                    not_available = getattr(self._pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
                    if isinstance(used_bytes, int) and used_bytes >= 0 and used_bytes != not_available:
                        per_pid[str(process.pid)] = _bytes_to_mib(used_bytes)
                process_memory[uuid] = per_pid
            self.samples.append(
                MemorySample(
                    monotonic_s=started,
                    gpu_memory_mib=gpu_memory,
                    process_memory_mib=process_memory,
                )
            )
            elapsed = time.perf_counter() - started
            self._stop_event.wait(max(0.0, self.interval_s - elapsed))


def _decode_nvml_text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _bytes_to_mib(value: int) -> float:
    return round(value / (1024 * 1024), 3)


def _data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    content_type = {
        "image": "image_url",
        "audio": "audio_url",
        "video": "video_url",
    }.get(args.modality)
    if content_type is not None:
        if not args.asset:
            raise ValueError(f"--asset is required for modality {args.modality}")
        for raw_path in args.asset:
            content.append(
                {
                    "type": content_type,
                    content_type: {"url": _data_url(Path(raw_path).expanduser().resolve())},
                }
            )
    elif args.asset:
        raise ValueError("--asset is not valid for text-only requests")

    content.append({"type": "text", "text": args.prompt})
    return {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are MiniCPM-o, a helpful multimodal assistant that can "
                    "understand images, audio and video. Respond briefly in text."
                ),
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "modalities": ["text"],
        "return_token_ids": True,
        "return_stage_metrics": True,
    }


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def update_stage_metrics(
    stage_metrics: dict[str, dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    """Merge the latest per-stage snapshot from one streamed response chunk."""
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return
    snapshot = metrics.get("stage_metrics")
    if not isinstance(snapshot, dict):
        return
    for stage_id, values in snapshot.items():
        if isinstance(values, dict):
            stage_metrics[str(stage_id)] = dict(values)


def send_streaming_request(
    *,
    index: int,
    endpoint: str,
    payload_bytes: bytes,
    headers: dict[str, str],
    timeout_s: float,
    start_barrier: threading.Barrier | None = None,
) -> RequestResult:
    if start_barrier is not None:
        start_barrier.wait(timeout=timeout_s)

    started = time.perf_counter()
    first_token_at: float | None = None
    completion_tokens: int | None = None
    completion_tokens_seen = 0
    response_parts: list[str] = []
    token_ids: list[int] = []
    stage_metrics: dict[str, dict[str, Any]] = {}
    try:
        request = urllib.request.Request(endpoint, data=payload_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line.removeprefix("data:").strip()
                if not chunk or chunk == "[DONE]":
                    continue
                data = json.loads(chunk)
                update_stage_metrics(stage_metrics, data)
                usage = data.get("usage")
                if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                    completion_tokens = usage["completion_tokens"]

                choices = data.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                chunk_token_ids = choice.get("token_ids") or []
                if content:
                    response_parts.append(content)
                if chunk_token_ids:
                    token_ids.extend(chunk_token_ids)

                token_progress = bool(content or chunk_token_ids)
                if completion_tokens is not None and completion_tokens > completion_tokens_seen:
                    token_progress = True
                    completion_tokens_seen = completion_tokens
                modality = data.get("modality")
                if first_token_at is None and modality in (None, "text") and token_progress:
                    first_token_at = time.perf_counter()
    except (OSError, TimeoutError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        ended = time.perf_counter()
        return RequestResult(
            index=index,
            success=False,
            ttft_ms=None,
            e2e_latency_ms=(ended - started) * 1000,
            completion_tokens=completion_tokens,
            response_text="".join(response_parts),
            response_sha256="",
            token_ids=token_ids,
            stage_metrics=stage_metrics,
            error=repr(exc),
        )

    ended = time.perf_counter()
    response_text = "".join(response_parts)
    return RequestResult(
        index=index,
        success=first_token_at is not None,
        ttft_ms=(first_token_at - started) * 1000 if first_token_at is not None else None,
        e2e_latency_ms=(ended - started) * 1000,
        completion_tokens=completion_tokens,
        response_text=response_text,
        response_sha256=hashlib.sha256(response_text.encode()).hexdigest(),
        token_ids=token_ids,
        stage_metrics=stage_metrics,
        error=None if first_token_at is not None else "stream ended before the first text token",
    )


def post_profile_control(
    base_url: str,
    route: str,
    stages: list[int],
    headers: dict[str, str],
    timeout_s: float,
) -> None:
    body = json.dumps({"stages": stages}).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/{route}",
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        if response.status != 200:
            raise RuntimeError(f"{route} returned HTTP {response.status}")


def summarize_memory(
    samples: list[MemorySample],
    *,
    request_started_s: float,
    request_ended_s: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    uuids = sorted({uuid for sample in samples for uuid in sample.gpu_memory_mib})
    for uuid in uuids:
        baseline_values = [
            sample.gpu_memory_mib[uuid]
            for sample in samples
            if sample.monotonic_s < request_started_s and uuid in sample.gpu_memory_mib
        ]
        request_values = [
            sample.gpu_memory_mib[uuid]
            for sample in samples
            if request_started_s <= sample.monotonic_s <= request_ended_s and uuid in sample.gpu_memory_mib
        ]
        process_baselines: dict[str, float] = {}
        process_peaks: dict[str, float] = {}
        for sample in samples:
            if sample.monotonic_s < request_started_s:
                process_baselines = dict(sample.process_memory_mib.get(uuid, {}))
            elif sample.monotonic_s <= request_ended_s:
                for pid, used_mib in sample.process_memory_mib.get(uuid, {}).items():
                    process_peaks[pid] = max(process_peaks.get(pid, 0.0), used_mib)

        baseline_mib = baseline_values[-1] if baseline_values else None
        peak_mib = max(request_values) if request_values else None
        summary[uuid] = {
            "baseline_memory_mib": baseline_mib,
            "request_peak_memory_mib": peak_mib,
            "request_peak_delta_mib": (
                round(peak_mib - baseline_mib, 3) if peak_mib is not None and baseline_mib is not None else None
            ),
            "process_baseline_memory_mib": process_baselines,
            "process_peak_memory_mib": process_peaks,
            "process_peak_delta_mib": {
                pid: round(peak - process_baselines[pid], 3) if pid in process_baselines else None
                for pid, peak in process_peaks.items()
            },
        }
    return summary


def _latency_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "min": min(values),
        "max": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8099")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--modality", choices=("text", "image", "audio", "video"), required=True)
    parser.add_argument("--asset", action="append", default=[], help="Media path; repeat for multiple inputs")
    parser.add_argument("--prompt", default="Describe the input briefly.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--request-count", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--profile-stage", type=int, action="append", default=[])
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--baseline-seconds", type=float, default=1.0)
    parser.add_argument("--no-memory-sampling", action="store_true")
    parser.add_argument("--include-memory-samples", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-max-model-len", type=int)
    parser.add_argument("--server-max-num-seqs", type=int)
    parser.add_argument("--server-limit-mm-per-prompt", default="")
    args = parser.parse_args()
    if args.request_count < 1 or args.concurrency < 1 or args.concurrency > args.request_count:
        parser.error("require 1 <= concurrency <= request-count")
    if args.warmup_requests < 0:
        parser.error("warmup-requests cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
    headers = _headers(args.api_key)
    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"

    sampler = None if args.no_memory_sampling else NvmlMemorySampler(args.sample_interval)
    if sampler is not None:
        sampler.start()
        time.sleep(args.baseline_seconds)

    warmup_results: list[RequestResult] = []
    warmup_started_s = time.perf_counter()
    for index in range(args.warmup_requests):
        result = send_streaming_request(
            index=-(index + 1),
            endpoint=endpoint,
            payload_bytes=payload_bytes,
            headers=headers,
            timeout_s=args.timeout,
        )
        warmup_results.append(result)
        if not result.success:
            if sampler is not None:
                sampler.stop()
            raise RuntimeError(f"warmup request failed: {result.error}")
    warmup_ended_s = time.perf_counter()

    if sampler is not None:
        # Capture a stable post-warmup baseline separately from the cold
        # modality allocation peak. This makes allocator growth during the
        # first image/audio/video request visible without contaminating the
        # steady-state request delta.
        time.sleep(args.baseline_seconds)

    profile_started = False
    request_started_s = 0.0
    try:
        if args.profile_stage:
            post_profile_control(args.base_url, "start_profile", args.profile_stage, headers, args.timeout)
            profile_started = True

        request_started_s = time.perf_counter()
        barrier = threading.Barrier(args.concurrency) if args.concurrency > 1 else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    send_streaming_request,
                    index=index,
                    endpoint=endpoint,
                    payload_bytes=payload_bytes,
                    headers=headers,
                    timeout_s=args.timeout,
                    start_barrier=barrier if index < args.concurrency else None,
                )
                for index in range(args.request_count)
            ]
            results = [future.result() for future in futures]
    finally:
        request_ended_s = time.perf_counter()
        if profile_started:
            post_profile_control(args.base_url, "stop_profile", args.profile_stage, headers, args.timeout)
        if sampler is not None:
            sampler.stop()

    successful = [result for result in results if result.success]
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "label": args.label,
        "request": {
            "base_url": args.base_url,
            "model": args.model,
            "modality": args.modality,
            "asset_paths": [str(Path(path).expanduser().resolve()) for path in args.asset],
            "asset_count": len(args.asset),
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "warmup_requests": args.warmup_requests,
            "request_count": args.request_count,
            "concurrency": args.concurrency,
        },
        "server_config": {
            "max_model_len": args.server_max_model_len,
            "max_num_seqs": args.server_max_num_seqs,
            "limit_mm_per_prompt": (
                json.loads(args.server_limit_mm_per_prompt) if args.server_limit_mm_per_prompt else None
            ),
        },
        "profile_stages": args.profile_stage,
        "summary": {
            "successful_requests": len(successful),
            "failed_requests": len(results) - len(successful),
            "ttft_ms": _latency_summary([result.ttft_ms for result in successful if result.ttft_ms is not None]),
            "e2e_latency_ms": _latency_summary([result.e2e_latency_ms for result in successful]),
        },
        "warmup_results": [asdict(result) for result in warmup_results],
        "warmup_summary": {
            "successful_requests": sum(result.success for result in warmup_results),
            "failed_requests": sum(not result.success for result in warmup_results),
            "ttft_ms": _latency_summary(
                [result.ttft_ms for result in warmup_results if result.success and result.ttft_ms is not None]
            ),
            "e2e_latency_ms": _latency_summary([result.e2e_latency_ms for result in warmup_results if result.success]),
        },
        "results": [asdict(result) for result in sorted(results, key=lambda item: item.index)],
    }
    if sampler is not None:
        report["gpus"] = sampler.gpus
        if warmup_results:
            report["warmup_memory_summary"] = summarize_memory(
                sampler.samples,
                request_started_s=warmup_started_s,
                request_ended_s=warmup_ended_s,
            )
        report["memory_summary"] = summarize_memory(
            sampler.samples,
            request_started_s=request_started_s,
            request_ended_s=request_ended_s,
        )
        if args.include_memory_samples:
            report["memory_samples"] = [asdict(sample) for sample in sampler.samples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    if len(successful) != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
