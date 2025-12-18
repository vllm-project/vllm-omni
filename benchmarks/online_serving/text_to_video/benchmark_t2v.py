# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


DEFAULT_T2V_PROMPTS: list[str] = [
    "A serene lakeside sunrise with mist over the water, cinematic, slow pan.",
    "A futuristic city skyline at night, neon lights flickering, aerial shot.",
    "A close-up of raindrops on a window, bokeh lights in the background, gentle movement.",
    "A snowboarder carving down a slope, dynamic camera follow, crisp winter air.",
    "雨夜街头霓虹闪烁，路人撑伞走过，镜头缓慢推进",
]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(cwd: Path) -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_prompts(prompt_file: Optional[Path], prompt: Optional[str], num_prompts: Optional[int]) -> dict[str, Any]:
    if prompt is not None:
        prompts = [prompt]
        source = "cli"
        version = None
    elif prompt_file is not None:
        raw = prompt_file.read_text(encoding="utf-8")
        prompts = [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]
        source = str(prompt_file)
        version = _sha256_bytes(raw.encode("utf-8"))
    else:
        prompts = list(DEFAULT_T2V_PROMPTS)
        source = "builtin"
        version = _sha256_bytes(("\n".join(prompts)).encode("utf-8"))

    if not prompts:
        raise ValueError("No prompts found (empty prompt set).")

    if num_prompts is not None:
        prompts = prompts[:num_prompts]

    return {
        "prompts": prompts,
        "prompt_set": {
            "source": source,
            "count": len(prompts),
            "version_sha256": version,
        },
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Empty input for percentile().")
    if q <= 0:
        return float(sorted_values[0])
    if q >= 100:
        return float(sorted_values[-1])
    k = (len(sorted_values) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def _summary_stats(values: list[float]) -> dict[str, float]:
    values_sorted = sorted(values)
    return {
        "p50": _percentile(values_sorted, 50),
        "p90": _percentile(values_sorted, 90),
        "p95": _percentile(values_sorted, 95),
        "p99": _percentile(values_sorted, 99),
        "mean": float(sum(values_sorted) / len(values_sorted)),
        "min": float(values_sorted[0]),
        "max": float(values_sorted[-1]),
        "count": float(len(values_sorted)),
    }


def _join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    rel = path.lstrip("/")
    return f"{base}/{rel}"


def _http_post_json(url: str, api_key: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return int(getattr(resp, "status", 200)), resp.read()


def _extract_error_message(body: bytes) -> str:
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            if isinstance(err, dict) and "message" in err:
                return str(err["message"])
            return str(err)
        return str(data)
    except Exception:
        return body.decode("utf-8", errors="replace")[:1000]


def _try_parse_json(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8", errors="replace"))
    except Exception:
        pass
    return None


def _try_import_pynvml():
    try:
        import pynvml  # type: ignore[import-not-found]

        return pynvml
    except Exception:
        return None


class GPUMonitor(threading.Thread):
    def __init__(self, interval_s: float, gpu_indices: Optional[list[int]]):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.gpu_indices = gpu_indices
        self._running = True
        self.max_memory_used_mb: float = 0.0
        self.device_count: int = 0
        self.error: Optional[str] = None
        self._pynvml = None

    def run(self) -> None:
        self._pynvml = _try_import_pynvml()
        if self._pynvml is None:
            self.error = "pynvml not available"
            return

        try:
            self._pynvml.nvmlInit()
            self.device_count = int(self._pynvml.nvmlDeviceGetCount())
            indices = self.gpu_indices if self.gpu_indices is not None else list(range(self.device_count))

            handles = []
            for idx in indices:
                if idx < 0 or idx >= self.device_count:
                    continue
                handles.append(self._pynvml.nvmlDeviceGetHandleByIndex(idx))

            while self._running:
                total_used = 0
                for h in handles:
                    info = self._pynvml.nvmlDeviceGetMemoryInfo(h)
                    total_used += int(info.used)
                total_mb = total_used / 1024 / 1024
                if total_mb > self.max_memory_used_mb:
                    self.max_memory_used_mb = float(total_mb)
                time.sleep(self.interval_s)
        except Exception as e:
            self.error = str(e)
        finally:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        self.join(timeout=5.0)


def _send_video_request(
    api_base: str,
    endpoint_path: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    n: int,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: Optional[int],
    guidance_scale: Optional[float],
    guidance_scale_2: Optional[float],
    negative_prompt: Optional[str],
    response_format: str,
    timeout_s: float,
    extra_body: dict[str, Any],
) -> tuple[float, bool, Optional[str], Optional[int]]:
    url = _join_url(api_base, endpoint_path)

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "response_format": response_format,
    }
    if num_inference_steps is not None:
        payload["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale
    if guidance_scale_2 is not None:
        payload["guidance_scale_2"] = guidance_scale_2
    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt
    for k, v in extra_body.items():
        if v is not None:
            payload[k] = v

    t0 = time.perf_counter()
    try:
        status, body = _http_post_json(url, api_key, payload, timeout_s)
        latency_s = time.perf_counter() - t0
        if status < 200 or status >= 300:
            return latency_s, False, _extract_error_message(body), None
        parsed = _try_parse_json(body)
        if isinstance(parsed, dict) and "error" in parsed:
            return latency_s, False, _extract_error_message(body), None
        out_n = int(len(parsed.get("data", []))) if isinstance(parsed, dict) and isinstance(parsed.get("data"), list) else None
        return latency_s, True, None, out_n
    except urllib.error.HTTPError as e:
        latency_s = time.perf_counter() - t0
        body = e.read() if hasattr(e, "read") else b""
        return latency_s, False, _extract_error_message(body) or str(e), None
    except Exception as e:
        latency_s = time.perf_counter() - t0
        return latency_s, False, str(e), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online serving benchmark for Text-to-Video via OpenAI-compatible API (defaults to /v1/images/generations)."
    )

    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="OpenAI base URL.")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="OpenAI API key (use 'EMPTY' for local).")
    parser.add_argument("--endpoint-path", type=str, default="/images/generations", help="Endpoint path.")
    parser.add_argument("--model", type=str, required=True, help="Served model name.")

    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (overrides prompt set).")
    parser.add_argument("--prompt-file", type=str, default=None, help="Text file with one prompt per line.")
    parser.add_argument("--num-prompts", type=int, default=None, help="Use only the first N prompts from the set.")

    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent clients (threads).")
    parser.add_argument("--num-requests", type=int, default=10, help="Total measured requests.")
    parser.add_argument("--warmup-requests", type=int, default=1, help="Warmup requests (not measured).")

    parser.add_argument("--videos-per-request", type=int, default=1, help="OpenAI 'n': videos per request.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--response-format", type=str, default="b64_json", help="Response format.")
    parser.add_argument("--timeout-s", type=float, default=600.0, help="HTTP timeout per request.")

    parser.add_argument("--num-inference-steps", type=int, default=None, help="(vLLM-Omni ext) num_inference_steps.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="(vLLM-Omni ext) guidance_scale.")
    parser.add_argument("--guidance-scale-2", type=float, default=None, help="(vLLM-Omni ext) guidance_scale_2.")
    parser.add_argument("--negative-prompt", type=str, default=None, help="(vLLM-Omni ext) negative_prompt.")
    parser.add_argument("--extra-body-json", type=str, default=None, help="Extra JSON fields to merge into request.")

    parser.add_argument("--gpu-monitor", action="store_true", help="Enable GPU peak VRAM monitor (NVML).")
    parser.add_argument("--gpu-monitor-interval-s", type=float, default=0.1, help="GPU monitor sampling interval.")
    parser.add_argument(
        "--gpu-indices",
        type=str,
        default=None,
        help="Comma-separated GPU indices for monitoring (default: all).",
    )

    parser.add_argument("--output-json", type=str, default=None, help="Optional summary JSON output path.")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Optional per-request JSONL output path.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]

    prompt_file = Path(args.prompt_file).resolve() if args.prompt_file else None
    prompt_data = _read_prompts(prompt_file, args.prompt, args.num_prompts)
    prompts: list[str] = prompt_data["prompts"]

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be >= 1")
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be >= 1")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be >= 0")
    if args.videos_per_request <= 0:
        raise ValueError("--videos-per-request must be >= 1")
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be >= 1")

    extra_body: dict[str, Any] = {}
    if args.extra_body_json:
        extra_body.update(json.loads(args.extra_body_json))

    gpu_indices = None
    if args.gpu_indices:
        gpu_indices = [int(x.strip()) for x in args.gpu_indices.split(",") if x.strip()]

    monitor = None
    if args.gpu_monitor:
        monitor = GPUMonitor(interval_s=args.gpu_monitor_interval_s, gpu_indices=gpu_indices)
        monitor.start()

    def _task(i: int) -> dict[str, Any]:
        prompt = prompts[i % len(prompts)]
        latency_s, ok, err, out_n = _send_video_request(
            args.api_base,
            args.endpoint_path,
            args.api_key,
            args.model,
            prompt,
            n=args.videos_per_request,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            negative_prompt=args.negative_prompt,
            response_format=args.response_format,
            timeout_s=args.timeout_s,
            extra_body=extra_body,
        )
        return {
            "request_idx": i,
            "latency_ms": latency_s * 1000.0,
            "success": ok,
            "error": err,
            "outputs": out_n,
        }

    print("---- Benchmark Config ----")
    print(f"api_base:            {args.api_base}")
    print(f"endpoint_path:       {args.endpoint_path}")
    print(f"model:               {args.model}")
    print(f"concurrency:         {args.concurrency}")
    print(f"warmup_requests:     {args.warmup_requests}")
    print(f"num_requests:        {args.num_requests}")
    print(f"videos_per_request:  {args.videos_per_request}")
    print(f"height x width:      {args.height} x {args.width}")
    print(f"num_frames:          {args.num_frames}")
    if args.num_inference_steps is not None:
        print(f"num_inference_steps: {args.num_inference_steps}")
    if args.guidance_scale is not None:
        print(f"guidance_scale:      {args.guidance_scale}")
    if args.guidance_scale_2 is not None:
        print(f"guidance_scale_2:    {args.guidance_scale_2}")
    if args.negative_prompt is not None:
        print("negative_prompt:     (set)")
    print(f"response_format:     {args.response_format}")
    if extra_body:
        print(f"extra_body:          {extra_body}")

    for i in range(args.warmup_requests):
        _ = _task(-1 - i)

    results: list[dict[str, Any]] = []
    start_benchmark = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_task, i) for i in range(args.num_requests)]
        for fut in futures:
            results.append(fut.result())
    total_time_s = time.perf_counter() - start_benchmark

    if monitor is not None:
        monitor.stop()

    if args.output_jsonl:
        jsonl_path = Path(args.output_jsonl).resolve()
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    success = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    lat_ms_ok = [float(r["latency_ms"]) for r in success]

    outputs_ok = 0
    for r in success:
        out_n = r.get("outputs")
        if isinstance(out_n, int) and out_n > 0:
            outputs_ok += out_n
        else:
            outputs_ok += args.videos_per_request

    throughput_vid_s = float(outputs_ok / total_time_s) if total_time_s > 0 else 0.0
    throughput_frames_s = float((outputs_ok * args.num_frames) / total_time_s) if total_time_s > 0 else 0.0

    print("\n" + "=" * 40)
    print("       BENCHMARK REPORT       ")
    print("=" * 40)
    print(f"Total Time Taken:       {total_time_s:.2f} s")
    print(f"Total Videos Gen:       {outputs_ok}")
    print(f"Success Rate:           {len(success) / len(results) * 100:.1f}%")
    print("-" * 40)
    print(f"1. Throughput:          {throughput_vid_s:.2f} videos/s")
    print(f"2. Throughput:          {throughput_frames_s:.2f} frames/s")
    if monitor is not None and monitor.error is None:
        print(f"3. Memory Peak (Total): {monitor.max_memory_used_mb:.2f} MB")
    else:
        print("3. Memory Peak (Total): N/A")
    print("-" * 40)
    if lat_ms_ok:
        stats = _summary_stats(lat_ms_ok)
        print(f"Avg Latency (ok):       {stats['mean'] / 1000.0:.2f} s")
        print(f"P99 Latency (ok):       {stats['p99'] / 1000.0:.2f} s")
    else:
        print("Avg Latency:            N/A")
        print("P99 Latency:            N/A")
    print("=" * 40)
    if failures:
        print(f"Failures: {len(failures)} (showing up to 3)")
        for r in failures[:3]:
            print(f"- idx={r['request_idx']}: {r.get('error')}")

    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "task": "t2v",
            "timestamp_utc": _utc_timestamp(),
            "git_commit": _git_commit(repo_root),
            "system": {
                "platform": platform.platform(),
                "python": sys.version.replace("\n", " "),
            },
            "prompt_set": prompt_data["prompt_set"],
            "config": {
                "api_base": args.api_base,
                "endpoint_path": args.endpoint_path,
                "model": args.model,
                "concurrency": args.concurrency,
                "warmup_requests": args.warmup_requests,
                "num_requests": args.num_requests,
                "videos_per_request": args.videos_per_request,
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "guidance_scale_2": args.guidance_scale_2,
                "negative_prompt": bool(args.negative_prompt),
                "response_format": args.response_format,
                "timeout_s": args.timeout_s,
                "extra_body": extra_body,
            },
            "metrics_summary": {
                "e2e_latency_ms": (_summary_stats(lat_ms_ok) if lat_ms_ok else None),
                "throughput_videos_per_s": throughput_vid_s,
                "throughput_frames_per_s": throughput_frames_s,
                "success_rate": float(len(success) / len(results)) if results else None,
                "failure_rate": float(len(failures) / len(results)) if results else None,
                "peak_vram_mb": (monitor.max_memory_used_mb if monitor is not None and monitor.error is None else None),
            },
        }
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote summary JSON to {out_path}")


if __name__ == "__main__":
    main()
