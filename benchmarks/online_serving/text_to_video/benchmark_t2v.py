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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.omni_diffusion import prepare_requests
from vllm_omni.utils.platform_utils import detect_device_type, is_npu


DEFAULT_T2V_PROMPTS: list[str] = [
    "A serene lakeside sunrise with mist over the water, cinematic, slow pan.",
    "A cat wearing a tiny backpack walks through a bustling market, handheld camera.",
    "A futuristic city skyline at night, neon lights flickering, aerial shot.",
    "A close-up of raindrops on a window, bokeh lights in the background, gentle movement.",
    "A paper boat floating down a small stream in a forest, soft light, calm mood.",
    "A snowboarder carving down a slope, dynamic camera follow, crisp winter air.",
    "A robot assembling a flower bouquet in a bright workshop, smooth motion.",
    "A time-lapse of clouds rolling over mountains, wide shot, dramatic lighting.",
    "一只小狗追逐泡泡，草地上阳光明媚，镜头轻微摇摆",
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


class _CudaMemorySampler:
    def __init__(self, backend: str, interval_s: float, gpu_indices: Optional[list[int]]):
        self.backend = backend
        self.interval_s = interval_s
        self.gpu_indices = gpu_indices
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_total_mib: Optional[float] = None
        self._nvml = None
        self._nvml_handles = None

    def _get_used_mib_total(self) -> Optional[float]:
        if self.backend == "nvml":
            try:
                if self._nvml is None or self._nvml_handles is None:
                    return None
                total = 0.0
                for handle in self._nvml_handles:
                    info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                    total += info.used / (1024 * 1024)
                return total
            except Exception:
                return None

        if self.backend == "nvidia-smi":
            try:
                args = [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ]
                res = subprocess.run(args, check=True, capture_output=True, text=True)
                lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
                used_mib = [float(x) for x in lines]
                if self.gpu_indices is not None:
                    used_mib = [used_mib[i] for i in self.gpu_indices if i < len(used_mib)]
                return float(sum(used_mib)) if used_mib else None
            except Exception:
                return None

        return None

    def start(self) -> None:
        self._stop.clear()
        self._peak_total_mib = None
        if self.backend == "nvml":
            try:
                import pynvml  # type: ignore[import-not-found]

                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                indices = self.gpu_indices if self.gpu_indices is not None else list(range(count))
                self._nvml = pynvml
                self._nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices]
            except Exception:
                self._nvml = None
                self._nvml_handles = None

        def _run() -> None:
            while not self._stop.is_set():
                used = self._get_used_mib_total()
                if used is not None:
                    if self._peak_total_mib is None or used > self._peak_total_mib:
                        self._peak_total_mib = used
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=_run, name="gpu-mem-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> Optional[float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self.backend == "nvml" and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
            self._nvml_handles = None
        return self._peak_total_mib


def _detect_cuda_mem_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import pynvml  # type: ignore[import-not-found]

        _ = pynvml
        return "nvml"
    except Exception:
        try:
            subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
            return "nvidia-smi"
        except Exception:
            return "none"


def _device_sync(device_type: str) -> None:
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_type == "npu" and hasattr(torch, "npu"):
        try:
            torch.npu.synchronize()  # type: ignore[attr-defined]
        except Exception:
            pass


def _is_oom_error(msg: str) -> bool:
    s = msg.lower()
    return ("out of memory" in s) or ("cuda oom" in s) or (("cublas" in s) and ("alloc" in s))


def _infer_num_frames(video: Any) -> Optional[int]:
    try:
        import numpy as np

        if isinstance(video, np.ndarray):
            arr = video
            if arr.ndim == 5:
                arr = arr[0]
            if arr.ndim == 4:
                return int(arr.shape[0])
            return None
        if isinstance(video, torch.Tensor):
            t = video.detach().cpu()
            if t.dim() == 5:
                # [B, C, F, H, W] or [B, F, H, W, C]
                if t.shape[1] in (3, 4):
                    return int(t.shape[2])
                return int(t.shape[1])
            if t.dim() == 4 and t.shape[0] in (3, 4):
                return int(t.shape[1])
            return None
    except Exception:
        pass

    if isinstance(video, list):
        return len(video)

    return None


@dataclass
class _RequestResult:
    request_idx: int
    batch_size: int
    videos_generated: int
    frames_generated: Optional[int]
    success: bool
    error: Optional[str]
    timings_ms: dict[str, float]
    peak_vram_mib: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DiT Text-to-Video generation via vLLM-Omni (in-process).")

    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Diffusers Wan2.2 model ID or local path.",
    )
    parser.add_argument("--prompt", default=None, help="Single prompt (overrides prompt set).")
    parser.add_argument("--prompt-file", type=str, default=None, help="Text file with one prompt per line.")
    parser.add_argument("--num-prompts", type=int, default=None, help="Use only the first N prompts from the set.")

    parser.add_argument("--height", type=int, default=720, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--fps", type=int, default=24, help="FPS for exported video (if enabled).")
    parser.add_argument("--num-inference-steps", type=int, default=40, help="Sampling steps.")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG scale (low/high).")
    parser.add_argument("--guidance-scale-high", type=float, default=None, help="Optional separate CFG for high-noise.")
    parser.add_argument("--num-outputs-per-prompt", type=int, default=1, help="Videos per prompt.")

    parser.add_argument("--boundary-ratio", type=float, default=0.875, help="Boundary split ratio for low/high DiT.")
    parser.add_argument("--flow-shift", type=float, default=5.0, help="Scheduler flow_shift (e.g., 5.0 for 720p).")

    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for diffusion engine.")
    parser.add_argument(
        "--cache-backend",
        type=str,
        default=None,
        choices=["cache_dit", "tea_cache"],
        help="Optional cache backend for acceleration.",
    )
    parser.add_argument(
        "--cache-config-json",
        type=str,
        default=None,
        help="Optional JSON string for cache_config (passed to Omni).",
    )

    parser.add_argument("--warmup-requests", type=int, default=3, help="Warmup requests (not measured).")
    parser.add_argument("--requests", type=int, default=10, help="Measured requests.")
    parser.add_argument("--batch-size", type=int, default=1, help="Prompts per request.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed; increments per prompt.")

    parser.add_argument("--output-json", type=str, default="t2v_benchmark_summary.json", help="Summary JSON path.")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Optional per-request JSONL path.")

    parser.add_argument("--export-videos-dir", type=str, default=None, help="Optional directory to export mp4 samples.")
    parser.add_argument("--export-videos-max", type=int, default=0, help="Max measured requests to export videos for.")
    parser.add_argument(
        "--include-export-in-e2e",
        action="store_true",
        help="If set, include export time (mp4 encoding) in e2e latency.",
    )

    parser.add_argument(
        "--cuda-mem-monitor",
        type=str,
        default="auto",
        choices=["auto", "none", "nvml", "nvidia-smi"],
        help="Best-effort CUDA peak VRAM monitor backend.",
    )
    parser.add_argument(
        "--cuda-mem-sample-ms",
        type=int,
        default=0,
        help="Memory sampling interval in ms (0 disables sampling).",
    )
    parser.add_argument(
        "--cuda-gpu-indices",
        type=str,
        default=None,
        help="Comma-separated physical GPU indices for memory monitoring (default: all).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]

    prompt_file = Path(args.prompt_file).resolve() if args.prompt_file else None
    prompt_data = _read_prompts(prompt_file, args.prompt, args.num_prompts)
    prompts: list[str] = prompt_data["prompts"]

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.requests <= 0:
        raise ValueError("--requests must be >= 1")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be >= 0")

    device_type = detect_device_type()
    generator_device = device_type if device_type in ("cuda", "npu") else "cpu"

    vae_use_slicing = is_npu()
    vae_use_tiling = is_npu()

    cache_config = None
    if args.cache_config_json:
        cache_config = json.loads(args.cache_config_json)

    omni = Omni(
        model=args.model,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        num_gpus=args.num_gpus,
        boundary_ratio=args.boundary_ratio,
        flow_shift=args.flow_shift,
    )

    try:
        engine = getattr(omni.instance, "engine", None)
        if engine is None:
            raise RuntimeError("Loaded model does not expose diffusion engine (is this a diffusion model?).")

        cuda_mem_backend = _detect_cuda_mem_backend(args.cuda_mem_monitor)
        gpu_indices = None
        if args.cuda_gpu_indices:
            gpu_indices = [int(x.strip()) for x in args.cuda_gpu_indices.split(",") if x.strip()]

        sample_interval_s = (args.cuda_mem_sample_ms / 1000.0) if args.cuda_mem_sample_ms > 0 else 0.0

        jsonl_path = Path(args.output_jsonl).resolve() if args.output_jsonl else None
        jsonl_f = jsonl_path.open("w", encoding="utf-8") if jsonl_path else None

        export_dir = Path(args.export_videos_dir).resolve() if args.export_videos_dir else None
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)

        results: list[_RequestResult] = []

        total_requests = args.warmup_requests + args.requests
        prompt_idx = 0
        prompt_seed = args.base_seed

        for req_idx in range(total_requests):
            batch_prompts = []
            batch_generators: list[torch.Generator] = []
            for _ in range(args.batch_size):
                batch_prompts.append(prompts[prompt_idx % len(prompts)])
                prompt_idx += 1
                g = torch.Generator(device=generator_device).manual_seed(prompt_seed)
                prompt_seed += 1
                batch_generators.append(g)

            req = prepare_requests(
                batch_prompts,
                generator=batch_generators,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                guidance_scale=args.guidance_scale,
                guidance_scale_2=args.guidance_scale_high,
                num_inference_steps=args.num_inference_steps,
                num_outputs_per_prompt=args.num_outputs_per_prompt,
            )

            timings_ms: dict[str, float] = {}

            peak_vram_mib = None
            mem_sampler: Optional[_CudaMemorySampler] = None
            if device_type == "cuda" and sample_interval_s > 0 and cuda_mem_backend != "none":
                mem_sampler = _CudaMemorySampler(cuda_mem_backend, sample_interval_s, gpu_indices)

            success = False
            err: Optional[str] = None
            video = None
            export_ms = 0.0

            try:
                _device_sync(device_type)
                if mem_sampler is not None:
                    mem_sampler.start()

                t0 = time.perf_counter()

                pre_t0 = time.perf_counter()
                reqs = [req]
                if engine.pre_process_func is not None:
                    reqs = engine.pre_process_func(reqs)
                timings_ms["preprocess_ms"] = (time.perf_counter() - pre_t0) * 1000.0

                eng_t0 = time.perf_counter()
                out = engine.add_req_and_wait_for_response(reqs)
                timings_ms["engine_ms"] = (time.perf_counter() - eng_t0) * 1000.0

                if out.error:
                    raise RuntimeError(out.error)

                post_t0 = time.perf_counter()
                video = engine.post_process_func(out.output)
                timings_ms["postprocess_ms"] = (time.perf_counter() - post_t0) * 1000.0

                _device_sync(device_type)
                e2e_ms = (time.perf_counter() - t0) * 1000.0

                success = video is not None

                is_measured = req_idx >= args.warmup_requests
                if (
                    success
                    and is_measured
                    and export_dir is not None
                    and args.export_videos_max > 0
                    and (req_idx - args.warmup_requests) < args.export_videos_max
                ):
                    try:
                        from diffusers.utils import export_to_video

                        export_t0 = time.perf_counter()
                        out_path = export_dir / f"req{req_idx:04d}.mp4"
                        export_to_video(video, str(out_path), fps=args.fps)
                        export_ms = (time.perf_counter() - export_t0) * 1000.0
                        timings_ms["export_ms"] = export_ms
                        if args.include_export_in_e2e:
                            e2e_ms += export_ms
                    except Exception as export_exc:
                        timings_ms["export_error_ms"] = export_ms
                        raise RuntimeError(f"export_to_video failed: {export_exc}") from export_exc

                timings_ms["e2e_ms"] = e2e_ms
            except Exception as e:
                err = str(e)
                success = False
            finally:
                if mem_sampler is not None:
                    peak_vram_mib = mem_sampler.stop()

            is_measured = req_idx >= args.warmup_requests
            if not is_measured:
                continue

            inferred_frames = _infer_num_frames(video) if success else None
            videos_generated = args.batch_size * args.num_outputs_per_prompt if success else 0
            frames_generated = (inferred_frames * videos_generated) if (success and inferred_frames is not None) else None

            result = _RequestResult(
                request_idx=req_idx - args.warmup_requests,
                batch_size=args.batch_size,
                videos_generated=videos_generated,
                frames_generated=frames_generated,
                success=success,
                error=err,
                timings_ms=timings_ms,
                peak_vram_mib=peak_vram_mib,
            )
            results.append(result)

            if jsonl_f is not None:
                jsonl_f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                jsonl_f.flush()

        if jsonl_f is not None:
            jsonl_f.close()

        latencies_ms = [r.timings_ms["e2e_ms"] for r in results if r.success and "e2e_ms" in r.timings_ms]
        engine_ms = [r.timings_ms["engine_ms"] for r in results if r.success and "engine_ms" in r.timings_ms]
        pre_ms = [r.timings_ms["preprocess_ms"] for r in results if r.success and "preprocess_ms" in r.timings_ms]
        post_ms = [r.timings_ms["postprocess_ms"] for r in results if r.success and "postprocess_ms" in r.timings_ms]
        export_ms = [r.timings_ms["export_ms"] for r in results if r.success and "export_ms" in r.timings_ms]
        vram_samples = [r.peak_vram_mib for r in results if r.success and r.peak_vram_mib is not None]

        failures = [r for r in results if not r.success]
        oom_failures = [r for r in failures if r.error and _is_oom_error(r.error)]

        total_videos = sum(r.videos_generated for r in results)
        total_time_s = sum((r.timings_ms.get("e2e_ms", 0.0) / 1000.0) for r in results if r.success)
        total_frames = 0
        for r in results:
            if r.success:
                frames = r.frames_generated
                if frames is not None:
                    total_frames += frames
                else:
                    total_frames += r.videos_generated * args.num_frames

        summary = {
            "task": "t2v",
            "timestamp_utc": _utc_timestamp(),
            "git_commit": _git_commit(repo_root),
            "system": {
                "platform": platform.platform(),
                "python": sys.version.replace("\n", " "),
                "device_type": device_type,
            },
            "prompt_set": prompt_data["prompt_set"],
            "config": {
                "model": args.model,
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "fps": args.fps,
                "num_inference_steps": args.num_inference_steps,
                "negative_prompt": args.negative_prompt,
                "guidance_scale": args.guidance_scale,
                "guidance_scale_high": args.guidance_scale_high,
                "num_outputs_per_prompt": args.num_outputs_per_prompt,
                "num_gpus": args.num_gpus,
                "boundary_ratio": args.boundary_ratio,
                "flow_shift": args.flow_shift,
                "cache_backend": args.cache_backend,
                "cache_config_json": args.cache_config_json,
                "include_export_in_e2e": bool(args.include_export_in_e2e),
            },
            "run": {
                "warmup_requests": args.warmup_requests,
                "measured_requests": args.requests,
                "batch_size": args.batch_size,
                "base_seed": args.base_seed,
            },
            "metrics_summary": {
                "e2e_latency_ms": (_summary_stats(latencies_ms) if latencies_ms else None),
                "engine_ms": (_summary_stats(engine_ms) if engine_ms else None),
                "preprocess_ms": (_summary_stats(pre_ms) if pre_ms else None),
                "postprocess_ms": (_summary_stats(post_ms) if post_ms else None),
                "export_ms": (_summary_stats(export_ms) if export_ms else None),
                "throughput_videos_per_s": (float(total_videos / total_time_s) if total_time_s > 0 else None),
                "throughput_frames_per_s": (float(total_frames / total_time_s) if total_time_s > 0 else None),
                "peak_vram_mib": (_summary_stats(vram_samples) if vram_samples else None),
                "failure_rate": float(len(failures) / len(results)) if results else None,
                "oom_rate": float(len(oom_failures) / len(results)) if results else None,
            },
        }

        out_json_path = Path(args.output_json).resolve()
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"Wrote summary JSON to {out_json_path}")
        if jsonl_path is not None:
            print(f"Wrote per-request JSONL to {jsonl_path}")
        if export_dir is not None:
            print(f"Exported sample videos to {export_dir}")
    finally:
        try:
            omni.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

