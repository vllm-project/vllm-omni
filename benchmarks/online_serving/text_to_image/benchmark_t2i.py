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


DEFAULT_T2I_PROMPTS: list[str] = [
    "a cup of coffee on the table",
    "a photo of a corgi wearing sunglasses, studio lighting",
    "a cinematic landscape with mountains and fog, ultra wide angle",
    "a watercolor painting of cherry blossoms in spring, soft pastel",
    "a futuristic city at night with neon signs, rain, reflections",
    "a macro shot of a ladybug on a leaf, shallow depth of field",
    "a cozy reading nook with warm lamp light, detailed interior",
    "an astronaut riding a horse on the moon, surreal",
    "a robot chef preparing sushi, 3d render, high detail",
    "a vintage poster of a seaside town, art deco style",
    "一只橘猫在窗台上晒太阳，温暖的午后，写实风格",
    "未来感机甲站在沙漠中，夕阳逆光，电影质感",
    "赛博朋克街道，雨夜霓虹，路面反光，细节丰富",
    "传统水墨画风格的山水，云雾缭绕，意境悠远",
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
        prompts = list(DEFAULT_T2I_PROMPTS)
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


@dataclass
class _RequestResult:
    request_idx: int
    batch_size: int
    images_generated: int
    success: bool
    error: Optional[str]
    timings_ms: dict[str, float]
    peak_vram_mib: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DiT Text-to-Image generation via vLLM-Omni (in-process).")

    parser.add_argument("--model", default="Qwen/Qwen-Image", help="Diffusion model name or local path.")
    parser.add_argument("--prompt", default=None, help="Single prompt (overrides prompt set).")
    parser.add_argument("--prompt-file", type=str, default=None, help="Text file with one prompt per line.")
    parser.add_argument("--num-prompts", type=int, default=None, help="Use only the first N prompts from the set.")

    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Denoising steps.")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="Qwen-Image true CFG scale.")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt (optional).")
    parser.add_argument("--num-outputs-per-prompt", type=int, default=1, help="Images per prompt.")

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

    parser.add_argument("--warmup-requests", type=int, default=5, help="Warmup requests (not measured).")
    parser.add_argument("--requests", type=int, default=30, help="Measured requests.")
    parser.add_argument("--batch-size", type=int, default=1, help="Prompts per request.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed; increments per prompt.")

    parser.add_argument("--output-json", type=str, default="t2i_benchmark_summary.json", help="Summary JSON path.")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Optional per-request JSONL path.")

    parser.add_argument("--save-outputs-dir", type=str, default=None, help="Optional directory to save sample outputs.")
    parser.add_argument("--save-outputs-max", type=int, default=0, help="Max measured requests to save outputs for.")

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

        save_dir = Path(args.save_outputs_dir).resolve() if args.save_outputs_dir else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

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
                true_cfg_scale=args.true_cfg_scale,
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
            images = None

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
                images = engine.post_process_func(out.output)
                timings_ms["postprocess_ms"] = (time.perf_counter() - post_t0) * 1000.0

                _device_sync(device_type)
                timings_ms["e2e_ms"] = (time.perf_counter() - t0) * 1000.0

                success = images is not None
            except Exception as e:
                err = str(e)
                success = False
            finally:
                if mem_sampler is not None:
                    peak_vram_mib = mem_sampler.stop()

            is_measured = req_idx >= args.warmup_requests
            if not is_measured:
                continue

            save_ms = 0.0
            if (
                success
                and images is not None
                and save_dir is not None
                and args.save_outputs_max > 0
                and (req_idx - args.warmup_requests) < args.save_outputs_max
            ):
                save_t0 = time.perf_counter()
                # Images are expected to be a list of PIL images.
                for img_i, img in enumerate(images):
                    out_path = save_dir / f"req{req_idx:04d}_img{img_i:02d}.png"
                    img.save(out_path)
                save_ms = (time.perf_counter() - save_t0) * 1000.0

            if save_ms > 0:
                timings_ms["save_ms"] = save_ms

            images_generated = args.batch_size * args.num_outputs_per_prompt if success else 0
            result = _RequestResult(
                request_idx=req_idx - args.warmup_requests,
                batch_size=args.batch_size,
                images_generated=images_generated,
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
        vram_samples = [r.peak_vram_mib for r in results if r.success and r.peak_vram_mib is not None]

        failures = [r for r in results if not r.success]
        oom_failures = [r for r in failures if r.error and _is_oom_error(r.error)]

        total_images = sum(r.images_generated for r in results)
        total_time_s = sum((r.timings_ms.get("e2e_ms", 0.0) / 1000.0) for r in results if r.success)

        summary = {
            "task": "t2i",
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
                "num_inference_steps": args.num_inference_steps,
                "true_cfg_scale": args.true_cfg_scale,
                "negative_prompt": args.negative_prompt,
                "num_outputs_per_prompt": args.num_outputs_per_prompt,
                "num_gpus": args.num_gpus,
                "cache_backend": args.cache_backend,
                "cache_config_json": args.cache_config_json,
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
                "throughput_images_per_s": (float(total_images / total_time_s) if total_time_s > 0 else None),
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
        if save_dir is not None:
            print(f"Saved sample outputs to {save_dir}")
    finally:
        try:
            omni.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
