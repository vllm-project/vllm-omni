# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM-Omni offline benchmark for GLM-Image.

Supports T2I and I2I modes with the prompt.json dataset.
Downloads source images for I2I from image_url on first run and caches locally.

Usage:
    # T2I mode
    python benchmarks/glm_image/vllm-omni/inference.py \
        --model-path zai-org/GLM-Image \
        --mode t2i --num-prompts 10

    # I2I mode (downloads source images)
    python benchmarks/glm_image/vllm-omni/inference.py \
        --model-path zai-org/GLM-Image \
        --mode i2i --num-prompts 10
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_JSON = BENCHMARK_DIR / "prompt" / "prompt.json"
IMAGE_CACHE_DIR = BENCHMARK_DIR / "prompt" / "images"
DEFAULT_DEPLOY_CONFIG = "vllm_omni/deploy/glm_image.yaml"

DATASET_REPO = "JaredforReal/glm-image-bench"
DATASET_FILE = "prompt.json"


def _ensure_prompt_json(dataset_path: str | None) -> str:
    """Return path to prompt.json, downloading from HuggingFace if needed."""
    if dataset_path:
        return dataset_path
    local = DEFAULT_PROMPT_JSON
    if local.exists():
        return str(local)
    print(f"Downloading {DATASET_FILE} from {DATASET_REPO} ...")
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=DATASET_FILE,
            repo_type="dataset",
        )
        local.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(downloaded, local)
        print(f"Saved to {local}")
    except ImportError:
        url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{DATASET_FILE}"
        import urllib.request

        local.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, local)
        print(f"Saved to {local}")
    return str(local)


SEED = 42
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 1.5

GLM_IMAGE_EOS_TOKEN_ID = 16385
GLM_IMAGE_VISION_VOCAB_SIZE = 16512


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_dataset(
    dataset_path: str | None,
    mode: str,
    num_prompts: int,
) -> list[dict]:
    path = _ensure_prompt_json(dataset_path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    items = []
    for entry in raw:
        prompt_key = "t2i_prompt" if mode == "t2i" else "i2i_prompt"
        prompt_text = entry.get(prompt_key, "").strip()
        if not prompt_text:
            continue

        item = {"prompt": prompt_text}
        if mode == "i2i":
            item["image_url"] = entry.get("image_url", "")
        items.append(item)

    if num_prompts and len(items) > num_prompts:
        items = items[:num_prompts]
    return items


def download_image(url: str, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = url.rsplit("/", 1)[-1]
    local_path = cache_dir / fname
    if local_path.exists():
        return str(local_path)
    print(f"  Downloading {url} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    local_path.write_bytes(resp.content)
    return str(local_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_max_tokens(height: int, width: int, is_i2i: bool = False) -> int:
    factor = 32
    token_h = height // factor
    token_w = width // factor
    large_tokens = token_h * token_w

    # Small preview tokens (half resolution in each dimension)

    ratio = token_h / token_w if token_w > 0 else 1.0
    small_token_h = max(1, int(math.sqrt(ratio) * (factor // 2)))
    small_token_w = max(1, int(math.sqrt(1 / ratio) * (factor // 2)))
    small_tokens = small_token_h * small_token_w

    # Mode-dependent totals:
    # - t2i: small + large + EOS
    # - i2i: large + EOS
    if is_i2i:
        return large_tokens + 1
    return small_tokens + large_tokens + 1


def build_prompt_t2i(prompt: str, height: int, width: int, **gen_kw) -> dict:
    return {
        "prompt": prompt,
        "height": height,
        "width": width,
        "mm_processor_kwargs": {"target_h": height, "target_w": width},
        **gen_kw,
    }


def build_prompt_i2i(prompt: str, image_path: str, height: int, width: int, **gen_kw) -> dict:
    return {
        "prompt": prompt,
        "height": height,
        "width": width,
        "mm_processor_kwargs": {"target_h": height, "target_w": width},
        "multi_modal_data": {"image": Image.open(image_path).convert("RGB")},
        **gen_kw,
    }


def resolve_deploy_config(args: argparse.Namespace) -> str:
    if args.deploy_config:
        return args.deploy_config
    if os.path.exists(DEFAULT_DEPLOY_CONFIG):
        return DEFAULT_DEPLOY_CONFIG
    fallback = Path(__file__).resolve().parents[3] / DEFAULT_DEPLOY_CONFIG
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("Deploy config not found. Specify --deploy-config.")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def benchmark(args: argparse.Namespace) -> None:
    is_i2i = args.mode == "i2i"

    print("=" * 60)
    print("GLM-Image vLLM-Omni Benchmark")
    print(f"Mode: {args.mode}  |  Model: {args.model_path}")
    print(f"Size: {args.height}x{args.width}  |  Steps: {args.num_inference_steps}")
    print("=" * 60)

    # Load dataset
    items = load_dataset(args.dataset_path, args.mode, args.num_prompts)
    if not items:
        print("No prompts loaded. Exiting.")
        return
    print(f"Loaded {len(items)} prompts for {args.mode} mode")

    # Download I2I source images
    if is_i2i:
        print("Preparing source images...")
        for item in items:
            url = item.get("image_url", "")
            if url:
                item["image_path"] = download_image(url, IMAGE_CACHE_DIR)
            else:
                item["image_path"] = None

    # Init Omni
    deploy_config = resolve_deploy_config(args)
    print(f"\nInitializing vLLM-Omni (deploy config: {deploy_config}) ...")
    t0 = time.perf_counter()

    omni = Omni(
        model=args.model_path,
        deploy_config=deploy_config,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        enable_ar_profiler=args.enable_ar_profiler,
    )

    init_time = time.perf_counter() - t0
    print(f"Initialized in {init_time:.2f}s")

    # Sampling params
    max_tokens = compute_max_tokens(args.height, args.width, is_i2i=is_i2i)
    ar_params = SamplingParams(
        temperature=0.9,
        top_p=0.75,
        top_k=GLM_IMAGE_VISION_VOCAB_SIZE,
        max_tokens=max_tokens,
        stop_token_ids=[GLM_IMAGE_EOS_TOKEN_ID],
        seed=args.seed,
        detokenize=False,
        extra_args={"target_h": args.height, "target_w": args.width},
    )
    diff_params = OmniDiffusionSamplingParams(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    sampling_params_list = [ar_params, diff_params]

    # Build all prompts
    gen_kw = {
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
    }
    all_prompts = []
    for item in items:
        if is_i2i:
            img_path = item.get("image_path")
            if not img_path or not os.path.exists(img_path):
                continue
            all_prompts.append(build_prompt_i2i(item["prompt"], img_path, args.height, args.width, **gen_kw))
        else:
            all_prompts.append(build_prompt_t2i(item["prompt"], args.height, args.width, **gen_kw))

    valid = len(all_prompts)
    print(f"Valid prompts: {valid}")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Warmup: run 1 request to prime caches, CUDA graphs, etc.
    if all_prompts:
        print("Running warmup request...")
        try:
            warmup_prompt = [all_prompts[0]]
            omni.generate(warmup_prompt, sampling_params_list, py_generator=False)
            print("Warmup done.\n")
        except Exception as e:
            print(f"Warmup failed (continuing): {e}")

    # Run
    print(f"\nRunning {valid} requests...")
    print("-" * 60)

    latencies = []
    all_stage_durations: list[dict[str, float]] = []
    success = 0
    failed = 0
    wall_start = time.perf_counter()

    try:
        output_idx = 0
        for stage_outputs in omni.generate(all_prompts, sampling_params_list, py_generator=True):
            if stage_outputs.final_output_type == "image":
                request_output = stage_outputs.request_output
                request_id = getattr(request_output, "request_id", "")

                images = getattr(request_output, "images", [])
                if not images and hasattr(request_output, "multimodal_output"):
                    mm = request_output.multimodal_output
                    if isinstance(mm, dict):
                        images = mm.get("images", [])

                elapsed = time.perf_counter() - wall_start
                if images:
                    for img in images:
                        if isinstance(img, Image.Image):
                            out_path = os.path.join(args.output_dir, f"{output_idx:04d}.png")
                            img.save(out_path)
                    success += 1
                    latencies.append(elapsed)
                    stage_durations = getattr(stage_outputs, "stage_durations", {})
                    if stage_durations:
                        all_stage_durations.append(stage_durations)
                    # Show wall-clock elapsed and pipeline breakdown if available
                    preprocess_str = ""
                    if "preprocess_ms" in stage_durations:
                        preprocess_str = f" preprocess={stage_durations['preprocess_ms'] / 1000.0:.2f}s"
                    print(f"  [{success}/{valid}] id={request_id[:8]} {elapsed:.2f}s{preprocess_str}")
                    output_idx += 1
                else:
                    failed += 1
    except Exception as e:
        print(f"Error: {e}")
        failed = valid - success

    total_gen_time = time.perf_counter() - wall_start

    # Diff stage_0_gen_ms with previous request to remove accumulated wait time.
    # stage_0_gen_ms is measured from submit_ts (same for all requests submitted
    # at once), so it accumulates queue/scheduling overhead across requests.
    # Other stages and pipeline timings are per-request already.
    _TIMING_ORDER = [
        "preprocess_ms",
        "stage_0_gen_ms",
        "ar2diffusion_ms",
        "stage_1_gen_ms",
        "queue_wait_ms",
    ]

    per_request_actual: list[dict[str, float]] = []
    prev_stage_0_ms = 0.0
    for sd in all_stage_durations:
        actual = dict(sd)
        s0 = sd.get("stage_0_gen_ms", 0.0)
        actual["stage_0_gen_ms"] = s0 - prev_stage_0_ms
        prev_stage_0_ms = s0
        per_request_actual.append(actual)

    per_request_e2e_ms: list[float] = []
    for actual in per_request_actual:
        e2e_ms = sum(v for k, v in actual.items() if k in _TIMING_ORDER)
        if e2e_ms > 0:
            per_request_e2e_ms.append(e2e_ms)

    # Report
    print("\n" + "=" * 60)
    print("vLLM-Omni Benchmark Results")
    print("=" * 60)
    print(f"{'Mode:':<40} {args.mode}")
    print(f"{'Model:':<40} {args.model_path}")
    print(f"{'Image size:':<40} {args.height}x{args.width}")
    print(f"{'Num inference steps:':<40} {args.num_inference_steps}")
    print("-" * 50)
    print(f"{'Init time (s):':<40} {init_time:.2f}")
    print(f"{'Successful:':<40} {success}/{valid}")
    print(f"{'Failed:':<40} {failed}")
    print("-" * 50)

    if per_request_e2e_ms:
        per_request_s = np.array(per_request_e2e_ms) / 1000.0
        print(f"{'Total generation time (s):':<40} {total_gen_time:.2f}")
        print(f"{'Throughput (img/s):':<40} {success / total_gen_time:.4f}")
        print(f"{'Latency Mean (s):':<40} {per_request_s.mean():.4f}")
        print(f"{'Latency Median (s):':<40} {np.median(per_request_s):.4f}")
        print(f"{'Latency P95 (s):':<40} {np.percentile(per_request_s, 95):.4f}")
        print(f"{'Latency P99 (s):':<40} {np.percentile(per_request_s, 99):.4f}")
        print(f"{'Latency Min (s):':<40} {per_request_s.min():.4f}")
        print(f"{'Latency Max (s):':<40} {per_request_s.max():.4f}")
    elif latencies:
        per_request = np.diff([0.0] + list(latencies))
        print(f"{'Total generation time (s):':<40} {total_gen_time:.2f}")
        print(f"{'Throughput (img/s):':<40} {success / total_gen_time:.4f}")
        print(f"{'Latency Mean (s) [wall-clock]:':<40} {per_request.mean():.4f}")
        print(f"{'Latency Median (s) [wall-clock]:':<40} {np.median(per_request):.4f}")
        print(f"{'Latency P95 (s) [wall-clock]:':<40} {np.percentile(per_request, 95):.4f}")
        print(f"{'Latency P99 (s) [wall-clock]:':<40} {np.percentile(per_request, 99):.4f}")
        print(f"{'Latency Min (s) [wall-clock]:':<40} {per_request.min():.4f}")
        print(f"{'Latency Max (s) [wall-clock]:':<40} {per_request.max():.4f}")

    if per_request_actual:
        print("-" * 50)
        print("Pipeline Timings Mean:")
        for key in _TIMING_ORDER:
            vals = [d.get(key, 0.0) for d in per_request_actual]
            if any(v != 0 for v in vals):
                unit = "ms" if key.endswith("_ms") else "s"
                print(f"  {key + ':':<38} {np.mean(vals):.4f} ({unit})")
        # Show any extra keys not in the ordered list
        ordered_set = set(_TIMING_ORDER)
        extra_keys = sorted(k for k in per_request_actual[0].keys() if k not in ordered_set)
        for key in extra_keys:
            vals = [d.get(key, 0.0) for d in per_request_actual]
            if any(v != 0 for v in vals):
                unit = "ms" if key.endswith("_ms") else "s"
                print(f"  {key + ':':<38} {np.mean(vals):.4f} ({unit})")

    print(f"\n{'Output dir:':<40} {args.output_dir}")
    print("=" * 60)

    # Metrics JSON
    metrics = {
        "backend": "vllm-omni",
        "mode": args.mode,
        "model": args.model_path,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "init_time_s": init_time,
        "completed_requests": success,
        "failed_requests": failed,
        "total_gen_time_s": total_gen_time,
        "throughput_qps": success / total_gen_time if total_gen_time > 0 else 0,
    }
    if per_request_e2e_ms:
        per_request_s = np.array(per_request_e2e_ms) / 1000.0
        metrics["latency_mean"] = float(per_request_s.mean())
        metrics["latency_median"] = float(np.median(per_request_s))
        metrics["latency_p95"] = float(np.percentile(per_request_s, 95))
        metrics["latency_p99"] = float(np.percentile(per_request_s, 99))
    elif latencies:
        per_request = np.diff([0.0] + list(latencies))
        metrics["latency_mean"] = float(per_request.mean())
        metrics["latency_median"] = float(np.median(per_request))
        metrics["latency_p95"] = float(np.percentile(per_request, 95))
        metrics["latency_p99"] = float(np.percentile(per_request, 99))
    else:
        metrics["latency_mean"] = 0
        metrics["latency_median"] = 0
        metrics["latency_p95"] = 0
        metrics["latency_p99"] = 0
    if per_request_actual:
        all_keys = list(_TIMING_ORDER) + sorted(k for k in per_request_actual[0].keys() if k not in set(_TIMING_ORDER))
        stage_metrics = {}
        for key in all_keys:
            vals = [d.get(key, 0.0) for d in per_request_actual]
            stage_metrics[key] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p95": float(np.percentile(vals, 95)),
            }
        metrics["stage_durations"] = stage_metrics
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output_file}")

    omni.close()
    print("Done!")


def main() -> None:
    parser = argparse.ArgumentParser(description="GLM-Image vLLM-Omni offline benchmark")
    parser.add_argument("--model-path", type=str, default="zai-org/GLM-Image")
    parser.add_argument("--deploy-config", type=str, default=None, help="Deploy config YAML")
    parser.add_argument("--mode", type=str, default="t2i", choices=["t2i", "i2i"])
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to prompt.json")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default="benchmarks/glm_image/vllm-omni/outputs")
    parser.add_argument("--output-file", type=str, default=None, help="JSON file for metrics")
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler for stage-level timing",
    )
    parser.add_argument(
        "--enable-ar-profiler",
        action="store_true",
        help="Enable AR stage profiler to include AR timing in stage_durations",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Enable detailed per-request pipeline stats logging",
    )
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
