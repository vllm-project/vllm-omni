# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HuggingFace (transformers + diffusers) baseline benchmark for GLM-Image.

Supports T2I and I2I modes with the prompt.json dataset.
Downloads source images for I2I from image_url on first run and caches locally.

Usage:
    # T2I mode (text-to-image, no source images needed)
    python benchmarks/glm_image/huggingface/inference.py \
        --model-path zai-org/GLM-Image \
        --mode t2i --num-prompts 10

    # I2I mode (image-to-image, downloads source images)
    python benchmarks/glm_image/huggingface/inference.py \
        --model-path zai-org/GLM-Image \
        --mode i2i --num-prompts 10

    # With custom prompt.json
    python benchmarks/glm_image/huggingface/inference.py \
        --model-path zai-org/GLM-Image \
        --mode i2i --dataset-path prompts.json --num-prompts 5
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_JSON = BENCHMARK_DIR / "prompt" / "prompt.json"
IMAGE_CACHE_DIR = BENCHMARK_DIR / "prompt" / "images"

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


HEIGHT = 1024
WIDTH = 1024
SEED = 42
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 1.5


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_dataset(
    dataset_path: str | None,
    mode: str,
    num_prompts: int,
) -> list[dict]:
    """Load prompts from prompt.json and prepare per-request data."""
    path = _ensure_prompt_json(dataset_path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    items = []
    for entry in raw:
        if mode == "t2i":
            prompt_key = "t2i_prompt"
        else:
            prompt_key = "i2i_prompt"

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
    """Download an image to cache_dir and return the local path."""
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
# Benchmark
# ---------------------------------------------------------------------------


def benchmark(args: argparse.Namespace) -> None:
    from diffusers.pipelines.glm_image import GlmImagePipeline

    print("=" * 60)
    print("GLM-Image HuggingFace Baseline Benchmark")
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
    if args.mode == "i2i":
        print("Preparing source images...")
        for item in items:
            url = item.get("image_url", "")
            if url:
                item["image_path"] = download_image(url, IMAGE_CACHE_DIR)
            else:
                item["image_path"] = None

    # Load pipeline
    print(f"\nLoading pipeline from {args.model_path} ...")
    t0 = time.perf_counter()
    pipe = GlmImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    init_time = time.perf_counter() - t0
    print(f"Pipeline loaded in {init_time:.2f}s")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Run benchmark
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    latencies = []
    success = 0
    failed = 0

    print(f"\nRunning {len(items)} requests sequentially...")
    print("-" * 60)

    for i, item in enumerate(items):
        prompt = item["prompt"]
        gen_kwargs: dict = {
            "prompt": prompt,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "generator": generator,
        }

        if args.mode == "i2i":
            img_path = item.get("image_path")
            if img_path and os.path.exists(img_path):
                gen_kwargs["image"] = [Image.open(img_path).convert("RGB")]
            else:
                print(f"  [{i + 1}] SKIP: no source image")
                failed += 1
                continue

        t_start = time.perf_counter()
        try:
            result = pipe(**gen_kwargs)
            image = result.images[0]
            elapsed = time.perf_counter() - t_start
            latencies.append(elapsed)
            success += 1

            out_path = os.path.join(args.output_dir, f"{i:04d}.png")
            image.save(out_path)
            print(f"  [{i + 1}/{len(items)}] {elapsed:.3f}s -> {out_path}")
        except Exception as e:
            elapsed = time.perf_counter() - t_start
            failed += 1
            print(f"  [{i + 1}/{len(items)}] FAILED ({elapsed:.3f}s): {e}")

    # Report
    total_gen_time = sum(latencies) if latencies else 0
    print("\n" + "=" * 60)
    print("HuggingFace Baseline Results")
    print("=" * 60)
    print(f"{'Mode:':<40} {args.mode}")
    print(f"{'Model:':<40} {args.model_path}")
    print(f"{'Image size:':<40} {args.height}x{args.width}")
    print(f"{'Num inference steps:':<40} {args.num_inference_steps}")
    print("-" * 50)
    print(f"{'Pipeline init time (s):':<40} {init_time:.2f}")
    print(f"{'Successful:':<40} {success}/{len(items)}")
    print(f"{'Failed:':<40} {failed}")
    print("-" * 50)
    if latencies:
        arr = np.array(latencies)
        print(f"{'Total generation time (s):':<40} {total_gen_time:.2f}")
        print(f"{'Throughput (img/s):':<40} {success / total_gen_time:.4f}")
        print(f"{'Latency Mean (s):':<40} {arr.mean():.4f}")
        print(f"{'Latency Median (s):':<40} {np.median(arr):.4f}")
        print(f"{'Latency P95 (s):':<40} {np.percentile(arr, 95):.4f}")
        print(f"{'Latency P99 (s):':<40} {np.percentile(arr, 99):.4f}")

    print(f"\n{'Output dir:':<40} {args.output_dir}")
    print("=" * 60)

    # Save metrics JSON
    metrics = {
        "backend": "huggingface",
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
        "latency_mean": float(np.mean(latencies)) if latencies else 0,
        "latency_median": float(np.median(latencies)) if latencies else 0,
        "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0,
        "latency_p99": float(np.percentile(latencies, 99)) if latencies else 0,
    }
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GLM-Image HuggingFace baseline benchmark")
    parser.add_argument("--model-path", type=str, default="zai-org/GLM-Image")
    parser.add_argument("--mode", type=str, default="t2i", choices=["t2i", "i2i"])
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to prompt.json")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default="benchmarks/glm_image/huggingface/outputs")
    parser.add_argument("--output-file", type=str, default=None, help="JSON file for metrics")
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
