# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Online serving benchmark for GLM-Image (T2I and I2I modes).

Sends requests to the /v1/chat/completions endpoint and reports end-to-end
latency, throughput, and per-stage durations (when the server is started with
--enable-diffusion-pipeline-profiler and/or --enable-ar-profiler).

Supports three dataset types:
  - prompt:   Use prompt.json (default). T2I uses t2i_prompt, I2I uses i2i_prompt
              and sends source images from image_url.
  - random:   Generate synthetic prompts (and random images for I2I).
  - custom:   Load from a user-specified JSON file.

Usage:
    # T2I with prompt.json (default)
    python benchmarks/glm_image/benchmark_glm_image.py \
        --mode t2i --num-prompts 10

    # I2I with prompt.json (downloads source images automatically)
    python benchmarks/glm_image/benchmark_glm_image.py \
        --mode i2i --num-prompts 10

    # Random dataset
    python benchmarks/glm_image/benchmark_glm_image.py \
        --mode t2i --dataset random --num-prompts 20

    # Custom dataset
    python benchmarks/glm_image/benchmark_glm_image.py \
        --mode i2i --dataset custom \
        --dataset-path my_prompts.json --num-prompts 5
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import requests as sync_requests
from PIL import Image
from tqdm.asyncio import tqdm

# Import backends from the diffusion benchmark (add parent dirs to path)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "diffusion"))
from backends import RequestFuncOutput

BENCHMARK_DIR = Path(__file__).resolve().parent
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class GLMImageRequest:
    prompt: str
    image_path: str | None = None  # Only for I2I mode


def download_image(url: str) -> str:
    """Download an image to cache and return the local path."""
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = url.rsplit("/", 1)[-1]
    local_path = IMAGE_CACHE_DIR / fname
    if local_path.exists():
        return str(local_path)
    resp = sync_requests.get(url, timeout=30)
    resp.raise_for_status()
    local_path.write_bytes(resp.content)
    return str(local_path)


def encode_image_as_data_url(path: str) -> str:
    """Encode a local image file as a base64 data URL."""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(path).suffix.lower()
    mime = {"png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")
    return f"data:{mime};base64,{encoded}"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class PromptDataset:
    """Load from prompt.json. T2I uses t2i_prompt, I2I uses i2i_prompt + image_url."""

    def __init__(self, args: argparse.Namespace):
        path = _ensure_prompt_json(args.dataset_path)
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        prompt_key = "t2i_prompt" if args.mode == "t2i" else "i2i_prompt"
        self.items: list[GLMImageRequest] = []

        for entry in raw:
            prompt = entry.get(prompt_key, "").strip()
            if not prompt:
                continue
            image_path = None
            if args.mode == "i2i":
                url = entry.get("image_url", "")
                if url:
                    image_path = download_image(url)
            self.items.append(GLMImageRequest(prompt=prompt, image_path=image_path))

        if args.num_prompts and len(self.items) > args.num_prompts:
            self.items = self.items[: args.num_prompts]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> GLMImageRequest:
        return self.items[idx]

    def get_requests(self) -> list[GLMImageRequest]:
        return list(self.items)


class RandomDataset:
    """Generate synthetic prompts (and optional random images for I2I)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.num_prompts = args.num_prompts
        self._random_image_paths: list[str] | None = None
        if args.mode == "i2i":
            self._random_image_paths = self._generate_random_images()

    def _generate_random_images(self) -> list[str]:
        paths: list[str] = []
        for i in range(self.args.num_input_images):
            img = Image.new("RGB", (512, 512), (128 + i * 30 % 128, 64, 192))
            path = os.path.join(tempfile.gettempdir(), f"glm_image_bench_input_{i}.png")
            img.save(path)
            paths.append(path)
        return paths

    def __len__(self) -> int:
        return self.num_prompts

    def __getitem__(self, idx: int) -> GLMImageRequest:
        image_path = None
        if self._random_image_paths is not None:
            image_path = self._random_image_paths[idx % len(self._random_image_paths)]
        return GLMImageRequest(
            prompt=f"A beautiful scene with vivid colors and intricate details, prompt {idx}",
            image_path=image_path,
        )

    def get_requests(self) -> list[GLMImageRequest]:
        return [self[i] for i in range(len(self))]


class CustomDataset:
    """Load from a user-specified JSON file.

    Expected format:
    [
        {"prompt": "A cat sitting on a windowsill"},
        {"prompt": "Make it look like winter", "image_path": "/path/to/img.png"}
    ]
    """

    def __init__(self, args: argparse.Namespace):
        if not args.dataset_path:
            raise ValueError("--dataset-path is required for custom dataset")
        with open(args.dataset_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.items: list[GLMImageRequest] = []
        for item in raw:
            self.items.append(
                GLMImageRequest(
                    prompt=item.get("prompt", ""),
                    image_path=item.get("image_path"),
                )
            )
        if args.num_prompts and len(self.items) > args.num_prompts:
            self.items = self.items[: args.num_prompts]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> GLMImageRequest:
        return self.items[idx]

    def get_requests(self) -> list[GLMImageRequest]:
        return list(self.items)


# ---------------------------------------------------------------------------
# Async request for GLM-Image (chat completions with image support)
# ---------------------------------------------------------------------------


async def async_glm_image_request(
    req: GLMImageRequest,
    api_url: str,
    model: str,
    session: aiohttp.ClientSession,
    pbar: Any,
    args: argparse.Namespace,
) -> RequestFuncOutput:
    """Send a single T2I or I2I request via chat completions endpoint."""
    output = RequestFuncOutput()
    output.start_time = time.perf_counter()

    # Build messages
    if req.image_path and args.mode == "i2i":
        data_url = encode_image_as_data_url(req.image_path)
        content = [
            {"type": "text", "text": req.prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    else:
        content = req.prompt

    messages = [{"role": "user", "content": content}]

    extra_body: dict[str, Any] = {}
    if args.height:
        extra_body["height"] = args.height
    if args.width:
        extra_body["width"] = args.width
    if args.num_inference_steps:
        extra_body["num_inference_steps"] = args.num_inference_steps
    if args.seed is not None:
        extra_body["seed"] = args.seed

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if extra_body:
        payload["extra_body"] = extra_body

    try:
        async with session.post(api_url, json=payload) as response:
            if response.status == 200:
                resp_json = await response.json()
                output.response_body = resp_json
                output.success = True
                try:
                    choices = resp_json.get("choices", [])
                    if choices and isinstance(choices, list):
                        msg = choices[0].get("message", {})
                        if isinstance(msg, dict):
                            resp_content = msg.get("content", [])
                            if resp_content and isinstance(resp_content, list) and len(resp_content) > 0:
                                first_item = resp_content[0]
                                if isinstance(first_item, dict):
                                    output.stage_durations = first_item.get("stage_durations") or {}
                                    output.peak_memory_mb = first_item.get("peak_memory_mb", 0.0)
                except (IndexError, TypeError, AttributeError):
                    pass
            else:
                output.error = f"HTTP {response.status}: {await response.text()}"
                output.success = False
    except Exception as e:
        output.error = str(e)
        output.success = False

    output.latency = time.perf_counter() - output.start_time
    if pbar:
        pbar.update(1)
    return output


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


async def iter_requests(n: int, request_rate: float) -> Any:
    import random as _random

    for i in range(n):
        if request_rate != float("inf") and i > 0:
            await asyncio.sleep(_random.expovariate(request_rate))
        yield i


def calculate_metrics(outputs: list[RequestFuncOutput], total_duration: float) -> dict[str, Any]:
    success = [o for o in outputs if o.success]
    errors = [o for o in outputs if not o.success]
    latencies = [o.latency for o in success]
    peak_mems = [o.peak_memory_mb for o in success if o.peak_memory_mb > 0]

    stage_duration_lists: dict[str, list[float]] = {}
    for o in success:
        for stage, dur in (o.stage_durations or {}).items():
            stage_duration_lists.setdefault(stage, []).append(dur)

    return {
        "duration": total_duration,
        "completed_requests": len(success),
        "failed_requests": len(errors),
        "throughput_qps": len(success) / total_duration if total_duration > 0 else 0,
        "latency_mean": float(np.mean(latencies)) if latencies else 0,
        "latency_median": float(np.median(latencies)) if latencies else 0,
        "latency_p99": float(np.percentile(latencies, 99)) if latencies else 0,
        "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0,
        "peak_memory_mb_max": max(peak_mems) if peak_mems else 0,
        "stage_durations_mean": {s: float(np.mean(v)) for s, v in stage_duration_lists.items()},
        "stage_durations_p50": {s: float(np.percentile(v, 50)) for s, v in stage_duration_lists.items()},
    }


async def benchmark(args: argparse.Namespace) -> None:
    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"

    # Load dataset
    if args.dataset == "prompt":
        dataset = PromptDataset(args)
    elif args.dataset == "random":
        dataset = RandomDataset(args)
    elif args.dataset == "custom":
        dataset = CustomDataset(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    glm_requests = dataset.get_requests()
    print(f"Prepared {len(glm_requests)} requests (mode={args.mode}, dataset={args.dataset})")

    semaphore = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None

    async def limited_request(idx: int, req: GLMImageRequest, session: aiohttp.ClientSession, pbar: Any):
        if semaphore:
            async with semaphore:
                return await async_glm_image_request(req, api_url, args.model, session, pbar, args)
        return await async_glm_image_request(req, api_url, args.model, session, pbar, args)

    async with aiohttp.ClientSession() as session:
        # Warmup
        if args.warmup_requests and glm_requests:
            print(f"Running {args.warmup_requests} warmup request(s)...")
            for i in range(args.warmup_requests):
                await limited_request(i, glm_requests[i % len(glm_requests)], session, None)

        # Main benchmark
        pbar = tqdm(total=len(glm_requests), disable=args.disable_tqdm)
        start_time = time.perf_counter()
        tasks = []
        async for idx in iter_requests(len(glm_requests), args.request_rate):
            tasks.append(asyncio.create_task(limited_request(idx, glm_requests[idx], session, pbar)))
        outputs = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time
        pbar.close()

    # Metrics
    metrics = calculate_metrics(outputs, total_duration)
    metrics["mode"] = args.mode
    metrics["model"] = args.model
    metrics["dataset"] = args.dataset

    print(f"\n{' GLM-Image Online Benchmark Result ':=^60}")
    print(f"{'Mode:':<40} {args.mode}")
    print(f"{'Model:':<40} {args.model}")
    print(f"{'Dataset:':<40} {args.dataset}")
    print("-" * 50)
    print(f"{'Benchmark duration (s):':<40} {metrics['duration']:.2f}")
    print(f"{'Request rate:':<40} {args.request_rate}")
    print(f"{'Max concurrency:':<40} {args.max_concurrency}")
    print(f"{'Successful requests:':<40} {metrics['completed_requests']}/{len(glm_requests)}")
    print("-" * 50)
    print(f"{'Throughput (req/s):':<40} {metrics['throughput_qps']:.2f}")
    print(f"{'Latency Mean (s):':<40} {metrics['latency_mean']:.4f}")
    print(f"{'Latency Median (s):':<40} {metrics['latency_median']:.4f}")
    print(f"{'Latency P95 (s):':<40} {metrics['latency_p95']:.4f}")
    print(f"{'Latency P99 (s):':<40} {metrics['latency_p99']:.4f}")

    if metrics["peak_memory_mb_max"] > 0:
        print("-" * 50)
        print(f"{'Peak Memory Max (MB):':<40} {metrics['peak_memory_mb_max']:.2f}")

    if metrics["stage_durations_mean"]:
        print("-" * 50)
        print("Stage Durations Mean:")
        for stage, val in sorted(metrics["stage_durations_mean"].items()):
            unit = "ms" if stage.endswith("_ms") else "s"
            print(f"  {stage + ':':<38} {val:.4f} ({unit})")

    print("=" * 60)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GLM-Image T2I/I2I online serving.")
    parser.add_argument("--mode", type=str, default="t2i", choices=["t2i", "i2i"])
    parser.add_argument("--dataset", type=str, default="prompt", choices=["prompt", "random", "custom"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--num-input-images", type=int, default=1, help="For random I2I dataset.")
    args = parser.parse_args()
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
