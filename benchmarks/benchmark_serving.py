# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark online serving for vLLM-Omni diffusion models via OpenAI-style
/v1/chat/completions.

Examples:
  # Text-to-Image
  python benchmarks/benchmark_serving.py --task t2i --model Qwen-Image --num-prompts 20

  # Image-to-Image (VBench I2V dataset provides images + captions)
  python benchmarks/benchmark_serving.py --task i2i --dataset vbench --num-prompts 20

  # Text-to-Video
  python benchmarks/benchmark_serving.py --task t2v --model Wan2.2 --num-prompts 20 --num-frames 81

  # Image-to-Video
  python benchmarks/benchmark_serving.py --task i2v --dataset vbench --num-prompts 20 --num-frames 81
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import json
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

import aiohttp
try:
    from tqdm.asyncio import tqdm
except Exception:  # pragma: no cover - optional dependency
    class _NoopTqdm:
        def __init__(self, *args, **kwargs) -> None:
            return

        def update(self, n: int = 1) -> None:
            return

        def close(self) -> None:
            return

    def tqdm(*args, **kwargs):  # type: ignore[no-redef]
        return _NoopTqdm()


DUMMY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2+"
    "8mIAAAAASUVORK5CYII="
)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    model: Optional[str]
    extra_body: Dict[str, Any] = field(default_factory=dict)
    image_paths: Optional[List[str]] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    error: str = ""
    start_time: float = 0.0
    response_body: Dict[str, Any] = field(default_factory=dict)
    output_count: Optional[int] = None


class BaseDataset(ABC):
    def __init__(self, args, api_url: str, model: Optional[str]):
        self.args = args
        self.api_url = api_url
        self.model = model

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> RequestFuncInput:
        pass

    @abstractmethod
    def get_requests(self) -> List[RequestFuncInput]:
        pass


class VBenchDataset(BaseDataset):
    """
    Dataset loader for VBench prompts.
    Supports text-only (t2i/t2v) and image+text (i2i/i2v).
    """

    T2V_PROMPT_URL = (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/prompts/"
        "prompts_per_dimension/subject_consistency.txt"
    )
    I2V_DOWNLOAD_SCRIPT_URL = (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/vbench2_beta_i2v/"
        "download_data.sh"
    )

    def __init__(self, args, api_url: str, model: Optional[str]):
        super().__init__(args, api_url, model)
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vllm_omni")
        self.items = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        if self.args.task in {"t2i", "t2v"}:
            return self._load_t2v_prompts()
        return self._load_i2v_data()

    def _download_text(self, url: str, dest_path: str) -> None:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with urlopen(url) as resp:
            data = resp.read().decode("utf-8")
        Path(dest_path).write_text(data, encoding="utf-8")

    def _load_t2v_prompts(self) -> List[Dict[str, Any]]:
        path = self.args.dataset_path

        if not path:
            path = os.path.join(self.cache_dir, "vbench_subject_consistency.txt")
            if not os.path.exists(path):
                print(f"Downloading VBench prompts to {path}...")
                try:
                    self._download_text(self.T2V_PROMPT_URL, path)
                except Exception as e:
                    print(f"Failed to download VBench prompts: {e}")
                    return [{"prompt": "A cat sitting on a bench"}] * 50

        prompts = []
        raw = Path(path).read_text(encoding="utf-8")
        for line in raw.splitlines():
            line = line.strip()
            if line:
                prompts.append({"prompt": line})

        return self._resize_data(prompts)

    def _auto_download_i2v_dataset(self) -> Optional[str]:
        vbench_i2v_dir = os.path.join(self.cache_dir, "vbench_i2v", "vbench2_beta_i2v")
        info_json_path = os.path.join(vbench_i2v_dir, "data", "i2v-bench-info.json")

        if os.path.exists(info_json_path):
            return vbench_i2v_dir

        if os.name == "nt":
            print("Auto-download for VBench I2V is not supported on Windows.")
            print("Please download the dataset and pass --dataset-path.")
            return None

        print(f"Downloading VBench I2V dataset to {vbench_i2v_dir}...")
        try:
            cache_root = os.path.join(self.cache_dir, "vbench_i2v")
            script_path = os.path.join(cache_root, "download_data.sh")
            self._download_text(self.I2V_DOWNLOAD_SCRIPT_URL, script_path)
            os.chmod(script_path, 0o755)

            import subprocess

            result = subprocess.run(
                ["bash", script_path],
                cwd=cache_root,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Download script failed: {result.stderr}")
        except Exception as e:
            print(f"Failed to download VBench I2V dataset: {e}")
            print(
                "Please manually download following instructions at:\n"
                "https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v#22-download"
            )
            return None

        return vbench_i2v_dir if os.path.exists(info_json_path) else None

    def _load_from_i2v_json(self, json_path: str) -> List[Dict[str, Any]]:
        items = json.loads(Path(json_path).read_text(encoding="utf-8"))
        base_dir = os.path.dirname(os.path.dirname(json_path))
        origin_dir = os.path.join(base_dir, "data", "origin")

        data: List[Dict[str, Any]] = []
        for item in items:
            img_path = os.path.join(origin_dir, item.get("file_name", ""))
            if os.path.exists(img_path):
                data.append({"prompt": item.get("caption", ""), "image_path": img_path})
            else:
                print(f"Warning: Image not found: {img_path}")

        print(f"Loaded {len(data)} I2V samples from VBench I2V dataset")
        return data

    def _scan_directory_for_images(self, path: str) -> List[Dict[str, Any]]:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        files: List[str] = []

        for ext in exts:
            files.extend(glob.glob(os.path.join(path, ext)))
            files.extend(glob.glob(os.path.join(path, ext.upper())))
            origin_dir = os.path.join(path, "data", "origin")
            if os.path.exists(origin_dir):
                files.extend(glob.glob(os.path.join(origin_dir, ext)))
                files.extend(glob.glob(os.path.join(origin_dir, ext.upper())))

        return [
            {"prompt": os.path.splitext(os.path.basename(f))[0], "image_path": f}
            for f in files
        ]

    def _create_dummy_data(self) -> List[Dict[str, Any]]:
        print("No I2V data found. Using dummy placeholders.")
        dummy_path = _ensure_dummy_image(self.cache_dir)
        if not dummy_path:
            return []
        return [{"prompt": "A moving cat", "image_path": dummy_path}] * 10

    def _load_i2v_data(self) -> List[Dict[str, Any]]:
        path = self.args.dataset_path

        if not path:
            path = self._auto_download_i2v_dataset()
            if not path:
                return self._resize_data(self._create_dummy_data())

        info_json_candidates = [
            os.path.join(path, "data", "i2v-bench-info.json"),
            path if path.endswith(".json") else None,
        ]

        for json_path in info_json_candidates:
            if json_path and os.path.exists(json_path):
                try:
                    return self._resize_data(self._load_from_i2v_json(json_path))
                except Exception as e:
                    print(f"Failed to load {json_path}: {e}")

        if os.path.isdir(path):
            data = self._scan_directory_for_images(path)
            if data:
                return self._resize_data(data)

        return self._resize_data(self._create_dummy_data())

    def _resize_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.args.num_prompts:
            return data

        if len(data) < self.args.num_prompts:
            factor = (self.args.num_prompts // len(data)) + 1
            data = data * factor

        return data[: self.args.num_prompts]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestFuncInput:
        item = self.items[idx]
        image_paths = [item["image_path"]] if "image_path" in item else None
        return RequestFuncInput(
            prompt=item.get("prompt", ""),
            api_url=self.api_url,
            model=self.model,
            extra_body=dict(self.args.extra_body),
            image_paths=image_paths,
        )

    def get_requests(self) -> List[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


class RandomDataset(BaseDataset):
    def __init__(self, args, api_url: str, model: Optional[str]):
        super().__init__(args, api_url, model)
        self.num_prompts = args.num_prompts or 100
        self.needs_image = args.task in {"i2i", "i2v", "ti2i", "ti2v"}
        self.dummy_image_path = _ensure_dummy_image(os.path.join(os.path.expanduser("~"), ".cache", "vllm_omni"))

    def __len__(self) -> int:
        return self.num_prompts

    def __getitem__(self, idx: int) -> RequestFuncInput:
        image_paths = [self.dummy_image_path] if self.needs_image and self.dummy_image_path else None
        return RequestFuncInput(
            prompt=f"Random prompt {idx} for benchmarking diffusion models",
            api_url=self.api_url,
            model=self.model,
            extra_body=dict(self.args.extra_body),
            image_paths=image_paths,
        )

    def get_requests(self) -> List[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


def _ensure_dummy_image(cache_dir: str) -> Optional[str]:
    os.makedirs(cache_dir, exist_ok=True)
    dummy_image = os.path.join(cache_dir, "dummy_image.png")
    if not os.path.exists(dummy_image):
        try:
            Path(dummy_image).write_bytes(base64.b64decode(DUMMY_PNG_BASE64))
        except Exception as e:
            print(f"Failed to create dummy image: {e}")
            return None
    return dummy_image


def _guess_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _encode_image_as_data_url(path: str) -> str:
    mime = _guess_mime_type(path)
    b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_output_count(response_body: Dict[str, Any]) -> Optional[int]:
    try:
        content = response_body["choices"][0]["message"]["content"]
        if not isinstance(content, list):
            return None
        image_cnt = 0
        video_cnt = 0
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url":
                image_cnt += 1
            elif item.get("type") == "video_url":
                video_cnt += 1
        if video_cnt > 0:
            return int(video_cnt)
        if image_cnt > 0:
            return int(image_cnt)
        return None
    except Exception:
        return None


def _summary_stats(values: List[float]) -> Dict[str, float]:
    values_sorted = sorted(values)
    if not values_sorted:
        return {"mean": 0.0, "median": 0.0, "p99": 0.0, "p50": 0.0}
    n = len(values_sorted)
    mean = sum(values_sorted) / n
    median = values_sorted[n // 2] if n % 2 else (values_sorted[n // 2 - 1] + values_sorted[n // 2]) / 2
    p50 = values_sorted[int(0.50 * (n - 1))]
    p99 = values_sorted[int(0.99 * (n - 1))]
    return {"mean": float(mean), "median": float(median), "p99": float(p99), "p50": float(p50)}


def _join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    rel = path.lstrip("/")
    return f"{base}/{rel}"


def _root_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return root[:-3]
    return root


def wait_for_service(base_url: str, timeout: int = 120) -> None:
    print(f"Waiting for service at {base_url}...")
    start_time = time.time()
    health_url = _root_url(base_url).rstrip("/") + "/health"
    while True:
        try:
            with urlopen(health_url, timeout=1) as resp:
                if resp.status == 200:
                    print("Service is ready.")
                    return
        except Exception:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Service at {base_url} did not start within {timeout} seconds.")
        time.sleep(1)


async def async_request_vllm_omni(
    input: RequestFuncInput,
    session: aiohttp.ClientSession,
    api_key: str,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    output.start_time = time.perf_counter()

    prompt_text = input.prompt or "Generate an image."
    if input.image_paths and len(input.image_paths) > 0:
        try:
            image_data_url = _encode_image_as_data_url(input.image_paths[0])
        except Exception as e:
            output.error = f"Failed to read image: {e}"
            output.success = False
            output.latency = time.perf_counter() - output.start_time
            if pbar:
                pbar.update(1)
            return output
        content = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
    else:
        content = prompt_text

    payload: Dict[str, Any] = {
        "messages": [{"role": "user", "content": content}],
    }
    if input.model:
        payload["model"] = input.model
    if input.extra_body:
        payload["extra_body"] = input.extra_body

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with session.post(input.api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                resp_json = await response.json()
                output.response_body = resp_json
                output.output_count = _extract_output_count(resp_json)
                output.success = True
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


async def benchmark(args):
    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"

    wait_for_service(args.base_url)

    api_url = _join_url(args.base_url, args.endpoint_path)
    if args.dataset == "vbench":
        dataset = VBenchDataset(args, api_url, args.model)
    elif args.dataset == "random":
        dataset = RandomDataset(args, api_url, args.model)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    requests_list = dataset.get_requests()
    print(f"Prepared {len(requests_list)} requests from {args.dataset} dataset.")

    if args.max_concurrency is not None:
        semaphore = asyncio.Semaphore(args.max_concurrency)
    else:
        semaphore = None

    async def limited_request_func(req, session, pbar):
        if semaphore:
            async with semaphore:
                return await async_request_vllm_omni(req, session, args.api_key, pbar)
        return await async_request_vllm_omni(req, session, args.api_key, pbar)

    pbar = tqdm(total=len(requests_list), disable=args.disable_tqdm)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        start_time = time.perf_counter()
        tasks = []
        for req in requests_list:
            if args.request_rate != float("inf"):
                interval = random.expovariate(args.request_rate)
                await asyncio.sleep(interval)
            tasks.append(asyncio.create_task(limited_request_func(req, session, pbar)))
        outputs = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time

    pbar.close()

    success_outputs = [o for o in outputs if o.success]
    error_outputs = [o for o in outputs if not o.success]
    latencies = [o.latency for o in success_outputs]

    metrics = {
        "duration": total_duration,
        "completed_requests": len(success_outputs),
        "failed_requests": len(error_outputs),
        "throughput_qps": len(success_outputs) / total_duration if total_duration > 0 else 0,
        "latency_mean": _summary_stats(latencies)["mean"] if latencies else 0,
        "latency_median": _summary_stats(latencies)["median"] if latencies else 0,
        "latency_p99": _summary_stats(latencies)["p99"] if latencies else 0,
        "latency_p50": _summary_stats(latencies)["p50"] if latencies else 0,
    }

    outputs_ok = 0
    for out in success_outputs:
        if out.output_count is not None:
            outputs_ok += out.output_count
        else:
            outputs_ok += args.num_outputs_per_prompt
    outputs_throughput = outputs_ok / total_duration if total_duration > 0 else 0.0
    frames_throughput = (
        (outputs_ok * args.num_frames) / total_duration
        if total_duration > 0 and args.num_frames
        else 0.0
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=60, c="="))
    print("{:<40} {:<15}".format("Task:", args.task))
    print("{:<40} {:<15}".format("Model:", args.model or "(server default)"))
    print("{:<40} {:<15}".format("Dataset:", args.dataset))
    print("{:<40} {:<15}".format("Endpoint:", api_url))
    print(f"{'-' * 50}")
    print("{:<40} {:<15.2f}".format("Benchmark duration (s):", metrics["duration"]))
    print("{:<40} {:<15}".format("Request rate:", str(args.request_rate)))
    print(
        "{:<40} {:<15}".format(
            "Max request concurrency:",
            str(args.max_concurrency) if args.max_concurrency else "not set",
        )
    )
    print(
        "{:<40} {}/{:<15}".format(
            "Successful requests:", metrics["completed_requests"], len(requests_list)
        )
    )
    print(f"{'-' * 50}")
    print("{:<40} {:<15.2f}".format("Request throughput (req/s):", metrics["throughput_qps"]))
    print("{:<40} {:<15.2f}".format("Output throughput (out/s):", outputs_throughput))
    if frames_throughput > 0:
        print("{:<40} {:<15.2f}".format("Frame throughput (fps):", frames_throughput))
    print("{:<40} {:<15.4f}".format("Latency Mean (s):", metrics["latency_mean"]))
    print("{:<40} {:<15.4f}".format("Latency Median (s):", metrics["latency_median"]))
    print("{:<40} {:<15.4f}".format("Latency P99 (s):", metrics["latency_p99"]))
    print("\n" + "=" * 60)

    if error_outputs:
        print(f"Failed requests: {len(error_outputs)} (showing up to 3)")
        for err in error_outputs[:3]:
            print(f"- {err.error}")

    if args.output_file:
        output_payload = {
            "metrics": metrics,
            "outputs_throughput": outputs_throughput,
            "frames_throughput": frames_throughput if frames_throughput > 0 else None,
        }
        Path(args.output_file).write_text(
            json.dumps(output_payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Metrics saved to {args.output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark serving for vLLM-Omni diffusion models via /v1/chat/completions."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL of the server (e.g., http://localhost:8000).",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--endpoint-path",
        type=str,
        default="/v1/chat/completions",
        help="Endpoint path under base URL.",
    )
    parser.add_argument("--api-key", type=str, default="EMPTY", help="OpenAI API key.")
    parser.add_argument("--model", type=str, default=None, help="Optional served model name.")

    parser.add_argument(
        "--dataset",
        type=str,
        default="vbench",
        choices=["vbench", "random"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v",
        choices=["t2i", "i2i", "t2v", "i2v", "ti2v", "ti2i"],
        help="Task type.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local dataset file or directory.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts to benchmark."
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent requests (default: 1).",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second; inf sends all at time 0.",
    )
    parser.add_argument("--timeout-s", type=float, default=600.0, help="HTTP timeout per request.")
    parser.add_argument("--width", type=int, default=None, help="Image/Video width.")
    parser.add_argument("--height", type=int, default=None, help="Image/Video height.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames (video).")
    parser.add_argument("--fps", type=int, default=None, help="FPS (video).")

    parser.add_argument("--num-inference-steps", type=int, default=None, help="Diffusion steps.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Guidance scale.")
    parser.add_argument("--guidance-scale-2", type=float, default=None, help="Guidance scale for video.")
    parser.add_argument("--true-cfg-scale", type=float, default=None, help="Qwen-Image specific.")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for diffusion.")
    parser.add_argument(
        "--num-outputs-per-prompt",
        type=int,
        default=1,
        help="Outputs per prompt (num_outputs_per_prompt).",
    )
    parser.add_argument(
        "--extra-body-json",
        type=str,
        default=None,
        help="Extra JSON fields to merge into extra_body.",
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="Output JSON file for metrics."
    )
    parser.add_argument(
        "--disable-tqdm", action="store_true", help="Disable progress bar."
    )
    return parser.parse_args()


def build_extra_body(args: argparse.Namespace) -> Dict[str, Any]:
    extra_body: Dict[str, Any] = {}
    if args.width:
        extra_body["width"] = args.width
    if args.height:
        extra_body["height"] = args.height
    if args.num_frames:
        extra_body["num_frames"] = args.num_frames
    if args.fps:
        extra_body["fps"] = args.fps
    if args.num_inference_steps is not None:
        extra_body["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        extra_body["guidance_scale"] = args.guidance_scale
    if args.guidance_scale_2 is not None:
        extra_body["guidance_scale_2"] = args.guidance_scale_2
    if args.true_cfg_scale is not None:
        extra_body["true_cfg_scale"] = args.true_cfg_scale
    if args.negative_prompt is not None:
        extra_body["negative_prompt"] = args.negative_prompt
    if args.seed is not None:
        extra_body["seed"] = args.seed
    if args.num_outputs_per_prompt:
        extra_body["num_outputs_per_prompt"] = args.num_outputs_per_prompt
    if args.extra_body_json:
        extra_body.update(json.loads(args.extra_body_json))
    return extra_body


if __name__ == "__main__":
    args = parse_args()
    args.extra_body = build_extra_body(args)
    asyncio.run(benchmark(args))
