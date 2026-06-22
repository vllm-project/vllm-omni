# adapted from fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark online serving for diffusion models (Image/Video Generation).
If you want to use i2v, i2i dataset, you should `uv pip install gdown` first

Supports multiple endpoints:
    - /v1/chat/completions: OpenAI chat-compatible image requests (e.g. t2i, Qwen i2i)
    - /v1/images/edits: OpenAI image edit / IT2I (multipart; e.g. Hunyuan --bot-task think)
    - /v1/images/generations: OpenAI image generation requests
    - /v1/videos: Async video jobs

Legacy --backend vllm-omni and openai are aliases for chat/completions and images/generations.

Usage:
    # Video (/v1/videos endpoint)
    t2v:
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --endpoint /v1/videos --dataset vbench --task t2v --num-prompts 10 \
        --height 480 --width 640 --fps 16 --num-frames 80

    i2v:
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --endpoint /v1/videos --dataset vbench --task i2v --num-prompts 10


    # Image (/v1/chat/completions endpoint)
    t2i:
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --endpoint /v1/chat/completions --dataset vbench --task t2i --num-prompts 10 \
        --height 1024 --width 1024

    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --endpoint /v1/chat/completions --dataset random --task t2i --num-prompts 1 \
        --max-concurrency 1 --enable-negative-prompt \
        --random-request-config '[
            {"width":512,"height":512,"num_inference_steps":20,"weight":0.15},
            {"width":768,"height":768,"num_inference_steps":20,"weight":0.25},
            {"width":1024,"height":1024,"num_inference_steps":25,"weight":0.45},
            {"width":1536,"height":1536,"num_inference_steps":35,"weight":0.15}
        ]'

    ti2i (Hunyuan / OpenAI image edit API):
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --endpoint /v1/images/edits --dataset random --task ti2i --num-prompts 10 \
        --bot-task think

    python benchmarks/diffusion/diffusion_benchmark_serving.py \
      --endpoint /v1/images/edits \
      --dataset custom \
      --dataset-path custom_requests.jsonl \
      --task ti2i \
      --bot-task think

    i2i (chat-based models such as Qwen-Image-Edit):
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --endpoint /v1/chat/completions --dataset vbench --task i2i --num-prompts 10

    # Image (/v1/images/generations endpoint)
    t2i:
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --endpoint /v1/images/generations --dataset vbench --task t2i --num-prompts 10 \
        --height 1024 --width 1024 --port 3000

    # Video (v1/videos)
    t2v:
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --endpoint /v1/videos --dataset random --task t2v --num-prompts 1 \
        --max-concurrency 1 --enable-negative-prompt \
        --random-request-config '[
            {"width":854,"height":480,"num_inference_steps":18,"num_frames":120,"fps":24,"weight":1}
        ]'


"""

import argparse
import ast
import asyncio
import base64
import glob
import json
import logging
import os
import random
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import replace
from typing import Any

import aiohttp
import numpy as np
import requests
from backends import (
    RequestFuncInput,
    RequestFuncOutput,
    backends_function_mapping,
    normalize_endpoint,
)
from PIL import Image
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)

_STAGE_METRICS_ENDPOINTS = {"/v1/chat/completions"}
_RETURN_STAGE_METRICS_FIELD = "return_stage_metrics"


class BaseDataset(ABC):
    def __init__(self, args, api_url: str, model: str):
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
    def get_requests(self) -> list[RequestFuncInput]:
        pass


class VBenchDataset(BaseDataset):
    """
    Dataset loader for VBench prompts.
    Supports t2v, i2v.
    """

    T2V_PROMPT_URL = (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/prompts/prompts_per_dimension/subject_consistency.txt"
    )
    I2V_DOWNLOAD_SCRIPT_URL = (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/vbench2_beta_i2v/download_data.sh"
    )

    def __init__(self, args, api_url: str, model: str):
        super().__init__(args, api_url, model)
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vllm-omni")
        self.items = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        if self.args.task == "t2v":
            return self._load_t2v_prompts()
        elif self.args.task in ["i2v", "ti2v", "ti2i", "i2i", "it2i"]:
            return self._load_i2v_data()
        else:
            return self._load_t2v_prompts()

    def _download_file(self, url: str, dest_path: str) -> None:
        """Download a file from URL to destination path."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        resp = requests.get(url)
        resp.raise_for_status()
        with open(dest_path, "w") as f:
            f.write(resp.text)

    def _load_t2v_prompts(self) -> list[dict[str, Any]]:
        path = self.args.dataset_path

        if not path:
            path = os.path.join(self.cache_dir, "vbench_subject_consistency.txt")
            if not os.path.exists(path):
                print(f"Downloading VBench T2V prompts to {path}...")
                try:
                    self._download_file(self.T2V_PROMPT_URL, path)
                except Exception as e:
                    print(f"Failed to download VBench prompts: {e}")
                    return [{"prompt": "A cat sitting on a bench"}] * 50

        prompts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append({"prompt": line})

        return self._resize_data(prompts)

    def _auto_download_i2v_dataset(self) -> str:
        """Auto-download VBench I2V dataset and return the dataset directory."""
        vbench_i2v_dir = os.path.join(self.cache_dir, "vbench_i2v", "vbench2_beta_i2v")
        info_json_path = os.path.join(vbench_i2v_dir, "data", "i2v-bench-info.json")

        if os.path.exists(info_json_path):
            return vbench_i2v_dir

        print(f"Downloading VBench I2V dataset to {vbench_i2v_dir}...")
        try:
            cache_root = os.path.join(self.cache_dir, "vbench_i2v")
            script_path = os.path.join(cache_root, "download_data.sh")

            self._download_file(self.I2V_DOWNLOAD_SCRIPT_URL, script_path)
            os.chmod(script_path, 0o755)

            print("Executing download_data.sh (this may take a while)...")
            import subprocess

            result = subprocess.run(
                ["bash", script_path],
                cwd=cache_root,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Download script failed: {result.stderr}")

            print(f"Successfully downloaded VBench I2V dataset to {vbench_i2v_dir}")
        except Exception as e:
            print(f"Failed to download VBench I2V dataset: {e}")
            print("Please manually download following instructions at:")
            print("https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v#22-download")
            return None

        return vbench_i2v_dir if os.path.exists(info_json_path) else None

    def _load_from_i2v_json(self, json_path: str) -> list[dict[str, Any]]:
        """Load I2V data from i2v-bench-info.json format."""
        with open(json_path) as f:
            items = json.load(f)

        base_dir = os.path.dirname(os.path.dirname(json_path))  # Go up to vbench2_beta_i2v
        origin_dir = os.path.join(base_dir, "data", "origin")

        data = []
        for item in items:
            img_path = os.path.join(origin_dir, item.get("file_name", ""))
            if os.path.exists(img_path):
                data.append({"prompt": item.get("caption", ""), "image_path": img_path})
            else:
                print(f"Warning: Image not found: {img_path}")

        print(f"Loaded {len(data)} I2V samples from VBench I2V dataset")
        return data

    def _scan_directory_for_images(self, path: str) -> list[dict[str, Any]]:
        """Scan directory for image files."""
        exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        files = []

        for ext in exts:
            files.extend(glob.glob(os.path.join(path, ext)))
            files.extend(glob.glob(os.path.join(path, ext.upper())))

            # Also check in data/origin subdirectory
            origin_dir = os.path.join(path, "data", "origin")
            if os.path.exists(origin_dir):
                files.extend(glob.glob(os.path.join(origin_dir, ext)))
                files.extend(glob.glob(os.path.join(origin_dir, ext.upper())))

        return [{"prompt": os.path.splitext(os.path.basename(f))[0], "image_path": f} for f in files]

    def _create_dummy_data(self) -> list[dict[str, Any]]:
        """Create dummy data with a placeholder image in cache directory."""
        print("No I2V data found. Using dummy placeholders.")

        dummy_image = os.path.join(self.cache_dir, "dummy_image.jpg")
        if not os.path.exists(dummy_image):
            try:
                from PIL import Image

                os.makedirs(self.cache_dir, exist_ok=True)
                img = Image.new("RGB", (100, 100), color="red")
                img.save(dummy_image)
                print(f"Created dummy image at {dummy_image}")
            except ImportError:
                print("PIL not installed, cannot create dummy image.")
                return []

        return [{"prompt": "A moving cat", "image_path": dummy_image}] * 10

    def _load_i2v_data(self) -> list[dict[str, Any]]:
        """Load I2V data from VBench I2V dataset or user-provided path."""
        path = self.args.dataset_path

        # Auto-download if no path provided
        if not path:
            path = self._auto_download_i2v_dataset()
            if not path:
                return self._resize_data(self._create_dummy_data())

        # Try to load from i2v-bench-info.json
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

        # Fallback: scan directory for images
        if os.path.isdir(path):
            data = self._scan_directory_for_images(path)
            if data:
                return self._resize_data(data)

        # Last resort: dummy data
        return self._resize_data(self._create_dummy_data())

    def _resize_data(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resize data to match num_prompts."""
        if not data:
            raise ValueError("No benchmark data available. Install Pillow or provide --dataset-path.")

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
            width=self.args.width,
            height=self.args.height,
            num_frames=self.args.num_frames,
            num_inference_steps=self.args.num_inference_steps,
            seed=self.args.seed,
            fps=self.args.fps,
            image_paths=image_paths,
        )

    def get_requests(self) -> list[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


class TraceDataset(BaseDataset):
    """Trace-based dataset loader for heterogeneous diffusion requests."""

    DEFAULT_REPO_ID = "asukaqaqzz/Dit_Trace"
    DEFAULT_FILENAME = "sd3_trace.txt"
    DEFAULT_FILENAME_BY_TASK: dict[str, str] = {
        # Text-to-image traces (e.g., SD3)
        "t2i": "sd3_trace.txt",
        # Text-to-video traces (e.g., CogVideoX)
        "t2v": "cogvideox_trace.txt",
    }

    def __init__(self, args, api_url: str, model: str):
        super().__init__(args, api_url, model)
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vllm-omni", "trace")
        self.default_filename = self.DEFAULT_FILENAME_BY_TASK.get(getattr(args, "task", ""), self.DEFAULT_FILENAME)
        dataset_root = args.dataset_path
        if not dataset_root:
            dataset_root = self._download_default_trace()
        self.items = self._load_items(dataset_root)

    @staticmethod
    def _coerce_int(x: Any) -> int | None:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        try:
            s = str(x).strip()
            if not s:
                return None
            return int(float(s))
        except Exception:
            return None

    @staticmethod
    def _coerce_float(x: Any) -> float | None:
        if x is None:
            return None
        if isinstance(x, float):
            return x
        if isinstance(x, int):
            return float(x)
        try:
            s = str(x).strip()
            if not s:
                return None
            return float(s)
        except Exception:
            return None

    def _download_default_trace(self) -> str:
        """Download default trace file from HuggingFace Hub if not provided."""

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download the default trace dataset. "
                "Install via `pip install huggingface_hub`."
            ) from exc

        os.makedirs(self.cache_dir, exist_ok=True)
        return hf_hub_download(
            repo_id=self.DEFAULT_REPO_ID,
            filename=self.default_filename,
            repo_type="dataset",
            local_dir=self.cache_dir,
            local_dir_use_symlinks=False,
        )

    def _expand_paths(self, dataset_path: str | None) -> list[str]:
        if not dataset_path:
            return []

        parts = [p.strip() for p in str(dataset_path).split(",") if p.strip()]
        paths: list[str] = []
        for p in parts:
            if any(ch in p for ch in ["*", "?", "["]):
                paths.extend(sorted(glob.glob(p)))
            elif os.path.isdir(p):
                paths.extend(sorted(glob.glob(os.path.join(p, "**", "*.txt"), recursive=True)))
            else:
                paths.append(p)

        seen = set()
        unique_paths = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
        return unique_paths

    def _parse_trace_file(self, path: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        def parse_request_repr_line(line: str) -> dict[str, Any] | None:
            text = line.strip()
            if not text:
                return None
            if not (text.startswith("Request(") and text.endswith(")")):
                return None
            inner = text[len("Request(") : -1]
            try:
                expr = ast.parse(f"f({inner})", mode="eval")
                if not isinstance(expr.body, ast.Call):
                    return None
                call = expr.body
                out: dict[str, Any] = {}
                for kw in call.keywords:
                    if kw.arg is None:
                        continue
                    out[kw.arg] = ast.literal_eval(kw.value)
                return out
            except Exception:
                return None

        # detect first non-empty line to pick parser
        first_non_empty = None
        with open(path, encoding="utf-8") as f:
            for _ in range(50):
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    first_non_empty = line.strip()
                    f.seek(pos)
                    break

        if first_non_empty is None:
            return rows

        if first_non_empty.startswith("Request("):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parsed = parse_request_repr_line(line)
                    if isinstance(parsed, dict):
                        rows.append(parsed)
            return rows

        # txt fallback: parse Request(...) lines only
        with open(path, encoding="utf-8") as f:
            for line in f:
                parsed = parse_request_repr_line(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
        return rows

    def _load_items(self, dataset_root: str) -> list[dict[str, Any]]:
        paths = self._expand_paths(dataset_root)
        if not paths:
            raise ValueError("No trace files found. Provide --dataset-path or rely on default HuggingFace download.")

        items: list[dict[str, Any]] = []
        for p in paths:
            if not os.path.exists(p):
                continue
            for row in self._parse_trace_file(p):
                if isinstance(row, dict):
                    row = dict(row)
                    row.setdefault("_source", p)
                    items.append(row)

        if not items:
            raise ValueError("Trace dataset is empty after parsing provided paths.")

        if self.args.num_prompts is not None:
            items = items[: self.args.num_prompts]

        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestFuncInput:
        row = self.items[idx]
        prompt = row.get("prompt") or row.get("text") or ""

        row_height = self._coerce_int(row.get("height"))
        row_width = self._coerce_int(row.get("width"))
        num_frames = self._coerce_int(row.get("num_frames"))
        num_steps = self._coerce_int(row.get("num_inference_steps"))
        seed = self._coerce_int(row.get("seed"))
        fps = self._coerce_int(row.get("fps"))
        timestamp = self._coerce_float(row.get("timestamp"))
        slo_ms = self._coerce_float(row.get("slo_ms"))
        image_paths = row.get("image_paths")
        if not image_paths:
            single = row.get("image_path")
            image_paths = [single] if single else None

        if not image_paths and self.args.task in ["i2v", "i2i", "ti2v", "ti2i", "it2i"]:
            raise ValueError(
                f"Task {self.args.task} requires image input, but no image_path or image_paths found in trace row."
            )

        override_w = self.args.width
        override_h = self.args.height
        if override_w is not None or override_h is not None:
            width = override_w
            height = override_h
        else:
            width = row_width
            height = row_height

        return RequestFuncInput(
            prompt=str(prompt),
            api_url=self.api_url,
            model=self.model,
            width=width,
            height=height,
            num_frames=num_frames if num_frames is not None else self.args.num_frames,
            num_inference_steps=num_steps if num_steps is not None else self.args.num_inference_steps,
            seed=seed if seed is not None else self.args.seed,
            fps=fps if fps is not None else self.args.fps,
            timestamp=timestamp,
            slo_ms=slo_ms,
            image_paths=image_paths,
            request_id=str(row.get("request_id")) if row.get("request_id") is not None else str(uuid.uuid4()),
        )

    def get_requests(self) -> list[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


class CustomDataset(BaseDataset):
    """
    Custom dataset that loads requests from a JSONL file.

    Each line in the JSONL file should be a JSON object with the following fields:
    - prompt (required): The text prompt for the request
    - width (optional): Image/video width
    - height (optional): Image height
    - num_inference_steps (optional): Number of diffusion steps
    - seed (optional): Random seed
    - image_paths (optional): List of input image paths for i2i/i2v/ti2i tasks
    - image_urls (optional): List of input image URLs (alternative to image_paths)

    Example JSONL for ti2i:
    {"prompt": "Add sunset lighting", "width": 1024, "height": 1024, "image_urls": ["https://example.com/image.jpg"]}
    """

    def __init__(self, args, api_url: str, model: str):
        super().__init__(args, api_url, model)
        self.dataset_path = args.dataset_path
        if not self.dataset_path:
            raise ValueError("--dataset-path must be provided when using 'custom' dataset")
        self.load_data()

    def load_data(self) -> None:
        """Load data from JSONL file."""
        import pandas as pd

        if not self.dataset_path.endswith(".jsonl"):
            raise ValueError("Custom dataset must be a JSONL file")

        try:
            df = pd.read_json(path_or_buf=self.dataset_path, lines=True)
        except Exception as e:
            raise ValueError(f"Failed to load JSONL file {self.dataset_path}: {e}")

        if "prompt" not in df.columns:
            raise ValueError("JSONL file must contain a 'prompt' column")

        self.data = df.to_dict("records")
        print(f"Loaded {len(self.data)} requests from {self.dataset_path}")

    def _resolve_image_paths(self, item: dict) -> list[str] | None:
        # Handle local file paths
        if "image_paths" in item and item["image_paths"]:
            paths = item["image_paths"]
            if isinstance(paths, str):
                paths = [paths]

            # Verify all paths exist
            valid_paths = []
            for path in paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    raise ValueError(f"Image file not found: {path}")

            return valid_paths if valid_paths else None

        # Handle URLs - download to temp files
        if "image_urls" in item and item["image_urls"]:
            urls = item["image_urls"]
            if isinstance(urls, str):
                urls = [urls]

            downloaded_paths = []
            for url in urls:
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()

                    # Create temp file with appropriate extension
                    suffix = os.path.splitext(url)[1] or ".png"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(response.content)
                        downloaded_paths.append(tmp.name)
                except Exception as e:
                    raise ValueError(f"Failed to download image from {url}: {e}")

            return downloaded_paths if downloaded_paths else None

        return None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> RequestFuncInput:
        item = self.data[idx]

        # Extract fields with fallback to args defaults
        prompt = item["prompt"]
        width = item.get("width", self.args.width)
        height = item.get("height", self.args.height)
        num_inference_steps = item.get("num_inference_steps", self.args.num_inference_steps)
        seed = item.get("seed", self.args.seed)
        reserved_keys = {"prompt", "width", "height", "num_inference_steps", "seed", "image_paths", "image_urls"}
        extra_body = {k: v for k, v in item.items() if k not in reserved_keys}

        # Handle image paths/URLs
        image_paths = self._resolve_image_paths(item)

        return RequestFuncInput(
            prompt=prompt,
            api_url=self.api_url,
            model=self.model,
            seed=seed,
            image_paths=image_paths,
            extra_body=extra_body,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
        )

    def get_requests(self) -> list[RequestFuncInput]:
        """Get all requests, cycling through data if num_prompts > dataset size."""
        num_requests = getattr(self.args, "num_prompts", len(self.data))

        if num_requests <= len(self.data):
            return [self[i] for i in range(num_requests)]

        # Cycle through the dataset to reach num_prompts
        requests = []
        for i in range(num_requests):
            idx = i % len(self.data)
            requests.append(self[idx])

        return requests


class RandomDataset(BaseDataset):
    def __init__(self, args, api_url: str, model: str, enable_negative_prompt: bool = False):
        super().__init__(args, api_url, model)
        self.num_prompts = args.num_prompts
        self.enable_negative_prompt = enable_negative_prompt
        self.num_input_images = max(1, args.num_input_images)
        self.random_request_config = getattr(args, "random_request_config", None)
        if self.random_request_config:
            self.random_request_config = json.loads(self.random_request_config)
            self._weights = [p["weight"] for p in self.random_request_config]

            self.random_request_config = [
                {k: v for k, v in p.items() if k != "weight"} for p in self.random_request_config
            ]

            seed = getattr(args, "random_request_seed", 42)
            self._rng = random.Random(seed)

            self._sampled_requests = self._rng.choices(
                self.random_request_config,
                weights=self._weights,
                k=self.num_prompts,
            )
        else:
            self._sampled_requests = None

        # Random image generate
        if self.args.task in ["i2v", "ti2v", "ti2i", "i2i", "it2i"]:
            self._random_image_path = self._generate_random_image_paths()
        else:
            self._random_image_path = None

    def __len__(self) -> int:
        return self.num_prompts

    def __getitem__(self, idx: int) -> RequestFuncInput:
        extra_body = {}
        if self.enable_negative_prompt:
            extra_body["negative_prompt"] = f"Negative prompt {idx} for benchmarking diffusion models"

        params = {
            "width": self.args.width,
            "height": self.args.height,
            "num_frames": self.args.num_frames,
            "num_inference_steps": self.args.num_inference_steps,
            "fps": self.args.fps,
        }
        if self._sampled_requests:
            profile = self._sampled_requests[idx]
            params.update(profile)
        return RequestFuncInput(
            prompt=f"Random prompt {idx} for benchmarking diffusion models",
            api_url=self.api_url,
            model=self.model,
            seed=self.args.seed,
            extra_body=extra_body,
            image_paths=self._random_image_path,
            **params,
        )

    def get_requests(self) -> list[RequestFuncInput]:
        return [self[i] for i in range(len(self))]

    def _generate_random_image_paths(self) -> list[str]:
        image_paths: list[str] = []
        for image_idx in range(self.num_input_images):
            img = Image.new("RGB", (512, 512), (255, 255, 255))
            image_path = os.path.join(
                tempfile.gettempdir(),
                f"diffusion_benchmark_random_image_{image_idx}.png",
            )
            img.save(image_path)
            image_paths.append(image_path)
        return image_paths


def _compute_expected_latency_ms_from_base(req: RequestFuncInput, args, base_time_ms: float | None) -> float | None:
    """Compute expected execution time (ms) based on a base per-step-per-frame unit time.

    Assumes linear scaling with pixel area, frame count, and num_inference_steps.
    The base unit represents latency for a 16x16 resolution, single frame, single step.
    """

    if base_time_ms is None:
        return None

    width = req.width if req.width is not None else args.width
    height = req.height if req.height is not None else args.height
    if width is None or height is None:
        return None

    frames = req.num_frames if req.num_frames is not None else args.num_frames
    steps = req.num_inference_steps if req.num_inference_steps is not None else args.num_inference_steps

    frame_scale = frames if isinstance(frames, int) and frames > 0 else 1
    step_scale = steps if isinstance(steps, int) and steps > 0 else 1

    area_units = max((float(width) * float(height)) / float(16 * 16), 1.0)
    return float(base_time_ms) * area_units * frame_scale * step_scale


def _infer_slo_base_time_ms_from_warmups(
    warmup_pairs: list[tuple[RequestFuncInput, RequestFuncOutput]],
    args,
) -> float | None:
    """Infer base SLO unit time from warmup requests.

    Returns the median base latency (ms) for a 16x16 resolution, single-frame,
    single-step request. Only uses warmups that succeeded and have resolvable
    width/height.
    """

    candidates_ms: list[float] = []
    for req, out in warmup_pairs:
        if not out.success or out.latency <= 0:
            continue

        width = req.width if req.width is not None else args.width
        height = req.height if req.height is not None else args.height
        if width is None or height is None:
            continue

        frames = req.num_frames if req.num_frames is not None else args.num_frames
        steps = req.num_inference_steps if req.num_inference_steps is not None else args.num_inference_steps

        frame_scale = int(frames) if isinstance(frames, int) and frames > 0 else 1
        step_scale = int(steps) if isinstance(steps, int) and steps > 0 else 1

        area_units = max((float(width) * float(height)) / float(16 * 16), 1.0)
        denom = area_units * float(frame_scale) * float(step_scale)
        if denom <= 0:
            continue

        candidates_ms.append((out.latency * 1000.0) / denom)

    if not candidates_ms:
        return None
    return float(np.median(candidates_ms))


def _populate_slo_ms_from_warmups(
    requests_list: list[RequestFuncInput],
    warmup_pairs: list[tuple[RequestFuncInput, RequestFuncOutput]],
    args,
) -> list[RequestFuncInput]:
    """Populate missing RequestFuncInput.slo_ms using warmup outputs.

    - If a request already has slo_ms (e.g., trace-provided), it is kept as-is.
    - If any request has slo_ms is None and we can infer base time from warmups,
      we estimate each missing request's expected execution time and set:
        req.slo_ms = expected_latency_ms * args.slo_scale

    Returns updated requests_list.
    """

    if not any(req.slo_ms is None for req in requests_list):
        return requests_list

    base_time_ms = _infer_slo_base_time_ms_from_warmups(warmup_pairs, args)
    if base_time_ms is None:
        return requests_list

    slo_scale = float(getattr(args, "slo_scale", 3.0))
    if slo_scale <= 0:
        raise ValueError(f"slo_scale must be positive, got {slo_scale}.")

    updated: list[RequestFuncInput] = []
    for req in requests_list:
        if req.slo_ms is not None:
            updated.append(req)
            continue
        expected_ms = _compute_expected_latency_ms_from_base(req, args, base_time_ms)
        updated.append(replace(req, slo_ms=(expected_ms * slo_scale) if expected_ms is not None else None))

    return updated


async def iter_requests(
    requests_list: list[RequestFuncInput],
    request_rate: float,
) -> AsyncGenerator[RequestFuncInput, None]:
    """Yield requests using a Poisson process if request_rate is set.

    - If request_rate is inf, all requests are yielded immediately (no sleep).
    - Otherwise, inter-arrival times follow an exponential distribution.
    """

    if request_rate != float("inf"):
        if request_rate <= 0:
            raise ValueError(f"request_rate must be positive or inf, got {request_rate}.")

    for i, req in enumerate(requests_list):
        if request_rate != float("inf") and i > 0:
            interval_s = random.expovariate(request_rate)
            await asyncio.sleep(interval_s)
        yield req


def _make_warmup_request(
    requests_list: list[RequestFuncInput],
    index: int,
    args,
) -> RequestFuncInput:
    warm_req = requests_list[index % len(requests_list)]
    if args.warmup_num_inference_steps is not None:
        warm_req = replace(
            warm_req,
            num_inference_steps=args.warmup_num_inference_steps,
        )
    if args.task == "t2v":
        warm_req = replace(warm_req, num_frames=1)
    return warm_req


async def _run_warmups(
    requests_list: list[RequestFuncInput],
    args,
    session: aiohttp.ClientSession,
    request_func,
) -> list[tuple[RequestFuncInput, RequestFuncOutput]]:
    if not args.warmup_requests or not requests_list:
        return []

    warmup_requests = [_make_warmup_request(requests_list, i, args) for i in range(args.warmup_requests)]
    warmup_concurrency = min(int(args.warmup_concurrency), len(warmup_requests))
    warmup_semaphore = asyncio.Semaphore(warmup_concurrency)

    print(
        f"Running {len(warmup_requests)} warmup request(s) "
        f"with num_inference_steps={args.warmup_num_inference_steps} "
        f"and warmup_concurrency={warmup_concurrency}..."
    )

    async def limited_warmup_request_func(
        req: RequestFuncInput,
    ) -> RequestFuncOutput:
        async with warmup_semaphore:
            return await request_func(req, session, None)

    warmup_tasks = [asyncio.create_task(limited_warmup_request_func(req)) for req in warmup_requests]
    warmup_outputs = await asyncio.gather(*warmup_tasks)
    return list(zip(warmup_requests, warmup_outputs))


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    total_duration: float,
    requests_list: list[RequestFuncInput],
    args,
    slo_enabled: bool,
):
    success_outputs = [o for o in outputs if o.success]
    error_outputs = [o for o in outputs if not o.success]

    num_success = len(success_outputs)
    latencies = [o.latency for o in success_outputs]
    peak_memories = [o.peak_memory_mb for o in success_outputs if o.peak_memory_mb > 0]

    # Aggregate per-stage durations across all successful requests that reported them.
    stage_duration_lists: dict[str, list[float]] = {}
    for o in success_outputs:
        for stage, duration in (o.stage_durations or {}).items():
            stage_duration_lists.setdefault(stage, []).append(duration)
    stage_durations_mean = {s: float(np.mean(v)) for s, v in stage_duration_lists.items()}
    stage_durations_p50 = {s: float(np.percentile(v, 50)) for s, v in stage_duration_lists.items()}
    stage_durations_p99 = {s: float(np.percentile(v, 99)) for s, v in stage_duration_lists.items()}

    metrics = {
        "duration": total_duration,
        "completed_requests": num_success,
        "failed_requests": len(error_outputs),
        "throughput_qps": num_success / total_duration if total_duration > 0 else 0,
        "latency_mean": np.mean(latencies) if latencies else 0,
        "latency_median": np.median(latencies) if latencies else 0,
        "latency_p99": np.percentile(latencies, 99) if latencies else 0,
        "latency_p95": np.percentile(latencies, 95) if latencies else 0,
        "latency_p50": np.percentile(latencies, 50) if latencies else 0,
        "peak_memory_mb_max": max(peak_memories) if peak_memories else 0,
        "peak_memory_mb_mean": np.mean(peak_memories) if peak_memories else 0,
        "peak_memory_mb_median": np.median(peak_memories) if peak_memories else 0,
        "stage_durations_mean": stage_durations_mean,
        "stage_durations_p50": stage_durations_p50,
        "stage_durations_p99": stage_durations_p99,
    }

    if slo_enabled:
        slo_defined_total = 0
        slo_met_success = 0

        for req, out in zip(requests_list, outputs):
            if req.slo_ms is None:
                continue
            slo_defined_total += 1
            if out.slo_achieved is None:
                continue
            if out.slo_achieved:
                slo_met_success += 1

        slo_attain_all = (slo_met_success / slo_defined_total) if slo_defined_total > 0 else 0.0

        metrics.update(
            {
                "slo_attainment_rate": slo_attain_all,
                "slo_met_success": slo_met_success,
                "slo_scale": getattr(args, "slo_scale", 3.0),
            }
        )

    return metrics


def _save_generated_outputs(
    outputs: list[RequestFuncOutput],
    requests_list: list[RequestFuncInput],
    save_dir: str,
) -> None:
    """Decode and save base64 images/videos from successful responses."""
    os.makedirs(save_dir, exist_ok=True)
    saved = 0
    failed = 0

    for idx, (req, out) in enumerate(zip(requests_list, outputs)):
        if not out.success or not out.response_body:
            continue

        if isinstance(out.response_body, bytes):
            fname = f"req_{idx:04d}.mp4"
            fpath = os.path.join(save_dir, fname)
            with open(fpath, "wb") as f:
                f.write(out.response_body)
            saved += 1
            continue

        media_urls: list[str] = []

        # Chat-completions style: choices[*].message.content[*].image_url.url
        choices = out.response_body.get("choices", [])
        if isinstance(choices, list):
            for choice in choices:
                content = (choice or {}).get("message", {}).get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "image_url":
                        continue
                    url = (item.get("image_url") or {}).get("url", "")
                    if isinstance(url, str) and url.startswith("data:"):
                        media_urls.append(url)

        # Images endpoint style: data[*].b64_json
        data_items = out.response_body.get("data", [])
        if isinstance(data_items, list):
            for data_item in data_items:
                if not isinstance(data_item, dict):
                    continue
                b64_json = data_item.get("b64_json", "")
                if isinstance(b64_json, str) and b64_json:
                    media_urls.append(f"data:image/png;base64,{b64_json}")

        for img_idx, url in enumerate(media_urls):
            if "," not in url:
                continue

            try:
                header, b64_data = url.split(",", 1)
                ext = "png"
                if "image/jpeg" in header:
                    ext = "jpg"
                elif "image/webp" in header:
                    ext = "webp"
                elif "video/mp4" in header:
                    ext = "mp4"

                img_bytes = base64.b64decode(b64_data)
                fname = f"req_{idx:04d}_{img_idx}.{ext}"
                fpath = os.path.join(save_dir, fname)
                with open(fpath, "wb") as f:
                    f.write(img_bytes)
                saved += 1
            except Exception as e:
                failed += 1
                logger.warning(f"Failed to save image for request {idx}: {e}", exc_info=True)

    logger.info(f"Saved {saved} generated image(s) to {save_dir}. Failed to save {failed} image(s).")


def wait_for_service(base_url: str, timeout: int = 120) -> None:
    print(f"Waiting for service at {base_url}...")
    start_time = time.time()
    while True:
        try:
            # Try /health endpoint first
            resp = requests.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                print("Service is ready.")
                break
        except requests.exceptions.RequestException:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Service at {base_url} did not start within {timeout} seconds.")

        time.sleep(1)


def _default_endpoint_for_task(task: str) -> str:
    if task in {"t2v", "i2v", "ti2v"}:
        return "/v1/videos"
    if task in {"i2i", "ti2i", "it2i"}:
        return "/v1/images/edits"
    if task == "t2i":
        return "/v1/chat/completions"
    raise ValueError(f"Unsupported task for endpoint resolution: {task}")


async def benchmark(args):
    # Construct base_url if not provided
    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"

    VIDEO_TASKS = {"t2v", "i2v", "ti2v"}
    IMAGE_TASKS = {"t2i", "i2i", "ti2i", "it2i"}

    if args.task in VIDEO_TASKS:
        task_type = "2v"
    elif args.task in IMAGE_TASKS:
        task_type = "2i"
    else:
        raise ValueError(
            f"Unsupported task: '{args.task}'. "
            f"Valid video tasks: {sorted(VIDEO_TASKS)}, "
            f"Valid image tasks: {sorted(IMAGE_TASKS)}"
        )

    raw_endpoint = args.endpoint if args.endpoint is not None else args.backend
    if raw_endpoint is None:
        raw_endpoint = _default_endpoint_for_task(args.task)
    args.endpoint = normalize_endpoint(raw_endpoint)

    valid_endpoints = sorted(backends_function_mapping[task_type].keys())

    if args.endpoint not in valid_endpoints:
        logger.error(
            f"Invalid endpoint '{args.endpoint}' for task '{args.task}' (task type: '{task_type}').\n"
            f"Valid endpoints for this task type: {valid_endpoints}\n"
            f"Example usage: --task {args.task} --endpoint {valid_endpoints[0]}"
        )
        raise ValueError("Endpoint validation failed. See log above for valid options.")

    # Setup API URL and request function based on endpoint.
    request_func, api_url = backends_function_mapping[task_type][args.endpoint]
    api_url = f"{args.base_url}{api_url}"

    if args.dataset == "vbench":
        dataset = VBenchDataset(args, api_url, args.model)
    elif args.dataset == "trace":
        dataset = TraceDataset(args, api_url, args.model)
    elif args.dataset == "random":
        dataset = RandomDataset(args, api_url, args.model, args.enable_negative_prompt)
    elif args.dataset == "custom":
        dataset = CustomDataset(args, api_url, args.model)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print("Loading requests...")
    requests_list = dataset.get_requests()
    print(f"Prepared {len(requests_list)} requests from {args.dataset} dataset.")

    if args.return_stage_metrics and args.endpoint in _STAGE_METRICS_ENDPOINTS:
        for req in requests_list:
            req.extra_body.setdefault(_RETURN_STAGE_METRICS_FIELD, True)

    if args.endpoint == "/v1/images/edits":
        for req in requests_list:
            req.default_bot_task = args.bot_task

    # Limit concurrency
    if args.max_concurrency is not None:
        semaphore = asyncio.Semaphore(args.max_concurrency)
    else:
        semaphore = None

    async def limited_request_func(req, session, pbar):
        if semaphore:
            async with semaphore:
                return await request_func(req, session, pbar)
        else:
            return await request_func(req, session, pbar)

    # Run benchmark
    pbar = tqdm(total=len(requests_list), disable=args.disable_tqdm)

    async with aiohttp.ClientSession() as session:
        warmup_pairs = await _run_warmups(
            requests_list=requests_list,
            args=args,
            session=session,
            request_func=request_func,
        )

        if args.slo:
            # Prefer trace-provided per-request slo_ms. Only populate when missing.
            requests_list = _populate_slo_ms_from_warmups(
                requests_list=requests_list,
                warmup_pairs=warmup_pairs,
                args=args,
            )

        start_time = time.perf_counter()
        tasks = []
        async for req in iter_requests(requests_list=requests_list, request_rate=args.request_rate):
            task = asyncio.create_task(limited_request_func(req, session, pbar))
            tasks.append(task)

        outputs = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time

    pbar.close()

    # Calculate metrics
    metrics = calculate_metrics(outputs, total_duration, requests_list, args, args.slo)

    # Add configuration info to metrics for JSON output
    metrics["endpoint"] = args.endpoint
    metrics["model"] = args.model
    metrics["dataset"] = args.dataset
    metrics["task"] = args.task
    if args.endpoint == "/v1/images/edits":
        metrics["bot_task"] = args.bot_task

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))

    # Section 1: Configuration
    print("{:<40} {:<15}".format("Endpoint:", args.endpoint))
    print("{:<40} {:<15}".format("Model:", args.model))
    print("{:<40} {:<15}".format("Dataset:", args.dataset))
    print("{:<40} {:<15}".format("Task:", args.task))

    # Section 2: Execution & Traffic
    print(f"{'-' * 50}")
    print("{:<40} {:<15.2f}".format("Benchmark duration (s):", metrics["duration"]))
    print("{:<40} {:<15}".format("Request rate:", str(args.request_rate)))
    print(
        "{:<40} {:<15}".format(
            "Max request concurrency:",
            str(args.max_concurrency) if args.max_concurrency else "not set",
        )
    )
    print("{:<40} {}/{:<15}".format("Successful requests:", metrics["completed_requests"], len(requests_list)))

    # Section 3: Performance Metrics
    print(f"{'-' * 50}")

    print("{:<40} {:<15.2f}".format("Request throughput (req/s):", metrics["throughput_qps"]))
    print("{:<40} {:<15.4f}".format("Latency Mean (s):", metrics["latency_mean"]))
    print("{:<40} {:<15.4f}".format("Latency Median (s):", metrics["latency_median"]))
    print("{:<40} {:<15.4f}".format("Latency P99 (s):", metrics["latency_p99"]))
    print("{:<40} {:<15.4f}".format("Latency P95 (s):", metrics["latency_p95"]))

    if args.slo:
        print(f"{'-' * 50}")
        print("{:<40} {:<15.2%}".format("SLO Attainment Rate (all):", metrics.get("slo_attainment_rate", 0.0)))
        print("{:<40} {:<15}".format("SLO Met (success count):", str(metrics.get("slo_met_success", 0))))
        print("{:<40} {:<15}".format("SLO Scale:", str(metrics.get("slo_scale", 3.0))))

    if metrics["peak_memory_mb_max"] > 0:
        print(f"{'-' * 50}")
        print("{:<40} {:<15.2f}".format("Peak Memory Max (MB):", metrics["peak_memory_mb_max"]))
        print("{:<40} {:<15.2f}".format("Peak Memory Mean (MB):", metrics["peak_memory_mb_mean"]))
        print("{:<40} {:<15.2f}".format("Peak Memory Median (MB):", metrics["peak_memory_mb_median"]))

    if metrics["stage_durations_mean"]:
        print(f"{'-' * 50}")
        print("Stage Durations Mean (s):")
        for stage, val in metrics["stage_durations_mean"].items():
            print("{:<40} {:<15.4f}".format(f"  {stage}:", val))

    print("\n" + "=" * 60)

    if args.save_dir:
        _save_generated_outputs(outputs, requests_list, args.save_dir)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark serving for diffusion models.")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL of the server (e.g., http://localhost:8091). Overrides host/port.",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=8091, help="Server port.")
    parser.add_argument("--model", type=str, default="default", help="Model name.")
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help=("Endpoint path to target. Leading '/' is optional, e.g. /v1/videos or v1/videos."),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="vbench",
        choices=["vbench", "trace", "random", "custom"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v",
        choices=["t2v", "i2v", "ti2v", "ti2i", "i2i", "it2i", "t2i"],
        help="Task type.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local dataset file (optional).",
    )
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts to benchmark.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent requests, default to `1`. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Number of warmup requests to run before measurement.",
    )
    # NOTE Changed default from 1 to 2 because some models (e.g., Bagel) run
    # `num_timesteps - 1` denoising iterations. A default of 1 results in 0 steps,
    # which causes errors.
    # TODO If this slightly longer warmup causes regression issues for other
    # diffusion pipelines in the future, consider implementing model-specific
    # overrides instead of a global default.
    parser.add_argument(
        "--warmup-num-inference-steps",
        type=int,
        default=2,
        help="Number of inference steps used for warmup requests. "
        "Default is 2 to ensure at least one denoising step is executed.",
    )
    parser.add_argument(
        "--warmup-concurrency",
        type=int,
        default=1,
        help="Maximum number of warmup requests to run concurrently. "
        "Set this to match the real batch shape when warming up torch.compile or CUDA graphs.",
    )
    parser.add_argument("--width", type=int, default=None, help="Image/Video width.")
    parser.add_argument("--height", type=int, default=None, help="Image/Video height.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames (for video).")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps (for diffusion models).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (for diffusion models).",
    )
    parser.add_argument("--fps", type=int, default=None, help="FPS (for video).")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSON file for metrics.")
    parser.add_argument(
        "--slo",
        action="store_true",
        help=(
            "Enable SLO calculation and reporting. If trace provides per-request slo_ms, it is used. "
            "Otherwise, warmup request(s) are used to infer expected execution time assuming linear "
            "scaling by resolution, frames, and steps, then slo_ms = expected_time * --slo-scale."
        ),
    )
    parser.add_argument(
        "--slo-scale",
        type=float,
        default=3.0,
        help="SLO target multiplier: slo_ms = estimated_exec_time_ms * slo_scale (default: 3).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save generated images/outputs for visual inspection. "
        "If not set, generated outputs are discarded after metric collection.",
    )
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable progress bar.")
    parser.add_argument(
        "--enable-negative-prompt",
        action="store_true",
        default=False,
        help="Generate negative prompts when using the random dataset.",
    )
    parser.add_argument(
        "--random-request-config",
        type=str,
        default=None,
        help=(
            "JSON string defining random request profiles. "
            "Each profile may contain: width, height, num_inference_steps, etc. "
            "The 'weight' field controls sampling probability (relative weight). "
            "Example: "
            '[{"width":512,"height":512,"num_inference_steps":20,"weight":0.15},'
            '{"width":768,"height":768,"num_inference_steps":20,"weight":0.85}]'
        ),
    )
    parser.add_argument(
        "--num-input-images",
        type=int,
        default=1,
        help=(
            "Number of synthetic input images to attach for image-conditioned tasks "
            "(i2v, ti2v, ti2i, i2i, it2i) when using random dataset."
        ),
    )
    parser.add_argument(
        "--bot-task",
        type=str,
        default="think",
        help=("bot_task form field for --endpoint /v1/images/edits (think, recaption, think_recaption, vanilla)."),
    )
    parser.add_argument(
        "--return-stage-metrics",
        action="store_true",
        help="Request stage duration metrics from endpoints that support return_stage_metrics.",
    )

    args = parser.parse_args()
    asyncio.run(benchmark(args))
