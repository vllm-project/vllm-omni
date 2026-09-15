# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Diffusion dataset loaders for ``vllm bench serve``.

Ports prompt-loading logic from the standalone
``benchmarks/diffusion/diffusion_benchmark_serving.py`` script so that
diffusion models can be benchmarked through the unified CLI.

Each loader returns ``list[SampleRequest]`` with ``prompt_len=0`` and
``expected_output_len=0``.  Generation parameters (width, height,
num_inference_steps, etc.) are placed in ``request_overrides`` so they
flow into the HTTP request body via ``_merge_overrides()`` at dispatch
time.
"""

from __future__ import annotations

import json
import logging
import os
import random
from argparse import Namespace
from typing import Any

import requests as http_requests

from vllm.benchmarks.datasets import SampleRequest

logger = logging.getLogger(__name__)

VBENCH_T2V_PROMPT_URL = (
    "https://raw.githubusercontent.com/Vchitect/VBench/master/"
    "prompts/prompts_per_dimension/subject_consistency.txt"
)

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "vllm-omni")


def _build_overrides(args: Namespace) -> dict[str, Any]:
    """Build ``request_overrides`` from CLI args for diffusion generation."""
    overrides: dict[str, Any] = {}
    for key in ("width", "height", "num_inference_steps", "num_frames", "fps"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    seed = getattr(args, "seed", None)
    if seed is not None:
        overrides["seed"] = seed
    return overrides


def _resize(
    items: list[dict[str, Any]], num_prompts: int
) -> list[dict[str, Any]]:
    """Repeat or truncate *items* to exactly *num_prompts* entries."""
    if not items:
        raise ValueError("No diffusion prompts available")
    if num_prompts <= 0:
        return items
    if len(items) < num_prompts:
        factor = (num_prompts // len(items)) + 1
        items = items * factor
    return items[:num_prompts]


def _load_vbench_prompts(args: Namespace) -> list[dict[str, Any]]:
    """Load VBench text prompts (t2v subject-consistency set)."""
    path = getattr(args, "dataset_path", None)
    if not path:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = os.path.join(_CACHE_DIR, "vbench_subject_consistency.txt")
        if not os.path.exists(path):
            logger.info("Downloading VBench T2V prompts to %s", path)
            try:
                resp = http_requests.get(VBENCH_T2V_PROMPT_URL, timeout=30)
                resp.raise_for_status()
                with open(path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
            except Exception as exc:
                logger.warning("Failed to download VBench prompts: %s", exc)
                return [{"prompt": "A cat sitting on a bench"}] * 50

    prompts: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append({"prompt": line})
    return prompts


def _load_random_prompts(args: Namespace) -> list[dict[str, Any]]:
    """Generate synthetic prompts, optionally using weighted resolution profiles."""
    num_prompts = getattr(args, "num_prompts", 10) or 10
    config_str = getattr(args, "random_request_config", None)

    if config_str:
        profiles = json.loads(config_str)
        weights = [p.pop("weight", 1.0) for p in profiles]
        rng = random.Random(getattr(args, "seed", 42))
        sampled = rng.choices(profiles, weights=weights, k=num_prompts)
        items: list[dict[str, Any]] = []
        for idx, profile in enumerate(sampled):
            prompt_text = profile.pop(
                "prompt",
                f"Random prompt {idx} for benchmarking diffusion models",
            )
            items.append({"prompt": prompt_text, "overrides": dict(profile)})
        return items

    return [
        {"prompt": f"Random prompt {i} for benchmarking diffusion models"}
        for i in range(num_prompts)
    ]


def _load_custom_prompts(args: Namespace) -> list[dict[str, Any]]:
    """Load prompts from a JSONL file."""
    path = getattr(args, "dataset_path", None)
    if not path:
        raise ValueError(
            "--dataset-path is required when using --diffusion-dataset custom"
        )
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}: {exc}"
                ) from exc
            if "prompt" not in row:
                raise ValueError(
                    f"Missing 'prompt' field at {path}:{line_no}"
                )
            items.append(row)
    logger.info("Loaded %d custom diffusion prompts from %s", len(items), path)
    return items


_DATASET_LOADERS = {
    "vbench": _load_vbench_prompts,
    "random": _load_random_prompts,
    "custom": _load_custom_prompts,
}


def load_diffusion_samples(args: Namespace) -> list[SampleRequest]:
    """Load diffusion benchmark prompts and return ``SampleRequest`` objects.

    Dispatches by ``args.diffusion_dataset`` (default: ``"random"``).
    """
    dataset_name = getattr(args, "diffusion_dataset", "random") or "random"
    loader = _DATASET_LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(
            f"Unknown diffusion dataset: {dataset_name!r}. "
            f"Choose from {sorted(_DATASET_LOADERS)}"
        )

    raw_items = loader(args)
    num_prompts = getattr(args, "num_prompts", 0) or 0
    if num_prompts > 0:
        raw_items = _resize(raw_items, num_prompts)

    base_overrides = _build_overrides(args)
    requests: list[SampleRequest] = []

    for idx, item in enumerate(raw_items):
        prompt = item.get("prompt", "")
        per_request = dict(base_overrides)

        if "overrides" in item:
            per_request.update(item["overrides"])

        for key in (
            "width",
            "height",
            "num_inference_steps",
            "num_frames",
            "fps",
            "seed",
        ):
            if key in item and key not in (item.get("overrides") or {}):
                per_request.setdefault(key, item[key])

        requests.append(
            SampleRequest(
                prompt=prompt,
                prompt_len=0,
                expected_output_len=0,
                request_overrides=per_request if per_request else None,
                request_id=str(idx),
            )
        )

    logger.info(
        "Prepared %d diffusion SampleRequest(s) from %s dataset",
        len(requests),
        dataset_name,
    )
    return requests
