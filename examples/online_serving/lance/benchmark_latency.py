#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repeatable Lance online latency benchmark.

This measures the OpenAI-compatible HTTP path end to end.  Server-side
``--log-stats`` and ``--enable-diffusion-pipeline-profiler`` can be enabled to
collect stage timings without using the heavier torch profiler.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
VIDEO_PAD = "<|video_pad|>"
_VISION_BLOCK = f"{VISION_START}{VIDEO_PAD}{VISION_END}"

_SYSTEM_PROMPTS = {
    "t2i": (
        "Describe the image by detailing the color, quantity, text, shape, "
        "size, texture, spatial relationships of the objects and background:"
    ),
    "image_edit": (
        "Describe the key features of the input image (color, shape, size, "
        "texture, objects, background), then explain how the user's text "
        "instruction should alter or modify the image. Generate a new image "
        "that meets the user's requirements while maintaining consistency "
        "with the original input where appropriate."
    ),
}


def _render_lance_prompt(task: str, user_text: str, vision_token: str | None = None) -> str:
    user_msg = user_text if vision_token is None else f"{vision_token}{user_text}"
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPTS[task]}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    )


def _image_url(path_or_url: str) -> str:
    path = Path(path_or_url)
    if not path.exists():
        return path_or_url
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.modality == "text2img":
        prompt = _render_lance_prompt("t2i", args.prompt)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    elif args.modality == "img2img":
        if not args.image_url:
            raise ValueError("--image-url is required for img2img")
        prompt = _render_lance_prompt("image_edit", args.prompt, vision_token=_VISION_BLOCK)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_url(args.image_url)}},
        ]
    else:
        raise ValueError(f"Unsupported modality: {args.modality}")

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image"],
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
    }
    if args.negative_prompt:
        payload["negative_prompt"] = args.negative_prompt
    return payload


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    stage_durations = out.pop("stage_durations", None)
    if isinstance(stage_durations, dict):
        normalized = {}
        for name, value in stage_durations.items():
            if not isinstance(value, int | float):
                continue
            key = str(name)
            normalized[key if key.endswith("_ms") else f"{key}_ms"] = float(value) if key.endswith("_ms") else float(value) * 1000.0
        out["stage_durations_ms"] = normalized
    return out


def _extract_metrics(data: dict[str, Any]) -> dict[str, Any]:
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        return _normalize_metrics(metrics)
    for choice in data.get("choices", []):
        content = choice.get("message", {}).get("content")
        if isinstance(content, list):
            for item in content:
                stage_durations = item.get("stage_durations")
                if isinstance(stage_durations, dict):
                    return _normalize_metrics(
                        {
                            "stage_durations": stage_durations,
                            "peak_memory_mb": item.get("peak_memory_mb"),
                        }
                    )
    return {}


def _post_once(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
    started = time.perf_counter()
    response = requests.post(
        f"{args.server.rstrip('/')}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=args.timeout,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()
    return elapsed_ms, len(response.content), _extract_metrics(response.json())


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Lance image end-to-end online latency.")
    parser.add_argument("--server", default="http://localhost:8091")
    parser.add_argument("--model", default="bytedance-research/Lance")
    parser.add_argument("--modality", choices=["text2img", "img2img"], default="text2img")
    parser.add_argument("--prompt", default="A cute corgi astronaut on the moon, cinematic")
    parser.add_argument("--image-url")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    payload = _build_payload(args)
    records: list[dict[str, Any]] = []

    for i in range(args.warmup):
        elapsed_ms, response_bytes, metrics = _post_once(args, payload)
        print(
            json.dumps(
                {
                    "phase": "warmup",
                    "iter": i,
                    "client_latency_ms": elapsed_ms,
                    "response_bytes": response_bytes,
                    "metrics": metrics,
                },
                sort_keys=True,
            )
        )

    for i in range(args.iters):
        elapsed_ms, response_bytes, metrics = _post_once(args, payload)
        rec = {
            "phase": "measure",
            "iter": i,
            "client_latency_ms": elapsed_ms,
            "response_bytes": response_bytes,
            "metrics": metrics,
        }
        records.append(rec)
        print(json.dumps(rec, sort_keys=True))

    latencies = [r["client_latency_ms"] for r in records]
    summary = {
        "client_latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "client_latency_ms_min": min(latencies) if latencies else None,
        "client_latency_ms_max": max(latencies) if latencies else None,
        "client_latency_ms_p50": statistics.median(latencies) if latencies else None,
        "iters": args.iters,
        "warmup": args.warmup,
        "payload": payload,
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps({"records": records, "summary": summary}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
