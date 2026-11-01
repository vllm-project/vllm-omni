#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Calibrate TeaCache polynomial coefficients for GlmImageTransformer2DModel.

Workflow (matches the GLM-Image TeaCache plan):

1. Dump AR prior tokens for ~70 Parti prompts (or load a pre-dumped ``.pt``).
2. Run the DiT with ``DataCollectionHook`` via ``GlmImageAdapter`` at
   1024x1024 / 50 steps, feeding ``prior_token_ids`` through
   ``OmniDiffusionSamplingParams.extra_args``.
3. Fit a 4th-order polynomial and print coeffs + R^2 for
   ``vllm_omni/diffusion/cache/teacache/config.py``.

Example::

    python tools/teacache/calibrate_glm_image.py \\
        --model zai-org/GLM-Image \\
        --prior-tokens /data/glm_image_parti_priors.pt \\
        --prompts /data/parti_prompts.txt \\
        --height 1024 --width 1024 --steps 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.cache.teacache.coefficient_estimator import (
    TeaCacheCoefficientEstimator,
    calculate_relative_l1,
    estimate_teacache_coefficients,
)


def _load_prompts(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _load_prior_tokens(path: Path) -> dict[str, Any]:
    """Load a torch dump of ``{prompt: prior_token_ids tensor/list}`` or a list."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"__list__": payload}
    raise TypeError(f"Unsupported prior-token dump type: {type(payload)}")


def _r_squared(collected: list[list[tuple[np.ndarray, np.ndarray]]], coeffs: list[float]) -> float:
    xs: list[float] = []
    ys: list[float] = []
    for sample in collected:
        for t in range(len(sample) - 1):
            feat_in_curr, feat_out_curr = sample[t]
            feat_in_next, feat_out_next = sample[t + 1]
            xs.append(calculate_relative_l1(feat_in_curr, feat_in_next))
            ys.append(calculate_relative_l1(feat_out_curr, feat_out_next))
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    pred = np.poly1d(coeffs)(x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF id or local path for GlmImagePipeline")
    parser.add_argument("--prompts", type=Path, required=True, help="Text file, one prompt per line")
    parser.add_argument(
        "--prior-tokens",
        type=Path,
        required=True,
        help="Torch dump of prior_token_ids keyed by prompt (or a list aligned to prompts)",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--poly-order", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=70)
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to write fit summary")
    args = parser.parse_args()

    prompts = _load_prompts(args.prompts)[: args.max_prompts]
    priors = _load_prior_tokens(args.prior_tokens)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    estimator = TeaCacheCoefficientEstimator(
        model_path=args.model,
        model_type="GlmImage",
        device=args.device,
        dtype=dtype,
    )

    for idx, prompt in enumerate(prompts):
        if "__list__" in priors:
            prior = priors["__list__"][idx]
        else:
            prior = priors[prompt]
        print(f"[{idx + 1}/{len(prompts)}] collecting: {prompt[:80]!r}")
        estimator.collect_from_prompt(
            prompt,
            num_inference_steps=args.steps,
            seed=args.seed,
            height=args.height,
            width=args.width,
            prior_token_ids=prior,
        )

    coeffs = estimate_teacache_coefficients(estimator.collected_data, poly_order=args.poly_order)
    r2 = _r_squared(estimator.collected_data, coeffs)

    xs = []
    for sample in estimator.collected_data:
        for t in range(len(sample) - 1):
            xs.append(calculate_relative_l1(sample[t][0], sample[t + 1][0]))
    x = np.asarray(xs, dtype=np.float64)

    print("\n=== GLM-Image TeaCache fit ===")
    print(f"prompts={len(estimator.collected_data)} step_pairs={len(x)} R^2={r2:.4f}")
    print(f"input rel_l1 range: {x.min():.4f} .. {x.max():.4f} (mean {x.mean():.4f})")
    print("coefficients:")
    print(json.dumps(coeffs, indent=2))

    if args.out_json is not None:
        args.out_json.write_text(
            json.dumps(
                {
                    "coefficients": coeffs,
                    "r_squared": r2,
                    "input_rel_l1_min": float(x.min()),
                    "input_rel_l1_max": float(x.max()),
                    "input_rel_l1_mean": float(x.mean()),
                    "num_prompts": len(estimator.collected_data),
                    "num_step_pairs": len(x),
                    "height": args.height,
                    "width": args.width,
                    "steps": args.steps,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
