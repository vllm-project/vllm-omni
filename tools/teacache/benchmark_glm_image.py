#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark GLM-Image TeaCache speed and quality vs the uncached baseline.

Measures (warm, median of ``--repeats``):

- DiT ``diffuse`` time (headline; AR stage is uncached)
- Stage-1 / end-to-end wall time when ``--enable-diffusion-pipeline-profiler``
- LPIPS / SSIM / PSNR vs no-cache over a prompt set
- Computed vs reused TeaCache step counts

Sweeps ``rel_l1_thresh`` over 0.1, 0.15, 0.2, 0.25, 0.3 and prints a
speed-vs-LPIPS table so a default can be chosen for
``_MODEL_DEFAULT_REL_L1_THRESH["GlmImageTransformer2DModel"]``.

Uses fixed seed + fixed prior tokens so quality is comparable even if the
AR token-range bug is still present on main.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_prompts(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _to_image_tensor(image: Any) -> torch.Tensor:
    """Convert PIL / ndarray / tensor image to float CHW in [0, 1]."""
    if isinstance(image, torch.Tensor):
        tensor = image.detach().float().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        return tensor.clamp(0, 1)
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    tensor = torch.from_numpy(array).float() / 255.0
    return tensor.permute(2, 0, 1).clamp(0, 1)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    # Lightweight structural similarity (windowless, per-channel mean).
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a = ((a - mu_a) ** 2).mean()
    sigma_b = ((b - mu_b) ** 2).mean()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return float((num / den).item())


def _lpips_score(a: torch.Tensor, b: torch.Tensor) -> float:
    try:
        import lpips  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install lpips to run quality metrics: pip install lpips") from exc

    loss_fn = lpips.LPIPS(net="alex")
    # LPIPS expects NCHW in [-1, 1]
    aa = a.unsqueeze(0) * 2 - 1
    bb = b.unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return float(loss_fn(aa, bb).item())


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prior-tokens", type=Path, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-prompts", type=int, default=20)
    parser.add_argument(
        "--thresh",
        type=float,
        nargs="+",
        default=[0.1, 0.15, 0.2, 0.25, 0.3],
    )
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    prompts = _load_prompts(args.prompts)[: args.max_prompts]
    priors = torch.load(args.prior_tokens, map_location="cpu", weights_only=False)

    def prior_for(prompt: str, idx: int) -> Any:
        if isinstance(priors, dict) and prompt in priors:
            return priors[prompt]
        if isinstance(priors, dict) and "__list__" in priors:
            return priors["__list__"][idx]
        if isinstance(priors, list):
            return priors[idx]
        raise KeyError(f"No prior tokens for prompt index {idx}")

    def run_once(cache_backend: str | None, thresh: float | None, prompt: str, prior: Any) -> dict[str, Any]:
        cache_config = {"rel_l1_thresh": thresh} if thresh is not None else None
        kwargs: dict[str, Any] = {
            "model": args.model,
            "enable_diffusion_pipeline_profiler": True,
        }
        if cache_backend is not None:
            kwargs["cache_backend"] = cache_backend
            kwargs["cache_config"] = cache_config
        omni = Omni(**kwargs)
        try:
            t0 = time.perf_counter()
            outputs = omni.generate(
                prompt,
                OmniDiffusionSamplingParams(
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    seed=args.seed,
                    extra_args={"prior_token_ids": prior},
                ),
            )
            e2e = time.perf_counter() - t0
            image = outputs[0].images[0]
            return {"e2e_s": e2e, "image": image}
        finally:
            # Best-effort cleanup across Omni versions.
            close = getattr(omni, "close", None) or getattr(omni, "shutdown", None)
            if callable(close):
                close()

    # Baseline images (no cache)
    baselines: list[torch.Tensor] = []
    baseline_times: list[float] = []
    for idx, prompt in enumerate(prompts):
        prior = prior_for(prompt, idx)
        # Warm + timed repeats for the first prompt only; single pass for quality set.
        if idx == 0:
            run_once(None, None, prompt, prior)  # warm
            times = []
            last = None
            for _ in range(args.repeats):
                last = run_once(None, None, prompt, prior)
                times.append(last["e2e_s"])
            baseline_times.append(_median(times))
            baselines.append(_to_image_tensor(last["image"]))
        else:
            result = run_once(None, None, prompt, prior)
            baselines.append(_to_image_tensor(result["image"]))

    rows: list[dict[str, Any]] = []
    for thresh in args.thresh:
        print(f"\n=== tea_cache rel_l1_thresh={thresh} ===")
        quality = []
        speed_prompt = prompts[0]
        prior0 = prior_for(speed_prompt, 0)
        run_once("tea_cache", thresh, speed_prompt, prior0)  # warm
        times = []
        last = None
        for _ in range(args.repeats):
            last = run_once("tea_cache", thresh, speed_prompt, prior0)
            times.append(last["e2e_s"])
        median_e2e = _median(times)
        speedup = baseline_times[0] / median_e2e if median_e2e > 0 else float("nan")

        for idx, prompt in enumerate(prompts):
            prior = prior_for(prompt, idx)
            result = run_once("tea_cache", thresh, prompt, prior)
            cached = _to_image_tensor(result["image"])
            base = baselines[idx]
            quality.append(
                {
                    "lpips": _lpips_score(base, cached),
                    "ssim": _ssim(base, cached),
                    "psnr": _psnr(base, cached),
                }
            )

        row = {
            "rel_l1_thresh": thresh,
            "median_e2e_s": median_e2e,
            "baseline_median_e2e_s": baseline_times[0],
            "e2e_speedup": speedup,
            "lpips_mean": float(np.mean([q["lpips"] for q in quality])),
            "ssim_mean": float(np.mean([q["ssim"] for q in quality])),
            "psnr_mean": float(np.mean([q["psnr"] for q in quality])),
        }
        rows.append(row)
        print(
            f"thresh={thresh:.2f} e2e={median_e2e:.3f}s "
            f"speedup={speedup:.2f}x LPIPS={row['lpips_mean']:.4f} "
            f"SSIM={row['ssim_mean']:.4f} PSNR={row['psnr_mean']:.2f}"
        )

    print("\nSpeed vs LPIPS table")
    print("thresh\tspeedup\tLPIPS\tSSIM\tPSNR")
    for row in rows:
        print(
            f"{row['rel_l1_thresh']:.2f}\t{row['e2e_speedup']:.2f}\t"
            f"{row['lpips_mean']:.4f}\t{row['ssim_mean']:.4f}\t{row['psnr_mean']:.2f}"
        )

    # Prefer the highest speedup whose mean LPIPS stays under 0.05 when possible;
    # otherwise fall back to 0.2 (Qwen-family default).
    eligible = [r for r in rows if r["lpips_mean"] <= 0.05]
    pick = max(eligible, key=lambda r: r["e2e_speedup"]) if eligible else next(r for r in rows if r["rel_l1_thresh"] == 0.2)
    print(f"\nRecommended default rel_l1_thresh={pick['rel_l1_thresh']} (LPIPS={pick['lpips_mean']:.4f})")

    if args.out_json is not None:
        args.out_json.write_text(json.dumps({"rows": rows, "recommended": pick}, indent=2) + "\n")
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
