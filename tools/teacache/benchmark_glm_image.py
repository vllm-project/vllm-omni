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

One Omni engine is constructed per cache configuration, warmed once, then
reused for all timed repeats and quality prompts for that configuration.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
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


def _close_omni(omni: Any) -> None:
    close = getattr(omni, "close", None) or getattr(omni, "shutdown", None)
    if callable(close):
        close()


class OmniSession:
    """Own one Omni engine for warmup + measured repeats of a cache config."""

    def __init__(self, factory: Callable[..., Any], **engine_kwargs: Any):
        self.omni = factory(**engine_kwargs)
        self._closed = False
        self.generate_calls = 0

    def generate(self, prompt: str, sampling_params: Any) -> Any:
        self.generate_calls += 1
        return self.omni.generate(prompt, sampling_params)

    def close(self) -> None:
        if not self._closed:
            _close_omni(self.omni)
            self._closed = True

    def __enter__(self) -> OmniSession:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def warm_and_time(
    session: OmniSession,
    *,
    prompt: str,
    sampling_params: Any,
    repeats: int,
) -> tuple[list[float], Any]:
    """Warm once on ``session``, then time ``repeats`` generations.

    Construction must happen before this helper; the warmup generate is not
    timed, and every timed sample reuses the same engine instance.
    """
    session.generate(prompt, sampling_params)  # warm (untimed)
    times: list[float] = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = session.generate(prompt, sampling_params)
        times.append(time.perf_counter() - t0)
    return times, last


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

    def make_params(prior: Any) -> OmniDiffusionSamplingParams:
        return OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            seed=args.seed,
            extra_args={"prior_token_ids": prior},
        )

    def engine_kwargs(cache_backend: str | None, thresh: float | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": args.model,
            "enable_diffusion_pipeline_profiler": True,
        }
        if cache_backend is not None:
            kwargs["cache_backend"] = cache_backend
            kwargs["cache_config"] = {"rel_l1_thresh": thresh}
        return kwargs

    # Baseline images (no cache): one engine, warm once, time repeats, then quality.
    baselines: list[torch.Tensor] = []
    baseline_times: list[float] = []
    with OmniSession(Omni, **engine_kwargs(None, None)) as baseline_session:
        speed_prompt = prompts[0]
        prior0 = prior_for(speed_prompt, 0)
        times, last = warm_and_time(
            baseline_session,
            prompt=speed_prompt,
            sampling_params=make_params(prior0),
            repeats=args.repeats,
        )
        baseline_times.append(_median(times))
        baselines.append(_to_image_tensor(last[0].images[0]))
        for idx, prompt in enumerate(prompts):
            if idx == 0:
                continue
            prior = prior_for(prompt, idx)
            outputs = baseline_session.generate(prompt, make_params(prior))
            baselines.append(_to_image_tensor(outputs[0].images[0]))

    rows: list[dict[str, Any]] = []
    for thresh in args.thresh:
        print(f"\n=== tea_cache rel_l1_thresh={thresh} ===")
        quality = []
        with OmniSession(Omni, **engine_kwargs("tea_cache", thresh)) as session:
            times, last = warm_and_time(
                session,
                prompt=speed_prompt,
                sampling_params=make_params(prior0),
                repeats=args.repeats,
            )
            median_e2e = _median(times)
            speedup = baseline_times[0] / median_e2e if median_e2e > 0 else float("nan")
            # Reuse the last timed image for prompt 0 quality.
            quality.append(
                {
                    "lpips": _lpips_score(baselines[0], _to_image_tensor(last[0].images[0])),
                    "ssim": _ssim(baselines[0], _to_image_tensor(last[0].images[0])),
                    "psnr": _psnr(baselines[0], _to_image_tensor(last[0].images[0])),
                }
            )
            for idx, prompt in enumerate(prompts):
                if idx == 0:
                    continue
                prior = prior_for(prompt, idx)
                outputs = session.generate(prompt, make_params(prior))
                cached = _to_image_tensor(outputs[0].images[0])
                quality.append(
                    {
                        "lpips": _lpips_score(baselines[idx], cached),
                        "ssim": _ssim(baselines[idx], cached),
                        "psnr": _psnr(baselines[idx], cached),
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
    # otherwise fall back to 0.2 (Qwen-family default). Still provisional until
    # a real GLM Parti sweep is attached to the PR.
    eligible = [r for r in rows if r["lpips_mean"] <= 0.05]
    pick = (
        max(eligible, key=lambda r: r["e2e_speedup"])
        if eligible
        else next(r for r in rows if r["rel_l1_thresh"] == 0.2)
    )
    print(f"\nRecommended default rel_l1_thresh={pick['rel_l1_thresh']} (LPIPS={pick['lpips_mean']:.4f})")

    if args.out_json is not None:
        args.out_json.write_text(json.dumps({"rows": rows, "recommended": pick}, indent=2) + "\n")
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
