# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 vs FP8 perf/mem/quality bench for HunyuanVideo-1.5 and Wan2.2.

Runs each (model, precision) pair with the same seed, records peak VRAM and
wall time, then computes frame-averaged LPIPS / PSNR / SSIM between the BF16
and FP8 outputs. Emits a markdown table suitable for pasting into a PR.

Target: 1×H100 80GB. Wan2.2 uses the TI2V-5B dense checkpoint (fits in 80GB
BF16); for A14B MoE you need 2×H100 + TP=2 or CPU offload.

Deps:
    pip install lpips scikit-image pynvml imageio[ffmpeg]

Usage:
    # single model, default config
    python benchmarks/diffusion/bench_video_fp8.py --only hv15

    # HV-1.5 sweep: 4 FP8 presets × 2 frame counts (short + long).
    # S1 = FFN only, S2 = video stream only, S3 = all FP8 except encoder cross-attn,
    # S4 = everything FP8. BF16 is captured once per frame count and reused.
    python benchmarks/diffusion/bench_video_fp8.py --only hv15 \\
        --presets S1,S2,S3,S4 --frames-list 33,121 --steps 30 --tag sweep

    # Winning preset + block-wise FP8 (per-tile scales, closer to BF16 quality).
    # Block size [128, 128] is the standard; try [1, 128] for per-output-row precision.
    python benchmarks/diffusion/bench_video_fp8.py --only hv15 \\
        --presets S4 --weight-block-size 128,128 --num-frames 33 --tag bs128

    # Custom prompt + long-only run
    python benchmarks/diffusion/bench_video_fp8.py --only hv15 \\
        --prompt "An astronaut riding a horse on Mars" \\
        --num-frames 121 --steps 40 --tag astronaut
"""

from __future__ import annotations

import argparse
import gc
import os
import threading
import time

import numpy as np
import torch

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

CASES = {
    "hv15": dict(
        id="HunyuanVideo-1.5",
        model="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        task="T2V 480x832, 33 frames, 30 steps",
        kwargs=dict(
            height=480,
            width=832,
            num_inference_steps=30,
            num_frames=33,
            guidance_scale=6.0,
        ),
    ),
    "wan22": dict(
        id="Wan2.2 (TI2V 5B)",
        model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        task="T2V 704x1280, 49 frames, 30 steps",
        kwargs=dict(
            height=704,
            width=1280,
            num_inference_steps=30,
            num_frames=49,
            guidance_scale=5.0,
        ),
    ),
}
PROMPT = "A dog running across a field of golden wheat."
SEED = 42


def _extract_video(outputs) -> np.ndarray:
    """Return [T, H, W, 3] float in [0, 1]. Handles any dim order / batch wrappers."""
    from vllm_omni.outputs import OmniRequestOutput

    first = outputs[0]
    if hasattr(first, "request_output") and isinstance(first.request_output, list):
        inner = first.request_output[0]
        frames = inner.images[0] if isinstance(inner, OmniRequestOutput) and inner.images else inner
    elif hasattr(first, "images") and first.images:
        frames = first.images
    else:
        raise ValueError("Cannot extract frames from output.")

    if isinstance(frames, torch.Tensor):
        v = frames.detach().cpu().float().numpy()
    else:
        v = np.asarray(frames)
        if v.dtype == np.uint8:
            v = v.astype(np.float32) / 255.0
        else:
            v = v.astype(np.float32)

    print(f"    _extract_video raw shape: {v.shape}, dtype: {v.dtype}", flush=True)

    # Clamp [-1, 1] → [0, 1] if signed
    if v.dtype.kind == "f" and v.min() < -1e-3:
        v = np.clip(v, -1.0, 1.0) * 0.5 + 0.5

    # Strip leading size-1 dims, then any extra leading dims (take first)
    while v.ndim > 4 and v.shape[0] == 1:
        v = v[0]
    while v.ndim > 4:
        v = v[0]

    if v.ndim != 4:
        raise ValueError(f"Expected 4D video after reduction, got {v.shape}")

    # Permute to [T, H, W, C]. Try in order: already-correct, [C, T, H, W], [T, C, H, W].
    if v.shape[-1] in (1, 3, 4):
        pass  # already [T, H, W, C]
    elif v.shape[0] in (1, 3, 4):
        v = np.transpose(v, (1, 2, 3, 0))  # [C, T, H, W] -> [T, H, W, C]
    elif v.shape[1] in (1, 3, 4):
        v = np.transpose(v, (0, 2, 3, 1))  # [T, C, H, W] -> [T, H, W, C]
    else:
        raise ValueError(f"Cannot identify channel axis in shape {v.shape}")

    print(f"    _extract_video normalized shape: {v.shape}", flush=True)
    return v


def _nvml_used_gib() -> float:
    """Process-wide GPU-0 used memory in GiB. Captures the worker subprocess too."""
    import pynvml

    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024**3)
    finally:
        pynvml.nvmlShutdown()


class _PeakPoller:
    """Polls process-wide NVML 'used' memory every 100ms, tracks peak."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _PeakPoller:
        def loop() -> None:
            while not self._stop.is_set():
                self.peak = max(self.peak, _nvml_used_gib())
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def run(
    model: str,
    quantization: str | dict | None,
    kwargs: dict,
    prompt: str = PROMPT,
) -> tuple[float, float, np.ndarray]:
    """One (model, precision) pair: warmup + timed run. Returns (peak_gib, seconds, video).

    `quantization` may be None (BF16), a string like "fp8", or a dict like
    `{"method": "fp8", "weight_block_size": [128, 128]}` for finer-grained FP8 config.

    Uses integer `seed=` via sampling params (pickles to the worker subprocess correctly —
    `torch.Generator` handles do not propagate across processes). Peak VRAM is captured via
    pynvml polling so the worker's memory counts.
    """
    omni_kw: dict = {"model": model}
    if quantization:
        omni_kw["quantization"] = quantization
    omni = Omni(**omni_kw)
    try:
        # warmup
        omni.generate(prompt, OmniDiffusionSamplingParams(**kwargs, seed=0))
        torch.cuda.synchronize()

        with _PeakPoller() as poll:
            t0 = time.perf_counter()
            outputs = omni.generate(prompt, OmniDiffusionSamplingParams(**kwargs, seed=SEED))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        peak = poll.peak
        video = _extract_video(outputs)
        return peak, elapsed, video
    finally:
        del omni
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def save_video(video: np.ndarray, path: str, fps: int = 24) -> None:
    """Save [T, H, W, 3] float video in [0, 1] as MP4."""
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames_uint8 = (np.clip(video, 0.0, 1.0) * 255).astype(np.uint8)
    imageio.mimsave(path, list(frames_uint8), fps=fps, codec="libx264", quality=8)


def compute_metrics(bf16: np.ndarray, fp8: np.ndarray) -> dict[str, float]:
    """Frame-averaged LPIPS (lower better), PSNR (higher better), SSIM (higher better)."""
    import lpips
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    t = min(bf16.shape[0], fp8.shape[0])
    bf16, fp8 = bf16[:t], fp8[:t]

    # LPIPS expects [B, 3, H, W] in [-1, 1]
    lp = lpips.LPIPS(net="alex").cuda().eval()

    def to_lp(x: np.ndarray) -> torch.Tensor:
        return (torch.from_numpy(x).permute(0, 3, 1, 2).cuda() * 2 - 1).float()

    with torch.no_grad():
        lpips_scores = lp(to_lp(bf16), to_lp(fp8)).squeeze().cpu().numpy()
    mean_lpips = float(np.mean(lpips_scores))

    psnrs, ssims = [], []
    for a, b in zip(bf16, fp8, strict=False):
        psnrs.append(peak_signal_noise_ratio(a, b, data_range=1.0))
        ssims.append(structural_similarity(a, b, data_range=1.0, channel_axis=-1))
    return dict(lpips=mean_lpips, psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=list(CASES), default=None)
    p.add_argument("--output-dir", default="./bench_output", help="Where to save MP4s.")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--prompt", type=str, default=None, help="Override the default prompt.")
    p.add_argument("--num-frames", type=int, default=None, help="Override num_frames (single value).")
    p.add_argument("--steps", type=int, default=None, help="Override num_inference_steps.")
    p.add_argument("--tag", type=str, default="", help="Suffix for output filenames.")
    p.add_argument(
        "--presets",
        type=str,
        default="S4",
        help="Comma-separated FP8 presets for HV-1.5 sweep (BF16,S1,S2,S3,S4). Ignored for other models.",
    )
    p.add_argument(
        "--frames-list",
        type=str,
        default=None,
        help="Comma-separated list of num_frames to sweep (e.g. '33,121'). Overrides --num-frames.",
    )
    p.add_argument(
        "--weight-block-size",
        type=str,
        default=None,
        help="FP8 weight quantization block size as 'M,N' (e.g. '128,128' for block-wise scales). "
        "Default: per-tensor (one scale per linear). Requires activation_scheme=dynamic (default).",
    )
    args = p.parse_args()

    weight_block_size: list[int] | None = None
    if args.weight_block_size:
        parts = [int(x) for x in args.weight_block_size.split(",") if x.strip()]
        if len(parts) != 2:
            raise SystemExit(f"--weight-block-size must be 'M,N' (2 ints), got {args.weight_block_size!r}")
        weight_block_size = parts
        print(f"FP8 weight_block_size: {weight_block_size}", flush=True)

    prompt = args.prompt or PROMPT
    print(f"Prompt: {prompt}", flush=True)

    presets = [s.strip().upper() for s in args.presets.split(",") if s.strip()]
    valid_presets = {"BF16", "S1", "S2", "S3", "S4"}
    for pr in presets:
        if pr not in valid_presets:
            raise SystemExit(f"Invalid preset {pr!r}. Expected one of {sorted(valid_presets)}")

    keys = [args.only] if args.only else list(CASES)
    suffix_tag = f"_{args.tag}" if args.tag else ""

    rows: list[str] = []
    for case_key in keys:
        base = CASES[case_key]
        base_kwargs = dict(base["kwargs"])
        if args.num_frames is not None:
            base_kwargs["num_frames"] = args.num_frames
        if args.steps is not None:
            base_kwargs["num_inference_steps"] = args.steps
        frames_values = (
            [int(x) for x in args.frames_list.split(",") if x.strip()]
            if args.frames_list
            else [base_kwargs["num_frames"]]
        )

        for frames in frames_values:
            kwargs_f = dict(base_kwargs)
            kwargs_f["num_frames"] = frames
            task = (
                f"T2V {kwargs_f['height']}x{kwargs_f['width']}, "
                f"{kwargs_f['num_frames']} frames, {kwargs_f['num_inference_steps']} steps"
            )
            frame_suffix = f"_f{frames}"

            # BF16 baseline once per frame count.
            print(f"\n=== {base['id']} BF16 frames={frames} ===", flush=True)
            os.environ["HV15_FP8_PRESET"] = "BF16"  # harmless on other models
            mem_bf16, t_bf16, v_bf16 = run(base["model"], None, kwargs_f, prompt=prompt)
            bf16_path = os.path.join(args.output_dir, f"{case_key}_bf16_seed{SEED}{frame_suffix}{suffix_tag}.mp4")
            save_video(v_bf16, bf16_path, fps=args.fps)
            print(f"  peak {mem_bf16:.2f} GiB, {t_bf16:.2f}s -> {bf16_path}", flush=True)

            # Sweep FP8 presets against the cached BF16 video.
            for preset in presets:
                if preset == "BF16":
                    continue
                print(f"\n=== {base['id']} FP8 preset={preset} frames={frames} ===", flush=True)
                os.environ["HV15_FP8_PRESET"] = preset

                # Per-tensor (string "fp8") vs block-wise (dict with weight_block_size).
                quant_spec: str | dict = "fp8"
                bs_tag = ""
                if weight_block_size is not None:
                    quant_spec = {"method": "fp8", "weight_block_size": weight_block_size}
                    bs_tag = f"_bs{weight_block_size[0]}x{weight_block_size[1]}"

                mem_fp8, t_fp8, v_fp8 = run(base["model"], quant_spec, kwargs_f, prompt=prompt)
                fp8_path = os.path.join(
                    args.output_dir,
                    f"{case_key}_fp8_{preset}{bs_tag}_seed{SEED}{frame_suffix}{suffix_tag}.mp4",
                )
                save_video(v_fp8, fp8_path, fps=args.fps)
                print(f"  peak {mem_fp8:.2f} GiB, {t_fp8:.2f}s -> {fp8_path}", flush=True)

                print("\n  computing LPIPS/PSNR/SSIM...", flush=True)
                try:
                    m = compute_metrics(v_bf16, v_fp8)
                    print(
                        f"  LPIPS={m['lpips']:.4f}  PSNR={m['psnr']:.2f}dB  SSIM={m['ssim']:.4f}",
                        flush=True,
                    )
                    quality_cols = f"{m['lpips']:.4f} | {m['psnr']:.2f} | {m['ssim']:.4f}"
                except Exception as e:
                    print(f"  metric computation failed: {e}", flush=True)
                    quality_cols = "n/a | n/a | n/a"

                mem_red = (1 - mem_fp8 / mem_bf16) if mem_bf16 > 0 else 0.0
                speedup = (1 - t_fp8 / t_bf16) if t_bf16 > 0 else 0.0
                preset_label = (
                    f"{preset}+bs{weight_block_size[0]}x{weight_block_size[1]}" if weight_block_size else preset
                )
                row = (
                    f"| {base['id']} | {task} | **{preset_label}** "
                    f"| {mem_bf16:.2f} GiB | {mem_fp8:.2f} GiB | {mem_red:.0%} "
                    f"| {t_bf16:.2f}s | {t_fp8:.2f}s | {speedup:.0%} "
                    f"| {quality_cols} |"
                )
                print(f"\n  row: {row}", flush=True)
                rows.append(row)

    header = [
        "\n## Benchmark (H100 80GB × 1)\n",
        f"Prompt: `{prompt}`\n",
        "| Model | Task | Preset | Mem BF16 | Mem FP8 | Mem Red | Time BF16 | Time FP8 | Speedup | LPIPS ↓ | PSNR ↑ | SSIM ↑ |",
        "|-------|------|--------|----------|---------|---------|-----------|----------|---------|---------|--------|--------|",
        *rows,
    ]
    md = "\n".join(header)
    print("\n" + md)

    results_path = os.path.join(args.output_dir, "results.md")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"\nResults saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
