# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark quantization quality loss for diffusion models (image & video).

Generates outputs with BF16 (baseline) and a quantized config using the same
seed, then computes LPIPS perceptual distance between them. Results are printed
as a Markdown table ready to paste into a PR description.

Requirements:
    pip install lpips Pillow numpy

Image example (text-to-image):
    python benchmarks/diffusion/quantization_quality.py \
        --model Tongyi-MAI/Z-Image-Turbo \
        --task t2i \
        --quantization fp8 \
        --prompts \
            "an aerial view of a coral reef with crystal clear turquoise water" \
            "a campfire in a dark forest with sparks rising into a starry sky" \
            "a gourmet dessert plate with chocolate mousse and gold leaf" \
        --height 1024 --width 1024 \
        --num-inference-steps 50 --seed 42

Video example (text-to-video):
    python benchmarks/diffusion/quantization_quality.py \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --task t2v \
        --quantization fp8 \
        --prompts \
            "A serene lakeside sunrise with mist over the water" \
            "A cat walking across a wooden bridge in autumn" \
        --height 720 --width 1280 \
        --num-frames 81 --num-inference-steps 40 --seed 42

LTX-2 example (text-to-video; audio output is generated but not scored by LPIPS):
    python benchmarks/diffusion/quantization_quality.py \
        --model Lightricks/LTX-Video-2 \
        --task t2v \
        --quantization fp8 int8 \
        --prompts \
            "A serene lakeside sunrise with mist over the water" \
        --height 704 --width 1216 \
        --num-frames 121 --num-inference-steps 40 --seed 42

MagiHuman example (text-to-video+audio; LPIPS scores video frames only):
    python benchmarks/diffusion/quantization_quality.py \
        --model SII-GAIR/daVinci-MagiHuman-Base-1080p \
        --model-type magi-human \
        --task t2v \
        --quantization fp8 \
        --prompts "A woman speaks to the camera in a sunlit garden." \
        --height 256 --width 448 \
        --seconds 5 --num-inference-steps 8 --seed 52

Multiple quantization methods:
    python benchmarks/diffusion/quantization_quality.py \
        --model Tongyi-MAI/Z-Image-Turbo \
        --task t2i \
        --quantization fp8 int8 bitsandbytes \
        --prompts "a cup of coffee on the table" \
        --height 1024 --width 1024 \
        --num-inference-steps 50 --seed 42

Output directory structure (--output-dir, default: ./quant_bench_output):
    quant_bench_output/
        baseline/           # BF16 outputs
        <method>/           # Quantized outputs per method
        results.md          # Markdown table
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch


def _extract_peak_memory_gib(output, fallback_gib: float) -> float:
    """Return worker-reported peak memory when the engine runs out of process."""
    candidates = [output]
    request_output = getattr(output, "request_output", None)
    if isinstance(request_output, list):
        candidates.extend(request_output)
    elif request_output is not None:
        candidates.append(request_output)

    for candidate in candidates:
        peak_memory_mb = getattr(candidate, "peak_memory_mb", None)
        if peak_memory_mb:
            return float(peak_memory_mb) / 1024.0
    return fallback_gib


def compute_lpips_images(
    baseline_images: list,
    quantized_images: list,
    net: str = "alex",
) -> list[float]:
    """Compute LPIPS between paired lists of PIL images."""
    import lpips
    from torchvision import transforms

    loss_fn = lpips.LPIPS(net=net).eval()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    scores = []
    for img_bl, img_qt in zip(baseline_images, quantized_images):
        t_bl = transform(img_bl.convert("RGB")).unsqueeze(0)
        t_qt = transform(img_qt.convert("RGB")).unsqueeze(0)
        if torch.cuda.is_available():
            t_bl, t_qt = t_bl.cuda(), t_qt.cuda()
        with torch.no_grad():
            score = loss_fn(t_bl, t_qt).item()
        scores.append(score)
    return scores


def compute_lpips_video(
    baseline_frames: np.ndarray,
    quantized_frames: np.ndarray,
    net: str = "alex",
) -> float:
    """Compute mean per-frame LPIPS for a video pair.

    Args:
        baseline_frames: (F, H, W, C) float array in [0, 1].
        quantized_frames: same shape.

    Returns:
        Mean LPIPS across all frames.
    """
    import lpips

    loss_fn = lpips.LPIPS(net=net).eval()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    num_frames = min(len(baseline_frames), len(quantized_frames))
    scores = []
    for i in range(num_frames):
        # Convert (H, W, C) float [0,1] -> (1, C, H, W) float [-1, 1]
        f_bl = torch.from_numpy(baseline_frames[i]).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        f_qt = torch.from_numpy(quantized_frames[i]).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        if torch.cuda.is_available():
            f_bl, f_qt = f_bl.cuda(), f_qt.cuda()
        with torch.no_grad():
            score = loss_fn(f_bl, f_qt).item()
        scores.append(score)
    return float(np.mean(scores))


def _build_omni_kwargs(args, quantization=None):
    """Build kwargs dict for Omni() constructor."""
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    kwargs = {
        "model": args.model,
        "parallel_config": parallel_config,
        "enforce_eager": args.enforce_eager,
    }
    if args.model_type is not None:
        kwargs["model_type"] = args.model_type
    if quantization:
        kwargs["quantization"] = quantization
    return kwargs


def _output_fps(args) -> int:
    """Return the native FPS when the model type provides one."""
    if args.fps is not None:
        return args.fps
    if args.model_type == "magi-human":
        return 25
    return 24


def _prepare_video_frames_for_export(frames):
    """Normalize integer NumPy video frames for diffusers' float-only exporter."""
    array = np.asarray(frames)
    if array.ndim == 4 and np.issubdtype(array.dtype, np.integer):
        return list(array.astype(np.float32) / np.iinfo(array.dtype).max)
    return list(array) if isinstance(frames, np.ndarray) and array.ndim == 4 else frames


def _save_video(frames, output_path: Path, fps: int) -> None:
    try:
        from diffusers.utils import export_to_video

        export_to_video(_prepare_video_frames_for_export(frames), str(output_path), fps=fps)
    except ImportError:
        np.save(output_path.with_suffix(".npy"), frames)


def _generate_image(omni, args, prompt, seed):
    """Generate a single image and return (PIL.Image, time_seconds, memory_gib)."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
    torch.accelerator.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs = omni.generate(
        {"prompt": prompt},
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
        ),
    )
    elapsed = time.perf_counter() - start
    peak_mem = torch.accelerator.max_memory_allocated() / (1024**3)

    req_out = OmniRequestOutput.unwrap_result(outputs)
    if not req_out.images:
        raise ValueError("Could not extract image output from result.")
    img = req_out.images[0]
    return img, elapsed, peak_mem


def _generate_video(omni, args, prompt, seed):
    """Generate a video and return (np.ndarray [F,H,W,C], time_seconds, memory_gib)."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(seed)
    torch.accelerator.reset_peak_memory_stats()
    start = time.perf_counter()
    extra_args = {}
    if args.seconds is not None:
        extra_args["seconds"] = args.seconds
    if args.sr_height is not None:
        extra_args["sr_height"] = args.sr_height
    if args.sr_width is not None:
        extra_args["sr_width"] = args.sr_width
    if args.sr_num_inference_steps is not None:
        extra_args["sr_num_inference_steps"] = args.sr_num_inference_steps

    outputs = omni.generate(
        {"prompt": prompt, "negative_prompt": ""},
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            seed=seed,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            extra_args=extra_args,
        ),
    )
    elapsed = time.perf_counter() - start
    peak_mem = torch.accelerator.max_memory_allocated() / (1024**3)

    first = outputs[0]
    if hasattr(first, "request_output") and isinstance(first.request_output, list):
        inner = first.request_output[0]
        if isinstance(inner, OmniRequestOutput) and hasattr(inner, "images"):
            frames = inner.images[0] if inner.images else None
        else:
            frames = inner
    elif hasattr(first, "images") and first.images:
        frames = first.images[0]
    else:
        raise ValueError("Could not extract video frames from output.")

    # LTX-2 (and similar audio+video models) may surface a dict or (video, audio) tuple
    if isinstance(frames, dict):
        frames = frames.get("video") or frames.get("frames")
    elif isinstance(frames, tuple) and len(frames) == 2:
        frames = frames[0]

    if frames is None:
        raise ValueError("Could not extract video frames from output.")

    if isinstance(frames, torch.Tensor):
        video = frames.detach().cpu()
        if video.dim() == 5:
            video = video[0].permute(1, 2, 3, 0) if video.shape[1] in (3, 4) else video[0]
        elif video.dim() == 4 and video.shape[0] in (3, 4):
            video = video.permute(1, 2, 3, 0)
        if video.is_floating_point():
            video = video.clamp(-1, 1) * 0.5 + 0.5
        frames_array = video.float().numpy()
    else:
        frames_array = np.asarray(frames)
        if frames_array.ndim == 5:
            frames_array = frames_array[0]

    return frames_array, elapsed, _extract_peak_memory_gib(first, peak_mem)


def _free_gpu_memory():
    """Force GC and release cached GPU memory.

    Must be called AFTER the caller has dropped (i.e., via `del`)
    every reference to the Omni instance"""
    gc.collect()
    if torch.cuda.is_available():
        torch.accelerator.empty_cache()
        torch.accelerator.synchronize()


def run_benchmark(args):
    from vllm_omni.entrypoints.omni import Omni

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_video = args.task == "t2v"
    prompts = args.prompts
    seed = args.seed

    # Determine configs to benchmark
    configs = []  # list of (label, quantization_method)
    for method in args.quantization:
        configs.append((method, method))

    # --- Baseline run ---
    print("\n" + "=" * 60)
    print("Running BF16 baseline...")
    print("=" * 60)
    bl_kwargs = _build_omni_kwargs(args, quantization=None)
    omni_bl = Omni(**bl_kwargs)

    baseline_outputs = {}  # prompt -> (output, time, mem)
    for prompt in prompts:
        print(f"  Generating: {prompt[:60]}...")
        if is_video:
            out, t, mem = _generate_video(omni_bl, args, prompt, seed)
        else:
            out, t, mem = _generate_image(omni_bl, args, prompt, seed)
        baseline_outputs[prompt] = (out, t, mem)

    bl_avg_time = np.mean([v[1] for v in baseline_outputs.values()])
    bl_mem = baseline_outputs[prompts[0]][2]  # use first prompt's memory
    video_frame_count = len(baseline_outputs[prompts[0]][0]) if is_video else None
    omni_bl.shutdown()
    del omni_bl
    _free_gpu_memory()

    # Save baseline outputs
    bl_dir = output_dir / "baseline"
    bl_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(prompts):
        out = baseline_outputs[prompt][0]
        if is_video:
            _save_video(out, bl_dir / f"prompt_{i}.mp4", fps=_output_fps(args))
        else:
            out.save(bl_dir / f"prompt_{i}.png")

    # --- Quantized runs ---
    all_results = []  # list of dicts

    for config_label, quant_method in configs:
        print(f"\n{'=' * 60}")
        print(f"Running: {config_label}...")
        print("=" * 60)

        qt_kwargs = _build_omni_kwargs(args, quantization=quant_method)
        omni_qt = Omni(**qt_kwargs)

        qt_outputs = {}
        for prompt in prompts:
            print(f"  Generating: {prompt[:60]}...")
            if is_video:
                out, t, mem = _generate_video(omni_qt, args, prompt, seed)
            else:
                out, t, mem = _generate_image(omni_qt, args, prompt, seed)
            qt_outputs[prompt] = (out, t, mem)

        qt_avg_time = np.mean([v[1] for v in qt_outputs.values()])
        qt_mem = qt_outputs[prompts[0]][2]
        omni_qt.shutdown()
        del omni_qt
        _free_gpu_memory()

        # Save quantized outputs
        qt_dir = output_dir / config_label.replace(" ", "_")
        qt_dir.mkdir(parents=True, exist_ok=True)

        # Compute LPIPS per prompt
        per_prompt = []
        for i, prompt in enumerate(prompts):
            bl_out = baseline_outputs[prompt][0]
            qt_out = qt_outputs[prompt][0]
            if is_video:
                lpips_score = compute_lpips_video(bl_out, qt_out, net=args.lpips_net)
                _save_video(qt_out, qt_dir / f"prompt_{i}.mp4", fps=_output_fps(args))
            else:
                lpips_score = compute_lpips_images([bl_out], [qt_out], net=args.lpips_net)[0]
                qt_out.save(qt_dir / f"prompt_{i}.png")
            per_prompt.append({"prompt": prompt, "lpips": lpips_score})

        mean_lpips = np.mean([p["lpips"] for p in per_prompt])
        speedup = bl_avg_time / qt_avg_time if qt_avg_time > 0 else float("inf")
        mem_reduction = (bl_mem - qt_mem) / bl_mem * 100

        all_results.append(
            {
                "config": config_label,
                "avg_time": qt_avg_time,
                "speedup": speedup,
                "memory_gib": qt_mem,
                "mem_reduction_pct": mem_reduction,
                "mean_lpips": mean_lpips,
                "per_prompt": per_prompt,
            }
        )

    # --- Print results ---
    print("\n\n")
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Summary table
    lines = []
    lines.append(f"## Quantization Quality Benchmark — {args.model.split('/')[-1]}")
    lines.append(
        f"Setup: {args.height}x{args.width}, {args.num_inference_steps} steps, "
        f"seed={args.seed}, LPIPS ({args.lpips_net})"
    )
    if is_video:
        lines.append(f"Video: {video_frame_count} frames")
    lines.append("")
    lines.append("### Summary")
    lines.append("")
    lines.append("| Config | Avg Time | Speedup | Memory (GiB) | Mem Reduction | Mean LPIPS |")
    lines.append("|--------|----------|---------|--------------|---------------|------------|")
    lines.append(f"| BF16 baseline | {bl_avg_time:.2f}s | 1.00x | {bl_mem:.2f} | — | (ref) |")
    for r in all_results:
        lines.append(
            f"| {r['config']} | {r['avg_time']:.2f}s | {r['speedup']:.2f}x "
            f"| {r['memory_gib']:.2f} | {r['mem_reduction_pct']:.0f}% "
            f"| {r['mean_lpips']:.4f} |"
        )
    lines.append("")
    lines.append("> LPIPS < 0.01 = imperceptible, > 0.1 = clearly noticeable.")
    lines.append("")

    # Per-prompt table
    if len(prompts) > 1:
        lines.append("### Per-Prompt LPIPS")
        lines.append("")
        header = "| Prompt |"
        sep = "|--------|"
        for r in all_results:
            header += f" {r['config']} |"
            sep += "--------|"
        lines.append(header)
        lines.append(sep)
        for i, prompt in enumerate(prompts):
            short = prompt[:50] + "..." if len(prompt) > 50 else prompt
            row = f"| {short} |"
            for r in all_results:
                row += f" {r['per_prompt'][i]['lpips']:.4f} |"
            lines.append(row)
        lines.append("")

    md = "\n".join(lines)
    print(md)

    # Save markdown
    results_path = output_dir / "results.md"
    results_path.write_text(md, encoding="utf-8")
    print(f"\nResults saved to {results_path}")
    print(f"Baseline outputs in {bl_dir}")
    for r in all_results:
        qt_dir = output_dir / r["config"].replace(" ", "_")
        print(f"Quantized outputs in {qt_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark quantization quality loss for diffusion models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model name or local path.")
    parser.add_argument(
        "--model-type",
        default=None,
        help="Optional vLLM-Omni model type override, for example magi-human.",
    )
    parser.add_argument(
        "--task",
        default="t2i",
        choices=["t2i", "t2v"],
        help="Task type: t2i (text-to-image) or t2v (text-to-video).",
    )
    parser.add_argument(
        "--quantization",
        nargs="+",
        required=True,
        help="One or more quantization methods to benchmark (e.g. fp8 int8 bitsandbytes).",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["a cup of coffee on the table"],
        help="One or more prompts to generate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=81, help="Number of video frames (t2v only).")
    parser.add_argument("--seconds", type=int, default=None, help="MagiHuman output duration in seconds.")
    parser.add_argument("--sr-height", type=int, default=None, help="MagiHuman SR output height.")
    parser.add_argument("--sr-width", type=int, default=None, help="MagiHuman SR output width.")
    parser.add_argument(
        "--sr-num-inference-steps",
        type=int,
        default=None,
        help="MagiHuman SR denoising steps.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Video FPS for saving (defaults to 25 for MagiHuman and 24 otherwise).",
    )
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG scale (used for video).")
    parser.add_argument("--output-dir", type=str, default="./quant_bench_output", help="Directory to save outputs.")
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone network.",
    )
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
