# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LingBot dense parity and timing harness.")
    parser.add_argument("--model", default="/home/models/lingbot-video-dense-1.3b")
    parser.add_argument("--official-repo", default="/tmp/lingbot-video")
    parser.add_argument("--output-dir", default="/tmp/lingbot_dense_parity")
    parser.add_argument("--prompt", default="a robotic arm picks up a red block")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--runs", type=int, default=0, help="Optional in-process steady-state native runs.")
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[float, str]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        print(proc.stdout)
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout)
    return elapsed, proc.stdout


def _read_video(path: Path) -> np.ndarray:
    frames = iio.imread(path, plugin="pyav")
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32) / 255.0
    return frames


def _compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "shape_match": False}
    diff = a - b
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse == 0 else float(20.0 * np.log10(1.0 / np.sqrt(mse)))
    return {
        "shape": list(a.shape),
        "shape_match": True,
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
    }


def _extract_total_generation_seconds(log: str) -> float | None:
    match = re.search(r"Total generation time:\s*([0-9.]+)\s*seconds", log)
    return float(match.group(1)) if match else None


def _peak_memory_mb(output: Any) -> float:
    result = output[0] if isinstance(output, list) and output else output
    return float(getattr(result, "peak_memory_mb", 0.0) or 0.0)


def _benchmark_native_requests(args: argparse.Namespace) -> dict[str, Any]:
    load_start = time.perf_counter()
    omni = Omni(
        model=args.model,
        model_class_name="LingBotVideoPipeline",
        flow_shift=args.shift,
        parallel_config=DiffusionParallelConfig(),
    )
    load_seconds = time.perf_counter() - load_start
    prompt = {"prompt": args.prompt}
    latencies: list[float] = []
    peak_mb = 0.0

    for _ in range(args.runs):
        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
        sampling_params = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type="pt",
            extra_args={"shift": args.shift},
        )
        start = time.perf_counter()
        output = omni.generate(prompt, sampling_params)
        latencies.append(time.perf_counter() - start)
        peak_mb = max(peak_mb, _peak_memory_mb(output))
        del output
    omni.close()

    return {
        "load_seconds": load_seconds,
        "run_seconds": latencies,
        "mean_seconds": statistics.mean(latencies) if latencies else None,
        "median_seconds": statistics.median(latencies) if latencies else None,
        "min_seconds": min(latencies) if latencies else None,
        "max_seconds": max(latencies) if latencies else None,
        "peak_mb": peak_mb,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    official_out = output_dir / "official_diffusers.mp4"
    native_out = output_dir / "vllm_omni_native.mp4"

    official_cmd = [
        sys.executable,
        str(Path(args.official_repo) / "scripts" / "inference.py"),
        "--backend",
        "diffusers",
        "--model_dir",
        args.model,
        "--mode",
        "t2v",
        "--prompt",
        args.prompt,
        "--output",
        str(official_out),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_frames",
        str(args.num_frames),
        "--steps",
        str(args.steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--shift",
        str(args.shift),
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
        "--quiet_progress",
    ]
    native_cmd = [
        sys.executable,
        "examples/offline_inference/text_to_video/text_to_video_lingbot.py",
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--output",
        str(native_out),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num-frames",
        str(args.num_frames),
        "--num-inference-steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--flow-shift",
        str(args.shift),
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
    ]

    official_time, official_log = _run(official_cmd, cwd=Path(args.official_repo))
    native_time, native_log = _run(native_cmd, cwd=Path.cwd())

    result = {
        "official_seconds": official_time,
        "native_process_seconds": native_time,
        "native_request_seconds": _extract_total_generation_seconds(native_log),
        "official_output": str(official_out),
        "native_output": str(native_out),
        "accuracy": _compare(_read_video(official_out), _read_video(native_out)),
        "official_log_tail": official_log.splitlines()[-20:],
        "native_log_tail": native_log.splitlines()[-20:],
    }
    if args.runs > 0:
        result["native_steady_state"] = _benchmark_native_requests(args)

    result_path = output_dir / "comparison.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
