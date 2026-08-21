# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare LTX-2.5 global SP with Stage-2 tiled data parallelism."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MODEL_CLASS = "LTX2DistilledTwoStagePipeline"
STAGE_2_SIGMAS = [0.625, 0.4, 0.0]
VAE_SPATIAL_COMPRESSION_RATIO = 32
TWO_STAGE_ALIGNMENT = 64

_TOTAL_TIME_RE = re.compile(r"Total generation time:\s*(?P<seconds>[0-9.]+)\s*seconds")
_PEAK_MEMORY_RE = re.compile(r"Worker peak GPU memory \(reserved\):\s*(?P<mib>[0-9.]+)\s*MiB")
_PROFILER_RE = re.compile(r"\[DiffusionPipelineProfiler\]\s+(?P<name>.+?)\s+took\s+(?P<seconds>[0-9.]+)s")
_SSIM_RE = re.compile(r"All:(?P<score>-?[0-9.]+)")
_PSNR_RE = re.compile(r"average:(?P<score>inf|[0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and compare matched-schedule LTX-2.5 global-SP and Stage-2 TDP videos."
    )
    parser.add_argument("--model", default="Lightricks/LTX-2.5-Diffusers")
    parser.add_argument("--devices", default="0,1,2,3", help="Exactly four comma-separated GPU indices.")
    parser.add_argument("--output-dir", default="/tmp/ltx25_tdp_ab")
    parser.add_argument(
        "--prompt",
        default="A cinematic aerial shot following a sailboat through a fjord at sunrise.",
    )
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlap", type=int, default=5, help="Stage-2 overlap in latent cells.")
    parser.add_argument(
        "--review-width",
        type=int,
        default=1920,
        help="Width of each half in the side-by-side review video.",
    )
    parser.add_argument("--analyze-only", action="store_true", help="Reuse existing outputs and logs.")
    parser.add_argument("--dry-run", action="store_true", help="Print generation commands without running them.")
    return parser.parse_args()


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def _split_dimension(size: int, tile_count: int, overlap: int) -> list[tuple[int, int]]:
    """Mirror the runtime's balanced overlapping interval split."""
    if tile_count < 1 or overlap < 0:
        raise ValueError("tile_count must be positive and overlap must be non-negative")
    if tile_count == 1:
        return [(0, size)]

    tile_size, remainder = divmod(size + overlap * (tile_count - 1), tile_count)
    if tile_size <= overlap:
        raise ValueError(f"Cannot split latent dimension {size} into {tile_count} tiles with overlap {overlap}.")

    intervals: list[tuple[int, int]] = []
    start = 0
    for index in range(tile_count):
        end = start + tile_size + int(index < remainder)
        intervals.append((start, end))
        start = end - overlap
    return intervals


def _overlap_bands(
    *,
    target_width: int,
    target_height: int,
    internal_width: int,
    internal_height: int,
    overlap: int,
) -> list[dict[str, int | str]]:
    latent_height = internal_height // VAE_SPATIAL_COMPRESSION_RATIO
    latent_width = internal_width // VAE_SPATIAL_COMPRESSION_RATIO
    height_intervals = _split_dimension(latent_height, 2, overlap)
    width_intervals = _split_dimension(latent_width, 2, overlap)

    bands: list[dict[str, int | str]] = []
    y_start = height_intervals[1][0] * VAE_SPATIAL_COMPRESSION_RATIO
    y_end = min(height_intervals[0][1] * VAE_SPATIAL_COMPRESSION_RATIO, target_height)
    if y_start < y_end:
        bands.append(
            {
                "axis": "horizontal",
                "x": 0,
                "y": y_start,
                "width": target_width,
                "height": y_end - y_start,
            }
        )

    x_start = width_intervals[1][0] * VAE_SPATIAL_COMPRESSION_RATIO
    x_end = min(width_intervals[0][1] * VAE_SPATIAL_COMPRESSION_RATIO, target_width)
    if x_start < x_end:
        bands.append(
            {
                "axis": "vertical",
                "x": x_start,
                "y": 0,
                "width": x_end - x_start,
                "height": target_height,
            }
        )
    return bands


def _generation_command(
    args: argparse.Namespace,
    *,
    output: Path,
    tiled: bool,
    internal_width: int,
    internal_height: int,
) -> list[str]:
    extra_body: dict[str, Any] = {"stage_2_sigmas": STAGE_2_SIGMAS}
    if tiled:
        extra_body.update(
            {
                "ltx_tiled_data_parallel": True,
                "ltx_tiled_data_parallel_overlap": args.overlap,
            }
        )

    command = [
        sys.executable,
        "examples/offline_inference/text_to_video/text_to_video.py",
        "--model",
        args.model,
        "--model-class-name",
        MODEL_CLASS,
        "--prompt",
        args.prompt,
        "--width",
        str(args.width if tiled else internal_width),
        "--height",
        str(args.height if tiled else internal_height),
        "--num-frames",
        str(args.num_frames),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--frame-rate",
        str(args.frame_rate),
        "--fps",
        str(args.fps),
        "--seed",
        str(args.seed),
        "--ulysses-degree",
        "4",
        "--enforce-eager",
        "--enable-diffusion-pipeline-profiler",
        "--extra-body",
        json.dumps(extra_body, separators=(",", ":")),
        "--output",
        str(output),
    ]
    if args.negative_prompt is not None:
        command.extend(["--negative-prompt", args.negative_prompt])
    return command


def _parse_run_log(log: str) -> dict[str, Any]:
    total_match = _TOTAL_TIME_RE.search(log)
    peak_match = _PEAK_MEMORY_RE.search(log)
    profiler_values: dict[str, list[float]] = {}
    for match in _PROFILER_RE.finditer(log):
        profiler_values.setdefault(match.group("name"), []).append(float(match.group("seconds")))

    profiler_summary = {
        name: {
            "samples": len(values),
            "min_seconds": min(values),
            "mean_seconds": statistics.mean(values),
            "max_seconds": max(values),
        }
        for name, values in profiler_values.items()
    }
    return {
        "request_seconds": float(total_match.group("seconds")) if total_match else None,
        "peak_memory_mib": float(peak_match.group("mib")) if peak_match else None,
        "profiler_events": profiler_summary,
    }


def _run_generation(command: list[str], *, repo_root: Path, devices: str, log_path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["PYTHONUNBUFFERED"] = "1"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    wall_seconds = time.perf_counter() - started
    if return_code != 0:
        raise RuntimeError(f"Generation failed with exit code {return_code}; see {log_path}.")

    result = _parse_run_log(log_path.read_text(encoding="utf-8"))
    result["process_wall_seconds"] = wall_seconds
    result["log"] = str(log_path)
    result["command"] = command
    return result


def _run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _probe_video(path: Path) -> dict[str, int | float]:
    result = _run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": int(numerator) / int(denominator),
        "frames": int(stream["nb_read_frames"]),
    }


def _similarity(
    baseline: Path,
    tiled: Path,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> dict[str, float]:
    crop = f"crop={width}:{height}:{x}:{y}"
    scores: dict[str, float] = {}
    for metric, pattern in (("ssim", _SSIM_RE), ("psnr", _PSNR_RE)):
        filter_graph = f"[0:v]{crop}[a];[1:v]{crop}[b];[a][b]{metric}[metric]"
        result = _run_checked(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-i",
                str(baseline),
                "-i",
                str(tiled),
                "-filter_complex",
                filter_graph,
                "-map",
                "[metric]",
                "-f",
                "null",
                "-",
            ]
        )
        match = pattern.search(result.stderr)
        if match is None:
            raise ValueError(f"Could not parse {metric.upper()} from ffmpeg output:\n{result.stderr}")
        score = match.group("score")
        scores[f"{metric}_mean" if metric == "ssim" else "psnr_mean_db"] = math.inf if score == "inf" else float(score)
    return scores


def _create_side_by_side(
    baseline: Path,
    tiled: Path,
    output: Path,
    *,
    target_width: int,
    target_height: int,
    review_width: int,
) -> None:
    if review_width < 2 or review_width % 2:
        raise ValueError("--review-width must be a positive even number.")
    common = f"crop={target_width}:{target_height}:0:0,scale={review_width}:-2"
    filter_graph = (
        f"[0:v]{common},drawbox=x=0:y=0:w=iw:h=60:color=black@0.65:t=fill,"
        "drawtext=text='Global SP (matched schedule)':x=24:y=16:fontsize=28:fontcolor=white[left];"
        f"[1:v]{common},drawbox=x=0:y=0:w=iw:h=60:color=black@0.65:t=fill,"
        "drawtext=text='Stage 2 TDP':x=24:y=16:fontsize=28:fontcolor=white[right];"
        "[left][right]hstack=inputs=2[review]"
    )
    _run_checked(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(baseline),
            "-i",
            str(tiled),
            "-filter_complex",
            filter_graph,
            "-map",
            "[review]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )


def _validate_inputs(args: argparse.Namespace) -> str:
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError("--devices must contain exactly four distinct GPU indices.")
    if args.width < 1 or args.height < 1 or args.overlap < 0:
        raise ValueError("Width and height must be positive; overlap must be non-negative.")
    if (args.num_frames - 1) % 8:
        raise ValueError("LTX-2.5 num_frames must be 8 * k + 1.")
    return ",".join(devices)


def main() -> None:
    args = parse_args()
    devices = _validate_inputs(args)
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise RuntimeError(f"{binary} is required for the LTX TDP A/B benchmark.")

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_video = output_dir / "global_sp_internal.mp4"
    tiled_video = output_dir / "stage2_tdp.mp4"
    side_by_side = output_dir / "side_by_side.mp4"
    baseline_log = output_dir / "global_sp.log"
    tiled_log = output_dir / "stage2_tdp.log"

    internal_width = _ceil_to_multiple(args.width, TWO_STAGE_ALIGNMENT)
    internal_height = _ceil_to_multiple(args.height, TWO_STAGE_ALIGNMENT)
    bands = _overlap_bands(
        target_width=args.width,
        target_height=args.height,
        internal_width=internal_width,
        internal_height=internal_height,
        overlap=args.overlap,
    )
    baseline_command = _generation_command(
        args,
        output=baseline_video,
        tiled=False,
        internal_width=internal_width,
        internal_height=internal_height,
    )
    tiled_command = _generation_command(
        args,
        output=tiled_video,
        tiled=True,
        internal_width=internal_width,
        internal_height=internal_height,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "CUDA_VISIBLE_DEVICES": devices,
                    "global_sp": baseline_command,
                    "stage2_tdp": tiled_command,
                    "internal_resolution": [internal_width, internal_height],
                    "target_resolution": [args.width, args.height],
                    "overlap_bands": bands,
                },
                indent=2,
            )
        )
        return

    if args.analyze_only:
        for path in (baseline_video, tiled_video):
            if not path.exists():
                raise FileNotFoundError(f"--analyze-only expected {path}.")
        baseline_run = _parse_run_log(baseline_log.read_text(encoding="utf-8") if baseline_log.exists() else "")
        tiled_run = _parse_run_log(tiled_log.read_text(encoding="utf-8") if tiled_log.exists() else "")
    else:
        print("\n=== Global-SP matched-schedule baseline ===")
        baseline_run = _run_generation(
            baseline_command,
            repo_root=repo_root,
            devices=devices,
            log_path=baseline_log,
        )
        print("\n=== Stage-2 tiled data parallel ===")
        tiled_run = _run_generation(
            tiled_command,
            repo_root=repo_root,
            devices=devices,
            log_path=tiled_log,
        )

    baseline_metadata = _probe_video(baseline_video)
    tiled_metadata = _probe_video(tiled_video)
    expected_baseline = {
        "width": internal_width,
        "height": internal_height,
        "fps": float(args.fps),
        "frames": args.num_frames,
    }
    expected_tiled = {
        "width": args.width,
        "height": args.height,
        "fps": float(args.fps),
        "frames": args.num_frames,
    }
    if baseline_metadata != expected_baseline:
        raise ValueError(f"Unexpected global-SP video metadata: {baseline_metadata}, expected {expected_baseline}.")
    if tiled_metadata != expected_tiled:
        raise ValueError(f"Unexpected TDP video metadata: {tiled_metadata}, expected {expected_tiled}.")

    full_frame = _similarity(
        baseline_video,
        tiled_video,
        x=0,
        y=0,
        width=args.width,
        height=args.height,
    )
    seam_metrics = []
    for band in bands:
        metrics = _similarity(
            baseline_video,
            tiled_video,
            x=int(band["x"]),
            y=int(band["y"]),
            width=int(band["width"]),
            height=int(band["height"]),
        )
        seam_metrics.append({**band, **metrics})

    _create_side_by_side(
        baseline_video,
        tiled_video,
        side_by_side,
        target_width=args.width,
        target_height=args.height,
        review_width=args.review_width,
    )

    baseline_seconds = baseline_run.get("request_seconds")
    tiled_seconds = tiled_run.get("request_seconds")
    speedup = baseline_seconds / tiled_seconds if baseline_seconds and tiled_seconds else None
    result = {
        "comparison": "global_sp_matched_schedule_vs_stage2_tdp",
        "controlled_inputs": {
            "model": args.model,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "target_width": args.width,
            "target_height": args.height,
            "internal_width": internal_width,
            "internal_height": internal_height,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "num_inference_steps": args.num_inference_steps,
            "stage_2_sigmas": STAGE_2_SIGMAS,
            "gpus": devices,
        },
        "geometry": {"grid": [2, 2], "latent_overlap": args.overlap, "pixel_overlap_bands": bands},
        "global_sp": {"video": str(baseline_video), "metadata": baseline_metadata, **baseline_run},
        "stage2_tdp": {"video": str(tiled_video), "metadata": tiled_metadata, **tiled_run},
        "performance": {"request_speedup": speedup},
        "video_similarity": {"full_frame": full_frame, "overlap_bands": seam_metrics},
        "side_by_side": str(side_by_side),
        "notes": [
            "The baseline uses global Ulysses SP in both stages and the same two-step Stage-2 sigma schedule as TDP.",
            "The baseline generates the aligned internal resolution; metrics crop it to the requested "
            "target resolution.",
            "Video metrics exclude audio. TDP intentionally returns full-context Stage-1 audio.",
            "TDP is approximate local attention, so SSIM/PSNR are diagnostics rather than bit-parity gates.",
        ],
    }
    result_path = output_dir / "comparison.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))
    print(f"\nSaved comparison report to {result_path}")


if __name__ == "__main__":
    main()
