# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from vllm_omni.outputs import OmniRequestOutput


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure one complete Omni video request at a time.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument(
        "--revision",
        help="Model revision passed to Omni; use a pinned local snapshot for local weights.",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=positive_int,
        help="Override PP; otherwise use the deploy config.",
    )
    parser.add_argument(
        "--stage-overrides",
        type=json.loads,
        help="Omni's JSON stage_id -> overrides mapping.",
    )
    parser.add_argument("--frames", type=positive_int, required=True)
    parser.add_argument("--chunks", type=positive_int, required=True)
    parser.add_argument(
        "--fps",
        type=positive_float,
        required=True,
        help="Playback FPS, distinct from generation throughput.",
    )
    parser.add_argument("--steps", type=positive_int, default=4)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--width", type=positive_int, default=832)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=positive_float, default=1.0)
    parser.add_argument("--cell-schedule", choices=("self_forcing", "latest_kv", "same_step"), default="same_step")
    parser.add_argument("--kv-history-chunks", type=nonnegative_int, default=18)
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def video_frame_count(shape: tuple[int, ...]) -> int:
    # The np video postprocessor returns [batch, frames, height, width, channels].
    if len(shape) != 5 or shape[0] != 1 or shape[-1] != 3 or any(dimension < 1 for dimension in shape):
        raise ValueError(f"Expected one RGB video [1, F, H, W, 3], received {shape}.")
    return shape[1]


def video_payload(
    outputs: list[OmniRequestOutput],
) -> tuple[OmniRequestOutput, np.ndarray]:
    import numpy as np

    if len(outputs) != 1:
        raise ValueError(f"Expected one request output, received {len(outputs)}.")
    output = outputs[0]
    if output.error:
        raise RuntimeError(output.error)
    if not output.finished or output.final_output_type not in {"image", "video"} or len(output.images) != 1:
        raise ValueError("Expected one finished video in OmniRequestOutput.images.")
    video = output.images[0]
    # The legacy formatter retains the video mapping; the normalized formatter unwraps it.
    if isinstance(video, dict):
        video = video["video"]
    if not isinstance(video, np.ndarray):
        raise TypeError(f"Expected np video output, received {type(video).__name__}.")
    return output, video


def request_metrics(frames: int, elapsed_seconds: float, stage_durations: dict[str, float] | None) -> dict:
    stage_durations = stage_durations or {}
    physis_seconds = {
        key: value for key, value in stage_durations.items() if key == "physis.dit" or key.startswith("physis.chunk_")
    }
    dit_seconds = physis_seconds.get("physis.dit")
    return {
        "frames": frames,
        "e2e_seconds": elapsed_seconds,
        "e2e_ms": elapsed_seconds * 1000.0,
        "e2e_fps": frames / elapsed_seconds,
        "stage_durations": dict(stage_durations),
        "physis_stage_durations_seconds": physis_seconds,
        "dit_ms": dit_seconds * 1000.0 if dit_seconds is not None else None,
        "dit_fps": frames / dit_seconds if dit_seconds is not None and dit_seconds > 0 else None,
    }


def summarize(runs: list[dict]) -> dict:
    elapsed = [run["e2e_seconds"] for run in runs]
    throughput = [run["e2e_fps"] for run in runs]
    result = {
        "requests": len(runs),
        "total_frames": sum(run["frames"] for run in runs),
        "e2e_seconds_mean": statistics.mean(elapsed),
        "e2e_seconds_median": statistics.median(elapsed),
        "e2e_fps_median": statistics.median(throughput),
        "e2e_fps_aggregate": sum(run["frames"] for run in runs) / sum(elapsed),
    }
    dit_fps = [run["dit_fps"] for run in runs if run["dit_fps"] is not None]
    if len(dit_fps) == len(runs):
        result["dit_fps_median"] = statistics.median(dit_fps)
    return result


def source_metadata(root: Path) -> dict:
    probe = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if probe.returncode != 0:
        return {"path": str(root), "head": None, "working_tree_status": None}
    root = Path(probe.stdout.strip())

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    return {
        "path": str(root),
        "head": git("rev-parse", "HEAD"),
        "working_tree_status": git("status", "--porcelain").splitlines(),
    }


def main() -> None:
    args = parse_args()

    import torch

    import vllm_omni
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    init_kwargs: dict[str, object] = {
        "model_class_name": args.class_name,
        "deploy_config": args.deploy_config,
    }
    if args.revision is not None:
        init_kwargs["revision"] = args.revision
    if args.pipeline_parallel_size is not None:
        init_kwargs["pipeline_parallel_size"] = args.pipeline_parallel_size
    if args.stage_overrides is not None:
        init_kwargs["stage_overrides"] = args.stage_overrides

    extra_args = {
        "num_chunks": args.chunks,
        "num_denoise_steps": args.steps,
        "cell_schedule": args.cell_schedule,
        "kv_history_chunks": args.kv_history_chunks,
        "collect_chunk_timing": True,
    }

    def sampling_params() -> OmniDiffusionSamplingParams:
        return OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_type="np",
            extra_args=dict(extra_args),
        )

    prompt = {"prompt": args.prompt, "negative_prompt": args.negative_prompt}
    source = {
        "benchmark": source_metadata(Path(__file__).resolve().parents[2]),
        "runtime": source_metadata(Path(vllm_omni.__file__).resolve().parent),
        "runtime_package_file": str(Path(vllm_omni.__file__).resolve()),
    }
    omni = Omni(model=args.model, **init_kwargs)
    try:
        model_path = Path(omni.model).expanduser()
        resolved_model = str(model_path.resolve()) if model_path.exists() else omni.model
        runs = []
        for index in range(args.warmup + args.repeats):
            params = sampling_params()
            started = time.perf_counter()
            outputs = omni.generate(prompt, params, use_tqdm=False)
            elapsed = time.perf_counter() - started
            output, video = video_payload(outputs)
            frames = video_frame_count(video.shape)
            if video.shape != (1, args.frames, args.height, args.width, 3):
                raise ValueError(
                    f"Expected [1, {args.frames}, {args.height}, {args.width}, 3], received {video.shape}."
                )
            if index >= args.warmup:
                runs.append(
                    {
                        "repeat": index - args.warmup + 1,
                        **request_metrics(frames, elapsed, output.stage_durations),
                        "worker_peak_memory_mb": output.peak_memory_mb,
                    }
                )
            del outputs, output, video

        result = {
            "status": "complete",
            "benchmark": {
                "model": args.model,
                "resolved_model": resolved_model,
                "model_revision": args.revision,
                "class_name": args.class_name,
                "deploy_config": args.deploy_config,
                "pipeline_parallel_size_override": args.pipeline_parallel_size,
                "stage_overrides": args.stage_overrides,
                "source": source,
                "device_type": current_omni_platform.device_type,
                "device_name": current_omni_platform.get_device_name(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_version": torch.__version__,
                "frames": args.frames,
                "chunks": args.chunks,
                "playback_fps": args.fps,
                "steps": args.steps,
                "height": args.height,
                "width": args.width,
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "seed": args.seed,
                "guidance_scale": args.guidance_scale,
                "extra_args": extra_args,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "timing_scope": "Omni.generate wall clock: request text encoding, DiT, VAE, postprocess and return; "
                "excludes initialization, warmup, sampling-parameter creation and file writing.",
            },
            "runs": runs,
            "summary": summarize(runs),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output_json": str(args.output_json), **result["summary"]}))
    finally:
        omni.close()


if __name__ == "__main__":
    main()
