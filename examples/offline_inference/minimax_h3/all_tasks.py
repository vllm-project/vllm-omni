# SPDX-License-Identifier: Apache-2.0
"""Run MiniMax-H3 tasks with a configurable distributed topology.

The shell wrapper launches this file once per checkpoint partition. Keeping
FL2VA and Ref2VA in separate processes avoids reinitializing distributed
process groups after an ``Omni`` engine has been closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

ASPECT_RATIOS = {
    "21:9": 21 / 9,
    "16:9": 16 / 9,
    "4:3": 4 / 3,
    "1:1": 1.0,
    "3:4": 3 / 4,
    "9:16": 9 / 16,
}

TASK_IDS = (
    "t2va",
    "fl2va_first_frame",
    "ref2va_image_audio",
    "ref2va_two_videos",
)

DEFAULT_PROMPTS = {
    "t2va": (
        "At night, three cats march into a bedroom playing tiny brass "
        "instruments, then abruptly file out, with synchronized room ambience."
    ),
    "fl2va_first_frame": (
        "Continue naturally from the supplied first frame. The cats march "
        "forward while playing, with coherent motion and synchronized sound."
    ),
    "ref2va_image_audio": (
        "Use Picture 1 as the visual subject and Audio 1 as the sound reference. "
        "Create coherent natural motion synchronized to the complete audio."
    ),
    "ref2va_two_videos": (
        "Combine the subjects and motion of Video 1 with the continuation in "
        "Video 2, preserving coherent timing and synchronized sound."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("fl2va", "ref2va"), required=True)
    parser.add_argument("--expect-ref2va", action="store_true")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=1101)
    parser.add_argument("--num-gpus", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--text-encoder-tp-size", type=int, required=True)
    parser.add_argument("--vae-patch-parallel-size", type=int, required=True)
    parser.add_argument("--enable-distributed-layerwise-offload", action="store_true")
    parser.add_argument("--dlo-resident-layers", type=int, default=0)
    parser.add_argument("--profiler-dir", type=Path)
    parser.add_argument("--attention-backend", default="CUDNN_ATTN")
    parser.add_argument("--fp8-q-scale", type=float)
    parser.add_argument("--fp8-k-scale", type=float)
    parser.add_argument("--fp8-v-scale", type=float)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_metadata(expected_gpus: int) -> list[dict[str, object]]:
    hardware = []
    for index in range(torch.accelerator.device_count()):
        properties = torch.cuda.get_device_properties(index)
        hardware.append(
            {
                "logical_index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_gib": round(properties.total_memory / 2**30, 2),
            }
        )
    if len(hardware) != expected_gpus:
        raise RuntimeError(f"Expected {expected_gpus} visible GPUs, found {len(hardware)}")
    return hardware


def validate_parallel_args(args: argparse.Namespace) -> None:
    expected_gpus = args.tensor_parallel_size * args.ulysses_degree * args.ring_degree
    if expected_gpus != args.num_gpus:
        raise ValueError(
            "num_gpus must equal tensor_parallel_size * ulysses_degree * "
            f"ring_degree, got {args.num_gpus} != {expected_gpus}"
        )
    if 56 % args.tensor_parallel_size or 14336 % args.tensor_parallel_size:
        raise ValueError(
            "MiniMax-H3 tensor_parallel_size must divide both 56 attention "
            f"heads and the 14336-wide FFN, got {args.tensor_parallel_size}"
        )
    local_heads = 56 // args.tensor_parallel_size
    if local_heads % args.ulysses_degree:
        raise ValueError(
            "MiniMax-H3 local attention heads must be divisible by "
            f"ulysses_degree, got {local_heads} % {args.ulysses_degree} != 0"
        )
    if args.text_encoder_tp_size > args.num_gpus:
        raise ValueError("text_encoder_tp_size must not exceed num_gpus")
    if 64 % args.text_encoder_tp_size or 8 % args.text_encoder_tp_size:
        raise ValueError(
            "text_encoder_tp_size must divide the Qwen3-VL 64 query heads and "
            f"8 KV heads, got {args.text_encoder_tp_size}"
        )
    if not 1 <= args.vae_patch_parallel_size <= args.num_gpus:
        raise ValueError("vae_patch_parallel_size must be in [1, num_gpus]")
    if not 0 <= args.dlo_resident_layers <= 50:
        raise ValueError("dlo_resident_layers must be in [0, 50]")
    if args.dlo_resident_layers and not args.enable_distributed_layerwise_offload:
        raise ValueError("dlo_resident_layers requires distributed layerwise offload")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    for name in ("fp8_q_scale", "fp8_k_scale", "fp8_v_scale"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be > 0")
    has_fp8_scale = any(getattr(args, name) is not None for name in ("fp8_q_scale", "fp8_k_scale", "fp8_v_scale"))
    if has_fp8_scale and args.attention_backend.upper() != "FLASHINFER_SM120_ATTN":
        raise ValueError("FP8 attention scales require --attention-backend FLASHINFER_SM120_ATTN")


def attention_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    backend = args.attention_backend.upper()
    if backend != "FLASHINFER_SM120_ATTN":
        return {"diffusion_attention_backend": backend}

    quant: dict[str, Any] = {
        "dtype_qk": "fp8_e4m3",
        "flashinfer_backend": "cute-dsl-prims",
    }
    for arg_name, config_name in (
        ("fp8_q_scale", "q_scale"),
        ("fp8_k_scale", "k_scale"),
        ("fp8_v_scale", "v_scale"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            quant[config_name] = value
    return {"diffusion_attention_config": {"default": {"backend": backend, "quant": quant}}}


def make_engine(model_dir: Path, args: argparse.Namespace) -> Omni:
    profiler_config = None
    if args.profiler_dir is not None:
        args.profiler_dir.mkdir(parents=True, exist_ok=True)
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profiler_dir),
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_with_stack": False,
            "torch_profiler_dump_cuda_time_total": True,
        }
    return Omni(
        model=str(model_dir),
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=args.tensor_parallel_size,
            ulysses_degree=args.ulysses_degree,
            ring_degree=args.ring_degree,
            text_encoder_tp_size=args.text_encoder_tp_size,
            vae_patch_parallel_size=args.vae_patch_parallel_size,
            vae_parallel_mode="tile",
        ),
        trust_remote_code=True,
        enable_cpu_offload=False,
        enable_distributed_layerwise_offload=(args.enable_distributed_layerwise_offload),
        dlo_use_allgather=False,
        dlo_resident_layers=args.dlo_resident_layers,
        enforce_eager=args.enforce_eager,
        **attention_engine_kwargs(args),
        enable_diffusion_pipeline_profiler=True,
        profiler_config=profiler_config,
    )


def closest_aspect_ratio(width: int, height: int) -> str:
    ratio = width / height
    return min(
        ASPECT_RATIOS,
        key=lambda label: abs(math.log(ratio / ASPECT_RATIOS[label])),
    )


def sampling_params(
    args: argparse.Namespace,
    *,
    task: str,
    seed: int,
    num_inference_steps: int | None = None,
) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        fps=24,
        num_inference_steps=(args.num_inference_steps if num_inference_steps is None else num_inference_steps),
        seed=seed,
        output_type="np",
        extra_args={
            "task": task,
            "duration": args.duration,
            "aspect_ratio": closest_aspect_ratio(args.width, args.height),
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        },
    )


def prompt_for(task_id: str) -> str:
    environment_key = f"MINIMAX_H3_{task_id.upper()}_PROMPT"
    return os.environ.get(environment_key, DEFAULT_PROMPTS[task_id])


def save_first_frame(frames: np.ndarray, output_path: Path) -> None:
    from PIL import Image

    first_frame = np.asarray(frames[0])
    if np.issubdtype(first_frame.dtype, np.floating):
        first_frame = np.clip(first_frame, 0.0, 1.0) * 255.0
    Image.fromarray(first_frame.astype(np.uint8)).save(output_path)


def extract_reference_audio(video_path: Path, audio_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(audio_path),
        ],
        check=True,
    )


def run_task(
    engine: Omni,
    args: argparse.Namespace,
    *,
    task_id: str,
    task: str,
    prompt: str | dict[str, Any],
    prompt_text: str,
    seed: int,
    output_path: Path,
) -> tuple[dict[str, object], np.ndarray]:
    if args.warmup_steps:
        engine.generate(
            prompt,
            sampling_params(
                args,
                task=task,
                seed=seed,
                num_inference_steps=args.warmup_steps,
            ),
            use_tqdm=False,
        )
    started = time.perf_counter()
    torch.cuda.nvtx.range_push(f"minimax_h3_task:{task_id}")
    try:
        outputs = engine.generate(
            prompt,
            sampling_params(args, task=task, seed=seed),
            use_tqdm=False,
        )
    finally:
        torch.cuda.nvtx.range_pop()
    wall_time = time.perf_counter() - started
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one output for {task_id}, found {len(outputs)}")

    result = outputs[0]
    frames = np.asarray(result.images[0])
    multimodal = result.multimodal_output
    if multimodal is None:
        raise RuntimeError(f"{task_id} returned no audio metadata")
    audio = np.asarray(multimodal["audio"])
    fps = int(multimodal["fps"])
    sample_rate = int(multimodal["audio_sample_rate"])

    if frames.ndim != 4 or tuple(frames.shape[1:]) != (
        args.height,
        args.width,
        3,
    ):
        raise RuntimeError(f"Unexpected {task_id} video shape: {frames.shape}")
    if audio.ndim not in (2, 3) or 2 not in audio.shape:
        raise RuntimeError(f"Unexpected {task_id} audio shape: {audio.shape}")
    if fps != 24 or sample_rate != 32000:
        raise RuntimeError(f"Unexpected {task_id} media rates: fps={fps}, audio={sample_rate}")

    output_path.write_bytes(
        _encode_video_bytes(
            frames,
            fps=fps,
            audio=audio,
            audio_sample_rate=sample_rate,
        )
    )
    record: dict[str, object] = {
        "task_id": task_id,
        "task": task,
        "partition": args.partition,
        "prompt": prompt_text,
        "seed": seed,
        "wall_time_s": round(wall_time, 4),
        "stage_durations": dict(getattr(result, "stage_durations", {}) or {}),
        "worker_peak_memory_mb": float(getattr(result, "peak_memory_mb", 0.0) or 0.0),
        "frames_shape": list(frames.shape),
        "audio_shape": list(audio.shape),
        "fps": fps,
        "audio_sample_rate": sample_rate,
        "frames_sha256": array_sha256(frames),
        "audio_sha256": array_sha256(audio),
        "mp4_sha256": file_sha256(output_path),
        "output": str(output_path),
        "completed_at": utc_now(),
    }
    print("TASK_RESULT " + json.dumps(record, sort_keys=True), flush=True)
    return record, frames


def update_summary(
    args: argparse.Namespace,
    *,
    hardware: list[dict[str, object]],
    records: list[dict[str, object]],
) -> None:
    summary_path = args.output_dir / "summary.json"
    expected_tasks = TASK_IDS if args.expect_ref2va else TASK_IDS[:2]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "started_at": utc_now(),
            "tasks": [],
        }
    summary["expected_tasks"] = list(expected_tasks)

    previous = {record["task_id"]: record for record in summary["tasks"]}
    previous.update({record["task_id"]: record for record in records})
    summary.update(
        {
            "updated_at": utc_now(),
            "status": ("completed" if all(task_id in previous for task_id in expected_tasks) else "in_progress"),
            "model_root": str(args.model_root),
            "hardware": hardware,
            "torch_version": torch.__version__,
            "parallel_config": (
                f"tp{args.tensor_parallel_size}_ulysses{args.ulysses_degree}_"
                f"ring{args.ring_degree}_text_encoder_tp{args.text_encoder_tp_size}_"
                f"vae_tile{args.vae_patch_parallel_size}"
            ),
            "attention_backend": args.attention_backend.upper(),
            "attention_fp8_scales": {
                "q": args.fp8_q_scale,
                "k": args.fp8_k_scale,
                "v": args.fp8_v_scale,
                "mode": (
                    "static"
                    if all(
                        value is not None
                        for value in (
                            args.fp8_q_scale,
                            args.fp8_k_scale,
                            args.fp8_v_scale,
                        )
                    )
                    else "first_call_calibration"
                ),
            }
            if args.attention_backend.upper() == "FLASHINFER_SM120_ATTN"
            else None,
            "precision": "checkpoint BF16/FP32",
            "memory_placement": (
                f"dlo_no_allgather_resident_{args.dlo_resident_layers}"
                if args.enable_distributed_layerwise_offload
                else "fully_resident"
            ),
            "regional_compile": not args.enforce_eager,
            "height": args.height,
            "width": args.width,
            "duration_seconds": args.duration,
            "num_inference_steps": args.num_inference_steps,
            "warmup_steps": args.warmup_steps,
            "tasks": [previous[task_id] for task_id in TASK_IDS if task_id in previous],
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_fl2va_partition(
    engine: Omni,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    t2va_path = args.output_dir / "01_t2va.mp4"
    first_frame_path = args.output_dir / "t2va_first_frame.png"
    reference_audio_path = args.output_dir / "t2va_reference_audio.wav"
    fl2va_path = args.output_dir / "02_fl2va_first_frame.mp4"

    t2va_prompt = prompt_for("t2va")
    t2va, frames = run_task(
        engine,
        args,
        task_id="t2va",
        task="t2va",
        prompt=t2va_prompt,
        prompt_text=t2va_prompt,
        seed=args.seed_base,
        output_path=t2va_path,
    )
    save_first_frame(frames, first_frame_path)
    extract_reference_audio(t2va_path, reference_audio_path)

    fl2va_prompt = prompt_for("fl2va_first_frame")
    fl2va, _ = run_task(
        engine,
        args,
        task_id="fl2va_first_frame",
        task="fl2va",
        prompt={
            "prompt": fl2va_prompt,
            "multi_modal_data": {"image": str(first_frame_path)},
        },
        prompt_text=fl2va_prompt,
        seed=args.seed_base + 1000,
        output_path=fl2va_path,
    )
    return [t2va, fl2va]


def require_generated_assets(output_dir: Path) -> dict[str, Path]:
    assets = {
        "first_frame": output_dir / "t2va_first_frame.png",
        "reference_audio": output_dir / "t2va_reference_audio.wav",
        "t2va_video": output_dir / "01_t2va.mp4",
        "fl2va_video": output_dir / "02_fl2va_first_frame.mp4",
    }
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Ref2VA phase requires the FL2VA phase outputs; missing: " + ", ".join(missing))
    return assets


def run_ref2va_partition(
    engine: Omni,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    assets = require_generated_assets(args.output_dir)
    image_audio_path = args.output_dir / "03_ref2va_image_audio.mp4"
    two_videos_path = args.output_dir / "04_ref2va_two_videos.mp4"

    image_audio_prompt = prompt_for("ref2va_image_audio")
    image_audio, _ = run_task(
        engine,
        args,
        task_id="ref2va_image_audio",
        task="ref2va",
        prompt={
            "prompt": image_audio_prompt,
            "multi_modal_data": {
                "image": str(assets["first_frame"]),
                "audio": str(assets["reference_audio"]),
            },
        },
        prompt_text=image_audio_prompt,
        seed=args.seed_base + 2000,
        output_path=image_audio_path,
    )

    two_videos_prompt = prompt_for("ref2va_two_videos")
    two_videos, _ = run_task(
        engine,
        args,
        task_id="ref2va_two_videos",
        task="ref2va",
        prompt={
            "prompt": two_videos_prompt,
            "multi_modal_data": {
                "video": [
                    str(assets["t2va_video"]),
                    str(assets["fl2va_video"]),
                ]
            },
        },
        prompt_text=two_videos_prompt,
        seed=args.seed_base + 3000,
        output_path=two_videos_path,
    )
    return [image_audio, two_videos]


def main() -> None:
    args = parse_args()
    validate_parallel_args(args)
    args.model_root = args.model_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    partition_name = "FL2VA" if args.partition == "fl2va" else "Ref2VA"
    model_dir = args.model_root / partition_name
    if not (model_dir / "model_index.json").is_file():
        raise FileNotFoundError(
            f"Missing {partition_name} checkpoint at {model_dir}. "
            "Download the selected MiniMax-H3 partition before running."
        )

    hardware = hardware_metadata(args.num_gpus)
    engine = make_engine(model_dir, args)
    profile_started = False
    try:
        if args.profiler_dir is not None:
            engine.start_profile(profile_prefix=args.partition)
            profile_started = True
        if args.partition == "fl2va":
            records = run_fl2va_partition(engine, args)
        else:
            records = run_ref2va_partition(engine, args)
    finally:
        if profile_started:
            engine.stop_profile()
        engine.close()

    update_summary(args, hardware=hardware, records=records)


if __name__ == "__main__":
    main()
