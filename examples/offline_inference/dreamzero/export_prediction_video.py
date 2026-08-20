#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from client_schedule import (
    CAMERA_FILES,
    DEFAULT_NUM_AR_CHUNKS,
    build_ar_observations,
)
from PIL import Image

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

WORKER_EXTENSION = "vllm_omni.diffusion.models.dreamzero.video_export_worker.DreamZeroVideoExportWorkerExtension"
ASSET_REPO_ID = "YangshenDeng/vllm-omni-dreamzero-assets"
DEFAULT_SESSION_PREFIX = "dreamzero-export"


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Export DreamZero prediction video from downloaded example inputs.")
    parser.add_argument("--model", default="GEAR-Dreams/DreamZero-DROID")
    parser.add_argument("--deploy-config", type=Path, required=True)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=repo_root / "outputs" / "dreamzero" / "assets",
        help="Directory containing the three camera MP4 files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "outputs" / "dreamzero" / "generated_predictions"
    )
    parser.add_argument("--output-stem", default="dreamzero_prediction")
    parser.add_argument(
        "--prompt",
        default="Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_NUM_AR_CHUNKS,
        help="Number of 4-frame chunks to send after the initial frame (default: 15).",
    )
    parser.add_argument(
        "--repeat-chunk-observations",
        action="store_true",
        help="Repeat the last valid chunk observation when bundled assets run out of frames.",
    )
    parser.add_argument("--save-input-video", action="store_true")
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--save-actions", action="store_true")
    parser.add_argument("--fps", type=int, default=5)
    return parser.parse_args()


def _load_all_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def _load_camera_frames(video_dir: Path) -> dict[str, np.ndarray]:
    camera_frames: dict[str, np.ndarray] = {}
    for camera_key, file_name in CAMERA_FILES.items():
        video_path = video_dir / file_name
        if not video_path.exists():
            raise FileNotFoundError(
                f"Missing DreamZero example asset: {video_path}. "
                "Download the example videos with: "
                f"`hf download {ASSET_REPO_ID} --repo-type dataset --local-dir {video_dir}`"
            )
        camera_frames[camera_key] = _load_all_frames(video_path)
    return camera_frames


def _build_observations(
    video_dir: Path,
    prompt: str,
    session_id: str,
    *,
    num_chunks: int,
    repeat_chunk_observations: bool,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    camera_frames = _load_camera_frames(video_dir)
    observations = build_ar_observations(
        camera_frames,
        prompt=prompt,
        session_id=session_id,
        num_chunks=num_chunks,
        repeat_chunk_observations=repeat_chunk_observations,
    )
    if len(observations) < 2:
        raise RuntimeError("Need at least two DreamZero observations to export a prediction video.")
    return camera_frames, observations


def _extract_latents(output: OmniRequestOutput) -> torch.Tensor:
    if not isinstance(output, OmniRequestOutput):
        raise TypeError(f"Expected OmniRequestOutput, got {type(output)!r}")
    if not output.images:
        raise RuntimeError("DreamZero output does not contain video latents in `images`.")

    latents = output.images[0]
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"Expected tensor latents, got {type(latents)!r}")

    latents = latents.detach().cpu()
    if latents.dim() == 4:
        latents = latents.unsqueeze(0)
    if latents.dim() != 5:
        raise ValueError(f"Unexpected latent shape: {tuple(latents.shape)}")

    if latents.shape[1] < latents.shape[2]:
        latents = latents.transpose(1, 2).contiguous()
    return latents


def _write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _write_gif(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = max(int(round(1000 / max(fps, 1))), 1)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def _stitch_input_frames(camera_frames: dict[str, np.ndarray]) -> np.ndarray:
    total_frames = min(frames.shape[0] for frames in camera_frames.values())
    stitched = []
    for frame_index in range(total_frames):
        left = camera_frames["observation/exterior_image_0_left"][frame_index]
        right = camera_frames["observation/exterior_image_1_left"][frame_index]
        wrist = camera_frames["observation/wrist_image_left"][frame_index]
        pad = np.zeros((left.shape[0], left.shape[1], 3), dtype=np.uint8)
        canvas = np.concatenate([left, right], axis=1)
        bottom = np.concatenate([wrist, pad], axis=1)
        stitched.append(np.concatenate([canvas, bottom], axis=0))
    return np.stack(stitched, axis=0)


def _run_generation(
    model: str, deploy_config_path: Path, observations: list[dict]
) -> tuple[Omni, list[OmniRequestOutput]]:
    omni = Omni(
        model=model,
        deploy_config=str(deploy_config_path),
        enforce_eager=True,
        worker_extension_cls=WORKER_EXTENSION,
    )

    outputs: list[OmniRequestOutput] = []
    for index, obs in enumerate(observations):
        sampling_params = OmniDiffusionSamplingParams(
            extra_args={
                "reset": index == 0,
                "session_id": obs["session_id"],
                "robot_obs": obs,
            }
        )
        result = omni.generate(obs["prompt"], sampling_params_list=[sampling_params])
        if not result:
            raise RuntimeError(f"No output returned for DreamZero request {index}")
        outputs.append(result[0])
    return omni, outputs


def _decode_accumulated_with_worker(omni: Omni, session_id: str) -> np.ndarray:
    """Decode server-side accumulated latents when the pipeline supports it."""
    stage_client = omni.engine.stage_clients[0]
    engine = getattr(stage_client, "_engine", None)
    if engine is None:
        raise RuntimeError("DreamZero export requires inline diffusion stage access.")

    decoded = engine.executor.collective_rpc(
        "decode_accumulated_video_latents_to_uint8",
        args=(session_id,),
        unique_reply_rank=0,
        exec_all_ranks=True,
    )
    if isinstance(decoded, torch.Tensor):
        decoded = decoded.numpy()
    if not isinstance(decoded, np.ndarray):
        raise TypeError(f"Unexpected decoded output type: {type(decoded)!r}")
    return decoded


def _decode_with_worker(omni: Omni, full_latents: torch.Tensor) -> np.ndarray:
    stage_client = omni.engine.stage_clients[0]
    engine = getattr(stage_client, "_engine", None)
    if engine is None:
        raise RuntimeError("DreamZero export requires inline diffusion stage access.")

    decoded = engine.executor.collective_rpc(
        "decode_video_latents_to_uint8",
        args=(full_latents,),
        unique_reply_rank=0,
        exec_all_ranks=True,
    )
    if isinstance(decoded, torch.Tensor):
        decoded = decoded.numpy()
    if not isinstance(decoded, np.ndarray):
        raise TypeError(f"Unexpected decoded output type: {type(decoded)!r}")
    return decoded


def _decode_video(omni: Omni, session_id: str, outputs: list[OmniRequestOutput]) -> np.ndarray:
    """Decode full AR rollout by concatenating per-step latents from all outputs."""
    latent_steps = [_extract_latents(output) for output in outputs]
    latent_t = [step.shape[2] for step in latent_steps]
    print(f"Decoding {len(latent_steps)} latent steps, latent_t={latent_t}, total_t={sum(latent_t)}")
    full_latents = torch.cat(latent_steps, dim=2)
    return _decode_with_worker(omni, full_latents)


def main() -> None:
    args = _parse_args()
    session_id = args.session_id or f"{DEFAULT_SESSION_PREFIX}-{uuid.uuid4()}"

    camera_frames, observations = _build_observations(
        video_dir=args.video_dir,
        prompt=args.prompt,
        session_id=session_id,
        num_chunks=args.num_chunks,
        repeat_chunk_observations=args.repeat_chunk_observations,
    )
    print(
        f"Running {len(observations)} inferences "
        f"(1 initial + {len(observations) - 1} chunks; requested {args.num_chunks} post-initial chunks)"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_input_video:
        input_frames = _stitch_input_frames(camera_frames)
        _write_mp4(args.output_dir / f"{args.output_stem}_input.mp4", input_frames, fps=15)
        if args.save_gif:
            _write_gif(args.output_dir / f"{args.output_stem}_input.gif", input_frames[::3], fps=5)

    omni = None
    try:
        omni, outputs = _run_generation(
            model=args.model,
            deploy_config_path=args.deploy_config,
            observations=observations,
        )
        frames = _decode_video(omni, session_id, outputs)
    finally:
        if omni is not None:
            omni.close()

    expected_frames = 8 * len(observations) + 1
    mp4_path = args.output_dir / f"{args.output_stem}.mp4"

    _write_mp4(mp4_path, frames, fps=args.fps)
    print(
        f"SAVED_MP4={mp4_path} frames={len(frames)} "
        f"expected≈{expected_frames} duration≈{len(frames) / max(args.fps, 1):.1f}s"
    )

    if args.save_gif:
        gif_path = args.output_dir / f"{args.output_stem}.gif"
        _write_gif(gif_path, frames, fps=args.fps)
        print(f"SAVED_GIF={gif_path}")

    if args.save_actions:
        npz_path = args.output_dir / f"{args.output_stem}_actions.npz"
        np.savez(
            npz_path,
            **{
                f"step{index}": np.asarray(output.multimodal_output.get("actions"))
                for index, output in enumerate(outputs)
            },
        )
        print(f"SAVED_ACTIONS={npz_path}")


if __name__ == "__main__":
    main()
