#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = Path(__file__).resolve().parent
EXPORT_SCRIPT = EXAMPLE_DIR / "export_prediction_video.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "dreamzero" / "comparison_videos"
DEFAULT_MODEL = "GEAR-Dreams/DreamZero-DROID"
ASSETS_DIR = EXAMPLE_DIR / "assets"

CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}

STAGE_CONFIGS = {
    "tp1_cfg1": REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "dreamzero.yaml",
    "tp1_cfg2": REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "dreamzero_tp1_cfg2.yaml",
    "tp2_cfg1": REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "dreamzero_tp2_cfg1.yaml",
    "tp2_cfg2": REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "dreamzero_tp2_cfg2.yaml",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DreamZero comparison videos for four vLLM configs.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--upstream-video", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
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


def _load_camera_frames() -> dict[str, np.ndarray]:
    camera_frames: dict[str, np.ndarray] = {}
    for camera_key, file_name in CAMERA_FILES.items():
        camera_frames[camera_key] = _load_all_frames(ASSETS_DIR / file_name)
    return camera_frames


def _write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _write_input_reference(output_dir: Path) -> Path:
    output_path = output_dir / "dreamzero_input_reference.mp4"
    camera_frames = _load_camera_frames()
    total_frames = min(frames.shape[0] for frames in camera_frames.values())
    stitched = []
    for frame_index in range(total_frames):
        left = camera_frames["observation/exterior_image_0_left"][frame_index]
        right = camera_frames["observation/exterior_image_1_left"][frame_index]
        wrist = camera_frames["observation/wrist_image_left"][frame_index]
        pad = np.zeros((left.shape[0], left.shape[1], 3), dtype=np.uint8)
        top = np.concatenate([left, right], axis=1)
        bottom = np.concatenate([wrist, pad], axis=1)
        stitched.append(np.concatenate([top, bottom], axis=0))
    _write_mp4(output_path, np.stack(stitched, axis=0), fps=15)
    return output_path


def _run_export(args: argparse.Namespace, config_name: str, stage_config_path: Path) -> Path:
    output_stem = f"{config_name}_vllm_example"
    output_path = args.output_dir / f"{output_stem}.mp4"
    if args.skip_existing and output_path.exists():
        return output_path

    cmd = [
        args.python,
        str(EXPORT_SCRIPT),
        "--model",
        args.model,
        "--stage-configs-path",
        str(stage_config_path),
        "--output-dir",
        str(args.output_dir),
        "--output-stem",
        output_stem,
        "--fps",
        str(args.fps),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return output_path


def _copy_upstream_video(output_dir: Path, upstream_video: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "dreamzero_upstream_reference.mp4"
    shutil.copy2(upstream_video, dst)
    return dst


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}
    failures: dict[str, str] = {}
    manifest["input_reference"] = _display_path(_write_input_reference(args.output_dir))

    for config_name, stage_config_path in STAGE_CONFIGS.items():
        try:
            manifest[config_name] = _display_path(_run_export(args, config_name, stage_config_path))
        except subprocess.CalledProcessError as exc:
            if not args.continue_on_error:
                raise
            failures[config_name] = str(exc).replace(str(REPO_ROOT) + "/", "")

    if args.upstream_video is not None:
        manifest["upstream_reference"] = _display_path(_copy_upstream_video(args.output_dir, args.upstream_video))

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"videos": manifest, "failures": failures}, indent=2) + "\n")

    for name, path in manifest.items():
        print(f"{name}={path}")
    for name, error in failures.items():
        print(f"FAILED_{name}={error}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
