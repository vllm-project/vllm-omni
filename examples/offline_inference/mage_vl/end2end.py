#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run Mage-VL offline end-to-end inference through the upstream Mage scripts.

Mage-VL currently ships its custom Mage-ViT and codec processors as
Transformers remote code in the Microsoft Mage repository. This example keeps
the vLLM-Omni entrypoint stable while delegating the checkpoint-specific forward
path to that upstream implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path

DEFAULT_MODEL = "microsoft/Mage-VL"
DEFAULT_IMAGE_QUESTION = "Describe this image in detail."
DEFAULT_VIDEO_QUESTION = "Describe this video."


def _default_mage_repo() -> Path | None:
    value = os.getenv("MAGE_REPO_DIR")
    if value:
        return Path(value)
    sibling = Path(__file__).resolve().parents[3].parent / "Mage"
    if sibling.exists():
        return sibling
    return None


def _resolve_mage_repo(value: str | None) -> Path:
    repo = Path(value) if value else _default_mage_repo()
    if repo is None:
        raise ValueError("Set --mage-repo or MAGE_REPO_DIR to a checkout of https://github.com/microsoft/Mage.")
    repo = repo.expanduser().resolve()
    if not (repo / "mage_vl" / "inference_base.py").is_file():
        raise FileNotFoundError(f"Missing Mage-VL scripts under {repo}. Expected mage_vl/inference_base.py.")
    return repo


def _default_media_path(mage_repo: Path, task: str) -> Path:
    if task == "image":
        return mage_repo / "mage_vl" / "assets" / "examples" / "dog.jpg"
    return mage_repo / "mage_vl" / "assets" / "examples" / "soccer-broadcast.mp4"


def _resolve_media_path(path: str | None, mage_repo: Path, task: str) -> Path:
    resolved = Path(path).expanduser().resolve() if path else _default_media_path(mage_repo, task)
    if not resolved.is_file():
        raise FileNotFoundError(f"Input media file not found: {resolved}")
    return resolved


def build_command(args: argparse.Namespace, mage_repo: Path, media_path: Path) -> list[str]:
    if args.task == "streaming":
        return [
            sys.executable,
            str(mage_repo / "mage_vl" / "inference_streaming.py"),
            "--video",
            str(media_path),
            "--checkpoint",
            args.model,
            "--video_backend",
            args.video_backend,
            "--num_frames",
            str(args.num_frames),
            "--cur_fps",
            str(args.cur_fps),
            "--segment_sec",
            str(args.segment_sec),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--max_segments",
            str(args.max_segments),
            "--gate_threshold",
            str(args.gate_threshold),
            "--device",
            args.device,
            "--attn_impl",
            args.attn_impl,
        ]

    cmd = [
        sys.executable,
        str(mage_repo / "mage_vl" / "inference_base.py"),
        "--mode",
        "offline",
        "--question",
        args.question or (DEFAULT_IMAGE_QUESTION if args.task == "image" else DEFAULT_VIDEO_QUESTION),
        "--model",
        args.model,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.task == "image":
        cmd.extend(["--image", str(media_path)])
        return cmd

    cmd.extend(
        [
            "--video",
            str(media_path),
            "--video-backend",
            args.video_backend,
            "--num-frames",
            str(args.num_frames),
            "--max-pixels",
            str(args.max_pixels),
        ]
    )
    if args.video_backend == "codec":
        cmd.extend(["--codec-engine", args.codec_engine])
    return cmd


def check_runtime_dependencies(task: str) -> None:
    required = ["torch"]
    if task == "streaming":
        required.append("mamba_ssm")
    missing = [name for name in required if find_spec(name) is None]
    if not missing:
        return

    install_hint = (
        "Install Mage-VL runtime dependencies in this environment first:\n"
        "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
        "  grep -v '^mamba-ssm' /path/to/Mage/mage_vl/requirements.txt > /tmp/mage_vl_requirements_no_mamba.txt\n"
        "  pip install -r /tmp/mage_vl_requirements_no_mamba.txt\n"
        "  pip install 'causal-conv1d>=1.4.0' --no-build-isolation\n"
        "  pip install 'mamba-ssm>=2.2' --no-build-isolation"
    )
    raise RuntimeError(f"Missing Python packages: {', '.join(missing)}.\n{install_hint}")


def run_command(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        input="y\n",
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def write_outputs(
    *,
    args: argparse.Namespace,
    mage_repo: Path,
    media_path: Path,
    cmd: list[str],
    result: subprocess.CompletedProcess[str] | None,
    elapsed_s: float | None,
) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": args.task,
        "model": args.model,
        "mage_repo": str(mage_repo),
        "media_path": str(media_path),
        "command": cmd,
        "dry_run": args.dry_run,
        "elapsed_s": elapsed_s,
        "returncode": None if result is None else result.returncode,
    }
    if result is not None:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        summary["stdout"] = stdout
        summary["stderr"] = stderr
        (output_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (output_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mage-VL offline end-to-end inference example.")
    parser.add_argument(
        "--mage-repo",
        default=None,
        help="Path to a Microsoft Mage checkout. Defaults to MAGE_REPO_DIR.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local checkpoint path.")
    parser.add_argument("--task", choices=("image", "video", "streaming"), default="image")
    parser.add_argument("--image-path", default=None, help="Image input. Defaults to the Mage sample image.")
    parser.add_argument("--video-path", default=None, help="Video input. Defaults to the Mage sample video.")
    parser.add_argument("--question", default=None, help="Question for image/video offline QA.")
    parser.add_argument("--video-backend", choices=("frames", "codec"), default="frames")
    parser.add_argument("--codec-engine", choices=("traditional", "neural"), default="traditional")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=150000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cur-fps", type=float, default=2.0)
    parser.add_argument("--segment-sec", type=float, default=8.0)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-impl", choices=("flash_attention_2", "sdpa", "eager"), default="flash_attention_2")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output-dir", default="outputs/mage_vl")
    parser.add_argument("--dry-run", action="store_true", help="Build the command and summary without executing it.")
    args = parser.parse_args()
    if args.num_frames <= 0:
        parser.error("--num-frames must be positive")
    if args.segment_sec <= 0:
        parser.error("--segment-sec must be positive")
    if args.task == "image" and args.video_path:
        parser.error("--video-path cannot be used with --task image")
    if args.task in {"video", "streaming"} and args.image_path:
        parser.error("--image-path cannot be used with video tasks")
    return args


def main() -> None:
    args = parse_args()
    mage_repo = _resolve_mage_repo(args.mage_repo)
    media_arg = args.image_path if args.task == "image" else args.video_path
    media_path = _resolve_media_path(media_arg, mage_repo, args.task)
    cmd = build_command(args, mage_repo, media_path)

    result = None
    elapsed_s = None
    if not args.dry_run:
        check_runtime_dependencies(args.task)
        start = time.perf_counter()
        result = run_command(cmd, cwd=mage_repo, timeout=args.timeout)
        elapsed_s = time.perf_counter() - start

    summary_path = write_outputs(
        args=args,
        mage_repo=mage_repo,
        media_path=media_path,
        cmd=cmd,
        result=result,
        elapsed_s=elapsed_s,
    )
    print(f"summary saved to {summary_path}")
    if result is not None:
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, file=sys.stderr, end="")
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
