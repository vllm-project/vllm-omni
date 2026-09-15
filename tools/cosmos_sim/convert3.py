#!/usr/bin/env python3
"""Convert AgiBotWorld-Beta clips for offline and interactive Cosmos3-Nano-Sim-Bimanual.

This standalone tool saves NPZ/JSONL evaluation artifacts. Every selected
clip also emits:

* ``agibot_scene_NNN.json``: a calibrated interactive scene bundle;
* ``interactive_seed_NNN.png``: the frame-0 RGB image at deployment size; and
* ``replay_NNN.pt``: raw AgiBot actions shaped ``[chunks, 16, 29]``.

The initial head/wrist transforms and gripper open fractions all come from
observation frame 0.  Replay row 0 remains the frame-0-to-frame-1 action, so
the scene and replay share an exact temporal boundary.

Example (run with the imaginaire4 Cosmos environment)::

    /path/to/imaginaire4/packages/cosmos3/.venv/bin/python convert3.py \
        --imaginaire-root /path/to/imaginaire4 \
        --dataset-root /data/agibotworld/concrete_lerobot_shard \
        --output-dir /data/agibot_interactive_eval \
        --num-frames 65 \
        --sample-count 4

``num_frames - 1`` must be divisible by 16 because one interactive tick
contains exactly 16 raw action rows.  The default 65 frames produces four
replay ticks.

Offline payloads are saved as ``sample_NNN.npz`` with numeric arrays and
Unicode string scalars, readable with ``np.load(..., allow_pickle=False)``.
``samples.jsonl`` references these files using ``npz_path``. Run the tool
with the imaginaire4 Cosmos environment, which provides the dataset, NumPy,
PyTorch, Pillow, and imageio/FFmpeg dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_IMAGINAIRE_MARKER = Path("projects/cosmos3/cosmos3/datasets/action/agibotworld_beta_dataset.py")
_TEMPORAL_COMPRESSION_FACTOR = 4
_AGIBOT_ACTION_DIM = 29
_AGIBOT_DOMAIN_ID = 15
_AGIBOT_EMBODIMENT = "agibotworld"
_ACTION_STEPS_PER_TICK = 16
_ACTION_COORDINATE_VERSION = "agibotworld.backward_framewise.rot6d.opencv.v1"
_DEFAULT_PROMPT = "An AgiBot robot interacts with its workspace."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imaginaire-root",
        type=Path,
        help=(
            "Path to the imaginaire4 checkout. If omitted, the script searches "
            "its directory, the working directory, and their parents."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more concrete AgiBotWorld-Beta LeRobot dataset roots. "
            "Pass the directories containing meta/info.json, not the common "
            "parent that still needs LEROBOT_ROOTS expansion."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=65,
        help="Pixel frames per clip. Must be 16*k+1; 65, 129, and 257 are useful quality checks.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--sample-count",
        type=int,
        default=1,
        help="Number of deterministic, evenly spaced dataset samples to export.",
    )
    selection.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="Explicit flat dataset indices to export instead of --sample-count.",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--sample-stride", type=int, default=200)
    parser.add_argument(
        "--min-episode-length-frames",
        type=int,
        default=900,
        help="Filter on native episode length; use 0 to disable it.",
    )
    parser.add_argument("--max-loaded-datasets", type=int, default=32)
    parser.add_argument("--fast-init", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-init-max-workers", type=int, default=64)
    parser.add_argument("--use-subtask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--viewpoint",
        choices=("ego_view", "concat_view"),
        default="ego_view",
        help="Use ego_view for the egocentric Cosmos3-Nano-Sim-Bimanual checkpoint.",
    )
    parser.add_argument("--expected-height", type=int, default=720)
    parser.add_argument("--expected-width", type=int, default=1280)
    parser.add_argument(
        "--skip-ground-truth-video",
        action="store_true",
        help="Write the NPZ and conditioning PNG, but skip the visual-reference MP4.",
    )
    parser.add_argument(
        "--video-quality",
        type=int,
        default=9,
        help="imageio/FFmpeg quality for the visual-reference MP4 (0-10).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files with the same generated names. Unrelated files are left untouched.",
    )
    parser.add_argument(
        "--scene-prompt",
        help=(
            "Override the dataset caption in every interactive scene. By default, "
            "each sample's ai_caption is used, with a non-empty generic fallback."
        ),
    )
    parser.add_argument(
        "--linear-velocity-m-s",
        type=float,
        default=0.12,
        help="Interactive controller translation limit (default: 0.12 m/s).",
    )
    parser.add_argument(
        "--angular-velocity-rad-s",
        type=float,
        default=math.radians(30.0),
        help="Interactive controller rotation limit (default: pi/6 rad/s).",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_frames < 17:
        raise ValueError("--num-frames must be at least 17.")
    if (args.num_frames - 1) % _ACTION_STEPS_PER_TICK != 0:
        raise ValueError(
            "Interactive replay requires --num-frames = 16*k+1 so every action "
            f"row belongs to a complete tick; got {args.num_frames}. "
            "Try 17, 65, 129, 257, 385, or 401."
        )
    if args.sample_count is not None and args.sample_count <= 0:
        raise ValueError("--sample-count must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be positive.")
    if args.min_episode_length_frames < 0:
        raise ValueError("--min-episode-length-frames cannot be negative.")
    if args.max_loaded_datasets <= 0:
        raise ValueError("--max-loaded-datasets must be positive.")
    if args.fast_init_max_workers <= 0:
        raise ValueError("--fast-init-max-workers must be positive.")
    if args.expected_height <= 0 or args.expected_width <= 0:
        raise ValueError("--expected-height and --expected-width must be positive.")
    if not 0 <= args.video_quality <= 10:
        raise ValueError("--video-quality must be in [0, 10].")
    if args.viewpoint != "ego_view":
        raise ValueError(
            "Interactive Cosmos3-Nano-Sim-Bimanual uses the egocentric artifact; "
            "convert3.py therefore requires --viewpoint ego_view."
        )
    for name in ("linear_velocity_m_s", "angular_velocity_rad_s"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} must be positive and finite.")
    if args.scene_prompt is not None and not args.scene_prompt.strip():
        raise ValueError("--scene-prompt cannot be empty or whitespace-only.")


def _is_imaginaire_root(path: Path) -> bool:
    return (path / _IMAGINAIRE_MARKER).is_file()


def _find_imaginaire_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not _is_imaginaire_root(root):
            raise FileNotFoundError(f"--imaginaire-root {root} does not contain {_IMAGINAIRE_MARKER}.")
        return root

    candidates: list[Path] = []
    env_root = os.environ.get("IMAGINAIRE4_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for parent in (start, *start.parents):
            candidates.extend((parent, parent / "imaginaire4"))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_imaginaire_root(resolved):
            return resolved
    raise FileNotFoundError("Could not find the imaginaire4 checkout. Pass --imaginaire-root /path/to/imaginaire4.")


def _select_indices(dataset_length: int, count: int | None, requested: list[int] | None) -> list[int]:
    if dataset_length <= 0:
        raise RuntimeError(
            "The registered dataset has no eligible windows. Check the concrete dataset roots; "
            "for a small local dataset, try --split train; for long clips, lower "
            "--num-frames or --min-episode-length-frames."
        )

    if requested is not None:
        selected: list[int] = []
        seen: set[int] = set()
        for raw_index in requested:
            index = raw_index + dataset_length if raw_index < 0 else raw_index
            if index < 0 or index >= dataset_length:
                raise IndexError(f"Dataset index {raw_index} resolves to {index}, outside [0, {dataset_length}).")
            if index not in seen:
                selected.append(index)
                seen.add(index)
        if not selected:
            raise ValueError("--indices did not contain any usable indices.")
        return selected

    assert count is not None
    if count > dataset_length:
        raise ValueError(f"--sample-count {count} exceeds the eligible dataset length {dataset_length}.")
    if count == 1:
        return [0]
    return [round(position * (dataset_length - 1) / (count - 1)) for position in range(count)]


def _scalar_float(value: Any, *, name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected scalar {name}, got {value!r}.") from exc


def _scalar_int(value: Any, *, name: str) -> int:
    numeric = _scalar_float(value, name=name)
    integer = int(numeric)
    if numeric != integer:
        raise ValueError(f"Expected integer-valued {name}, got {numeric}.")
    return integer


def _copy_optional_metadata(sample: dict[str, Any], payload: dict[str, Any]) -> None:
    for key in (
        "initial_pose",
        "initial_pose_right",
        "initial_pose_left",
        "debug_caption",
        "additional_view_description",
    ):
        if key not in sample:
            continue
        value = sample[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().contiguous()
        payload[key] = value


def _prepare_payload(
    sample: dict[str, Any],
    *,
    dataset_index: int,
    num_frames: int,
    expected_fps: float,
) -> tuple[dict[str, Any], tuple[int, int]]:
    import torch

    video = sample.get("video")
    action = sample.get("action")
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"Expected sample['video'] to be a tensor, got {type(video).__name__}.")
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"Expected sample['action'] to be a tensor, got {type(action).__name__}.")

    video = video.detach().cpu().contiguous()
    action = action.detach().cpu().to(dtype=torch.float32).contiguous()
    expected_video_prefix = (3, num_frames)
    if video.ndim != 4 or tuple(video.shape[:2]) != expected_video_prefix:
        raise ValueError(f"Expected dataset video shape [3, {num_frames}, H, W], got {tuple(video.shape)}.")
    if video.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 RGB video, got {video.dtype}.")
    expected_action_shape = (num_frames - 1, _AGIBOT_ACTION_DIM)
    if tuple(action.shape) != expected_action_shape:
        raise ValueError(f"Expected raw action shape {expected_action_shape}, got {tuple(action.shape)}.")
    if not bool(torch.isfinite(action).all()):
        raise ValueError(f"Sample {dataset_index} contains non-finite action values.")

    fps = _scalar_float(sample.get("conditioning_fps", expected_fps), name="conditioning_fps")
    if abs(fps - expected_fps) > 1e-5:
        raise ValueError(f"Dataset returned conditioning_fps={fps}, expected {expected_fps}.")
    domain_id = _scalar_int(sample.get("domain_id", _AGIBOT_DOMAIN_ID), name="domain_id")
    if domain_id != _AGIBOT_DOMAIN_ID:
        raise ValueError(f"AgiBotWorld must use domain_id={_AGIBOT_DOMAIN_ID}, got {domain_id}.")

    caption = str(sample.get("ai_caption", "")).strip()
    payload: dict[str, Any] = {
        "ai_caption": caption,
        "video": video,
        "action": action,
        "conditioning_fps": fps,
        "domain_id": domain_id,
        "domain_name": _AGIBOT_EMBODIMENT,
        "embodiment": _AGIBOT_EMBODIMENT,
        "mode": str(sample.get("mode", "forward_dynamics")),
        "viewpoint": str(sample.get("viewpoint", "ego_view")),
        "source_dataset_index": dataset_index,
        "num_frames": num_frames,
    }
    _copy_optional_metadata(sample, payload)
    return payload, (int(video.shape[2]), int(video.shape[3]))


def _prepare_interactive_payload(
    sample: dict[str, Any],
    *,
    dataset_index: int,
    num_frames: int,
    expected_fps: float,
) -> tuple[dict[str, Any], tuple[int, int]]:
    payload, resolution = _prepare_payload(
        sample,
        dataset_index=dataset_index,
        num_frames=num_frames,
        expected_fps=expected_fps,
    )
    for key in ("initial_gripper_right", "initial_gripper_left"):
        if key not in sample:
            raise KeyError(
                f"Dataset sample is missing {key!r}; use the interactive dataset subclass created by convert3.py."
            )
        value = _scalar_float(sample[key], name=key)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be a finite open fraction in [0,1], got {value}.")
        payload[key] = value
    return payload, resolution


def _write_npz(path: Path, payload: dict[str, Any]) -> None:
    """Save the full payload without object arrays or pickled tensors."""

    import numpy as np
    import torch

    arrays = {}
    for key, value in payload.items():
        # Missing optional metadata must not become an object array for None.
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            value = value.numpy()
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(
                f"Payload field {key!r} cannot be stored without pickle; expected numeric arrays or string scalars."
            )
        arrays[key] = array
    np.savez(path, **arrays)


def _transform_to_json(value: Any, *, name: str) -> list[list[float]]:
    """Validate a calibrated rigid transform before putting it in a scene."""

    import torch

    transform = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tuple(transform.shape) != (4, 4):
        raise ValueError(f"{name} must have shape [4,4], got {tuple(transform.shape)}.")
    if not bool(torch.isfinite(transform).all()):
        raise ValueError(f"{name} contains NaN or Inf values.")
    expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0])
    if not torch.allclose(transform[3], expected_bottom, atol=1e-5, rtol=0):
        raise ValueError(f"{name} must be a homogeneous rigid transform.")
    rotation = transform[:3, :3]
    if not torch.allclose(
        rotation.T @ rotation,
        torch.eye(3, dtype=torch.float32),
        atol=1e-4,
        rtol=1e-4,
    ):
        raise ValueError(f"{name} rotation must be orthonormal.")
    if not torch.allclose(torch.det(rotation), torch.tensor(1.0), atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} rotation must have determinant +1.")
    return transform.tolist()


def _write_conditioning_png(path: Path, video: Any) -> None:
    import numpy as np
    from PIL import Image

    frame = video[:, 0].permute(1, 2, 0).numpy()
    Image.fromarray(np.ascontiguousarray(frame), mode="RGB").save(path)


def _write_ground_truth_video(path: Path, video: Any, *, fps: float, quality: int) -> None:
    """Write uint8 RGB frames without the accidental second 255x scaling."""

    import imageio.v2 as imageio
    import numpy as np

    try:
        writer = imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            quality=quality,
            macro_block_size=None,
            pixelformat="yuv420p",
            ffmpeg_log_level="error",
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize MP4 export. Install imageio and imageio-ffmpeg, "
            "or rerun with --skip-ground-truth-video."
        ) from exc

    try:
        for frame_index in range(video.shape[1]):
            frame = video[:, frame_index].permute(1, 2, 0).numpy()
            writer.append_data(np.ascontiguousarray(frame))
    finally:
        writer.close()


def _write_interactive_seed(
    path: Path,
    video: Any,
    *,
    target_height: int,
    target_width: int,
) -> None:
    """Write frame 0 using the model's aspect-resize/center-crop algorithm."""

    import numpy as np
    from PIL import Image

    frame = video[:, 0].permute(1, 2, 0).numpy()
    image = Image.fromarray(np.ascontiguousarray(frame), mode="RGB")
    scale = max(target_width / image.width, target_height / image.height)
    resize_width = int(math.ceil(scale * image.width))
    resize_height = int(math.ceil(scale * image.height))
    image = image.resize(
        (resize_width, resize_height),
        Image.Resampling.LANCZOS,
    )
    left = (resize_width - target_width) // 2
    top = (resize_height - target_height) // 2
    image.crop((left, top, left + target_width, top + target_height)).save(path)


def _write_replay(path: Path, action: Any) -> int:
    import torch

    actions = torch.as_tensor(action, dtype=torch.float32).detach().cpu().contiguous()
    if actions.ndim != 2 or actions.shape[1] != _AGIBOT_ACTION_DIM:
        raise ValueError(f"Replay source actions must have shape [steps,29], got {tuple(actions.shape)}.")
    if actions.shape[0] % _ACTION_STEPS_PER_TICK != 0:
        raise ValueError(f"Replay has {actions.shape[0]} rows, which is not divisible by {_ACTION_STEPS_PER_TICK}.")
    chunks = actions.reshape(
        -1,
        _ACTION_STEPS_PER_TICK,
        _AGIBOT_ACTION_DIM,
    ).contiguous()
    torch.save(chunks, path)
    return int(chunks.shape[0])


def _scene_payload(
    payload: dict[str, Any],
    *,
    seed_name: str,
    prompt_override: str | None,
    linear_velocity_m_s: float,
    angular_velocity_rad_s: float,
) -> dict[str, Any]:
    prompt = prompt_override or str(payload.get("ai_caption", "")).strip()
    if not prompt:
        prompt = _DEFAULT_PROMPT
    return {
        "seed_rgb": seed_name,
        "prompt": prompt,
        "fps": float(payload["conditioning_fps"]),
        "domain_id": _AGIBOT_DOMAIN_ID,
        "embodiment": _AGIBOT_EMBODIMENT,
        "action_coordinate_version": _ACTION_COORDINATE_VERSION,
        "head_transform": _transform_to_json(payload["initial_pose"], name="head_transform"),
        "right_wrist_transform": _transform_to_json(payload["initial_pose_right"], name="right_wrist_transform"),
        "left_wrist_transform": _transform_to_json(payload["initial_pose_left"], name="left_wrist_transform"),
        "right_gripper": float(payload["initial_gripper_right"]),
        "left_gripper": float(payload["initial_gripper_left"]),
        "limits": {
            "linear_velocity_m_s": float(linear_velocity_m_s),
            "angular_velocity_rad_s": float(angular_velocity_rad_s),
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _planned_paths(
    output_dir: Path,
    sample_total: int,
    *,
    write_video: bool,
) -> list[Path]:
    paths = [output_dir / "samples.jsonl", output_dir / "conversion_summary.json"]
    for ordinal in range(sample_total):
        paths.extend(
            (
                output_dir / f"sample_{ordinal:03d}.npz",
                output_dir / f"conditioning_{ordinal:03d}.png",
                output_dir / f"interactive_seed_{ordinal:03d}.png",
                output_dir / f"agibot_scene_{ordinal:03d}.json",
                output_dir / f"replay_{ordinal:03d}.pt",
            )
        )
        if write_video:
            paths.append(output_dir / f"ground_truth_{ordinal:03d}.mp4")
    return paths


def _check_collisions(paths: list[Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    collisions = [path for path in paths if path.exists()]
    if collisions:
        preview = "\n  ".join(str(path) for path in collisions[:10])
        suffix = "\n  ..." if len(collisions) > 10 else ""
        raise FileExistsError(
            f"Refusing to replace existing output files; pass --overwrite if intentional:\n  {preview}{suffix}"
        )


def main() -> int:
    args = _build_parser().parse_args()
    _validate_args(args)
    imaginaire_root = _find_imaginaire_root(args.imaginaire_root)
    sys.path.insert(0, str(imaginaire_root))

    try:
        import numpy as np
        import torch
        from projects.cosmos3.cosmos3.datasets.action.agibot_gear_fk import (
            convert_gripper_state_to_open_fraction,
        )
        from projects.cosmos3.cosmos3.datasets.action.agibotworld_beta_dataset import (
            AgiBotWorldBetaDataset,
        )
    except ImportError as exc:
        raise ImportError(
            "Failed to import the AgiBotWorld converter dependencies. Run this "
            "script with the imaginaire4 Cosmos Python environment and pass the "
            "matching --imaginaire-root."
        ) from exc

    class InteractiveAgiBotWorldBetaDataset(AgiBotWorldBetaDataset):
        """Add exact observation-frame-0 grippers to the normal sample."""

        def _build_fk_action(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
            action, extras = super()._build_fk_action(sample)
            effector = (
                sample["observation.states.effector.position"].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            if effector.ndim != 2 or effector.shape[0] == 0 or effector.shape[1] != 2:
                raise ValueError(
                    f"observation.states.effector.position must have shape [T+1,2], got {tuple(effector.shape)}."
                )
            # Beta column order is left, right. Convert the whole observed
            # interval so encoding detection is identical to action creation,
            # then retain the scene's frame-0 absolute state.
            left = convert_gripper_state_to_open_fraction(effector[:, 0])
            right = convert_gripper_state_to_open_fraction(effector[:, 1])
            extras["initial_gripper_left"] = float(left[0])
            extras["initial_gripper_right"] = float(right[0])
            return action, extras

    chunk_length = args.num_frames - 1
    concrete_roots = [str(path.expanduser().resolve()) for path in args.dataset_root]
    missing_roots = [root for root in concrete_roots if not Path(root).is_dir()]
    if missing_roots:
        raise FileNotFoundError(f"Dataset roots do not exist: {missing_roots}")

    print(
        f"Registering AgiBotWorld-Beta {args.split} split for "
        f"{args.num_frames} video frames and {chunk_length} raw action rows...",
        flush=True,
    )
    dataset = InteractiveAgiBotWorldBetaDataset(
        root=concrete_roots,
        fps=args.fps,
        chunk_length=chunk_length,
        split_seed=args.split_seed,
        split_val_ratio=args.val_ratio,
        split=args.split,
        action_normalization=None,
        mode="forward_dynamics",
        viewpoint=args.viewpoint,
        pose_convention="backward_framewise",
        rotation_format="rot6d",
        max_loaded_datasets=args.max_loaded_datasets,
        sample_stride=args.sample_stride,
        enable_fast_init=args.fast_init,
        fast_init_max_workers=args.fast_init_max_workers,
        min_episode_length_frames=(args.min_episode_length_frames if args.min_episode_length_frames > 0 else None),
        use_subtask=args.use_subtask,
    )
    dataset._register_sources()  # noqa: SLF001 - deferred-init eval contract
    selected_indices = _select_indices(len(dataset), args.sample_count, args.indices)

    planned = _planned_paths(
        args.output_dir,
        len(selected_indices),
        write_video=not args.skip_ground_truth_video,
    )
    _check_collisions(planned, overwrite=args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for ordinal, dataset_index in enumerate(selected_indices):
        print(
            f"[{ordinal + 1}/{len(selected_indices)}] Loading dataset index {dataset_index}...",
            flush=True,
        )
        sample = dataset[dataset_index]
        payload, (height, width) = _prepare_interactive_payload(
            sample,
            dataset_index=dataset_index,
            num_frames=args.num_frames,
            expected_fps=args.fps,
        )
        if (height, width) != (args.expected_height, args.expected_width):
            print(
                "NOTICE: source video resolution is "
                f"{height}x{width}; interactive_seed_{ordinal:03d}.png will use "
                "the model's aspect-resize/center-crop preprocessing to produce "
                f"{args.expected_height}x{args.expected_width}.",
                file=sys.stderr,
            )

        npz_name = f"sample_{ordinal:03d}.npz"
        conditioning_name = f"conditioning_{ordinal:03d}.png"
        ground_truth_name = f"ground_truth_{ordinal:03d}.mp4"
        interactive_seed_name = f"interactive_seed_{ordinal:03d}.png"
        scene_name = f"agibot_scene_{ordinal:03d}.json"
        replay_name = f"replay_{ordinal:03d}.pt"

        _write_npz(args.output_dir / npz_name, payload)
        _write_conditioning_png(args.output_dir / conditioning_name, payload["video"])
        _write_interactive_seed(
            args.output_dir / interactive_seed_name,
            payload["video"],
            target_height=args.expected_height,
            target_width=args.expected_width,
        )
        chunk_count = _write_replay(
            args.output_dir / replay_name,
            payload["action"],
        )
        scene = _scene_payload(
            payload,
            seed_name=interactive_seed_name,
            prompt_override=args.scene_prompt,
            linear_velocity_m_s=args.linear_velocity_m_s,
            angular_velocity_rad_s=args.angular_velocity_rad_s,
        )
        _write_json(args.output_dir / scene_name, scene)
        if not args.skip_ground_truth_video:
            _write_ground_truth_video(
                args.output_dir / ground_truth_name,
                payload["video"],
                fps=payload["conditioning_fps"],
                quality=args.video_quality,
            )

        record = {"npz_path": npz_name, "num_frames": args.num_frames}
        records.append(record)
        summary_records.append(
            {
                **record,
                "source_dataset_index": dataset_index,
                "conditioning_image": conditioning_name,
                "ground_truth_video": (None if args.skip_ground_truth_video else ground_truth_name),
                "interactive_seed": interactive_seed_name,
                "scene": scene_name,
                "replay_actions": replay_name,
                "replay_chunks": chunk_count,
                "caption": payload["ai_caption"],
                "scene_prompt": scene["prompt"],
                "video_shape": list(payload["video"].shape),
                "action_shape": list(payload["action"].shape),
                "initial_gripper_right": payload["initial_gripper_right"],
                "initial_gripper_left": payload["initial_gripper_left"],
            }
        )

    jsonl_path = args.output_dir / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "schema": "cosmos-dreams-agibot-interactive-eval-v3",
        "imaginaire_root": str(imaginaire_root),
        "dataset_roots": concrete_roots,
        "dataset_length": len(dataset),
        "split": args.split,
        "split_seed": args.split_seed,
        "val_ratio": args.val_ratio,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "chunk_length": chunk_length,
        "action_steps_per_tick": _ACTION_STEPS_PER_TICK,
        "replay_chunks_per_sample": chunk_length // _ACTION_STEPS_PER_TICK,
        "temporal_compression_factor": _TEMPORAL_COMPRESSION_FACTOR,
        "latent_frames": ((args.num_frames - 1) // _TEMPORAL_COMPRESSION_FACTOR + 1),
        "action_normalization": None,
        "action_coordinate_version": _ACTION_COORDINATE_VERSION,
        "embodiment": _AGIBOT_EMBODIMENT,
        "domain_id": _AGIBOT_DOMAIN_ID,
        "interactive_resolution": [args.expected_height, args.expected_width],
        "controller_limits": {
            "linear_velocity_m_s": args.linear_velocity_m_s,
            "angular_velocity_rad_s": args.angular_velocity_rad_s,
        },
        "records": summary_records,
    }
    summary_path = args.output_dir / "conversion_summary.json"
    _write_json(summary_path, summary)

    duration = (args.num_frames - 1) / args.fps
    print(
        f"Wrote {len(records)} sample(s) to {args.output_dir}. Each rollout has "
        f"{args.num_frames} pixel frames ({summary['latent_frames']} latent "
        f"frames, {duration:.2f} seconds at {args.fps:g} fps) and "
        f"{summary['replay_chunks_per_sample']} interactive replay chunks."
    )
    print(f"Offline vLLM input: {jsonl_path}")
    if len(summary_records) == 1:
        only = summary_records[0]
        print(f"Interactive scene: {args.output_dir / only['scene']}")
        print(f"Interactive replay: {args.output_dir / only['replay_actions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
