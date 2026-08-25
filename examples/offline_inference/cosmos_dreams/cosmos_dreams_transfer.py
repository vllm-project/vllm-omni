# SPDX-License-Identifier: Apache-2.0
"""Run Cosmos-Dreams-Transfer from an imaginaire4 Transfer JSON record."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

TRANSFER_HINTS = ("edge", "blur", "depth", "seg")


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _resolve_num_frames(cli_value: int | None, record: dict[str, Any]) -> int:
    value = _first_not_none(cli_value, record.get("num_frames"), 97)
    try:
        num_frames = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Transfer num_frames must be an integer, got {value!r}.") from exc
    if num_frames <= 0:
        raise ValueError(f"Transfer num_frames must be positive, got {num_frames}.")
    return num_frames


def _resolve_fps(cli_value: float | None, record: dict[str, Any]) -> float:
    value = _first_not_none(cli_value, record.get("fps"), record.get("frame_rate"), 15.0)
    try:
        fps = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Transfer FPS must be numeric, got {value!r}.") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Transfer FPS must be positive and finite, got {fps}.")
    return fps


def _resolve_path(value: Any, base_dir: Path) -> Any:
    if not isinstance(value, str) or "://" in value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else base_dir / path)


def _load_video_frames(value: str, *, max_frames: int) -> list[Image.Image]:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError("The Transfer example requires imageio[ffmpeg] to read vision_path.") from exc
    frames: list[Image.Image] = []
    for frame in iio.imiter(value):
        frames.append(Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8), mode="RGB"))
        if len(frames) >= max_frames:
            break
    if len(frames) != max_frames:
        raise ValueError(f"vision_path must provide at least {max_frames} frames, got {len(frames)}.")
    return frames


def _unwrap_video(output: Any) -> Any:
    seen: set[int] = set()
    while isinstance(output, list | OmniRequestOutput | dict):
        identity = id(output)
        if identity in seen:
            raise ValueError("Cosmos-Dreams-Transfer returned a cyclic video output envelope.")
        seen.add(identity)

        if isinstance(output, list):
            if not output:
                raise ValueError("Cosmos-Dreams-Transfer returned an empty video output list.")
            output = output[0]
        elif isinstance(output, OmniRequestOutput):
            if not output.images:
                raise ValueError("Cosmos-Dreams-Transfer returned no video frames.")
            output = output.images
        elif "video" in output:
            output = output["video"]
        elif "payload" in output:
            output = output["payload"]
        else:
            raise ValueError(
                "Cosmos-Dreams-Transfer returned an unsupported output mapping; "
                f"expected 'video' or 'payload', got keys {sorted(map(str, output))}."
            )
        if output is None:
            raise ValueError("Cosmos-Dreams-Transfer returned an empty video payload.")
    return output


def _video_frames(video: Any) -> list[np.ndarray]:
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu()
        if video.ndim == 5:
            video = video[0]
        if video.ndim == 4 and video.shape[0] in (3, 4):
            video = video[:3].permute(1, 2, 3, 0)
        if video.is_floating_point():
            video = video.clamp(-1, 1).mul(0.5).add(0.5)
        video = video.numpy()
    array = np.asarray(video)
    if array.ndim == 5:
        array = array[0]
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / 255.0
    return [frame for frame in array]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Converted Cosmos-Dreams-Transfer Diffusers directory.")
    parser.add_argument("--input-json", type=Path, required=True, help="imaginaire4 Transfer JSON record.")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/cosmos_dreams_transfer.yaml")
    parser.add_argument("--output", type=Path, default=Path("cosmos_dreams_transfer.mp4"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--output-type", choices=("video", "latent"), default="video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = json.loads(args.input_json.read_text())
    if not isinstance(record, dict):
        raise TypeError("Transfer input JSON must contain one object.")
    selected_hints = [hint for hint in TRANSFER_HINTS if record.get(hint) is not None]
    if len(selected_hints) != 1:
        raise ValueError(f"Transfer input JSON must select exactly one of {list(TRANSFER_HINTS)}.")
    hint = selected_hints[0]
    hint_config = record[hint]
    if hint_config is True:
        hint_config = {}
    if not isinstance(hint_config, dict):
        raise TypeError(f"Transfer hint {hint!r} must be an object or true.")
    hint_config = dict(hint_config)
    if hint_config.get("control_path") is not None:
        hint_config["control_path"] = _resolve_path(hint_config["control_path"], args.input_json.parent)

    num_frames = _resolve_num_frames(args.num_frames, record)
    fps = _resolve_fps(args.fps, record)
    prompt_data: dict[str, Any] = {"prompt": str(record.get("prompt", ""))}
    vision_path = record.get("vision_path")
    if vision_path is not None:
        resolved_vision = _resolve_path(vision_path, args.input_json.parent)
        prompt_data["multi_modal_data"] = {
            "video": _load_video_frames(str(resolved_vision), max_frames=num_frames),
        }

    extra_args = {
        "session_id": f"transfer-{args.input_json.stem}",
        "reset": True,
        "close_session": True,
        "resolution": "480",
        hint: hint_config,
    }
    omni = Omni(
        model=args.model,
        model_class_name="CosmosDreamsTransferPipeline",
        deploy_config=args.deploy_config,
        enforce_eager=True,
    )
    sampling_params = OmniDiffusionSamplingParams(
        height=480,
        width=832,
        num_frames=num_frames,
        num_inference_steps=4,
        guidance_scale=1.0,
        frame_rate=fps,
        seed=args.seed,
        output_type="latent" if args.output_type == "latent" else None,
        generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed),
        extra_args=extra_args,
    )
    result = _unwrap_video(omni.generate(prompt_data, sampling_params))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output_type == "latent":
        torch.save(result, args.output)
    else:
        from diffusers.utils import export_to_video

        export_to_video(_video_frames(result), str(args.output), fps=fps)
    print(f"Saved Cosmos-Dreams-Transfer {args.output_type} output to {args.output}")


if __name__ == "__main__":
    main()
