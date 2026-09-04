# SPDX-License-Identifier: Apache-2.0
"""Run a Cosmos-Dreams sample from the reference JSONL/NPZ format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def _load_record(jsonl_path: Path, sample_index: int) -> dict[str, Any]:
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    if sample_index < 0 or sample_index >= len(records):
        raise IndexError(f"sample-index {sample_index} is outside a {len(records)}-record jsonl file.")
    record = records[sample_index]
    npz_path = record.get("npz_path") or record.get("data_path")
    if npz_path is not None:
        resolved = Path(npz_path)
        if not resolved.is_absolute():
            resolved = jsonl_path.parent / resolved
        if resolved.suffix.lower() != ".npz":
            raise ValueError(f"Expected an .npz data file, got {resolved}.")
        with np.load(resolved, allow_pickle=False) as archive:
            payload = {}
            for key in archive.files:
                value = archive[key]
                payload[key] = value.item() if value.ndim == 0 else value
        record = {**payload, **record}
    return record


def _first_image(record: dict[str, Any], *, base_dir: Path) -> Image.Image | None:
    value = record.get(
        "input_video",
        record.get("video", record.get("frames", record.get("image"))),
    )
    if value is None:
        return None
    if isinstance(value, str | Path):
        path = Path(value)
        if not path.is_absolute():
            path = base_dir / path
        try:
            return Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            import decord

            reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
            if len(reader) == 0:
                raise ValueError(f"Source video has no frames: {path}.")
            return Image.fromarray(reader[0].asnumpy()).convert("RGB")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4:
        if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.moveaxis(array[:, 0], 0, -1)
        else:
            array = array[0]
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if array.min() < 0 or array.max() > 1:
            array = array * 0.5 + 0.5
        array = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


def _unwrap_video(output: Any) -> Any:
    if isinstance(output, list):
        output = output[0]
    if isinstance(output, OmniRequestOutput):
        if not output.images:
            raise ValueError("Cosmos-Dreams returned no video frames.")
        return _unwrap_video(output.images)
    if isinstance(output, dict):
        return output.get("video", output.get("frames", output))
    return output


def _video_frames(video: Any) -> list[np.ndarray]:
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu()
        if video.ndim == 5:
            video = video[0]
        if video.ndim == 4 and video.shape[0] in (3, 4):
            video = video.permute(1, 2, 3, 0)
        if video.is_floating_point():
            video = video.clamp(-1, 1) * 0.5 + 0.5
        video = video.numpy()
    array = np.asarray(video)
    if array.ndim == 5:
        array = array[0]
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / 255.0
    return [frame for frame in array]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Converted Cosmos-Dreams Diffusers directory.")
    parser.add_argument("--jsonl", type=Path, required=True, help="Reference inference jsonl file.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/cosmos_dreams.yaml")
    parser.add_argument("--output", type=Path, default=Path("cosmos_dreams.mp4"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--output-type", choices=("video", "latent"), default="video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = _load_record(args.jsonl, args.sample_index)
    prompt = str(record.get("prompt", record.get("ai_caption", record.get("text", ""))))
    image = _first_image(record, base_dir=args.jsonl.parent)
    action_value = record.get("action", record.get("actions"))
    action = None if action_value is None else torch.as_tensor(action_value, dtype=torch.float32)
    fps = float(
        args.fps
        or record.get(
            "fps",
            record.get("conditioning_fps", record.get("frame_rate", 15.0)),
        )
    )
    num_frames = args.num_frames
    if num_frames is None:
        num_frames = int(record.get("num_frames", action.shape[0] + 1 if action is not None else 17))

    prompt_data: dict[str, Any] = {"prompt": prompt}
    if image is not None:
        prompt_data["multi_modal_data"] = {"image": image}
    extra_args: dict[str, Any] = {
        "session_id": f"offline-{args.sample_index}",
        "reset": True,
        "close_session": True,
        "domain_id": int(record.get("domain_id", 15)),
    }
    domain_name = record.get("domain_name", record.get("embodiment"))
    if domain_name is not None:
        extra_args["domain_name"] = str(domain_name)
    if action is not None:
        extra_args["action"] = action

    omni = Omni(
        model=args.model,
        model_class_name="CosmosDreamsPipeline",
        deploy_config=args.deploy_config,
        enforce_eager=True,
    )
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
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
    print(f"Saved Cosmos-Dreams {args.output_type} output to {args.output}")


if __name__ == "__main__":
    main()
