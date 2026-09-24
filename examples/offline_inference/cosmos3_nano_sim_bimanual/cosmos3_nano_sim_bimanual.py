# SPDX-License-Identifier: Apache-2.0
"""Run Cosmos3-Nano-Sim-Bimanual from reference JSONL/NPZ or action-sidecar and camera JSONL inputs."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, UnidentifiedImageError


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
    from vllm_omni.outputs import OmniRequestOutput

    if isinstance(output, list):
        output = output[0]
    if isinstance(output, OmniRequestOutput):
        if not output.images:
            raise ValueError("Cosmos3-Nano-Sim-Bimanual returned no video frames.")
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
            video = video.float().clamp(-1, 1) * 0.5 + 0.5
        video = video.numpy()
    array = np.asarray(video)
    if array.ndim == 5:
        array = array[0]
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / 255.0
    return [frame for frame in array]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Converted Cosmos3-Nano-Sim-Bimanual Diffusers directory.")
    parser.add_argument("--jsonl", type=Path, required=True, help="Input JSONL records.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-index", type=int, default=0)
    selection.add_argument("--all-samples", action="store_true")
    parser.add_argument("--input-format", choices=("reference", "cookbook"), default="reference")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/cosmos3_nano_sim_bimanual.yaml")
    parser.add_argument("--output", type=Path, default=Path("cosmos3_nano_sim_bimanual.mp4"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--output-type", choices=("video", "latent"), default="video")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--camera-trajectory")
    parser.add_argument("--camera-translation-scale", type=float)
    parser.add_argument("--camera-action-normalization", choices=("global_asinh", "scale"))
    parser.add_argument("--camera-pose-convention", choices=("backward_chunk_anchored_16f", "backward_framewise"))
    parser.add_argument("--camera-num-frames", type=int)
    args = parser.parse_args()
    if args.all_samples and args.output_dir is None:
        parser.error("--all-samples requires --output-dir")
    if args.input_format == "reference" and any(
        getattr(args, key) is not None
        for key in (
            "resolution",
            "camera_trajectory",
            "camera_translation_scale",
            "camera_action_normalization",
            "camera_pose_convention",
            "camera_num_frames",
        )
    ):
        parser.error("Resolution/camera recipe flags require --input-format cookbook")
    return args


def _reference_request(
    record: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prompt = str(record.get("prompt", record.get("ai_caption", record.get("text", ""))))
    image = _first_image(record, base_dir=args.jsonl.parent)
    action_value = record.get("action", record.get("actions"))
    action = None if action_value is None else torch.as_tensor(action_value, dtype=torch.float32)
    fps = float(
        args.fps
        if args.fps is not None
        else record.get("fps", record.get("conditioning_fps", record.get("frame_rate", 15.0)))
    )
    num_frames = (
        args.num_frames
        if args.num_frames is not None
        else int(record.get("num_frames", action.shape[0] + 1 if action is not None else 17))
    )
    prompt_data: dict[str, Any] = {"prompt": prompt}
    if image is not None:
        prompt_data["multi_modal_data"] = {"image": image}
    extra: dict[str, Any] = {}
    domain_name = record.get("domain_name", record.get("embodiment"))
    if domain_name is not None:
        extra["domain_name"] = str(domain_name)
    if record.get("domain_id") is not None:
        extra["domain_id"] = int(record["domain_id"])
    if action is not None:
        extra["action"] = action
    if "action_space" in record:
        extra["action_space"] = record["action_space"]
    params = {
        "height": args.height,
        "width": args.width,
        "num_frames": num_frames,
        "frame_rate": fps,
        "seed": args.seed if args.seed is not None else 42,
    }
    return prompt_data, {**params, "extra_args": extra}, {**params, "synthetic": True}


def _video_metadata(path: Path) -> dict[str, Any]:
    import imageio.v2 as imageio

    count = 0
    with imageio.get_reader(str(path)) as reader:
        fps = float(reader.get_meta_data()["fps"])
        height = width = 0
        for frame in reader:
            height, width = frame.shape[:2]
            count += 1
    return {"actual_num_frames": count, "actual_fps": fps, "actual_height": height, "actual_width": width}


def main() -> None:
    args = parse_args()
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    manifest = None
    inference_config = None
    if args.input_format == "cookbook":
        from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import Cosmos3NanoSimBimanualManifest

        config_path = Path(args.model) / "transformer" / "config.json"
        manifest = Cosmos3NanoSimBimanualManifest.from_od_config(
            SimpleNamespace(tf_model_config=json.loads(config_path.read_text(encoding="utf-8")))
        )
        manifest.require_exported_artifact()
        from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.inference_config import (
            Cosmos3NanoSimBimanualInferenceConfig,
        )

        deployment = yaml.safe_load(Path(args.deploy_config).read_text())
        inference_config = Cosmos3NanoSimBimanualInferenceConfig.from_od_config(
            SimpleNamespace(model_config=deployment["stages"][0].get("model_config", {})), manifest
        )
    count = sum(bool(line.strip()) for line in args.jsonl.read_text().splitlines())
    indexes = range(count) if args.all_samples else [args.sample_index]
    if count == 0:
        raise ValueError("The input JSONL is empty.")
    output_dir = args.output_dir or args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: list[dict[str, Any]] = []
    omni = None
    try:
        for index in indexes:
            status: dict[str, Any] = {"sample_index": index, "status": "failed"}
            started = time.perf_counter()
            try:
                record = _load_record(args.jsonl, index)
                status["name"] = str(record.get("name", f"sample_{index}"))
                if args.output_dir:
                    name = re.sub(r"[^A-Za-z0-9_.-]", "_", status["name"]).strip(".") or "sample"
                    output = output_dir / f"{index:04d}_{name}{'.pt' if args.output_type == 'latent' else '.mp4'}"
                else:
                    output = args.output
                status["output"] = str(output)
                if manifest is not None:
                    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.cookbook import prepare_cookbook_input

                    keys = (
                        "seed",
                        "fps",
                        "num_frames",
                        "height",
                        "width",
                        "resolution",
                        "camera_trajectory",
                        "camera_translation_scale",
                        "camera_action_normalization",
                        "camera_pose_convention",
                        "camera_num_frames",
                    )
                    prepared = prepare_cookbook_input(
                        record,
                        manifest=manifest,
                        base_dir=args.jsonl.parent,
                        cache_dir=output_dir / ".inputs",
                        overrides={key: getattr(args, key) for key in keys},
                    )
                    prompt_data = {"prompt": prepared.prompt}
                    if prepared.image is not None:
                        prompt_data["multi_modal_data"] = {"image": prepared.image}
                    params = {
                        "height": prepared.height,
                        "width": prepared.width,
                        "num_frames": prepared.num_frames,
                        "frame_rate": prepared.fps,
                        "seed": prepared.seed,
                        "extra_args": prepared.extra_args,
                    }
                    status.update(prepared.metadata)
                    status.pop("num_steps", None)
                    status.update(
                        window_frames=inference_config.window_frames,
                        sink_frames=inference_config.sink_frames,
                        history_mode=inference_config.history_mode,
                        sampler=manifest.sample_type,
                        frame_sigma_schedules=inference_config.frame_sigma_schedules,
                        num_steps_by_frame=[len(row) for row in inference_config.frame_sigma_schedules],
                        inference_id=inference_config.digest,
                    )
                    if prepared.poses is not None:
                        pose_path = output.with_suffix(".camera_trajectory.json")
                        pose_path.write_text(json.dumps(prepared.poses.tolist()) + "\n", encoding="utf-8")
                        status["camera_trajectory_path"] = str(pose_path)
                else:
                    prompt_data, params, metadata = _reference_request(record, args)
                    status.update(metadata)
                params["extra_args"].update(session_id=f"offline-{index}", reset=True, close_session=True)
                print(
                    json.dumps(
                        {
                            "sample_index": index,
                            "seed": params["seed"],
                            "num_frames": params["num_frames"],
                            "fps": params["frame_rate"],
                            "guidance_scale": 1.0,
                        }
                    )
                )
                if omni is None:
                    omni = Omni(
                        model=args.model,
                        model_class_name="Cosmos3NanoSimBimanualPipeline",
                        deploy_config=args.deploy_config,
                        enforce_eager=True,
                    )
                sampling_params = OmniDiffusionSamplingParams(
                    **params,
                    num_inference_steps=None,
                    guidance_scale=1.0,
                    output_type="latent" if args.output_type == "latent" else None,
                    generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(params["seed"]),
                )
                result = _unwrap_video(omni.generate(prompt_data, sampling_params))
                if args.output_type == "latent":
                    torch.save(result, output)
                    status["latent_shape"] = list(result.shape)
                else:
                    from diffusers.utils import export_to_video

                    export_to_video(_video_frames(result), str(output), fps=params["frame_rate"])
                    status.update(_video_metadata(output))
                    if status["actual_num_frames"] != params["num_frames"]:
                        raise ValueError("Output frame count does not match the effective request.")
                    if abs(status["actual_fps"] - params["frame_rate"]) > 0.01:
                        raise ValueError("Output FPS does not match the request.")
                    if params["height"] is not None and (status["actual_height"], status["actual_width"]) != (
                        params["height"],
                        params["width"],
                    ):
                        raise ValueError("Output dimensions do not match the request.")
                status["status"] = "success"
                print(f"Saved Cosmos3-Nano-Sim-Bimanual {args.output_type} output to {output}")
            except Exception as exc:
                status["error"] = f"{type(exc).__name__}: {exc}"
                print(f"Sample {index} failed: {status['error']}")
            finally:
                status["elapsed_seconds"] = time.perf_counter() - started
                statuses.append(status)
                (output_dir / "sample_outputs.json").write_text(json.dumps(statuses, indent=2) + "\n", encoding="utf-8")
    finally:
        if omni is not None:
            omni.close()
    if any(status["status"] != "success" for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
