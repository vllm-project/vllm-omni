# SPDX-License-Identifier: Apache-2.0
"""Standalone JSONL inputs for action-sidecar and camera-conditioned inference."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from vllm_omni.diffusion.models.cosmos3.resolution import VIDEO_RES_SIZE_INFO, find_closest_target_size
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_inputs import validate_action_values
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.camera import camera_poses_to_actions, resolve_camera_poses
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import Cosmos3NanoSimBimanualManifest
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.geometry import Cosmos3NanoSimBimanualResolutionPolicy

_VIEWPOINTS = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": (
        "This video is captured from a third-person perspective looking towards the agent from the front."
    ),
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
    "top_down_2d_view": "This video is captured from a top-down view of a flat 2D scene.",
    "video_game_view": "This video is captured from an in-game camera perspective in a virtual environment.",
}


def resolve_asset(value: str, *, base_dir: Path, cache_dir: Path) -> Path:
    """Resolve local files and download HTTP assets; HF uses its normal token cache."""
    url = urllib.parse.urlsplit(value)
    if url.scheme in ("https", "http"):
        parts = url.path.strip("/").split("/")
        if url.scheme == "https" and url.netloc == "huggingface.co" and len(parts) >= 5 and parts[2] == "resolve":
            from huggingface_hub import hf_hub_download

            return Path(
                hf_hub_download(
                    repo_id="/".join(parts[:2]),
                    revision=urllib.parse.unquote(parts[3]),
                    filename=urllib.parse.unquote("/".join(parts[4:])),
                )
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        destination = cache_dir / (hashlib.sha256(value.encode()).hexdigest() + Path(url.path).suffix)
        if not destination.is_file():
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                with urllib.request.urlopen(value, timeout=60) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination
    if url.scheme:
        raise ValueError("Media/action paths must be local files or HTTP(S) URLs.")
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def first_media_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except UnidentifiedImageError:
        import decord

        reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
        if len(reader) == 0:
            raise ValueError("Conditioning video contains no frames.")
        return Image.fromarray(reader[0].asnumpy()).convert("RGB")


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def format_action_prompt(prompt: str, viewpoint: str, *, num_frames: int, fps: float, height: int, width: int) -> str:
    """Match the action cookbook's viewpoint, timing and resolution fields."""
    prompt = prompt.strip()
    if not prompt:
        return prompt
    if viewpoint not in _VIEWPOINTS:
        raise ValueError(f"Unknown view_point {viewpoint!r}.")
    duration = num_frames / fps
    seconds = round(duration)
    aspect = next(
        (key for sizes in VIDEO_RES_SIZE_INFO.values() for key, size in sizes.items() if size == (width, height)), None
    )
    if aspect is None:
        divisor = math.gcd(width, height)
        aspect = f"{width // divisor},{height // divisor}"
    return json.dumps(
        {
            "cinematography": {"framing": _VIEWPOINTS[viewpoint]},
            "actions": [
                {
                    "time": f"0:00-{seconds // 60}:{seconds % 60:02d}",
                    "description": prompt if prompt.endswith((".", "!", "?")) else prompt + ".",
                }
            ],
            "duration": f"{int(duration)}s",
            "fps": float(fps),
            "resolution": {"H": height, "W": width},
            "aspect_ratio": aspect,
        }
    )


@dataclass(frozen=True)
class CookbookInput:
    prompt: str
    image: Image.Image | None
    action: torch.Tensor
    extra_args: dict[str, Any]
    num_frames: int
    fps: float
    seed: int
    height: int
    width: int
    poses: np.ndarray | None
    metadata: dict[str, Any]


def prepare_cookbook_input(
    record: dict[str, Any],
    *,
    manifest: Cosmos3NanoSimBimanualManifest,
    base_dir: Path,
    cache_dir: Path,
    overrides: dict[str, Any] | None = None,
) -> CookbookInput:
    """Validate and translate one record, applying explicit CLI overrides first."""
    values = {**record, **{key: value for key, value in (overrides or {}).items() if value is not None}}
    mode = values.get("model_mode", "forward_dynamics")
    trajectory = values.get("camera_trajectory")
    camera = trajectory is not None
    if values.get("autoregressive") is False:
        raise ValueError("The Bimanual cookbook requires autoregressive inference.")
    if camera and any(values.get(key) is not None for key in ("action_path", "action", "actions")):
        raise ValueError("camera_trajectory and action inputs are mutually exclusive.")
    if mode not in (("image2video", "text2video", "forward_dynamics") if camera else ("forward_dynamics",)):
        raise ValueError("This adapter supports forward dynamics and camera-conditioned AR generation only.")
    if not camera and values.get("action_path") is None:
        raise ValueError("forward_dynamics requires action_path.")
    if not camera and any(values.get(key) is not None for key in values if key.startswith("camera_")):
        raise ValueError("Camera options require camera_trajectory.")
    domain = values.get("domain_name", "camera_pose" if camera else "agibotworld")
    schema = manifest.require_action_schema()
    embodiment = schema.resolve_embodiment(domain, values.get("domain_id"))
    contract = schema.embodiments[embodiment]
    if camera and embodiment != "camera_pose":
        raise ValueError("Camera trajectories require domain_name=camera_pose.")
    requested_frames = _integer(values.get("num_frames", 61 if camera else 901), "num_frames", 2)
    num_frames = (
        _integer(
            values.get("camera_num_frames") if values.get("camera_num_frames") is not None else requested_frames,
            "camera_num_frames",
            2,
        )
        if camera
        else requested_frames
    )
    if (num_frames - 1) % manifest.temporal_compression_factor:
        raise ValueError(
            f"num_frames - 1 must be divisible by {manifest.temporal_compression_factor}; got {num_frames}."
        )
    if not camera and _integer(values.get("action_chunk_size", num_frames - 1), "action_chunk_size") != num_frames - 1:
        raise ValueError("action_chunk_size must equal num_frames - 1.")
    fps = values.get("fps", 30)
    if isinstance(fps, bool) or not isinstance(fps, int | float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive.")
    seed = _integer(values.get("seed", 42), "seed", 0)
    poses = None
    camera_recipe = None
    if camera:
        convention = values.get("camera_pose_convention", "backward_chunk_anchored_16f")
        normalization = values.get("camera_action_normalization", "global_asinh")
        translation_scale = values.get("camera_translation_scale", 1.0)
        expected_method = {"global_asinh": "global_asinh", "scale": "pose_scale"}.get(normalization)
        if contract.layout.pose_convention != convention or contract.normalizer.method != expected_method:
            raise ValueError(
                "Camera recipe disagrees with the exported contract; re-export with the intended inference profile."
            )
        expected_scale = 1.0 if normalization == "global_asinh" else contract.normalizer.derivation.translation_scale
        if isinstance(translation_scale, bool) or translation_scale != expected_scale:
            raise ValueError(f"Camera translation scale must match the exported value {expected_scale}.")
        poses = resolve_camera_poses(trajectory, num_frames)
        action = validate_action_values(camera_poses_to_actions(poses, convention), width=9)
        action_space = "raw"
        camera_recipe = {
            "pose_convention": convention,
            "action_normalization": normalization,
            "translation_scale": float(translation_scale),
            "rotation_scale": 1.0 if normalization == "global_asinh" else contract.normalizer.derivation.rotation_scale,
            "camera_num_frames": num_frames,
            "normalizer_method": contract.normalizer.method,
            "normalizer_transform_sha256": contract.normalizer.transform_sha256,
        }
    else:
        action_path = resolve_asset(str(values["action_path"]), base_dir=base_dir, cache_dir=cache_dir)
        action = validate_action_values(
            json.loads(action_path.read_text(encoding="utf-8")), width=contract.raw_action_dim
        )
        if values.get("raw_action_dim", contract.raw_action_dim) != contract.raw_action_dim:
            raise ValueError("raw_action_dim disagrees with the exported embodiment width.")
        if len(action) != num_frames - 1:
            raise ValueError(f"Expected {num_frames - 1} action rows, got {len(action)}.")
        action_space = "model"
    if values.get("action_space", action_space) != action_space:
        raise ValueError(f"This cookbook input requires action_space={action_space!r}.")
    vision = values.get("vision_path")
    if vision is None and (not camera or mode == "image2video"):
        raise ValueError("This cookbook record requires vision_path.")
    image = first_media_image(resolve_asset(str(vision), base_dir=base_dir, cache_dir=cache_dir)) if vision else None
    height, width = values.get("height"), values.get("width")
    if (height is None) != (width is None):
        raise ValueError("height and width must be supplied together.")
    if height is None:
        source_w, source_h = image.size if image is not None else (832, 480)
        width, height = find_closest_target_size(
            source_h, source_w, values.get("resolution", values.get("image_size", 480))
        )
    geometry = Cosmos3NanoSimBimanualResolutionPolicy().resolve(height, width)
    prompt = str(values.get("prompt", "")).strip()
    prompt_templates = None
    if camera:
        # Apply default templates unless the record supplies an override.
        prompt_templates = {}
        for key, default in (
            ("duration_template", "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."),
            ("resolution_template", "This video is of {height}x{width} resolution."),
        ):
            template = values.get(key, default)
            prompt_templates[key] = template
            if template:
                prompt += template.format(duration=int(num_frames / fps), fps=fps, height=height, width=width)
    else:
        prompt = format_action_prompt(
            prompt, values.get("view_point", "ego_view"), num_frames=num_frames, fps=fps, height=height, width=width
        )
    metadata = {
        "requested_num_frames": requested_frames,
        "effective_num_frames": num_frames,
        "fps": fps,
        "height": height,
        "width": width,
        "seed": seed,
        "checkpoint_id": manifest.checkpoint_id,
        "checkpoint_hash": manifest.checkpoint_hash,
        "action_contract_sha256": schema.contract_sha256,
        "domain_name": embodiment,
        "action_space": action_space,
        "window_frames": manifest.window_frames,
        "sink_frames": manifest.sink_frames,
        "sampler": manifest.sampler_id,
        "num_steps": len(manifest.t_list),
        "guidance_scale": 1.0,
        "effective_camera_recipe": camera_recipe,
        "prompt_templates": prompt_templates,
        "effective_prompt": prompt,
        "inference_camera_profile": schema.inference_camera_profile.model_dump()
        if schema.inference_camera_profile
        else None,
        "synthetic": True,
    }
    return CookbookInput(
        prompt,
        image,
        action,
        {"action": action, "action_space": action_space, "domain_name": embodiment, "action_mode": "forward_dynamics"},
        num_frames,
        float(fps),
        seed,
        geometry.height,
        geometry.width,
        poses,
        metadata,
    )
