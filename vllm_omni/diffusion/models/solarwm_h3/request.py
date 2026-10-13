# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public image/prompt/camera request contract for SolarWM-H3 Stage2."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from .camera import CameraCondition, prepare_camera

CAMERA_SCHEMA = "solarwm.camera_trajectory.v1"


@dataclass(frozen=True)
class SolarRequest:
    prompt: str
    image: Image.Image
    image_digest: str
    camera: CameraCondition
    height: int
    width: int
    num_frames: int
    fps: float
    seed: int
    output_type: str
    tick: ARDiffusionTickRequest | None


def parse_request(prompt: object, sampling: OmniDiffusionSamplingParams, *, height: int, width: int) -> SolarRequest:
    if not isinstance(prompt, Mapping) or not isinstance(prompt.get("prompt"), str):
        raise ValueError("SolarWM-H3 requires a prompt mapping with a string 'prompt'")
    text = prompt["prompt"]
    if not text.strip():
        raise ValueError("SolarWM-H3 prompt must be nonempty")
    if getattr(sampling, "num_outputs_per_prompt", 1) != 1:
        raise ValueError("SolarWM-H3 requires num_outputs_per_prompt=1")
    if getattr(sampling, "num_inference_steps", None) not in (None, 4):
        raise ValueError("SolarWM-H3 Stage2 requires num_inference_steps=4")
    if any(getattr(sampling, field, None) is not None for field in ("timesteps", "sigmas")):
        raise ValueError("SolarWM-H3 Stage2 uses a fixed schedule; custom timesteps and sigmas are unsupported")
    if getattr(sampling, "guidance_scale", None) not in (None, 1, 1.0):
        raise ValueError("SolarWM-H3 Stage2 requires guidance_scale=1")
    if prompt.get("negative_prompt") or getattr(sampling, "negative_prompt", None):
        raise ValueError("SolarWM-H3 Stage2 does not use a negative prompt")
    requested_height = getattr(sampling, "height", None)
    requested_width = getattr(sampling, "width", None)
    requested_height = height if requested_height is None else requested_height
    requested_width = width if requested_width is None else requested_width
    if (requested_height, requested_width) != (height, width):
        raise ValueError(f"SolarWM-H3 request must match fixed cache resolution {height}x{width}")
    num_frames = getattr(sampling, "num_frames", None)
    num_frames = 158 if num_frames in (None, 1) else num_frames
    fps = getattr(sampling, "fps", None)
    fps = 24.0 if fps is None else fps
    if isinstance(fps, bool) or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("SolarWM-H3 fps must be finite and positive")
    output_type = getattr(sampling, "output_type", None) or "pil"
    if output_type not in ("pil", "np", "pt", "latent"):
        raise ValueError("SolarWM-H3 output_type must be pil, np, pt or latent")
    extra = getattr(sampling, "extra_args", None) or {}
    if not isinstance(extra, Mapping):
        raise ValueError("SolarWM-H3 extra_args must be a mapping")
    for key, expected in (("flow_shift", 12.0), ("audio_flow_shift", 3.0)):
        if key in extra and float(extra[key]) != expected:
            raise ValueError(f"SolarWM-H3 Stage2 requires {key}={expected:g}")
    tick = ARDiffusionTickRequest.from_extra_args(extra)
    if tick is None:
        camera = extra.get("camera")
    else:
        if tick.prompt is not None and tick.prompt != text:
            raise ValueError("SolarWM-H3 tick prompt must match the standard request prompt")
        controls = [control for control in tick.controls if control.track == "camera"]
        if len(controls) != 1 or len(tick.controls) != 1:
            raise ValueError("SolarWM-H3 ticks require exactly one camera control")
        control = controls[0]
        if control.schema != CAMERA_SCHEMA or "schema" in control.data:
            raise ValueError(f"SolarWM-H3 camera control requires schema '{CAMERA_SCHEMA}' outside its data")
        camera = {"schema": control.schema, **control.data}
    if not isinstance(camera, Mapping) or camera.get("schema") != CAMERA_SCHEMA:
        raise ValueError(f"SolarWM-H3 requires extra_args.camera with schema '{CAMERA_SCHEMA}'")
    if "c2w" not in camera or "intrinsics" not in camera:
        raise ValueError("SolarWM-H3 camera requires absolute c2w poses and intrinsics")
    if set(camera) - {"schema", "c2w", "intrinsics", "intrinsics_normalized"}:
        raise ValueError("SolarWM-H3 camera contains unknown fields")
    normalized = camera.get("intrinsics_normalized", False)
    if type(normalized) is not bool:
        raise ValueError("SolarWM-H3 intrinsics_normalized must be a boolean")
    prepared = prepare_camera(
        camera["c2w"],
        camera["intrinsics"],
        num_frames=num_frames,
        height=height,
        width=width,
        intrinsics_normalized=normalized,
        device="cpu",
    )
    if prepared.viewmats.shape[0] != 1:
        raise ValueError("SolarWM-H3 supports one camera trajectory per request")
    multimodal = prompt.get("multi_modal_data")
    if not isinstance(multimodal, Mapping) or set(multimodal) != {"image"}:
        raise ValueError("SolarWM-H3 requires exactly one multi_modal_data.image")
    image = multimodal["image"]
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            image = ImageOps.exif_transpose(source).convert("RGB").copy()
    elif isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image).convert("RGB").copy()
    else:
        raise ValueError("SolarWM-H3 image must be a PIL image or local file path")
    # Solar's inference boundary resizes the initial frame to the output grid.
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image_digest = hashlib.sha256(image.tobytes()).hexdigest()
    generator = getattr(sampling, "generator", None)
    if isinstance(generator, list):
        raise ValueError("SolarWM-H3 accepts one generator, not a generator list")
    seed = getattr(sampling, "seed", None)
    if seed is None:
        seed = generator.initial_seed() if generator is not None else 0
    return SolarRequest(
        text, image, image_digest, prepared, height, width, num_frames, float(fps), int(seed), output_type, tick
    )
