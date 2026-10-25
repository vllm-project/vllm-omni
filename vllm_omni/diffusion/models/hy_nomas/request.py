# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import torch

from vllm_omni.diffusion.request import resolve_video_num_frames
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
from vllm_omni.model_extras.hy_nomas import HY_NOMAS_EXTRA_BODY_PARAMS, get_hy_nomas_video_generation_defaults


@dataclass(frozen=True)
class HyNomasRequest:
    prompts: list[str]
    boundaries: list[int]
    num_frames: int
    height: int
    width: int
    steps: int
    seed: int
    negative_prompt: str
    camera_trajectory: str | None
    segment_has_camera: list[bool] | None


def normalize_request(prompt: OmniPromptType, params: OmniDiffusionSamplingParams) -> HyNomasRequest:
    if not isinstance(prompt, (str, dict)):
        raise OmniClientError("Hy-Nomas requires a text prompt.")
    text = prompt if isinstance(prompt, str) else prompt.get("prompt")
    if isinstance(prompt, dict) and prompt.get("multi_modal_data"):
        raise OmniClientError("Hy-Nomas currently accepts text and camera trajectories, without input media.")
    if not isinstance(text, str) or not text.strip():
        raise OmniClientError("Hy-Nomas requires a nonempty text prompt.")
    prompts = [segment.strip() for segment in text.split("|||")]
    if not all(prompts):
        raise OmniClientError("Hy-Nomas prompt segments separated by ||| must be nonempty.")
    extra = params.extra_args
    unknown = extra.keys() - HY_NOMAS_EXTRA_BODY_PARAMS
    if unknown:
        raise OmniClientError(f"Unsupported Hy-Nomas extra_args: {sorted(unknown)}")
    if "session_id" in extra and (not isinstance(extra["session_id"], str) or not extra["session_id"].strip()):
        raise OmniClientError("Hy-Nomas session_id must be a nonempty string.")
    for name in ("reset", "close_session"):
        if name in extra and type(extra[name]) is not bool:
            raise OmniClientError(f"Hy-Nomas {name} must be a boolean.")
    try:
        defaults = get_hy_nomas_video_generation_defaults(extra)
    except (TypeError, ValueError) as exc:
        raise OmniClientError(str(exc)) from exc
    if extra.get("generate_sound") not in (None, False):
        raise OmniClientError("Hy-Nomas generates video without audio; generate_sound must be false.")
    if params.num_outputs_per_prompt != 1:
        raise OmniClientError("Hy-Nomas supports one output per prompt.")
    if params.output_type == "latent":
        raise OmniClientError("Hy-Nomas currently returns decoded video only.")
    if params.guidance_scale not in (None, 1.0) or params.guidance_scale_2 not in (None, 1.0):
        raise OmniClientError("The distilled Hy-Nomas checkpoint requires guidance_scale=1.")
    if extra.get("flow_shift") not in (None, 5.0):
        raise OmniClientError("Hy-Nomas uses the reference flow_shift=5.")
    if params.resolved_frame_rate not in (None, 24.0):
        raise OmniClientError("Hy-Nomas uses 24 fps for text and camera conditioning.")
    frames = resolve_video_num_frames(params.num_frames, default_num_frames=defaults.num_frames, is_dummy_run=False)
    if type(frames) is not int or frames < 5 or (frames - 1) % 4:
        raise OmniClientError("Hy-Nomas num_frames must be 4*n+1 with n >= 1.")
    if extra.get("duration") is not None and frames != defaults.num_frames:
        raise OmniClientError("Hy-Nomas num_frames conflicts with duration at 24 fps (four frame intervals plus one).")
    height = defaults.height if params.height is None else params.height
    width = defaults.width if params.width is None else params.width
    for name, value in (("height", height), ("width", width)):
        if type(value) is not int or value < 32 or value % 32:
            raise OmniClientError(f"Hy-Nomas {name} must be a positive multiple of 32.")
    steps = defaults.num_inference_steps if params.num_inference_steps is None else params.num_inference_steps
    if type(steps) is not int or steps not in (4, 6, 8):
        raise OmniClientError("Hy-Nomas distilled inference supports 4, 6, or 8 steps.")
    boundaries = extra.get("segment_frame_boundaries")
    if boundaries is None:
        if len(prompts) != 1:
            raise OmniClientError("Multiple Hy-Nomas segments require segment_frame_boundaries in pixel frames.")
        boundaries = [0, frames]
    if (
        not isinstance(boundaries, list)
        or len(boundaries) != len(prompts) + 1
        or any(type(value) is not int for value in boundaries)
        or boundaries[0] != 0
        or boundaries[-1] != frames
        or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
    ):
        raise OmniClientError(
            "segment_frame_boundaries must increase from 0 to num_frames, with one interval per segment."
        )
    if any(boundary % 4 for boundary in boundaries[1:-1]):
        raise OmniClientError("Internal segment boundaries must be multiples of 4 pixel frames.")
    camera = extra.get("camera_trajectory")
    if camera is not None and (not isinstance(camera, str) or not camera.strip()):
        raise OmniClientError("camera_trajectory must be a nonempty reference trajectory path or generator name.")
    camera_flags = extra.get("segment_has_camera")
    if camera_flags is not None and (
        not isinstance(camera_flags, list)
        or len(camera_flags) != len(prompts)
        or any(type(value) is not bool for value in camera_flags)
        or (any(camera_flags) and camera is None)
    ):
        raise OmniClientError(
            "segment_has_camera requires one boolean per segment and a trajectory for camera segments."
        )
    generator = params.generator
    if isinstance(generator, list):
        if len(generator) != 1:
            raise OmniClientError("Hy-Nomas accepts one generator per request.")
        generator = generator[0]
    if generator is not None and not isinstance(generator, torch.Generator):
        raise OmniClientError("Hy-Nomas generator must be a torch.Generator.")
    seed = generator.initial_seed() if generator is not None else (42 if params.seed is None else params.seed)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise OmniClientError("Hy-Nomas seed must be an integer in [0, 2**63).")
    negative = "" if isinstance(prompt, str) else prompt.get("negative_prompt", "")
    if negative is None:
        negative = ""
    if not isinstance(negative, str):
        raise OmniClientError("negative_prompt must be a string.")
    return HyNomasRequest(prompts, list(boundaries), frames, height, width, steps, seed, negative, camera, camera_flags)
