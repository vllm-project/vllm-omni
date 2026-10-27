# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from dataclasses import dataclass
from typing import Any

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
    if negative:
        raise OmniClientError("The distilled Hy-Nomas checkpoint does not support negative_prompt (guidance_scale=1).")
    return HyNomasRequest(prompts, list(boundaries), frames, height, width, steps, seed, camera, camera_flags)


@dataclass(frozen=True)
class HyNomasChunk:
    segment: int
    start: int
    end: int
    pixel_frames: int


def plan_chunks(request: HyNomasRequest, chunk_latents: int) -> list[HyNomasChunk]:
    boundaries = [value // 4 for value in request.boundaries[:-1]] + [(request.num_frames - 1) // 4 + 1]
    chunks = []
    for segment, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        for left in range(start, end, chunk_latents):
            right = min(left + chunk_latents, end)
            chunks.append(HyNomasChunk(segment, left, right, min(request.num_frames, 4 * right) - 4 * left))
    return chunks


def action_conditioning(
    prompt: str, text_encoder: Any, text_length: int, segment_frames: int
) -> tuple[int, list[int] | None, list[list[int]] | None]:
    # Plain captions have no structured action mask.
    try:
        obj = json.loads(prompt)
    except json.JSONDecodeError:
        return -1, None, None
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list) or not segments or not isinstance(segments[0], dict):
        return -1, None, None
    segment = segments[0]
    actions = segment.get("contained_actions")
    if not isinstance(actions, (dict, list)) or not actions:
        return -1, None, None

    inputs = text_encoder.text2tokens([prompt], data_type="video", max_length=text_length)
    full_ids = inputs["input_ids"][0].tolist()
    template = text_encoder.prompt_template_video
    crop = max(0, int(template.get("crop_start", 0) or 0)) if isinstance(template, dict) else 0
    offsets = []
    for count in range(len(actions)):
        truncated = dict(segment)
        if count == 0:
            truncated.pop("contained_actions")
        else:
            truncated["contained_actions"] = (
                dict(list(actions.items())[:count]) if isinstance(actions, dict) else actions[:count]
            )
        variant = json.dumps(
            {key: [truncated] if key == "segments" else value for key, value in obj.items()}, ensure_ascii=False
        )
        ids = text_encoder.text2tokens([variant], data_type="video", max_length=text_length)["input_ids"][0].tolist()
        common = 0
        for left, right in zip(full_ids, ids):
            if left != right:
                break
            common += 1
        offsets.append(max(offsets[-1] if offsets else 0, min(text_length - 1, max(0, common - crop))))
    start = offsets[0]
    mask = inputs.get("attention_mask")
    offsets.append(min(int(mask[0].sum()) if mask is not None else text_length, text_length))

    from hymm.inference import _parse_timestamp

    action_list = list(actions.values()) if isinstance(actions, dict) else actions
    spans = []
    for index, action in enumerate(action_list):
        if not isinstance(action, dict) or "start_time" not in action or "end_time" not in action:
            return start, None, None
        left = min(segment_frames, int(max(0, _parse_timestamp(action["start_time"])) * 24) // 4)
        right = min(segment_frames, int(max(0, _parse_timestamp(action["end_time"])) * 24) // 4)
        if right > left:
            spans.append([left, right, index])
    if not spans:
        return start, None, None
    spans.sort()
    windows = []
    cursor = 0
    for left, right, index in spans:
        if left < cursor:
            raise OmniClientError("Hy-Nomas action video windows must not overlap.")
        if left > cursor:
            windows.append([cursor, left, -1])
        windows.append([left, right, index])
        cursor = right
    if cursor < segment_frames:
        windows.append([cursor, segment_frames, -1])
    if len(windows) == 1 and windows[0][2] == 0 and len(actions) == 1:
        return start, None, None
    return start, offsets, windows


def prepare_camera(
    transformer: Any,
    config: Any,
    trajectory: str | None,
    *,
    num_frames: int,
    latent_h: int,
    latent_w: int,
    boundaries: list[int] | None,
    device: torch.device,
) -> torch.Tensor | None:
    if trajectory is None:
        return None
    if transformer.camera_injection_mode != "extrinsics_attn":
        raise ValueError("Hy-Nomas camera control requires extrinsics_attn.")
    from hymm.pipeline.bidirectional import prepare_camera_inputs

    inference = config.inference
    camera = prepare_camera_inputs(
        trajectory_source=trajectory,
        num_frames=num_frames,
        device=device,
        dtype=torch.bfloat16,
        normalize_relative_to_first_frame=inference.normalize_relative_to_first_frame,
        normalize_framewise=inference.normalize_framewise,
        normalize_displacement=inference.normalize_displacement,
        normalize_displacement_per_video=bool(config.get("normalize_displacement_per_video", False)),
        segment_boundaries_pixel=boundaries,
        translation_scale=float(inference.trajectory_translation_scale),
        rotation_scale=float(inference.trajectory_rotation_scale),
    )
    offsets = camera.get("turn_camera_offsets", [0, camera["num_frames"]])
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        embeddings = [
            transformer._preprocess_camera(
                norm_extrinsics=camera["norm_extrinsics"][:, start:end],
                norm_intrinsics=camera["norm_intrinsics"][:, start:end],
                action_onehot=camera["action_onehot"][:, start:end],
                oh=latent_h,
                ow=latent_w,
                img_dtype=torch.bfloat16,
                device=device,
            )
            for start, end in zip(offsets, offsets[1:])
        ]
    result = torch.cat(embeddings, dim=1)
    expected = (num_frames - 1) // 4 + 1
    if result.shape[1] != expected:
        raise OmniClientError(f"Hy-Nomas camera trajectory has {result.shape[1]} latent frames; expected {expected}.")
    return result
