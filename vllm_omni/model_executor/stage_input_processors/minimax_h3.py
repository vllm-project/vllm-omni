# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 text-encoder stage input and output adapters."""

from __future__ import annotations

import copy
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

from vllm_omni.data_entry_keys import REQUEST_ARTIFACT_DIRS_KEY
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_align_frame_count
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MINIMAX_H3_CONDITION_LABELS_KEY,
    MINIMAX_H3_PRESENTATION_TASK_KEY,
    MiniMaxH3TextConditioning,
)
from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
    load_minimax_h3_images,
    resolve_minimax_h3_reference_image_shape,
    resolve_minimax_h3_target_canvas,
)
from vllm_omni.model_executor.models.minimax_h3.reference_video import (
    MINIMAX_H3_PREPARED_REFERENCE_VIDEOS_KEY,
    MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS,
    prepare_reference_videos,
    sample_reference_video_frames,
    serialize_prepared_reference_videos,
)
from vllm_omni.model_executor.models.minimax_h3.timeline_guides import GUIDES_EXTRA_KEY


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) and not (len(value) == 2 and isinstance(value[1], Mapping)):
        return list(value)
    return [value]


def _audio_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], (int, np.integer)):
        return [value]
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _resolve_task(
    extra_args: Mapping[str, Any],
    multi_modal_data: Mapping[str, Any],
) -> str:
    requested = extra_args.get("task")
    if requested is not None:
        return str(requested).lower()
    if multi_modal_data.get("video") is not None or multi_modal_data.get("audio") is not None:
        return "ref2va"
    if multi_modal_data.get("image") is not None:
        return "fl2va"
    return "t2va"


def _diffusion_sampling_params(sampling_params_list: Sequence[Any]) -> Any:
    diffusion_params = [
        sampling_params
        for sampling_params in sampling_params_list
        if isinstance(sampling_params, OmniDiffusionSamplingParams)
    ]
    if len(diffusion_params) != 1:
        raise RuntimeError(
            "MiniMax H3 text encoding requires exactly one OmniDiffusionSamplingParams stage parameter, "
            f"got {len(diffusion_params)}"
        )
    return diffusion_params[0]


def _ref2va_target_frame_count(sampling_params_list: Sequence[Any]) -> int:
    sampling = _diffusion_sampling_params(sampling_params_list)
    extra_args = sampling.extra_args or {}
    target = extra_args.get("target")
    target = target if isinstance(target, Mapping) else {}
    duration = target.get("duration_seconds", extra_args.get("duration_seconds", extra_args.get("duration")))
    if duration is not None:
        requested = int(round(float(duration) * 24))
    elif int(getattr(sampling, "num_frames", None) or 1) > 1:
        requested = int(sampling.num_frames)
    else:
        requested = 124
    return minimax_h3_align_frame_count(requested)


def _prepare_qwen_images(
    task: str,
    values: list[Any],
    sampling_params_list: Sequence[Any],
) -> list[Any]:
    if not values:
        return []
    images = load_minimax_h3_images(values)
    if task not in ("fl2va", "ref2va"):
        return images

    sampling = _diffusion_sampling_params(sampling_params_list)
    extra_args = sampling.extra_args or {}
    if task == "ref2va":
        # Same guided predicate as the diffusion pipeline's request preparation.
        if extra_args.get(GUIDES_EXTRA_KEY, []) == []:
            return [
                image.resize(
                    resolve_minimax_h3_reference_image_shape(image),
                    Image.Resampling.LANCZOS,
                )
                for image in images
            ]
        # Guided Ref2VA: resolve the very same target canvas the diffusion
        # pipeline resolves, then apply the identical down-only reference
        # policy, so the Qwen presentation and the VAE reference agree.
        target_height, target_width = resolve_minimax_h3_target_canvas(
            task,
            height=sampling.height,
            width=sampling.width,
            extra_args=extra_args,
            image=images[0],
        )
        return [
            image.resize(
                resolve_minimax_h3_reference_image_shape(image, target=(target_width, target_height)),
                Image.Resampling.LANCZOS,
            )
            for image in images
        ]

    height, width = resolve_minimax_h3_target_canvas(
        task,
        height=sampling.height,
        width=sampling.width,
        extra_args=extra_args,
        image=images[0],
    )
    return [image.resize((width, height), Image.Resampling.LANCZOS) for image in images]


def prepare_text_encoder_prompt(
    prompt: Any,
    sampling_params_list: Sequence[Any],
) -> Any:
    """Build H3's labeled Qwen3-VL presentation for Stage 0.

    The upstream Qwen3-VL multimodal processor expands each image/video
    placeholder into the exact number of vision tokens and adds timestamped
    video blocks.  Audio is represented only by its H3 text label and is not
    sent to Qwen3-VL.
    """
    # Reject request-visible incompatibilities before split-stage Qwen work.
    # Startup cache, fused adapters and resolved attention roles are checked by
    # the diffusion pipeline, which owns those model instances/configurations.
    for sampling in sampling_params_list:
        if not isinstance(sampling, OmniDiffusionSamplingParams) or not (sampling.extra_args or {}).get(
            GUIDES_EXTRA_KEY
        ):
            continue
        if sampling.quality == "high":
            raise OmniClientError("MiniMax H3 timeline guides require cache-free execution; use quality=lossless")
        if sampling.lora_request is not None and float(sampling.lora_scale) != 0.0:
            raise OmniClientError("MiniMax H3 timeline guides do not support active LoRA/Turbo adapters")
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, dict):
        raise TypeError(f"MiniMax H3 expects a string or dict prompt, got {type(prompt)!r}")

    prompt = copy.copy(prompt)
    additional_information = dict(prompt.get("additional_information") or {})
    meta = dict(additional_information.get("meta") or {})
    meta.pop(MINIMAX_H3_PREPARED_REFERENCE_VIDEOS_KEY, None)
    additional_information["meta"] = meta
    prompt["additional_information"] = additional_information

    text = str(prompt.get("prompt") or "")
    if not text:
        raise OmniClientError("MiniMax H3 requires a non-empty prompt")
    multi_modal_data = prompt.get("multi_modal_data") or {}
    if not isinstance(multi_modal_data, Mapping):
        raise TypeError("multi_modal_data must be a mapping")

    image_values = _items(multi_modal_data.get("image"))
    videos = _items(multi_modal_data.get("video"))
    audios = _audio_items(multi_modal_data.get("audio"))
    diffusion_sampling = _diffusion_sampling_params(sampling_params_list)
    extra_args = diffusion_sampling.extra_args or {}
    task = _resolve_task(extra_args, multi_modal_data)
    images = _prepare_qwen_images(task, image_values, sampling_params_list)
    qwen_video_inputs: list[tuple[np.ndarray, dict[str, Any]]] = []
    condition_labels: list[tuple[str, int]] = []

    if task == "t2va":
        if images or videos or audios:
            raise OmniClientError("t2va does not accept image, video, or audio conditions")
    elif task == "fl2va":
        if not images or videos or audios:
            raise OmniClientError("fl2va requires image conditions only")
        condition_labels.extend(("image", index) for index in range(1, len(images) + 1))
    elif task == "ref2va":
        if not images and not videos:
            raise OmniClientError("ref2va requires an image or video condition")
        condition_labels.extend(("image", index) for index in range(1, len(images) + 1))
        prepared_videos: list[dict[str, Any]] = []
        artifact_dir: str | None = None
        if videos:
            artifact_dir = tempfile.mkdtemp(prefix="minimax_h3_ref2va_")
            try:
                prepared_videos = prepare_reference_videos(
                    videos,
                    target_frame_count=_ref2va_target_frame_count(sampling_params_list),
                    workdir=artifact_dir,
                    start_time_seconds=extra_args.get("start_time_seconds"),
                )
                for item in prepared_videos:
                    sampled = sample_reference_video_frames(item["prepared_path"])
                    frames = np.stack(sampled["frames"])
                    frame_count = int(frames.shape[0])
                    qwen_video_inputs.append(
                        (
                            frames,
                            {
                                "total_num_frames": frame_count,
                                "fps": MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS,
                                "duration": frame_count / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS,
                                "video_backend": "minimax_h3",
                                "frames_indices": list(range(frame_count)),
                                "do_sample_frames": False,
                            },
                        )
                    )
            except BaseException:
                shutil.rmtree(artifact_dir, ignore_errors=True)
                raise
        audio_index = 0
        for video_index, item in enumerate(prepared_videos, start=1):
            if item["input_has_audio"]:
                audio_index += 1
                condition_labels.append(("audio", audio_index))
            condition_labels.append(("video", video_index))
        for _ in audios:
            audio_index += 1
            condition_labels.append(("audio", audio_index))
    else:
        raise OmniClientError(f"unsupported MiniMax H3 task {task!r}")

    transformed = copy.copy(prompt)
    if isinstance(prompt.get("additional_information"), Mapping):
        transformed["additional_information"] = dict(prompt["additional_information"])
    transformed["prompt"] = text
    if task == "ref2va" and prepared_videos and artifact_dir is not None:
        additional_information = dict(transformed.get("additional_information") or {})
        meta = dict(additional_information.get("meta") or {})
        meta[MINIMAX_H3_PREPARED_REFERENCE_VIDEOS_KEY] = serialize_prepared_reference_videos(
            prepared_videos,
            artifact_dir,
        )
        additional_information["meta"] = meta
        transformed["additional_information"] = additional_information
        transformed[REQUEST_ARTIFACT_DIRS_KEY] = [artifact_dir]
    qwen_mm_data = dict(multi_modal_data)
    qwen_mm_data.pop("audio", None)
    if images:
        qwen_mm_data["image"] = images
    if qwen_video_inputs:
        qwen_mm_data["video"] = qwen_video_inputs
    transformed["multi_modal_data"] = qwen_mm_data or None

    mm_processor_kwargs = dict(prompt.get("mm_processor_kwargs") or {})
    mm_processor_kwargs[MINIMAX_H3_PRESENTATION_TASK_KEY] = task
    mm_processor_kwargs[MINIMAX_H3_CONDITION_LABELS_KEY] = condition_labels
    transformed["mm_processor_kwargs"] = mm_processor_kwargs
    return transformed


def _original_prompt(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else {}
    if isinstance(prompt, dict):
        return copy.copy(prompt)
    if isinstance(prompt, str):
        return {"prompt": prompt}
    raise TypeError(f"invalid MiniMax H3 prompt type {type(prompt)!r}")


def _global_request_id(prompt: Mapping[str, Any]) -> str | None:
    additional_information = prompt.get("additional_information")
    if not isinstance(additional_information, Mapping):
        return None
    value = additional_information.get("global_request_id")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return str(value) if value is not None else None


def text_encoder2diffusion(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> dict[str, Any] | None:
    """Attach Stage 0 hidden states and token tags to the original request."""
    del requires_multimodal_data, streaming_context
    if not source_outputs:
        return None
    if len(source_outputs) != 1:
        raise RuntimeError(f"MiniMax H3 diffusion requires exactly one text-encoder source, got {len(source_outputs)}")

    diffusion_prompt = _original_prompt(prompt)
    source_output = source_outputs[0]
    source_request_id = getattr(source_output, "request_id", None)
    expected_request_id = _global_request_id(diffusion_prompt)
    if (
        source_request_id is not None
        and expected_request_id is not None
        and str(source_request_id) != expected_request_id
    ):
        raise RuntimeError(
            "MiniMax H3 text-encoder request ID does not match the diffusion request: "
            f"source={source_request_id!r}, expected={expected_request_id!r}"
        )

    outputs = getattr(source_output, "outputs", None)
    if not isinstance(outputs, list) or len(outputs) != 1:
        output_count = len(outputs) if isinstance(outputs, list) else 0
        raise RuntimeError(f"MiniMax H3 text encoder must return exactly one completion, got {output_count}")

    completion = outputs[0]
    payload = completion.multimodal_output
    if not isinstance(payload, Mapping):
        raise RuntimeError("MiniMax H3 text encoder returned no conditioning payload")
    try:
        conditioning = MiniMaxH3TextConditioning.from_omni_payload(payload)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    additional_information = dict(diffusion_prompt.get("additional_information") or {})
    additional_information["text_encoder_output"] = conditioning.to_payload()
    diffusion_prompt["additional_information"] = additional_information
    return diffusion_prompt
