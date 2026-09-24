# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapters between MiniMax H3 conditioning and the Omni stage runner."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from vllm_omni.data_entry_keys import OmniPayloadStruct, to_dict, unflatten_payload
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MINIMAX_H3_CONDITION_LABELS_KEY,
    MINIMAX_H3_ENCODER_REQUEST_KEY,
    MINIMAX_H3_PRESENTATION_TASK_KEY,
    MiniMaxH3EncoderConditioning,
)
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import prepare_encoder_inputs


def _diffusion_sampling_params(sampling_params_list: Sequence[Any]) -> Any:
    diffusion_params = [
        sampling_params
        for sampling_params in sampling_params_list
        if isinstance(sampling_params, OmniDiffusionSamplingParams)
    ]
    if len(diffusion_params) != 1:
        raise RuntimeError(
            "MiniMax H3 encoding requires exactly one OmniDiffusionSamplingParams stage parameter, "
            f"got {len(diffusion_params)}"
        )
    return diffusion_params[0]


def prepare_encoder_prompt(
    prompt: Any,
    sampling_params_list: Sequence[Any],
) -> Any:
    prepared = prepare_encoder_inputs(prompt, _diffusion_sampling_params(sampling_params_list))
    if isinstance(prompt, str):
        prompt = {"prompt": prompt}
    text = prepared.prompt
    images = prepared.images
    qwen_video_inputs = prepared.qwen_videos
    condition_labels = prepared.condition_labels
    media_input = prepared.media
    task = media_input.task
    transformed = copy.copy(prompt)
    additional_information = dict(prompt.get("additional_information") or {})
    transformed["prompt"] = text
    qwen_mm_data: dict[str, Any] = {}
    if images:
        qwen_mm_data["image"] = images
    if qwen_video_inputs:
        qwen_mm_data["video"] = qwen_video_inputs
    transformed["multi_modal_data"] = qwen_mm_data or None

    mm_processor_kwargs = dict(prompt.get("mm_processor_kwargs") or {})
    mm_processor_kwargs[MINIMAX_H3_PRESENTATION_TASK_KEY] = task
    mm_processor_kwargs[MINIMAX_H3_CONDITION_LABELS_KEY] = condition_labels
    media_tensors = media_input.to_mm_tensors()
    transformed["mm_processor_kwargs"] = mm_processor_kwargs

    hidden_states = dict(additional_information.get("hidden_states") or {})
    hidden_states["layers"] = dict(enumerate(media_tensors))
    additional_information["hidden_states"] = hidden_states
    meta = dict(additional_information.get("meta") or {})
    meta[MINIMAX_H3_ENCODER_REQUEST_KEY] = media_input.to_metadata()
    additional_information["meta"] = meta
    transformed["additional_information"] = additional_information
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


def _encoder_conditioning(payload: Any) -> MiniMaxH3EncoderConditioning:
    if isinstance(payload, OmniPayloadStruct):
        payload = to_dict(payload)
    if not isinstance(payload, Mapping):
        raise RuntimeError("MiniMax H3 encoder returned no conditioning payload")
    try:
        return MiniMaxH3EncoderConditioning.from_omni_payload(unflatten_payload(dict(payload)))
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def encoder2diffusion_full_payload(
    *,
    pooling_output: Any = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Pack all three encoder components for direct worker-to-worker transfer.

    Returning the diffusion-ready structure here keeps the DiT worker free of
    any H3-specific unpacking: the generic receive path merges this dict into
    the request's ``additional_information``.
    """
    del kwargs
    if pooling_output is None:
        return None
    return {"encoder_output": _encoder_conditioning(pooling_output).to_omni_payload()}


def encoder2diffusion(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> dict[str, Any] | None:
    """Reuse the encoder handoff for all three H3 encoder components."""
    del requires_multimodal_data, streaming_context
    if not source_outputs:
        return None
    if len(source_outputs) != 1:
        raise RuntimeError(f"MiniMax H3 DiT requires exactly one encoder source, got {len(source_outputs)}")
    if not getattr(source_outputs[0], "finished", True):
        return None

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
            "MiniMax H3 encoder request ID does not match the diffusion request: "
            f"source={source_request_id!r}, expected={expected_request_id!r}"
        )

    outputs = getattr(source_output, "outputs", None)
    if not isinstance(outputs, list) or len(outputs) != 1:
        output_count = len(outputs) if isinstance(outputs, list) else 0
        raise RuntimeError(f"MiniMax H3 encoder must return exactly one completion, got {output_count}")
    payload = getattr(outputs[0], "multimodal_output", None)
    # Successful connector sends omit the inline payload. The diffusion runner
    # merges encoder_output before forward; original media still needs cleanup.
    conditioning = _encoder_conditioning(payload) if payload is not None else None

    additional_information = dict(diffusion_prompt.get("additional_information") or {})
    hidden_states = dict(additional_information.get("hidden_states") or {})
    hidden_states.pop("layers", None)
    if hidden_states:
        additional_information["hidden_states"] = hidden_states
    else:
        additional_information.pop("hidden_states", None)
    meta = dict(additional_information.get("meta") or {})
    meta.pop(MINIMAX_H3_ENCODER_REQUEST_KEY, None)
    if meta:
        additional_information["meta"] = meta
    else:
        additional_information.pop("meta", None)
    if conditioning is not None:
        additional_information["encoder_output"] = conditioning.to_omni_payload()
    diffusion_prompt["additional_information"] = additional_information
    diffusion_prompt["multi_modal_data"] = None
    diffusion_prompt.pop("model_intermediate_buffer", None)
    return diffusion_prompt
