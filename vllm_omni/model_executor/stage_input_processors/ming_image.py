# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict MLLM-to-diffusion bridge for Ming-Image."""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import _ensure_list

_IMAGE_PATCH_TOKEN_ID = 157157
_IMAGE_END_TOKEN_ID = 157159
_NUM_QUERY_TOKENS = 256
_CAPTURE_LAYERS = (5, 12, 20)


def _output_payload(source_output: Any) -> tuple[list[int], dict[str, torch.Tensor]]:
    if not getattr(source_output, "outputs", None):
        raise ValueError("Ming-Image thinker produced no outputs.")
    output = source_output.outputs[0]
    multimodal = getattr(output, "multimodal_output", None) or {}
    return _ensure_list(source_output.prompt_token_ids), multimodal


def _validate_query_suffix(prompt_ids: list[int]) -> tuple[int, int]:
    expected = _NUM_QUERY_TOKENS + 2
    if len(prompt_ids) < expected:
        raise ValueError("Ming-Image prompt is shorter than its image-query suffix.")
    start = len(prompt_ids) - _NUM_QUERY_TOKENS - 1
    end = len(prompt_ids) - 1
    if prompt_ids[end] != _IMAGE_END_TOKEN_ID or any(token != _IMAGE_PATCH_TOKEN_ID for token in prompt_ids[start:end]):
        raise ValueError("Ming-Image query suffix must end with 256 image-patch tokens and an image-end token.")
    return start, end


def _as_sequence(hidden: torch.Tensor, prompt_length: int, name: str) -> torch.Tensor:
    if hidden.ndim == 3:
        if hidden.shape[0] != 1:
            raise ValueError(f"{name} must have batch size 1, got {tuple(hidden.shape)}")
        hidden = hidden[0]
    if hidden.ndim != 2 or hidden.shape[0] != prompt_length:
        raise ValueError(f"{name} shape {tuple(hidden.shape)} does not match prompt length {prompt_length}.")
    return hidden


def _extract_conditions(source_output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids, multimodal = _output_payload(source_output)
    query_start, query_end = _validate_query_suffix(prompt_ids)
    final_hidden = multimodal.get("final_hidden_states")
    if not isinstance(final_hidden, torch.Tensor):
        raise ValueError("Ming-Image thinker output is missing final_hidden_states.")
    final_hidden = _as_sequence(final_hidden, len(prompt_ids), "final_hidden_states")
    query_hidden = final_hidden[query_start:query_end].detach().contiguous()

    selected: list[torch.Tensor] = []
    for layer_idx in _CAPTURE_LAYERS:
        value = multimodal.get(f"hidden_states_{layer_idx}")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Ming-Image thinker output is missing hidden_states_{layer_idx}.")
        selected.append(_as_sequence(value, len(prompt_ids), f"hidden_states_{layer_idx}"))

    # Vendor labels mark the full generated query block, including delimiters.
    # Direct VLM conditioning is restricted to the original prompt/reference
    # prefix and therefore excludes all 258 appended tokens.
    direct_end = len(prompt_ids) - (_NUM_QUERY_TOKENS + 2)
    direct_hidden = torch.cat([value[:direct_end] for value in selected], dim=-1)
    return query_hidden, direct_hidden.detach().contiguous()


def thinker2image(
    source_outputs: list[Any],
    prompt: Any | None = None,
    requires_multimodal_data: bool = False,
    sampling_params: Any | None = None,
) -> list[dict[str, Any]]:
    del requires_multimodal_data
    if len(source_outputs) != 1:
        raise ValueError(f"Ming-Image expects one thinker output, got {len(source_outputs)}.")

    extra_args = getattr(sampling_params, "extra_args", None) or {}
    negative_prompt = extra_args.get("negative_prompt")
    if isinstance(prompt, dict):
        negative_prompt = prompt.get("negative_prompt", negative_prompt)
    if isinstance(negative_prompt, str) and negative_prompt.strip():
        raise ValueError("Ming-Image uses zero negative conditioning and does not accept negative_prompt.")

    query_hidden, direct_hidden = _extract_conditions(source_outputs[0])
    extra: dict[str, Any] = {
        "query_hidden_states": query_hidden,
        "direct_hidden_states": direct_hidden,
    }

    if isinstance(prompt, dict):
        mm_data = prompt.get("multi_modal_data") or {}
        reference = mm_data.get("img2img", mm_data.get("image"))
        if isinstance(reference, list):
            reference = reference[0] if reference else None
        if reference is not None:
            extra["reference_image"] = reference

    num_layers = int(extra_args.get("num_layers", 1))
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    extra["num_layers"] = num_layers
    return [{"prompt": "", "extra": extra}]


__all__ = ["thinker2image"]
