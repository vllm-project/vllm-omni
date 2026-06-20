# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek Janus stage bridges.

The functions in this module are referenced from Janus' pipeline config and
intentionally live with the Janus model package.  The public
``vllm_omni.model_executor.stage_input_processors.deepseek_janus`` module
re-exports them for older deploy configs.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def _normalize_prompt(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, dict):
        return prompt
    if hasattr(prompt, "_asdict"):
        return prompt._asdict()
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    return {}


def _normalize_extra(prompt_dict: dict[str, Any]) -> dict[str, Any]:
    extra = prompt_dict.get("extra")
    return extra if isinstance(extra, dict) else {}


def ar2generation(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Build diffusion-stage request dicts from text AR outputs."""
    del requires_multimodal_data
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid source stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    ar_outputs = stage_list[source_stage_id].engine_outputs
    out: list[dict[str, Any]] = []

    if not isinstance(prompt, list):
        prompt = [prompt] if prompt is not None else [{}]

    for i, ar_row in enumerate(ar_outputs):
        ao = ar_row.outputs[0]
        ar_text = getattr(ao, "text", "") or ""

        original = _normalize_prompt(prompt[i] if i < len(prompt) else {})
        base_prompt = original.get("prompt", "") or ""

        if ar_text.strip():
            merged = f"{base_prompt}\n{ar_text}".strip() if base_prompt else ar_text.strip()
        else:
            merged = base_prompt

        height = original.get("height", 384)
        width = original.get("width", 384)

        prompt_extra = _normalize_extra(original)
        diffusion_input: dict[str, Any] = {
            "prompt": merged,
            "height": height,
            "width": width,
            "extra": {
                **prompt_extra,
                "ar_generated_text": ar_text,
                "base_prompt": base_prompt,
            },
        }

        mm_data = original.get("multi_modal_data")
        if mm_data:
            diffusion_input["multi_modal_data"] = mm_data

        for key in ("seed", "num_inference_steps", "guidance_scale"):
            if key in original:
                diffusion_input[key] = original[key]

        logger.debug(
            "[ar2generation] merged prompt len=%d (base=%d ar=%d)",
            len(merged),
            len(base_prompt),
            len(ar_text),
        )
        out.append(diffusion_input)

    return out


def ar_tokens_to_vq(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Build VQ-decode request dicts from AR image-token outputs."""
    del requires_multimodal_data
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid source stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    ar_outputs = stage_list[source_stage_id].engine_outputs
    out: list[dict[str, Any]] = []

    if not isinstance(prompt, list):
        prompt = [prompt] if prompt is not None else [{}]

    for i, ar_row in enumerate(ar_outputs):
        image_tokens = None

        if hasattr(ar_row, "image_tokens"):
            image_tokens = ar_row.image_tokens
        elif hasattr(ar_row, "image_token_ids"):
            image_tokens = ar_row.image_token_ids
        elif len(ar_row.outputs) > 0:
            sampled_ids = getattr(ar_row.outputs[0], "token_ids", None) or []
            if sampled_ids and len(sampled_ids) >= 576:
                image_tokens = torch.tensor(sampled_ids[:576], dtype=torch.long)
            else:
                all_ids = getattr(ar_row, "sampled_token_ids", None)
                if all_ids is not None and len(all_ids) >= 576:
                    image_tokens = all_ids[-576:]

        if image_tokens is None:
            logger.error("[ar_tokens_to_vq] Could not extract image tokens from AR output %d", i)
            continue

        original = _normalize_prompt(prompt[i] if i < len(prompt) else {})
        height = original.get("height", 384)
        width = original.get("width", 384)
        img_size = original.get("img_size", max(height, width))
        patch_size = original.get("patch_size", _normalize_extra(original).get("patch_size", 16))

        diffusion_input: dict[str, Any] = {
            "prompt": original.get("prompt", ""),
            "height": height,
            "width": width,
            "extra": {
                "image_tokens": image_tokens,
                "img_size": img_size,
                "patch_size": patch_size,
            },
        }

        for key in ("seed", "num_inference_steps", "guidance_scale"):
            if key in original:
                diffusion_input[key] = original[key]

        logger.debug(
            "[ar_tokens_to_vq] request %d: image_tokens shape=%s",
            i,
            getattr(image_tokens, "shape", "?") if hasattr(image_tokens, "shape") else len(image_tokens),
        )
        out.append(diffusion_input)

    return out


__all__ = ["ar2generation", "ar_tokens_to_vq"]
