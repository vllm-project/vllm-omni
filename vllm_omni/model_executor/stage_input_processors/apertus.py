# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors and helpers for Apertus."""

from collections.abc import Sequence
from typing import Any

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def merge_image_placeholders(
    prompt: str,
    image_prompts: Sequence[str],
    image_placeholder: str = "<|image|>",
) -> str:
    """Replace image placeholders in order with serialized image token strings.

    If no placeholder appears in the prompt, this method prepends one
    placeholder per image (same behavior used by simple prompt style).
    """
    if not image_prompts:
        return prompt

    merged_prompt = prompt
    placeholder_count = merged_prompt.count(image_placeholder)

    if placeholder_count == 0:
        image_prefix = " ".join([image_placeholder] * len(image_prompts))
        merged_prompt = f"{image_prefix}\n{merged_prompt}" if merged_prompt else image_prefix
        placeholder_count = len(image_prompts)

    if placeholder_count != len(image_prompts):
        raise ValueError(
            f"Mismatch: found {placeholder_count} '{image_placeholder}' placeholders, "
            f"but got {len(image_prompts)} images."
        )

    for image_prompt in image_prompts:
        merged_prompt = merged_prompt.replace(image_placeholder, image_prompt, 1)

    return merged_prompt


def _ensure_list(x: Any) -> list[Any]:
    if hasattr(x, "_x"):
        return list(x._x)
    if isinstance(x, list):
        return list(x)
    if x is None:
        return []
    return list(x)


def _validate_stage_inputs(stage_list: list[Any], engine_input_source: list[int]) -> Any:
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    return stage.engine_outputs


def prefill_to_decode(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Forward tokenized prompt + generated tokens from a prefill stage.

    This is useful for a prefill/decode style split where both stages run
    `ApertusForCausalLM` and the downstream stage consumes token-only inputs.
    """
    del prompt, requires_multimodal_data
    source_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    decode_inputs: list[OmniTokensPrompt] = []

    for source_output in source_outputs:
        output = source_output.outputs[0]
        prompt_token_ids = _ensure_list(source_output.prompt_token_ids)
        generated_token_ids = _ensure_list(output.token_ids)

        decode_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=prompt_token_ids + generated_token_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return decode_inputs
