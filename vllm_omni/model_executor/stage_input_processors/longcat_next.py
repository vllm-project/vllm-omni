"""Stage input processors bridging LongCat-Next thinker -> decoders.

The thinker's flattened output stream interleaves text ids with multimodal
code rows (8 consecutive per position, one id per codebook level, each id
carrying its level offset) delimited by marker tokens:

    <longcat_img_start> [8 ids] ... <longcat_img_newline> ... <longcat_img_end>
    <longcat_audiogen_start> [8 ids] ... <longcat_audiogen_end>

Extraction is segment-aware rather than range-based: the audio and visual
flat-id ranges overlap (audio level 2+ ids exceed the visual base offset), so
only the surrounding markers disambiguate the modality.

These are ``sync_process_input_func`` hooks, invoked by the stage engine as
``func(source_outputs, prompt, requires_multimodal_data)`` where each source
output carries ``prompt_token_ids``, ``finished`` and ``outputs[0]`` with the
generated ids (see cosyvoice3.text2flow_token_only for the reference shape).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.longcat_next.longcat_next_utils import (
    extract_audio_codes,
    extract_visual_codes,
    infer_visual_grid,
)


def _ensure_list(ids: Any) -> list[int]:
    if ids is None:
        return []
    if hasattr(ids, "tolist"):
        return list(ids.tolist())
    return list(ids)


def _generated_ids(source_output: Any) -> list[int]:
    """Generated ids for one finished request, prompt prefix stripped."""
    output = source_output.outputs[0]
    output_ids = _ensure_list(getattr(output, "cumulative_token_ids", None)) or _ensure_list(
        getattr(output, "token_ids", None)
    )
    prefix = _ensure_list(getattr(source_output, "prompt_token_ids", None))
    if prefix and output_ids[: len(prefix)] == prefix:
        output_ids = output_ids[len(prefix):]
    return output_ids


def thinker2image_decoder_token_only(
    source_outputs: Sequence[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = True,
) -> list[OmniTokensPrompt]:
    del prompt
    engine_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        if not source_output.finished:
            continue
        output_ids = _generated_ids(source_output)
        additional_information: dict[str, Any] = {
            "visual_token_ids": extract_visual_codes(output_ids),
        }
        grid = infer_visual_grid(output_ids)
        if grid is not None:
            additional_information["token_h"] = grid[0]
            additional_information["token_w"] = grid[1]
        engine_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return engine_inputs


def thinker2audio_decoder_token_only(
    source_outputs: Sequence[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = True,
) -> list[OmniTokensPrompt]:
    del prompt
    engine_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        if not source_output.finished:
            continue
        output_ids = _generated_ids(source_output)
        engine_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information={
                    "audio_token_ids": extract_audio_codes(output_ids),
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return engine_inputs
