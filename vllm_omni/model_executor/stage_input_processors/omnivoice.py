# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inter-stage processor for OmniVoice: Generator → Decoder."""

from typing import Any

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def tokens2audio(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = True,
):
    """Build stage-1 (decoder) inputs from stage-0 (generator) outputs.

    Takes the 8-codebook audio tokens from the generator and packages
    them for the HiggsAudioV2 decoder.
    """
    source_stage_id = engine_input_source[0]
    source_outputs = stage_list[source_stage_id].engine_outputs

    if not isinstance(prompt, list):
        prompt = [prompt]

    source_output = source_outputs[0]
    output = source_output.outputs[0]

    multi_modal_data = output.multimodal_output
    if multi_modal_data is None:
        raise RuntimeError(f"Missing multimodal_output for request {source_output.request_id}")

    # Pass audio_tokens from generator to decoder
    engine_input = OmniTokensPrompt(
        prompt_token_ids=output.cumulative_token_ids,
        additional_information=multi_modal_data,
    )
    return [engine_input]
