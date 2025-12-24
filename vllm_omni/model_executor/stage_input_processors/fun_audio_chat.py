# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for Fun-Audio-Chat: Main → CosyVoice transition."""

from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def main2cosyvoice(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process main stage outputs to create CosyVoice inputs.

    Workflow:
    1. Extract speech tokens from main stage output
    2. Package for CosyVoice3 synthesis

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for main)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for CosyVoice stage
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    main_outputs = stage_list[source_stage_id].engine_outputs
    cosyvoice_inputs = []

    # Process each main stage output
    for i, main_output in enumerate(main_outputs):
        output = main_output.outputs[0]

        # Extract speech tokens from multimodal_output
        speech_tokens = output.multimodal_output.get("speech_tokens")

        if speech_tokens is None:
            # If no speech tokens, create empty input
            # CosyVoice will handle this case
            cosyvoice_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                )
            )
            continue

        # Convert to list if tensor
        if isinstance(speech_tokens, torch.Tensor):
            speech_tokens = speech_tokens.cpu().to(torch.long).reshape(-1).tolist()

        cosyvoice_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=speech_tokens,
                additional_information={
                    "text_response": output.text if hasattr(output, "text") else None,
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return cosyvoice_inputs


__all__ = ["main2cosyvoice"]
