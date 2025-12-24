# SPDX-License-Identifier: Apache-2.0
"""Stage input processor for Step-Audio2: Thinker → Token2Wav transition."""

from typing import Any, Union

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def thinker2token2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Union[OmniTokensPrompt, TextPrompt, None] = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create token2wav inputs.

    Workflow:
    1. Extract generated token IDs from thinker output
    2. Separate audio tokens from text tokens
    3. Package audio tokens for token2wav stage

    Step-Audio2 token ranges:
    - Text tokens: 0 - 151688
    - Audio tokens: 151696 - 158257 (vocab size 6562)

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for thinker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for token2wav stage
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    thinker_outputs = stage_list[source_stage_id].engine_outputs
    token2wav_inputs = []

    # Step-Audio2 constants
    STEP_AUDIO2_AUDIO_START = 151696
    STEP_AUDIO2_AUDIO_EOS = 6561  # Relative to audio start

    # Process each thinker output
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]

        # Get all generated tokens (prompt + generated)
        all_token_ids = thinker_output.prompt_token_ids + output.token_ids

        # Separate audio tokens from text tokens
        # Audio tokens are >= STEP_AUDIO2_AUDIO_START
        audio_tokens = [
            tid - STEP_AUDIO2_AUDIO_START  # Convert to 0-based for Token2Wav
            for tid in all_token_ids
            if tid >= STEP_AUDIO2_AUDIO_START
        ]

        # Remove padding tokens (anything >= EOS)
        audio_tokens = [t for t in audio_tokens if t < STEP_AUDIO2_AUDIO_EOS]

        if not audio_tokens:
            # No audio tokens generated, skip or use empty
            # For now, we'll pass empty list
            audio_tokens = []

        # Package for token2wav
        token2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=audio_tokens,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return token2wav_inputs
