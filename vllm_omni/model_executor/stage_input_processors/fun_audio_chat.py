# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for Fun-Audio-Chat 3-stage S2S pipeline.

Pipeline:
- Stage 0 (Main): Audio → Text + Hidden States
- Stage 1 (CRQ): Hidden States → Speech Tokens
- Stage 2 (CosyVoice): Speech Tokens → Audio Waveform

Transitions:
- main2crq: Stage 0 → Stage 1
- crq2cosyvoice: Stage 1 → Stage 2
"""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def main2crq(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process main stage outputs for CRQ decoder input.

    Extracts hidden states and text embeddings from the main stage
    for speech token generation.

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs [0]
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for CRQ decoder stage
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    main_outputs = stage_list[source_stage_id].engine_outputs
    crq_inputs = []

    for i, main_output in enumerate(main_outputs):
        output = main_output.outputs[0]

        # Get hidden states (latent) from main stage
        latent = output.multimodal_output.get("latent")
        if latent is None:
            # Try alternative key
            latent = output.multimodal_output.get("hidden_states")

        if latent is None:
            logger.warning(f"No hidden states found in main output {i}")
            crq_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    additional_information={"thinker_hidden_states": None},
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                )
            )
            continue

        # Clone and move to correct device/dtype
        thinker_hidden_states = latent.clone().detach().cuda().to(torch.float32)

        # Get prompt token IDs for position info
        prompt_token_ids = main_output.prompt_token_ids
        generated_token_ids = output.token_ids
        prompt_len = len(prompt_token_ids) if prompt_token_ids else 0

        # Extract only the generated portion hidden states (response part)
        # The CRQ decoder needs hidden states corresponding to the response
        if thinker_hidden_states.dim() == 2:
            # [seq_len, hidden_size] -> split by prompt length
            response_hidden = thinker_hidden_states[prompt_len:]
            prompt_hidden = thinker_hidden_states[:prompt_len]
        else:
            # Assume [batch, seq_len, hidden_size]
            response_hidden = thinker_hidden_states[:, prompt_len:]
            prompt_hidden = thinker_hidden_states[:, :prompt_len]

        # Get text embeddings if available
        text_embeds = output.multimodal_output.get("text_embeds")

        crq_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=generated_token_ids,  # Use generated tokens as reference
                additional_information={
                    "thinker_hidden_states": response_hidden,
                    "prompt_embeds": prompt_hidden,
                    "text_embeds": text_embeds,
                    "prompt_token_ids": prompt_token_ids,
                    "generated_token_ids": generated_token_ids,
                    "text_response": output.text if hasattr(output, "text") else None,
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return crq_inputs


def crq2cosyvoice(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process CRQ decoder outputs for CosyVoice input.

    Extracts speech tokens from CRQ decoder and packages them
    for audio synthesis.

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs [1]
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

    crq_outputs = stage_list[source_stage_id].engine_outputs
    cosyvoice_inputs = []

    for i, crq_output in enumerate(crq_outputs):
        output = crq_output.outputs[0]

        # Extract speech tokens from CRQ output
        speech_tokens = output.multimodal_output.get("speech_tokens")

        if speech_tokens is None:
            logger.warning(f"No speech tokens found in CRQ output {i}")
            cosyvoice_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    additional_information={"speech_tokens": None},
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                )
            )
            continue

        # Convert to list if tensor
        if isinstance(speech_tokens, torch.Tensor):
            speech_tokens_list = speech_tokens.cpu().to(torch.long).reshape(-1).tolist()
        else:
            speech_tokens_list = list(speech_tokens)

        # Filter valid tokens (0 <= token < 6561 are valid codebook tokens)
        valid_tokens = [t for t in speech_tokens_list if 0 <= t < 6561]

        # Get text response from previous stage if available
        text_response = None
        if hasattr(output, "text"):
            text_response = output.text

        cosyvoice_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=valid_tokens,  # Speech tokens as input
                additional_information={
                    "speech_tokens": torch.tensor(valid_tokens, dtype=torch.long),
                    "raw_speech_tokens": speech_tokens_list,
                    "text_response": text_response,
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return cosyvoice_inputs


# Legacy function for backward compatibility
def main2cosyvoice(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Legacy function for 2-stage pipeline (Main → CosyVoice).

    For 3-stage pipeline, use main2crq and crq2cosyvoice instead.
    """
    logger.warning("main2cosyvoice is for 2-stage pipeline. For 3-stage S2S, use main2crq and crq2cosyvoice.")

    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    main_outputs = stage_list[source_stage_id].engine_outputs
    cosyvoice_inputs = []

    for i, main_output in enumerate(main_outputs):
        output = main_output.outputs[0]

        # Extract speech tokens from multimodal_output
        speech_tokens = output.multimodal_output.get("speech_tokens")

        if speech_tokens is None:
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
                    "speech_tokens": torch.tensor(speech_tokens, dtype=torch.long),
                    "text_response": output.text if hasattr(output, "text") else None,
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return cosyvoice_inputs


__all__ = ["main2crq", "crq2cosyvoice", "main2cosyvoice"]
