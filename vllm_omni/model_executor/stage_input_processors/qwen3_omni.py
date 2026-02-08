# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def _compute_talker_prompt_ids_length(info, device: torch.device | str = "cuda") -> int:
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091

    thinker_sequences = torch.tensor(info["thinker_sequences"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    input_ids = torch.tensor(info["thinker_input_ids"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    im_start_indexes = torch.cat(
        [
            torch.nonzero(input_ids[0] == im_start_token_id).squeeze(1),
            torch.tensor([thinker_sequences.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
        ],
        dim=0,
    )

    if len(im_start_indexes) <= 1:
        # Case for non-chat input (e.g. speech API)
        # If no im_start tokens found (only the length sentinel remains),
        # return the full length of input_ids
        return input_ids.shape[-1]

    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = int(im_start_indexes[i].item())
        e = int(im_start_indexes[i + 1].item())
        role = int(input_ids[0, s + 1].item())
        if role == system_token_id:
            continue
        elif role == user_token_id:
            sum_user_len += e - s
        elif role == assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_len += 9  # 3 + 4 + 1 + 1
        else:
            pass

    return sum_user_len + assistant_len


def _validate_stage_inputs(stage_list, engine_input_source):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    return stage.engine_outputs


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    async_chunk_stream: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create talker inputs.

    Workflow:
    1. Extract thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for thinker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for talker stage
    """
    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    talker_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    # Process each thinker output
    for thinker_output in thinker_outputs:
        output = thinker_output.outputs[0]
        thinker_embeddings = (
            torch.cat(output.multimodal_output["0"], dim=0).detach().to(device=device, dtype=torch.float)
            if isinstance(output.multimodal_output["0"], list)
            else output.multimodal_output["0"].detach().to(device=device, dtype=torch.float)
        )
        thinker_hidden_states = (
            torch.cat(output.multimodal_output["24"], dim=0).detach().to(device=device, dtype=torch.float)
            if isinstance(output.multimodal_output["24"], list)
            else output.multimodal_output["24"].detach().to(device=device, dtype=torch.float)
        )
        info = {
            "thinker_embeddings": thinker_embeddings,
            "thinker_hidden_states": thinker_hidden_states,
            "thinker_sequences": thinker_output.prompt_token_ids
            + output.token_ids,  # the thinker_sequences is the whole ids
            "thinker_input_ids": thinker_output.prompt_token_ids,
            # Provide thinker-side TTS token embeddings for talker projection
            "tts_bos_embed": (
                torch.cat(output.multimodal_output["tts_bos_embed"], dim=0)
                .detach()
                .to(device=device, dtype=torch.float)
                if isinstance(output.multimodal_output["tts_bos_embed"], list)
                else output.multimodal_output["tts_bos_embed"].detach().to(device=device, dtype=torch.float)
            ),
            "tts_eos_embed": (
                torch.cat(output.multimodal_output["tts_eos_embed"], dim=0)
                .detach()
                .to(device=device, dtype=torch.float)
                if isinstance(output.multimodal_output["tts_eos_embed"], list)
                else output.multimodal_output["tts_eos_embed"].detach().to(device=device, dtype=torch.float)
            ),
            "tts_pad_embed": (
                torch.cat(output.multimodal_output["tts_pad_embed"], dim=0)
                .detach()
                .to(device=device, dtype=torch.float)
                if isinstance(output.multimodal_output["tts_pad_embed"], list)
                else output.multimodal_output["tts_pad_embed"].detach().to(device=device, dtype=torch.float)
            ),
        }

        if async_chunk_stream:
            # Wait until atleast 2 output tokens before leaving prefill.
            # Because 1st token will be generated during last step of prefill phase
            # Keep this aligned with Qwen3ThinkerChunkProcessor.min_tokens_for_decode.
            info["is_prefill"] = [True] if len(output.token_ids) < 2 else [False]

        prompt_len = _compute_talker_prompt_ids_length(info, device=device)

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    async_chunk_stream: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.

    Workflow:
    1. Extract talker's codec code outputs (8-layer RVQ codes)
    2. Flatten codes for code2wav input
    3. Package for code2wav stage

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [1] for talker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for code2wav stage
    """
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []

    # The number of talker output tokens to accumulate
    # before invoking code2wav stage
    talker_tokens_batch_size = 25

    # Process each talker output
    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        # When async_chunk_stream is enabled, "is_prefill" key should be added to
        # additional_information. This helps orchestrator to forwarding the first chunk
        # to next stage when current stage is still in prefill phase
        if async_chunk_stream:
            prefill_len = len(talker_output.prompt_token_ids)
            seq_len = len(output.token_ids)

            additional_information = {"is_prefill": [True] if seq_len <= talker_tokens_batch_size else [False]}

            code_predictor_codes = output.multimodal_output["code_predictor_codes"]

            if isinstance(code_predictor_codes, list):
                code_predictor_codes = torch.cat(code_predictor_codes, dim=0)

            # Accumulate talker output tokens until talker_tokens_batch_size before invoking code2wav stage
            # The rest of the chunks arrive by scheduler-scheduler communication using omniconnector
            code_predictor_codes = code_predictor_codes[prefill_len : prefill_len + talker_tokens_batch_size]

            if code_predictor_codes.shape[0] > 0:
                # Extract codec codes from talker output
                codec_codes = (
                    code_predictor_codes.to(torch.long).transpose(0, 1).cpu().to(torch.long).reshape(-1).tolist()
                )
            else:
                codec_codes = []

            code2wav_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=codec_codes,
                    multi_modal_data=None,
                    additional_information=additional_information,
                    mm_processor_kwargs=None,
                )
            )
        else:
            seq_len = len(output.token_ids) - 1
            # Extract codec codes from talker output
            # Expected shape: [8, seq_len] (8-layer RVQ codes)
            codec_codes = (
                output.multimodal_output["code_predictor_codes"][-seq_len:]
                .to(torch.long)
                .transpose(0, 1)
                .cpu()
                .to(torch.long)
                .reshape(-1)
                .tolist()
            )  # 16, seq_len
            code2wav_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=codec_codes,
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                )
            )

    return code2wav_inputs
