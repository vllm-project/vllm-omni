# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
"""Stage input processors for MiniMind-O (thinker → talker → code2wav)."""

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import (
    EmbeddingsStruct,
    HiddenStatesStruct,
    IdsStruct,
    OmniPayloadStruct,
    to_dict,
)
from vllm_omni.inputs.data import OmniTokensPrompt

MIMI_CODEC_PAD_TOKEN_ID = 2049
MIMI_CODEC_STOP_TOKEN_ID = 2050
MIMI_NUM_CODEC_LAYERS = 8


def _build_initial_audio_ids(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """8-layer Mimi code buffer, all pads (HF stream_generate audio_buffer)."""
    return torch.full(
        (MIMI_NUM_CODEC_LAYERS, seq_len),
        MIMI_CODEC_PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )


def thinker2talker(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    talker_inputs = []
    if not isinstance(prompt, list):
        prompt = [prompt]
    multi_modal_data = {
        thinker_output.request_id: p.get("multi_modal_data", None)
        for thinker_output, p in zip(source_outputs, prompt)
    }

    for i, thinker_output in enumerate(source_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = list(thinker_output.prompt_token_ids)
        thinker_output_ids = list(output.cumulative_token_ids)
        mm = output.multimodal_output
        latent = mm["latent"]
        thinker_hidden_states = latent.clone().detach().to(latent.device)
        prompt_token_ids_len = len(prompt_token_ids)
        decode_hidden = thinker_hidden_states[prompt_token_ids_len:].to(torch.float32)
        prefill_hidden = thinker_hidden_states[:prompt_token_ids_len].to(torch.float32)

        # Talker prefill length: prompt + first generated text token (HF delay pattern).
        talker_seq_len = max(prompt_token_ids_len + 1, 1)
        audio_ids = _build_initial_audio_ids(talker_seq_len, device=prefill_hidden.device)

        additional_information = to_dict(
            OmniPayloadStruct(
                hidden_states=HiddenStatesStruct(
                    output=decode_hidden,
                    output_shape=list(decode_hidden.shape),
                ),
                embed=EmbeddingsStruct(
                    prefill=prefill_hidden,
                    prefill_shape=list(prefill_hidden.shape),
                    audio_ids=audio_ids,
                ),
                ids=IdsStruct(prompt=prompt_token_ids, output=thinker_output_ids),
            )
        )

        # Flat prompt ids for the talker AR stage (codec stream); fusion uses additional_information.
        flat_ids = [MIMI_CODEC_PAD_TOKEN_ID] * talker_seq_len
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=flat_ids,
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[thinker_output.request_id]
                    if requires_multimodal_data and multi_modal_data is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )
    return talker_inputs


def talker2code2wav(
    source_outputs,
    _prompt: OmniTokensPrompt | TextPrompt = None,
    _requires_multimodal_data: bool = False,
):
    code2wav_inputs = []
    for talker_output in source_outputs:
        output = talker_output.outputs[0]
        mm = getattr(output, "multimodal_output", None) or {}
        codes = None
        if isinstance(mm, dict):
            codes = mm.get("codes", {}).get("audio")
        if isinstance(codes, torch.Tensor) and codes.numel() > 0:
            if codes.dim() == 2 and codes.size(-1) == MIMI_NUM_CODEC_LAYERS:
                token_ids = codes.reshape(-1).tolist()
            else:
                token_ids = codes.reshape(-1).tolist()
        else:
            token_ids = list(output.cumulative_token_ids)
            while token_ids and token_ids[-1] == MIMI_CODEC_STOP_TOKEN_ID:
                token_ids = token_ids[:-1]
            while token_ids and token_ids[0] == MIMI_CODEC_PAD_TOKEN_ID:
                token_ids = token_ids[1:]
        if not token_ids:
            continue
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=token_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs
