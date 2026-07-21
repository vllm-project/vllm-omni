# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for MiniCPM-o 4.5.

Thinker -> Talker converts hidden states + token ids into the MiniCPMTTS
conditioning payload. Talker -> Token2Wav ships the complete generated
audio-token payload through the connector in non-async mode.
"""

import logging
from collections.abc import Mapping
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)


def llm2talker(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | dict | list | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
):
    """Extract the thinker token/hidden slice consumed by the Talker stage."""
    del streaming_context  # not used by MiniCPM-o 4.5 turn-taking pipeline

    if not source_outputs:
        raise ValueError("source_outputs cannot be empty")

    talker_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    multi_modal_data = {
        llm_output.request_id: p.get("multi_modal_data", None) if isinstance(p, dict) else None
        for llm_output, p in zip(source_outputs, prompt)
    }

    for llm_output in source_outputs:
        output = llm_output.outputs[0]
        prompt_token_ids = llm_output.prompt_token_ids
        llm_output_ids = output.token_ids
        latent = output.multimodal_output.get("latent", None)
        if latent is None:
            latent = output.hidden_states if hasattr(output, "hidden_states") else None
            if latent is None:
                raise ValueError("No latent or hidden_states found in thinker output")

        thinker_hidden_states = latent.clone().detach()

        # Build full token sequence and extract TTS region
        full_token_ids = list(prompt_token_ids) + (
            list(llm_output_ids) if not isinstance(llm_output_ids, list) else llm_output_ids
        )
        full_hidden = thinker_hidden_states.to(torch.float32)

        # Detect TTS token IDs (4.5: 151703/151704, 2.6: 151691/151692)
        tts_bos_id, tts_eos_id = 151691, 151692
        for _id in [151703, 151704]:
            if _id in full_token_ids:
                tts_bos_id, tts_eos_id = 151703, 151704
                break

        tts_bos_idx = tts_eos_idx = None
        for idx_t, tid in enumerate(full_token_ids):
            if tid == tts_bos_id:
                tts_bos_idx = idx_t + 1
            elif tid == tts_eos_id:
                tts_eos_idx = idx_t

        tts_token_ids_slice = tts_hidden_slice = None
        if tts_bos_idx is not None and full_hidden.shape[0] > tts_bos_idx:
            end_idx = tts_eos_idx if tts_eos_idx is not None else full_hidden.shape[0]
            tts_token_ids_slice = torch.tensor(full_token_ids[tts_bos_idx:end_idx], dtype=torch.long)
            tts_hidden_slice = full_hidden[tts_bos_idx:end_idx]

        additional_information: dict[str, Any] = {}
        if tts_token_ids_slice is not None:
            additional_information["tts_token_ids"] = tts_token_ids_slice
        if tts_hidden_slice is not None:
            additional_information["tts_hidden_states"] = tts_hidden_slice

        # Minimal prompt token IDs: the talker's AR framework needs *some* tokens
        # to do a single prefill step. We use [BOS, PAD, EOS] as a dummy.
        request_mm = multi_modal_data.get(llm_output.request_id)
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[1, 0, 2],
                additional_information=additional_information,
                multi_modal_data=request_mm if requires_multimodal_data and request_mm is not None else None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


# Backward-compatible alias for old deploys / tests. The alias now points to
# the Talker-only path; Token2Wav is a separate downstream stage.
llm2tts = llm2talker


def _audio_codes_from_mapping(mapping: Mapping[str, Any] | None) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    audio = mapping.get("codes.audio")
    if audio is not None:
        return audio
    codes = mapping.get("codes")
    if isinstance(codes, Mapping):
        return codes.get("audio")
    return None


def _audio_code_len(audio: Any) -> int:
    if audio is None:
        return 0
    if isinstance(audio, torch.Tensor):
        return int(audio.numel())
    try:
        return int(torch.as_tensor(audio).numel())
    except (TypeError, ValueError):
        try:
            return len(audio)
        except TypeError:
            return 1


def _minicpmo_empty_finished_payload() -> dict[str, Any]:
    return {
        "codes": {"audio": torch.empty(0, dtype=torch.long)},
        "meta": {
            "finished": torch.tensor(True, dtype=torch.bool),
            "code_flat_numel": 0,
            "next_stage_prompt_len": 1,
        },
    }


def talker2token2wav_token_only(
    source_outputs: list[Any],
    _prompt: OmniTokensPrompt | TextPrompt | None = None,
    _requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Sync-side placeholder for the Token2Wav stage.

    The actual audio token tensor is delivered by
    ``talker2token2wav_full_payload`` through the worker connector. This
    function only sizes the consumer prompt so the generation runner can
    schedule the request. Empty audio payloads still get one placeholder slot
    and are disambiguated by ``meta.code_flat_numel=0`` in the connector
    payload.
    """
    del _prompt, _requires_multimodal_data, streaming_context

    token2wav_inputs: list[OmniTokensPrompt] = []
    for talker_output in source_outputs:
        if getattr(talker_output, "finished", True) is False:
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output if hasattr(output, "multimodal_output") else None
        audio = _audio_codes_from_mapping(mm if isinstance(mm, Mapping) else None)
        prompt_len = max(1, _audio_code_len(audio))
        token2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return token2wav_inputs


def talker2token2wav_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any]:
    """Pack complete MiniCPM-o Talker audio tokens for Token2Wav."""
    del transfer_manager
    rid = getattr(request, "request_id", None)
    if not isinstance(pooling_output, Mapping):
        logger.warning(
            "talker2token2wav_full_payload: pooling_output not a dict (type=%s) for req=%s; "
            "sending empty finished payload.",
            type(pooling_output).__name__,
            rid,
        )
        return _minicpmo_empty_finished_payload()

    audio = _audio_codes_from_mapping(pooling_output)
    if audio is None:
        logger.warning(
            "talker2token2wav_full_payload: missing codes.audio (keys=%s) for req=%s; sending empty finished payload.",
            list(pooling_output.keys()),
            rid,
        )
        return _minicpmo_empty_finished_payload()
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio, dtype=torch.long)
    audio = audio.to(dtype=torch.long).reshape(-1)
    if audio.numel() == 0:
        logger.warning(
            "talker2token2wav_full_payload: empty codes.audio for req=%s; sending empty finished payload.",
            rid,
        )
        return _minicpmo_empty_finished_payload()

    audio_cpu = audio.detach().to("cpu").contiguous()
    return {
        "codes": {"audio": audio_cpu},
        "meta": {
            "finished": torch.tensor(True, dtype=torch.bool),
            "code_flat_numel": int(audio_cpu.numel()),
            "next_stage_prompt_len": int(audio_cpu.numel()),
        },
    }
