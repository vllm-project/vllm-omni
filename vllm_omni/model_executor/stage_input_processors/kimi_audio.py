# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Kimi-Audio request admission and complete AR-to-decoder input conversion."""

from collections.abc import Sequence
from typing import Any

import msgspec
import torch

from vllm_omni.data_entry_keys import deserialize_payload
from vllm_omni.engine import AdditionalInformationPayload
from vllm_omni.model_executor.models.kimi_audio.prompt import KimiAudioSpecialTokens
from vllm_omni.model_executor.models.kimi_audio.sampling import KimiAudioSamplingParams


def prepare_kimi_audio_request(prompt: dict[str, Any], sampling_params_list: Sequence[Any]) -> dict[str, Any]:
    """Validate stage-0 settings through Omni's existing prompt_transform_func.

    Call after prepare_kimi_audio_inputs and after resolving stage sampling
    defaults, before native input processing. No SamplingParams object crosses
    into the model buffer. Registration must also enable sampling_extra_args
    and include the tokenizer's msg_end in stage-0 stop_token_ids.
    """
    params = sampling_params_list[0]
    if not params.detokenize or not params.include_stop_str_in_output:
        raise ValueError("Kimi-Audio requires detokenize=True and include_stop_str_in_output=True")
    unsupported = [
        name
        for name in (
            "min_p",
            "min_tokens",
            "ignore_eos",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "allowed_token_ids",
            "bad_words",
            "logit_bias",
            "logits_processors",
            "structured_outputs",
            "repetition_detection",
        )
        if getattr(params, name, None)
    ]
    unsupported += [
        name
        for name in ("logprobs", "prompt_logprobs", "logprob_token_ids", "thinking_token_budget")
        if getattr(params, name, None) is not None
    ]
    if params.top_p != 1.0:
        unsupported.append("top_p")
    if unsupported:
        raise ValueError(f"Kimi-Audio sampling does not yet support: {', '.join(unsupported)}")

    overrides = (params.extra_args or {}).get("kimi_audio", {})
    if not isinstance(overrides, dict):
        raise ValueError("SamplingParams.extra_args.kimi_audio must be a parameter dictionary")
    values = dict(
        text_temperature=params.temperature,
        text_top_k=params.top_k,
        text_repetition_penalty=params.repetition_penalty,
    )
    values.update(overrides)
    KimiAudioSamplingParams(**values)

    info = prompt["model_intermediate_buffer"]
    payload = deserialize_payload(msgspec.convert(info["kimi_audio_input"], AdditionalInformationPayload))
    special = KimiAudioSpecialTokens(**payload["special_tokens"])
    # Only blanks are returned to the scheduler before completion, then one
    # msg_end. True text/audio IDs stay in the model's two histories, so the
    # native tokenizer's different EOS cannot prematurely end either stream.
    if set(params.stop_token_ids or ()) != {special.msg_end}:
        raise ValueError(f"Kimi-Audio requires stop_token_ids=[{special.msg_end}]; custom stops are not supported")
    if params.eos_token_id == special.kimia_text_blank:
        raise ValueError("Kimi-Audio scheduler blank must not be the tokenizer EOS")
    return {**prompt, "model_intermediate_buffer": {**info, "kimi_audio_request_validated": True}}


def kimi_audio_to_decoder(
    source_outputs: list, prompt: dict[str, Any], _requires_multimodal_data: bool = False
) -> list:
    """Build stage-1 inputs from completed stage-0 RequestOutputs.

    Uses Omni's existing custom_process_input_func signature. The source's
    cumulative codes.audio is authoritative; scheduler token IDs are blanks.
    Offset and vocabulary size come from the prepared prompt's model config.
    This adapter consumes full requests, not async_chunk deltas.
    """
    wire = prompt["model_intermediate_buffer"]["kimi_audio_input"]
    config = deserialize_payload(msgspec.convert(wire, AdditionalInformationPayload))
    if config["output_type"] != "both":
        raise ValueError("Kimi-Audio text-only requests must finish on stage 0")
    offset, vocab_size = config["audio_token_offset"], config["audio_vocab_size"]
    inputs = []
    for source in source_outputs:
        if not source.finished:
            raise ValueError("Kimi-Audio decoder requires the complete AR output")
        if len(source.outputs) != 1:
            raise ValueError("Kimi-Audio stage conversion expects one completion per source request")
        output = source.outputs[0]
        if output.finish_reason not in ("stop", "length"):
            raise ValueError("Kimi-Audio cannot decode an aborted or failed AR output")
        audio = output.multimodal_output["codes"]["audio"]
        if not isinstance(audio, torch.Tensor) or audio.ndim != 1 or audio.dtype not in (torch.int32, torch.int64):
            raise ValueError("Kimi-Audio codes.audio must be a one-dimensional integer tensor")
        # The AR stage already removed its six delay steps. Match official
        # generate(): filter control IDs, then remove the vocabulary offset ONCE.
        codes = audio[audio >= offset] - offset
        if torch.any(codes >= vocab_size):
            raise ValueError("Kimi-Audio output exceeds the audio codebook vocabulary")
        codes = codes.cpu().tolist()
        inputs.append(
            {
                # A zero-length prompt is not a schedulable generation request.
                # Preserve the real empty sequence separately; never synthesize
                # an audible code just to schedule the empty result.
                "prompt_token_ids": codes or [0],
                "model_intermediate_buffer": {"codes": {"audio": codes}, "meta": {"finished": True}},
            }
        )
    return inputs
