# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Kimi-Audio request admission and complete AR-to-decoder input conversion."""

import hashlib
from collections.abc import Sequence
from typing import Any

import msgspec
import torch

from vllm_omni.data_entry_keys import deserialize_payload
from vllm_omni.engine import AdditionalInformationPayload
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.kimi_audio.prompt import KimiAudioSpecialTokens
from vllm_omni.model_executor.models.kimi_audio.sampling import KimiAudioSamplingParams


def prepare_kimi_audio_request(prompt: dict[str, Any], sampling_params_list: Sequence[Any]) -> dict[str, Any]:
    """Validate stage-0 inputs through Omni's existing prompt_transform_func.

    Chat arrives after rendering; offline and speech prompts are not yet native
    EngineInputs. Both carry the same model buffer and resolved stage sampling
    settings. Finalize the layout/cache identity here, after HTTP prompt extras
    and before constructing the engine request. Never rebuild either stream.
    """
    if not isinstance(prompt, dict):
        raise OmniClientError("Kimi-Audio requires rendered chat or prepared token inputs")
    info = prompt.get("model_intermediate_buffer")
    if not isinstance(info, dict) or "kimi_audio_input" not in info:
        raise OmniClientError("Kimi-Audio input is missing; use the Kimi chat renderer or prepare_kimi_audio_inputs")
    additional_info = prompt.get("additional_information")
    if additional_info is None:
        additional_info = {}
    if not isinstance(additional_info, dict) or any(not isinstance(key, str) for key in additional_info):
        raise OmniClientError("Kimi-Audio additional_information must be a dictionary with string keys")
    if any(key.startswith("kimi_audio_") for key in additional_info):
        raise OmniClientError("Kimi-Audio internal request state cannot be supplied through additional_information")

    params = sampling_params_list[0]
    if params.n != 1:
        raise OmniClientError("Kimi-Audio currently requires one completion per request")
    if not params.detokenize or not params.include_stop_str_in_output:
        raise OmniClientError("Kimi-Audio requires detokenize=True and include_stop_str_in_output=True")
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
        raise OmniClientError(f"Kimi-Audio sampling does not yet support: {', '.join(unsupported)}")

    overrides = (params.extra_args or {}).get("kimi_audio", {})
    if not isinstance(overrides, dict):
        raise OmniClientError("SamplingParams.extra_args.kimi_audio must be a parameter dictionary")
    values = dict(
        text_temperature=params.temperature,
        text_top_k=params.top_k,
        text_repetition_penalty=params.repetition_penalty,
    )
    values.update(overrides)
    try:
        KimiAudioSamplingParams(**values)
    except (TypeError, ValueError) as exc:
        raise OmniClientError(str(exc)) from None

    payload = deserialize_payload(msgspec.convert(info["kimi_audio_input"], AdditionalInformationPayload))
    special = KimiAudioSpecialTokens(**payload["special_tokens"])
    # Only blanks are returned to the scheduler before completion, then one
    # msg_end. True text/audio IDs stay in the model's two histories, so the
    # native tokenizer's different EOS cannot prematurely end either stream.
    if set(params.stop_token_ids or ()) != {special.msg_end}:
        raise OmniClientError(f"Kimi-Audio requires stop_token_ids=[{special.msg_end}]; custom stops are not supported")
    if params.eos_token_id == special.kimia_text_blank:
        raise OmniClientError("Kimi-Audio scheduler blank must not be the tokenizer EOS")
    # Public chat preprocessing may supply a caller cache_salt after rendering.
    # Combine it with the full layout instead of letting it replace the text
    # stream and task identity absent from scheduler-side placeholder IDs.
    caller_salt = prompt.get("cache_salt")
    if caller_salt is not None and not isinstance(caller_salt, str):
        raise OmniClientError("Kimi-Audio cache_salt must be a string")
    cache_salt = hashlib.sha256(msgspec.msgpack.encode((info["kimi_audio_input"], caller_salt))).hexdigest()
    return {
        **prompt,
        "cache_salt": cache_salt,
        "model_intermediate_buffer": {**info, "kimi_audio_request_validated": True},
    }


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
        raise OmniClientError(
            "Kimi-Audio output_type='text' cannot feed the audio decoder; "
            "use output_type='both' for audio output or select only the text stage"
        )
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
