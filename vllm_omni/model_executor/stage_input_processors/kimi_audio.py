# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Kimi-Audio admission and complete/streaming AR-to-decoder conversion."""

import hashlib
import secrets
from collections.abc import Sequence
from typing import Any

import msgspec
import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct, deserialize_payload
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
    meta = additional_info.get("meta")
    if meta is None:
        meta = {}
    if not isinstance(meta, dict):
        raise OmniClientError("Kimi-Audio additional_information.meta must be a dictionary")

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
    special = KimiAudioSpecialTokens(**payload["meta"]["special_tokens"])
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
    # Downstream conversion receives the original prompt. Omni copies processed
    # additional_information.meta back to it, but not model_intermediate_buffer.
    # Resolve an omitted seed once: None is dropped on the wire and would leave
    # a stale audio_seed when a caller reuses a previously submitted prompt.
    audio_seed = params.seed if params.seed is not None else secrets.randbits(63)
    return {
        **prompt,
        "cache_salt": cache_salt,
        "additional_information": {**additional_info, "meta": {**meta, "audio_seed": audio_seed}},
        "model_intermediate_buffer": {
            **info,
            "kimi_audio_request_validated": True,
        },
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
    meta = config["meta"]
    if meta["output_type"] != "both":
        raise OmniClientError(
            "Kimi-Audio output_type='text' cannot feed the audio decoder; "
            "use output_type='both' for audio output or select only the text stage"
        )
    offset, vocab_size = meta["audio_token_offset"], meta["audio_vocab_size"]
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
                "model_intermediate_buffer": {
                    "codes": {"audio": codes},
                    "meta": {
                        "finished": True,
                        "audio_seed": prompt["additional_information"]["meta"]["audio_seed"],
                    },
                },
            }
        )
    return inputs


def kimi_audio_to_decoder_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Send new semantic codes through Omni's existing async chunk connector.

    AR outputs are deltas with the six delay steps already removed. Keep one
    code pending beyond each 30-code block: if generation stops on an exact
    block boundary, that last block still reaches the decoder with is_final.
    Acoustic lookahead and waveform overlap belong to the decoder, so no
    previously sent codes are replayed here.
    """
    request_id = request.external_req_id
    try:
        state = transfer_manager.request_payload.setdefault(request_id, {})
        if "kimi_audio" not in state:
            wire = request.model_intermediate_buffer["kimi_audio_input"]
            config = deserialize_payload(msgspec.convert(wire, AdditionalInformationPayload))
            meta = config["meta"]
            if meta["output_type"] != "both":
                raise ValueError("Kimi-Audio audio streaming requires output_type='both'")
            state["kimi_audio"] = {
                "offset": meta["audio_token_offset"],
                "vocab_size": meta["audio_vocab_size"],
                "chunk_seq": 0,
            }
        state = state["kimi_audio"]
        pending = transfer_manager.code_prompt_token_ids[request_id]
        audio = (multimodal_output or {}).get("codes", {}).get("audio")
        if audio is not None:
            if not isinstance(audio, torch.Tensor) or audio.ndim != 1 or audio.dtype not in (torch.int32, torch.int64):
                raise ValueError("Kimi-Audio codes.audio must be a one-dimensional integer tensor")
            codes = audio[audio >= state["offset"]] - state["offset"]
            if torch.any(codes >= state["vocab_size"]):
                raise ValueError("Kimi-Audio output exceeds the audio codebook vocabulary")
            pending.extend(codes.cpu().tolist())
    except Exception as exc:
        # The connector swallows processor exceptions. Record the request ID
        # through its existing reporting channel, too. This is diagnostic:
        # the scheduler only logs send failures; it does not propagate an
        # immediate client error. Receive deadlines remain framework-owned.
        transfer_manager.record_send_failure(request.request_id, str(exc))
        return None

    # The scheduler is authoritative, including max_tokens termination while
    # the audio/text streams have not both reached their native end markers.
    finished = bool(is_finished or request.is_finished())
    if not finished and len(pending) <= 30:
        return None
    length = len(pending) if finished else ((len(pending) - 1) // 30) * 30
    chunk = pending[:length]
    del pending[:length]
    payload = OmniPayloadStruct(
        # [frames, one codebook] uses the connector's existing tensor-payload
        # path. Each chunk is one scheduling unit, even with a small token
        # budget; the scheduler cannot split the acoustic block mid-forward.
        codes=CodesStruct(audio=torch.tensor(chunk, dtype=torch.long).reshape(-1, 1)),
        meta=MetaStruct(
            finished=torch.tensor(finished, dtype=torch.bool),
            stream_finished=torch.tensor(finished, dtype=torch.bool),
            chunk_seq=state["chunk_seq"],
            audio_seed=request.sampling_params.seed,
        ),
    )
    state["chunk_seq"] += 1
    # The transfer adapter owns cleanup on completion/abort. An empty terminal
    # sequence has no acoustic state to flush because of the holdback above.
    return payload
