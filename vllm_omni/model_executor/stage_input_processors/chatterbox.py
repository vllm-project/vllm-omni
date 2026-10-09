# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox stage handoff: T3's speech tokens to S3Gen.

``t3_to_s3gen_async_chunk`` streams (``async_chunk: true``). It runs in stage
0, once per sampled token, on either data plane, and sends stage 1 a growing
prefix of the utterance in chunks. ``t3_to_s3gen`` is the other mode: the
orchestrator calls it once per request and stage 1 receives the finished
utterance as a stream of one final chunk, so stage 1 has a single input
contract.
"""

from typing import Any

import torch

from vllm_omni.data_entry_keys import CodesStruct, EmbeddingsStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

SPEECH_TOKEN_LIMIT = ChatterboxConfig().speech_token_limit


def t3_to_s3gen_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Collect one request's sampled speech tokens and emit a chunk when one is due.

    The schedule is CosyVoice 2's: the first chunk is ``codec_chunk_frames``
    tokens plus the padding that aligns the reference prompt to that size,
    each later chunk is ``codec_stream_scale_factor`` times the one before up
    to ``codec_max_chunk_frames``, and a chunk that is not the last waits for
    ``codec_pre_lookahead_frames`` tokens beyond its own, which the flow
    reads ahead and stage 1 does not play. A chunk is sent as the whole
    utterance so far through that lookahead, with the offset of its new part.

    Tokens come from the request snapshot, never from ``multimodal_output``.
    The scheduler-side transport hands over the full output history on every
    call. The runner-side transport hands it over until the first chunk was
    sent and from then on only the count and the last id, one call per
    sampled token.

    Per-request state is the two entries the transport frees when a request
    finishes or is aborted: ``request_payload[external_req_id]`` and
    ``code_prompt_token_ids[external_req_id]``.

    Args:
        transfer_manager: The stage's transport, one of two classes with no
            common type. Its connector's config holds the four ``codec_*``
            settings of the deploy file.
        multimodal_output: The step's payload for the request, unused.
        request: The transport's snapshot of the request, a different type
            on each: ``external_req_id``, ``output_token_ids``,
            ``output_token_count``, ``last_output_token_id`` (runner-side
            only) and ``additional_information``.
        is_finished: Whether the request ended. The scheduler-side transport
            sets it on the call that brings the last token, the runner-side
            one on a call of its own, also after an abort.

    Returns:
        The chunk, or None when none is due. A finished request with nothing
        left to send gets an explicit empty ``codes.audio``: without the key
        stage 1 would run the placeholder prompt it was started with.

    Raises:
        KeyError: If the deploy file's connector lacks one of the settings.
        RuntimeError: If the request has sampled more tokens than were taken
            and the snapshot no longer has them: the audio would be missing
            them with no other symptom.
    """
    request_id = request.external_req_id
    # A connector's config is its ``extra`` section of the deploy file.
    settings = transfer_manager.connector.config
    chunk, lookahead = settings["codec_chunk_frames"], settings["codec_pre_lookahead_frames"]
    largest, growth = settings["codec_max_chunk_frames"], settings["codec_stream_scale_factor"]

    state = transfer_manager.request_payload.get(request_id)
    if state is None:
        reference = deserialize_additional_information(request.additional_information)["embed"]
        state = transfer_manager.request_payload[request_id] = {
            # Output tokens taken from the snapshot, control tokens included.
            "seen": 0,
            # Speech tokens whose audio earlier chunks already cover.
            "sent": 0,
            # The next chunk's new tokens.
            "hop": chunk + -reference["speech_token"].shape[1] % chunk,
            "chunks": 0,
            "terminal_sent": False,
            # Travels with the first chunk and is dropped here.
            "reference": EmbeddingsStruct(
                speech_token=reference["speech_token"],
                speech_feat=reference["speech_feat"],
                embedding=reference["embedding"],
            ),
        }
    if state["terminal_sent"]:
        return None

    seen = state["seen"]
    if len(request.output_token_ids) > seen:
        new_tokens = request.output_token_ids[seen:]
    elif request.output_token_count == seen + 1:
        new_tokens = [request.last_output_token_id]
    elif request.output_token_count == seen:
        new_tokens = []
    else:
        raise RuntimeError(
            f"Chatterbox request {request_id} has sampled {request.output_token_count} tokens, its chunk "
            f"processor took {seen} and the snapshot holds {len(request.output_token_ids)}: the tokens between "
            "would be missing from the audio"
        )
    state["seen"] = seen + len(new_tokens)
    # Drops the stop token.
    tokens = transfer_manager.code_prompt_token_ids[request_id]
    tokens.extend(token for token in new_tokens if token < SPEECH_TOKEN_LIMIT)

    sent, hop = state["sent"], state["hop"]
    if is_finished:
        state["terminal_sent"] = True
        if len(tokens) <= sent:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True)),
            )
        end = len(tokens)
    else:
        end = sent + hop + lookahead
        if len(tokens) < end:
            return None
        state["sent"] = sent + hop
        state["chunks"] += 1
        state["hop"] = min(max(chunk, largest), chunk * growth ** state["chunks"])
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor(tokens[:end], dtype=torch.long)),
        meta=MetaStruct(
            finished=torch.tensor(is_finished),
            stream_finished=torch.tensor(is_finished),
            left_context_size=sent,
        ),
        embed=state.pop("reference", None),
    )


# The scheduler-side transport calls a processor on a step that has sampled
# tokens but no tensor payload only when the processor asks for it.
setattr(t3_to_s3gen_async_chunk, "requires_token_updates", True)


def t3_to_s3gen(
    source_outputs: list,
    prompt: dict,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build stage 1's input from stage 0's finished output.

    The orchestrator calls this once per request, with that request's stage 0
    output and its own prompt. The payload is shaped as a stream of one final
    chunk, so stage 1 has a single input contract in both modes.

    Args:
        source_outputs: Stage 0's output for the request.
        prompt: The request's own prompt, from ``conditioning.build_prompt``.
        _requires_multimodal_data: Unused; part of the processor signature.

    Returns:
        One prompt per finished output: the valid speech tokens, with the
        request's reference and the stream metadata.

    Raises:
        RuntimeError: If stage 0 produced no speech tokens for the request.
    """
    reference = prompt["additional_information"]["embed"]
    stage_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        if not source_output.finished:
            continue
        # Drops the stop token, and the placeholder prompt ids should an
        # engine version report the prompt in the output history.
        codes = [token for token in source_output.outputs[0].cumulative_token_ids if token < SPEECH_TOKEN_LIMIT]
        if not codes:
            raise RuntimeError(f"Chatterbox T3 produced no speech tokens for request {source_output.request_id}")
        stage_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codes,
                additional_information={
                    "embed": {
                        "speech_token": reference["speech_token"],
                        "speech_feat": reference["speech_feat"],
                        "embedding": reference["embedding"],
                    },
                    "meta": {"stream_finished": torch.tensor(True), "left_context_size": 0},
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return stage_inputs
