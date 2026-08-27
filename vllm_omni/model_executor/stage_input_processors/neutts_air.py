# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage input processing for NeuTTS-Air."""

from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt

NEUTTS_SPEECH_GENERATION_START_TOKEN_ID = 151669
NEUTTS_SPEECH_TOKEN_OFFSET = 151671
NEUTTS_CODEC_VOCAB_SIZE = 65536
NEUTTS_SPEECH_TOKEN_LIMIT = NEUTTS_SPEECH_TOKEN_OFFSET + NEUTTS_CODEC_VOCAB_SIZE

NEUTTS_STREAMING_CHUNK_FRAMES = 25
NEUTTS_STREAMING_LOOKFORWARD_FRAMES = 5
NEUTTS_STREAMING_LOOKBACK_FRAMES = 50
NEUTTS_STREAMING_OVERLAP_FRAMES = 1


def _ensure_token_list(token_ids: Any) -> list[int]:
    """Convert vLLM ConstantList, tensors, and iterables to Python ints."""
    if hasattr(token_ids, "_x"):
        token_ids = token_ids._x
    if isinstance(token_ids, torch.Tensor):
        values = token_ids.detach().cpu().reshape(-1).tolist()
    elif isinstance(token_ids, list):
        values = token_ids
    else:
        values = list(token_ids)
    return [int(value) for value in values]


def filter_speech_codes(token_ids: list[int]) -> list[int]:
    """Extract NeuTTS speech tokens and convert them to NeuCodec codes."""
    return [
        token_id - NEUTTS_SPEECH_TOKEN_OFFSET
        for token_id in token_ids
        if NEUTTS_SPEECH_TOKEN_OFFSET <= token_id < NEUTTS_SPEECH_TOKEN_LIMIT
    ]


def _extract_reference_codes(prompt_token_ids: list[int]) -> list[int]:
    """Read reference codes appended after the prompt's generation marker."""
    try:
        marker_index = len(prompt_token_ids) - 1 - prompt_token_ids[::-1].index(NEUTTS_SPEECH_GENERATION_START_TOKEN_ID)
    except ValueError as exc:
        raise ValueError("NeuTTS-Air streaming prompt is missing <|SPEECH_GENERATION_START|>.") from exc

    reference_codes = filter_speech_codes(prompt_token_ids[marker_index + 1 :])
    if not reference_codes:
        raise ValueError("NeuTTS-Air streaming requires reference speech codes.")
    return reference_codes


def _streaming_config(transfer_manager: Any) -> tuple[int, int, int, int]:
    connector = getattr(transfer_manager, "connector", None)
    raw_config = getattr(connector, "config", {}) or {}
    if not isinstance(raw_config, Mapping):
        raw_config = {}
    config = raw_config.get("extra", raw_config)
    if not isinstance(config, Mapping):
        config = {}

    chunk_frames = int(config.get("codec_chunk_frames", NEUTTS_STREAMING_CHUNK_FRAMES))
    lookforward_frames = int(
        config.get(
            "codec_pre_lookahead_frames",
            NEUTTS_STREAMING_LOOKFORWARD_FRAMES,
        )
    )
    lookback_frames = int(
        config.get(
            "codec_left_context_frames",
            NEUTTS_STREAMING_LOOKBACK_FRAMES,
        )
    )
    overlap_frames = int(config.get("codec_overlap_frames", NEUTTS_STREAMING_OVERLAP_FRAMES))

    if chunk_frames <= 0 or lookforward_frames < 2 * overlap_frames or lookback_frames < 0 or overlap_frames < 0:
        raise ValueError(
            "Invalid NeuTTS-Air streaming config: "
            f"chunk={chunk_frames}, lookforward={lookforward_frames}, "
            f"lookback={lookback_frames}, overlap={overlap_frames}."
        )
    return chunk_frames, lookforward_frames, lookback_frames, overlap_frames


def _request_finished(request: Any, is_finished: bool) -> bool:
    request_is_finished = getattr(request, "is_finished", None)
    return bool(is_finished or (callable(request_is_finished) and request_is_finished()))


def _stream_payload(
    *,
    request_id: str,
    codes: list[int],
    left_context_size: int,
    right_holdback_size: int,
    num_processed_tokens: int,
    chunk_frames: int,
    lookback_frames: int,
    finished: bool,
) -> OmniPayloadStruct:
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor(codes, dtype=torch.long)),
        meta=MetaStruct(
            finished=torch.tensor(finished, dtype=torch.bool),
            stream_finished=torch.tensor(finished, dtype=torch.bool),
            last_chunk=finished,
            req_id=[request_id],
            left_context_size=left_context_size,
            right_holdback_size=right_holdback_size,
            num_processed_tokens=num_processed_tokens,
            codec_streaming=True,
            codec_chunk_frames=chunk_frames,
            codec_left_context_frames=lookback_frames,
        ),
    )


def llm2neucodec_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Build NeuCodec windows matching the official NeuTTS stream decoder."""
    del multimodal_output

    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    if not request_id:
        raise ValueError("NeuTTS-Air streaming request has no request ID.")
    request_id = str(request_id)

    prompt_token_ids = _ensure_token_list(request.prompt_token_ids)
    all_token_ids = _ensure_token_list(request.all_token_ids)
    if len(all_token_ids) < len(prompt_token_ids):
        raise ValueError("NeuTTS-Air all_token_ids is shorter than its prompt.")

    reference_codes = _extract_reference_codes(prompt_token_ids)
    generated_ids = all_token_ids[len(prompt_token_ids) :]
    target_codes = filter_speech_codes(generated_ids)
    finished = _request_finished(request, is_finished)

    state_by_request = getattr(transfer_manager, "code_prompt_token_ids", None)
    if state_by_request is None:
        state_by_request = {}
        transfer_manager.code_prompt_token_ids = state_by_request
    consumed_codes = state_by_request.setdefault(request_id, [])
    consumed = len(consumed_codes)
    if consumed > len(target_codes):
        raise RuntimeError("NeuTTS-Air streaming state is ahead of the generated code stream.")
    if [int(code) for code in consumed_codes] != target_codes[:consumed]:
        raise RuntimeError("NeuTTS-Air streaming state does not match the generated code prefix.")

    chunk_frames, lookforward_frames, lookback_frames, overlap_frames = _streaming_config(transfer_manager)
    available = len(target_codes) - consumed
    continuous_codes = reference_codes + target_codes

    if not finished:
        required_frames = chunk_frames + lookforward_frames
        if available < required_frames:
            return None

        cursor = len(reference_codes) + consumed
        window_start = max(cursor - lookback_frames - overlap_frames, 0)
        window_end = min(
            cursor + required_frames + overlap_frames,
            len(continuous_codes),
        )
        window_codes = continuous_codes[window_start:window_end]
        left_context_size = cursor - window_start
        decoded_window_frames = chunk_frames + 2 * overlap_frames
        right_holdback_size = len(window_codes) - left_context_size - decoded_window_frames
        if right_holdback_size < 0:
            raise RuntimeError("NeuTTS-Air streaming window is too short to decode.")

        newly_consumed = target_codes[consumed : consumed + chunk_frames]
        consumed_codes.extend(newly_consumed)
        return _stream_payload(
            request_id=request_id,
            codes=window_codes,
            left_context_size=left_context_size,
            right_holdback_size=right_holdback_size,
            num_processed_tokens=len(newly_consumed),
            chunk_frames=chunk_frames,
            lookback_frames=lookback_frames,
            finished=False,
        )

    remaining_codes = target_codes[consumed:]
    if not remaining_codes:
        # A zero-token non-AR request is finished by the scheduler without
        # calling Stage 1. Send one context-only code so Stage 1 can observe
        # stream_finished and release its overlap-add cache. The metadata says
        # that this transport wake-up contains zero newly processed frames.
        return _stream_payload(
            request_id=request_id,
            codes=continuous_codes[-1:],
            left_context_size=1,
            right_holdback_size=0,
            num_processed_tokens=0,
            chunk_frames=chunk_frames,
            lookback_frames=lookback_frames,
            finished=True,
        )

    window_start = max(
        len(continuous_codes) - (lookback_frames + overlap_frames + len(remaining_codes)),
        0,
    )
    left_context_size = max(
        len(continuous_codes) - window_start - len(remaining_codes) - overlap_frames,
        0,
    )
    window_codes = continuous_codes[window_start:]
    consumed_codes.extend(remaining_codes)
    return _stream_payload(
        request_id=request_id,
        codes=window_codes,
        left_context_size=left_context_size,
        right_holdback_size=0,
        num_processed_tokens=len(remaining_codes),
        chunk_frames=chunk_frames,
        lookback_frames=lookback_frames,
        finished=True,
    )


def llm2neucodec_sync(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Forward generated NeuCodec codes directly in synchronous mode."""
    code2wav_inputs: list[OmniTokensPrompt] = []
    for output_wrapper in source_outputs:
        output = output_wrapper.outputs[0]
        speech_codes = filter_speech_codes(list(output.token_ids))
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=speech_codes,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs
