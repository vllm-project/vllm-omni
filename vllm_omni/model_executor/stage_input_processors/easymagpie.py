# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Transfer stacked acoustic codes from the talker to the native codec.

Stage 0 emits ``[frames, codebooks]`` codes. The stateful Stage 1 consumes one
placeholder and one code row per newly generated acoustic frame.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.request import RequestStatus

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayload, OmniPayloadStruct
from vllm_omni.engine.serialization import deserialize_additional_information

logger = init_logger(__name__)


# Base codebook size, excluding control tokens.
_CODEBOOK_SIZE = 1024
_CODEC_PADDING = -1
_SENDER_FRAME_FASTPATH = os.environ.get("EASYMAGPIE_SENDER_CPU_FRAME_FASTPATH", "1") == "1"
_SENDER_FRAME_COUNTS: Counter[str] = Counter()


def sender_frame_status():
    return {"enabled": _SENDER_FRAME_FASTPATH, **dict(_SENDER_FRAME_COUNTS)}


def _owned_cpu_integer_frame(tensor):
    # The selected sender receives an already-completed owning CPU snapshot.
    # Reusing a row preserves the same storage as the old detach/view chain.
    return (
        type(tensor) is torch.Tensor
        and tensor.device.type == "cpu"
        and tensor.dtype == torch.long
        and not tensor.requires_grad
        and tensor.ndim in (1, 2)
        and tensor.shape[-1] <= 64
        and tensor.is_contiguous()
    )


def _empty_finished_payload() -> dict[str, Any]:
    """Release Stage 1 when no usable codec frames were produced."""
    return {
        "codes": {"audio": torch.zeros(0, dtype=torch.long)},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


def _filter_audio_codes(audio_codes: torch.Tensor) -> torch.Tensor:
    """Drop all-zero (padding/warm-up), negative, and special-token frames.

    Special audio tokens (bos/eos/mask) live at ``codebook_size + offset`` — any
    frame containing one is an out-of-band control frame, not audio, so it is
    removed here (the decoder additionally clamps as a safety net).
    """
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0 or audio_codes.ndim != 2:
        return audio_codes
    valid_mask = (
        (audio_codes >= 0).all(dim=1) & audio_codes.any(dim=1) & (audio_codes.max(dim=1).values < _CODEBOOK_SIZE)
    )
    return audio_codes[valid_mask]


def talker2code2wav_token_only(
    source_outputs: list,
    prompt=None,
    _requires_multimodal_data: bool = False,
) -> list:
    """Sync-side one-placeholder-per-frame Stage-1 input.

    The real codec rows ship via the worker connector payload built by
    :func:`talker2code2wav_full_payload`.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    code2wav_inputs: list = []
    for talker_output in source_outputs:
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output if isinstance(getattr(output, "multimodal_output", None), dict) else {}
        audio = _extract_audio_codes(mm)
        token_ids = getattr(output, "cumulative_token_ids", []) or []
        seq_len = max(len(token_ids) - 1, 0)

        if isinstance(audio, torch.Tensor) and audio.numel() > 0:
            audio = _filter_audio_codes(audio.to(torch.long))
            if seq_len > 0 and audio.ndim == 2 and int(audio.shape[0]) > seq_len:
                audio = audio[-seq_len:]
            num_frames = int(audio.shape[0]) if audio.ndim == 2 else 0
        else:
            num_frames = 0

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * num_frames,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def talker2code2wav_full_payload(transfer_manager, multimodal_output, request, is_finished: bool = False):
    """Send accumulated codec frames through the worker connector.

    ``is_finished`` is part of the transfer-adapter callback contract.
    """
    pooling_output = multimodal_output
    del transfer_manager, is_finished
    rid = getattr(request, "request_id", "?")
    if not isinstance(pooling_output, dict):
        logger.warning(
            "easymagpie.talker2code2wav_full_payload: pooling_output is %s (not dict) for req=%s",
            type(pooling_output).__name__,
            rid,
        )
        return _empty_finished_payload()

    audio = pooling_output.get("codes.audio")
    if audio is None:
        codes_nested = pooling_output.get("codes")
        if isinstance(codes_nested, dict):
            audio = codes_nested.get("audio")
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        logger.warning(
            "easymagpie.talker2code2wav_full_payload: missing/empty codes.audio (keys=%s) for req=%s",
            list(pooling_output.keys()),
            rid,
        )
        return _empty_finished_payload()

    raw_frames = int(audio.shape[0]) if audio.ndim == 2 else 0
    audio = _filter_audio_codes(audio.to(torch.long))
    kept_frames = int(audio.shape[0]) if audio.ndim == 2 else 0
    logger.debug(
        "easymagpie.talker2code2wav_full_payload: req=%s accumulated %d frames (%d after filtering control/padding).",
        rid,
        raw_frames,
        kept_frames,
    )
    if audio.numel() == 0:
        return _empty_finished_payload()

    output_token_ids = list(getattr(request, "output_token_ids", None) or [])
    seq_len = max(len(output_token_ids) - 1, 0)
    if seq_len > 0 and audio.ndim == 2 and int(audio.shape[0]) > seq_len:
        audio = audio[-seq_len:]

    return {
        "codes": {"audio": audio.to(device="cpu", dtype=torch.long).contiguous()},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


def _extract_last_frame(multimodal_output: OmniPayload | dict[str, Any]) -> torch.Tensor | None:
    audio_codes = _extract_audio_codes(multimodal_output)
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if _SENDER_FRAME_FASTPATH and _owned_cpu_integer_frame(audio_codes):
        frame = audio_codes[-1] if audio_codes.ndim == 2 else audio_codes
        if audio_codes.ndim == 2 and not any(frame.tolist()):
            _SENDER_FRAME_COUNTS["zero_rows"] += 1
            return None
        _SENDER_FRAME_COUNTS["extracted_rows"] += 1
        return frame
    if _SENDER_FRAME_FASTPATH:
        _SENDER_FRAME_COUNTS["extract_fallbacks"] += 1
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None
        return frame.to(torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        return audio_codes.to(torch.long).reshape(-1)
    raise ValueError(f"Invalid audio_codes shape for EasyMagpie async_chunk: {tuple(audio_codes.shape)}")


def _resolve_speech_delay(transfer_manager: Any) -> int:
    """Return and cache the number of non-audio frames before speech starts."""
    cached = getattr(transfer_manager, "_easymagpie_speech_delay", None)
    if cached is not None:
        return cached

    delay = int(getattr(transfer_manager.config.hf_config, "streaming_speech_delay", 0) or 0)
    transfer_manager._easymagpie_speech_delay = delay
    logger.debug("easymagpie: resolved streaming_speech_delay=%d (leading warm-up frames dropped)", delay)
    return delay


def _persistent_state(transfer_manager: Any, attr: str) -> dict:
    """Lazily create a request-keyed ``int`` dict on the transfer manager.

    These survive the scheduler's per-segment reset of ``output_token_ids`` /
    the codec buffer, so warm-up and emission accounting stay continuous across
    the segment stops that a streaming (chunk-by-chunk) request goes through.
    """
    state = getattr(transfer_manager, attr, None)
    if state is None:
        state = defaultdict(int)
        setattr(transfer_manager, attr, state)
    return state


def _persistent_list_state(transfer_manager: Any, attr: str) -> dict:
    """Lazily create a request-keyed ``list`` dict on the transfer manager."""
    state = getattr(transfer_manager, attr, None)
    if state is None:
        state = defaultdict(list)
        setattr(transfer_manager, attr, state)
    return state


def _is_true_request_finish(request: Any) -> bool:
    """True only at the real end of the utterance, not at a segment stop.

    Mirrors the transfer adapter's own ``request.is_finished() and not
    request.resumable`` rule: a resumable streaming request reports
    ``is_finished()`` at every segment boundary while still expecting more
    input, so it must not be treated as the terminal finish.
    """
    return bool(getattr(request, "is_finished", lambda: False)()) and not bool(getattr(request, "resumable", False))


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Emit newly generated time-major codec rows to the stateful native codec.

    ``multimodal_output`` must retain this name because the transfer adapter
    passes it by keyword.
    """
    request_id = request.external_req_id
    if _is_failed_codec_request(request):
        _cleanup_codec_request(transfer_manager, request_id)
        return None
    # Queued sends outlive Request status changes; use their captured finish flags.
    send_finish = getattr(transfer_manager, "_easymagpie_send_finish", None)
    if send_finish is None:
        finished = bool(is_finished or request.is_finished())
        true_finished = _is_true_request_finish(request)
    else:
        true_finished, segment_finished = send_finish
        finished = bool(true_finished or segment_finished)
    info = deserialize_additional_information(getattr(request, "additional_information", None))
    reset_codec = finished and bool(info.get("reset_codec_on_segment", False))
    true_finished = true_finished or reset_codec
    if not isinstance(multimodal_output, Mapping) and not finished:
        return None

    if isinstance(multimodal_output, Mapping):
        frame = _extract_last_frame(multimodal_output)
        # Keep the terminal control row so the codec can retain any valid subframe before it.
        delay_state = _persistent_state(transfer_manager, "_emp_request_speech_delay")
        if request_id not in delay_state:
            base_speech_delay = _resolve_speech_delay(transfer_manager)
            text_prefill_num = int(info.get("text_prefill_num", 0) or 0)
            if not 0 <= text_prefill_num <= base_speech_delay:
                raise ValueError(
                    f"Invalid EasyMagpie text_prefill_num={text_prefill_num} for speech delay {base_speech_delay}"
                )
            delay_state[request_id] = base_speech_delay - text_prefill_num
        speech_delay = delay_state[request_id]

        # Count actual predicted code frames across segment stops. The prefill
        # callback has no frame and must not consume one of the remaining warm-up
        # positions.
        seen_state = _persistent_state(transfer_manager, "_emp_seen_frames")
        _ = seen_state[request_id]
        is_warmup = False
        if frame is not None:
            seen_state[request_id] += 1
            frame_index = seen_state[request_id]
            is_warmup = speech_delay > 0 and frame_index <= speech_delay
        # Accumulate real frames into a request-persistent buffer. The framework's
        # per-segment buffer can reset; this one keeps the acoustic stream
        # continuous regardless of how the text was chunked.
        frame_buffer = _persistent_list_state(transfer_manager, "_emp_frame_buffer")
        if frame is not None and not is_warmup:
            # ``multimodal_output`` is already a CPU snapshot. Keep it as a
            # tensor so codec chunks can be assembled with a single stack
            # instead of a Python list round-trip.
            if _SENDER_FRAME_FASTPATH and _owned_cpu_integer_frame(frame) and frame.ndim == 1:
                frame_row = frame
                _SENDER_FRAME_COUNTS["reused_rows"] += 1
            else:
                frame_row = frame.detach().to(device="cpu", dtype=torch.long).reshape(-1).contiguous()
                if _SENDER_FRAME_FASTPATH:
                    _SENDER_FRAME_COUNTS["row_fallbacks"] += 1
            frame_buffer[request_id].append(frame_row)
            # Keep the framework buffer populated too (some connector bookkeeping
            # counts active requests by its non-empty per-request lists).
            transfer_manager.code_prompt_token_ids[request_id].append(frame_row)
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    raw_startup_chunks = cfg.get("codec_startup_chunk_frames", [])
    if not isinstance(raw_startup_chunks, (list, tuple)):
        raise ValueError(
            "Invalid EasyMagpie codec chunk config: codec_startup_chunk_frames "
            f"must be a list, got {type(raw_startup_chunks).__name__}"
        )
    startup_chunk_sizes = [int(value) for value in raw_startup_chunks]
    if chunk_size <= 0 or any(value <= 0 for value in startup_chunk_sizes):
        raise ValueError(
            f"Invalid EasyMagpie codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_startup_chunk_frames={startup_chunk_sizes}"
        )
    # Track one absolute emission high-water mark across resumable text
    # segments so repeated segment-finish notifications cannot duplicate frames.
    frame_buffer = _persistent_list_state(transfer_manager, "_emp_frame_buffer")
    buffer = frame_buffer[request_id]
    base_state = _persistent_state(transfer_manager, "_emp_frame_buffer_base")
    base_index = base_state[request_id]
    length = base_index + len(buffer)

    emitted_state = _persistent_state(transfer_manager, "_emp_emitted_frames")
    emitted = emitted_state[request_id]
    emitted_chunks_state = _persistent_state(transfer_manager, "_emp_emitted_chunks")
    emitted_chunks = emitted_chunks_state[request_id]

    if length <= 0:
        if finished:
            if true_finished:
                _cleanup_codec_request(transfer_manager, request_id)
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(
                    finished=torch.tensor(True, dtype=torch.bool),
                    codec_streaming=not reset_codec,
                ),
            )
        return None

    pending = length - emitted
    if pending <= 0:
        # Nothing new to emit. Never re-emit already-sent frames. Dialogue
        # boundaries still carry a reset marker to the codec scheduler.
        if true_finished:
            _cleanup_codec_request(transfer_manager, request_id)
        if reset_codec:
            return OmniPayloadStruct(meta=MetaStruct(codec_streaming=False))
        return None

    # Startup targets ramp independently from the steady codec chunk size.
    target = startup_chunk_sizes[emitted_chunks] if emitted_chunks < len(startup_chunk_sizes) else chunk_size
    # A resumable segment stop is not an audio flush: retain a partial body
    # across text-input segments so steady codec chunks keep one logical size.
    # Terminal request finishes and dialogue resets may emit a short final tail.
    if not true_finished and pending < target:
        return None
    context_length = pending if true_finished else min(pending, target)

    new_end = emitted + context_length
    relative_start = emitted - base_index
    relative_end = new_end - base_index
    code_predictor_codes = torch.stack(buffer[relative_start:relative_end], dim=0).contiguous()
    if true_finished and context_length < chunk_size:
        padding = code_predictor_codes.new_full(
            (chunk_size - context_length, code_predictor_codes.shape[1]),
            _CODEC_PADDING,
        )
        code_predictor_codes = torch.cat((code_predictor_codes, padding), dim=0)

    emitted_state[request_id] = new_end
    emitted_chunks_state[request_id] += 1
    drop = new_end - base_index
    if drop > 0:
        del buffer[:drop]
        base_state[request_id] = new_end
    if true_finished:
        _cleanup_codec_request(transfer_manager, request_id)

    return OmniPayloadStruct(
        codes=CodesStruct(audio=code_predictor_codes),
        meta=MetaStruct(
            left_context_size=0,
            finished=torch.tensor(finished, dtype=torch.bool),
            codec_streaming=not reset_codec,
        ),
    )


def _is_failed_codec_request(request: Any) -> bool:
    return getattr(request, "status", None) in (RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR)


def _cleanup_codec_request(transfer_manager: Any, request_id: str) -> None:
    for name in (
        "_emp_emitted_frames",
        "_emp_emitted_chunks",
        "_emp_seen_frames",
        "_emp_request_speech_delay",
        "_emp_frame_buffer_base",
        "_emp_frame_buffer",
    ):
        getattr(transfer_manager, name, {}).pop(request_id, None)


def _extract_audio_codes(mm: Mapping | dict[str, Any] | None) -> torch.Tensor | None:
    """Read the talker acoustic codes, preferring nested ``codes.audio`` then
    the single-stage ``audio_codes`` key."""
    if not isinstance(mm, Mapping):
        return None
    codes = mm.get("codes")
    if isinstance(codes, Mapping):
        audio = codes.get("audio")
        if isinstance(audio, torch.Tensor):
            return audio
    audio = mm.get("audio_codes")
    if isinstance(audio, torch.Tensor):
        return audio
    return None
