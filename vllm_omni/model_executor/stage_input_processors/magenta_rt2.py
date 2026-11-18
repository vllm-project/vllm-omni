# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Talker -> SpectroStream Code2Wav input processors for Magenta RealTime 2.

Stage 0 (``MagentaRT2ForConditionalGeneration``) emits 12 SpectroStream RVQ
tokens per 40 ms frame under ``("codes", "audio")``. Unlike PersonaPlex, there is
no inter-codebook acoustic delay in Magenta RealTime 2, so frames map 1-to-1 to
SpectroStream without dropping warmup frames. The processors flatten valid
``[F, 12]`` code frames into contiguous ``[F * 12]`` sequences for
``SpectroStreamCode2Wav``.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
)

NUM_CODEBOOKS = 12


def _empty_finished_payload() -> OmniPayloadStruct:
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )


def _flatten_frames(audio: Any) -> torch.Tensor:
    if audio is None:
        return torch.empty(0, dtype=torch.long)
    tensor = audio if isinstance(audio, torch.Tensor) else torch.as_tensor(audio)
    if tensor.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    tensor = tensor.to(dtype=torch.long)
    if tensor.ndim == 1:
        if tensor.numel() % NUM_CODEBOOKS != 0:
            return torch.empty(0, dtype=torch.long)
        tensor = tensor.view(-1, NUM_CODEBOOKS)
    elif tensor.ndim == 2:
        if tensor.shape[1] < NUM_CODEBOOKS:
            return torch.empty(0, dtype=torch.long)
        tensor = tensor[:, :NUM_CODEBOOKS]
    else:
        return torch.empty(0, dtype=torch.long)
    valid = (tensor >= 0).all(dim=1)
    tensor = tensor[valid]
    if tensor.shape[0] == 0:
        return torch.empty(0, dtype=torch.long)
    return tensor.contiguous().reshape(-1)


def _extract_codes(src: Any) -> Any:
    if not isinstance(src, dict):
        return None
    nested = src.get("codes")
    audio = nested.get("audio") if isinstance(nested, dict) else None
    return audio if audio is not None else src.get("codes.audio")


def talker2code2wav_token_only(
    source_outputs: list,
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list:
    """Sync ``process_engine_inputs``: build SpectroStreamCode2Wav placeholder tokens."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    del prompt, _requires_multimodal_data
    inputs: list = []
    for talker_output in source_outputs:
        if not getattr(talker_output, "finished", False):
            continue
        output = talker_output.outputs[0]
        token_ids = getattr(output, "cumulative_token_ids", None) or getattr(output, "token_ids", None) or []
        n_frames = max(len(token_ids) - 1, 0)
        inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * (NUM_CODEBOOKS * n_frames),
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return inputs


def talker2code2wav_full_payload(
    transfer_manager: Any = None,
    pooling_output: Any = None,
    request: Any = None,
    is_finished: bool = False,
    **kwargs: Any,
) -> OmniPayloadStruct:
    """Collect accumulated ``[F, 12]`` Stage 0 codes into flat ``[F * 12]`` payload."""
    del transfer_manager, is_finished
    audio = None
    for source in (
        getattr(request, "additional_information", None),
        getattr(request, "additional_information_cpu", None),
        pooling_output,
        kwargs.get("multimodal_output"),
    ):
        audio = _extract_codes(source)
        if audio is not None:
            break
    flat = _flatten_frames(audio)
    if flat.numel() == 0:
        return _empty_finished_payload()
    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any = None,
    request: Any = None,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Streaming chunk extraction for Stage 0 -> SpectroStreamCode2Wav (40 ms ticks).

    Unlike PersonaPlex, Magenta RealTime 2 has zero inter-codebook acoustic delay,
    so every emitted ``[12]`` code row maps immediately to one 40 ms audio frame.
    """
    request_id = getattr(request, "external_req_id", getattr(request, "request_id", "?"))
    finished = bool(is_finished and not getattr(request, "resumable", False))

    request_payload = getattr(transfer_manager, "request_payload", None)
    if not isinstance(request_payload, dict):
        request_payload = {}
        if transfer_manager is not None:
            transfer_manager.request_payload = request_payload
    state = request_payload.setdefault(request_id, {})
    frames: list[torch.Tensor] = state.setdefault("magenta_rt2_frames", [])

    # Explicit None fallback (avoid truthiness on Tensors).
    # MRv2 Stage 0 emits per-step fresh codes via multimodal_output (`mm_outputs_fresh_per_step=True`);
    # fall back to cumulative `request.additional_information` if needed.
    audio = _extract_codes(multimodal_output)
    is_cumulative_fallback = False
    if audio is None:
        audio = _extract_codes(getattr(request, "additional_information", None))
        is_cumulative_fallback = True
    if isinstance(audio, torch.Tensor) and audio.numel() > 0:
        a = audio if audio.ndim == 2 else audio.reshape(-1, NUM_CODEBOOKS)
        rows = a[-1:] if is_cumulative_fallback else a
        for row in rows:
            row_l = row.to(torch.long).cpu()
            if bool((row_l >= 0).all()):
                frames.append(row_l)

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk = int(cfg.get("codec_chunk_frames", 1))
    initial_chunk = int(cfg.get("initial_codec_chunk_frames") or 0)
    if chunk <= 0 or initial_chunk < 0:
        raise ValueError(
            "MagentaRT2 codec chunk sizes must be positive/non-negative: "
            f"codec_chunk_frames={chunk}, initial_codec_chunk_frames={initial_chunk}"
        )
    target_frames = initial_chunk if not state.get("magenta_rt2_emitted") and initial_chunk > 0 else chunk

    if len(frames) < target_frames and not finished:
        pending = torch.tensor(False, dtype=torch.bool)
        return OmniPayloadStruct(
            meta=MetaStruct(
                finished=pending,
                is_segment_finished=pending,
            )
        )

    emit_frames = len(frames) if finished else target_frames
    if emit_frames <= 0:
        if finished:
            request_payload.pop(request_id, None)
            return _empty_finished_payload()
        return None

    stacked = torch.stack(frames[:emit_frames], dim=0)  # [F, 12]
    flat = stacked.contiguous().reshape(-1)
    if finished:
        request_payload.pop(request_id, None)
    else:
        state["magenta_rt2_frames"] = frames[emit_frames:]
        state["magenta_rt2_emitted"] = True

    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(finished=torch.tensor(bool(finished), dtype=torch.bool)),
    )


__all__ = [
    "NUM_CODEBOOKS",
    "talker2code2wav_async_chunk",
    "talker2code2wav_full_payload",
    "talker2code2wav_token_only",
]
