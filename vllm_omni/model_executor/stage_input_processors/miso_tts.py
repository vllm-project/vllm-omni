# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for Miso TTS: Talker → Mimi decoder."""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct, to_dict
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
    max_ic_for_chunk_size,
)

logger = init_logger(__name__)

_MISO_NUM_CODEBOOKS = 32


def _extract_last_frame(pooling_output: dict) -> torch.Tensor | None:
    audio_codes = pooling_output.get("codes", {}).get("audio")
    if isinstance(audio_codes, list):
        tensors = [x for x in audio_codes if isinstance(x, torch.Tensor) and x.numel() > 0]
        if not tensors:
            return None
        frame = tensors[-1]
    elif isinstance(audio_codes, torch.Tensor):
        if audio_codes.numel() == 0:
            return None
        if audio_codes.ndim == 2:
            frame = audio_codes[-1]
        elif audio_codes.ndim == 1:
            frame = audio_codes
        else:
            return None
    else:
        return None
    if frame.numel() == 0 or not bool(frame.any().item()):
        return None
    return frame.to(torch.long).reshape(-1)


def talker_preprocess_input(
    request: Any,
    model_intermediate_buffer: dict[str, dict] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Extract additional_information (text, speaker) into runtime info for talker."""
    if model_intermediate_buffer is None:
        return {}
    
    # Try different possible req_id attribute names
    req_id = getattr(request, "req_id", None)
    if req_id is None:
        req_id = getattr(request, "request_id", None)
    if req_id is None:
        req_id = getattr(request, "external_req_id", None)
    
    if req_id is None:
        return {}
    
    info = model_intermediate_buffer.get(req_id, {})
    
    # Return the info dict which will be merged into runtime_additional_information
    return info


def _audio_codes_as_frames(audio_codes: object) -> torch.Tensor | None:
    """Normalize talker ``codes.audio`` to ``[num_frames, Q]``."""
    if isinstance(audio_codes, list):
        rows = [x.reshape(-1).to(torch.long) for x in audio_codes if isinstance(x, torch.Tensor) and x.numel() > 0]
        if not rows:
            return None
        if all(r.numel() == _MISO_NUM_CODEBOOKS for r in rows):
            return torch.stack(rows, dim=0)
        if len(rows) == 1 and rows[0].numel() % _MISO_NUM_CODEBOOKS == 0:
            return rows[0].reshape(-1, _MISO_NUM_CODEBOOKS)
        return None
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    audio_codes = audio_codes.to(torch.long)
    if audio_codes.ndim == 1 and audio_codes.numel() % _MISO_NUM_CODEBOOKS == 0:
        return audio_codes.reshape(-1, _MISO_NUM_CODEBOOKS)
    if audio_codes.ndim == 2:
        return audio_codes
    return None


def talker2mimi(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async: pass full codec sequence to Mimi after talker finishes."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    code2wav_inputs: list[OmniTokensPrompt] = []
    for talker_output in source_outputs:
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output
        mm_codes = mm.get("codes", {})
        audio_codes = mm_codes.get("audio")
        audio_codes = _audio_codes_as_frames(audio_codes)
        if audio_codes is None:
            continue
        valid_mask = (audio_codes >= 0).all(dim=1) & audio_codes.any(dim=1)
        audio_codes = audio_codes[valid_mask]
        flat = audio_codes.transpose(0, 1).reshape(-1).tolist()
        if not flat:
            continue
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=flat,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )
    return code2wav_inputs


def talker2mimi_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict | None,
    request: Any,
    is_finished: bool = False,
    **_: Any,
) -> OmniPayloadStruct | None:
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    # Check if talker signaled done via multimodal_output
    if isinstance(multimodal_output, dict):
        frame = _extract_last_frame(multimodal_output)
        if frame is not None:
            transfer_manager.code_prompt_token_ids[request_id].append(frame.cpu().tolist())
        # Check done flag from talker
        done_flags = multimodal_output.get("done")
        if isinstance(done_flags, (list, tuple)) and len(done_flags) > 0:
            finished = finished or bool(done_flags[0])
        elif hasattr(done_flags, 'item'):  # torch.Tensor
            finished = finished or bool(done_flags.item())

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    initial_chunk = int(cfg.get("initial_codec_chunk_frames") or 0)

    length = len(transfer_manager.code_prompt_token_ids[request_id])
    if length <= 0:
        if finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    if initial_chunk > 0 and length < initial_chunk and not finished:
        return None

    if length < chunk_size and not finished:
        return None

    # Sliding window (chunk + left-context) just like Qwen3-TTS / Moss.
    context_length = min(length, chunk_size)
    end_index = min(length, left_context_size_config + context_length)
    left_context_size = max(0, end_index - context_length)
    window = transfer_manager.code_prompt_token_ids[request_id][-end_index:]
    if not window:
        return None

    num_frames = len(window)
    
    # Fix: Use frame-major order [T*Q] instead of codebook-major [Q*T]
    # This matches what the decoder expects in _frames_from_runtime_info
    # Optimized: use torch operations instead of list comprehension
    code_tensor = torch.tensor(window, dtype=torch.long).reshape(-1)
    meta = MetaStruct(
        left_context_size=left_context_size,
        codec_chunk_frames=chunk_size,
        codec_left_context_frames=left_context_size_config,
        code_flat_numel=int(code_tensor.numel()),
        finished=torch.tensor(finished, dtype=torch.bool),
    )
    if finished:
        transfer_manager.code_prompt_token_ids[request_id].clear()

    return OmniPayloadStruct(codes=CodesStruct(audio=code_tensor), meta=meta)


def talker2mimi_full_payload(
    transfer_manager: Any,
    pooling_output: dict | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    if not is_finished and not request.is_finished():
        frame = _extract_last_frame(pooling_output) if isinstance(pooling_output, dict) else None
        if frame is not None:
            rid = request.external_req_id
            transfer_manager.code_prompt_token_ids[rid].append(frame.cpu().tolist())
        return None

    rid = request.external_req_id
    frames = transfer_manager.code_prompt_token_ids.get(rid, [])
    if not frames:
        return OmniPayloadStruct(
            codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
            meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
        )
    flat: list[int] = []
    for frame in frames:
        flat.extend(frame)
    transfer_manager.code_prompt_token_ids[rid].clear()
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor(flat, dtype=torch.long)),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )


def talker2mimi_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Sync mode placeholder tokens; real codec payload arrives via connector."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    return [
        OmniTokensPrompt(
            prompt_token_ids=[0],
            multi_modal_data=None,
            mm_processor_kwargs=None,
            additional_information=to_dict(OmniPayloadStruct()),
        )
        for out in source_outputs
        if getattr(out, "finished", False)
    ]


# Re-export dynamic IC helper for tests / parity with Qwen3-TTS processors.
__all__ = [
    "talker2mimi",
    "talker2mimi_async_chunk",
    "talker2mimi_full_payload",
    "talker2mimi_token_only",
    "compute_dynamic_initial_chunk_size",
    "max_ic_for_chunk_size",
]
