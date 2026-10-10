from collections.abc import Mapping
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
)

logger = init_logger(__name__)


def _codec_audio_tensors(multimodal_output: OmniPayload | dict[str, Any]) -> list[torch.Tensor] | None:
    """Inter-stage codec frames from ``codes.audio``."""
    if not isinstance(multimodal_output, Mapping):
        return None
    codes = multimodal_output.get("codes")
    if not isinstance(codes, Mapping) or "audio" not in codes:
        return None
    audio = codes["audio"]
    if isinstance(audio, torch.Tensor):
        return [audio]
    if isinstance(audio, list) and audio:
        return audio
    return None


def _extract_last_frame(multimodal_output: OmniPayload | dict[str, Any]) -> torch.Tensor | None:
    audio_tensors = _codec_audio_tensors(multimodal_output)
    if not audio_tensors:
        return None
    frame = audio_tensors[-1]
    if not isinstance(frame, torch.Tensor) or frame.numel() == 0:
        return None
    return frame.flatten()


def generator2tokenizer_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | dict[str, Any],
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if isinstance(multimodal_output, Mapping):
        frame = _extract_last_frame(multimodal_output)
        if frame is not None:
            # Keep the frame on its producing device: appending must not force a
            # per-frame D2H sync. The whole window is copied to host once per
            # flushed chunk below instead.
            transfer_manager.code_prompt_token_ids[request_id].append(frame.detach())
    elif not finished:
        # Some steps may not produce multimodal_output. Only flush on finish.
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    chunk_size_at_begin = int(cfg.get("codec_chunk_frames_at_begin", 5))
    left_context_size = int(cfg.get("codec_left_context_frames", 25))
    if chunk_size <= 0 or left_context_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size}"
        )
    length = len(transfer_manager.code_prompt_token_ids[request_id])

    # Avoid emitting empty chunks during normal streaming. If the request is
    # finished and nothing was produced, emit an EOF marker.
    if length <= 0:
        if finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    # Use a small chunk size at begin
    if length <= chunk_size:
        chunk_size = chunk_size_at_begin

    chunk_length = length % chunk_size

    if chunk_length != 0 and not finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size
    end_index = min(length, left_context_size + context_length)
    ctx_frames = max(0, int(end_index - context_length))
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Pack context + chunk into codebook-major flat codes for adapter.
    # Frames may still be on the GPU, so flatten on-device and issue a single
    # D2H copy for the whole window (one per chunk instead of one per frame).
    # torch.as_tensor passes tensors through zero-copy and still accepts the
    # plain-int frames older seeds/tests may have stored; stragglers on another
    # device (e.g. pre-seeded CPU lists against CUDA frames) are moved to the
    # producing device of the newest frame first. .to() is a no-op when the
    # window is homogeneous.
    frames = [torch.as_tensor(f).reshape(-1) for f in window_frames]
    code_predictor_codes = torch.cat([f.to(frames[-1].device) for f in frames]).cpu().tolist()

    return OmniPayloadStruct(
        codes=CodesStruct(
            audio=torch.tensor(
                [int(ctx_frames), int(context_length)] + code_predictor_codes,
                dtype=torch.long,
            ),
        ),
        meta=MetaStruct(finished=torch.tensor(finished, dtype=torch.bool)),
    )
