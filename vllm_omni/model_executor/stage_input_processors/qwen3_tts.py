"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _extract_last_frame(pooling_output: dict[str, Any]) -> torch.Tensor | None:
    audio_codes = pooling_output.get("audio_codes")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None
        return frame.to(torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        return audio_codes.to(torch.long).reshape(-1)
    raise ValueError(f"Invalid audio_codes shape for Qwen3-TTS async_chunk: {tuple(audio_codes.shape)}")


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            codec_codes = frame.cpu().tolist()
            transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)
    elif not finished:
        # Some steps may not produce pooling_output. Only flush on finish.
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))
    initial_chunk_size = int(cfg.get("initial_codec_chunk_frames", 0))
    # Per-request override (takes priority over stage config)
    additional_information = getattr(request, "additional_information", None)
    if (
        additional_information is not None
        and hasattr(additional_information, "entries")
        and "initial_codec_chunk_frames" in additional_information.entries
    ):
        entry = additional_information.entries["initial_codec_chunk_frames"]
        if entry.list_data is not None and len(entry.list_data) == 1:
            initial_chunk_size = int(entry.list_data[0])
    if chunk_size <= 0 or left_context_size_config < 0 or initial_chunk_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={initial_chunk_size}"
        )
    if initial_chunk_size > chunk_size:
        logger.warning(
            "initial_codec_chunk_frames=%d > codec_chunk_frames=%d, clamping to codec_chunk_frames.",
            initial_chunk_size,
            chunk_size,
        )
        initial_chunk_size = chunk_size
    length = len(transfer_manager.code_prompt_token_ids[request_id])

    # Avoid emitting empty chunks during normal streaming. If the request is
    # finished and nothing was produced, emit an EOF marker.
    if length <= 0:
        if finished:
            return {
                "code_predictor_codes": [],
                "finished": torch.tensor(True, dtype=torch.bool),
            }
        return None

    in_initial_phase = initial_chunk_size > 0 and length <= chunk_size

    if in_initial_phase:
        # Initial-chunk phase: emit every initial_chunk_size frames with full accumulated context.
        already_sent = transfer_manager.put_req_chunk[request_id] * initial_chunk_size
        pending = length - already_sent
        if pending <= 0:
            return None
        if pending < initial_chunk_size and not finished:
            return None
        context_length = min(pending, initial_chunk_size)
        end_index = length
        left_context_size = max(0, length - context_length)
        window_frames = transfer_manager.code_prompt_token_ids[request_id][:length]
    else:
        # Normal phase: standard chunk_size cadence with left_context sliding window.
        # Offset by initial_coverage so normal starts from where the initial-chunk phase left off.
        initial_coverage = (chunk_size // initial_chunk_size) * initial_chunk_size if initial_chunk_size > 0 else 0
        adjusted = length - initial_coverage
        chunk_length = adjusted % chunk_size
        if chunk_length != 0 and not finished:
            return None
        context_length = chunk_length if chunk_length != 0 else chunk_size
        end_index = min(length, left_context_size_config + context_length)
        left_context_size = max(0, int(end_index - context_length))
        window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Pack context + chunk into codebook-major flat codes for adapter.
    code_predictor_codes = torch.tensor(window_frames).transpose(0, 1).reshape(-1).tolist()

    return {
        "code_predictor_codes": code_predictor_codes,
        "left_context_size": left_context_size,
        "finished": torch.tensor(finished, dtype=torch.bool),
    }
