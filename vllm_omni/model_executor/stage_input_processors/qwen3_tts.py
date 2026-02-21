"""Stage input processor for Qwen3-TTS: Talker -> Code2Wav."""

from typing import Any

import torch


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
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    if not isinstance(pooling_output, dict):
        return None

    request_id = request.external_req_id

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size = int(cfg.get("codec_left_context_frames", 25))
    if chunk_size <= 0 or left_context_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size}"
        )

    finished = bool(request.is_finished())

    frame = _extract_last_frame(pooling_output)
    if frame is not None:
        codec_codes = frame.cpu().tolist()
        transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)

    length = len(transfer_manager.code_prompt_token_ids[request_id])
    chunk_length = length % chunk_size

    if chunk_length != 0 and not finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size

    if length <= 0:
        return {
            "code_predictor_codes": [],
            "finished": torch.tensor(bool(finished), dtype=torch.bool),
        }

    end_index = min(length, left_context_size + context_length)
    ctx_frames = max(0, int(end_index - context_length))
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Pack context + chunk into codebook-major flat codes for adapter.
    code_predictor_codes = torch.tensor(window_frames).transpose(0, 1).reshape(-1).tolist()

    # Build final prompt_token_ids with ctx_frames header for Qwen3-TTS Code2Wav.
    # The model expects input_ids layout: [ctx_frames, *flat_codes].
    return {
        "code_predictor_codes": [int(ctx_frames)] + code_predictor_codes,
        "finished": torch.tensor(bool(finished), dtype=torch.bool),
    }
