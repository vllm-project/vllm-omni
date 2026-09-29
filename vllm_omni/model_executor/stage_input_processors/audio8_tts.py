# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Preview: Slow AR -> codec decoder stage input processor."""

from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    MetaStruct,
    OmniPayloadStruct,
)

#: Chunking defaults; ~21.5 codec frames per second, so 25 frames ~= 1.16 s.
DEFAULT_CHUNK_FRAMES = 25
DEFAULT_LEFT_CONTEXT_FRAMES = 25


def _connector_extra(transfer_manager: Any) -> dict[str, Any]:
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    return raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}


def _cfg_int(cfg: dict[str, Any], key: str, default: int) -> int:
    value = cfg.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Audio8 TTS integer config {key}={value!r}") from exc


def _request_initial_chunk_frames(request: Any, default: int) -> int:
    """Per-request override of ``initial_codec_chunk_frames``, if present."""
    additional_information = getattr(request, "additional_information", None)
    entries = getattr(additional_information, "entries", None)
    if not entries or "initial_codec_chunk_frames" not in entries:
        return default
    entry = entries["initial_codec_chunk_frames"]
    if entry.list_data is not None and len(entry.list_data) == 1:
        return int(entry.list_data[0])
    return default


def extract_last_frame(multimodal_output: Mapping[str, Any]) -> torch.Tensor | None:
    """Return the newest frame of codec codes, or ``None`` if this step had none.

    Prefill steps emit an all-zero placeholder frame, which must not be
    forwarded to the decoder.
    """
    audio_codes = multimodal_output.get("audio_codes")
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 1:
        return audio_codes.to(device="cpu", dtype=torch.long).reshape(-1)
    if audio_codes.ndim != 2:
        raise ValueError(f"Invalid audio_codes shape for Audio8 TTS async_chunk: {tuple(audio_codes.shape)}")

    frame = audio_codes[-1]
    if frame.numel() == 0:
        return None
    valid = multimodal_output.get("audio_code_valid")
    if isinstance(valid, torch.Tensor) and valid.numel() > 0:
        is_valid = bool(valid.reshape(-1)[-1].item())
    elif valid is not None:
        is_valid = bool(valid)
    else:
        is_valid = bool(frame.any().item())
    if not is_valid:
        return None
    return frame.to(device="cpu", dtype=torch.long).reshape(-1)


def slow_ar_to_codec_decoder_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Emit ``[left_context | new_frames]`` chunks as the Slow AR produces codes.

    Codes accumulate across AR steps and are shipped as a
    ``[num_codebooks, frames]`` tensor. ``meta.left_context_size`` tells the
    decoder how many leading frames to drop, keeping chunk boundaries
    artefact-free. Returns ``None`` when the current chunk is not full yet.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    cfg = _connector_extra(transfer_manager)

    appended_this_call = False
    if isinstance(multimodal_output, Mapping):
        frame = extract_last_frame(multimodal_output)
        if frame is not None:
            transfer_manager.code_prompt_token_ids[request_id].append(frame.detach())
            appended_this_call = True
    elif not finished:
        return None

    chunk_frames = _cfg_int(cfg, "codec_chunk_frames", DEFAULT_CHUNK_FRAMES)
    left_context_frames = _cfg_int(cfg, "codec_left_context_frames", DEFAULT_LEFT_CONTEXT_FRAMES)
    initial_chunk_frames = _request_initial_chunk_frames(request, _cfg_int(cfg, "initial_codec_chunk_frames", 0))
    if chunk_frames <= 0 or left_context_frames < 0 or initial_chunk_frames < 0:
        raise ValueError(
            "Invalid Audio8 TTS codec chunk config: "
            f"codec_chunk_frames={chunk_frames}, codec_left_context_frames={left_context_frames}, "
            f"initial_codec_chunk_frames={initial_chunk_frames}"
        )
    initial_chunk_frames = min(initial_chunk_frames, chunk_frames)

    frames = transfer_manager.code_prompt_token_ids[request_id]
    length = len(frames)
    if length <= 0:
        if finished:
            # Still emit a terminal payload so the decoder can close the stream.
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    # A smaller first chunk trades a little quality for time-to-first-audio.
    if initial_chunk_frames > 0 and length <= chunk_frames:
        already_sent = transfer_manager.put_req_chunk[request_id] * initial_chunk_frames
        pending = length - already_sent
        if pending <= 0:
            return None
        if pending < initial_chunk_frames and not finished:
            return None
        new_frames = min(pending, initial_chunk_frames)
        left_context_size = max(0, length - new_frames)
        window = frames[:length]
    else:
        initial_coverage = (
            (chunk_frames // initial_chunk_frames) * initial_chunk_frames if initial_chunk_frames > 0 else 0
        )
        pending = (length - initial_coverage) % chunk_frames
        if pending != 0 and not finished:
            return None
        if pending == 0 and finished and not appended_this_call:
            # This boundary's chunk was already shipped when the stream first
            # reached it; a terminal call carrying no new frame must close the
            # stream, not re-send that window (otherwise the last chunk_frames
            # frames ship twice).
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        new_frames = pending if pending != 0 else chunk_frames
        end_index = min(length, left_context_frames + new_frames)
        left_context_size = max(0, end_index - new_frames)
        window = frames[-end_index:]

    # Codebook-major [num_codebooks, frames]; the tensor payload avoids
    # expanding codec indices into Python ints across the connector boundary.
    codes_qf = torch.stack(window, dim=0).transpose(0, 1).contiguous()
    return OmniPayloadStruct(
        codes=CodesStruct(audio=codes_qf),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(finished, dtype=torch.bool),
        ),
    )


__all__ = [
    "DEFAULT_CHUNK_FRAMES",
    "DEFAULT_LEFT_CONTEXT_FRAMES",
    "extract_last_frame",
    "slow_ar_to_codec_decoder_async_chunk",
]
