# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hands the CSM talker's frames to the Mimi stage.

The Mimi stage keeps its decoder state per request, so each frame is sent
exactly once and never again as left context.
"""

from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

DEFAULT_CHUNK_FRAMES = 5


def _chunk_sizes(transfer_manager: Any) -> tuple[int, int]:
    """Frames per chunk and frames in a request's first chunk, from the connector config."""
    config = getattr(getattr(transfer_manager, "connector", None), "config", None) or {}
    extra = config.get("extra", config)
    chunk = int(extra.get("codec_chunk_frames", DEFAULT_CHUNK_FRAMES))
    initial = int(extra.get("initial_codec_chunk_frames") or chunk)
    left_context = int(extra.get("codec_left_context_frames") or 0)
    if chunk <= 0 or initial <= 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk}, initial_codec_chunk_frames={initial}"
        )
    if left_context != 0:
        raise ValueError(
            f"codec_left_context_frames={left_context} is not supported: the CSM codec stage keeps its decoder state"
        )
    return chunk, min(initial, chunk)


def _published_frame(multimodal_output: Any) -> torch.Tensor | None:
    """The frame a talker step produced (``[num_codebooks]``, on the CPU), or None if there is none to play."""
    if not isinstance(multimodal_output, Mapping):
        return None
    codes = multimodal_output.get("codes")
    codes = codes.get("audio") if isinstance(codes, Mapping) else None
    meta = multimodal_output.get("meta")
    valid = meta.get("codec_frame_valid") if isinstance(meta, Mapping) else None
    if not isinstance(codes, torch.Tensor) or not isinstance(valid, torch.Tensor):
        return None
    if codes.numel() == 0 or valid.numel() == 0:
        return None
    # A prefill step carries one row per prompt token; its frame is in the last row.
    frame = codes.reshape(-1, codes.shape[-1])[-1]
    # The flag and the frame reach the host in one copy.
    row = torch.cat((valid.reshape(-1)[-1:].to(torch.long), frame.to(torch.long))).cpu()
    frame = row[1:]
    # Not played: the step after a stop, and the all-zero frame where the audio ends.
    if not bool(row[0]) or not bool(frame.any()):
        return None
    return frame


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Collect the step's frame; return the next chunk once enough frames wait or the request ends.

    The chunk holds only frames not sent before, flattened codebook-major
    (``[num_codebooks * frames]``). The last chunk of a request is marked
    finished and may be empty.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    pending = transfer_manager.code_prompt_token_ids[request_id]
    frame = _published_frame(multimodal_output)
    if frame is not None:
        pending.append(frame)

    chunk, initial = _chunk_sizes(transfer_manager)
    progress = transfer_manager.request_payload.setdefault(request_id, {"chunks": 0})
    if not finished and len(pending) < (initial if progress["chunks"] == 0 else chunk):
        return None
    codes = torch.stack(pending).T.reshape(-1) if pending else torch.empty(0, dtype=torch.long)
    pending.clear()
    progress["chunks"] += 1
    return OmniPayloadStruct(
        codes=CodesStruct(audio=codes),
        meta=MetaStruct(request_id=request_id, finished=torch.tensor(finished)),
    )
