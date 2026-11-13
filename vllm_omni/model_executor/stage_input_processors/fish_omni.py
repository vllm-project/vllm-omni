# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processor for Fish Audio Drama 3: frame stream -> DAC decoder.

Stage 0 emits one frame ``[main, c0..c9]`` per request per step. Frames whose
main token is not semantic (the end-of-speech frame) carry no audio. Windows
follow a ramp (``codec_chunk_ramp``, default ``6, 8, 12, 16, 24, 32``, then
``codec_chunk_frames``) so the first audio leaves early, and every window
carries up to ``codec_left_context_frames`` earlier frames, which the DAC
decoder trims (or skips, when it keeps the stream's codec state). Windows
travel as ``[codebooks, frames]`` tensors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

SEMANTIC_BEGIN_ID = 248099
CODEBOOK_SIZE = 4096
NUM_CODEBOOKS = 10
DEFAULT_RAMP = (6, 8, 12, 16, 24, 32)
DEFAULT_CHUNK = 64
DEFAULT_LEFT_CONTEXT = 32
_STATE_KEY = "_fish_omni_stream"


def _config(transfer_manager: Any) -> tuple[tuple[int, ...], int, int]:
    cached = getattr(transfer_manager, "_fish_omni_chunk_config", None)
    if cached is not None:
        return cached
    connector = getattr(transfer_manager, "connector", None)
    raw = getattr(connector, "config", {}) or {}
    cfg = raw.get("extra", raw) if isinstance(raw, dict) else {}
    cfg = cfg if isinstance(cfg, dict) else {}
    chunk = int(cfg.get("codec_chunk_frames", DEFAULT_CHUNK))
    left = int(cfg.get("codec_left_context_frames", DEFAULT_LEFT_CONTEXT))
    ramp_cfg = cfg.get("codec_chunk_ramp", DEFAULT_RAMP)
    ramp = tuple(int(v) for v in (ramp_cfg.split(",") if isinstance(ramp_cfg, str) else ramp_cfg))
    if chunk <= 0 or left < 0 or any(v <= 0 for v in ramp):
        raise ValueError(f"invalid fish_omni codec chunk config chunk={chunk} left={left} ramp={ramp}")
    transfer_manager._fish_omni_chunk_config = (ramp, chunk, left)
    return transfer_manager._fish_omni_chunk_config


def _state(transfer_manager: Any, request_id: str) -> dict[str, Any]:
    # request_payload entries are reclaimed when a request completes or is cancelled.
    return transfer_manager.request_payload.setdefault(request_id, {}).setdefault(
        # ``codes`` holds frames ``base..``: frames no later window reads are dropped.
        _STATE_KEY,
        {"codes": [], "base": 0, "emitted": 0, "windows": 0, "done": False},
    )


def _frames(output: Any) -> torch.Tensor | None:
    if not isinstance(output, Mapping):
        return None
    codes = output.get("codes", {})
    frames = codes.get("audio") if isinstance(codes, Mapping) else None
    if frames is None:
        frames = output.get("codes.audio")
    if not isinstance(frames, torch.Tensor) or frames.numel() == 0:
        return None
    return frames


def _semantic(main: np.ndarray) -> np.ndarray:
    return (main >= SEMANTIC_BEGIN_ID) & (main < SEMANTIC_BEGIN_ID + CODEBOOK_SIZE)


def _request_id(request: Any) -> str:
    return str(getattr(request, "external_req_id", None) or getattr(request, "request_id", id(request)))


def fish_omni_to_dac_decoder_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    finished = bool(is_finished or (hasattr(request, "is_finished") and request.is_finished()))
    codes = None
    frames = _frames(multimodal_output)
    if frames is not None:
        rows = frames.detach().cpu().numpy().reshape(-1, frames.shape[-1])
        keep = _semantic(rows[:, 0])
        codes = rows[keep, 1:] if keep.any() else None
    return _advance(transfer_manager, _request_id(request), codes, finished)


def fish_omni_to_dac_decoder_async_chunk_batch(
    transfer_manager: Any,
    pooling_outputs: list[Any],
    requests: list[Any],
    is_finished: list[bool],
) -> list[OmniPayloadStruct | None]:
    """One model step of :func:`fish_omni_to_dac_decoder_async_chunk`, in request order.

    The step's frames are validated in one NumPy pass instead of one small
    conversion per request; most requests only append a frame.
    """
    present: list[int] = []
    arrays: list[np.ndarray] = []
    for index, output in enumerate(pooling_outputs):
        frames = _frames(output)
        if frames is not None:
            present.append(index)
            arrays.append(frames.detach().cpu().numpy().reshape(-1, frames.shape[-1]))
    rows: list[np.ndarray | None] = [None] * len(pooling_outputs)
    if arrays:
        stacked = np.concatenate(arrays)
        keep = _semantic(stacked[:, 0]).tolist()
        offset = 0
        for index, array in zip(present, arrays, strict=True):
            count = array.shape[0]
            if count == 1:
                rows[index] = stacked[offset : offset + 1, 1:] if keep[offset] else None
            else:
                mask = keep[offset : offset + count]
                rows[index] = stacked[offset : offset + count][mask, 1:] if any(mask) else None
            offset += count
    return [
        _advance(transfer_manager, _request_id(request), rows[index], bool(is_finished[index]))
        for index, request in enumerate(requests)
    ]


def _advance(
    transfer_manager: Any, request_id: str, codes: np.ndarray | None, finished: bool
) -> OmniPayloadStruct | None:
    ramp, chunk, left_context = _config(transfer_manager)
    state = _state(transfer_manager, request_id)
    if state["done"]:
        return None
    if codes is not None:
        state["codes"].extend(codes)

    base = state["base"]
    total = base + len(state["codes"])
    emitted = state["emitted"]
    window = ramp[state["windows"]] if state["windows"] < len(ramp) else chunk
    pending = total - emitted
    if not finished and pending < window:
        return None
    if pending <= 0:
        state["done"] = True
        # Still [codebooks, frames]: the receiver replaces the previous
        # window's codes only with a 2-D tensor, and a kept window would
        # decode again.
        return OmniPayloadStruct(
            codes=CodesStruct(audio=torch.empty((NUM_CODEBOOKS, 0), dtype=torch.long)),
            meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
        )
    end = total if finished else emitted + window
    context = min(emitted, left_context)
    # The window's index in its stream: the codec starts a stream's history at
    # chunk 0 (left context alone cannot tell, e.g. with no left context).
    seq = state["windows"]
    frames = np.stack(state["codes"][emitted - context - base : end - base])  # [frames, 10]
    state["emitted"] = end
    state["windows"] += 1
    done = finished and end == total
    if done:
        state["done"] = True
        state["codes"] = []
    else:
        keep_from = max(end - left_context, base)
        del state["codes"][: keep_from - base]
        state["base"] = keep_from
    # A [codebooks, frames] tensor: Stage 1 schedules one placeholder token per
    # window and the decoder reads the codes from runtime info, instead of a
    # Python int per code through the scheduler and back.
    # A fresh C-order copy: for a one-frame window ``ascontiguousarray`` keeps the
    # transposed strides, which the shared-memory connector cannot view as bytes.
    codes = torch.from_numpy(np.array(frames.T, dtype=np.int64, order="C", copy=True))
    return OmniPayloadStruct(
        codes=CodesStruct(audio=codes),
        meta=MetaStruct(
            left_context_size=context,
            chunk_seq=seq,
            finished=torch.tensor(done, dtype=torch.bool),
        ),
    )


# Per-step saves (OmniChunkTransferAdapter.save_async_batch) skip the request
# snapshot: this batch form reads only request ids and the finish flags given.
fish_omni_to_dac_decoder_async_chunk_batch.per_step_batch = True  # type: ignore[attr-defined]

__all__ = ["fish_omni_to_dac_decoder_async_chunk", "fish_omni_to_dac_decoder_async_chunk_batch"]
