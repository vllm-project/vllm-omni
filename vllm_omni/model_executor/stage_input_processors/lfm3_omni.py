# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3 frame handoff: [frames, codebooks] to LiquidCodec [codebooks, frames]."""

import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

CODEC_CHUNK_FRAMES = 4


def _frames(output):
    if not output:
        return torch.empty((0, 12), dtype=torch.long)
    value = output.get("codes.audio")
    if value is None:
        value = output.get("codes", {}).get("audio")
    if value is None:
        return torch.empty((0, 12), dtype=torch.long)
    frames = torch.as_tensor(value, dtype=torch.long).reshape(-1, 12)
    if torch.any((frames < 0) | (frames >= 2016)):
        raise ValueError("LiquidCodec payload contains an invalid or terminal audio code")
    return frames


def temporal_to_codec_full_payload(transfer_manager, pooling_output, request):
    frames = _frames(pooling_output)
    codes = frames.transpose(0, 1).contiguous() if frames.numel() else torch.empty((12, 0), dtype=torch.long)
    return {"codes": {"audio": codes}, "meta": {"finished": True}}


def temporal_to_codec_async_chunk(transfer_manager, multimodal_output, request, is_finished=False):
    request_id = request.external_req_id
    state = transfer_manager.request_payload.setdefault(request_id, {}).setdefault(
        "lfm3_codec", {"pending": [], "finished": False}
    )
    if state["finished"]:
        return None
    frames = _frames(multimodal_output)
    if frames.numel():
        state["pending"].append(frames)
    finished = bool(is_finished or request.is_finished())
    # Four frames = 320 ms. The decoder retains its exact recurrent and
    # overlap-add state, so there is no approximate left-context truncation.
    count = sum(len(part) for part in state["pending"])
    if not finished and count < CODEC_CHUNK_FRAMES:
        return None
    codes = torch.cat(state["pending"]) if count else torch.empty((0, 12), dtype=torch.long)
    state["pending"] = []
    state["finished"] = finished
    return OmniPayloadStruct(
        codes=CodesStruct(
            audio=codes.transpose(0, 1).contiguous() if codes.numel() else torch.empty((12, 0), dtype=torch.long)
        ),
        meta=MetaStruct(
            finished=torch.tensor(finished),
            stream_finished=torch.tensor(finished),
            req_id=[request_id],
        ),
    )


def temporal_to_codec_token_only(source_outputs, prompt=None, requires_multimodal_data=False):
    return [{"prompt_token_ids": [0]} for _ in source_outputs]
