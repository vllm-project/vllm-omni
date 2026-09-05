# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transfer raw Breeze RVQ frames to the stateful Qwen3 codec."""

from typing import TYPE_CHECKING

import torch
from vllm.v1.request import Request

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayload, OmniPayloadStruct

if TYPE_CHECKING:
    from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter


def talker2code2wav_async_chunk(
    transfer_manager: "OmniChunkTransferAdapter",
    multimodal_output: OmniPayload | None,
    request: Request,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    request_id = request.external_req_id
    finished = is_finished or request.is_finished()
    frames = transfer_manager.code_prompt_token_ids[request_id]
    if multimodal_output is not None:
        codes = multimodal_output.get("codes", {}).get("audio")
        if codes is not None and codes.numel():
            if codes.ndim != 2 or codes.shape[1] != 16:
                raise ValueError("Breeze codec payload must have shape (frames, 16)")
            rows = codes.to(device="cpu", dtype=torch.long)
            if not bool(((rows >= 0) & (rows < 2048)).all()):
                raise ValueError("Breeze codec payload contains a reserved or invalid code")
            frames.extend(rows.unbind(0))
    connector = transfer_manager.connector
    if connector is None:
        raise RuntimeError("Breeze streaming requires a stage connector")
    config = connector.config
    chunk_size = int(config.get("extra", config).get("codec_chunk_frames", 5))
    if chunk_size <= 0:
        raise ValueError("codec_chunk_frames must be positive")
    emitted = transfer_manager.request_payload.get(request_id, 0)
    pending = len(frames) - emitted
    if not finished and pending < chunk_size:
        return None
    if pending < 0:
        raise RuntimeError("Breeze codec transfer emitted more frames than generated")
    # The shared codec retains its causal state; each frame is delivered once.
    flat = torch.stack(frames[emitted:]).T.contiguous().reshape(-1) if pending else torch.empty(0, dtype=torch.long)
    transfer_manager.request_payload[request_id] = len(frames)
    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(request_id=request_id, left_context_size=0, finished=torch.tensor(finished)),
    )
