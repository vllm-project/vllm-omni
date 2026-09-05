# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transfer raw Breeze RVQ frames to the stateful Qwen3 codec."""

from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.request import Request

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayload, OmniPayloadStruct
from vllm_omni.model_executor.models.breeze_tts.prompt import CFG_UNCOND_SUFFIX
from vllm_omni.model_executor.stage_input_processors.bagel import ExpandedPrompt

if TYPE_CHECKING:
    from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter


def expand_cfg_prompts(prompt: dict[str, Any] | str, sampling_params: Any) -> list[ExpandedPrompt]:
    if not isinstance(prompt, dict):
        raise ValueError("Breeze requires build_breeze_prompt to encode its conditioning")
    info = prompt["additional_information"]
    conditioning = info["breeze_prompt"]
    if conditioning["guidance_scale"] == 1.0:
        return []
    negative = conditioning["negative_ids"]
    companion_conditioning = {**conditioning, "role": "uncond", "target_ids": negative}
    companion_conditioning.pop("negative_ids")
    length = len(prompt["prompt_token_ids"]) - len(conditioning["target_ids"]) + len(negative)
    companion = {
        "prompt_token_ids": [0] * length,
        "additional_information": {
            **info,
            "breeze_prompt": companion_conditioning,
            "cfg_group": {"role": "uncond", "uncond_suffix": CFG_UNCOND_SUFFIX},
        },
    }
    return [ExpandedPrompt(prompt=companion, role="uncond", request_id_suffix=CFG_UNCOND_SUFFIX)]


def talker2code2wav_async_chunk(
    transfer_manager: "OmniChunkTransferAdapter",
    multimodal_output: OmniPayload | None,
    request: Request,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    request_id = request.external_req_id
    if request_id.endswith(CFG_UNCOND_SUFFIX):
        return None
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
    extra = config.get("extra", config)
    state = transfer_manager.request_payload.setdefault(request_id, {"frames": 0, "chunks": 0})
    emitted = state["frames"]
    ramp = extra.get("codec_chunk_ramp", [])
    if state["chunks"] < len(ramp):
        chunk_size = int(ramp[state["chunks"]])
    else:
        chunk_size = int(
            extra.get("initial_codec_chunk_frames", 1) if emitted == 0 else extra.get("codec_chunk_frames", 5)
        )
    if chunk_size <= 0:
        raise ValueError("codec_chunk_frames must be positive")
    pending = len(frames) - emitted
    if not finished and pending < chunk_size:
        return None
    if pending < 0:
        raise RuntimeError("Breeze codec transfer emitted more frames than generated")
    # The shared codec retains its causal state; each frame is delivered once.
    flat = torch.stack(frames[emitted:]).T.contiguous().reshape(-1) if pending else torch.empty(0, dtype=torch.long)
    state["frames"] = len(frames)
    state["chunks"] += 1
    return OmniPayloadStruct(
        codes=CodesStruct(audio=flat),
        meta=MetaStruct(request_id=request_id, left_context_size=0, finished=torch.tensor(finished)),
    )
