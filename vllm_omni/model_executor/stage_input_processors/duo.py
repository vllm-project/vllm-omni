# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Official stage adapters for computed H50 -> Realization -> speculative codec."""

from collections.abc import Mapping, Sequence
from typing import Any

from vllm_omni.diffusion.models.duo.codec import DuoCodecBlock, unpack_duo_latents
from vllm_omni.diffusion.models.duo.wire import codec_to_wire


def multimodal_payload(output):
    mm = getattr(output, "multimodal_output", None)
    if isinstance(mm, Mapping):
        return mm
    outputs: Sequence[Any] = getattr(output, "outputs", ()) or ()
    if len(outputs) == 1 and isinstance(getattr(outputs[0], "multimodal_output", None), Mapping):
        return outputs[0].multimodal_output
    raise ValueError("Realization stage has no inline multimodal handoff")


def intent_to_realization(source_outputs, prompt=None, requires_multimodal_data=False):
    if len(source_outputs) != 1:
        raise ValueError("Realization requires one computed Intent block")
    mm = multimodal_payload(source_outputs[0])
    condition = mm.get("realization_condition")
    if not isinstance(condition, Mapping):
        raise ValueError("Realization condition was not selected by the session H50 ledger")
    return {"prompt": "frame-v5 Realization block", "additional_information": {"duo_condition": condition}}


def realization_to_codec(source_outputs, prompt=None, requires_multimodal_data=False):
    if len(source_outputs) != 1:
        raise ValueError("Codec requires one final Euler latent snapshot")
    mm = multimodal_payload(source_outputs[0])
    meta = mm["metadata"]
    if meta.get("phase") != "latent_ready" or meta.get("clean_committed") is not False:
        raise ValueError("Codec dispatch must precede the separate clean acknowledgement")
    route = mm.get("realization_codec_route")
    if not isinstance(route, Mapping):
        raise ValueError("Missing immutable codec routing geometry")
    video, audio = unpack_duo_latents(
        mm.get("payload", mm),
        height=route["height"],
        width=route["width"],
        codec_mode=route.get("codec_mode", "stream_4frame"),
    )
    block = DuoCodecBlock(
        meta["session_id"],
        meta["fence"],
        meta["block"],
        video,
        audio,
        reference=route["reference"] if meta["block"] == 0 else None,
        final=route.get("final", False),
        tenant=route["tenant"],
        dispatch_ns=route.get("dispatch_ns"),
        codec_mode=route.get("codec_mode", "stream_4frame"),
        reset_media_cache=meta.get("media_cache_reset", False),
    )
    return {"prompt": "Realization codec block", "additional_information": {"duo_codec": codec_to_wire(block)}}
