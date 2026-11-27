# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage transport for Ming3 continuous latents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.ming3_tts.config_ming3_tts import (
    resolve_latent_chunk_sizes,
)
from vllm_omni.model_executor.models.ming3_tts.constants import LATENT_DIM
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_CHUNK_ID as STATE_CHUNK_ID,
)
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_REQUEST_ID as STATE_REQUEST_ID,
)


def _patches(output: Mapping[str, Any] | None) -> torch.Tensor | None:
    if not isinstance(output, Mapping):
        return None
    patch = output.get("ming3_latent_patch")
    active = output.get("ming3_has_patch")
    if not isinstance(patch, torch.Tensor) or patch.numel() == 0:
        return None
    if patch.ndim == 2:
        patch = patch.unsqueeze(0)
    if patch.ndim != 3 or patch.shape[-2:] != (1, LATENT_DIM):
        raise ValueError(f"Invalid Ming3 latent tensor shape: {tuple(patch.shape)}")
    if isinstance(active, torch.Tensor) and active.numel():
        # Payloads arrive on the host with one row per scheduled token, so
        # plain Python indexing avoids several tiny tensor ops per request.
        flags = active.reshape(-1).tolist()
        keep = [index for index, flag in enumerate(flags) if flag > 0]
        if not keep:
            return None
        if len(keep) != len(flags) or len(flags) != patch.shape[0]:
            patch = patch[keep]
    return patch.detach().to("cpu", dtype=torch.float32)


def _chunk_sizes(transfer_manager: Any) -> tuple[int, int, bool]:
    connector = getattr(transfer_manager, "connector", None)
    raw = getattr(connector, "config", {}) or {}
    config = raw.get("extra", raw) if isinstance(raw, Mapping) else {}
    chunk, initial = resolve_latent_chunk_sizes(config)
    left = int(config.get("latent_left_context", 0))
    ramp = bool(config.get("latent_chunk_ramp", False))
    if left != 0:
        raise ValueError("Ming3 streaming AudioVAE uses cached state; latent_left_context must be 0")
    return chunk, initial, ramp


def _chunk_target(emitted: int, chunk_size: int, initial_size: int, ramp: bool) -> int:
    """Return how many latents the next chunk waits for.

    With ``ramp``, chunks double from the initial size up to ``chunk_size``,
    so a stream does not wait a full chunk of AR steps while its player holds
    only the small first chunk.
    """
    if emitted == 0:
        return initial_size
    if not ramp:
        return chunk_size
    target = initial_size
    while emitted >= target and target < chunk_size:
        target *= 2
    return min(target, chunk_size)


def llm2audio_vae_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    # TODO: Share the Ming/Ling chunk-buffer state machine after streaming parity.
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    state_root = transfer_manager.request_payload.get(request_id)
    if not isinstance(state_root, dict) or "_ming3_async_state" not in state_root:
        state_root = {"_ming3_async_state": {"seen": 0, "terminal": False}}
        transfer_manager.request_payload[request_id] = state_root
    state = state_root["_ming3_async_state"]
    if state["terminal"]:
        return None

    current = _patches(multimodal_output)
    if current is not None:
        transfer_manager.code_prompt_token_ids[request_id].extend(list(current))
    buffered = transfer_manager.code_prompt_token_ids[request_id]
    seen = int(state["seen"])
    available = len(buffered) - seen
    chunk_size, initial_size, ramp = _chunk_sizes(transfer_manager)
    if available <= 0:
        if not finished:
            return None
        state["terminal"] = True
        return OmniPayloadStruct(
            codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
            meta=MetaStruct(
                finished=torch.tensor(True),
                stream_finished=torch.tensor(True),
            ),
            kv_metadata={
                STATE_REQUEST_ID: request_id,
                STATE_CHUNK_ID: int(transfer_manager.put_req_chunk[request_id]),
            },
        )

    target = _chunk_target(seen, chunk_size, initial_size, ramp)
    if not finished and available < target:
        return None
    count = available if finished else target
    latent = torch.stack(buffered[seen : seen + count], dim=0)
    state["seen"] = seen + count
    state["terminal"] = finished
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor([0], dtype=torch.long)),
        meta=MetaStruct(
            finished=torch.tensor(finished),
            stream_finished=torch.tensor(finished),
        ),
        latent=latent,
        kv_metadata={
            STATE_REQUEST_ID: request_id,
            STATE_CHUNK_ID: int(transfer_manager.put_req_chunk[request_id]),
            "ming3_emit_patch_count": count,
        },
    )


def llm2audio_vae(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any = None,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data, streaming_context
    outputs: list[OmniTokensPrompt] = []
    for stage_output in source_outputs:
        if not bool(getattr(stage_output, "finished", True)):
            continue
        output = stage_output.outputs[0]
        latent = _patches(output.multimodal_output)
        outputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information={
                    "latent": latent if latent is not None else torch.zeros((0, 1, LATENT_DIM), dtype=torch.float32),
                    STATE_REQUEST_ID: getattr(stage_output, "request_id", None),
                    "finished": torch.tensor(True),
                },
            )
        )
    return outputs


__all__ = ["llm2audio_vae", "llm2audio_vae_async_chunk"]
