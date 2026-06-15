# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_CHUNK_ID,
    KEY_REQUEST_ID,
    LATENT_CHUNK_SIZE,
    LATENT_DIM,
    LATENT_LEFT_CONTEXT,
    PATCH_SIZE,
)
from vllm_omni.model_executor.models.ming_tts.patch_emission import (
    MING_STOP_REASON_CODES,
    MING_STOP_REASON_KEY,
)

logger = init_logger(__name__)

MING_EMIT_PATCH_COUNT_KEY = "ming_emit_patch_count"
MING_LATENT_SHAPE_KEY = "ming_latent_shape"
MING_ESTIMATED_BYTES_KEY = "ming_estimated_bytes"
MING_FINAL_FLUSH_KEY = "ming_final_flush"
MING_FINAL_DECODE_STEP_KEY = "ming_final_decode_step"
MING_STOP_REASON_BY_CODE = {code: reason for reason, code in MING_STOP_REASON_CODES.items()}


def _extract_last_patch(pooling_output: Mapping[str, Any] | None) -> torch.Tensor | None:
    if not isinstance(pooling_output, Mapping):
        return None
    has_patch = pooling_output.get("ming_has_patch")
    patch = pooling_output.get("ming_latent_patch")
    if not isinstance(patch, torch.Tensor) or patch.numel() == 0:
        return None

    if isinstance(has_patch, torch.Tensor) and has_patch.numel() > 0:
        active = (has_patch.reshape(-1) > 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return None
        patch = patch[int(active[-1].item())]
    elif patch.ndim == 3:
        patch = patch[-1]

    if patch.ndim != 2:
        raise ValueError(f"Invalid Ming latent patch shape: {tuple(patch.shape)}")
    return patch.to(torch.float32).cpu()


def _extract_all_patches(pooling_output: Mapping[str, Any] | None) -> torch.Tensor | None:
    if not isinstance(pooling_output, Mapping):
        return None
    has_patch = pooling_output.get("ming_has_patch")
    patch = pooling_output.get("ming_latent_patch")
    if not isinstance(patch, torch.Tensor) or patch.numel() == 0:
        return None

    if patch.ndim == 2:
        patch = patch.unsqueeze(0)
    if patch.ndim != 3:
        raise ValueError(f"Invalid Ming latent patch tensor shape: {tuple(patch.shape)}")

    if isinstance(has_patch, torch.Tensor) and has_patch.numel() > 0:
        active = (has_patch.reshape(-1) > 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return None
        patch = patch.index_select(0, active.to(device=patch.device))

    if patch.numel() == 0:
        return None
    return patch.to(torch.float32).cpu()


def _extract_last_value(pooling_output: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(pooling_output, Mapping):
        return None
    value = pooling_output.get(key)
    if value is None:
        return None

    has_patch = pooling_output.get("ming_has_patch")
    selected_index = -1
    if isinstance(has_patch, torch.Tensor) and has_patch.numel() > 0:
        active = (has_patch.reshape(-1) > 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return None
        selected_index = int(active[-1].item())

    if isinstance(value, torch.Tensor):
        flat = value.reshape(-1)
        if flat.numel() == 0:
            return None
        return flat[min(selected_index, flat.numel() - 1)].item()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if selected_index < 0:
            return value[-1]
        return value[min(selected_index, len(value) - 1)]
    return value


def _decode_stop_reason(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return MING_STOP_REASON_BY_CODE.get(int(value))


def _get_async_chunk_config(transfer_manager: Any) -> tuple[int, int]:
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, Mapping) else {}
    if "latent_chunk_size" not in cfg:
        logger.warning(
            "Ming async chunk config missing latent_chunk_size, using fallback value %s",
            LATENT_CHUNK_SIZE,
        )
    if "latent_left_context" not in cfg:
        logger.warning(
            "Ming async chunk config missing latent_left_context, using fallback value %s",
            LATENT_LEFT_CONTEXT,
        )
    chunk_size = int(cfg.get("latent_chunk_size", LATENT_CHUNK_SIZE))
    left_context = int(cfg.get("latent_left_context", LATENT_LEFT_CONTEXT))
    if chunk_size <= 0:
        raise ValueError(f"Invalid Ming latent_chunk_size={chunk_size}")
    # Stage-2 VAE caches past_key_values and stream_state by request_id.
    # Replaying left-context latents would double-feed cached decoder state.
    if left_context != 0:
        raise ValueError(
            "Ming async chunk transport does not support latent_left_context replay. "
            "Ming boundary continuity is handled by per-request decoder state cache, not "
            f"latent replay. Got latent_left_context={left_context}."
        )
    return chunk_size, left_context


def _build_chunk_observability(
    latent_patches: torch.Tensor | None,
    *,
    final_flush: bool,
) -> dict[str, Any]:
    if latent_patches is None:
        emit_patch_count = 0
        latent_shape = None
        estimated_bytes = 0
    else:
        emit_patch_count = int(latent_patches.shape[0])
        latent_shape = tuple(latent_patches.shape)
        estimated_bytes = int(latent_patches.numel() * latent_patches.element_size())
    return {
        MING_EMIT_PATCH_COUNT_KEY: emit_patch_count,
        MING_LATENT_SHAPE_KEY: latent_shape,
        MING_ESTIMATED_BYTES_KEY: estimated_bytes,
        MING_FINAL_FLUSH_KEY: bool(final_flush),
    }


def llm2audio_vae_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    pooling_output = multimodal_output
    request_id = request.external_req_id
    chunk_id = int(transfer_manager.put_req_chunk[request_id])
    finished = bool(is_finished or request.is_finished())
    final_decode_step = _extract_last_value(pooling_output, "ming_decode_step")
    stop_reason = _decode_stop_reason(_extract_last_value(pooling_output, MING_STOP_REASON_KEY))
    request_payload = transfer_manager.request_payload
    request_state = request_payload.get(request_id)
    if not isinstance(request_state, dict) or "_ming_async_state" not in request_state:
        request_state = {
            "_ming_async_state": {
                "seen_patch_len": 0,
                "terminal_sent": False,
            }
        }
        request_payload[request_id] = request_state
    state = request_state["_ming_async_state"]
    if bool(state.get("terminal_sent", False)):
        return None

    patch = _extract_last_patch(pooling_output)
    if patch is not None:
        transfer_manager.code_prompt_token_ids[request_id].append(patch)

    chunk_size, _ = _get_async_chunk_config(transfer_manager)

    patches = transfer_manager.code_prompt_token_ids[request_id]
    seen_patch_len = int(state.get("seen_patch_len", 0))
    new_patches = patches[seen_patch_len:] if seen_patch_len < len(patches) else []
    length = len(new_patches)
    if length <= 0:
        if finished and not bool(state.get("terminal_sent", False)):
            observability = _build_chunk_observability(None, final_flush=True)
            kv_meta = {
                KEY_CHUNK_ID: chunk_id,
                KEY_REQUEST_ID: request_id,
                **observability,
            }
            if final_decode_step is not None:
                kv_meta[MING_FINAL_DECODE_STEP_KEY] = int(final_decode_step)
            if stop_reason is not None:
                kv_meta[MING_STOP_REASON_KEY] = stop_reason
            state["terminal_sent"] = True
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(
                    finished=torch.tensor(True, dtype=torch.bool),
                    stream_finished=torch.tensor(True, dtype=torch.bool),
                ),
                kv_metadata=kv_meta,
            )
        return None

    chunk_length = length % chunk_size
    if chunk_length != 0 and not finished:
        return None

    emit_count = chunk_length if chunk_length != 0 else chunk_size
    emit_patches = list(new_patches[:emit_count])
    state["seen_patch_len"] = seen_patch_len + len(emit_patches)
    latent_patches = torch.stack(emit_patches, dim=0)
    observability = _build_chunk_observability(latent_patches, final_flush=finished)

    kv_meta = {
        KEY_CHUNK_ID: chunk_id,
        KEY_REQUEST_ID: request_id,
        **observability,
    }
    if final_decode_step is not None:
        kv_meta[MING_FINAL_DECODE_STEP_KEY] = int(final_decode_step)
    if stop_reason is not None:
        kv_meta[MING_STOP_REASON_KEY] = stop_reason
    if finished:
        state["terminal_sent"] = True
    return OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor([0], dtype=torch.long)),
        meta=MetaStruct(
            finished=torch.tensor(finished, dtype=torch.bool),
            stream_finished=torch.tensor(finished, dtype=torch.bool),
        ),
        latent=latent_patches,
        kv_metadata=kv_meta,
    )


def llm2audio_vae(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any = None,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data, streaming_context
    if not source_outputs:
        raise ValueError("source_outputs cannot be empty")

    outputs = []
    for stage_output in source_outputs:
        finished = bool(getattr(stage_output, "finished", True))
        if not finished:
            continue
        output = stage_output.outputs[0]
        patches = _extract_all_patches(output.multimodal_output)
        additional_information = {
            "ming_latent_patches": patches
            if patches is not None
            else torch.zeros((0, PATCH_SIZE, LATENT_DIM), dtype=torch.float32),
            KEY_REQUEST_ID: getattr(stage_output, "request_id", None),
            "finished": torch.tensor(finished, dtype=torch.bool),
        }
        final_decode_step = _extract_last_value(output.multimodal_output, "ming_decode_step")
        stop_reason = _decode_stop_reason(_extract_last_value(output.multimodal_output, MING_STOP_REASON_KEY))
        if final_decode_step is not None:
            additional_information[MING_FINAL_DECODE_STEP_KEY] = int(final_decode_step)
        if stop_reason is not None:
            additional_information[MING_STOP_REASON_KEY] = stop_reason
        outputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information,
            )
        )
    return outputs
