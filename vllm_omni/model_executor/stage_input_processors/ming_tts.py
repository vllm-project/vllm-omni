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
    INITIAL_LATENT_CHUNK_SIZE,
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


def _select_snapshot_value(value: Any, selected_index: int | torch.Tensor) -> Any:
    if isinstance(value, torch.Tensor):
        flat = value.reshape(-1)
        if flat.numel() == 0:
            return None
        if isinstance(selected_index, torch.Tensor):
            value_index = selected_index.to(device=flat.device).clamp_max(flat.numel() - 1)
        else:
            value_index = min(selected_index, flat.numel() - 1)
        return flat[value_index]
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value_index = int(selected_index.item()) if isinstance(selected_index, torch.Tensor) else selected_index
        return value[-1] if value_index < 0 else value[min(value_index, len(value) - 1)]
    return value


def _extract_snapshot_metadata(
    pooling_output: Mapping[str, Any],
    selected_index: int | torch.Tensor,
) -> tuple[Any, Any]:
    decode_step_value = pooling_output.get("ming_decode_step")
    stop_reason_value = pooling_output.get(MING_STOP_REASON_KEY)
    if isinstance(selected_index, torch.Tensor) and (
        isinstance(decode_step_value, (list, tuple)) or isinstance(stop_reason_value, (list, tuple))
    ):
        selected_index = int(selected_index.item())

    final_decode_step = _select_snapshot_value(decode_step_value, selected_index)
    stop_reason_code = _select_snapshot_value(stop_reason_value, selected_index)

    if (
        isinstance(final_decode_step, torch.Tensor)
        and isinstance(stop_reason_code, torch.Tensor)
        and final_decode_step.device == stop_reason_code.device
    ):
        final_decode_step, stop_reason_code = (
            torch.stack(
                (
                    final_decode_step.to(torch.int64),
                    stop_reason_code.to(torch.int64),
                )
            )
            .cpu()
            .tolist()
        )
    else:
        if isinstance(final_decode_step, torch.Tensor):
            final_decode_step = final_decode_step.item()
        if isinstance(stop_reason_code, torch.Tensor):
            stop_reason_code = stop_reason_code.item()
    return final_decode_step, stop_reason_code


def _extract_ming_output_snapshot(
    pooling_output: Mapping[str, Any] | None,
    *,
    all_patches: bool = False,
) -> tuple[torch.Tensor | None, Any, str | None]:
    if not isinstance(pooling_output, Mapping):
        return None, None, None

    has_patch = pooling_output.get("ming_has_patch")
    active: torch.Tensor | None = None
    selected_index: int | torch.Tensor = -1
    if isinstance(has_patch, torch.Tensor) and has_patch.numel() > 0:
        patch_mask = has_patch.reshape(-1) > 0
        if all_patches:
            active = patch_mask.nonzero(as_tuple=True)[0]
            if active.numel() == 0:
                return None, None, None
            selected_index = active[-1]
        else:
            reversed_mask = patch_mask.flip(0)
            reverse_offset, has_active = (
                torch.stack(
                    (
                        reversed_mask.to(torch.int64).argmax(),
                        reversed_mask.any().to(torch.int64),
                    )
                )
                .cpu()
                .tolist()
            )
            if not has_active:
                return None, None, None
            selected_index = patch_mask.numel() - 1 - reverse_offset

    patch = pooling_output.get("ming_latent_patch")
    selected_patch: torch.Tensor | None = None
    if isinstance(patch, torch.Tensor) and patch.numel() > 0:
        if all_patches:
            if patch.ndim == 2:
                patch = patch.unsqueeze(0)
            if patch.ndim != 3:
                raise ValueError(f"Invalid Ming latent patch tensor shape: {tuple(patch.shape)}")
            selected_patch = patch if active is None else patch.index_select(0, active.to(device=patch.device))
        else:
            if patch.ndim == 3:
                if isinstance(selected_index, torch.Tensor):
                    selected_index = selected_index.to(device=patch.device)
                patch = patch[selected_index]
            if patch.ndim != 2:
                raise ValueError(f"Invalid Ming latent patch shape: {tuple(patch.shape)}")
            selected_patch = patch

        selected_patch = selected_patch.to(torch.float32).cpu() if selected_patch.numel() > 0 else None

    final_decode_step, stop_reason_code = _extract_snapshot_metadata(pooling_output, selected_index)
    return selected_patch, final_decode_step, _decode_stop_reason(stop_reason_code)


def _decode_stop_reason(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return MING_STOP_REASON_BY_CODE.get(int(value))


def _get_async_chunk_config(transfer_manager: Any) -> tuple[int, int, int]:
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
    initial_chunk_size = int(cfg.get("initial_latent_chunk_size", INITIAL_LATENT_CHUNK_SIZE))
    left_context = int(cfg.get("latent_left_context", LATENT_LEFT_CONTEXT))
    if chunk_size <= 0:
        raise ValueError(f"Invalid Ming latent_chunk_size={chunk_size}")
    if initial_chunk_size < 0:
        raise ValueError(f"Invalid Ming initial_latent_chunk_size={initial_chunk_size}")
    if initial_chunk_size > chunk_size:
        logger.warning(
            "Ming initial_latent_chunk_size=%d > latent_chunk_size=%d, clamping to latent_chunk_size.",
            initial_chunk_size,
            chunk_size,
        )
        initial_chunk_size = chunk_size
    # Stage-2 VAE caches past_key_values and stream_state by request_id.
    # Replaying left-context latents would double-feed cached decoder state.
    if left_context != 0:
        raise ValueError(
            "Ming async chunk transport does not support latent_left_context replay. "
            "Ming boundary continuity is handled by per-request decoder state cache, not "
            f"latent replay. Got latent_left_context={left_context}."
        )
    return chunk_size, initial_chunk_size, left_context


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
    patch, final_decode_step, stop_reason = _extract_ming_output_snapshot(pooling_output)
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

    if patch is not None:
        transfer_manager.code_prompt_token_ids[request_id].append(patch)

    chunk_size, initial_chunk_size, _ = _get_async_chunk_config(transfer_manager)

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

    use_first_chunk = 0 < initial_chunk_size < chunk_size and seen_patch_len == 0
    if finished:
        emit_count = length
    elif use_first_chunk:
        if length < initial_chunk_size:
            return None
        emit_count = initial_chunk_size
    else:
        if length < chunk_size:
            return None
        emit_count = chunk_size
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
        patches, final_decode_step, stop_reason = _extract_ming_output_snapshot(
            output.multimodal_output,
            all_patches=True,
        )
        additional_information = {
            "ming_latent_patches": patches
            if patches is not None
            else torch.zeros((0, PATCH_SIZE, LATENT_DIM), dtype=torch.float32),
            KEY_REQUEST_ID: getattr(stage_output, "request_id", None),
            "finished": torch.tensor(finished, dtype=torch.bool),
        }
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
