# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/modeling_bailingmm.py
from __future__ import annotations

from typing import Any

import torch
from vllm.forward_context import get_forward_context, is_forward_context_available

from .config_ming_tts import KEY_MAX_DECODE_STEPS, KEY_MIN_DECODE_STEPS, KEY_REQUEST_ID, KEY_TEXT_MODE, MingTTSConfig

MING_STOP_REASON_CONTINUE = "continue"
MING_STOP_REASON_STOP_HEAD = "stop_head"
MING_STOP_REASON_MAX_DECODE_STEPS = "max_decode_steps"
MING_STOP_REASON_KEY = "ming_stop_reason"
MING_STOP_REASON_CODES = {
    MING_STOP_REASON_CONTINUE: 0,
    MING_STOP_REASON_STOP_HEAD: 1,
    MING_STOP_REASON_MAX_DECODE_STEPS: 2,
}


def _normalize_request_infos(model_intermediate_buffer: object) -> list[dict[str, Any]]:
    if not isinstance(model_intermediate_buffer, list):
        return []
    infos: list[dict[str, Any]] = []
    for item in model_intermediate_buffer:
        infos.append(item if isinstance(item, dict) else {})
    return infos


def _get_request_token_counts(
    hidden_states: torch.Tensor,
    request_infos: list[dict[str, Any]],
    seq_token_counts: list[int] | None,
) -> list[int]:
    if seq_token_counts:
        return [int(x) for x in seq_token_counts]

    if is_forward_context_available():
        slices = getattr(get_forward_context(), "ubatch_slices", None)
        if slices is not None and len(slices) > 0:
            counts: list[int] = []
            for item in slices:
                if isinstance(item, int):
                    counts.append(int(item))
                elif hasattr(item, "stop") and hasattr(item, "start"):
                    counts.append(int(item.stop) - int(item.start))
            if counts:
                return counts

    if request_infos:
        if len(request_infos) == hidden_states.shape[0]:
            return [1] * hidden_states.shape[0]
        return [hidden_states.shape[0]]

    return []


def _coerce_latent_history(
    value: object,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cfg: MingTTSConfig,
) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    history = value.detach()
    if history.ndim == 2:
        history = history.unsqueeze(0)
    if history.ndim != 3:
        raise RuntimeError(f"Expected latent_history rank-3 [B,T,D], got {tuple(history.shape)}")
    if history.shape[1] != cfg.history_patch_size or history.shape[2] != cfg.latent_dim:
        raise RuntimeError(
            f"latent_history shape mismatch: got {tuple(history.shape)}, "
            f"expected [B,{cfg.history_patch_size},{cfg.latent_dim}]"
        )
    return history.to(device=device, dtype=dtype)


def _resolve_runtime_float(req_info: dict[str, Any], key: str, default_value: float) -> float:
    raw = req_info.get(key, default_value)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {key}: expected float-like value, got {raw!r}") from exc
    if not value >= 0.0:
        raise RuntimeError(f"Invalid {key}: expected non-negative value, got {value}")
    return value


def _resolve_runtime_int(req_info: dict[str, Any], key: str, default_value: int) -> int:
    raw = req_info.get(key, default_value)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {key}: expected int-like value, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"Invalid {key}: expected positive value, got {value}")
    return value


def _resolve_optional_runtime_int(req_info: dict[str, Any], key: str, default_value: int) -> int:
    raw = req_info.get(key, default_value)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {key}: expected int-like value, got {raw!r}") from exc
    if value < 0:
        raise RuntimeError(f"Invalid {key}: expected non-negative value, got {value}")
    return value


def _validate_ming_decode_window(
    request_infos: list[dict[str, Any]],
    *,
    min_stop_step: int,
    default_max_decode_steps: int,
) -> None:
    for i, info in enumerate(request_infos):
        if info.get(KEY_TEXT_MODE):
            continue
        max_steps = _resolve_runtime_int(info, KEY_MAX_DECODE_STEPS, default_max_decode_steps)
        min_steps = _resolve_optional_runtime_int(info, KEY_MIN_DECODE_STEPS, 0)
        min_required = max(min_stop_step + 1, min_steps)
        if max_steps < min_required:
            req_id = info.get(KEY_REQUEST_ID, f"idx={i}")
            raise ValueError(
                f"Ming request {req_id!r}: max_decode_steps={max_steps} < "
                f"min_required_decode_steps={min_required} "
                f"(min_stop_step={min_stop_step}, min_decode_steps={min_steps})"
            )


def _resolve_ming_stop_decision(
    *,
    step: int,
    stop_prob: float,
    stop_threshold: float,
    min_stop_step: int,
    min_decode_steps: int,
    max_decode_steps: int,
    audio_dummy_token_id: int,
    text_eos_token_id: int,
) -> tuple[str, bool, bool, int, int]:
    min_required_decode_steps = max(min_stop_step + 1, min_decode_steps)
    if max_decode_steps < min_required_decode_steps:
        raise RuntimeError(
            "Invalid Ming decode window: "
            f"max_decode_steps={max_decode_steps} is smaller than "
            f"min_required_decode_steps={min_required_decode_steps}"
        )
    should_force_stop = (step + 1) >= max_decode_steps
    should_stop_head = ((step + 1) >= min_required_decode_steps) and stop_prob > stop_threshold

    if should_force_stop:
        return (
            MING_STOP_REASON_MAX_DECODE_STEPS,
            True,
            True,
            min_required_decode_steps,
            text_eos_token_id,
        )
    if should_stop_head:
        return (
            MING_STOP_REASON_STOP_HEAD,
            True,
            False,
            min_required_decode_steps,
            text_eos_token_id,
        )
    return (
        MING_STOP_REASON_CONTINUE,
        False,
        False,
        min_required_decode_steps,
        audio_dummy_token_id,
    )
