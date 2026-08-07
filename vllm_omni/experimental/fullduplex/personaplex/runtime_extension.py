# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import binascii
from typing import Any

from vllm.sampling_params import SamplingParams

from vllm_omni.experimental.fullduplex.engine.duplex_runtime import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputDecision,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

_FRAME_BYTES = 1920 * 4


def _validated_frame_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("PersonaPlex duplex append payload must be a mapping")
    if payload.get("format") != "pcm_f32le":
        raise ValueError("PersonaPlex duplex append format must be pcm_f32le")
    if payload.get("sample_rate_hz") != 24000:
        raise ValueError("PersonaPlex duplex append sample_rate_hz must be 24000")
    audio = payload.get("audio")
    if not isinstance(audio, str):
        raise ValueError("PersonaPlex duplex append audio must be base64 pcm_f32le")
    try:
        raw = base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("PersonaPlex duplex append audio is not valid base64") from exc
    if len(raw) != _FRAME_BYTES:
        raise ValueError("PersonaPlex duplex append must contain exactly 1920 samples")
    return dict(payload)


class PersonaPlexDuplexRuntimeExtension:
    """PersonaPlex policy for the engine-owned resumable duplex request."""

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, Any],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        if not defaults:
            return defaults
        configured = list(defaults)
        stage0 = defaults[0]
        if isinstance(stage0, SamplingParams):
            stage0 = stage0.clone()
            stage0.temperature = 0.0
            stage0.top_k = 1
            stage0.max_tokens = 1
            configured[0] = stage0
        return tuple(configured)

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, Any],
        runtime_config: dict[str, Any],
        seq: int,
        turn_seq: int,
        mode: DuplexInputMode,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del sampling_params
        if mode is not DuplexInputMode.APPEND_AUDIO_CHUNK:
            raise ValueError(f"PersonaPlex does not support duplex input mode {mode.value!r}")
        normalized_payload = _validated_frame_payload(payload)
        prefill_slots = runtime_config.get("personaplex_prefill_slots", 0)
        try:
            prefill_slots = max(0, int(prefill_slots))
        except (TypeError, ValueError) as exc:
            raise ValueError("personaplex_prefill_slots must be a non-negative integer") from exc
        prompt_slots = 1 + (prefill_slots if seq <= 1 else 0)
        return DuplexAppendPlan(
            prompt={
                "prompt_token_ids": [0] * prompt_slots,
                "model_intermediate_buffer": {
                    "request_id": request_id,
                    "global_request_id": [fence.session_id],
                    "duplex": {
                        "data_plane": True,
                        "fence": fence,
                        "session_id": fence.session_id,
                        "incarnation": fence.incarnation,
                        "epoch": fence.epoch,
                        "turn_id": fence.turn_id,
                        "response_seq": fence.response_seq,
                        "seq": seq,
                        "turn_seq": turn_seq,
                        "mode": mode.value,
                        "payload": normalized_payload,
                        "final": final,
                        "session_config": dict(session_config),
                        "runtime_config": dict(runtime_config),
                        "scheduler_token_budget": prompt_slots,
                    },
                },
            }
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, Any],
        output: object,
    ) -> DuplexOutputDecision | None:
        del stage_id, final_stage_id, segment_finished
        del segment_token_ids, segment_output_metadata, output
        return None


__all__ = ["PersonaPlexDuplexRuntimeExtension"]
