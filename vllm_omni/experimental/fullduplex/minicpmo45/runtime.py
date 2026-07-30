# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from base64 import b64decode
from binascii import Error as BinasciiError
from typing import Any

from vllm_omni.experimental.fullduplex.base.data_plane import (
    coerce_int,
    completion_token_ids,
    first_completion,
    multimodal_output,
    special_token_ids,
)
from vllm_omni.experimental.fullduplex.base.runtime_extension import (
    BaseDuplexRuntimeExtension,
)
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputAction,
    DuplexOutputDecision,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

_DUPLEX_CHUNK_SAMPLES = 16000
_DUPLEX_SAMPLES_PER_AUDIO_TOKEN = 1600
_DUPLEX_VISION_TOKENS_PER_FRAME = 66


def _duplex_frame_count(payload: object) -> int:
    if not isinstance(payload, dict):
        return 0
    frames = payload.get("video_frames")
    if not isinstance(frames, list):
        return 0
    return sum(1 for frame in frames if isinstance(frame, str) and frame)


def _duplex_pcm_sample_count(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    audio = payload.get("audio") or payload.get("data")
    if payload.get("format") != "pcm_f32le" or not isinstance(audio, str):
        return None
    try:
        raw = b64decode(audio, validate=True)
    except (BinasciiError, ValueError):
        return None
    return len(raw) // 4


def duplex_payload_is_exact_chunks(payload: object) -> bool:
    sample_count = _duplex_pcm_sample_count(payload)
    return bool(sample_count) and sample_count % _DUPLEX_CHUNK_SAMPLES == 0


def duplex_first_append_unit_count(payload: object) -> int | None:
    sample_count = _duplex_pcm_sample_count(payload)
    if not sample_count or sample_count % _DUPLEX_CHUNK_SAMPLES != 0:
        return None
    return max(1, sample_count // _DUPLEX_CHUNK_SAMPLES - 1)


def duplex_scheduler_token_budget(payload: object, *, default: int = 64) -> int:
    vision_tokens = _duplex_frame_count(payload) * _DUPLEX_VISION_TOKENS_PER_FRAME
    sample_count = _duplex_pcm_sample_count(payload)
    if sample_count is None:
        return max(1, int(default)) + vision_tokens
    sample_count = max(1, sample_count)
    if sample_count % _DUPLEX_CHUNK_SAMPLES == 0:
        units = sample_count // _DUPLEX_CHUNK_SAMPLES
        return units * (2 + _DUPLEX_CHUNK_SAMPLES // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN) + vision_tokens
    return max(16, min(768, sample_count // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8)) + vision_tokens


def duplex_first_append_context_reserve(runtime_config: object) -> int:
    if not isinstance(runtime_config, dict):
        return 48
    exact = runtime_config.get("duplex_first_append_context_tokens")
    if isinstance(exact, int) and exact >= 0:
        return exact
    reserve = 48
    ref = runtime_config.get("ref_audio_data")
    if isinstance(ref, str) and ref:
        try:
            raw = b64decode(ref, validate=True)
        except (BinasciiError, ValueError):
            raw = b""
        if raw:
            reserve += max(0, (len(raw) // 4) // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8)
    return reserve


def _duplex_force_listen_count(extra_body: object) -> int:
    raw = extra_body.get("force_listen_count") if isinstance(extra_body, dict) else None
    try:
        return 0 if raw is None else max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def build_duplex_data_plane_prompt(
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
) -> dict[str, Any]:
    token_budget = duplex_scheduler_token_budget(payload)
    if seq <= 1:
        context_reserve = duplex_first_append_context_reserve(runtime_config)
        token_budget += context_reserve
        first_units = duplex_first_append_unit_count(payload)
        if first_units is not None:
            token_budget = (
                context_reserve + first_units * 12 - 1 + _duplex_frame_count(payload) * _DUPLEX_VISION_TOKENS_PER_FRAME
            )
    if seq > 1 and duplex_payload_is_exact_chunks(payload):
        token_budget += 1
    if final and duplex_payload_is_exact_chunks(payload):
        token_budget += 12
    extra_body = session_config.get("extra_body")
    raw_token_id = runtime_config.get("duplex_scheduler_token_id")
    try:
        token_id = max(0, int(raw_token_id))
    except (TypeError, ValueError):
        token_id = 0
    force_listen_count = _duplex_force_listen_count(extra_body)
    if (
        force_listen_count > 0
        and turn_seq <= force_listen_count
        and isinstance(payload, dict)
        and payload.get("force_listen") is not True
    ):
        payload = {**payload, "force_listen": True}
    return {
        "prompt_token_ids": [token_id] * token_budget,
        "model_intermediate_buffer": {
            "request_id": request_id,
            "global_request_id": [fence.session_id],
            "duplex": {
                "fence": fence,
                "session_id": fence.session_id,
                "incarnation": fence.incarnation,
                "epoch": fence.epoch,
                "seq": seq,
                "turn_id": fence.turn_id,
                "response_seq": fence.response_seq,
                "turn_seq": turn_seq,
                "mode": mode.value,
                "payload": payload,
                "final": final,
                "data_plane": True,
                "session_config": dict(session_config),
                "runtime_config": dict(runtime_config),
                "scheduler_token_budget": token_budget,
                "scheduler_token_id": token_id,
            },
        },
    }


class MiniCPMO45DuplexRuntimeExtension(BaseDuplexRuntimeExtension):
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
        tokenizer: object = None,
    ) -> DuplexAppendPlan:
        del sampling_params, tokenizer
        return DuplexAppendPlan(
            prompt=build_duplex_data_plane_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                mode=mode,
                payload=payload,
                final=final,
            )
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
        if stage_id >= final_stage_id or not segment_finished:
            return None

        completion = first_completion(output)
        output_metadata = multimodal_output(output, completion)
        stids = special_token_ids(segment_output_metadata)
        stids.update(special_token_ids(output_metadata))
        listen_id = stids.get("listen_token_id")
        if listen_id is None:
            return None

        stop_reason = getattr(completion, "stop_reason", None) if completion is not None else None
        token_ids = completion_token_ids(completion) or list(segment_token_ids)
        if coerce_int(stop_reason) != listen_id and (not token_ids or token_ids[-1] != listen_id):
            return None

        metadata = dict(output_metadata)
        for key, value in stids.items():
            metadata.setdefault(f"meta.{key}", value)
        metadata.update(
            {
                "duplex_direct_response": True,
                "duplex_native_decision": "listen",
                "model_listen": True,
                "listen_source": "model_listen",
            }
        )
        return DuplexOutputDecision(
            action=DuplexOutputAction.DIRECT_RESPONSE,
            metadata=metadata,
        )


__all__ = [
    "MiniCPMO45DuplexRuntimeExtension",
    "build_duplex_data_plane_prompt",
    "duplex_first_append_context_reserve",
    "duplex_first_append_unit_count",
    "duplex_payload_is_exact_chunks",
    "duplex_scheduler_token_budget",
]
