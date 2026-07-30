# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from base64 import b64decode
from binascii import Error as BinasciiError
from typing import Any

from vllm_omni.experimental.fullduplex.base.runtime_extension import (
    BaseDuplexRuntimeExtension,
)
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputDecision,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.qwen3omni.policy import Qwen3OmniDuplexPolicy

_CHUNK_SAMPLES = Qwen3OmniDuplexPolicy.CHUNK_SAMPLES
_SAMPLES_PER_AUDIO_TOKEN = Qwen3OmniDuplexPolicy.SAMPLES_PER_AUDIO_TOKEN


def _pcm_sample_count(payload: object) -> int | None:
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


def _whisper_output_tokens(sample_count: int, hop_length: int = 160) -> int:
    mel_frames = -(-sample_count // hop_length)
    leave = mel_frames % 100
    feat = (leave - 1) // 2 + 1
    return max(0, ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel_frames // 100) * 13)


def qwen3omni_scheduler_token_budget(payload: object, *, default: int = 64) -> int:
    sample_count = _pcm_sample_count(payload)
    if sample_count is None:
        return max(1, int(default))
    return max(1, _whisper_output_tokens(sample_count))


def _runtime_config_int(runtime_config: object, key: str, fallback: int) -> int:
    if not isinstance(runtime_config, dict):
        return fallback
    val = runtime_config.get(key)
    if isinstance(val, int) and val >= 0:
        return val
    return fallback


def qwen3omni_first_append_context_reserve(runtime_config: object) -> int:
    return _runtime_config_int(runtime_config, "duplex_context_prefix_tokens", 48)


def qwen3omni_suffix_reserve(runtime_config: object) -> int:
    return _runtime_config_int(runtime_config, "duplex_context_suffix_tokens", 12)


def tokenize_qwen3omni_duplex_prompt(
    tokenizer: Any,
    session_config: dict[str, Any],
    payload: object,
    *,
    final: bool,
    seq: int,
) -> list[int]:
    """Build real ChatML token IDs for a duplex prompt (CPU-only, no GPU).

    Mirrors the token ID construction in stage0.py _build_context_embeds /
    _append_suffix / stage_prefill_embeddings, but without embedding.
    """

    def _encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    token_ids: list[int] = []

    if seq <= 1:
        prefix, _ = Qwen3OmniDuplexPolicy.session_context_texts(
            session_config.get("instructions"),
        )
        token_ids.extend(_encode(prefix))

        history = session_config.get("conversation_history")
        last_role = "user"
        if isinstance(history, list):
            for msg in history:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                content = msg.get("content", "")
                if not role or not content:
                    continue
                if role == "user":
                    token_ids.extend(_encode(f"{content}<|im_end|>\n"))
                else:
                    token_ids.extend(
                        _encode(f"<|im_start|>assistant\n{content}<|im_end|>\n<|im_start|>user\n"),
                    )
                last_role = role
            if last_role == "user" and history:
                token_ids.extend(_encode("<|im_start|>user\n"))

        token_ids.append(Qwen3OmniDuplexPolicy.AUDIO_START_TOKEN_ID)

    audio_tokens = qwen3omni_scheduler_token_budget(payload)
    token_ids.extend([Qwen3OmniDuplexPolicy.AUDIO_PAD_TOKEN_ID] * audio_tokens)

    if final:
        token_ids.append(Qwen3OmniDuplexPolicy.AUDIO_END_TOKEN_ID)
        _, suffix = Qwen3OmniDuplexPolicy.session_context_texts(
            session_config.get("instructions"),
        )
        token_ids.extend(_encode(suffix))

    return token_ids


def build_qwen3omni_data_plane_prompt(
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
    tokenizer: Any = None,
) -> dict[str, Any]:
    if tokenizer is not None:
        prompt_token_ids = tokenize_qwen3omni_duplex_prompt(
            tokenizer,
            session_config,
            payload,
            final=final,
            seq=seq,
        )
        token_budget = len(prompt_token_ids)
    else:
        token_budget = qwen3omni_scheduler_token_budget(payload)
        if seq <= 1:
            token_budget += qwen3omni_first_append_context_reserve(runtime_config)
            history = session_config.get("conversation_history")
            if isinstance(history, list) and history:
                token_budget += sum(len(str(m.get("content", ""))) // 3 + 10 for m in history if isinstance(m, dict))
        if final:
            token_budget += qwen3omni_suffix_reserve(runtime_config)

        raw_token_id = runtime_config.get("duplex_scheduler_token_id")
        try:
            token_id = max(0, int(raw_token_id))
        except (TypeError, ValueError):
            token_id = 0
        prompt_token_ids = [token_id] * token_budget

    raw_token_id = runtime_config.get("duplex_scheduler_token_id")
    try:
        scheduler_token_id = max(0, int(raw_token_id))
    except (TypeError, ValueError):
        scheduler_token_id = 0

    return {
        "prompt_token_ids": prompt_token_ids,
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
                "scheduler_token_id": scheduler_token_id,
            },
        },
    }


class Qwen3OmniDuplexRuntimeExtension(BaseDuplexRuntimeExtension):
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
        tokenizer: Any = None,
    ) -> DuplexAppendPlan:
        del sampling_params
        return DuplexAppendPlan(
            prompt=build_qwen3omni_data_plane_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                mode=mode,
                payload=payload,
                final=final,
                tokenizer=tokenizer,
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
        del segment_token_ids, segment_output_metadata
        del final_stage_id, segment_finished, output
        return None
