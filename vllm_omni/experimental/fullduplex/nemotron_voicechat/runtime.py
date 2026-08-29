# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Engine policy for Nemotron VoiceChat's frame-locked duplex timeline."""

from __future__ import annotations

from typing import Any

from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputAction,
    DuplexOutputDecision,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.nemotron_voicechat.input import (
    decode_pcm_f32le,
)


def _plain_token_ids(value: object, *, name: str) -> list[int]:
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"Nemotron VoiceChat runtime requires non-empty {name}")
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Nemotron VoiceChat {name} must contain token ids") from exc


def _positive_int(value: object, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Nemotron VoiceChat requires a positive {name}") from exc
    if parsed <= 0:
        raise ValueError(f"Nemotron VoiceChat requires a positive {name}")
    return parsed


class NemotronVoiceChatDuplexRuntimeExtension:
    """Append one 80 ms acoustic frame to one resumable thinker request."""

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, Any],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        if not defaults:
            return defaults
        # Every scheduler segment is one transport wake of a long-lived
        # request.  CUMULATIVE output makes the multimodal output processor
        # concatenate all earlier Stage-2 waveforms and resend them on every
        # wake (80, 160, 240, ... ms), causing O(n^2) audio replay.  Duplex
        # consumers require per-wake deltas at every stage.
        configured = [params.clone() if isinstance(params, SamplingParams) else params for params in defaults]
        for params in configured:
            if isinstance(params, SamplingParams):
                params.output_kind = RequestOutputKind.DELTA
        stage0 = defaults[0]
        if isinstance(stage0, SamplingParams):
            stage0 = stage0.clone()
            stage0.temperature = 0.0
            stage0.top_p = 1.0
            stage0.top_k = 0
            stage0.max_tokens = 1
            stage0.ignore_eos = True
            stage0.output_kind = RequestOutputKind.DELTA
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
        del sampling_params, turn_seq
        if mode is not DuplexInputMode.APPEND_AUDIO_CHUNK:
            raise ValueError(f"Nemotron VoiceChat does not support duplex input mode {mode.value!r}")
        decode_pcm_f32le(payload, exact_frame=True)
        normalized_payload = dict(payload)
        prompt_ids = _plain_token_ids(
            runtime_config.get("nvc_prompt_token_ids"),
            name="nvc_prompt_token_ids",
        )
        max_model_len = _positive_int(
            runtime_config.get("nvc_max_model_len"),
            name="nvc_max_model_len",
        )
        required_model_len = len(prompt_ids) + seq + 1
        if required_model_len > max_model_len:
            raise ValueError(
                "Nemotron VoiceChat native duplex session exceeds the Stage-0 "
                f"max_model_len: prompt_tokens={len(prompt_ids)} + input_frames={seq} + "
                f"sampled_token=1 gives {required_model_len} > {max_model_len}; "
                "start a new session or raise Stage 0 max_model_len"
            )
        try:
            pad_id = int(runtime_config["nvc_text_pad_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Nemotron VoiceChat runtime requires nvc_text_pad_id") from exc
        scheduler_prompt = prompt_ids + [pad_id] if seq <= 1 else [pad_id]
        return DuplexAppendPlan(
            prompt={
                "prompt_token_ids": scheduler_prompt,
                "model_intermediate_buffer": {
                    "request_id": request_id,
                    "global_request_id": [fence.session_id],
                    "duplex": {
                        "data_plane": True,
                        "fence": fence,
                        "session_id": fence.session_id,
                        "incarnation": fence.incarnation,
                        "epoch": fence.epoch,
                        "source_input_seq": seq,
                        "seq": seq,
                        "mode": mode.value,
                        "payload": normalized_payload,
                        "final": final,
                        "session_config": dict(session_config),
                        "runtime_config": dict(runtime_config),
                        "scheduler_token_budget": len(scheduler_prompt),
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
        del final_stage_id, segment_finished, output
        if stage_id != 0:
            return None
        metadata = dict(segment_output_metadata)
        metadata["nvc_text_token_ids"] = list(segment_token_ids)
        # Stage 0 is a client-visible side channel even though Stage 2 is the
        # configured audio response stage. Mark it explicitly so the shared
        # collector does not discard it while waiting for the final stage.
        metadata["duplex_direct_response"] = True
        return DuplexOutputDecision(
            action=DuplexOutputAction.DIRECT_RESPONSE,
            metadata=metadata,
            final_output_type="text",
        )


__all__ = ["NemotronVoiceChatDuplexRuntimeExtension"]
