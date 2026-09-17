# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen3-Omni duplex plugin: one ephemeral Thinker→Talker→Code2Wav request per turn."""

from __future__ import annotations

import binascii
from collections.abc import Mapping

import numpy as np
import pybase64 as base64
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexOutputDecision,
)
from vllm_omni.engine.duplex.plugin import DuplexModelPlugin, EncodeAudio
from vllm_omni.model_executor.models.qwen3_omni.duplex.capabilities import (
    qwen3_omni_duplex_capabilities,
)
from vllm_omni.model_executor.models.qwen3_omni.duplex.data_plane import (
    Qwen3OmniDataPlaneContext,
    Qwen3OmniDataPlaneSession,
)
from vllm_omni.model_executor.models.qwen3_omni.duplex.session import Qwen3OmniDuplexSessionState

# Matches ``QWEN3_OMNI_PIPELINE`` talker ``sampling_constraints.stop_token_ids``.
QWEN3_OMNI_TALKER_EOS_TOKEN_ID = 2150

DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)

_THINKER_STAGE_ID = 0
_TALKER_STAGE_ID = 1

_PRIVATE_KEYS = frozenset(
    {
        "qwen3_system_prompt",
        "initial_user_text",
    }
)


def _decode_pcm_f32le(payload: Mapping[str, object]) -> tuple[np.ndarray, int]:
    audio = payload.get("audio")
    sample_rate_hz = payload.get("sample_rate_hz", 16000)
    if not isinstance(audio, str) or not isinstance(sample_rate_hz, int):
        raise ValueError("Qwen3-Omni duplex commit payload requires pcm_f32le audio and sample_rate_hz")
    try:
        raw = base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Qwen3-Omni duplex audio is not valid base64") from exc
    if len(raw) % 4:
        raise ValueError("Qwen3-Omni duplex pcm_f32le payload has a partial sample")
    values = np.frombuffer(raw, dtype="<f4").copy()
    return values, sample_rate_hz


def _thinker_prompt(*, system_prompt: str, user_text: str) -> str:
    extra = user_text if user_text else ""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        f"{extra}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class Qwen3OmniDuplexPlugin(DuplexModelPlugin):
    """Turn-commit Qwen3-Omni: one ephemeral three-stage request per utterance."""

    plugin_id = "qwen3_omni"
    private_runtime_config_keys = _PRIVATE_KEYS

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane = Qwen3OmniDataPlaneSession(encode_audio)

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        configured = list(defaults)
        if len(configured) > _TALKER_STAGE_ID and isinstance(configured[_TALKER_STAGE_ID], SamplingParams):
            talker = configured[_TALKER_STAGE_ID].clone()
            stop_ids = list(talker.stop_token_ids or [])
            if QWEN3_OMNI_TALKER_EOS_TOKEN_ID not in stop_ids:
                stop_ids.append(QWEN3_OMNI_TALKER_EOS_TOKEN_ID)
            talker.stop_token_ids = stop_ids
            configured[_TALKER_STAGE_ID] = talker
        return tuple(configured)

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del request_id, session_config, seq, turn_seq, sampling_params

        if not final:
            raise ValueError("Qwen3-Omni duplex plan_append only accepts a committed final turn")
        if not isinstance(payload, Mapping):
            raise ValueError("Qwen3-Omni duplex plan_append expects a mapping payload")

        audio_b64 = payload.get("audio")
        has_audio = isinstance(audio_b64, str) and bool(audio_b64)
        if not has_audio:
            raise ValueError("Qwen3-Omni duplex commit requires audio")
        wav, sample_rate_hz = _decode_pcm_f32le(payload)
        if wav.size == 0:
            raise ValueError("Qwen3-Omni duplex commit requires non-empty audio")

        system_prompt = str(runtime_config.get("qwen3_system_prompt") or DEFAULT_SYSTEM_PROMPT)
        user_text = runtime_config.get("initial_user_text")
        user_text = user_text if isinstance(user_text, str) else ""

        additional_information: dict[str, object] = {
            "qwen3_system_prompt": system_prompt,
            "qwen3_duplex": True,
            "session_id": fence.session_id,
            "epoch": fence.epoch,
            "turn_id": fence.turn_id,
            "is_speech": bool(payload.get("is_speech", True)),
        }
        prompt: dict[str, object] = {
            "prompt": _thinker_prompt(system_prompt=system_prompt, user_text=user_text),
            "multi_modal_data": {"audio": (wav, sample_rate_hz)},
            "additional_information": additional_information,
        }
        return DuplexAppendPlan(prompt=prompt)

    def observe_stage_output(
        self,
        *,
        stage_id: int,
        output: object,
        context: object,
    ) -> bool:
        """Project Thinker text to the client and keep forwarding Talker/Code2Wav."""
        del output, context
        return stage_id == _THINKER_STAGE_ID

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, object],
        output: object,
    ) -> DuplexOutputDecision | None:
        del stage_id, final_stage_id, segment_finished, segment_token_ids, segment_output_metadata, output
        return None

    def create_session_state(self) -> Qwen3OmniDuplexSessionState:
        return Qwen3OmniDuplexSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return qwen3_omni_duplex_capabilities(max_sessions=max_sessions)

    def validate_client_extra_body(self, extra_body: object) -> None:
        if extra_body is None:
            return
        if not isinstance(extra_body, dict):
            raise ValueError("Qwen3-Omni duplex extra_body must be an object")

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: object | None
    ) -> dict[str, object]:
        del model_config
        extra = config.extra_body if isinstance(config.extra_body, dict) else {}
        system_prompt = extra.get("qwen3_system_prompt") or config.instructions or DEFAULT_SYSTEM_PROMPT
        runtime: dict[str, object] = {
            "qwen3_system_prompt": str(system_prompt),
            "instructions": str(system_prompt),
        }
        initial_user_text = extra.get("duplex_initial_user_text")
        if not (isinstance(initial_user_text, str) and initial_user_text):
            initial_user_text = config.initial_user_text
        if isinstance(initial_user_text, str) and initial_user_text:
            runtime["initial_user_text"] = initial_user_text
        return runtime

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        updated = dict(current)
        extra = config.extra_body if isinstance(config.extra_body, dict) else {}
        if "qwen3_system_prompt" in extra:
            updated["qwen3_system_prompt"] = str(extra["qwen3_system_prompt"])
        if config.instructions:
            updated["instructions"] = config.instructions
            updated.setdefault("qwen3_system_prompt", config.instructions)
        extra_user_text = extra.get("duplex_initial_user_text")
        if isinstance(extra_user_text, str) and extra_user_text:
            updated["initial_user_text"] = extra_user_text
        elif isinstance(config.initial_user_text, str) and config.initial_user_text:
            updated["initial_user_text"] = config.initial_user_text
        return updated

    def data_plane_context(
        self,
        *,
        epoch: int,
        turn_id: int,
        active_response_turn_id: int | None,
        active_response_id: str | None,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> Qwen3OmniDataPlaneContext:
        del active_response_turn_id, active_response_id
        return Qwen3OmniDataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )


__all__ = ["DEFAULT_SYSTEM_PROMPT", "QWEN3_OMNI_TALKER_EOS_TOKEN_ID", "Qwen3OmniDuplexPlugin"]
