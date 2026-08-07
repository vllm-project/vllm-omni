# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexCapabilities,
)
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    ServingRuntimeConfigError,
)
from vllm_omni.experimental.fullduplex.personaplex.config import DEFAULT_PERSONA
from vllm_omni.experimental.fullduplex.personaplex.data_plane import (
    PersonaPlexDataPlaneContext,
    PersonaPlexDataPlaneSession,
)
from vllm_omni.experimental.fullduplex.personaplex.input import (
    PersonaPlexPcmAppendBuffer,
)

EncodeAudio = Callable[[object, int, str, float | None], str | None]

_PRIVATE_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "personaplex_prefill_slots",
        "personaplex_model_path",
    }
)


@dataclass(slots=True)
class PersonaPlexServingSessionState:
    audio_buffer: PersonaPlexPcmAppendBuffer = field(default_factory=PersonaPlexPcmAppendBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    data_plane_task: asyncio.Task[None] | None = None
    data_plane_restart_requested: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None
    silence_continuation_scheduler: Callable[..., Awaitable[bool]] | None = None

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes += max(0, int(reserved_bytes))

    def clear_committed_audio(self) -> int:
        reserved_bytes = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        return reserved_bytes

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0
        self.pending_silence_task = None
        self.pending_silence_owner_id = None


class PersonaPlexServingRuntimeAdapter:
    adapter_id = "personaplex"
    clean_response_done_prefix = ""
    interrupted_tts_prefix = ""
    private_runtime_config_keys = _PRIVATE_RUNTIME_CONFIG_KEYS

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self.session_states: dict[str, PersonaPlexServingSessionState] = {}
        self.data_plane = PersonaPlexDataPlaneSession(encode_audio)

    def create_session_state(self) -> PersonaPlexServingSessionState:
        return PersonaPlexServingSessionState()

    def session_state(self, session_id: str) -> PersonaPlexServingSessionState:
        return self.session_states.setdefault(session_id, self.create_session_state())

    def remove_session_state(self, session_id: str) -> None:
        self.session_states.pop(session_id, None)

    @staticmethod
    def is_enabled(config: object) -> bool:
        del config
        return True

    @staticmethod
    def capabilities(*, max_sessions: int) -> DuplexCapabilities:
        supports_multi_session = max_sessions > 1
        return DuplexCapabilities(
            supports_model_native_turn_policy=True,
            supports_external_turn_signal=False,
            supports_client_commit=False,
            supports_barge_in=False,
            supports_playback_ack=True,
            supports_input_append=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=False,
            supports_rollback_to_checkpoint=False,
            supports_turn_commit_only=False,
            supports_model_internal_state=True,
            supports_stage_resumption=True,
            supports_core_resumable_request=True,
            supports_stage_connector_handoff=True,
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=supports_multi_session,
            supports_multi_session_same_replica=supports_multi_session,
            supports_session_lease=True,
            supports_session_resume=False,
            session_admission_mode="engine_managed",
            supports_audio_truncate=False,
            requires_model_runner_kv=True,
            requires_native_stage_role=True,
            implementation_level="model_native_duplex",
            adapter_patterns=["scheduler_data_plane"],
            input_modes=["append_audio_chunk"],
            signal_sources=["model_native", "client_event"],
            stage_handoff_transport="scheduler_data_plane",
            chunk_period_ms=80,
            target_barge_in_latency_ms=None,
        )

    @staticmethod
    def validate_client_extra_body(extra_body: object) -> None:
        if not isinstance(extra_body, dict):
            return
        private = sorted(_PRIVATE_RUNTIME_CONFIG_KEYS.intersection(extra_body))
        if private:
            raise ServingRuntimeConfigError("PersonaPlex runtime configuration is server-owned: " + ", ".join(private))

    @classmethod
    async def prepare_runtime_config(
        cls,
        config: object,
        *,
        model_config: Any,
    ) -> dict[str, object]:
        extra_body = getattr(config, "extra_body", None)
        cls.validate_client_extra_body(extra_body)
        model_path = getattr(model_config, "model", None)
        if not isinstance(model_path, str) or not model_path:
            raise ServingRuntimeConfigError("PersonaPlex model path is unavailable")
        voice = cls._voice_name(getattr(config, "voice", None))
        instructions = getattr(config, "instructions", None) or DEFAULT_PERSONA
        from vllm_omni.experimental.fullduplex.personaplex.stage0 import (
            personaplex_prefill_slots,
        )

        try:
            prefill_slots = await asyncio.to_thread(
                personaplex_prefill_slots,
                model_path,
                voice,
                str(instructions),
            )
        except Exception as exc:
            raise ServingRuntimeConfigError(f"PersonaPlex voice/persona prefill could not be prepared: {exc}") from exc
        return {
            "personaplex_model_path": model_path,
            "personaplex_voice_prompt": voice,
            "personaplex_persona": str(instructions),
            "personaplex_prefill_slots": prefill_slots,
        }

    @classmethod
    def runtime_config_for_update(
        cls,
        config: object,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        cls.validate_client_extra_body(getattr(config, "extra_body", None))
        runtime_config = deepcopy(dict(current))
        runtime_config["personaplex_voice_prompt"] = cls._voice_name(getattr(config, "voice", None))
        runtime_config["personaplex_persona"] = str(getattr(config, "instructions", None) or DEFAULT_PERSONA)
        return runtime_config

    @staticmethod
    def data_plane_context(
        *,
        epoch: int,
        turn_id: int,
        active_response_turn_id: int | None,
        active_response_id: str | None,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> PersonaPlexDataPlaneContext:
        return PersonaPlexDataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            active_response_turn_id=active_response_turn_id,
            active_response_id=active_response_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )

    @staticmethod
    def _voice_name(value: object) -> str:
        voice = value if isinstance(value, str) and value else "NATF2.pt"
        path = PurePath(voice)
        if path.name != voice or path.suffix != ".pt" or any(part == ".." for part in path.parts):
            raise ServingRuntimeConfigError("PersonaPlex voice must be a bundled .pt basename")
        return voice


__all__ = [
    "PersonaPlexServingRuntimeAdapter",
    "PersonaPlexServingSessionState",
]
