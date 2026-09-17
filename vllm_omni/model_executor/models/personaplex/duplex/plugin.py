# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex full-duplex model plugin: engine policy and session policy in one class.

PersonaPlex is a pure-lockstep model. Engine policy is one 80 ms frame per
resumable Stage 0 append with greedy one-token decoding; session policy is
the bundled voice prompt and the persona text, resolved once at open into the
server-owned runtime configuration the worker replays as its prefill.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import TYPE_CHECKING

from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexOutputDecision,
)
from vllm_omni.engine.duplex.plugin import (
    DefaultDuplexModelSessionState,
    DuplexModelPlugin,
    DuplexRuntimeConfigError,
    EncodeAudio,
    reject_changed_runtime_value,
)
from vllm_omni.model_executor.common.duplex.payload import decode_pcm_f32le_payload
from vllm_omni.model_executor.models.personaplex.duplex.capabilities import personaplex_capabilities
from vllm_omni.model_executor.models.personaplex.duplex.config import (
    DEFAULT_PERSONA,
    DEFAULT_VOICE,
    FRAME_SIZE,
    SAMPLE_RATE,
)
from vllm_omni.model_executor.models.personaplex.duplex.data_plane import PersonaPlexDataPlaneSession
from vllm_omni.model_executor.models.personaplex.duplex.input import PersonaPlexPcmAppendBuffer

if TYPE_CHECKING:
    from vllm.config import ModelConfig

#: Runtime-config keys the server derives from the session fields; a client
#: cannot set them through ``extra_body``.
PRIVATE_RUNTIME_CONFIG_KEYS = frozenset(
    {
        "personaplex_prefill_slots",
        "personaplex_model_path",
        "personaplex_voice_prompt",
        "personaplex_persona",
    }
)


@dataclass
class PersonaPlexSessionState(DefaultDuplexModelSessionState):
    """Per-session model state: the framework flags plus the 80 ms framing buffer."""

    audio_buffer: PersonaPlexPcmAppendBuffer = field(default_factory=PersonaPlexPcmAppendBuffer)


def voice_name(value: object) -> str:
    """A bundled ``.pt`` basename (``NATF2.pt``); paths are refused so the worker only reads the checkpoint."""
    voice = value if isinstance(value, str) and value else DEFAULT_VOICE
    path = PurePath(voice)
    if path.name != voice or path.suffix != ".pt" or any(part == ".." for part in path.parts):
        raise DuplexRuntimeConfigError("PersonaPlex voice must be a bundled .pt basename", code="invalid_voice")
    return voice


def persona_text(value: object) -> str:
    return str(value) if isinstance(value, str) and value else DEFAULT_PERSONA


def prefill_slot_count(runtime_config: Mapping[str, object]) -> int:
    raw = runtime_config.get("personaplex_prefill_slots", 0)
    try:
        return max(0, int(raw))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("personaplex_prefill_slots must be a non-negative integer") from exc


class PersonaPlexDuplexPlugin(DuplexModelPlugin):
    """PersonaPlex-owned sampling policy, append planning, session state and output projection."""

    plugin_id = "personaplex"
    private_runtime_config_keys = PRIVATE_RUNTIME_CONFIG_KEYS
    #: One codec frame: the runner keeps a model turn clocked with these when the client pauses.
    silence_continuation_samples = FRAME_SIZE
    silence_continuation_sample_rate_hz = SAMPLE_RATE

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self.data_plane = PersonaPlexDataPlaneSession(encode_audio)
        # The prefill length depends only on (model, voice, persona); the
        # voice bundle and tokenizer are read once per distinct triple.
        self._prefill_slots: dict[tuple[str, str, str], int] = {}
        self._prefill_lock = asyncio.Lock()

    # ---- engine policy (the resumable Stage 0 request) ----

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        del runtime_config
        if not defaults:
            return defaults
        configured = list(defaults)
        stage0 = defaults[0]
        if isinstance(stage0, SamplingParams):
            # Greedy, one temporal token per frame: the native stepper takes
            # argmax for the text head and the depformer.
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
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del sampling_params
        decode_pcm_f32le_payload(payload, sample_rate_hz=SAMPLE_RATE, exact_samples=FRAME_SIZE, model="PersonaPlex")
        normalized_payload = dict(payload)  # type: ignore[call-overload]
        # One scheduler slot per frame; the first append of an epoch also
        # carries the voice/persona prefill (a new epoch is a new Stage 0
        # request with fresh KV, so the worker replays it).
        prompt_slots = 1 + (prefill_slot_count(runtime_config) if seq <= 1 else 0)
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
                        "epoch": fence.epoch,
                        "turn_id": fence.turn_id,
                        "seq": seq,
                        "turn_seq": turn_seq,
                        "mode": "append_audio_chunk",
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
        segment_output_metadata: dict[str, object],
        output: object,
    ) -> DuplexOutputDecision | None:
        # Always-clocked model: what the client hears comes from the final
        # stage data plane, never from a Stage 0 listen/speak decision.
        del stage_id, final_stage_id, segment_finished, segment_token_ids, segment_output_metadata, output
        return None

    # ---- session policy ----

    def create_session_state(self) -> PersonaPlexSessionState:
        return PersonaPlexSessionState()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return personaplex_capabilities(max_sessions=max_sessions)

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        model_path = getattr(model_config, "model", None)
        if not isinstance(model_path, str) or not model_path:
            raise DuplexRuntimeConfigError("PersonaPlex model path is unavailable", code="model_path_unavailable")
        voice = voice_name(config.voice)
        persona = persona_text(config.instructions)
        prefill_slots = await self._prefill_slots_for(model_path, voice, persona)
        return {
            "personaplex_model_path": model_path,
            "personaplex_voice_prompt": voice,
            "personaplex_persona": persona,
            "personaplex_prefill_slots": prefill_slots,
        }

    async def _prefill_slots_for(self, model_path: str, voice: str, persona: str) -> int:
        key = (model_path, voice, persona)
        cached = self._prefill_slots.get(key)
        if cached is not None:
            return cached
        from vllm_omni.model_executor.models.personaplex.duplex.stage0 import personaplex_prefill_slots

        async with self._prefill_lock:
            cached = self._prefill_slots.get(key)
            if cached is not None:
                return cached
            try:
                # Reads the voice bundle and the tokenizer: off the orchestrator loop.
                slots = await asyncio.to_thread(personaplex_prefill_slots, model_path, voice, persona)
            except Exception as exc:
                raise DuplexRuntimeConfigError(
                    f"PersonaPlex voice/persona prefill could not be prepared: {exc}",
                    code="prefill_unavailable",
                ) from exc
            self._prefill_slots[key] = slots
            return slots

    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        new_persona = persona_text(config.instructions)
        reject_changed_runtime_value(
            new_persona,
            current.get("personaplex_persona"),
            message="PersonaPlex persona (instructions) cannot be changed after the session is created",
            code="persona_update_unsupported",
        )
        new_voice = voice_name(config.voice)
        reject_changed_runtime_value(
            new_voice,
            current.get("personaplex_voice_prompt"),
            message="PersonaPlex voice cannot be changed after the session is created",
            code="voice_update_unsupported",
        )
        runtime_config = deepcopy(dict(current))
        runtime_config["personaplex_voice_prompt"] = new_voice
        runtime_config["personaplex_persona"] = new_persona
        return runtime_config


__all__ = [
    "PRIVATE_RUNTIME_CONFIG_KEYS",
    "PersonaPlexDuplexPlugin",
    "PersonaPlexSessionState",
    "persona_text",
    "prefill_slot_count",
    "voice_name",
]
