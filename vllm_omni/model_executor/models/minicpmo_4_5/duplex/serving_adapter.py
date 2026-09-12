# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from vllm_omni.entrypoints.duplex.protocol import DuplexCapabilities
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.adapter import (
    MiniCPMO45NativeDuplexServingAdapter,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.capabilities import (
    minicpmo45_native_capabilities,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import (
    MiniCPMO45DataPlaneContext,
    MiniCPMO45DataPlaneSession,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.session import (
    MiniCPMO45ServingSessionState,
)

EncodeAudio = Callable[[object, int, str, float | None], str | None]


class MiniCPMO45ServingRuntimeAdapter:
    """MiniCPM-owned serving state, input packing, and output projection."""

    adapter_id = "minicpmo45"
    # MiniCPM may need silent decision units after the user's final chunk.
    # Frame-clocked adapters must not inherit this model-turn policy.
    supports_silence_continuation = True
    clean_response_done_prefix = ""
    interrupted_tts_prefix = ""
    private_runtime_config_keys = MiniCPMO45NativeDuplexServingAdapter.PRIVATE_RUNTIME_CONFIG_KEYS

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self.session_states: dict[str, MiniCPMO45ServingSessionState] = {}
        self.data_plane = MiniCPMO45DataPlaneSession(encode_audio)

    def create_session_state(self) -> MiniCPMO45ServingSessionState:
        return MiniCPMO45ServingSessionState()

    def session_state(self, session_id: str) -> MiniCPMO45ServingSessionState:
        state = self.session_states.get(session_id)
        if state is None:
            state = self.create_session_state()
            self.session_states[session_id] = state
        return state

    def remove_session_state(self, session_id: str) -> None:
        self.session_states.pop(session_id, None)

    @staticmethod
    def is_enabled(config: object) -> bool:
        return MiniCPMO45NativeDuplexServingAdapter.is_enabled(config)  # type: ignore[arg-type]

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        caps = minicpmo45_native_capabilities(max_sessions=max_sessions)
        if getattr(self, "_gander_enabled", False) and caps.supports_prompt_replay:
            caps.supports_context_replacement = True
        return caps

    @staticmethod
    def validate_client_extra_body(extra_body: object) -> None:
        MiniCPMO45NativeDuplexServingAdapter.validate_client_extra_body(extra_body)

    async def prepare_runtime_config(self, config: object, *, model_config: Any) -> dict[str, object]:
        self._gander_enabled = bool(getattr(getattr(model_config, "hf_config", None), "gander_unit8", False))
        return await MiniCPMO45NativeDuplexServingAdapter.prepare_runtime_config(
            config,  # type: ignore[arg-type]
            model_config=model_config,
        )

    @staticmethod
    def runtime_config_for_update(
        config: object,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        return MiniCPMO45NativeDuplexServingAdapter.runtime_config_for_update(
            config,  # type: ignore[arg-type]
            dict(current),
        )

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
    ) -> MiniCPMO45DataPlaneContext:
        return MiniCPMO45DataPlaneContext(
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
    def prepare_context_input(item: dict, current: dict, *, epoch: int):
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_tools import prepare_context_input

        return prepare_context_input(item, current, epoch=epoch)

    @staticmethod
    def reconcile_context_config(candidate: dict, current: dict) -> dict:
        # Output registration runs while the engine configuration RPC awaits.
        # Keep newly registered calls, while candidate owns result updates for
        # calls that were already present when this context input was prepared.
        calls = dict(current.get("gander_calls", {}))
        calls.update(candidate.get("gander_calls", {}))
        return {**candidate, "gander_calls": calls}

    @staticmethod
    def register_function_call(native: dict, runtime: dict, *, epoch: int) -> dict:
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_tools import register_call

        return register_call(native, runtime, epoch=epoch)

    @staticmethod
    def prepare_context_replacement(item, current, *, epoch):
        from vllm_omni.model_executor.models.minicpmo_4_5.gander_tools import prepare_context_replacement

        return prepare_context_replacement(item, current, epoch=epoch)

    @staticmethod
    def context_input_requires_replacement(item):
        return isinstance(item, dict) and (item.get("kind") == "task_slate" or item.get("preempt") is True)

    @staticmethod
    def context_wakeup_payload(runtime):
        return {
            "gander_control": True,
            "gander_wake": True,
            "token_ids": [],
            "context_version": runtime["duplex_context_version"],
        }

    @staticmethod
    def prepare_append_context_policy(payload, runtime, *, response_active):
        if runtime.get("gander_enabled") and isinstance(payload, dict):
            # Keep this decision on the operation payload, including retries.
            payload.setdefault("gander_defer_rollover", bool(response_active))
        return payload
