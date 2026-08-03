# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3-Omni implementation of the model-neutral ``ServingRuntimeAdapter``.

Selected by ``PipelineConfig.duplex_serving_adapter`` and instantiated by
``load_serving_runtime_adapter`` with a single ``encode_audio`` callable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.experimental.fullduplex.qwen3omni.adapter import (
    Qwen3OmniNativeDuplexServingAdapter,
)
from vllm_omni.experimental.fullduplex.qwen3omni.data_plane import (
    EncodeAudio,
    Qwen3OmniDataPlaneContext,
    Qwen3OmniDataPlaneSession,
)
from vllm_omni.experimental.fullduplex.qwen3omni.session import (
    Qwen3OmniServingSessionState,
)


class Qwen3OmniServingRuntimeAdapter:
    """Model-owned serving state and policy for Qwen3-Omni duplex sessions."""

    adapter_id = "qwen3omni"
    # No response-text prefixes are stripped for this model; MiniCPM uses
    # these to clean up its own control-token artifacts.
    clean_response_done_prefix = ""
    interrupted_tts_prefix = ""
    private_runtime_config_keys = Qwen3OmniNativeDuplexServingAdapter.PRIVATE_RUNTIME_CONFIG_KEYS

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self.session_states: dict[str, Qwen3OmniServingSessionState] = {}
        self.data_plane = Qwen3OmniDataPlaneSession(encode_audio)

    # ---- session state ----------------------------------------------------

    def create_session_state(self) -> Qwen3OmniServingSessionState:
        return Qwen3OmniServingSessionState()

    def session_state(self, session_id: str) -> Qwen3OmniServingSessionState:
        state = self.session_states.get(session_id)
        if state is None:
            state = self.create_session_state()
            self.session_states[session_id] = state
        return state

    def remove_session_state(self, session_id: str) -> None:
        self.session_states.pop(session_id, None)

    # ---- policy delegation ------------------------------------------------

    @staticmethod
    def is_enabled(config: object) -> bool:
        return Qwen3OmniNativeDuplexServingAdapter.is_enabled(config)

    @staticmethod
    def capabilities(*, max_sessions: int) -> object:
        return Qwen3OmniNativeDuplexServingAdapter.capabilities(max_sessions=max_sessions)

    @staticmethod
    def validate_client_extra_body(extra_body: object) -> None:
        Qwen3OmniNativeDuplexServingAdapter.validate_client_extra_body(extra_body)

    @staticmethod
    async def prepare_runtime_config(config: object, *, model_config: Any) -> dict[str, object]:
        return await Qwen3OmniNativeDuplexServingAdapter.prepare_runtime_config(config, model_config=model_config)

    @staticmethod
    def runtime_config_for_update(config: object, current: Mapping[str, object]) -> dict[str, object]:
        return Qwen3OmniNativeDuplexServingAdapter.runtime_config_for_update(config, current)

    # ---- data-plane context ----------------------------------------------

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
    ) -> Qwen3OmniDataPlaneContext:
        return Qwen3OmniDataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            active_response_turn_id=active_response_turn_id,
            active_response_id=active_response_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )
