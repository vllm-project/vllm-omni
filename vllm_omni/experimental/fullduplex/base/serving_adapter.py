# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from vllm_omni.experimental.fullduplex.base.data_plane import (
    BaseDataPlaneSession,
    DataPlaneContext,
)
from vllm_omni.experimental.fullduplex.base.session_state import (
    BaseDuplexServingSessionState,
)
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexCapabilities

EncodeAudio = Callable[[object, int, str, float | None], str | None]


class BaseServingRuntimeAdapter(ABC):
    """Session state CRUD and data-plane context factory."""

    clean_response_done_prefix: str = ""
    interrupted_tts_prefix: str = ""

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self.session_states: dict[str, BaseDuplexServingSessionState] = {}
        self.data_plane: BaseDataPlaneSession = self._create_data_plane(encode_audio)

    @abstractmethod
    def _create_data_plane(self, encode_audio: EncodeAudio) -> BaseDataPlaneSession: ...

    @property
    @abstractmethod
    def adapter_id(self) -> str: ...

    @property
    @abstractmethod
    def private_runtime_config_keys(self) -> frozenset[str]: ...

    def create_session_state(self) -> BaseDuplexServingSessionState:
        return BaseDuplexServingSessionState()

    def session_state(self, session_id: str) -> BaseDuplexServingSessionState:
        state = self.session_states.get(session_id)
        if state is None:
            state = self.create_session_state()
            self.session_states[session_id] = state
        return state

    def remove_session_state(self, session_id: str) -> None:
        self.session_states.pop(session_id, None)

    @staticmethod
    @abstractmethod
    def is_enabled(config: object) -> bool: ...

    @staticmethod
    @abstractmethod
    def capabilities(*, max_sessions: int) -> DuplexCapabilities: ...

    @staticmethod
    @abstractmethod
    def validate_client_extra_body(extra_body: object) -> None: ...

    @staticmethod
    @abstractmethod
    async def prepare_runtime_config(config: object, *, model_config: Any) -> dict[str, object]: ...

    @staticmethod
    @abstractmethod
    def runtime_config_for_update(
        config: object,
        current: Mapping[str, object],
    ) -> dict[str, object]: ...

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
    ) -> DataPlaneContext:
        return DataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            active_response_turn_id=active_response_turn_id,
            active_response_id=active_response_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )
