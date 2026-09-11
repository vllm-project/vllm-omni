# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model plugin contract for full-duplex models.

One ``DuplexModelPlugin`` subclass per model binds what used to be two dotted
paths (the engine ``DuplexRuntimeExtension`` and the serving
``ServingRuntimeAdapter``). Everything runs engine-side now, so the plugin is
loaded once by ``DuplexOmniEngine`` and handed to ``DuplexOrchestrator`` /
``DuplexSessionManager``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from importlib import import_module
from typing import TYPE_CHECKING

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexAppendPlan,
    DuplexFence,
    DuplexOutputDecision,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig


class DuplexRuntimeConfigError(ValueError):
    """A model plugin rejected client-visible runtime configuration."""

    def __init__(self, message: str, *, code: str = "invalid_duplex_runtime_config") -> None:
        super().__init__(message)
        self.code = code


def reject_changed_runtime_value(
    new_value: object,
    current_value: object,
    *,
    message: str,
    code: str,
    error_cls: type[DuplexRuntimeConfigError] = DuplexRuntimeConfigError,
) -> None:
    if new_value != current_value:
        raise error_cls(message, code=code)


class PcmAppendReservation(ABC):
    operation_id: str
    payload: dict[str, object] | None

    @property
    @abstractmethod
    def active(self) -> bool: ...

    @property
    @abstractmethod
    def byte_count(self) -> int: ...

    @abstractmethod
    def commit(self) -> None: ...

    @abstractmethod
    def rollback(self) -> None: ...


class PcmAppendBuffer(ABC):
    @property
    @abstractmethod
    def pending_byte_count(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def clear_force_listen(self) -> None: ...

    @abstractmethod
    def has_pending(self) -> bool: ...

    @abstractmethod
    def has_reserved(self) -> bool: ...

    @abstractmethod
    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool,
    ) -> PcmAppendReservation | None: ...

    @abstractmethod
    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> PcmAppendReservation: ...

    @abstractmethod
    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None: ...


class DuplexModelSessionState(ABC):
    """Model-owned per-session state; owned by the session runner (one per session)."""

    audio_buffer: PcmAppendBuffer
    input_since_commit: bool
    speech_since_commit: bool
    context_locked: bool
    committed_audio_payload: dict[str, object] | None
    committed_audio_operation_id: str | None
    committed_audio_reserved_bytes: int
    deferred_response_create: bool
    deferred_precreate_response: bool
    continuation_owner_id: str | None
    continuation_units: int
    pending_silence_task: asyncio.Task[bool] | None
    pending_silence_owner_id: str | None

    @abstractmethod
    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None: ...

    @abstractmethod
    def clear_committed_audio(self) -> int: ...

    @abstractmethod
    def clear_continuation(self) -> None: ...


class DuplexDataPlane(ABC):
    """Projects raw stage outputs of one model into internal duplex events."""

    @abstractmethod
    def begin_request(self, request_id: str) -> None: ...

    @abstractmethod
    def is_terminal(self, request_id: str | None) -> bool: ...

    @abstractmethod
    def mark_terminal(self, request_id: str) -> None: ...

    @abstractmethod
    def close_stream(self, request_id: str) -> None: ...

    @abstractmethod
    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None: ...

    @abstractmethod
    def project(self, result: object, *, context: object | None = None) -> Iterable[dict[str, object]]: ...


EncodeAudio = Callable[[object, int, str, float | None], str | None]


class DuplexModelPlugin(ABC):
    """Everything vLLM-Omni needs to know about one full-duplex model.

    Engine policy (sampling params, append planning, output decisions) and
    session policy (capabilities, runtime configuration, per-session state,
    data-plane projection) live on the same object so a mismatch between the
    two halves is impossible by construction.
    """

    plugin_id: str = ""
    private_runtime_config_keys: frozenset[str] = frozenset()
    #: Samples per silence unit the runner appends to keep a model turn going.
    silence_continuation_samples: int = 16000
    data_plane: DuplexDataPlane

    def __init__(self, encode_audio: EncodeAudio) -> None:
        # Constructor-only: concrete plugins hand the encoder to their data plane.
        del encode_audio

    # ---- engine policy (was DuplexRuntimeExtension) ----

    @abstractmethod
    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]: ...

    @abstractmethod
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
    ) -> DuplexAppendPlan: ...

    @abstractmethod
    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, object],
        output: object,
    ) -> DuplexOutputDecision | None: ...

    # ---- session policy (was ServingRuntimeAdapter) ----

    @abstractmethod
    def create_session_state(self) -> DuplexModelSessionState: ...

    @abstractmethod
    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities: ...

    @abstractmethod
    def validate_client_extra_body(self, extra_body: object) -> None: ...

    @abstractmethod
    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]: ...

    @abstractmethod
    def runtime_config_for_update(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
    ) -> dict[str, object]: ...

    @abstractmethod
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
    ) -> object: ...

    # Optional hook: build the runtime config patch for a function-call output
    # item. Plugins without tools keep the default (no change).
    def runtime_config_for_function_output(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
        item: Mapping[str, object],
    ) -> dict[str, object] | None:
        del config, current, item
        return None


def load_duplex_plugin(path: str, encode_audio: EncodeAudio) -> DuplexModelPlugin:
    module_name, separator, attribute_name = path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid duplex plugin path: {path!r}")
    plugin_type = getattr(import_module(module_name), attribute_name)
    plugin = plugin_type(encode_audio)
    if not isinstance(plugin, DuplexModelPlugin):
        raise TypeError(f"{path!r} is not a DuplexModelPlugin")
    if not plugin.plugin_id:
        raise TypeError("Duplex plugin must declare plugin_id")
    if not isinstance(getattr(plugin, "data_plane", None), DuplexDataPlane):
        raise TypeError("Duplex plugin must declare a DuplexDataPlane as data_plane")
    return plugin


def validate_duplex_plugin_sampling(plugin: DuplexModelPlugin, *, sampling_defaults: tuple[object, ...]) -> None:
    """Fail fast when the plugin cannot produce one sampling parameter per stage."""
    configured = plugin.configure_sampling_params(runtime_config={}, defaults=sampling_defaults)
    if not isinstance(configured, tuple):
        raise TypeError("Duplex plugin must return sampling parameters as a tuple")
    if len(configured) != len(sampling_defaults):
        raise ValueError("Duplex plugin must return one sampling parameter per stage")
    for stage_id, (value, default) in enumerate(zip(configured, sampling_defaults, strict=True)):
        if default is not None and not isinstance(value, type(default)):
            raise TypeError(
                "Duplex plugin sampling parameter type mismatch "
                f"for stage {stage_id}: expected {type(default).__name__}, got {type(value).__name__}"
            )


def payload_turn_id(payload: object) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    return coerce_int(payload.get("duplex_turn_id", payload.get("model_turn_id")))


def coerce_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "DuplexDataPlane",
    "DuplexModelPlugin",
    "DuplexModelSessionState",
    "DuplexRuntimeConfigError",
    "EncodeAudio",
    "PcmAppendBuffer",
    "PcmAppendReservation",
    "coerce_int",
    "load_duplex_plugin",
    "payload_turn_id",
    "reject_changed_runtime_value",
    "validate_duplex_plugin_sampling",
]
