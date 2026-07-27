from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping
from importlib import import_module
from typing import Any, Protocol


class ServingRuntimeConfigError(ValueError):
    """A model serving plugin rejected client-visible runtime configuration."""

    def __init__(self, message: str, *, code: str = "invalid_duplex_runtime_config") -> None:
        super().__init__(message)
        self.code = code


class PcmAppendReservation(Protocol):
    operation_id: str
    payload: dict[str, object] | None

    @property
    def active(self) -> bool: ...

    @property
    def byte_count(self) -> int: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class PcmAppendBuffer(Protocol):
    @property
    def pending_byte_count(self) -> int: ...

    def clear(self) -> None: ...

    def clear_force_listen(self) -> None: ...

    def has_pending(self) -> bool: ...

    def has_reserved(self) -> bool: ...

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool,
    ) -> PcmAppendReservation | None: ...

    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> PcmAppendReservation: ...

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None: ...


class ServingRuntimeSessionState(Protocol):
    audio_buffer: PcmAppendBuffer
    input_since_commit: bool
    speech_since_commit: bool
    committed_audio_payload: dict[str, object] | None
    committed_audio_operation_id: str | None
    committed_audio_reserved_bytes: int
    deferred_response_create: bool
    deferred_precreate_response: bool
    data_plane_task: asyncio.Task[None] | None
    data_plane_restart_requested: bool
    continuation_owner_id: str | None
    continuation_units: int
    pending_silence_task: asyncio.Task[bool] | None
    pending_silence_owner_id: str | None
    silence_continuation_scheduler: Callable[..., Awaitable[bool]] | None

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None: ...

    def clear_committed_audio(self) -> int: ...

    def clear_continuation(self) -> None: ...


class RuntimeDataPlane(Protocol):
    def begin_request(self, request_id: str) -> None: ...

    def is_terminal(self, request_id: str | None) -> bool: ...

    def mark_terminal(self, request_id: str) -> None: ...

    def close_stream(self, request_id: str) -> None: ...

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None: ...

    def project(self, result: object, *, context: object | None = None) -> Iterable[dict[str, object]]: ...


class ServingRuntimeAdapter(Protocol):
    adapter_id: str
    session_states: MutableMapping[str, ServingRuntimeSessionState]
    data_plane: RuntimeDataPlane
    clean_response_done_prefix: str
    interrupted_tts_prefix: str
    private_runtime_config_keys: frozenset[str]

    def create_session_state(self) -> ServingRuntimeSessionState: ...

    def session_state(self, session_id: str) -> ServingRuntimeSessionState: ...

    def remove_session_state(self, session_id: str) -> None: ...

    def is_enabled(self, config: object) -> bool: ...

    def capabilities(self, *, max_sessions: int) -> object: ...

    def validate_client_extra_body(self, extra_body: object) -> None: ...

    async def prepare_runtime_config(self, config: object, *, model_config: Any) -> dict[str, object]: ...

    def runtime_config_for_update(
        self,
        config: object,
        current: Mapping[str, object],
    ) -> dict[str, object]: ...

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


def load_serving_runtime_adapter(
    path: str,
    encode_audio,
) -> ServingRuntimeAdapter:
    module_name, separator, attribute_name = path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid duplex serving runtime adapter path: {path!r}")
    adapter_type = getattr(import_module(module_name), attribute_name)
    return validate_serving_runtime_adapter(adapter_type(encode_audio))


def validate_serving_runtime_adapter(adapter: object) -> ServingRuntimeAdapter:
    required_methods = (
        "create_session_state",
        "session_state",
        "remove_session_state",
        "is_enabled",
        "capabilities",
        "validate_client_extra_body",
        "prepare_runtime_config",
        "runtime_config_for_update",
        "data_plane_context",
    )
    missing = [name for name in required_methods if not callable(getattr(adapter, name, None))]
    if missing:
        raise TypeError(f"Duplex serving runtime adapter is missing callable method(s): {', '.join(missing)}")
    if not isinstance(getattr(adapter, "adapter_id", None), str) or not adapter.adapter_id:
        raise TypeError("Duplex serving runtime adapter must declare adapter_id")
    if not isinstance(getattr(adapter, "session_states", None), MutableMapping):
        raise TypeError("Duplex serving runtime adapter must declare mutable session_states")
    for prefix_name in ("clean_response_done_prefix", "interrupted_tts_prefix"):
        if not isinstance(getattr(adapter, prefix_name, None), str):
            raise TypeError(f"Duplex serving runtime adapter must declare {prefix_name}")
    private_keys = getattr(adapter, "private_runtime_config_keys", None)
    if not isinstance(private_keys, frozenset) or any(not isinstance(key, str) for key in private_keys):
        raise TypeError("Duplex serving runtime adapter must declare private_runtime_config_keys as frozenset[str]")
    data_plane = getattr(adapter, "data_plane", None)
    if data_plane is None:
        raise TypeError("Duplex serving runtime adapter must declare data_plane")
    data_plane_methods = (
        "begin_request",
        "is_terminal",
        "mark_terminal",
        "close_stream",
        "close_session",
        "project",
    )
    missing_data_plane_methods = [name for name in data_plane_methods if not callable(getattr(data_plane, name, None))]
    if missing_data_plane_methods:
        raise TypeError(
            "Duplex serving runtime adapter data_plane is missing callable method(s): "
            + ", ".join(missing_data_plane_methods)
        )
    return adapter  # type: ignore[return-value]


def payload_turn_id(payload: object) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    return coerce_int(payload.get("duplex_turn_id", payload.get("model_turn_id")))


def coerce_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
