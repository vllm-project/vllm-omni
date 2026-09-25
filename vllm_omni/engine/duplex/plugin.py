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
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING

import pybase64 as base64

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
    # Deadline-aligned silence continuation state: the monotonic submission
    # time of the most recent native input unit and the next silence
    # continuation deadline. A real (non-silence) input resets the chain.
    last_native_submit_monotonic: float | None
    silence_deadline_monotonic: float | None

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


@dataclass
class DefaultDuplexModelSessionState(DuplexModelSessionState):
    """The one sane implementation of the per-session flags every model shares.

    A model plugin subclasses it to supply its ``audio_buffer`` (the only
    model-specific member) and keeps the bookkeeping the runner drives.
    """

    audio_buffer: PcmAppendBuffer
    input_since_commit: bool = False
    speech_since_commit: bool = False
    context_locked: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None
    # Deadline-aligned silence continuation state: the monotonic submission
    # time of the most recent native input unit and the next silence
    # continuation deadline. A real (non-silence) input resets the chain.
    last_native_submit_monotonic: float | None = None
    silence_deadline_monotonic: float | None = None

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
        self.last_native_submit_monotonic = None
        self.silence_deadline_monotonic = None


@dataclass(frozen=True, slots=True)
class DuplexDataPlaneContext:
    """Session state the runner hands a data plane to project one stage output."""

    epoch: int = 0
    turn_id: int = 0
    active_response_turn_id: int | None = None
    active_response_id: str | None = None
    auto_responds: bool = False
    response_format: str = "wav"
    speed: float | None = None
    modalities: tuple[str, ...] = ()


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


@dataclass(frozen=True, slots=True)
class PartialStageForward:
    """One downstream update the orchestrator should submit.

    ``close_only`` is a final update with no new sentence. ``output`` is the
    model-built payload; the orchestrator does not interpret its text.

    ``queue_close_after`` means this chunk still has text, but Stage1 has
    finished and an earlier sentence is already in flight. The text must be
    submitted resumable. A non-resumable submit is an end sentinel
    (``StreamingUpdate.from_request`` returns None) and aborts that sentence.
    """

    output: object
    is_final_update: bool
    close_only: bool = False
    queue_close_after: bool = False


class DuplexModelPlugin(ABC):
    """Everything vLLM-Omni needs to know about one full-duplex model.

    Engine policy (sampling params, append planning, output decisions) and
    session policy (capabilities, runtime configuration, per-session state,
    data-plane projection) live on the same object so a mismatch between the
    two halves is impossible by construction.
    """

    projects_intermediate_outputs: bool = False
    plugin_id: str = ""
    private_runtime_config_keys: frozenset[str] = frozenset()
    #: Samples per silence unit the runner appends to keep a model turn going.
    silence_continuation_samples: int = 16000
    #: Sample rate of that unit: the runner submits it through ``plan_append``
    #: exactly like client audio, so it must be a unit the model accepts.
    silence_continuation_sample_rate_hz: int = 16000
    data_plane: DuplexDataPlane

    def __init__(self, encode_audio: EncodeAudio) -> None:
        # Constructor-only: concrete plugins hand the encoder to their data plane.
        del encode_audio

    def silence_unit_payload(self) -> dict[str, object]:
        """One silence unit as an append payload (``pcm_f32le`` zeros).

        Used by the runner's turn continuation and by the startup warmup; a
        model whose unit is not plain zero PCM overrides it.
        """
        samples = int(self.silence_continuation_samples)
        return {
            "type": "audio",
            "audio": _silence_pcm_f32le_base64(samples),
            "format": "pcm_f32le",
            "sample_rate_hz": int(self.silence_continuation_sample_rate_hz),
        }

    # ---- engine policy (was DuplexRuntimeExtension) ----

    @abstractmethod
    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, object],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]: ...

    def prepare_prompt_config(
        self, config: dict[str, object], *, state: DuplexModelSessionState, payload: dict[str, object]
    ) -> dict[str, object]:
        """Add model-owned context before planning an append on the session loop."""
        return config

    async def prepare_append_plan(self, **kwargs) -> DuplexAppendPlan:
        """Prepare a plan; plugins may offload expensive work on owned snapshots."""
        return self.plan_append(**kwargs)

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

    def project_intermediate_output(
        self,
        *,
        stage_id: int,
        output: object,
        context: object,
    ) -> bool:
        """Return True to project this intermediate stage to the client.

        Unlike ``decide_output``, projecting does **not** short-circuit the
        pipeline: the stage output is still forwarded to the next stage.
        Default is off. Orthogonal to ``projects_intermediate_outputs``
        (Qwen3 Stage0); this hook is per-stage.
        """
        del stage_id, output, context
        return False

    def user_transcript(
        self,
        *,
        stage_id: int,
        output: object,
        prompt: object,
        finished: bool,
    ) -> str | None:
        """ASR text to show as the user's words, or None.

        Default models do not surface Stage0. AURA uses this for a spoken
        turn only; vision-follow commits stay off the transcript.
        """
        del stage_id, output, prompt, finished
        return None

    def plan_partial_stage_output(
        self,
        orchestrator: object,
        stage_id: int,
        replica_id: int,
        output: object,
        req_state: object,
    ) -> PartialStageForward | None:
        """Return a Talker update the orchestrator should submit, or None.

        Default models do not split Stage1 text. The orchestrator owns the
        actual ``_forward_to_next_stage`` call.
        """
        del orchestrator, stage_id, replica_id, output, req_state
        return None

    def partial_stage_followup(self, plan: PartialStageForward, req_state: object) -> PartialStageForward | None:
        """Optional second submit after ``plan`` has already been forwarded.

        Default models have nothing to add. AURA uses this to queue the
        end sentinel only after the last sentence text is already resumable.
        """
        del plan, req_state
        return None

    def commit_model_context(self, *, session_id: str | None, assistant_text: str) -> None:
        """Persist model-context history at a turn boundary. Default is a no-op.

        This is not playback-ACK history. A model that keeps its own prompt
        transcript implements this; the session runner only decides when.
        """
        del session_id, assistant_text

    def release_concurrent_turn_requests(
        self,
        *,
        stage_id: int,
        segment_finished: bool,
        output: object,
        context: object,
    ) -> bool:
        """Return True when the next user commit may start while prior TTS drains.

        The plugin chooses when that is safe. The runner must not hard-code a
        stage id. Default off. Unlike barge-in, this path must not cancel the
        old TTS.
        """
        del stage_id, segment_finished, output, context
        return False

    def draining_stage_ids(self, *, stage_count: int) -> frozenset[int]:
        """Output stages that may keep running after the next user turn starts.

        Empty means a concurrent turn does not overlap a previous output
        stage. Shared lifecycle reads this instead of assuming a stage layout.
        """
        del stage_count
        return frozenset()

    # ---- session policy (was ServingRuntimeAdapter) ----

    @abstractmethod
    def create_session_state(self) -> DuplexModelSessionState: ...

    @abstractmethod
    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities: ...

    def validate_client_extra_body(self, extra_body: object) -> None:
        """Refuse client ``extra_body`` keys the server owns (``private_runtime_config_keys``)."""
        if not isinstance(extra_body, Mapping):
            return
        private_keys = sorted(self.private_runtime_config_keys.intersection(extra_body))
        if private_keys:
            raise DuplexRuntimeConfigError(
                f"{self.plugin_id} runtime configuration is server-owned: " + ", ".join(private_keys)
            )

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
    ) -> object:
        """The context handed to ``data_plane.project``; the default is the generic dataclass."""
        return DuplexDataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            active_response_turn_id=active_response_turn_id,
            active_response_id=active_response_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )

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


@lru_cache(maxsize=8)
def _silence_pcm_f32le_base64(samples: int) -> str:
    return base64.b64encode(bytes(max(0, samples) * 4)).decode("ascii")


def payload_turn_id(payload: object) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    return coerce_int(payload.get("duplex_turn_id", payload.get("model_turn_id")))


def coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float | str | bytes | bytearray):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    return None


__all__ = [
    "DefaultDuplexModelSessionState",
    "DuplexDataPlane",
    "DuplexDataPlaneContext",
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
