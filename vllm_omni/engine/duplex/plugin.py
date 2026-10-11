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
    DuplexOutputContext,
    DuplexOutputDecision,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig


class DuplexRuntimeConfigError(ValueError):
    """A model plugin rejected client-visible runtime configuration."""

    def __init__(self, message: str, *, code: str = "invalid_duplex_runtime_config") -> None:
        super().__init__(message)
        self.code = code


def reject_private_runtime_keys(
    extra_body: object,
    private_runtime_config_keys: frozenset[str],
    *,
    message: str,
    error_cls: type[DuplexRuntimeConfigError] = DuplexRuntimeConfigError,
) -> None:
    """Reject client overrides while preserving each plugin's error contract."""
    if not isinstance(extra_body, dict):
        return
    private_keys = sorted(private_runtime_config_keys.intersection(extra_body))
    if private_keys:
        raise error_cls(message + ", ".join(private_keys))


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

    def prepare_backlog(self, *, operation_id: str, chunk_period_ms: int) -> PcmAppendReservation | None:
        """Reserve one more whole unit from audio already buffered, if there is one (input-clocked sessions).

        After an append emitted its unit, an input-clocked session calls this
        until it returns ``None``, so an append longer than one unit submits
        every whole unit it completed and is acknowledged after all of them.
        Default: ``None`` (one unit per append; the rest waits for later
        appends).
        """
        del operation_id, chunk_period_ms
        return None


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


@dataclass(slots=True)
class DefaultDuplexModelSessionState(DuplexModelSessionState):
    """Framework-owned flag anatomy, implemented once.

    The flag set above is the session runner's contract with the model (commit
    retention, deferred response/creates, silence-continuation bookkeeping); it
    is identical for every lockstep or frame-locked model, so a plugin only
    supplies its input packetizer as ``audio_buffer``.
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


@dataclass(frozen=True, slots=True)
class DuplexUnitDecision:
    """How a plugin decision taken at a stage segment end affects the model unit (input-clocked sessions)."""

    #: Reported in ``input_audio_buffer.processed.units[].decision`` (e.g. ``"listen"``).
    label: str
    #: False when the decision is a side channel and the unit still produces
    #: final-stage output (the unit then completes through ``unit_output_complete``).
    ends_unit: bool = True


@dataclass(frozen=True, slots=True)
class DuplexUnitOutputs:
    """What the engine knows about the oldest model unit still producing final-stage output."""

    #: 0-based index of the unit among all units of the session (the order they were
    #: created in): a stable key, not a position in the output stream (units that
    #: never reached Stage 0, or whose epoch was cancelled, have indices too).
    index: int
    #: Whether the stage just before the final stage reported its segment end for this unit.
    upstream_segment_finished: bool
    #: Whether a final-stage output credited to this unit marked a segment end.
    final_segment_finished: bool
    #: Session epoch the unit was submitted in (a cancel starts a new one).
    epoch: int = 0
    #: 1-based position of the unit among the Stage-0 submissions its epoch accepted
    #: (the n-th Stage-0 segment end of the epoch decided it), counting units whose
    #: decision ended them; units that never reached Stage 0 take none.
    ordinal: int = 0


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
    #: Whether sessions of this model may set ``extra_body.clock == "input"``;
    #: otherwise such a session is refused (``input_clock_unsupported``). No
    #: model opts in yet. A plugin opts in once ``unit_decision`` /
    #: ``unit_output_complete`` match its real unit boundaries, and only if
    #: Stage 0 ends every accepted submission with exactly one segment end, in
    #: submission order, with the unit's decision taken on that segment-ending
    #: output: the n-th Stage-0 segment end of an epoch decides the n-th
    #: accepted unit. An extra or missing segment end (e.g. a prefill-only
    #: append that ends no segment) shifts the epoch's later units onto the
    #: wrong segment ends: an acknowledgement may then arrive before its unit's
    #: output, or wait for the timeouts, until the next cancel starts a new
    #: epoch; the timeouts keep acknowledgements coming but do not resynchronise.
    supports_input_clock: bool = False
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

    # ---- input-clocked sessions (extra_body.clock == "input") ----

    def unit_decision(
        self,
        *,
        stage_id: int,
        decision: DuplexOutputDecision | None,
        output: object = None,
        context: DuplexOutputContext | None = None,
        runtime_config: Mapping[str, object] | None = None,
    ) -> DuplexUnitDecision | None:
        """Classify a stage segment end for the input clock.

        The input clock tracks every model unit (one Stage-0 submission) until
        all of its output has been emitted, so it can acknowledge the client
        input that caused it. Called when a stage output arrives (so it reads
        the live output at that moment), for every output of a stage before
        the final one that carries a decision, and for every segment end of
        such a stage without a decision (``decision is None``; ``output`` and
        ``context`` are that stage output). ``runtime_config`` is the
        session's runtime configuration (passed by the engine; the default
        only keeps direct calls short). Decisions taken on the final stage are
        not consulted. Default: no classification without a decision;
        otherwise the label is ``"listen"`` when the decision metadata sets
        ``model_listen`` (the framework key several models' data planes set),
        else the action, and any decision ends the unit (``ends_unit=True``);
        a plugin whose decisions do not all end the unit overrides this. A
        plugin whose stage can end a segment that hands the next stage nothing
        (so the unit would wait for a segment end that never comes) returns a
        decision with ``ends_unit=True`` for it.
        """
        del stage_id, output, context, runtime_config
        if decision is None:
            return None
        metadata = decision.metadata
        if metadata.get("model_listen") is True:
            label = "listen"
        else:
            label = str(getattr(decision.action, "value", decision.action))
        return DuplexUnitDecision(label=label, ends_unit=True)

    def unit_output_complete(
        self,
        *,
        unit: DuplexUnitOutputs | None,
        final_stage_id: int,
        new_output: object | None,
        new_context: DuplexOutputContext | None,
        state: dict[str, object],
        runtime_config: Mapping[str, object],
    ) -> bool:
        """Input clock: is the final-stage output of ``unit`` (the oldest speaking unit) complete?

        Called after every final-stage output (``new_output`` / ``new_context``,
        credited to ``unit``), after the segment end of the stage just before
        the final stage, and, once a unit completed, again for the next one
        with no new output (so one output may complete several units). The
        engine only calls it with a speaking unit (``unit`` is never None; the
        type keeps ``None`` for compatibility).

        The engine keeps no output history: a plugin that needs more than the
        flags of ``unit`` keeps it in ``state``, a dict it owns for the current
        epoch: the engine hands it a new, empty one whenever a cancel starts a
        new epoch, since the cancelled epoch's outputs are dropped and its
        units settled. For example, ``state["flagged_unit"] = unit.index``
        when ``new_output`` carries an end-of-unit flag; or, for a final stage
        that emits one frame per Stage-0 submission, a count of the epoch's
        frames compared with ``unit.ordinal`` (not ``unit.index``, which also
        counts units that never reached Stage 0 and those of earlier epochs).
        Output of a later stage that reaches the session before the Stage-0
        segment end that makes its unit speaking (the orchestrator may deliver
        different stages' outputs in either order) is held by the engine and
        handed to the hook, in arrival order, once that unit is speaking.
        ``runtime_config`` is the session's runtime configuration.

        A hook that raises is logged (once per session) and its call counts
        as "not complete": the unit is then settled by the timeouts. The same
        holds for ``unit_decision`` ("no decision").

        Default: the final stage marks one streaming segment end per unit, and
        final-stage outputs of different units do not interleave. A model whose
        final stage streams across unit boundaries must override this. Turns
        never overlap in an input-clocked session (it does not open the
        concurrent-turn gate, ``supports_concurrent_turn_requests``), because
        the clock credits outputs to the speaking units in order; crediting by
        request instead is future work.
        """
        del final_stage_id, new_output, new_context, state, runtime_config
        return unit is not None and unit.final_segment_finished

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

    def runtime_config_after_model_output(
        self,
        current: Mapping[str, object],
        output_metadata: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Return a runtime-config patch after a model output is observed.

        Plugins may use model-owned output metadata to retire server-side
        runtime state that has been consumed by the worker. The default keeps
        the framework unaware of model-specific metadata.
        """
        del current, output_metadata
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
    "DuplexUnitDecision",
    "DuplexUnitOutputs",
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
