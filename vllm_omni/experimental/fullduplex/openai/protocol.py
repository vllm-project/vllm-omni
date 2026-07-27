from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from uuid import uuid4


class DuplexOverlapPolicy(str, Enum):
    AUTO = "auto"
    LISTEN_ONLY = "listen_only"
    BARGE_IN_ON_SPEECH = "barge_in_on_speech"


class DuplexPlaybackCommitPolicy(str, Enum):
    COMMIT_ALL_ON_DONE = "commit_all_on_done"
    ACK_ONLY = "ack_only"


class DuplexSessionState(str, Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class DuplexTurnState(str, Enum):
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    USER_COMMITTED = "user_committed"
    ASSISTANT_GENERATING = "assistant_generating"
    ASSISTANT_PLAYING = "assistant_playing"
    BARGE_IN = "barge_in"


class DuplexTurnEventType(str, Enum):
    USER_STARTED = "user_started"
    USER_COMMITTED = "user_committed"
    ASSISTANT_STARTED = "assistant_started"
    ASSISTANT_DONE = "assistant_done"
    BARGE_IN = "barge_in"
    PLAYBACK_ACK = "playback_ack"
    TIMEOUT = "timeout"
    CLOSE = "close"


@dataclass
class DuplexCapabilities:
    """Runtime/model capabilities exposed by the duplex serving protocol.

    These are intentionally explicit so the serving layer does not assume all
    duplex models support the same input append, rollback, or turn policy.
    ``supports_core_kv_lease`` is reserved for scheduler-owned KV lifecycle;
    model-owned decoder/TTS state must use ``supports_model_internal_state``.
    ``supports_core_resumable_request`` means the scheduler can resume the
    same request id across streaming updates, but it is not a KV lease by
    itself. Realtime support means this endpoint can speak the native Realtime
    event schema for the supported audio duplex paths while keeping model- or
    scheduler-specific limits explicit in the capability payload.
    """

    supports_session_adapter: bool = True
    supports_model_native_turn_policy: bool = False
    supports_external_turn_signal: bool = True
    supports_client_commit: bool = True
    supports_barge_in: bool = True
    supports_playback_ack: bool = True
    supports_input_append: bool = False
    supports_replace_latest_chunk: bool = True
    supports_reencode_context: bool = True
    supports_rollback_to_checkpoint: bool = False
    supports_turn_commit_only: bool = True
    supports_kv_lease: bool = False
    supports_core_kv_lease: bool = False
    supports_model_internal_state: bool = False
    supports_stage_resumption: bool = False
    supports_scheduler_native_append: bool = False
    supports_core_resumable_request: bool = False
    supports_stage_connector_handoff: bool = False
    supports_independent_io_streams: bool = False
    supports_realtime_endpoint: bool = False
    supports_multi_session: bool = False
    supports_multi_session_same_replica: bool = False
    supports_session_lease: bool = False
    supports_session_resume: bool = False
    session_admission_mode: str = "serving_managed"
    supports_audio_truncate: bool = False
    requires_model_runner_kv: bool = False
    requires_native_stage_role: bool = False
    implementation_level: str = "serving_session_adapter"
    adapter_patterns: list[str] = field(default_factory=lambda: ["chunk_group_append"])
    input_modes: list[str] = field(default_factory=lambda: ["turn_commit_only", "reencode_context"])
    signal_sources: list[str] = field(default_factory=lambda: ["client_event", "server_policy", "model_native"])
    stage_handoff_transport: str | None = None
    chunk_period_ms: int | None = 1000
    target_barge_in_latency_ms: int | None = 1000

    @classmethod
    def minicpmo45_native(cls, *, max_sessions: int = 1) -> DuplexCapabilities:
        supports_multi_session = max_sessions > 1
        return cls(
            supports_model_native_turn_policy=True,
            supports_barge_in=False,
            supports_input_append=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=False,
            supports_turn_commit_only=False,
            supports_kv_lease=False,
            supports_core_kv_lease=False,
            supports_model_internal_state=True,
            supports_stage_resumption=True,
            supports_scheduler_native_append=False,
            supports_core_resumable_request=True,
            supports_stage_connector_handoff=True,
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=supports_multi_session,
            supports_multi_session_same_replica=supports_multi_session,
            supports_session_lease=True,
            supports_session_resume=True,
            session_admission_mode="engine_managed",
            supports_audio_truncate=True,
            requires_model_runner_kv=True,
            requires_native_stage_role=True,
            implementation_level="model_native_duplex",
            adapter_patterns=["scheduler_data_plane"],
            input_modes=["append_audio_chunk"],
            signal_sources=["model_native", "client_event", "server_policy"],
            stage_handoff_transport="scheduler_data_plane",
            chunk_period_ms=1000,
            target_barge_in_latency_ms=None,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "supports_session_adapter": self.supports_session_adapter,
            "supports_model_native_turn_policy": self.supports_model_native_turn_policy,
            "supports_external_turn_signal": self.supports_external_turn_signal,
            "supports_client_commit": self.supports_client_commit,
            "supports_barge_in": self.supports_barge_in,
            "supports_playback_ack": self.supports_playback_ack,
            "supports_input_append": self.supports_input_append,
            "supports_replace_latest_chunk": self.supports_replace_latest_chunk,
            "supports_reencode_context": self.supports_reencode_context,
            "supports_rollback_to_checkpoint": self.supports_rollback_to_checkpoint,
            "supports_turn_commit_only": self.supports_turn_commit_only,
            "supports_kv_lease": self.supports_kv_lease,
            "supports_core_kv_lease": self.supports_core_kv_lease,
            "supports_model_internal_state": self.supports_model_internal_state,
            "supports_stage_resumption": self.supports_stage_resumption,
            "supports_scheduler_native_append": self.supports_scheduler_native_append,
            "supports_core_resumable_request": self.supports_core_resumable_request,
            "supports_stage_connector_handoff": self.supports_stage_connector_handoff,
            "supports_independent_io_streams": self.supports_independent_io_streams,
            "supports_realtime_endpoint": self.supports_realtime_endpoint,
            "supports_multi_session": self.supports_multi_session,
            "supports_multi_session_same_replica": self.supports_multi_session_same_replica,
            "supports_session_lease": self.supports_session_lease,
            "supports_session_resume": self.supports_session_resume,
            "session_admission_mode": self.session_admission_mode,
            "supports_audio_truncate": self.supports_audio_truncate,
            "requires_model_runner_kv": self.requires_model_runner_kv,
            "requires_native_stage_role": self.requires_native_stage_role,
            "implementation_level": self.implementation_level,
            "adapter_patterns": self.adapter_patterns,
            "input_modes": self.input_modes,
            "signal_sources": self.signal_sources,
            "stage_handoff_transport": self.stage_handoff_transport,
            "chunk_period_ms": self.chunk_period_ms,
            "target_barge_in_latency_ms": self.target_barge_in_latency_ms,
        }


@dataclass
class DuplexPlaybackCursor:
    generated_ms: int = 0
    sent_ms: int = 0
    played_ms: int = 0
    committed_ms: int = 0

    def acknowledge(self, played_ms: int, committed_ms: int | None = None) -> None:
        self.played_ms = max(self.played_ms, max(0, int(played_ms)))
        if committed_ms is None:
            committed_ms = self.played_ms
        self.committed_ms = max(self.committed_ms, max(0, int(committed_ms)))

    def truncate_committed(self, committed_ms: int) -> None:
        self.committed_ms = max(0, min(max(self.sent_ms, self.generated_ms), int(committed_ms)))

    def as_dict(self) -> dict[str, int]:
        return {
            "generated_ms": self.generated_ms,
            "sent_ms": self.sent_ms,
            "played_ms": self.played_ms,
            "committed_ms": self.committed_ms,
        }

    def snapshot(self) -> DuplexPlaybackView:
        return DuplexPlaybackView(**self.as_dict())


@dataclass(frozen=True, slots=True)
class DuplexPlaybackView:
    generated_ms: int = 0
    sent_ms: int = 0
    played_ms: int = 0
    committed_ms: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "generated_ms": self.generated_ms,
            "sent_ms": self.sent_ms,
            "played_ms": self.played_ms,
            "committed_ms": self.committed_ms,
        }


@dataclass
class DuplexAudioChunk:
    data: str
    format: str = "wav"
    sample_rate_hz: int | None = None


@dataclass
class DuplexSessionConfig:
    model: str | None = None
    modalities: list[str] = field(default_factory=lambda: ["text", "audio"])
    instructions: str | None = None
    voice: str | None = None
    ref_audio: str | None = None
    response_format: str = "wav"
    temperature: float | None = None
    max_tokens: int | None = None
    speed: float | None = None
    use_tts_template: bool = True
    idle_timeout_s: float = 300.0
    overlap_policy: str = DuplexOverlapPolicy.LISTEN_ONLY.value
    overlap_short_ack_ms: int = 700
    overlap_barge_in_ms: int = 1200
    overlap_silence_rms: float = 0.003
    playback_commit_policy: str = DuplexPlaybackCommitPolicy.COMMIT_ALL_ON_DONE.value
    extra_body: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "modalities": list(self.modalities),
            "instructions": self.instructions,
            "voice": self.voice,
            "ref_audio": self.ref_audio,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "speed": self.speed,
            "use_tts_template": self.use_tts_template,
            "idle_timeout_s": self.idle_timeout_s,
            "overlap_policy": self.overlap_policy,
            "overlap_short_ack_ms": self.overlap_short_ack_ms,
            "overlap_barge_in_ms": self.overlap_barge_in_ms,
            "overlap_silence_rms": self.overlap_silence_rms,
            "playback_commit_policy": self.playback_commit_policy,
            "extra_body": dict(self.extra_body),
        }

    @classmethod
    def from_event(cls, event: dict[str, object]) -> DuplexSessionConfig:
        payload = event.get("session")
        if isinstance(payload, dict):
            source = payload
        else:
            source = event

        config = cls()
        if isinstance(source.get("model"), str):
            config.model = source["model"]
        if isinstance(source.get("instructions"), str):
            config.instructions = source["instructions"]
        if isinstance(source.get("voice"), str):
            config.voice = source["voice"]
        if isinstance(source.get("ref_audio"), str):
            config.ref_audio = source["ref_audio"]
        if isinstance(source.get("response_format"), str):
            config.response_format = source["response_format"]
        if isinstance(source.get("use_tts_template"), bool):
            config.use_tts_template = bool(source["use_tts_template"])
        if isinstance(source.get("temperature"), int | float):
            config.temperature = float(source["temperature"])
        if isinstance(source.get("max_tokens"), int):
            config.max_tokens = int(source["max_tokens"])
        if isinstance(source.get("speed"), int | float):
            config.speed = float(source["speed"])
        if isinstance(source.get("idle_timeout_s"), int | float):
            config.idle_timeout_s = float(source["idle_timeout_s"])
        if isinstance(source.get("overlap_policy"), str):
            config.overlap_policy = cls._normalize_overlap_policy(source["overlap_policy"])
        if isinstance(source.get("overlap_short_ack_ms"), int | float):
            config.overlap_short_ack_ms = max(0, int(source["overlap_short_ack_ms"]))
        if isinstance(source.get("overlap_barge_in_ms"), int | float):
            config.overlap_barge_in_ms = max(0, int(source["overlap_barge_in_ms"]))
        if isinstance(source.get("overlap_silence_rms"), int | float):
            config.overlap_silence_rms = max(0.0, float(source["overlap_silence_rms"]))
        if isinstance(source.get("playback_commit_policy"), str):
            config.playback_commit_policy = cls._normalize_playback_commit_policy(source["playback_commit_policy"])
        if isinstance(source.get("modalities"), list) and all(isinstance(x, str) for x in source["modalities"]):
            config.modalities = list(source["modalities"])
        if isinstance(source.get("extra_body"), dict):
            config.extra_body = dict(source["extra_body"])
            extra = config.extra_body
            if isinstance(extra.get("overlap_policy"), str):
                config.overlap_policy = cls._normalize_overlap_policy(extra["overlap_policy"])
            if isinstance(extra.get("overlap_short_ack_ms"), int | float):
                config.overlap_short_ack_ms = max(0, int(extra["overlap_short_ack_ms"]))
            if isinstance(extra.get("overlap_barge_in_ms"), int | float):
                config.overlap_barge_in_ms = max(0, int(extra["overlap_barge_in_ms"]))
            if isinstance(extra.get("overlap_silence_rms"), int | float):
                config.overlap_silence_rms = max(0.0, float(extra["overlap_silence_rms"]))
            if isinstance(extra.get("playback_commit_policy"), str):
                config.playback_commit_policy = cls._normalize_playback_commit_policy(extra["playback_commit_policy"])
        return config

    @staticmethod
    def _normalize_overlap_policy(value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {policy.value for policy in DuplexOverlapPolicy}:
            return normalized
        return DuplexOverlapPolicy.LISTEN_ONLY.value

    @staticmethod
    def _normalize_playback_commit_policy(value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {policy.value for policy in DuplexPlaybackCommitPolicy}:
            return normalized
        return DuplexPlaybackCommitPolicy.COMMIT_ALL_ON_DONE.value


@dataclass(frozen=True)
class ResponseCreateOptions:
    instructions: str | None = None
    voice: str | None = None
    response_format: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    speed: float | None = None
    modalities: tuple[str, ...] | None = None
    extra_body: Mapping[str, object] = field(default_factory=dict)

    def apply_to(self, config: DuplexSessionConfig) -> None:
        for field_name in ("instructions", "voice", "response_format", "temperature", "max_tokens", "speed"):
            value = getattr(self, field_name)
            if value is not None:
                setattr(config, field_name, value)
        if self.modalities is not None:
            config.modalities = list(self.modalities)
        config.extra_body.update(self.extra_body)


@dataclass
class DuplexCommittedInput:
    message: dict[str, object]
    turn_id: int
    epoch: int
    input_commit_seq: int


@dataclass
class DuplexAssistantAudioTextMark:
    text_chars: int
    audio_end_ms: int


@dataclass
class InputBufferState:
    commit_seq: int = 0
    pending_text: list[str] = field(default_factory=list)
    pending_audio: list[DuplexAudioChunk] = field(default_factory=list)
    overlap_speech_ms: int = 0
    reserved_input_bytes: int = 0
    pending_turns: int = 0


@dataclass
class ResponseState:
    active_request_id: str | None = None
    active_response_id: str | None = None
    active_response_turn_id: int | None = None
    last_response_id: str | None = None
    assistant_text_buffer: list[str] = field(default_factory=list)
    assistant_audio_text_marks: list[DuplexAssistantAudioTextMark] = field(default_factory=list)
    pending_options: ResponseCreateOptions | None = None
    active_options: ResponseCreateOptions | None = None
    active_config: DuplexSessionConfig | None = None
    stage_metrics: dict[str, dict[str, object]] = field(default_factory=dict)
    stage_metric_tpot_weighted_ms: dict[str, float] = field(default_factory=dict)
    stage_metric_tpot_weight: dict[str, int] = field(default_factory=dict)


@dataclass
class PlaybackLedger:
    current: DuplexPlaybackCursor = field(default_factory=DuplexPlaybackCursor)
    by_response: dict[str, DuplexPlaybackCursor] = field(default_factory=dict)


@dataclass
class ConversationHistory:
    messages: list[dict[str, object]] = field(default_factory=list)
    item_ids: dict[str, dict[str, object]] = field(default_factory=dict)
    item_audio_text_marks: dict[str, list[DuplexAssistantAudioTextMark]] = field(default_factory=dict)
    pending_item_ids: dict[str, dict[str, object]] = field(default_factory=dict)
    pending_item_audio_text_marks: dict[str, list[DuplexAssistantAudioTextMark]] = field(default_factory=dict)
    pending_truncations_ms: dict[str, int] = field(default_factory=dict)
    last_assistant_full_message: dict[str, object] | None = None
    last_assistant_audio_text_marks: list[DuplexAssistantAudioTextMark] = field(default_factory=list)


@dataclass
class DuplexSession:
    session_id: str
    config: DuplexSessionConfig
    capabilities: DuplexCapabilities = field(default_factory=DuplexCapabilities)
    incarnation: int = 0
    state: DuplexSessionState = DuplexSessionState.OPEN
    turn_state: DuplexTurnState = DuplexTurnState.IDLE
    epoch: int = 0
    turn_id: int = 0
    _input: InputBufferState = field(default_factory=InputBufferState, repr=False)
    _response: ResponseState = field(default_factory=ResponseState, repr=False)
    _playback: PlaybackLedger = field(default_factory=PlaybackLedger, repr=False)
    _conversation: ConversationHistory = field(default_factory=ConversationHistory, repr=False)
    _runtime_config: dict[str, object] = field(default_factory=dict, repr=False)

    @property
    def response_config(self) -> DuplexSessionConfig:
        """Return the immutable-for-this-lifecycle response configuration.

        ``config`` remains the session defaults.  A response takes a deep
        snapshot when it begins so response.create overrides and concurrent
        session updates cannot mutate each other's ownership domains.
        """
        return self._response.active_config or self.config

    @property
    def runtime_config(self) -> Mapping[str, object]:
        return MappingProxyType(dict(self._runtime_config))

    def replace_runtime_config(self, runtime_config: Mapping[str, object]) -> None:
        self._runtime_config = dict(runtime_config)

    @property
    def input_commit_seq(self) -> int:
        return self._input.commit_seq

    @property
    def pending_text(self) -> tuple[str, ...]:
        return tuple(self._input.pending_text)

    @property
    def pending_audio(self) -> tuple[DuplexAudioChunk, ...]:
        return tuple(self._input.pending_audio)

    @property
    def history(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(message) for message in self._conversation.messages)

    @property
    def active_request_id(self) -> str | None:
        return self._response.active_request_id

    @property
    def active_response_id(self) -> str | None:
        return self._response.active_response_id

    @property
    def active_response_turn_id(self) -> int | None:
        return self._response.active_response_turn_id

    @property
    def last_response_id(self) -> str | None:
        return self._response.last_response_id

    @property
    def overlap_speech_ms(self) -> int:
        return self._input.overlap_speech_ms

    @property
    def assistant_text_buffer(self) -> tuple[str, ...]:
        return tuple(self._response.assistant_text_buffer)

    @property
    def assistant_audio_text_marks(self) -> tuple[DuplexAssistantAudioTextMark, ...]:
        return tuple(self._response.assistant_audio_text_marks)

    @property
    def last_assistant_full_message(self) -> dict[str, object] | None:
        message = self._conversation.last_assistant_full_message
        return dict(message) if message is not None else None

    @property
    def last_assistant_audio_text_marks(self) -> tuple[DuplexAssistantAudioTextMark, ...]:
        return tuple(self._conversation.last_assistant_audio_text_marks)

    @property
    def playback(self) -> DuplexPlaybackView:
        return self._playback.current.snapshot()

    @property
    def response_playbacks(self) -> Mapping[str, DuplexPlaybackView]:
        return MappingProxyType({key: value.snapshot() for key, value in self._playback.by_response.items()})

    @property
    def history_item_ids(self) -> Mapping[str, dict[str, object]]:
        return MappingProxyType({key: dict(value) for key, value in self._conversation.item_ids.items()})

    @property
    def pending_history_item_ids(self) -> Mapping[str, dict[str, object]]:
        return MappingProxyType({key: dict(value) for key, value in self._conversation.pending_item_ids.items()})

    @property
    def pending_history_truncations_ms(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._conversation.pending_truncations_ms))

    def replace_config(self, config: DuplexSessionConfig) -> None:
        self.config = config

    def replace_capabilities(self, capabilities: DuplexCapabilities) -> None:
        self.capabilities = capabilities

    def transition_turn(self, state: DuplexTurnState) -> None:
        self.turn_state = state

    def transition_session(self, state: DuplexSessionState) -> None:
        self.state = state

    def bind_request(self, request_id: str | None) -> None:
        self._response.active_request_id = request_id

    def clear_request(self, expected_request_id: str | None = None) -> bool:
        if expected_request_id is not None and self._response.active_request_id != expected_request_id:
            return False
        self._response.active_request_id = None
        return True

    def bind_response_turn(self, turn_id: int | None) -> None:
        self._response.active_response_turn_id = turn_id

    def active_response_accepts_model_turn(self, turn_id: int | None) -> bool:
        if self._response.active_response_id is None:
            return False
        if turn_id is None:
            return True
        active_turn_id = self._response.active_response_turn_id
        return active_turn_id is None or int(turn_id) == active_turn_id

    def append_history_message(self, message: dict[str, object]) -> None:
        self._conversation.messages.append(message)

    def stage_pending_history_item(self, item_id: str, message: dict[str, object]) -> None:
        self._conversation.pending_item_ids[item_id] = dict(message)

    def append_text(self, text: str) -> None:
        if not text:
            return
        self._input.pending_text.append(text)
        self.turn_state = DuplexTurnState.USER_SPEAKING

    @property
    def pending_input_bytes(self) -> int:
        return self._input.reserved_input_bytes

    @property
    def pending_input_turns(self) -> int:
        return self._input.pending_turns

    def reserve_input_bytes(self, size: int, *, limit: int) -> bool:
        size = max(0, int(size))
        if self._input.reserved_input_bytes + size > int(limit):
            return False
        self._input.reserved_input_bytes += size
        return True

    def release_input_bytes(self, size: int) -> None:
        self._input.reserved_input_bytes = max(0, self._input.reserved_input_bytes - max(0, int(size)))

    def release_all_input_bytes(self) -> None:
        self._input.reserved_input_bytes = 0

    def reserve_pending_turn(self, *, limit: int) -> bool:
        if self._input.pending_turns >= int(limit):
            return False
        self._input.pending_turns += 1
        return True

    def release_pending_turn(self) -> None:
        self._input.pending_turns = max(0, self._input.pending_turns - 1)

    def clear_pending_turn_reservations(self) -> None:
        self._input.pending_turns = 0

    def append_audio(self, data: str, *, fmt: str = "wav", sample_rate_hz: int | None = None) -> None:
        if not data:
            return
        self._input.pending_audio.append(DuplexAudioChunk(data=data, format=fmt, sample_rate_hz=sample_rate_hz))
        self.turn_state = DuplexTurnState.USER_SPEAKING

    def mark_user_input_activity(self) -> None:
        self.turn_state = DuplexTurnState.USER_SPEAKING

    def mark_assistant_generating(self) -> None:
        self.turn_state = DuplexTurnState.ASSISTANT_GENERATING

    def cancel_pending_input(self) -> dict[str, int]:
        cancelled = {
            "text_chunks": len(self._input.pending_text),
            "audio_chunks": len(self._input.pending_audio),
        }
        self._input.pending_text.clear()
        self._input.pending_audio.clear()
        self._input.reserved_input_bytes = 0
        self._input.pending_turns = 0
        self.turn_state = DuplexTurnState.IDLE
        return cancelled

    def commit_user_input(self) -> DuplexCommittedInput | None:
        text = "".join(self._input.pending_text).strip()
        audio_chunks = list(self._input.pending_audio)
        if not text and not audio_chunks:
            return None

        content: str | list[dict[str, object]]
        if audio_chunks:
            content_items: list[dict[str, object]] = []
            if text:
                content_items.append({"type": "text", "text": text})
            for chunk in audio_chunks:
                content_items.append(
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/{chunk.format};base64,{chunk.data}",
                        },
                    }
                )
            content = content_items
        else:
            content = text

        self._input.commit_seq += 1
        message = {"role": "user", "content": content}
        self._conversation.messages.append(message)
        self._input.pending_text.clear()
        self._input.pending_audio.clear()
        self._input.reserved_input_bytes = 0
        self.turn_state = DuplexTurnState.USER_COMMITTED
        return DuplexCommittedInput(
            message=message,
            turn_id=self.turn_id,
            epoch=self.epoch,
            input_commit_seq=self._input.commit_seq,
        )

    def commit_native_audio_input(
        self,
        *,
        transcript: str | None = None,
        turn_id: int | None = None,
    ) -> DuplexCommittedInput:
        input_audio_part: dict[str, object] = {
            "type": "audio_url",
            "audio_url": {"url": "native-duplex:input-audio"},
        }
        if transcript:
            input_audio_part["transcript"] = transcript
        self._input.commit_seq += 1
        message = {"role": "user", "content": [input_audio_part]}
        if transcript:
            message["transcript"] = transcript
        self._conversation.messages.append(message)
        self._input.pending_text.clear()
        self._input.pending_audio.clear()
        self.turn_state = DuplexTurnState.USER_COMMITTED
        return DuplexCommittedInput(
            message=message,
            turn_id=self.turn_id if turn_id is None else int(turn_id),
            epoch=self.epoch,
            input_commit_seq=self._input.commit_seq,
        )

    def complete_model_turn(self, turn_id: int) -> None:
        """Advance the model-owned output identity after its terminal signal."""
        completed_turn_id = int(turn_id)
        if completed_turn_id >= self.turn_id:
            self.turn_id = completed_turn_id + 1

    def reserve_response_options(self, options: ResponseCreateOptions) -> None:
        if self._response.active_response_id is not None:
            raise RuntimeError("response options cannot be reserved while a response is active")
        if self._response.pending_options is not None:
            raise RuntimeError("response options are already reserved")
        self._response.pending_options = options

    def discard_response_options(self) -> None:
        self._response.pending_options = None

    def _activate_response_options(self) -> None:
        options = self._response.pending_options
        self._response.active_config = copy.deepcopy(self.config)
        self._response.active_options = options
        self._response.pending_options = None
        if options is not None:
            options.apply_to(self._response.active_config)

    def _restore_response_config(self) -> None:
        self._response.active_config = None
        self._response.active_options = None
        self._response.pending_options = None

    def begin_response(self, *, turn_id: int | None = None) -> str:
        self._activate_response_options()
        response_id = f"resp-{self.session_id}-{self.epoch}-{uuid4().hex[:8]}"
        self._response.active_response_id = response_id
        self._response.active_response_turn_id = self.turn_id if turn_id is None else int(turn_id)
        self._response.last_response_id = response_id
        self._response.assistant_text_buffer.clear()
        self._response.assistant_audio_text_marks.clear()
        self._clear_response_metrics()
        self._conversation.last_assistant_full_message = None
        self._conversation.last_assistant_audio_text_marks.clear()
        self._playback.current = DuplexPlaybackCursor()
        self._playback.by_response[response_id] = self._playback.current
        self.turn_state = DuplexTurnState.ASSISTANT_GENERATING
        return response_id

    def _clear_response_metrics(self) -> None:
        self._response.stage_metrics.clear()
        self._response.stage_metric_tpot_weighted_ms.clear()
        self._response.stage_metric_tpot_weight.clear()

    def accumulate_response_stage_metrics(
        self,
        stage_metrics: Mapping[object, object] | None,
    ) -> dict[str, dict[str, object]]:
        if self.active_response_id is None or not isinstance(stage_metrics, Mapping):
            return copy.deepcopy(self._response.stage_metrics)

        additive_fields = (
            "num_tokens_in",
            "num_tokens_out",
            "stage_gen_time_ms",
            "postprocess_time_ms",
            "audio_generated_frames",
            "audio_duration_s",
            "image_pixels",
            "output_unit_count",
        )
        first_positive_fields = (
            "serving_time_to_first_output_ms",
            "vllm_ttft_ms",
        )
        interval_fields = (
            ("inter_output_latencies_ms", "inter_output_latency_ms"),
            ("vllm_itls_ms", "vllm_itl_ms"),
        )
        handled_fields = {
            *additive_fields,
            *first_positive_fields,
            "vllm_tpot_ms",
            *(name for pair in interval_fields for name in pair),
        }

        for raw_stage_id, raw_values in stage_metrics.items():
            if not isinstance(raw_values, Mapping):
                continue
            stage_id = str(raw_stage_id)
            current = self._response.stage_metrics.setdefault(stage_id, {})
            for name in additive_fields:
                value = raw_values.get(name)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    current[name] = current.get(name, 0) + value
            for name in first_positive_fields:
                value = raw_values.get(name)
                current_value = current.get(name)
                if (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and value > 0
                    and not (isinstance(current_value, int | float) and current_value > 0)
                ):
                    current[name] = value
            for list_name, mean_name in interval_fields:
                values = raw_values.get(list_name)
                if isinstance(values, list):
                    combined = list(current.get(list_name, []))
                    combined.extend(
                        value for value in values if isinstance(value, int | float) and not isinstance(value, bool)
                    )
                    current[list_name] = combined
                    current[mean_name] = sum(combined) / len(combined) if combined else 0.0

            tpot_ms = raw_values.get("vllm_tpot_ms")
            token_count = raw_values.get("num_tokens_out")
            if isinstance(tpot_ms, int | float) and tpot_ms > 0:
                weight = max(int(token_count) - 1, 1) if isinstance(token_count, int | float) else 1
                self._response.stage_metric_tpot_weighted_ms[stage_id] = (
                    self._response.stage_metric_tpot_weighted_ms.get(stage_id, 0.0) + float(tpot_ms) * weight
                )
                self._response.stage_metric_tpot_weight[stage_id] = (
                    self._response.stage_metric_tpot_weight.get(stage_id, 0) + weight
                )
                current["vllm_tpot_ms"] = self._response.stage_metric_tpot_weighted_ms[stage_id] / float(
                    self._response.stage_metric_tpot_weight[stage_id]
                )

            for name, value in raw_values.items():
                if name not in handled_fields:
                    current[str(name)] = copy.deepcopy(value)

        return copy.deepcopy(self._response.stage_metrics)

    def accumulate_overlap_speech(self, duration_ms: int) -> int:
        self._input.overlap_speech_ms += max(0, int(duration_ms))
        return self._input.overlap_speech_ms

    def reset_overlap_speech(self) -> int:
        previous = self._input.overlap_speech_ms
        self._input.overlap_speech_ms = 0
        return previous

    def append_assistant_text(self, text: str) -> None:
        if text:
            self._response.assistant_text_buffer.append(text)

    def mark_audio_sent(
        self,
        duration_ms: int | None = None,
        *,
        text_chars: int | None = None,
        audio_text_marks: list[dict[str, object]] | None = None,
    ) -> None:
        playback = self._playback.current
        if duration_ms is not None:
            playback.generated_ms = max(playback.generated_ms, duration_ms)
            playback.sent_ms = max(playback.sent_ms, duration_ms)
            if text_chars is not None and text_chars >= 0:
                self._response.assistant_audio_text_marks.append(
                    DuplexAssistantAudioTextMark(
                        text_chars=int(text_chars),
                        audio_end_ms=max(0, int(duration_ms)),
                    )
                )
        if audio_text_marks:
            for raw_mark in audio_text_marks:
                if not isinstance(raw_mark, dict):
                    continue
                raw_text_chars = raw_mark.get("text_chars")
                raw_audio_end_ms = raw_mark.get("audio_end_ms", raw_mark.get("audio_ms"))
                if not isinstance(raw_text_chars, int | float) or not isinstance(raw_audio_end_ms, int | float):
                    continue
                self._response.assistant_audio_text_marks.append(
                    DuplexAssistantAudioTextMark(
                        text_chars=max(0, int(raw_text_chars)),
                        audio_end_ms=max(0, int(raw_audio_end_ms)),
                    )
                )
        self.turn_state = DuplexTurnState.ASSISTANT_PLAYING

    def _playback_cursor_for_response(self, response_id: str | None = None) -> DuplexPlaybackCursor:
        if response_id is None:
            return self._playback.current
        playback = self._playback.by_response.get(response_id)
        if playback is None:
            # A restored or legacy session may not have response-scoped state.
            # Keep its acknowledgement isolated from the active response.
            playback = DuplexPlaybackCursor()
            self._playback.by_response[response_id] = playback
        return playback

    def playback_for_response(self, response_id: str | None = None) -> DuplexPlaybackView:
        return self._playback_cursor_for_response(response_id).snapshot()

    def acknowledge_playback(
        self,
        played_ms: int,
        committed_ms: int | None = None,
        *,
        response_id: str | None = None,
    ) -> DuplexPlaybackView:
        playback = self._playback_cursor_for_response(response_id)
        playback.acknowledge(played_ms, committed_ms)
        return playback.snapshot()

    def truncate_playback_commit(
        self,
        committed_ms: int,
        *,
        response_id: str | None = None,
    ) -> DuplexPlaybackView:
        playback = self._playback_cursor_for_response(response_id)
        playback.truncate_committed(committed_ms)
        return playback.snapshot()

    def release_response_playback(self, response_id: str | None) -> None:
        if response_id is None or response_id == self.active_response_id:
            return
        self._playback.by_response.pop(response_id, None)

    def clear_playback_cursor(self) -> None:
        self._playback.current = DuplexPlaybackCursor()
        if self.active_response_id is not None:
            self._playback.by_response[self.active_response_id] = self._playback.current

    def end_response(
        self,
        *,
        commit_text: bool = True,
        playback_commit_policy: str | None = None,
        preserve_request: bool = False,
    ) -> dict[str, object] | None:
        assistant_text = "".join(self._response.assistant_text_buffer).strip()
        message = None
        if assistant_text:
            self._conversation.last_assistant_full_message = {"role": "assistant", "content": assistant_text}
            self._conversation.last_assistant_audio_text_marks = list(self._response.assistant_audio_text_marks)
        if commit_text and assistant_text:
            committed_text = self._playback_committed_text(
                assistant_text,
                playback_commit_policy=playback_commit_policy,
            )
        else:
            committed_text = ""
        if commit_text and committed_text:
            message = {"role": "assistant", "content": committed_text}
            self._conversation.messages.append(message)
        self._response.assistant_text_buffer.clear()
        if not preserve_request:
            self._response.active_request_id = None
        self._response.active_response_id = None
        self._response.active_response_turn_id = None
        self._clear_response_metrics()
        self.turn_state = DuplexTurnState.IDLE
        self._restore_response_config()
        return message

    def register_history_item(self, item_id: str | None, message: dict[str, object] | None) -> None:
        if not item_id:
            return
        if message is None:
            last_message = self._conversation.last_assistant_full_message
            if last_message is None:
                return
            self._conversation.pending_item_ids[item_id] = dict(last_message)
            if self._conversation.last_assistant_audio_text_marks:
                self._conversation.pending_item_audio_text_marks[item_id] = list(
                    self._conversation.last_assistant_audio_text_marks
                )
            pending_audio_ms = self._conversation.pending_truncations_ms.pop(item_id, None)
            if pending_audio_ms is not None:
                self.truncate_history_item(
                    item_id,
                    audio_end_ms=pending_audio_ms,
                    playback=self._playback_cursor_for_item_id(item_id),
                )
            return
        pending_audio_ms = self._conversation.pending_truncations_ms.pop(item_id, None)
        if pending_audio_ms is not None:
            self._truncate_message_to_audio_ms(
                message,
                audio_end_ms=pending_audio_ms,
                marks=self._response.assistant_audio_text_marks or self._conversation.last_assistant_audio_text_marks,
                playback=self._playback_cursor_for_item_id(item_id),
            )
            if self._message_text_len(message) <= 0:
                try:
                    self._conversation.messages.remove(message)
                except ValueError:
                    pass
                return
        self._conversation.item_ids[item_id] = message
        self._conversation.pending_item_ids.pop(item_id, None)
        self._conversation.pending_item_audio_text_marks.pop(item_id, None)
        if message.get("role") == "assistant":
            marks = self._response.assistant_audio_text_marks or self._conversation.last_assistant_audio_text_marks
            if marks:
                self._conversation.item_audio_text_marks[item_id] = list(marks)

    def delete_history_item(self, item_id: str) -> bool:
        message = self._conversation.item_ids.pop(item_id, None)
        self._conversation.item_audio_text_marks.pop(item_id, None)
        pending = self._conversation.pending_item_ids.pop(item_id, None)
        self._conversation.pending_item_audio_text_marks.pop(item_id, None)
        self._conversation.pending_truncations_ms.pop(item_id, None)
        if message is None:
            return pending is not None
        try:
            self._conversation.messages.remove(message)
        except ValueError:
            pass
        return True

    def truncate_history_item(
        self,
        item_id: str,
        *,
        audio_end_ms: int,
        playback: DuplexPlaybackCursor | DuplexPlaybackView | None = None,
    ) -> bool:
        playback = playback or self._playback_cursor_for_item_id(item_id)
        message = self._conversation.item_ids.get(item_id)
        if message is None:
            pending = self._conversation.pending_item_ids.get(item_id)
            if pending is None:
                self._conversation.pending_truncations_ms[item_id] = max(0, int(audio_end_ms))
                return False
            message = dict(pending)
            changed = self._truncate_message_to_audio_ms(
                message,
                audio_end_ms=audio_end_ms,
                marks=self._conversation.pending_item_audio_text_marks.get(item_id),
                playback=playback,
            )
            if not changed or self._message_text_len(message) <= 0:
                if changed:
                    self._conversation.pending_item_ids.pop(item_id, None)
                    self._conversation.pending_item_audio_text_marks.pop(item_id, None)
                    self._conversation.pending_truncations_ms.pop(item_id, None)
                return changed
            self._conversation.messages.append(message)
            self._conversation.item_ids[item_id] = message
            if item_id in self._conversation.pending_item_audio_text_marks:
                self._conversation.item_audio_text_marks[item_id] = list(
                    self._conversation.pending_item_audio_text_marks[item_id]
                )
            self._conversation.pending_item_ids.pop(item_id, None)
            self._conversation.pending_item_audio_text_marks.pop(item_id, None)
            self._conversation.pending_truncations_ms.pop(item_id, None)
            return True
        changed = self._truncate_message_to_audio_ms(
            message,
            audio_end_ms=audio_end_ms,
            marks=self._conversation.item_audio_text_marks.get(item_id),
            playback=playback,
        )
        if changed and self._message_text_len(message) <= 0:
            self._conversation.item_ids.pop(item_id, None)
            self._conversation.item_audio_text_marks.pop(item_id, None)
            try:
                self._conversation.messages.remove(message)
            except ValueError:
                pass
        return changed

    def _truncate_message_to_audio_ms(
        self,
        message: dict[str, object],
        *,
        audio_end_ms: int,
        marks: list[DuplexAssistantAudioTextMark] | None = None,
        playback: DuplexPlaybackCursor | DuplexPlaybackView | None = None,
    ) -> bool:
        content = message.get("content")
        if isinstance(content, str):
            keep_chars = self._text_chars_for_audio_ms(
                audio_end_ms,
                len(content),
                marks=marks,
                playback=playback,
            )
            message["content"] = content[:keep_chars].rstrip()
            return True
        if not isinstance(content, list):
            return False
        changed = False
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"output_audio", "audio", "audio_transcript"}:
                transcript = part.get("transcript")
                if isinstance(transcript, str):
                    keep_chars = self._text_chars_for_audio_ms(
                        audio_end_ms,
                        len(transcript),
                        marks=marks,
                        playback=playback,
                    )
                    part["transcript"] = transcript[:keep_chars].rstrip()
                    changed = True
            if part_type in {"output_text", "text"}:
                text = part.get("text")
                if isinstance(text, str):
                    keep_chars = self._text_chars_for_audio_ms(
                        audio_end_ms,
                        len(text),
                        marks=marks,
                        playback=playback,
                    )
                    part["text"] = text[:keep_chars].rstrip()
                    changed = True
        if not changed:
            return False
        return True

    @staticmethod
    def _message_text_len(message: dict[str, object]) -> int:
        content = message.get("content")
        if isinstance(content, str):
            return len(content)
        if not isinstance(content, list):
            return 0
        total = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            for key in ("text", "transcript"):
                value = part.get(key)
                if isinstance(value, str):
                    total += len(value)
        return total

    def _playback_committed_text(
        self,
        assistant_text: str,
        *,
        playback_commit_policy: str | None = None,
    ) -> str:
        sent_ms = max(self.playback.sent_ms, self.playback.generated_ms)
        committed_ms = self.playback.committed_ms
        policy = playback_commit_policy or self.config.playback_commit_policy
        if sent_ms <= 0 or committed_ms >= sent_ms:
            return assistant_text
        if committed_ms <= 0:
            if policy == DuplexPlaybackCommitPolicy.COMMIT_ALL_ON_DONE.value:
                return assistant_text
            return ""
        keep_chars = self._text_chars_for_audio_ms(committed_ms, len(assistant_text))
        if keep_chars <= 0:
            return ""
        return assistant_text[:keep_chars].rstrip()

    def _text_chars_for_audio_ms(
        self,
        audio_end_ms: int,
        text_len: int,
        *,
        marks: list[DuplexAssistantAudioTextMark] | None = None,
        playback: DuplexPlaybackCursor | DuplexPlaybackView | None = None,
    ) -> int:
        if text_len <= 0:
            return 0
        audio_end_ms = max(0, int(audio_end_ms))
        marks = marks if marks is not None else self._response.assistant_audio_text_marks
        playback = playback or self._playback.current
        if not marks:
            sent_ms = max(1, playback.sent_ms, playback.generated_ms)
            return int(text_len * max(0.0, min(1.0, audio_end_ms / sent_ms)))
        marks = sorted(
            (mark for mark in marks if mark.audio_end_ms >= 0 and mark.text_chars >= 0),
            key=lambda mark: mark.audio_end_ms,
        )
        if not marks:
            return 0
        if audio_end_ms <= 0:
            return 0
        previous_ms = 0
        previous_chars = 0
        for mark in marks:
            mark_ms = max(previous_ms, mark.audio_end_ms)
            mark_chars = min(text_len, max(previous_chars, mark.text_chars))
            if audio_end_ms <= mark_ms:
                if mark_ms <= previous_ms:
                    return mark_chars
                ratio = (audio_end_ms - previous_ms) / max(1, mark_ms - previous_ms)
                return int(previous_chars + (mark_chars - previous_chars) * max(0.0, min(1.0, ratio)))
            previous_ms = mark_ms
            previous_chars = mark_chars
        final_ms = max(playback.sent_ms, playback.generated_ms, previous_ms)
        if audio_end_ms >= final_ms:
            return text_len
        ratio = (audio_end_ms - previous_ms) / max(1, final_ms - previous_ms)
        return int(previous_chars + (text_len - previous_chars) * max(0.0, min(1.0, ratio)))

    def _playback_cursor_for_item_id(self, item_id: str) -> DuplexPlaybackCursor | None:
        if not item_id.startswith("item_"):
            return None
        return self._playback.by_response.get(item_id.removeprefix("item_"))

    def barge_in(self) -> int:
        self.epoch += 1
        self._input.pending_text.clear()
        self._input.pending_audio.clear()
        self._response.assistant_text_buffer.clear()
        self._response.assistant_audio_text_marks.clear()
        self._response.active_request_id = None
        self._response.active_response_id = None
        self._response.active_response_turn_id = None
        self._clear_response_metrics()
        self._restore_response_config()
        self.turn_state = DuplexTurnState.BARGE_IN
        return self.epoch

    def mark_closing(self) -> None:
        if self.state != DuplexSessionState.CLOSED:
            self.state = DuplexSessionState.CLOSING

    def close(self) -> None:
        self.state = DuplexSessionState.CLOSED
        self.turn_state = DuplexTurnState.IDLE
        self._response.active_response_turn_id = None
        self._clear_response_metrics()
        self._restore_response_config()

    def as_public_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.session_id,
            "state": self.state.value,
            "turn_state": self.turn_state.value,
            "epoch": self.epoch,
            "turn_id": self.turn_id,
            "active_request_id": self.active_request_id,
            "active_response_id": self.active_response_id,
            "active_response_turn_id": self.active_response_turn_id,
            "model": self.config.model,
            "modalities": list(self.config.modalities),
            "instructions": self.config.instructions,
            "voice": self.config.voice,
            "response_format": self.config.response_format,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "speed": self.config.speed,
            "idle_timeout_s": self.config.idle_timeout_s,
            "overlap_policy": self.config.overlap_policy,
            "overlap_short_ack_ms": self.config.overlap_short_ack_ms,
            "overlap_barge_in_ms": self.config.overlap_barge_in_ms,
            "overlap_silence_rms": self.config.overlap_silence_rms,
            "playback_commit_policy": self.config.playback_commit_policy,
            "playback": self.playback.as_dict(),
            "capabilities": self.capabilities.as_dict(),
        }
        if isinstance(self.config.extra_body.get("realtime_tools"), list):
            payload["tools"] = self.config.extra_body["realtime_tools"]
        if isinstance(self.config.extra_body.get("realtime_tool_choice"), str | dict):
            payload["tool_choice"] = self.config.extra_body["realtime_tool_choice"]
        if isinstance(self.config.extra_body.get("realtime_metadata"), dict):
            payload["metadata"] = dict(self.config.extra_body["realtime_metadata"])
        if isinstance(self.config.extra_body.get("realtime_include"), list):
            payload["include"] = list(self.config.extra_body["realtime_include"])
        if isinstance(self.config.extra_body.get("realtime_prompt"), dict):
            payload["prompt"] = dict(self.config.extra_body["realtime_prompt"])
        if isinstance(self.config.extra_body.get("realtime_input_audio_transcription"), dict):
            payload["input_audio_transcription"] = dict(self.config.extra_body["realtime_input_audio_transcription"])
        if isinstance(self.config.extra_body.get("realtime_input_audio_noise_reduction"), dict):
            payload["input_audio_noise_reduction"] = dict(
                self.config.extra_body["realtime_input_audio_noise_reduction"]
            )
        if isinstance(self.config.extra_body.get("realtime_audio"), dict):
            payload["audio"] = dict(self.config.extra_body["realtime_audio"])
        if isinstance(self.config.extra_body.get("realtime_tracing"), str | dict):
            payload["tracing"] = self.config.extra_body["realtime_tracing"]
        raw_realtime_session = self.config.extra_body.get("realtime_session_payload")
        if isinstance(raw_realtime_session, dict):
            for key, value in raw_realtime_session.items():
                if key not in payload and key != "extra_body":
                    payload[key] = value
        return payload


class DuplexTurnController:
    """Interaction controller that accepts signals from multiple sources."""

    def signal(
        self,
        session: DuplexSession,
        event_type: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = payload or {}
        if event_type == DuplexTurnEventType.USER_STARTED.value:
            session.transition_turn(DuplexTurnState.USER_SPEAKING)
        elif event_type == DuplexTurnEventType.USER_COMMITTED.value:
            session.transition_turn(DuplexTurnState.USER_COMMITTED)
        elif event_type == DuplexTurnEventType.ASSISTANT_STARTED.value:
            session.transition_turn(DuplexTurnState.ASSISTANT_GENERATING)
        elif event_type == DuplexTurnEventType.ASSISTANT_DONE.value:
            session.transition_turn(DuplexTurnState.IDLE)
        elif event_type == DuplexTurnEventType.PLAYBACK_ACK.value:
            played_ms = int(payload.get("played_ms", 0) or 0)
            committed_ms = payload.get("committed_ms")
            session.acknowledge_playback(
                played_ms,
                int(committed_ms) if isinstance(committed_ms, int | float) else None,
            )
        elif event_type == DuplexTurnEventType.BARGE_IN.value:
            session.transition_turn(DuplexTurnState.BARGE_IN)
        elif event_type in {DuplexTurnEventType.CLOSE.value, DuplexTurnEventType.TIMEOUT.value}:
            session.transition_session(DuplexSessionState.CLOSING)
        return {
            "type": "turn.event",
            "session_id": session.session_id,
            "event": event_type,
            "turn_state": session.turn_state.value,
            "epoch": session.epoch,
        }


class DuplexSessionRegistry:
    def __init__(self, capabilities: DuplexCapabilities | None = None) -> None:
        self._capabilities = capabilities or DuplexCapabilities()
        self._sessions: dict[str, DuplexSession] = {}
        self._next_incarnation_by_session_id: dict[str, int] = {}

    def create(self, config: DuplexSessionConfig | None = None, session_id: str | None = None) -> DuplexSession:
        sid = session_id or f"duplex-{uuid4().hex}"
        if sid in self._sessions:
            raise ValueError(f"Duplex session already exists: {sid}")
        incarnation = self._next_incarnation_by_session_id.get(sid, 0)
        self._next_incarnation_by_session_id[sid] = incarnation + 1
        session = DuplexSession(
            session_id=sid,
            config=config or DuplexSessionConfig(),
            capabilities=self._capabilities,
            incarnation=incarnation,
        )
        self._sessions[sid] = session
        return session

    def active_count(self) -> int:
        return len(self._sessions)

    def get(self, session_id: str) -> DuplexSession | None:
        return self._sessions.get(session_id)

    def close(self, session_id: str) -> DuplexSession | None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()
        return session
