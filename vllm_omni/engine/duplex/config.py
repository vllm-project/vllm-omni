# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session configuration, capabilities and value types of the duplex session model.

These classes are the typed half of the public contract: ``DuplexOmni``
accepts a ``DuplexSessionConfig`` (or the equivalent OpenAI Realtime
``session`` object via :meth:`DuplexSessionConfig.from_realtime`) and reports
``DuplexCapabilities``. They used to live in ``entrypoints/duplex/protocol.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast


class DuplexConfigError(ValueError):
    """A Realtime session/response object was rejected while mapping it onto the duplex config."""

    def __init__(self, message: str, *, code: str = "bad_event", param: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.param = param


class DuplexOverlapPolicy(str, Enum):
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
    adapter_patterns: list[str] = field(default_factory=lambda: ["chunk_group_append"])
    signal_sources: list[str] = field(default_factory=lambda: ["client_event", "server_policy", "model_native"])
    stage_handoff_transport: str | None = None
    chunk_period_ms: int | None = 1000
    target_barge_in_latency_ms: int | None = 1000

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
            # Every duplex model is model-native and appends audio chunks (§7 D4);
            # kept on the wire as constants for clients that still read them.
            "implementation_level": "model_native_duplex",
            "adapter_patterns": self.adapter_patterns,
            "input_modes": ["append_audio_chunk"],
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

    def normalized(self) -> DuplexSessionConfig:
        """Apply the clamps / defaults ``from_event`` applies to a wire object (typed configs skip them)."""
        return DuplexSessionConfig.from_event({"session": self.as_dict()})

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

    @classmethod
    def from_realtime(
        cls,
        session_payload: Mapping[str, object],
        *,
        model: str | None = None,
    ) -> DuplexSessionConfig:
        """Build a config from an OpenAI Realtime ``session`` object (session.update / open_session).

        Applies the wire defaults, validates audio formats and turn detection
        (raising :class:`DuplexConfigError` with the same codes the Realtime
        translator used), fills ``turn_detection`` defaults / derives
        ``overlap_policy``, and stores the Realtime-only fields under the
        ``realtime_*`` keys of ``extra_body`` exactly like the old
        ``_session_create_from_realtime``.
        """
        from vllm_omni.engine.duplex.realtime_commands import (
            RealtimeInputDefaults,
            duplex_response_format,
            input_audio_transcription_config,
            json_safe_realtime_payload,
            realtime_max_output_tokens,
            realtime_overlap_fields,
            validate_realtime_session_audio_formats,
        )
        from vllm_omni.engine.duplex.turn_detection import (
            normalize_turn_detection_session_payload,
            validate_realtime_turn_detection,
        )

        payload: dict[str, object] = dict(session_payload)
        format_error = validate_realtime_session_audio_formats(payload)
        if format_error is not None:
            raise DuplexConfigError(format_error, code="unsupported_audio_format")
        turn_detection_error = validate_realtime_turn_detection(payload)
        if turn_detection_error is not None:
            raise DuplexConfigError(turn_detection_error, code="unsupported_turn_detection", param="turn_detection")
        normalize_turn_detection_session_payload(payload)
        defaults = RealtimeInputDefaults().with_session_payload(payload)
        payload.update(realtime_overlap_fields(payload))

        audio_config = payload.get("audio")
        audio_input = audio_config.get("input") if isinstance(audio_config, dict) else None
        audio_output = audio_config.get("output") if isinstance(audio_config, dict) else None
        extra_body_payload = payload.get("extra_body")
        extra_body: dict[str, object] = dict(extra_body_payload) if isinstance(extra_body_payload, dict) else {}
        extra_body["realtime_session_payload"] = json_safe_realtime_payload(payload)
        if isinstance(payload.get("tools"), list):
            extra_body["realtime_tools"] = payload["tools"]
        if isinstance(payload.get("tool_choice"), str | dict):
            extra_body["realtime_tool_choice"] = payload["tool_choice"]
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            extra_body["realtime_metadata"] = dict(metadata)
        include = payload.get("include")
        if isinstance(include, list):
            extra_body["realtime_include"] = list(include)
        prompt = payload.get("prompt")
        if isinstance(prompt, dict):
            extra_body["realtime_prompt"] = dict(prompt)
        transcription = input_audio_transcription_config(payload)
        if isinstance(transcription, dict):
            extra_body["realtime_input_audio_transcription"] = dict(transcription)
        noise_reduction = payload.get("input_audio_noise_reduction")
        if isinstance(noise_reduction, dict):
            extra_body["realtime_input_audio_noise_reduction"] = dict(noise_reduction)
        audio_noise_reduction = audio_input.get("noise_reduction") if isinstance(audio_input, dict) else None
        if isinstance(audio_noise_reduction, dict):
            extra_body["realtime_input_audio_noise_reduction"] = dict(audio_noise_reduction)
        if isinstance(audio_config, dict):
            extra_body["realtime_audio"] = dict(audio_config)
        if isinstance(payload.get("tracing"), str | dict):
            extra_body["realtime_tracing"] = payload["tracing"]
        if isinstance(payload.get("turn_detection"), dict):
            extra_body["realtime_turn_detection"] = dict(cast("Mapping[str, object]", payload["turn_detection"]))
        elif "turn_detection" in payload and payload.get("turn_detection") is None:
            extra_body["realtime_turn_detection"] = None
        extra_body.setdefault("realtime_output_audio_format", defaults.output_audio_format)
        voice = payload.get("voice")
        if not isinstance(voice, str) and isinstance(audio_output, dict):
            voice = audio_output.get("voice")
        speed = payload.get("speed")
        if not isinstance(speed, int | float) and isinstance(audio_output, dict):
            speed = audio_output.get("speed")
        resolved_model = payload.get("model") if isinstance(payload.get("model"), str) else model
        session: dict[str, object] = {
            "model": resolved_model,
            "modalities": payload.get("modalities") or payload.get("output_modalities") or ["text", "audio"],
            "instructions": payload.get("instructions"),
            "voice": voice,
            "ref_audio": payload.get("ref_audio"),
            "response_format": duplex_response_format(defaults.output_audio_format),
            "temperature": payload.get("temperature"),
            "max_tokens": realtime_max_output_tokens(
                payload.get("max_response_output_tokens")
                or payload.get("max_output_tokens")
                or payload.get("max_tokens")
            ),
            "speed": speed,
            "idle_timeout_s": payload.get("idle_timeout_s") or 300.0,
            **realtime_overlap_fields(payload),
            "extra_body": extra_body,
        }
        return cls.from_event({"session": session})

    def apply_realtime_update(
        self,
        patch: Mapping[str, object],
        *,
        session_id: str | None = None,
        audio_started: bool = False,
    ) -> None:
        """Apply a Realtime ``session.update`` patch in place.

        Raises :class:`DuplexConfigError` (``code`` in ``model_update_unsupported``,
        ``voice_update_after_audio_unsupported``, ``ref_audio_update_unsupported``)
        when the patch changes something a live session cannot change.
        ``audio_started`` is ``playback.generated_ms > 0 or playback.sent_ms > 0``.
        """
        from vllm_omni.engine.duplex.realtime_commands import (
            REALTIME_OUTPUT_AUDIO_FORMATS,
            duplex_response_format,
            input_audio_transcription_config,
            json_safe_realtime_payload,
            parse_realtime_audio_format,
            realtime_max_output_tokens,
        )

        # Any: the patch is read with ``isinstance(payload.get(...))`` guards and consumed
        # through ``payload[...]``; typing the values ``object`` would need a cast per read.
        payload: dict[str, Any] = dict(patch)
        model = payload.get("model")
        audio_config = payload.get("audio")
        audio_input = audio_config.get("input") if isinstance(audio_config, dict) else None
        audio_output = audio_config.get("output") if isinstance(audio_config, dict) else None
        voice = payload.get("voice")
        if not isinstance(voice, str) and isinstance(audio_output, dict):
            voice = audio_output.get("voice")
        if isinstance(model, str) and self.model is not None and model != self.model:
            raise DuplexConfigError(
                "session.update cannot change model for an open realtime duplex session",
                code="model_update_unsupported",
            )
        if isinstance(model, str) and self.model is None:
            self.model = model
        if isinstance(voice, str) and audio_started:
            raise DuplexConfigError(
                "session.update cannot change voice after audio output has started",
                code="voice_update_after_audio_unsupported",
            )
        if isinstance(payload.get("ref_audio"), str):
            raise DuplexConfigError(
                "session.update cannot change ref_audio after the session is open", code="ref_audio_update_unsupported"
            )
        if isinstance(payload.get("instructions"), str):
            self.instructions = str(payload["instructions"])
        elif "instructions" in payload and payload.get("instructions") is None:
            self.instructions = None
        if isinstance(voice, str):
            self.voice = str(voice)
        elif "voice" in payload and payload.get("voice") is None:
            self.voice = None
        response_format = payload.get("output_audio_format") or payload.get("response_format")
        if response_format is None and isinstance(audio_output, dict):
            response_format = audio_output.get("format")
        response_format, _ = parse_realtime_audio_format(response_format)
        if isinstance(response_format, str) and response_format.lower() in REALTIME_OUTPUT_AUDIO_FORMATS:
            self.response_format = duplex_response_format(response_format)
        if isinstance(payload.get("temperature"), int | float):
            self.temperature = float(payload["temperature"])
        speed = payload.get("speed")
        if not isinstance(speed, int | float) and isinstance(audio_output, dict):
            speed = audio_output.get("speed")
        if isinstance(speed, int | float):
            self.speed = float(speed)
        max_tokens = (
            payload.get("max_response_output_tokens")
            if "max_response_output_tokens" in payload
            else payload.get("max_output_tokens")
            if "max_output_tokens" in payload
            else payload.get("max_tokens")
        )
        if "max_response_output_tokens" in payload or "max_output_tokens" in payload or "max_tokens" in payload:
            self.max_tokens = realtime_max_output_tokens(max_tokens)
        if isinstance(payload.get("overlap_policy"), str):
            self.overlap_policy = self._normalize_overlap_policy(str(payload["overlap_policy"]))
        if isinstance(payload.get("overlap_short_ack_ms"), int | float):
            self.overlap_short_ack_ms = max(0, int(payload["overlap_short_ack_ms"]))
        if isinstance(payload.get("overlap_barge_in_ms"), int | float):
            self.overlap_barge_in_ms = max(0, int(payload["overlap_barge_in_ms"]))
        if isinstance(payload.get("overlap_silence_rms"), int | float):
            self.overlap_silence_rms = max(0.0, float(payload["overlap_silence_rms"]))
        if isinstance(payload.get("playback_commit_policy"), str):
            self.playback_commit_policy = self._normalize_playback_commit_policy(str(payload["playback_commit_policy"]))
        modalities = payload.get("modalities") or payload.get("output_modalities")
        if isinstance(modalities, list) and all(isinstance(item, str) for item in modalities):
            self.modalities = list(modalities)
        if isinstance(payload.get("extra_body"), dict):
            self.extra_body.update(payload["extra_body"])
            extra = payload["extra_body"]
            if isinstance(extra.get("overlap_policy"), str):
                self.overlap_policy = self._normalize_overlap_policy(str(extra["overlap_policy"]))
            if isinstance(extra.get("playback_commit_policy"), str):
                self.playback_commit_policy = self._normalize_playback_commit_policy(
                    str(extra["playback_commit_policy"])
                )
        if isinstance(payload.get("tools"), list):
            self.extra_body["realtime_tools"] = payload["tools"]
        elif "tools" in payload and payload.get("tools") is None:
            self.extra_body.pop("realtime_tools", None)
        if isinstance(payload.get("tool_choice"), str | dict):
            self.extra_body["realtime_tool_choice"] = payload["tool_choice"]
        elif "tool_choice" in payload and payload.get("tool_choice") is None:
            self.extra_body.pop("realtime_tool_choice", None)
        if isinstance(payload.get("metadata"), dict):
            self.extra_body["realtime_metadata"] = dict(payload["metadata"])
        elif "metadata" in payload and payload.get("metadata") is None:
            self.extra_body.pop("realtime_metadata", None)
        if isinstance(payload.get("include"), list):
            self.extra_body["realtime_include"] = list(payload["include"])
        elif "include" in payload and payload.get("include") is None:
            self.extra_body.pop("realtime_include", None)
        if isinstance(payload.get("prompt"), dict):
            self.extra_body["realtime_prompt"] = dict(payload["prompt"])
        elif "prompt" in payload and payload.get("prompt") is None:
            self.extra_body.pop("realtime_prompt", None)
        transcription = input_audio_transcription_config(payload)
        if isinstance(transcription, dict):
            self.extra_body["realtime_input_audio_transcription"] = dict(transcription)
        elif "input_audio_transcription" in payload and payload.get("input_audio_transcription") is None:
            self.extra_body.pop("realtime_input_audio_transcription", None)
        if isinstance(payload.get("input_audio_noise_reduction"), dict):
            self.extra_body["realtime_input_audio_noise_reduction"] = dict(payload["input_audio_noise_reduction"])
        elif "input_audio_noise_reduction" in payload and payload.get("input_audio_noise_reduction") is None:
            self.extra_body.pop("realtime_input_audio_noise_reduction", None)
        if isinstance(audio_input, dict) and isinstance(audio_input.get("noise_reduction"), dict):
            self.extra_body["realtime_input_audio_noise_reduction"] = dict(audio_input["noise_reduction"])
        elif isinstance(audio_input, dict) and audio_input.get("noise_reduction") is None:
            self.extra_body.pop("realtime_input_audio_noise_reduction", None)
        if isinstance(payload.get("audio"), dict):
            self.extra_body["realtime_audio"] = dict(payload["audio"])
        elif "audio" in payload and payload.get("audio") is None:
            self.extra_body.pop("realtime_audio", None)
        if isinstance(payload.get("tracing"), str | dict):
            self.extra_body["realtime_tracing"] = payload["tracing"]
        elif "tracing" in payload and payload.get("tracing") is None:
            self.extra_body.pop("realtime_tracing", None)
        if isinstance(payload.get("turn_detection"), dict):
            self.extra_body["realtime_turn_detection"] = dict(payload["turn_detection"])
        elif "turn_detection" in payload and payload.get("turn_detection") is None:
            self.extra_body["realtime_turn_detection"] = None
        self.extra_body["realtime_session_payload"] = json_safe_realtime_payload(payload)
        return None

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

    @classmethod
    def from_realtime(
        cls,
        response_payload: Mapping[str, object],
        *,
        private_runtime_config_keys: frozenset[str] = frozenset(),
    ) -> ResponseCreateOptions:
        """Parse an OpenAI Realtime ``response`` object into response-scoped options.

        Raises :class:`DuplexConfigError` (``code="unsupported_native_response_options"``)
        for options a model-native duplex session cannot apply per response.
        Private runtime keys in ``extra_body`` are dropped.
        """
        from vllm_omni.engine.duplex.realtime_commands import (
            REALTIME_OUTPUT_AUDIO_FORMATS,
            duplex_response_format,
            parse_realtime_audio_format,
            realtime_max_output_tokens,
        )

        payload: dict[str, object] = dict(response_payload)
        audio_config = payload.get("audio")
        audio_output = audio_config.get("output") if isinstance(audio_config, dict) else None
        nested_voice = audio_output.get("voice") if isinstance(audio_output, dict) else None
        unsupported = (
            payload.get("instructions") is not None
            or payload.get("voice") is not None
            or nested_voice is not None
            or payload.get("temperature") is not None
            or any(
                payload.get(field_name) is not None
                for field_name in ("max_response_output_tokens", "max_output_tokens", "max_tokens")
            )
            or payload.get("tools") is not None
            or payload.get("tool_choice") is not None
        )
        if unsupported:
            raise DuplexConfigError(
                "response.create options are not supported by a model-native duplex session",
                code="unsupported_native_response_options",
            )
        instructions = str(payload["instructions"]) if isinstance(payload.get("instructions"), str) else None
        voice = payload.get("voice")
        if not isinstance(voice, str) and isinstance(audio_output, dict):
            voice = audio_output.get("voice")
        voice = str(voice) if isinstance(voice, str) else None
        response_format = payload.get("output_audio_format") or payload.get("response_format")
        if response_format is None and isinstance(audio_output, dict):
            response_format = audio_output.get("format")
        response_format, _ = parse_realtime_audio_format(response_format)
        if isinstance(response_format, str) and response_format.lower() in REALTIME_OUTPUT_AUDIO_FORMATS:
            response_format = duplex_response_format(response_format)
        else:
            response_format = None
        temperature = (
            float(cast("int | float", payload["temperature"]))
            if isinstance(payload.get("temperature"), int | float)
            else None
        )
        speed = payload.get("speed")
        if not isinstance(speed, int | float) and isinstance(audio_output, dict):
            speed = audio_output.get("speed")
        speed = float(speed) if isinstance(speed, int | float) else None
        max_tokens = (
            payload.get("max_response_output_tokens")
            if "max_response_output_tokens" in payload
            else payload.get("max_output_tokens")
            if "max_output_tokens" in payload
            else payload.get("max_tokens")
        )
        if "max_response_output_tokens" in payload or "max_output_tokens" in payload or "max_tokens" in payload:
            max_tokens = realtime_max_output_tokens(max_tokens)
        else:
            max_tokens = None
        modalities = payload.get("modalities") or payload.get("output_modalities")
        if isinstance(modalities, list) and all(isinstance(item, str) for item in modalities):
            modalities = tuple(modalities)
        else:
            modalities = None
        response_extra: dict[str, object] = {}
        conversation = payload.get("conversation")
        if isinstance(conversation, str):
            response_extra["realtime_response_conversation"] = conversation
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            response_extra["realtime_response_metadata"] = dict(metadata)
        prompt = payload.get("prompt")
        if isinstance(prompt, dict):
            response_extra["realtime_response_prompt"] = dict(prompt)
        if isinstance(payload.get("tools"), list):
            response_extra["realtime_response_tools"] = payload["tools"]
        if isinstance(payload.get("tool_choice"), str | dict):
            response_extra["realtime_response_tool_choice"] = payload["tool_choice"]
        extra_body = payload.get("extra_body")
        if isinstance(extra_body, dict):
            response_extra.update(
                (key, value) for key, value in extra_body.items() if key not in private_runtime_config_keys
            )
        return cls(
            instructions=instructions,
            voice=voice,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            speed=speed,
            modalities=modalities,
            extra_body=response_extra,
        )


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


def realtime_item_to_history_message(item: object) -> dict[str, object] | None:
    """Convert a Realtime conversation item into a chat history message (None when empty)."""
    if not isinstance(item, dict):
        return None
    role = item.get("role")
    if role not in {"system", "user", "assistant"}:
        return None
    content = item.get("content")
    if isinstance(content, str):
        text = content.strip()
        return {"role": role, "content": text} if text else None
    if not isinstance(content, list):
        return None
    text_chunks: list[str] = []
    audio_chunks: list[dict[str, object]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "text", "output_text"} and isinstance(part.get("text"), str):
            text_chunks.append(str(part["text"]))
        elif part_type in {"input_audio", "audio"}:
            audio = part.get("audio") or part.get("data")
            fmt = part.get("format") if isinstance(part.get("format"), str) else "wav"
            if isinstance(audio, str) and audio:
                audio_chunks.append({"type": "audio_url", "audio_url": {"url": f"data:audio/{fmt};base64,{audio}"}})
        elif part_type in {"audio_transcript", "transcript"} and isinstance(part.get("text"), str):
            text_chunks.append(str(part["text"]))
    text = "".join(text_chunks).strip()
    if audio_chunks:
        content_items: list[dict[str, object]] = []
        if text:
            content_items.append({"type": "text", "text": text})
        content_items.extend(audio_chunks)
        return {"role": role, "content": content_items}
    if text:
        return {"role": role, "content": text}
    return None


def realtime_max_output_tokens(value: object) -> int | None:
    """Normalize Realtime max output tokens (``"inf"`` -> ``None``)."""
    from vllm_omni.engine.duplex.realtime_commands import realtime_max_output_tokens as _impl

    return _impl(value)


def input_audio_transcription_config(session_payload: Mapping[str, object]) -> dict[str, object] | None:
    """Return the ``input_audio_transcription`` object of a Realtime session payload."""
    from vllm_omni.engine.duplex.realtime_commands import input_audio_transcription_config as _impl

    return _impl(session_payload)


__all__ = [
    "DuplexAssistantAudioTextMark",
    "DuplexAudioChunk",
    "DuplexCapabilities",
    "DuplexCommittedInput",
    "DuplexConfigError",
    "DuplexOverlapPolicy",
    "DuplexPlaybackCommitPolicy",
    "DuplexPlaybackCursor",
    "DuplexPlaybackView",
    "DuplexSessionConfig",
    "DuplexSessionState",
    "DuplexTurnEventType",
    "DuplexTurnState",
    "ResponseCreateOptions",
    "input_audio_transcription_config",
    "realtime_item_to_history_message",
    "realtime_max_output_tokens",
]
