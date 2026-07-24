from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from fastapi import WebSocket
from vllm.logger import init_logger

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.entrypoints.openai.duplex_capability import (
    should_enable_duplex_endpoint,
)
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import duplex_resource_request_id
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence, DuplexSessionLifecycleMessage
from vllm_omni.experimental.fullduplex.openai.chat_fallback import (
    ChatFallbackProjectorMixin,
)
from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexCapabilities,
    DuplexCommittedInput,
    DuplexOverlapPolicy,
    DuplexPlaybackCommitPolicy,
    DuplexSession,
    DuplexSessionConfig,
    DuplexSessionRegistry,
    DuplexSessionState,
    DuplexTurnController,
    ResponseCreateOptions,
)
from vllm_omni.experimental.fullduplex.openai.realtime_session import (
    REALTIME_OUTPUT_AUDIO_FORMATS,
    NativeRealtimeSessionProtocol,
)
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    ServingRuntimeAdapter,
    ServingRuntimeConfigError,
    ServingRuntimeSessionState,
    load_serving_runtime_adapter,
    validate_serving_runtime_adapter,
)
from vllm_omni.experimental.fullduplex.openai.runtime_bridge import (
    NativeRuntimeBridgeMixin,
)
from vllm_omni.experimental.fullduplex.openai.session_attachment import (
    DuplexJournalGapError,
    DuplexSessionAttachmentRegistry,
    InvalidResumeTokenError,
)
from vllm_omni.experimental.fullduplex.openai.session_runner import (
    DuplexSessionRunnerMixin,
)
from vllm_omni.experimental.fullduplex.openai.websocket import (
    DOMAIN_TERMINAL_EVENTS,
    DuplexSessionTasks,
    DuplexWebSocketActor,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

logger = init_logger(__name__)

__all__ = ["OmniDuplexSessionHandler", "should_enable_duplex_endpoint"]

_DEFAULT_CONFIG_TIMEOUT_S = 10.0
_DEFAULT_IDLE_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class _DuplexSessionHandshake:
    session: DuplexSession
    resumed: bool = False
    attachment_generation: int | None = None


class OmniDuplexSessionHandler(
    DuplexSessionRunnerMixin,
    NativeRuntimeBridgeMixin,
    ChatFallbackProjectorMixin,
):
    """WebSocket handler for RFC-style full-duplex session control.

    This owns the serving-side session actor, ordered inbound mailbox,
    barge-in epoch, turn controller, and playback commit state. Generic sessions
    can still fall back to chat requests, while MiniCPM-o 4.5 native sessions
    route audio appends through scheduler data-plane stage requests. It
    deliberately does not claim core persistent KV lease support.
    """

    def __init__(
        self,
        *,
        chat_service: OmniOpenAIServingChat,
        config_timeout_s: float = _DEFAULT_CONFIG_TIMEOUT_S,
        idle_timeout_s: float = _DEFAULT_IDLE_TIMEOUT_S,
        duplex_session_config: DuplexSessionRuntimeConfig | None = None,
        serving_runtime_adapter: ServingRuntimeAdapter | None = None,
        serving_runtime_adapter_path: str | None = None,
    ) -> None:
        self._chat_service = chat_service
        self._config_timeout_s = config_timeout_s
        self._idle_timeout_s = idle_timeout_s
        self._duplex_session_config = duplex_session_config or DuplexSessionRuntimeConfig()
        self._registry = DuplexSessionRegistry(
            DuplexCapabilities(
                supports_model_native_turn_policy=False,
                supports_input_append=False,
                supports_replace_latest_chunk=True,
                supports_reencode_context=True,
                supports_turn_commit_only=True,
                supports_kv_lease=False,
            )
        )
        self._turn_controller = DuplexTurnController()
        adapter_path = serving_runtime_adapter_path or getattr(
            chat_service,
            "duplex_serving_adapter_path",
            None,
        )
        if serving_runtime_adapter is not None:
            self._serving_runtime_adapter = validate_serving_runtime_adapter(serving_runtime_adapter)
        elif isinstance(adapter_path, str) and adapter_path:
            self._serving_runtime_adapter = load_serving_runtime_adapter(
                adapter_path,
                self._encode_native_data_plane_audio,
            )
        else:
            raise ValueError("A duplex serving runtime adapter must be explicitly configured")
        self._session_tasks: dict[str, DuplexSessionTasks] = {}
        self._realtime_protocols: dict[str, NativeRealtimeSessionProtocol] = {}
        self._lease_generations: dict[str, int] = {}
        self._resync_required_sessions: set[str] = set()
        self._lifecycle_queue = getattr(self._chat_service.engine_client, "duplex_lifecycle_events", None)
        self._lifecycle_task: asyncio.Task[None] | None = None
        self._attachment_registry = DuplexSessionAttachmentRegistry(
            replay_ttl_s=self._duplex_session_config.resume_replay_ttl_s,
            replay_max_bytes_per_session=self._duplex_session_config.resume_replay_max_bytes_per_session,
            disconnect_grace_s=self._duplex_session_config.disconnect_grace_s,
        )

    async def handle_realtime_session(self, websocket: WebSocket) -> None:
        await self.handle_session(
            websocket,
            realtime_protocol=NativeRealtimeSessionProtocol(websocket.query_params),
        )

    def _ensure_lifecycle_listener(self) -> None:
        if not isinstance(self._lifecycle_queue, asyncio.Queue):
            return
        if self._lifecycle_task is None or self._lifecycle_task.done():
            self._lifecycle_task = asyncio.create_task(
                self._run_lifecycle_listener(),
                name="duplex-serving-lifecycle",
            )

    async def _run_lifecycle_listener(self) -> None:
        queue = self._lifecycle_queue
        if not isinstance(queue, asyncio.Queue):
            return
        try:
            while True:
                message = await queue.get()
                try:
                    if isinstance(message, DuplexSessionLifecycleMessage):
                        await self._apply_runtime_lifecycle(message)
                finally:
                    queue.task_done()
                if self._registry.active_count() == 0:
                    return
        finally:
            if self._lifecycle_task is asyncio.current_task():
                self._lifecycle_task = None

    async def _apply_runtime_lifecycle(self, message: DuplexSessionLifecycleMessage) -> None:
        session = self._registry.get(message.session_id)
        if session is None:
            return
        fence = message.fence
        if (
            fence.session_id != session.session_id
            or fence.incarnation != session.incarnation
            or fence.epoch < session.epoch
        ):
            return
        current_generation = self._lease_generations.get(session.session_id, 0)
        if message.lease_generation < current_generation:
            return
        protocol = self._realtime_protocols.get(session.session_id)
        expired_payload: dict[str, object] = {
            "type": "session.expired",
            "session_id": session.session_id,
            "incarnation": session.incarnation,
            "reason": message.reason,
        }
        if protocol is not None:
            expired_payload = protocol.encode_outbound_event(expired_payload)[0]
        with suppress(Exception):
            await self._attachment_registry.send_event(
                session.session_id,
                expired_payload,
                journal=False,
            )

        tasks = self._session_tasks.pop(session.session_id, None)
        if tasks is not None:
            await tasks.cancel_append_tasks()
            active_response_task = tasks.active_response_task
            if active_response_task is not None and not active_response_task.done():
                active_response_task.cancel()
                with suppress(asyncio.CancelledError):
                    await active_response_task
        native = self._serving_runtime_adapter.session_states.get(session.session_id)
        if native is not None and native.data_plane_task is not None:
            data_plane_task = native.data_plane_task
            native.data_plane_task = None
            data_plane_task.cancel()
            with suppress(asyncio.CancelledError):
                await data_plane_task
        if session.active_response_id is not None:
            session.end_response(commit_text=False)
        self._cleanup_duplex_session_state(session)
        self._registry.close(session.session_id)
        self._realtime_protocols.pop(session.session_id, None)
        self._lease_generations.pop(session.session_id, None)
        self._resync_required_sessions.discard(session.session_id)
        attachment = await self._attachment_registry.close(session.session_id)
        if attachment is not None:
            with suppress(Exception):
                await attachment.close("session_expired")

    def _stop_lifecycle_listener_if_idle(self) -> None:
        task = self._lifecycle_task
        if self._registry.active_count() != 0 or task is None or task.done():
            return
        if task is not asyncio.current_task():
            task.cancel()
            self._lifecycle_task = None

    @staticmethod
    def _native_audio_payload_size_bytes(payload: Mapping[str, object]) -> int:
        audio = payload.get("audio") or payload.get("data")
        if not isinstance(audio, str):
            return 0
        try:
            return len(base64.b64decode(audio, validate=True))
        except (ValueError, binascii.Error):
            return 0

    async def _apply_outbound_session_event(
        self,
        payload: dict[str, object],
        *,
        session: DuplexSession | None,
        actor: DuplexWebSocketActor,
        native: ServingRuntimeSessionState,
        realtime_protocol: NativeRealtimeSessionProtocol | None,
    ) -> tuple[bool, dict[str, object] | None]:
        """Apply domain transitions before an event is queued for transport."""
        payload_type = payload.get("type")
        is_terminal = payload_type in DOMAIN_TERMINAL_EVENTS
        if is_terminal and session is not None:
            payload_epoch = payload.get("epoch")
            if isinstance(payload_epoch, int) and payload_epoch != session.epoch:
                return False, None
            if payload_type in {"response.done", "response.listen"} and (
                actor.closing or session.state == DuplexSessionState.CLOSED
            ):
                return False, None
        elif actor._is_stale_model_output(payload):
            return False, None

        if payload_type == "session.closed":
            actor.close_reason = actor.close_reason or str(payload.get("reason") or "closed")
            if session is not None:
                session.mark_closing()

        if not is_terminal or session is None:
            return True, None

        terminal_status = payload.get("status")
        terminal_status_details = payload.get("status_details")
        if terminal_status is None and isinstance(terminal_status_details, dict):
            terminal_status = terminal_status_details.get("type")
        response_terminal = payload_type == "response.done" or (
            payload_type == "response.listen" and session.active_response_id is not None
        )
        can_promote_overlap = response_terminal and terminal_status not in {"cancelled", "failed"}
        deferred_overlap_payload: dict[str, object] | None = None
        continuous_input_crosses_terminal = (
            can_promote_overlap
            and realtime_protocol is not None
            and self._session_auto_responds(session)
            and native.input_since_commit
            and not native.deferred_response_create
        )
        if continuous_input_crosses_terminal:
            # Auto-response is a model-owned continuous stream. A response
            # terminal advances the model turn without requiring the browser
            # to close its Realtime input item. Keep any partial model unit so
            # later PCM can complete it. Serving does not create or promote a
            # new model generation here; Stage0 advances generation identity
            # only after the model's terminal token.
            session.reset_overlap_speech()
            return True, None
        realtime_input_still_open = (
            can_promote_overlap
            and realtime_protocol is not None
            and native.input_since_commit
            and not native.deferred_response_create
        )
        if realtime_input_still_open:
            # response.done only closes the assistant response. It does not
            # close the current Realtime input item, and it does not reserve
            # or advance a model generation.
            session.reset_overlap_speech()
            return True, None
        if can_promote_overlap and session.overlap_speech_ms > 0:
            has_deferred_overlap = native.audio_buffer.has_pending() or native.committed_audio_payload is not None
            should_promote_overlap = (
                session.state == DuplexSessionState.OPEN
                and self._uses_native_input_append(session)
                and has_deferred_overlap
                and session.overlap_speech_ms > session.config.overlap_short_ack_ms
            )
            if should_promote_overlap:
                flushed_reserved_bytes = native.audio_buffer.pending_byte_count
                deferred_overlap_payload = native.audio_buffer.flush(
                    chunk_period_ms=session.capabilities.chunk_period_ms or 1000
                )
                if native.committed_audio_payload is not None:
                    if deferred_overlap_payload is not None:
                        deferred_overlap_payload = self._merge_native_audio_payloads(
                            native.committed_audio_payload,
                            deferred_overlap_payload,
                        )
                    else:
                        deferred_overlap_payload = native.committed_audio_payload
                if self._session_auto_responds(session) and deferred_overlap_payload is not None:
                    deferred_overlap_payload = dict(deferred_overlap_payload)
                    deferred_overlap_payload["force_listen"] = False
                if deferred_overlap_payload is not None:
                    native.retain_committed_audio(
                        deferred_overlap_payload,
                        operation_id=native.committed_audio_operation_id,
                        reserved_bytes=flushed_reserved_bytes,
                    )
                native.input_since_commit = deferred_overlap_payload is not None
                if realtime_protocol is not None:
                    native.deferred_response_create = True
                    native.deferred_precreate_response = False
                    deferred_overlap_payload = None
            else:
                had_pending_overlap_audio = native.audio_buffer.has_pending()
                native.audio_buffer.clear()
                native.input_since_commit = False
                native.speech_since_commit = False
                if had_pending_overlap_audio and realtime_protocol is not None:
                    await realtime_protocol.discard_pending_input_audio(audio_end_ms=session.overlap_speech_ms)
                if payload_type in {"audio.cancelled", "input.cancelled", "session.closed"}:
                    session.release_input_bytes(native.clear_committed_audio())

        session.reset_overlap_speech()
        if can_promote_overlap and native.deferred_response_create and native.committed_audio_payload is not None:
            deferred_overlap_payload = native.committed_audio_payload
            native.deferred_response_create = False
            native.input_since_commit = False
            native.speech_since_commit = False
        return True, deferred_overlap_payload

    @staticmethod
    def _advance_barge_in_epoch(session: DuplexSession) -> tuple[int, dict[str, int]]:
        old_playback = session.playback.as_dict()
        new_epoch = session.barge_in()
        session.clear_playback_cursor()
        return new_epoch, old_playback

    @staticmethod
    def _commit_played_response_history(
        session: DuplexSession,
        response_id: str | None,
        committed_ms: int,
    ) -> None:
        if not response_id or committed_ms < 0:
            return
        session.truncate_history_item(
            f"item_{response_id}",
            audio_end_ms=committed_ms,
            playback=session.playback_for_response(response_id),
        )

    @staticmethod
    def _should_commit_response_to_history(session: DuplexSession, response_id: str | None) -> bool:
        if response_id is not None and response_id != session.active_response_id:
            return True
        mode = session.response_config.extra_body.get("realtime_response_conversation")
        return not isinstance(mode, str) or mode.strip().lower() != "none"

    def _response_created_payload(
        self,
        session: DuplexSession,
        response_id: str,
        *,
        epoch: int,
        request_id: str | None = None,
    ) -> dict[str, object]:
        response_config = session.response_config
        payload: dict[str, object] = {
            "type": "response.created",
            "session_id": session.session_id,
            "response_id": response_id,
            "epoch": epoch,
            "modalities": list(response_config.modalities),
        }
        if request_id is not None:
            payload["request_id"] = request_id
        metadata = response_config.extra_body.get("realtime_response_metadata")
        if not isinstance(metadata, dict):
            metadata = response_config.extra_body.get("realtime_metadata")
        if isinstance(metadata, dict):
            payload["metadata"] = dict(metadata)
        conversation = response_config.extra_body.get("realtime_response_conversation")
        if isinstance(conversation, str):
            payload["conversation"] = conversation
        prompt = response_config.extra_body.get("realtime_response_prompt")
        if isinstance(prompt, dict):
            payload["prompt"] = dict(prompt)
        return payload

    def _overlap_decision(
        self,
        session: DuplexSession,
        event: dict[str, object],
        payload: dict[str, object],
    ) -> dict[str, object]:
        """Classify input that arrives while assistant audio is active.

        This is a serving-side policy. The model still owns listen/speak
        decisions for normal chunks; overlap policy only decides whether the
        current assistant response should be interrupted before the new audio is
        appended.
        """
        duration_ms = self._input_audio_duration_ms(event, payload)
        is_speech = self._input_looks_like_speech(event, payload, session=session)
        if not session.capabilities.supports_barge_in and self._event_requests_barge_in(event):
            return self._defer_unsupported_barge_in(session, duration_ms=duration_ms, is_speech=is_speech)
        explicit = event.get("overlap_action") or event.get("overlap")
        if isinstance(explicit, str):
            normalized = explicit.strip().lower()
            if normalized in {"barge_in", "interrupt", "cancel"}:
                return {
                    "action": "barge_in",
                    "reason": "client_overlap_action",
                    "duration_ms": duration_ms,
                    "buffer_audio": True,
                }
            if normalized in {"listen", "continue", "continue_output", "ack"}:
                session.reset_overlap_speech()
                return {
                    "action": "listen",
                    "reason": "client_overlap_action",
                    "duration_ms": duration_ms,
                    "buffer_audio": (
                        normalized == "listen" and is_speech and duration_ms > session.config.overlap_short_ack_ms
                    ),
                    "defer_runtime_append": True,
                }
            if normalized in {"drop", "ignore", "silence"}:
                session.reset_overlap_speech()
                return {
                    "action": "drop",
                    "reason": "client_overlap_action",
                    "duration_ms": duration_ms,
                    "buffer_audio": False,
                }

        if bool(event.get("force_barge_in", False)):
            return {
                "action": "barge_in",
                "reason": "client_force_barge_in",
                "duration_ms": duration_ms,
                "buffer_audio": True,
            }
        if self._session_auto_responds(session):
            # Full-duplex input remains model-owned while output is active. Feed
            # complete model units into the existing Stage0 stream immediately;
            # playback only controls history ACKs, not model admission.
            if is_speech:
                session.accumulate_overlap_speech(duration_ms)
            return {
                "action": "listen",
                "reason": "auto_response_continuous",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
                "defer_runtime_append": False,
                "force_listen": False,
                "preserve_realtime_input": True,
            }
        if bool(event.get("force_listen", False)):
            session.reset_overlap_speech()
            return {
                "action": "listen",
                "reason": "client_force_listen",
                "duration_ms": duration_ms,
                "buffer_audio": is_speech,
                "defer_runtime_append": True,
            }

        policy = session.config.overlap_policy
        if not is_speech:
            if session.overlap_speech_ms <= 0:
                session.reset_overlap_speech()
            return {
                "action": "drop",
                "reason": "silence_or_noise",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": False,
            }

        if self._is_short_ack_transcript_hint(event, payload):
            session.reset_overlap_speech()
            return {
                "action": "listen",
                "reason": "short_ack_transcript",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": False,
                "defer_runtime_append": True,
            }

        if policy == DuplexOverlapPolicy.LISTEN_ONLY.value:
            session.accumulate_overlap_speech(duration_ms)
            return {
                "action": "listen",
                "reason": "policy_listen_only",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
                "defer_runtime_append": True,
            }

        if policy == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value and not session.capabilities.supports_barge_in:
            return self._defer_unsupported_barge_in(session, duration_ms=duration_ms, is_speech=True)

        session.accumulate_overlap_speech(duration_ms)
        if policy == DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value:
            return {
                "action": "barge_in",
                "reason": "policy_barge_in_on_speech",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
            }

        if (
            duration_ms <= session.config.overlap_short_ack_ms
            and session.overlap_speech_ms <= session.config.overlap_short_ack_ms
        ):
            return {
                "action": "listen",
                "reason": "short_ack",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
                "defer_runtime_append": True,
            }
        if session.overlap_speech_ms >= session.config.overlap_barge_in_ms:
            if not session.capabilities.supports_barge_in:
                return {
                    "action": "listen",
                    "reason": "barge_in_unsupported",
                    "duration_ms": duration_ms,
                    "overlap_speech_ms": session.overlap_speech_ms,
                    "buffer_audio": True,
                    "defer_runtime_append": True,
                }
            return {
                "action": "barge_in",
                "reason": "long_overlap_speech",
                "duration_ms": duration_ms,
                "overlap_speech_ms": session.overlap_speech_ms,
                "buffer_audio": True,
            }
        return {
            "action": "listen",
            "reason": "accumulating_overlap_speech",
            "duration_ms": duration_ms,
            "overlap_speech_ms": session.overlap_speech_ms,
            "buffer_audio": True,
            "defer_runtime_append": True,
        }

    @staticmethod
    def _event_requests_barge_in(event: Mapping[str, object]) -> bool:
        if event.get("force_barge_in") is True:
            return True
        explicit = event.get("overlap_action") or event.get("overlap")
        return isinstance(explicit, str) and explicit.strip().lower() in {
            "barge_in",
            "interrupt",
            "cancel",
        }

    @staticmethod
    def _defer_unsupported_barge_in(
        session: DuplexSession,
        *,
        duration_ms: int,
        is_speech: bool,
    ) -> dict[str, object]:
        if is_speech:
            session.accumulate_overlap_speech(duration_ms)
        return {
            "action": "listen",
            "reason": "barge_in_unsupported",
            "duration_ms": duration_ms,
            "overlap_speech_ms": session.overlap_speech_ms,
            "buffer_audio": is_speech,
            "defer_runtime_append": True,
        }

    @staticmethod
    def _barge_in_unsupported_error(session: DuplexSession) -> dict[str, object]:
        return {
            "type": "error",
            "session_id": session.session_id,
            "code": "barge_in_unsupported",
            "error": "Barge-in is not supported by this duplex model",
        }

    @staticmethod
    def _is_short_ack_transcript_hint(event: dict[str, object], payload: dict[str, object]) -> bool:
        raw_text = event.get("transcript") or event.get("text") or payload.get("transcript") or payload.get("text")
        if not isinstance(raw_text, str):
            return False
        normalized = raw_text.strip().lower()
        if not normalized:
            return False
        compact = "".join(ch for ch in normalized if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        if compact in {
            "嗯",
            "嗯嗯",
            "对",
            "对的",
            "好",
            "好的",
            "继续",
            "继续说",
            "可以",
            "是的",
            "yes",
            "yeah",
            "yep",
            "ok",
            "okay",
            "continue",
            "goon",
            "right",
        }:
            return True
        return normalized in {"go on", "keep going", "please continue"}

    @staticmethod
    def _input_audio_duration_ms(event: dict[str, object], payload: dict[str, object]) -> int:
        for key in ("duration_ms", "audio_duration_ms"):
            value = event.get(key)
            if isinstance(value, int | float):
                return max(0, int(value))
        fmt = payload.get("format")
        sample_rate_hz = payload.get("sample_rate_hz")
        audio = payload.get("audio")
        if fmt == "pcm_f32le" and isinstance(sample_rate_hz, int) and sample_rate_hz > 0 and isinstance(audio, str):
            try:
                raw = base64.b64decode(audio, validate=True)
            except (binascii.Error, ValueError):
                return 0
            return int((len(raw) // 4) * 1000 / sample_rate_hz)
        return 0

    @staticmethod
    def _merge_native_audio_payloads(
        first: dict[str, object],
        second: dict[str, object],
    ) -> dict[str, object]:
        if first.get("format") != "pcm_f32le" or second.get("format") != "pcm_f32le":
            return second
        first_rate = first.get("sample_rate_hz")
        second_rate = second.get("sample_rate_hz")
        if not isinstance(first_rate, int) or not isinstance(second_rate, int) or first_rate != second_rate:
            return second
        first_audio = first.get("audio")
        second_audio = second.get("audio")
        if not isinstance(first_audio, str) or not isinstance(second_audio, str):
            return second
        try:
            first_raw = base64.b64decode(first_audio, validate=True)
            second_raw = base64.b64decode(second_audio, validate=True)
        except (binascii.Error, ValueError):
            return second
        merged = dict(second)
        merged["audio"] = base64.b64encode(first_raw + second_raw).decode("ascii")
        merged["sample_rate_hz"] = first_rate
        merged_frames = [
            frame
            for source in (first.get("video_frames"), second.get("video_frames"))
            if isinstance(source, list)
            for frame in source
            if isinstance(frame, str) and frame
        ]
        if merged_frames:
            merged["video_frames"] = merged_frames
        else:
            merged.pop("video_frames", None)
        merged["force_listen"] = bool(first.get("force_listen", False)) or bool(second.get("force_listen", False))
        merged.pop("force_speak", None)
        merged["is_speech"] = bool(first.get("is_speech", False)) or bool(second.get("is_speech", False))
        return merged

    @classmethod
    def _should_force_listen_for_short_commit(
        cls,
        session: DuplexSession,
        event: dict[str, object],
        payload: dict[str, object],
    ) -> bool:
        """Keep very short committed Realtime chunks in listen mode.

        Realtime VAD can emit a commit for a short pause even when the user has
        not actually yielded the turn. For MiniCPM-o native duplex, make that
        policy explicit by steering the scheduler path to the model listen
        token instead of letting a sub-second chunk start a response.
        """
        if event.get("force_listen") is True or payload.get("force_listen") is True:
            return True
        if event.get("force_barge_in") is True:
            return False
        if event.get("response_create") is not True:
            return False
        duration_ms = cls._input_audio_duration_ms(event, payload)
        return 0 < duration_ms <= session.config.overlap_short_ack_ms

    def _runtime_session_state(self, session: DuplexSession) -> ServingRuntimeSessionState:
        return self._serving_runtime_adapter.session_state(session.session_id)

    # Temporary compatibility accessors for downstream tests and extensions.
    def _minicpmo_session_state(self, session: DuplexSession) -> ServingRuntimeSessionState:
        return self._runtime_session_state(session)

    @property
    def _minicpmo_sessions(self):
        return self._serving_runtime_adapter.session_states

    @property
    def _minicpmo_data_plane(self):
        return self._serving_runtime_adapter.data_plane

    def _should_force_listen_for_auto_response_overlap(
        self,
        session: DuplexSession,
        event: dict[str, object],
        payload: dict[str, object],
    ) -> bool:
        if not self._session_auto_responds(session):
            return False
        if event.get("force_barge_in") is True:
            return False
        return event.get("force_listen") is True or payload.get("force_listen") is True

    @staticmethod
    def _assistant_playback_active(session: DuplexSession) -> bool:
        return (
            session.config.playback_commit_policy == DuplexPlaybackCommitPolicy.ACK_ONLY.value
            and session.playback.sent_ms > session.playback.committed_ms
        )

    @staticmethod
    def _input_looks_like_speech(
        event: dict[str, object],
        payload: dict[str, object],
        *,
        session: DuplexSession,
    ) -> bool:
        for key in ("is_speech", "speech"):
            value = event.get(key)
            if isinstance(value, bool):
                return value
        vad = event.get("vad")
        if isinstance(vad, dict):
            value = vad.get("is_speech")
            if isinstance(value, bool):
                return value
            probability = vad.get("speech_probability", vad.get("probability"))
            if isinstance(probability, int | float):
                return float(probability) >= 0.5
        probability = event.get("speech_probability")
        if isinstance(probability, int | float):
            return float(probability) >= 0.5

        fmt = payload.get("format")
        audio = payload.get("audio")
        if fmt in {"pcm_f32le", "pcm16"} and isinstance(audio, str):
            try:
                raw = base64.b64decode(audio, validate=True)
            except (binascii.Error, ValueError):
                return True
            if fmt == "pcm_f32le":
                if len(raw) < 4 or len(raw) % 4 != 0:
                    return True
                samples = np.frombuffer(raw, dtype=np.float32)
            else:
                if len(raw) < 2 or len(raw) % 2 != 0:
                    return True
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size == 0:
                return False
            rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
            return rms >= session.config.overlap_silence_rms
        return True

    async def _emit_overlap_decision(
        self,
        send_json,
        session: DuplexSession,
        decision: dict[str, object],
    ) -> None:
        await send_json(
            {
                "type": "overlap.decision",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "policy": session.config.overlap_policy,
                **decision,
            }
        )

    async def _open_session(
        self,
        websocket: WebSocket,
        send_json,
        *,
        realtime_protocol: NativeRealtimeSessionProtocol | None = None,
        attachment_send=None,
        attachment_close=None,
    ) -> _DuplexSessionHandshake | None:
        raw = await self._receive_text(
            websocket,
            self._config_timeout_s,
            realtime_protocol=realtime_protocol,
        )
        if raw is None:
            await send_json({"type": "error", "error": "Timeout waiting for session.create", "code": "config_timeout"})
            return None
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            await send_json({"type": "error", "error": "Invalid JSON in session.create", "code": "invalid_json"})
            return None
        if isinstance(event, dict) and event.get("type") == "session.resume":
            if realtime_protocol is None:
                await send_json(
                    {
                        "type": "error",
                        "error": "session.resume requires the Realtime protocol",
                        "code": "unsupported_session_resume",
                    }
                )
                return None
            return await self._resume_session_handshake(
                event,
                send_json=send_json,
                realtime_protocol=realtime_protocol,
                attachment_send=attachment_send,
                attachment_close=attachment_close,
            )
        if not isinstance(event, dict) or event.get("type") not in {"session.create", "open_session", "session.config"}:
            await send_json(
                {
                    "type": "error",
                    "error": f"Expected session.create, got: {event.get('type') if isinstance(event, dict) else None}",
                    "code": "bad_event",
                }
            )
            return None

        config = DuplexSessionConfig.from_event(event)
        if config.idle_timeout_s == _DEFAULT_IDLE_TIMEOUT_S:
            config.idle_timeout_s = self._idle_timeout_s
        use_native_runtime = self._uses_serving_runtime_adapter(config)
        runtime_config: dict[str, object] = {}
        if use_native_runtime:
            try:
                runtime_config = await self._serving_runtime_adapter.prepare_runtime_config(
                    config,
                    model_config=getattr(self._chat_service, "model_config", None),
                )
            except ServingRuntimeConfigError as exc:
                await send_json({"type": "error", "error": str(exc), "code": exc.code})
                return None
            except ValueError as exc:
                await send_json({"type": "error", "error": str(exc), "code": "unsupported_ref_audio_path"})
                return None
        session_id = event.get("session_id") if isinstance(event.get("session_id"), str) else None
        session = self._registry.create(config=config, session_id=session_id)
        if use_native_runtime:
            session.replace_capabilities(
                self._serving_runtime_adapter.capabilities(
                    max_sessions=self._duplex_session_config.max_sessions,
                )
            )
            session.replace_runtime_config(runtime_config)
        return _DuplexSessionHandshake(session=session)

    async def _resume_session_handshake(
        self,
        event: dict[str, object],
        *,
        send_json,
        realtime_protocol: NativeRealtimeSessionProtocol,
        attachment_send,
        attachment_close,
    ) -> _DuplexSessionHandshake | None:
        session_id = event.get("session_id")
        incarnation = event.get("incarnation")
        resume_token = event.get("resume_token")
        last_received = event.get("last_received_server_event_seq", 0)
        if (
            not isinstance(session_id, str)
            or not session_id
            or not isinstance(incarnation, int)
            or incarnation < 0
            or not isinstance(resume_token, str)
            or not resume_token
            or not isinstance(last_received, int)
            or last_received < 0
        ):
            await send_json(
                {
                    "type": "error",
                    "error": (
                        "session.resume requires session_id, incarnation, resume_token, "
                        "and a non-negative event sequence"
                    ),
                    "code": "invalid_session_resume",
                }
            )
            return None
        session = self._registry.get(session_id)
        if session is None or session.state != DuplexSessionState.OPEN:
            await send_json(
                {
                    "type": "error",
                    "error": f"Unknown or expired duplex session: {session_id}",
                    "code": "session_resume_expired",
                }
            )
            return None
        if not session.capabilities.supports_session_resume:
            await send_json(
                {
                    "type": "error",
                    "error": f"Session does not support resume: {session_id}",
                    "code": "unsupported_session_resume",
                }
            )
            return None
        try:
            await self._attachment_registry.authenticate_resume(
                session_id,
                incarnation=incarnation,
                resume_token=resume_token,
                last_received_server_event_seq=last_received,
            )
        except InvalidResumeTokenError:
            await send_json(
                {
                    "type": "error",
                    "error": "Invalid duplex session resume token",
                    "code": "invalid_resume_token",
                }
            )
            return None
        except DuplexJournalGapError:
            await send_json(
                {
                    "type": "session.resync_required",
                    "session_id": session_id,
                    "reason": "journal_gap",
                }
            )
            return None
        except (KeyError, ValueError) as exc:
            await send_json(
                {
                    "type": "error",
                    "error": str(exc),
                    "code": "session_resume_conflict",
                }
            )
            return None

        resume_runtime = getattr(self._chat_service.engine_client, "resume_duplex_session_async", None)
        if not callable(resume_runtime):
            await send_json(
                {
                    "type": "error",
                    "error": "Duplex runtime does not expose session resume control",
                    "code": "runtime_resume_unsupported",
                }
            )
            return None
        expected_generation = self._lease_generations.get(session_id, 0)
        try:
            runtime_result = await resume_runtime(
                session_id,
                fence=DuplexFence(
                    session.session_id,
                    epoch=session.epoch,
                    turn_id=session.turn_id,
                    incarnation=session.incarnation,
                ),
                expected_lease_generation=expected_generation,
            )
            lease_generation = self._runtime_lease_generation(runtime_result)
            if lease_generation is None:
                raise RuntimeError("runtime resume result omitted lease_generation")
        except Exception as exc:
            await send_json(
                {
                    "type": "error",
                    "error": str(exc),
                    "code": "runtime_resume_failed",
                }
            )
            return None

        if not callable(attachment_send) or not callable(attachment_close):
            raise RuntimeError("session.resume requires transport attachment callbacks")

        def activation_payload_factory(token, generation: int) -> dict[str, object]:
            internal = {
                "type": "session.resumed",
                "session_id": session_id,
                "incarnation": incarnation,
                "attachment_generation": generation,
                "resume_token": token.plaintext,
            }
            return realtime_protocol.encode_outbound_event(internal)[0]

        try:
            resumed = await self._attachment_registry.resume(
                session_id,
                incarnation=incarnation,
                resume_token=resume_token,
                last_received_server_event_seq=last_received,
                send=attachment_send,
                close=attachment_close,
                activation_payload_factory=activation_payload_factory,
            )
        except Exception as exc:
            # The engine-side CAS already advanced even if the transport
            # vanished before it received the rotated token. Keep that
            # generation so the registry's one-shot recovery token can retry.
            self._lease_generations[session_id] = lease_generation
            await send_json(
                {
                    "type": "error",
                    "error": str(exc),
                    "code": "session_resume_conflict",
                }
            )
            return None
        self._lease_generations[session_id] = lease_generation
        replaced = resumed.replaced_attachment
        if replaced is not None:
            replaced_payload = realtime_protocol.encode_outbound_event(
                {
                    "type": "session.replaced",
                    "session_id": session_id,
                    "attachment_generation": replaced.generation,
                }
            )[0]
            with suppress(Exception):
                await replaced.send(replaced_payload)
            with suppress(Exception):
                await replaced.close("session_replaced")
        return _DuplexSessionHandshake(
            session=session,
            resumed=True,
            attachment_generation=resumed.attachment_generation,
        )

    @classmethod
    def _runtime_lease_generation(cls, result: object) -> int | None:
        if isinstance(result, dict):
            generation = result.get("lease_generation")
            if isinstance(generation, int):
                return generation
            for key in ("stage_results", "result"):
                generation = cls._runtime_lease_generation(result.get(key))
                if generation is not None:
                    return generation
            return None
        if isinstance(result, list | tuple):
            for item in result:
                generation = cls._runtime_lease_generation(item)
                if generation is not None:
                    return generation
        return None

    def _uses_serving_runtime_adapter(self, config: DuplexSessionConfig) -> bool:
        return self._serving_runtime_adapter.is_enabled(config)

    def _runtime_session_update_error(
        self,
        session: DuplexSession,
        payload: dict[str, object],
    ) -> dict[str, object] | None:
        if not self._uses_native_input_append(session):
            return None
        try:
            self._serving_runtime_adapter.validate_client_extra_body(payload.get("extra_body"))
        except ServingRuntimeConfigError as exc:
            return {
                "type": "error",
                "session_id": session.session_id,
                "code": exc.code,
                "error": str(exc),
            }
        return None

    def _runtime_config_for_session_update(
        self,
        session: DuplexSession,
        candidate_config: DuplexSessionConfig,
    ) -> dict[str, object]:
        if not self._uses_native_input_append(session):
            return dict(session.runtime_config)
        return self._serving_runtime_adapter.runtime_config_for_update(
            candidate_config,
            dict(session.runtime_config),
        )

    def _runtime_session_candidate_update_error(
        self,
        session: DuplexSession,
        candidate_config: DuplexSessionConfig,
    ) -> dict[str, object] | None:
        if not self._uses_native_input_append(session):
            return None
        if not self._config_requests_audio_output(candidate_config):
            return None
        if "ref_audio_data" in session.runtime_config:
            return None
        return {
            "type": "error",
            "session_id": session.session_id,
            "code": "ref_audio_required",
            "error": "MiniCPM-o native duplex audio output requires ref_audio",
        }

    @staticmethod
    def _uses_native_input_append(session: DuplexSession) -> bool:
        return (
            session.capabilities.implementation_level == "model_native_duplex"
            and session.capabilities.supports_input_append
        )

    @staticmethod
    def _config_requests_audio_output(config: DuplexSessionConfig) -> bool:
        return any(str(modality).lower() == "audio" for modality in config.modalities)

    @staticmethod
    def _native_stage0_request_id(session: DuplexSession, epoch: int) -> str:
        return duplex_resource_request_id(
            DuplexFence(
                session.session_id,
                epoch=epoch,
                incarnation=session.incarnation,
            ),
            "stage0",
        )

    @staticmethod
    def _session_auto_responds(session: DuplexSession) -> bool:
        """Full-duplex / model-driven mode.

        When set, the server runs per-chunk speak-generation continuously (like
        the official MiniCPM-o ``duplex_generate`` loop) instead of waiting for an
        explicit ``response.create``: each ~chunk_period of appended audio is
        emitted and fed to the stage0 stream so the model itself decides to speak
        or listen. Signaled by the client via ``extra_body.auto_response`` (or
        ``extra_body.full_duplex``).
        """
        extra = getattr(session.config, "extra_body", None)
        if not isinstance(extra, dict):
            return False
        return extra.get("auto_response") is True or extra.get("full_duplex") is True

    async def _receive_text(
        self,
        websocket: WebSocket,
        timeout_s: float,
        *,
        realtime_protocol: NativeRealtimeSessionProtocol | None = None,
    ) -> str | None:
        try:
            if realtime_protocol is not None:
                return await asyncio.wait_for(
                    realtime_protocol.receive_internal_event_text(websocket),
                    timeout=max(0.1, timeout_s),
                )
            return await asyncio.wait_for(websocket.receive_text(), timeout=max(0.1, timeout_s))
        except asyncio.TimeoutError:
            return None

    @staticmethod
    def _input_committed_payload(
        session: DuplexSession,
        committed: DuplexCommittedInput,
        *,
        realtime_item_id: object | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "input.committed",
            "session_id": session.session_id,
            "turn_id": committed.turn_id,
            "epoch": committed.epoch,
            "history_len": len(session.history),
            "message": committed.message,
        }
        if isinstance(realtime_item_id, str) and realtime_item_id:
            payload["realtime_item_id"] = realtime_item_id
        return payload

    @staticmethod
    def _commit_native_audio_input(
        session: DuplexSession,
        *,
        realtime_item_id: object | None = None,
        transcript: object | None = None,
        turn_id: int | None = None,
    ) -> DuplexCommittedInput:
        clean_transcript = transcript.strip() if isinstance(transcript, str) else None
        committed = session.commit_native_audio_input(
            transcript=clean_transcript or None,
            turn_id=turn_id,
        )
        if isinstance(realtime_item_id, str) and realtime_item_id:
            session.register_history_item(realtime_item_id, committed.message)
        return committed

    @staticmethod
    def _native_audio_committed_payload(
        session: DuplexSession,
        *,
        committed: DuplexCommittedInput | None = None,
        realtime_item_id: object | None = None,
        transcript: object | None = None,
    ) -> dict[str, object]:
        message = committed.message if committed is not None else None
        if not isinstance(message, dict):
            input_audio_part: dict[str, object] = {
                "type": "audio_url",
                "audio_url": {"url": "native-duplex:input-audio"},
            }
            if isinstance(transcript, str) and transcript:
                input_audio_part["transcript"] = transcript
            message = {
                "role": "user",
                "content": [input_audio_part],
            }
        payload: dict[str, object] = {
            "type": "input.committed",
            "session_id": session.session_id,
            "turn_id": committed.turn_id if committed is not None else session.turn_id,
            "epoch": committed.epoch if committed is not None else session.epoch,
            "history_len": len(session.history),
            "native_audio": True,
            "message": message,
        }
        if isinstance(transcript, str) and transcript:
            payload["transcript"] = transcript
        if isinstance(realtime_item_id, str) and realtime_item_id:
            payload["realtime_item_id"] = realtime_item_id
        return payload

    @staticmethod
    def _apply_session_update(session: DuplexSession, payload: dict[str, object]) -> dict[str, object] | None:
        model = payload.get("model")
        audio_config = payload.get("audio")
        audio_input = audio_config.get("input") if isinstance(audio_config, dict) else None
        audio_output = audio_config.get("output") if isinstance(audio_config, dict) else None
        voice = payload.get("voice")
        if not isinstance(voice, str) and isinstance(audio_output, dict):
            voice = audio_output.get("voice")
        if isinstance(model, str) and session.config.model is not None and model != session.config.model:
            return {
                "type": "error",
                "session_id": session.session_id,
                "code": "model_update_unsupported",
                "error": "session.update cannot change model for an open realtime duplex session",
            }
        if isinstance(model, str) and session.config.model is None:
            session.config.model = model
        if isinstance(voice, str) and (session.playback.generated_ms > 0 or session.playback.sent_ms > 0):
            return {
                "type": "error",
                "session_id": session.session_id,
                "code": "voice_update_after_audio_unsupported",
                "error": "session.update cannot change voice after audio output has started",
            }
        if isinstance(payload.get("ref_audio"), str):
            return {
                "type": "error",
                "session_id": session.session_id,
                "code": "ref_audio_update_unsupported",
                "error": "session.update cannot change ref_audio after the native duplex runtime is open",
            }
        if isinstance(payload.get("instructions"), str):
            session.config.instructions = str(payload["instructions"])
        elif "instructions" in payload and payload.get("instructions") is None:
            session.config.instructions = None
        if isinstance(voice, str):
            session.config.voice = str(voice)
        elif "voice" in payload and payload.get("voice") is None:
            session.config.voice = None
        response_format = payload.get("output_audio_format") or payload.get("response_format")
        if response_format is None and isinstance(audio_config, dict):
            if isinstance(audio_output, dict):
                response_format = audio_output.get("format")
        response_format, _ = NativeRealtimeSessionProtocol._parse_realtime_audio_format(response_format)
        if isinstance(response_format, str) and response_format.lower() in REALTIME_OUTPUT_AUDIO_FORMATS:
            session.config.response_format = NativeRealtimeSessionProtocol._duplex_response_format(response_format)
        if isinstance(payload.get("temperature"), int | float):
            session.config.temperature = float(payload["temperature"])
        speed = payload.get("speed")
        if not isinstance(speed, int | float) and isinstance(audio_output, dict):
            speed = audio_output.get("speed")
        if isinstance(speed, int | float):
            session.config.speed = float(speed)
        max_tokens = (
            payload.get("max_response_output_tokens")
            if "max_response_output_tokens" in payload
            else payload.get("max_output_tokens")
            if "max_output_tokens" in payload
            else payload.get("max_tokens")
        )
        if "max_response_output_tokens" in payload or "max_output_tokens" in payload or "max_tokens" in payload:
            session.config.max_tokens = NativeRealtimeSessionProtocol.realtime_max_output_tokens(max_tokens)
        if isinstance(payload.get("overlap_policy"), str):
            session.config.overlap_policy = DuplexSessionConfig._normalize_overlap_policy(
                str(payload["overlap_policy"])
            )
        if isinstance(payload.get("overlap_short_ack_ms"), int | float):
            session.config.overlap_short_ack_ms = max(0, int(payload["overlap_short_ack_ms"]))
        if isinstance(payload.get("overlap_barge_in_ms"), int | float):
            session.config.overlap_barge_in_ms = max(0, int(payload["overlap_barge_in_ms"]))
        if isinstance(payload.get("overlap_silence_rms"), int | float):
            session.config.overlap_silence_rms = max(0.0, float(payload["overlap_silence_rms"]))
        if isinstance(payload.get("playback_commit_policy"), str):
            session.config.playback_commit_policy = DuplexSessionConfig._normalize_playback_commit_policy(
                str(payload["playback_commit_policy"])
            )
        modalities = payload.get("modalities") or payload.get("output_modalities")
        if isinstance(modalities, list) and all(isinstance(item, str) for item in modalities):
            session.config.modalities = list(modalities)
        if isinstance(payload.get("extra_body"), dict):
            session.config.extra_body.update(payload["extra_body"])
            extra = payload["extra_body"]
            if isinstance(extra.get("overlap_policy"), str):
                session.config.overlap_policy = DuplexSessionConfig._normalize_overlap_policy(
                    str(extra["overlap_policy"])
                )
            if isinstance(extra.get("playback_commit_policy"), str):
                session.config.playback_commit_policy = DuplexSessionConfig._normalize_playback_commit_policy(
                    str(extra["playback_commit_policy"])
                )
        if isinstance(payload.get("tools"), list):
            session.config.extra_body["realtime_tools"] = payload["tools"]
        elif "tools" in payload and payload.get("tools") is None:
            session.config.extra_body.pop("realtime_tools", None)
        if isinstance(payload.get("tool_choice"), str | dict):
            session.config.extra_body["realtime_tool_choice"] = payload["tool_choice"]
        elif "tool_choice" in payload and payload.get("tool_choice") is None:
            session.config.extra_body.pop("realtime_tool_choice", None)
        if isinstance(payload.get("metadata"), dict):
            session.config.extra_body["realtime_metadata"] = dict(payload["metadata"])
        elif "metadata" in payload and payload.get("metadata") is None:
            session.config.extra_body.pop("realtime_metadata", None)
        if isinstance(payload.get("include"), list):
            session.config.extra_body["realtime_include"] = list(payload["include"])
        elif "include" in payload and payload.get("include") is None:
            session.config.extra_body.pop("realtime_include", None)
        if isinstance(payload.get("prompt"), dict):
            session.config.extra_body["realtime_prompt"] = dict(payload["prompt"])
        elif "prompt" in payload and payload.get("prompt") is None:
            session.config.extra_body.pop("realtime_prompt", None)
        input_audio_transcription = NativeRealtimeSessionProtocol._input_audio_transcription_config(payload)
        if isinstance(input_audio_transcription, dict):
            session.config.extra_body["realtime_input_audio_transcription"] = dict(input_audio_transcription)
        elif "input_audio_transcription" in payload and payload.get("input_audio_transcription") is None:
            session.config.extra_body.pop("realtime_input_audio_transcription", None)
        if isinstance(payload.get("input_audio_noise_reduction"), dict):
            session.config.extra_body["realtime_input_audio_noise_reduction"] = dict(
                payload["input_audio_noise_reduction"]
            )
        elif "input_audio_noise_reduction" in payload and payload.get("input_audio_noise_reduction") is None:
            session.config.extra_body.pop("realtime_input_audio_noise_reduction", None)
        if isinstance(audio_input, dict) and isinstance(audio_input.get("noise_reduction"), dict):
            session.config.extra_body["realtime_input_audio_noise_reduction"] = dict(audio_input["noise_reduction"])
        elif isinstance(audio_input, dict) and audio_input.get("noise_reduction") is None:
            session.config.extra_body.pop("realtime_input_audio_noise_reduction", None)
        if isinstance(payload.get("audio"), dict):
            session.config.extra_body["realtime_audio"] = dict(payload["audio"])
        elif "audio" in payload and payload.get("audio") is None:
            session.config.extra_body.pop("realtime_audio", None)
        if isinstance(payload.get("tracing"), str | dict):
            session.config.extra_body["realtime_tracing"] = payload["tracing"]
        elif "tracing" in payload and payload.get("tracing") is None:
            session.config.extra_body.pop("realtime_tracing", None)
        session.config.extra_body["realtime_session_payload"] = (
            NativeRealtimeSessionProtocol._json_safe_realtime_payload(payload)
        )
        return None

    def _apply_response_create_options(
        self,
        session: DuplexSession,
        payload: dict[str, object],
    ) -> str | None:
        """Reserve options that apply only to the next response lifecycle."""
        audio_config = payload.get("audio")
        audio_output = audio_config.get("output") if isinstance(audio_config, dict) else None
        if session.capabilities.implementation_level == "model_native_duplex":
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
                return "unsupported_native_response_options"

        instructions = str(payload["instructions"]) if isinstance(payload.get("instructions"), str) else None
        voice = payload.get("voice")
        if not isinstance(voice, str) and isinstance(audio_output, dict):
            voice = audio_output.get("voice")
        voice = str(voice) if isinstance(voice, str) else None
        response_format = payload.get("output_audio_format") or payload.get("response_format")
        if response_format is None and isinstance(audio_config, dict):
            if isinstance(audio_output, dict):
                response_format = audio_output.get("format")
        response_format, _ = NativeRealtimeSessionProtocol._parse_realtime_audio_format(response_format)
        if isinstance(response_format, str) and response_format.lower() in REALTIME_OUTPUT_AUDIO_FORMATS:
            response_format = NativeRealtimeSessionProtocol._duplex_response_format(response_format)
        else:
            response_format = None
        temperature = float(payload["temperature"]) if isinstance(payload.get("temperature"), int | float) else None
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
            max_tokens = NativeRealtimeSessionProtocol.realtime_max_output_tokens(max_tokens)
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
            if session.capabilities.implementation_level == "model_native_duplex":
                response_extra.update(
                    (key, value)
                    for key, value in extra_body.items()
                    if key not in self._serving_runtime_adapter.private_runtime_config_keys
                )
            else:
                response_extra.update(extra_body)
        try:
            session.reserve_response_options(
                ResponseCreateOptions(
                    instructions=instructions,
                    voice=voice,
                    response_format=response_format,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    speed=speed,
                    modalities=modalities,
                    extra_body=response_extra,
                )
            )
        except RuntimeError:
            return "response_already_active"
        return None

    @staticmethod
    def _realtime_item_to_history_message(item: object) -> dict[str, object] | None:
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
                    audio_chunks.append(
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/{fmt};base64,{audio}",
                            },
                        }
                    )
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

    async def _handle_playback_ack(self, session: DuplexSession, event: dict[str, object], send_json) -> None:
        played_ms = event.get("played_ms", event.get("audio_ms", 0))
        committed_ms = event.get("committed_ms")
        if not isinstance(played_ms, int | float):
            await send_json({"type": "error", "error": "playback.ack requires played_ms", "code": "bad_event"})
            return
        committed_cursor = int(committed_ms) if isinstance(committed_ms, int | float) else int(played_ms)
        item_id = event.get("item_id")
        response_id = event.get("response_id")
        response_id = response_id if isinstance(response_id, str) and response_id else None
        if not isinstance(item_id, str) or not item_id:
            item_id = f"item_{response_id}" if response_id is not None else None
        elif response_id is None and item_id.startswith("item_"):
            response_id = item_id.removeprefix("item_")
        if response_id is None and item_id is None and len(session.pending_history_item_ids) == 1:
            item_id = next(iter(session.pending_history_item_ids))
            if item_id.startswith("item_"):
                response_id = item_id.removeprefix("item_")
        if response_id is None and item_id is None and session.active_response_id is not None:
            response_id = session.active_response_id
            item_id = f"item_{response_id}"
        if event.get("truncate") is True:
            playback = session.acknowledge_playback(
                int(played_ms),
                committed_cursor,
                response_id=response_id,
            )
            playback = session.truncate_playback_commit(
                committed_cursor,
                response_id=response_id,
            )
        else:
            playback = session.acknowledge_playback(
                int(played_ms),
                committed_cursor,
                response_id=response_id,
            )
        committed_history = False
        if isinstance(item_id, str) and item_id:
            committed_history = session.truncate_history_item(
                item_id,
                audio_end_ms=committed_cursor,
                playback=playback,
            )
        elif session.pending_history_item_ids:
            # A plain playback ack has no OpenAI item id. Commit the only
            # uncommitted assistant candidate if the session has an unambiguous
            # pending response; otherwise wait for conversation.item.truncate.
            pending_ids = list(session.pending_history_item_ids)
            if len(pending_ids) == 1:
                item_id = pending_ids[0]
                committed_history = session.truncate_history_item(
                    item_id,
                    audio_end_ms=committed_cursor,
                    playback=playback,
                )
        elif session.active_response_id is not None:
            item_id = f"item_{session.active_response_id}"
            committed_history = session.truncate_history_item(
                item_id,
                audio_end_ms=committed_cursor,
                playback=playback,
            )
        elif session.last_assistant_full_message is not None:
            if item_id is None and session.history_item_ids:
                assistant_item_ids = [
                    known_item_id
                    for known_item_id, message in session.history_item_ids.items()
                    if message.get("role") == "assistant"
                ]
                if len(assistant_item_ids) == 1:
                    item_id = assistant_item_ids[0]
            if isinstance(item_id, str) and item_id:
                committed_history = session.truncate_history_item(
                    item_id,
                    audio_end_ms=committed_cursor,
                    playback=playback,
                )
        await send_json(
            {
                "type": "playback.acknowledged",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "item_id": item_id,
                "played_ms": int(played_ms),
                "committed_ms": committed_cursor,
                "truncate": event.get("truncate") is True,
                "playback": playback.as_dict(),
                "history_committed": committed_history,
            }
        )
        if committed_history and committed_cursor >= max(playback.sent_ms, playback.generated_ms):
            session.release_response_playback(response_id)

    async def _cancel_active_response(
        self,
        session: DuplexSession,
        active_task: asyncio.Task[None] | None,
        send_json,
        *,
        reason: str,
        notify: bool = True,
    ) -> bool:
        has_running_task = active_task is not None and not active_task.done()
        if not has_running_task and session.active_request_id is None and session.active_response_id is None:
            return False

        old_epoch = session.epoch
        old_request_id = session.active_request_id
        old_response_id = session.active_response_id
        committed_ms = session.playback.committed_ms
        committed_message = session.end_response(
            commit_text=self._should_commit_response_to_history(session, old_response_id),
            playback_commit_policy=DuplexPlaybackCommitPolicy.ACK_ONLY.value,
        )
        if old_response_id is not None:
            item_id = f"item_{old_response_id}"
            if committed_message is not None:
                session.register_history_item(item_id, committed_message)
            elif committed_ms > 0:
                session.truncate_history_item(item_id, audio_end_ms=committed_ms)
        new_epoch, old_playback = self._advance_barge_in_epoch(session)
        if old_request_id is not None:
            await self._abort_request_background(
                session,
                old_request_id,
                send_json,
                notify=notify,
            )
        if has_running_task and active_task is not None:
            active_task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(active_task, return_exceptions=True), timeout=0.25)
            except asyncio.TimeoutError:
                pass
        if notify:
            await send_json(
                {
                    "type": "audio.cancelled",
                    "session_id": session.session_id,
                    "response_id": old_response_id,
                    "reason": reason,
                    "cancelled_epoch": old_epoch,
                    "epoch": new_epoch,
                    "committed_ms": committed_ms,
                    "playback": old_playback,
                }
            )
        return True

    async def _abort_request_background(
        self,
        session: DuplexSession,
        request_id: str,
        send_json,
        *,
        notify: bool,
    ) -> None:
        try:
            abort_internal = getattr(self._chat_service.engine_client, "_abort_internal_requests", None)
            if callable(abort_internal):
                result = abort_internal([request_id])
            else:
                result = self._chat_service.engine_client.abort([request_id])
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.exception("Failed to abort duplex request %s: %s", request_id, exc)
            if notify and session.state != DuplexSessionState.CLOSED:
                await self._send_runtime_error(send_json, "runtime_abort_failed", exc, session=session)

    async def _cancel_pending_input(self, session: DuplexSession, send_json, *, reason: str) -> None:
        cancelled = session.cancel_pending_input()
        self._advance_barge_in_epoch(session)
        await send_json(
            {
                "type": "input.cancelled",
                "session_id": session.session_id,
                "reason": reason,
                "epoch": session.epoch,
                "cancelled": cancelled,
            }
        )
