from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import suppress
from copy import deepcopy

from fastapi import WebSocket, WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.engine.lease import DuplexLeaseActivity
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.openai.audio import convert_input_audio_with_rate
from vllm_omni.experimental.fullduplex.openai.commit_policy import (
    CommitAction,
    CommitSnapshot,
    decide_commit_action,
)
from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexPlaybackCommitPolicy,
    DuplexSession,
    DuplexSessionState,
    DuplexTurnEventType,
)
from vllm_omni.experimental.fullduplex.openai.realtime_session import (
    NativeRealtimeSessionProtocol,
)
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    PcmAppendReservation,
    ServingRuntimeSessionState,
    payload_turn_id,
)
from vllm_omni.experimental.fullduplex.openai.session_attachment import (
    DuplexJournalOverflowError,
)
from vllm_omni.experimental.fullduplex.openai.websocket import (
    DuplexWebSocketActor,
    is_input_event,
    normalize_duplex_input_event,
)

logger = init_logger(__name__)

_MAX_EVENT_BYTES = 15 * 1024 * 1024


class DuplexSessionRunnerMixin:
    """Run one ordered WebSocket mailbox for a duplex session."""

    async def handle_session(
        self,
        websocket: WebSocket,
        *,
        realtime_protocol: NativeRealtimeSessionProtocol | None = None,
    ) -> None:
        await websocket.accept()
        session: DuplexSession | None = None
        actor = DuplexWebSocketActor(
            websocket,
            current_epoch=lambda: session.epoch if session is not None else None,
            session_closed=lambda: session is not None and session.state == DuplexSessionState.CLOSED,
            outbound_protocol=realtime_protocol,
        )
        runtime_opened = False
        runtime_closed = False
        attachment_ready = False
        attachment_generation: int | None = None
        resume_credential_delivered = False
        transport_detached = False
        pending_turn_reservations = 0

        async def attachment_send(payload: dict[str, object]) -> None:
            nonlocal resume_credential_delivered
            try:
                await websocket.send_json(payload)
            except RuntimeError as exc:
                message = str(exc)
                if "after sending 'websocket.close'" in message or "response already completed" in message:
                    raise WebSocketDisconnect(code=1006) from exc
                raise
            if payload.get("type") == "session.created" and isinstance(payload.get("resume_token"), str):
                resume_credential_delivered = True

        async def attachment_close(reason: str) -> None:
            close = getattr(websocket, "close", None)
            if callable(close):
                await close(code=1000, reason=reason)

        async def send_attachment_payload(payload: dict[str, object], *, journal: bool) -> None:
            assert session is not None
            session_id = session.session_id
            should_journal = journal and session_id not in self._resync_required_sessions
            try:
                await self._attachment_registry.send_event(
                    session_id,
                    payload,
                    journal=should_journal,
                )
            except DuplexJournalOverflowError:
                first_overflow = session_id not in self._resync_required_sessions
                self._resync_required_sessions.add(session_id)
                if first_overflow:
                    resync_payload: dict[str, object] = {
                        "type": "session.resync_required",
                        "session_id": session_id,
                        "reason": "journal_overflow",
                    }
                    if realtime_protocol is not None:
                        resync_payload = realtime_protocol.encode_outbound_event(resync_payload)[0]
                    await self._attachment_registry.send_event(
                        session_id,
                        resync_payload,
                        journal=False,
                    )
                await self._attachment_registry.send_event(
                    session_id,
                    payload,
                    journal=False,
                )

        if realtime_protocol is not None:

            async def send_realtime_raw(payload: dict[str, object]) -> None:
                if attachment_ready and session is not None:
                    await send_attachment_payload(
                        payload,
                        journal=payload.get("type") not in {"session.created", "session.resumed"},
                    )
                    return
                raw_payload = dict(payload)
                raw_payload["_realtime_raw"] = True
                await actor.send_json(raw_payload)

            realtime_protocol.bind_sender(send_realtime_raw)
        native: ServingRuntimeSessionState = self._serving_runtime_adapter.create_session_state()

        def begin_close(reason: str) -> None:
            actor.closing = True
            actor.close_reason = reason
            if session is not None:
                session.mark_closing()

        event_emit_lock = asyncio.Lock()

        async def send_outbound(payload: dict[str, object]) -> None:
            if not attachment_ready or session is None:
                await actor.send_json(dict(payload))
                return
            if realtime_protocol is None:
                await send_attachment_payload(
                    payload,
                    journal=payload.get("type") not in {"session.created", "session.resumed"},
                )
                return
            for projected in realtime_protocol.encode_outbound_event(payload):
                await send_attachment_payload(
                    projected,
                    journal=projected.get("type") not in {"session.created", "session.resumed"},
                )

        async def emit_event(payload: dict[str, object]) -> None:
            deferred_precreate_response = False
            async with event_emit_lock:
                accepted, deferred_overlap_payload = await self._apply_outbound_session_event(
                    payload,
                    session=session,
                    actor=actor,
                    native=native,
                    realtime_protocol=realtime_protocol,
                )
                if not accepted:
                    return
                await send_outbound(payload)
                if deferred_overlap_payload is not None:
                    deferred_precreate_response = native.deferred_precreate_response
                    native.deferred_precreate_response = False
            if deferred_overlap_payload is not None and session is not None and not actor.closing:
                await start_native_append(
                    deferred_overlap_payload,
                    final=True,
                    precreate_response=deferred_precreate_response,
                    operation_id=native.committed_audio_operation_id,
                    retained_committed_payload=(
                        deferred_overlap_payload if native.committed_audio_payload is deferred_overlap_payload else None
                    ),
                )

        writer_task = asyncio.create_task(actor.writer_loop(), name="duplex-session-writer")
        reader_task: asyncio.Task[None] | None = None

        async def read_event_loop() -> None:
            nonlocal pending_turn_reservations
            assert session is not None
            try:
                while not actor.closing:
                    raw = await self._receive_text(
                        websocket,
                        session.config.idle_timeout_s,
                        realtime_protocol=realtime_protocol,
                    )
                    if raw is None:
                        await actor.enqueue_event({"type": "__timeout__"})
                        return
                    if len(raw.encode("utf-8")) > _MAX_EVENT_BYTES:
                        await emit_event(
                            {"type": "error", "error": "Duplex event too large", "code": "event_too_large"}
                        )
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        await emit_event({"type": "error", "error": "Invalid JSON event", "code": "invalid_json"})
                        continue
                    if not isinstance(event, dict):
                        await emit_event(
                            {
                                "type": "error",
                                "error": "Duplex event must be a JSON object",
                                "code": "bad_event",
                            }
                        )
                        continue
                    event = normalize_duplex_input_event(event)
                    event_type = event.get("type")
                    if not isinstance(event_type, str):
                        await emit_event(
                            {"type": "error", "error": "Duplex event missing string type", "code": "bad_event"}
                        )
                        continue
                    if event_type in {"input.commit", "input_audio_buffer.commit"}:
                        if not session.reserve_pending_turn(
                            limit=self._duplex_session_config.max_pending_turns_per_session
                        ):
                            await emit_event(
                                {
                                    "type": "error",
                                    "error": "Duplex session has too many pending input turns",
                                    "code": "input_backpressure",
                                }
                            )
                            continue
                        event["_duplex_pending_turn_reserved"] = True
                        pending_turn_reservations += 1
                    if is_input_event(event_type) and native_response_in_progress():
                        event["_duplex_overlap_candidate"] = True
                    await actor.enqueue_event(event)
            except WebSocketDisconnect:
                await actor.enqueue_event({"type": "__disconnect__"})

        async def next_actor_event() -> dict[str, object]:
            nonlocal pending_turn_reservations, transport_detached
            event = await actor.next_event()
            if event.pop("_duplex_pending_turn_reserved", False) and session is not None:
                session.release_pending_turn()
                pending_turn_reservations = max(0, pending_turn_reservations - 1)
            if (
                attachment_ready
                and attachment_generation is not None
                and session is not None
                and not await self._attachment_registry.is_current_attachment(
                    session.session_id,
                    attachment_generation,
                )
            ):
                transport_detached = True
                return {"type": "__replaced_attachment__"}
            return event

        def native_response_in_progress() -> bool:
            if session is None:
                return False
            if session.active_response_id is not None:
                return True
            if (
                session.config.playback_commit_policy == DuplexPlaybackCommitPolicy.ACK_ONLY.value
                and session.playback.sent_ms > session.playback.committed_ms
            ):
                return True
            if actor.active_response_task is not None and not actor.active_response_task.done():
                return True
            if actor.has_response_bound_append_tasks():
                return True
            return False

        def clear_completed_pending_silence() -> None:
            task = native.pending_silence_task
            if task is not None and task.done():
                native.pending_silence_task = None
                native.pending_silence_owner_id = None

        def mark_pending_silence_superseded() -> None:
            task = native.pending_silence_task
            if task is None:
                return
            if task.done():
                native.pending_silence_task = None
            # Do not cancel the task or clear native_append_tail here. A
            # silence append may already have reached the Engine, and local
            # cancellation cannot retract that RPC. The sequencer preserves
            # wire order; before_append will skip silence that has not started.
            native.pending_silence_owner_id = None

        def real_native_input_waiting() -> bool:
            clear_completed_pending_silence()
            return (
                native.audio_buffer.has_pending()
                or native.audio_buffer.has_reserved()
                or actor.has_queued_input_events()
            )

        async def start_native_append(
            payload: object,
            *,
            final: bool,
            precreate_response: bool = False,
            pcm_reservation: PcmAppendReservation | None = None,
            operation_id: str | None = None,
            retained_committed_payload: dict[str, object] | None = None,
            silence_continuation: bool = False,
            before_append=None,
        ) -> asyncio.Task[bool] | None:
            if session is None:
                return
            if not silence_continuation:
                mark_pending_silence_superseded()
            append_epoch = session.epoch
            append_turn_id = payload_turn_id(payload)
            if append_turn_id is None:
                append_turn_id = session.turn_id
            request_id = self._native_stage0_request_id(session, append_epoch)
            if final or precreate_response:
                session.bind_request(request_id)
            if precreate_response:
                session.bind_response_turn(append_turn_id)
            if precreate_response and session.active_response_id is None:
                response_id = session.begin_response(turn_id=append_turn_id)
                await emit_event(
                    self._response_created_payload(
                        session,
                        response_id,
                        epoch=append_epoch,
                    )
                )
            precreated_response_id = session.active_response_id if precreate_response else None

            async def _run() -> bool:
                nonlocal runtime_closed
                try:
                    append_ok, emitted_response = await self._append_runtime_input(
                        session,
                        payload,
                        operation_id=(pcm_reservation.operation_id if pcm_reservation is not None else operation_id),
                        final=final,
                        send_json=emit_event,
                        mode="append_audio_chunk",
                        expected_epoch=append_epoch,
                    )
                    if append_ok:
                        if pcm_reservation is not None:
                            pcm_reservation.commit()
                            session.release_input_bytes(pcm_reservation.byte_count)
                        if (
                            retained_committed_payload is not None
                            and native.committed_audio_payload is retained_committed_payload
                        ):
                            session.release_input_bytes(native.clear_committed_audio())
                    elif pcm_reservation is not None:
                        pcm_reservation.rollback()
                    if (
                        not append_ok
                        and precreated_response_id is not None
                        and session.active_response_id == precreated_response_id
                    ):
                        session.end_response(commit_text=False)
                        await emit_event(
                            {
                                "type": "response.done",
                                "session_id": session.session_id,
                                "response_id": precreated_response_id,
                                "epoch": session.epoch,
                                "committed": False,
                                "status": "failed",
                                "status_details": {
                                    "type": "failed",
                                    "reason": "runtime_append_failed",
                                },
                                "playback": session.playback.as_dict(),
                            }
                        )
                    if not append_ok and session.state == DuplexSessionState.CLOSED:
                        runtime_closed = True
                        return False
                    if not emitted_response and session.epoch == append_epoch:
                        if session.active_request_id == self._native_stage0_request_id(session, append_epoch):
                            session.clear_request(request_id)
                        if final:
                            await emit_event(
                                self._turn_controller.signal(session, DuplexTurnEventType.USER_STARTED.value)
                            )
                    return append_ok
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if pcm_reservation is not None:
                        pcm_reservation.rollback()
                    logger.exception("Native duplex append task failed: %s", exc)
                    await self._send_runtime_error(emit_event, "runtime_append_task_failed", exc, session=session)
                    if session.state != DuplexSessionState.CLOSED:
                        begin_close("runtime_append_task_failed")
                        if await self._close_runtime_session(
                            session,
                            reason="runtime_append_task_failed",
                            send_json=emit_event,
                        ):
                            runtime_closed = True
                            session.close()
                            await emit_event(
                                {
                                    "type": "session.closed",
                                    "session_id": session.session_id,
                                    "reason": "runtime_append_task_failed",
                                }
                            )
                    return False

            async def _run_in_wire_order(predecessor: asyncio.Task[bool] | None) -> bool:
                if predecessor is not None:
                    try:
                        predecessor_ok = await predecessor
                    except asyncio.CancelledError:
                        current = asyncio.current_task()
                        if current is not None and current.cancelling():
                            raise
                        predecessor_ok = False
                    except Exception:
                        predecessor_ok = False
                    if not predecessor_ok:
                        if pcm_reservation is not None:
                            pcm_reservation.rollback()
                        return False
                if actor.closing or runtime_closed or session.state != DuplexSessionState.OPEN:
                    if pcm_reservation is not None:
                        pcm_reservation.rollback()
                    return False
                if before_append is not None and not before_append():
                    if pcm_reservation is not None:
                        pcm_reservation.rollback()
                    return True
                if pcm_reservation is not None and not pcm_reservation.active:
                    return False
                return await _run()

            predecessor = actor.native_append_tail
            if predecessor is not None and predecessor.done():
                try:
                    predecessor_ok = predecessor.result()
                except (asyncio.CancelledError, Exception):
                    predecessor_ok = False
                if not predecessor_ok:
                    # Appends already queued behind a failed predecessor keep
                    # their captured dependency and stop. A later wire command
                    # is an explicit retry and starts a new chain.
                    predecessor = None
            task = asyncio.create_task(_run_in_wire_order(predecessor))
            actor.native_append_tail = task
            actor.track_append_task(
                task,
                epoch=append_epoch,
                mode="append_audio_chunk",
                final=final,
                response_bound=final or precreate_response,
            )
            if silence_continuation:
                native.pending_silence_task = task

                def _clear_done_pending_silence(done: asyncio.Task[bool]) -> None:
                    if native.pending_silence_task is done:
                        native.pending_silence_task = None
                        native.pending_silence_owner_id = None

                task.add_done_callback(_clear_done_pending_silence)
            # Let this wire-order effect start before the next mailbox event can
            # cancel it. Later native appends still serialize on predecessor.
            await asyncio.sleep(0)
            return task

        async def schedule_native_silence_continuation(
            payload: object,
            *,
            request_id: str,
            owner_id: str,
            response_id: str | None,
            response_owned: bool,
            expected_epoch: int | None,
            expected_incarnation: int,
            expected_model_turn_id: int | None,
            send_json,
        ) -> bool:
            del send_json
            if session is None:
                return False
            clear_completed_pending_silence()
            pending_silence = native.pending_silence_task
            if pending_silence is not None and not pending_silence.done():
                return False
            if actor.native_append_tail is not None and not actor.native_append_tail.done():
                return False
            if real_native_input_waiting():
                return False

            def _still_valid() -> bool:
                return not real_native_input_waiting() and not self._native_silence_continuation_is_stale(
                    session,
                    request_id=request_id,
                    response_id=response_id,
                    response_owned=response_owned,
                    expected_epoch=expected_epoch,
                    expected_incarnation=expected_incarnation,
                    expected_model_turn_id=expected_model_turn_id,
                )

            native.pending_silence_owner_id = owner_id
            task = await start_native_append(
                payload,
                final=False,
                silence_continuation=True,
                before_append=_still_valid,
            )
            if task is None:
                native.pending_silence_owner_id = None
                return False
            return True

        async def wait_for_native_append_tail() -> bool:
            predecessor = actor.native_append_tail
            if predecessor is None:
                return True
            try:
                return await predecessor
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                return False
            except Exception:
                # The append task already emitted its runtime error. Preserve
                # wire order before the update path decides whether to continue.
                return False

        async def start_runtime_append(
            payload: object,
            *,
            final: bool,
            mode: str = "append_tokens",
        ) -> None:
            """Schedule non-native runtime appends without blocking WS input.

            Model-native appends use ``start_native_append`` because they can
            create data-plane response streams. Generic runtime appends still
            need the same control-plane isolation so cancel/close can preempt a
            slow engine append.
            """
            if session is None:
                return
            append_epoch = session.epoch

            async def _run() -> None:
                nonlocal runtime_closed
                try:
                    append_ok, emitted_response = await self._append_runtime_input(
                        session,
                        payload,
                        final=final,
                        send_json=emit_event,
                        mode=mode,
                        expected_epoch=append_epoch,
                    )
                    if not append_ok:
                        if session.state == DuplexSessionState.CLOSED:
                            runtime_closed = True
                        return
                    if not emitted_response and session.epoch == append_epoch:
                        await emit_event(self._turn_controller.signal(session, DuplexTurnEventType.USER_STARTED.value))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("Duplex runtime append task failed: %s", exc)
                    await self._send_runtime_error(emit_event, "runtime_append_task_failed", exc, session=session)

            task = asyncio.create_task(_run())
            actor.track_append_task(
                task,
                epoch=append_epoch,
                mode=mode,
                final=final,
                response_bound=final,
            )

        try:
            handshake = await self._open_session(
                websocket,
                emit_event,
                realtime_protocol=realtime_protocol,
                attachment_send=attachment_send,
                attachment_close=attachment_close,
            )
            if handshake is None:
                return
            session = handshake.session
            if handshake.resumed:
                native = self._serving_runtime_adapter.session_states[session.session_id]
                actor.tasks = self._session_tasks[session.session_id]
                persisted_protocol = self._realtime_protocols.get(session.session_id)
                if persisted_protocol is None:
                    raise RuntimeError(f"Missing Realtime protocol state for resumed session {session.session_id}")
                realtime_protocol = persisted_protocol
                realtime_protocol.bind_sender(send_realtime_raw)
                attachment_generation = handshake.attachment_generation
                attachment_ready = True
                resume_credential_delivered = True
                runtime_opened = True
                self._ensure_lifecycle_listener()
                reader_task = asyncio.create_task(read_event_loop(), name="duplex-session-reader")
            else:
                self._serving_runtime_adapter.session_states[session.session_id] = native
                self._session_tasks[session.session_id] = actor.tasks
                if realtime_protocol is not None:
                    self._realtime_protocols[session.session_id] = realtime_protocol
            native.silence_continuation_scheduler = schedule_native_silence_continuation
            if realtime_protocol is not None:
                session.config.playback_commit_policy = DuplexPlaybackCommitPolicy.ACK_ONLY.value
            if not handshake.resumed:
                open_result = await self._open_runtime_session(session, emit_event)
                if open_result is False:
                    return
                runtime_opened = True
                created_attachment = await self._attachment_registry.create(
                    session.session_id,
                    incarnation=session.incarnation,
                    send=attachment_send,
                    close=attachment_close,
                )
                attachment_generation = created_attachment.attachment_generation
                attachment_ready = True
                self._lease_generations[session.session_id] = 0
                self._ensure_lifecycle_listener()
                created_payload: dict[str, object] = {
                    "type": "session.created",
                    "session": session.as_public_dict(),
                }
                if session.capabilities.supports_session_resume:
                    created_payload.update(
                        {
                            "incarnation": session.incarnation,
                            "attachment_generation": attachment_generation,
                            "resume_token": created_attachment.resume_token.plaintext,
                        }
                    )
                if isinstance(open_result, dict):
                    created_payload["runtime_control"] = self._redact_runtime_control_result(open_result)
                await emit_event(created_payload)
                resume_credential_delivered = session.capabilities.supports_session_resume
                reader_task = asyncio.create_task(read_event_loop(), name="duplex-session-reader")

            while True:
                event = await next_actor_event()
                event_type = event.get("type")

                if event_type == "__replaced_attachment__":
                    return

                if event_type == "__timeout__":
                    begin_close("timeout")
                    native.audio_buffer.clear()
                    session.release_all_input_bytes()
                    native.input_since_commit = False
                    native.speech_since_commit = False
                    native.clear_committed_audio()
                    await actor.cancel_append_tasks()
                    await self._cancel_native_data_plane_stream(session)
                    await self._cancel_active_response(
                        session,
                        actor.active_response_task,
                        emit_event,
                        reason="timeout",
                        notify=True,
                    )
                    actor.active_response_task = None
                    runtime_closed = await self._close_runtime_session(
                        session,
                        reason="timeout",
                        send_json=emit_event,
                    )
                    if not runtime_closed:
                        return
                    session.close()
                    await emit_event(
                        {
                            "type": "session.closed",
                            "session_id": session.session_id,
                            "reason": "timeout",
                        }
                    )
                    return

                if event_type == "__disconnect__":
                    transport_detached = True
                    return

                if not isinstance(event_type, str):
                    continue

                if event_type == "session.heartbeat":
                    touch_session = getattr(self._chat_service.engine_client, "touch_duplex_session_async", None)
                    if not callable(touch_session):
                        await emit_event(
                            {
                                "type": "error",
                                "error": "Duplex runtime does not expose heartbeat control",
                                "code": "runtime_touch_unsupported",
                            }
                        )
                        continue
                    try:
                        await touch_session(
                            session.session_id,
                            fence=DuplexFence(
                                session.session_id,
                                epoch=session.epoch,
                                turn_id=session.turn_id,
                                incarnation=session.incarnation,
                            ),
                            activity=DuplexLeaseActivity.HEARTBEAT,
                        )
                    except Exception as exc:
                        await emit_event(
                            {
                                "type": "error",
                                "error": str(exc),
                                "code": "runtime_touch_failed",
                            }
                        )
                        continue
                    await emit_event(
                        {
                            "type": "session.heartbeat_ack",
                            "session_id": session.session_id,
                        }
                    )
                    continue

                if event_type == "session.event_ack":
                    acknowledged = event.get("server_event_seq")
                    if not isinstance(acknowledged, int) or acknowledged < 0:
                        await emit_event(
                            {
                                "type": "error",
                                "error": "session.event_ack requires a non-negative server_event_seq",
                                "code": "invalid_event_ack",
                            }
                        )
                        continue
                    try:
                        await self._attachment_registry.acknowledge(session.session_id, acknowledged)
                    except ValueError as exc:
                        await emit_event(
                            {
                                "type": "error",
                                "error": str(exc),
                                "code": "invalid_event_ack",
                            }
                        )
                    continue

                if event_type == "turn.signal" and event.get("event") in {
                    "input.cancel",
                    "response.cancel",
                    "barge_in",
                }:
                    payload = event.get("payload")
                    normalized_event = dict(payload) if isinstance(payload, dict) else {}
                    normalized_event.update(event)
                    normalized_event["type"] = event["event"]
                    event = normalized_event
                    event_type = str(event["type"])

                if event_type == "session.close":
                    with suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(actor.output_queue.join(), timeout=1.0)
                    begin_close("session_close")
                    if session.state == DuplexSessionState.CLOSED:
                        runtime_closed = True
                        return
                    native.audio_buffer.clear()
                    session.release_all_input_bytes()
                    native.input_since_commit = False
                    native.speech_since_commit = False
                    native.clear_committed_audio()
                    await actor.cancel_append_tasks()
                    await self._cancel_native_data_plane_stream(session)
                    await self._cancel_active_response(
                        session,
                        actor.active_response_task,
                        emit_event,
                        reason="session_close",
                    )
                    runtime_closed = await self._close_runtime_session(
                        session,
                        reason="session_close",
                        send_json=emit_event,
                    )
                    actor.active_response_task = None
                    if not runtime_closed:
                        return
                    await emit_event({"type": "session.closed", "session_id": session.session_id})
                    session.close()
                    return

                if event_type == "input_audio_buffer.clear":
                    native.audio_buffer.clear()
                    session.release_all_input_bytes()
                    native.input_since_commit = False
                    native.speech_since_commit = False
                    native.clear_committed_audio()
                    cancelled = session.cancel_pending_input()
                    await emit_event(
                        {
                            "type": "input_audio_buffer.cleared",
                            "session_id": session.session_id,
                            "epoch": session.epoch,
                            "drained_input_events": 0,
                            "cancelled": cancelled,
                        }
                    )
                    continue

                if event_type == "barge_in" and not session.capabilities.supports_barge_in:
                    await emit_event(self._barge_in_unsupported_error(session))
                    continue

                if event_type in {"input.cancel", "response.cancel", "barge_in", "output_audio_buffer.clear"}:
                    cancel_reason = (
                        "output_audio_buffer_clear" if event_type == "output_audio_buffer.clear" else "barge_in"
                    )
                    cancelled_fence = DuplexFence(
                        session.session_id,
                        epoch=session.epoch,
                        turn_id=session.turn_id,
                        incarnation=session.incarnation,
                    )
                    if event_type == "response.cancel":
                        requested_response_id = event.get("response_id")
                        has_active_response_work = native_response_in_progress()
                        if (
                            isinstance(requested_response_id, str)
                            and session.active_response_id is not None
                            and requested_response_id != session.active_response_id
                        ):
                            await emit_event(
                                {
                                    "type": "error",
                                    "session_id": session.session_id,
                                    "code": "response_not_active",
                                    "error": f"Response is not active: {requested_response_id}",
                                }
                            )
                            continue
                        if not has_active_response_work:
                            if realtime_protocol is not None and isinstance(requested_response_id, str):
                                continue
                            await emit_event(
                                {
                                    "type": "error",
                                    "session_id": session.session_id,
                                    "code": "response_not_active",
                                    "error": "response.cancel requires an active response",
                                }
                            )
                            continue
                    had_native_unbuffered_append = (
                        self._uses_native_input_append(session)
                        and native.input_since_commit
                        and not native.audio_buffer.has_pending()
                    )
                    playback_was_active = self._assistant_playback_active(session)
                    if event_type in {"input.cancel", "barge_in"}:
                        native.audio_buffer.clear()
                        session.release_all_input_bytes()
                        native.input_since_commit = False
                        native.speech_since_commit = False
                        native.clear_committed_audio()
                    had_native_append = await actor.cancel_append_tasks(
                        response_bound_only=event_type in {"response.cancel", "output_audio_buffer.clear"},
                    )
                    had_native_stream = native.data_plane_task is not None
                    cancelled = await self._cancel_active_response(
                        session,
                        actor.active_response_task,
                        emit_event,
                        reason=cancel_reason,
                    )
                    had_native_stream = await self._cancel_native_data_plane_stream(session) or had_native_stream
                    if not cancelled and (had_native_stream or had_native_append or had_native_unbuffered_append):
                        old_epoch = session.epoch
                        old_response_id = session.active_response_id
                        committed_ms = session.playback.committed_ms
                        self._commit_played_response_history(session, old_response_id, committed_ms)
                        new_epoch, old_playback = self._advance_barge_in_epoch(session)
                        await emit_event(
                            {
                                "type": "audio.cancelled",
                                "session_id": session.session_id,
                                "response_id": old_response_id,
                                "reason": cancel_reason,
                                "cancelled_epoch": old_epoch,
                                "epoch": new_epoch,
                                "committed_ms": committed_ms,
                                "playback": old_playback,
                            }
                        )
                        cancelled = True
                    if not cancelled and playback_was_active:
                        old_epoch = session.epoch
                        committed_ms = session.playback.committed_ms
                        self._commit_played_response_history(session, session.last_response_id, committed_ms)
                        new_epoch, old_playback = self._advance_barge_in_epoch(session)
                        await emit_event(
                            {
                                "type": "audio.cancelled",
                                "session_id": session.session_id,
                                "response_id": session.last_response_id,
                                "reason": cancel_reason,
                                "cancelled_epoch": old_epoch,
                                "epoch": new_epoch,
                                "committed_ms": committed_ms,
                                "playback": old_playback,
                            }
                        )
                        cancelled = True
                    if not cancelled and event_type == "response.cancel":
                        old_epoch = session.epoch
                        old_response_id = session.active_response_id
                        committed_ms = session.playback.committed_ms
                        self._commit_played_response_history(session, old_response_id, committed_ms)
                        new_epoch, old_playback = self._advance_barge_in_epoch(session)
                        await emit_event(
                            {
                                "type": "audio.cancelled",
                                "session_id": session.session_id,
                                "response_id": old_response_id,
                                "reason": cancel_reason,
                                "cancelled_epoch": old_epoch,
                                "epoch": new_epoch,
                                "committed_ms": committed_ms,
                                "playback": old_playback,
                            }
                        )
                        cancelled = True
                    if not cancelled and event_type == "output_audio_buffer.clear":
                        old_playback = session.playback.as_dict()
                        committed_ms = session.playback.committed_ms
                        session.clear_playback_cursor()
                        await emit_event(
                            {
                                "type": "audio.cancelled",
                                "session_id": session.session_id,
                                "response_id": session.active_response_id,
                                "reason": cancel_reason,
                                "cancelled_epoch": session.epoch,
                                "epoch": session.epoch,
                                "committed_ms": committed_ms,
                                "playback": old_playback,
                            }
                        )
                        actor.active_response_task = None
                        continue
                    if not cancelled:
                        await self._cancel_pending_input(session, emit_event, reason="barge_in")
                    if not await self._signal_runtime_session(
                        session,
                        "barge_in",
                        emit_event,
                        fence=cancelled_fence,
                        next_fence=(
                            DuplexFence(
                                session.session_id,
                                epoch=session.epoch,
                                turn_id=session.turn_id,
                                incarnation=session.incarnation,
                            )
                            if session.epoch > cancelled_fence.epoch
                            else None
                        ),
                    ):
                        continue
                    actor.active_response_task = None
                    continue

                if event_type == "turn.signal":
                    turn_event = event.get("event")
                    if isinstance(turn_event, str):
                        if turn_event == "barge_in" and not session.capabilities.supports_barge_in:
                            await emit_event(self._barge_in_unsupported_error(session))
                            continue
                        if turn_event == "session.update":
                            payload = event.get("payload")
                            if not isinstance(payload, dict):
                                await emit_event(
                                    {
                                        "type": "error",
                                        "session_id": session.session_id,
                                        "code": "bad_event",
                                        "error": "session.update requires a session payload",
                                    }
                                )
                                continue
                            if not await wait_for_native_append_tail():
                                continue
                            runtime_update_error = self._runtime_session_update_error(session, payload)
                            if runtime_update_error is not None:
                                await emit_event(runtime_update_error)
                                continue
                            previous_config = session.config
                            candidate_config = deepcopy(previous_config)
                            session.replace_config(candidate_config)
                            try:
                                update_error = self._apply_session_update(session, payload)
                            finally:
                                session.replace_config(previous_config)
                            if update_error is not None:
                                await emit_event(update_error)
                                continue
                            runtime_update_error = self._runtime_session_candidate_update_error(
                                session,
                                candidate_config,
                            )
                            if runtime_update_error is not None:
                                await emit_event(runtime_update_error)
                                continue
                            candidate_runtime_config = self._runtime_config_for_session_update(
                                session,
                                candidate_config,
                            )
                            if not await self._signal_runtime_session(
                                session,
                                turn_event,
                                emit_event,
                                session_config=candidate_config.as_dict(),
                                runtime_config=candidate_runtime_config,
                            ):
                                continue
                            session.replace_config(candidate_config)
                            session.replace_runtime_config(candidate_runtime_config)
                            await emit_event(
                                {
                                    "type": "session.updated",
                                    "session": session.as_public_dict(),
                                }
                            )
                            continue
                        if turn_event == "conversation.item.create":
                            payload = event.get("payload")
                            item = payload.get("item") if isinstance(payload, dict) else None
                            message = self._realtime_item_to_history_message(item)
                            item_id = item.get("id") if isinstance(item, dict) else None
                            if message is not None:
                                session.append_history_message(message)
                                session.register_history_item(item_id if isinstance(item_id, str) else None, message)
                            await emit_event(
                                {
                                    "type": "conversation.item.created",
                                    "session_id": session.session_id,
                                    "item": item,
                                    "created": message is not None,
                                }
                            )
                            continue
                        if turn_event == "conversation.item.delete":
                            payload = event.get("payload")
                            item_id = payload.get("item_id") if isinstance(payload, dict) else None
                            deleted = session.delete_history_item(item_id) if isinstance(item_id, str) else False
                            await emit_event(
                                {
                                    "type": "conversation.item.deleted",
                                    "session_id": session.session_id,
                                    "item_id": item_id,
                                    "deleted": deleted,
                                }
                            )
                            continue
                        if turn_event == "conversation.item.truncate":
                            payload = event.get("payload")
                            item_id = payload.get("item_id") if isinstance(payload, dict) else None
                            audio_end_ms = payload.get("audio_end_ms") if isinstance(payload, dict) else None
                            truncated = (
                                session.truncate_history_item(
                                    item_id,
                                    audio_end_ms=int(audio_end_ms) if isinstance(audio_end_ms, int | float) else 0,
                                )
                                if isinstance(item_id, str)
                                else False
                            )
                            await emit_event(
                                {
                                    "type": "conversation.item.truncated",
                                    "session_id": session.session_id,
                                    "item_id": item_id,
                                    "content_index": (
                                        payload.get("content_index", 0) if isinstance(payload, dict) else 0
                                    ),
                                    "audio_end_ms": audio_end_ms,
                                    "truncated": truncated,
                                }
                            )
                            continue
                        await emit_event(self._turn_controller.signal(session, turn_event, event))
                    else:
                        await emit_event({"type": "error", "error": "turn.signal requires event", "code": "bad_event"})
                    continue

                if event_type == "playback.ack":
                    await self._handle_playback_ack(session, event, emit_event)
                    continue

                if event_type == "input.text.append":
                    session.mark_user_input_activity()
                    text = event.get("text")
                    if not isinstance(text, str):
                        await emit_event(
                            {
                                "type": "error",
                                "error": "input.text.append requires text",
                                "code": "bad_event",
                            }
                        )
                        continue
                    if self._uses_native_input_append(session):
                        await emit_event(
                            {
                                "type": "error",
                                "error": "The selected native duplex runtime accepts audio append only",
                                "code": "native_text_append_unsupported",
                            }
                        )
                        continue
                    else:
                        session.append_text(text)
                    if session.capabilities.supports_input_append:
                        await start_runtime_append(text, final=False, mode="append_tokens")
                    continue

                if event_type == "input_audio_buffer.append":
                    session.mark_user_input_activity()
                    audio = event.get("audio") or event.get("data")
                    if not isinstance(audio, str):
                        await emit_event(
                            {
                                "type": "error",
                                "error": "input.audio.append requires audio",
                                "code": "bad_event",
                            }
                        )
                        continue
                    if not session.capabilities.supports_barge_in and self._event_requests_barge_in(event):
                        await emit_event(self._barge_in_unsupported_error(session))
                        event = dict(event)
                        event.pop("force_barge_in", None)
                        for key in ("overlap_action", "overlap"):
                            value = event.get(key)
                            if isinstance(value, str) and value.strip().lower() in {
                                "barge_in",
                                "interrupt",
                                "cancel",
                            }:
                                event.pop(key, None)
                    fmt = event.get("format") if isinstance(event.get("format"), str) else "pcm16"
                    default_sample_rate_hz = 16000
                    sr_raw = event.get("sample_rate_hz") or event.get("sample_rate")
                    sample_rate_hz = sr_raw if isinstance(sr_raw, int | float) else default_sample_rate_hz
                    try:
                        audio, fmt, sample_rate_hz = convert_input_audio_with_rate(
                            audio,
                            fmt,
                            sample_rate_hz=sample_rate_hz,
                        )
                    except ValueError as exc:
                        await emit_event({"type": "error", "error": str(exc), "code": "bad_event"})
                        continue
                    if isinstance(fmt, str) and fmt.lower() in {"pcm16", "pcm_s16le", "s16le"}:
                        await emit_event(
                            {
                                "type": "error",
                                "error": "input_audio_buffer.append pcm16 audio could not be decoded",
                                "code": "bad_audio",
                            }
                        )
                        continue
                    force_listen = bool(event.get("force_listen", False))
                    payload = {
                        "type": "audio",
                        "audio": audio,
                        "format": fmt,
                        "sample_rate_hz": sample_rate_hz,
                        "force_listen": force_listen,
                    }
                    video_frames = event.get("video_frames")
                    if isinstance(video_frames, list):
                        frames = [frame for frame in video_frames if isinstance(frame, str) and frame]
                        if frames:
                            payload["video_frames"] = frames
                    # Speech/silence tag for the Stage0 turn-ended latch.
                    payload["is_speech"] = self._input_looks_like_speech(event, payload, session=session)
                    defer_native_append = False
                    buffer_overlap_audio = True
                    if self._uses_native_input_append(session):
                        mark_pending_silence_superseded()
                        overlap_active = not self._session_auto_responds(session) and native_response_in_progress()
                        if overlap_active:
                            decision = self._overlap_decision(session, event, payload)
                            await self._emit_overlap_decision(emit_event, session, decision)
                            action = decision.get("action")
                            if action == "drop":
                                if realtime_protocol is not None:
                                    await realtime_protocol.discard_pending_input_audio(
                                        audio_end_ms=self._input_audio_duration_ms(event, payload)
                                    )
                                continue
                            if action == "listen":
                                buffer_overlap_audio = bool(decision.get("buffer_audio", True))
                                defer_native_append = bool(decision.get("defer_runtime_append", True))
                                if (
                                    not buffer_overlap_audio
                                    and realtime_protocol is not None
                                    and decision.get("preserve_realtime_input") is not True
                                ):
                                    await realtime_protocol.discard_pending_input_audio(
                                        audio_end_ms=self._input_audio_duration_ms(event, payload)
                                    )
                                if decision.get("force_listen", True) is True:
                                    payload["force_listen"] = True
                            else:
                                event["force_barge_in"] = True
                                cancelled_fence = DuplexFence(
                                    session.session_id,
                                    epoch=session.epoch,
                                    turn_id=session.turn_id,
                                    incarnation=session.incarnation,
                                )
                                playback_was_active = self._assistant_playback_active(session)
                                buffer_overlap_audio = True
                                defer_native_append = False
                                native.audio_buffer.clear_force_listen()
                                session.reset_overlap_speech()
                                native.input_since_commit = False
                                native.speech_since_commit = False
                                await actor.cancel_append_tasks()
                                had_native_stream = native.data_plane_task is not None
                                cancelled = await self._cancel_active_response(
                                    session,
                                    actor.active_response_task,
                                    emit_event,
                                    reason="barge_in",
                                )
                                had_native_stream = (
                                    await self._cancel_native_data_plane_stream(session) or had_native_stream
                                )
                                if not cancelled and had_native_stream:
                                    old_epoch = session.epoch
                                    old_response_id = session.active_response_id
                                    committed_ms = session.playback.committed_ms
                                    self._commit_played_response_history(session, old_response_id, committed_ms)
                                    new_epoch, old_playback = self._advance_barge_in_epoch(session)
                                    await emit_event(
                                        {
                                            "type": "audio.cancelled",
                                            "session_id": session.session_id,
                                            "response_id": old_response_id,
                                            "reason": "barge_in",
                                            "cancelled_epoch": old_epoch,
                                            "epoch": new_epoch,
                                            "committed_ms": committed_ms,
                                            "playback": old_playback,
                                        }
                                    )
                                    cancelled = True
                                if not cancelled and playback_was_active:
                                    old_epoch = session.epoch
                                    committed_ms = session.playback.committed_ms
                                    self._commit_played_response_history(
                                        session, session.last_response_id, committed_ms
                                    )
                                    new_epoch, old_playback = self._advance_barge_in_epoch(session)
                                    await emit_event(
                                        {
                                            "type": "audio.cancelled",
                                            "session_id": session.session_id,
                                            "response_id": session.last_response_id,
                                            "reason": "barge_in",
                                            "cancelled_epoch": old_epoch,
                                            "epoch": new_epoch,
                                            "committed_ms": committed_ms,
                                            "playback": old_playback,
                                        }
                                    )
                                    cancelled = True
                                if session.epoch > cancelled_fence.epoch:
                                    if not await self._signal_runtime_session(
                                        session,
                                        "barge_in",
                                        emit_event,
                                        fence=cancelled_fence,
                                        next_fence=DuplexFence(
                                            session.session_id,
                                            epoch=session.epoch,
                                            turn_id=session.turn_id,
                                            incarnation=session.incarnation,
                                        ),
                                    ):
                                        continue
                                actor.active_response_task = None
                        elif not self._session_auto_responds(session) and not self._input_looks_like_speech(
                            event, payload, session=session
                        ):
                            # Turn-mode only: skip silent chunks so they don't open a
                            # response. In auto-respond (full-duplex) mode the model owns
                            # the speak/listen decision and MUST receive silence units --
                            # the official model typically starts speaking during the
                            # silence after a question.
                            await emit_event(
                                {
                                    "type": "response.listen",
                                    "session_id": session.session_id,
                                    "epoch": session.epoch,
                                    "reason": "silence_or_noise",
                                }
                            )
                            continue
                        if self._should_force_listen_for_auto_response_overlap(session, event, payload):
                            # Auto-response keeps a long-lived native Stage0 stream.
                            # While assistant audio is still active, silence from the
                            # browser should advance the model in listen mode rather
                            # than letting the same stream start another assistant
                            # segment. Speech/barge-in chunks remain model-owned.
                            payload["force_listen"] = True
                        if not buffer_overlap_audio:
                            continue
                        session.mark_user_input_activity()
                        native.input_since_commit = True
                        native.speech_since_commit = native.speech_since_commit or self._input_looks_like_speech(
                            event, payload, session=session
                        )
                        try:
                            raw_audio_bytes = self._native_audio_payload_size_bytes(payload)
                            if not session.reserve_input_bytes(
                                raw_audio_bytes,
                                limit=self._duplex_session_config.max_pending_input_bytes_per_session,
                            ):
                                await emit_event(
                                    {
                                        "type": "error",
                                        "error": "Duplex session pending input exceeds server limit",
                                        "code": "input_backpressure",
                                    }
                                )
                                continue
                            allow_emit = not defer_native_append and (
                                realtime_protocol is None
                                or event_type != "input_audio_buffer.append"
                                # Full-duplex: emit each ~chunk_period of audio so the model
                                # runs per-chunk generation (speak/listen) without an explicit
                                # response.create, matching the official duplex_generate loop.
                                or self._session_auto_responds(session)
                            )
                            pcm_reservation = native.audio_buffer.prepare_append(
                                payload,
                                operation_id=uuid.uuid4().hex,
                                chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
                                allow_emit=allow_emit,
                            )
                        except ValueError as exc:
                            session.release_input_bytes(raw_audio_bytes)
                            await emit_event({"type": "error", "error": str(exc), "code": "bad_event"})
                            continue
                        if pcm_reservation is None:
                            continue
                        if pcm_reservation.byte_count == 0:
                            session.release_input_bytes(raw_audio_bytes)
                        payload = pcm_reservation.payload
                    else:
                        session.append_audio(audio, fmt=fmt, sample_rate_hz=sample_rate_hz)
                    if self._uses_native_input_append(session):
                        await start_native_append(
                            payload,
                            final=False,
                            pcm_reservation=pcm_reservation,
                        )
                        continue
                    if session.capabilities.supports_input_append:
                        await start_runtime_append(payload, final=False, mode="append_audio_chunk")
                    continue

                if event_type in {"input.commit", "input_audio_buffer.commit", "response.create"}:
                    realtime_item_id = event.get("realtime_item_id")
                    realtime_validated_audio_commit = (
                        event_type == "input_audio_buffer.commit"
                        and isinstance(realtime_item_id, str)
                        and bool(realtime_item_id)
                    )
                    if (
                        self._uses_native_input_append(session)
                        and event_type in {"input.commit", "input_audio_buffer.commit"}
                        and not await wait_for_native_append_tail()
                    ):
                        continue
                    if event_type == "input_audio_buffer.commit" and event.get("is_speech") is False:
                        native.input_since_commit = False
                        native.speech_since_commit = False
                        native.audio_buffer.clear()
                        session.release_all_input_bytes()
                        native.clear_committed_audio()
                        await emit_event(
                            {
                                "type": "input.committed",
                                "session_id": session.session_id,
                                "turn_id": session.turn_id,
                                "epoch": session.epoch,
                                "empty": True,
                                "is_speech": False,
                                "no_response": True,
                            }
                        )
                        await emit_event(
                            {
                                "type": "response.listen",
                                "session_id": session.session_id,
                                "epoch": session.epoch,
                                "reason": "silence_or_noise",
                            }
                        )
                        continue
                    should_create_response = (
                        event_type == "response.create"
                        or bool(event.get("response_create", event_type == "input.commit"))
                        or (event_type == "input_audio_buffer.commit" and self._session_auto_responds(session))
                    )
                    precreate_response_requested = event_type == "response.create" or bool(
                        event.get("response_create", event_type == "input.commit")
                    )
                    if event_type == "response.create":
                        response_payload = event.get("response")
                        if isinstance(response_payload, dict):
                            response_options_error = self._apply_response_create_options(session, response_payload)
                            if response_options_error is not None:
                                error_message = (
                                    "The selected native duplex runtime does not support generation "
                                    "overrides for instructions, voice, temperature, max tokens, tools, or "
                                    "tool_choice."
                                    if response_options_error == "unsupported_native_response_options"
                                    else "response.create cannot reserve options while another response is active."
                                )
                                await emit_event(
                                    {
                                        "type": "error",
                                        "session_id": session.session_id,
                                        "epoch": session.epoch,
                                        "code": response_options_error,
                                        "error": error_message,
                                    }
                                )
                                continue
                    if self._uses_native_input_append(session) and event_type == "input_audio_buffer.commit":
                        has_pending_native_audio = (
                            native.input_since_commit
                            or native.audio_buffer.has_pending()
                            or native.committed_audio_payload is not None
                            or realtime_validated_audio_commit
                        )
                        if (
                            not has_pending_native_audio
                            and not actor.native_append_tasks
                            and native.data_plane_task is None
                        ):
                            await emit_event(
                                {
                                    "type": "error",
                                    "session_id": session.session_id,
                                    "epoch": session.epoch,
                                    "code": "input_audio_buffer_empty",
                                    "error": "input_audio_buffer.commit requires a non-empty input audio buffer.",
                                }
                            )
                            continue
                        commit_action = decide_commit_action(
                            CommitSnapshot(
                                auto_responds=self._session_auto_responds(session),
                                speech_since_commit=native.speech_since_commit,
                                active_response_id=session.active_response_id,
                                overlap_speech_ms=session.overlap_speech_ms,
                                native_response_in_progress=native_response_in_progress(),
                                playback_active=self._assistant_playback_active(session),
                            )
                        )
                        if commit_action is CommitAction.DEFER_ACTIVE_RESPONSE:
                            if session.overlap_speech_ms <= session.config.overlap_short_ack_ms:
                                native.audio_buffer.clear()
                                session.release_all_input_bytes()
                                native.input_since_commit = False
                                native.speech_since_commit = False
                                native.clear_committed_audio()
                                if realtime_protocol is not None:
                                    await realtime_protocol.discard_pending_input_audio(
                                        audio_end_ms=session.overlap_speech_ms
                                    )
                                await emit_event(
                                    {
                                        "type": "input.committed",
                                        "session_id": session.session_id,
                                        "turn_id": session.turn_id,
                                        "epoch": session.epoch,
                                        "empty": True,
                                        "is_speech": False,
                                        "overlap_ack": True,
                                        "no_response": True,
                                    }
                                )
                                session.reset_overlap_speech()
                                session.discard_response_options()
                                continue

                            commit_reservation = native.audio_buffer.prepare_commit(
                                operation_id=uuid.uuid4().hex,
                                chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
                            )
                            deferred_payload = commit_reservation.payload
                            if deferred_payload is None:
                                commit_reservation.commit()
                            else:
                                if native.committed_audio_payload is not None:
                                    deferred_payload = self._merge_native_audio_payloads(
                                        native.committed_audio_payload,
                                        deferred_payload,
                                    )
                                native.retain_committed_audio(
                                    deferred_payload,
                                    operation_id=commit_reservation.operation_id,
                                    reserved_bytes=commit_reservation.byte_count,
                                )
                                commit_reservation.commit()
                                native.deferred_response_create = should_create_response
                                native.deferred_precreate_response = precreate_response_requested
                                native.input_since_commit = False
                                native.speech_since_commit = False
                                committed = self._commit_native_audio_input(
                                    session,
                                    realtime_item_id=event.get("realtime_item_id"),
                                    transcript=event.get("transcript"),
                                )
                                committed_payload = self._native_audio_committed_payload(
                                    session,
                                    committed=committed,
                                    realtime_item_id=event.get("realtime_item_id"),
                                    transcript=event.get("transcript"),
                                )
                                committed_payload["overlap_deferred"] = True
                                committed_payload["response_create_deferred"] = should_create_response
                                await emit_event(committed_payload)
                                continue
                        if commit_action is CommitAction.START_AUTO_RESPONSE:
                            commit_reservation = native.audio_buffer.prepare_commit(
                                operation_id=uuid.uuid4().hex,
                                chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
                            )
                            committed_input = commit_reservation.payload
                            final_payload = committed_input
                            if native.committed_audio_payload is not None:
                                if final_payload is not None:
                                    final_payload = self._merge_native_audio_payloads(
                                        native.committed_audio_payload,
                                        final_payload,
                                    )
                                else:
                                    final_payload = native.committed_audio_payload
                            commit_reservation.commit()
                            if final_payload is not None:
                                native.retain_committed_audio(
                                    final_payload,
                                    operation_id=commit_reservation.operation_id,
                                    reserved_bytes=commit_reservation.byte_count,
                                )
                            native.deferred_response_create = False
                            native.input_since_commit = False
                            native.speech_since_commit = False
                            data_plane_turn_id = session.turn_id
                            committed = self._commit_native_audio_input(
                                session,
                                realtime_item_id=event.get("realtime_item_id"),
                                transcript=event.get("transcript"),
                                turn_id=data_plane_turn_id,
                            )
                            await emit_event(
                                self._native_audio_committed_payload(
                                    session,
                                    committed=committed,
                                    realtime_item_id=event.get("realtime_item_id"),
                                    transcript=event.get("transcript"),
                                )
                            )
                            if final_payload is not None:
                                await start_native_append(
                                    {
                                        **final_payload,
                                        "duplex_turn_id": data_plane_turn_id,
                                    },
                                    final=True,
                                    precreate_response=False,
                                    operation_id=commit_reservation.operation_id,
                                    retained_committed_payload=final_payload,
                                )
                            continue
                    if self._uses_native_input_append(session) and event_type == "response.create":
                        if (
                            native_response_in_progress()
                            or actor.native_append_tasks
                            or native.data_plane_task is not None
                        ):
                            if session.active_response_id is None and (
                                session.active_request_id is not None
                                or actor.native_append_tasks
                                or native.data_plane_task is not None
                            ):
                                continue
                            await emit_event(
                                {
                                    "type": "error",
                                    "session_id": session.session_id,
                                    "epoch": session.epoch,
                                    "code": "response_already_active",
                                    "error": "response.create cannot start while another response is active.",
                                }
                            )
                            session.discard_response_options()
                            continue
                        if native.committed_audio_payload is not None:
                            committed_payload = native.committed_audio_payload
                            operation_id = native.committed_audio_operation_id
                            if operation_id is None:
                                operation_id = uuid.uuid4().hex
                                native.committed_audio_operation_id = operation_id
                            await start_native_append(
                                committed_payload,
                                final=True,
                                precreate_response=True,
                                operation_id=operation_id,
                                retained_committed_payload=committed_payload,
                            )
                            continue
                        await emit_event(
                            {
                                "type": "error",
                                "session_id": session.session_id,
                                "epoch": session.epoch,
                                "code": "response_create_without_input",
                                "error": "Native duplex response.create requires committed audio input.",
                            }
                        )
                        session.discard_response_options()
                        continue
                    if self._uses_native_input_append(session) and not native_response_in_progress():
                        commit_reservation = (
                            native.audio_buffer.prepare_commit(
                                operation_id=uuid.uuid4().hex,
                                chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
                            )
                            if event_type in {"input.commit", "input_audio_buffer.commit"}
                            else None
                        )
                        flushed_buffer_reserved_bytes = (
                            native.audio_buffer.pending_byte_count if commit_reservation is None else 0
                        )
                        flushed = (
                            commit_reservation.payload
                            if commit_reservation is not None
                            else native.audio_buffer.flush(chunk_period_ms=session.capabilities.chunk_period_ms or 1000)
                        )
                        if native.committed_audio_payload is not None:
                            if flushed is not None:
                                flushed = self._merge_native_audio_payloads(
                                    native.committed_audio_payload,
                                    flushed,
                                )
                            else:
                                flushed = native.committed_audio_payload
                        if commit_reservation is not None:
                            commit_reservation.commit()
                        if flushed is not None:
                            if self._should_force_listen_for_short_commit(session, event, flushed):
                                flushed = dict(flushed)
                                flushed["force_listen"] = True
                            native.input_since_commit = False
                            committed = self._commit_native_audio_input(
                                session,
                                realtime_item_id=event.get("realtime_item_id"),
                                transcript=event.get("transcript"),
                            )
                            await emit_event(
                                self._native_audio_committed_payload(
                                    session,
                                    committed=committed,
                                    realtime_item_id=event.get("realtime_item_id"),
                                    transcript=event.get("transcript"),
                                )
                            )
                            if should_create_response:
                                native.retain_committed_audio(
                                    flushed,
                                    operation_id=(
                                        commit_reservation.operation_id
                                        if commit_reservation is not None
                                        else uuid.uuid4().hex
                                    ),
                                    reserved_bytes=(
                                        commit_reservation.byte_count
                                        if commit_reservation is not None
                                        else flushed_buffer_reserved_bytes
                                    ),
                                )
                                await start_native_append(
                                    flushed,
                                    final=True,
                                    precreate_response=True,
                                    operation_id=native.committed_audio_operation_id,
                                    retained_committed_payload=flushed,
                                )
                            else:
                                native.retain_committed_audio(
                                    flushed,
                                    operation_id=(
                                        commit_reservation.operation_id
                                        if commit_reservation is not None
                                        else uuid.uuid4().hex
                                    ),
                                    reserved_bytes=(
                                        commit_reservation.byte_count
                                        if commit_reservation is not None
                                        else flushed_buffer_reserved_bytes
                                    ),
                                )
                                native.deferred_response_create = False
                            continue
                    native_had_uncommitted_audio = self._uses_native_input_append(session) and (
                        native.input_since_commit
                        or native.audio_buffer.has_pending()
                        or native.committed_audio_payload is not None
                        or realtime_validated_audio_commit
                    )
                    committed = session.commit_user_input()
                    if self._uses_native_input_append(session) and event_type in {
                        "input_audio_buffer.commit",
                        "input.commit",
                    }:
                        native.input_since_commit = False
                        native.speech_since_commit = False
                    if committed is None and event_type != "response.create":
                        if self._uses_native_input_append(session):
                            committed = (
                                self._commit_native_audio_input(
                                    session,
                                    realtime_item_id=event.get("realtime_item_id"),
                                    transcript=event.get("transcript"),
                                )
                                if native_had_uncommitted_audio
                                else None
                            )
                            await emit_event(
                                self._native_audio_committed_payload(
                                    session,
                                    committed=committed,
                                    realtime_item_id=event.get("realtime_item_id"),
                                    transcript=event.get("transcript"),
                                )
                            )
                        else:
                            await emit_event(
                                {
                                    "type": "input.committed",
                                    "session_id": session.session_id,
                                    "empty": True,
                                }
                            )
                        continue
                    if committed is not None:
                        realtime_item_id = event.get("realtime_item_id")
                        if isinstance(realtime_item_id, str):
                            session.register_history_item(realtime_item_id, committed.message)
                        await emit_event(
                            self._input_committed_payload(
                                session,
                                committed,
                                realtime_item_id=realtime_item_id,
                            )
                        )
                    if not should_create_response:
                        continue
                    if actor.active_response_task is not None and not actor.active_response_task.done():
                        await self._cancel_active_response(
                            session,
                            actor.active_response_task,
                            emit_event,
                            reason="new_response",
                        )
                    actor.active_response_task = asyncio.create_task(self._run_response(session, emit_event))
                    continue

                await emit_event(
                    {
                        "type": "error",
                        "error": f"Unknown duplex event: {event_type}",
                        "code": "unknown_event",
                    }
                )

        except WebSocketDisconnect:
            logger.info("Duplex session disconnected")
            transport_detached = True
        except Exception as exc:
            logger.exception("Duplex session failed: %s", exc)
            with suppress(Exception):
                await emit_event({"type": "error", "error": str(exc), "code": "internal_error"})
        finally:
            if session is not None:
                if reader_task is not None and not reader_task.done():
                    reader_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await reader_task
                while pending_turn_reservations > 0:
                    session.release_pending_turn()
                    pending_turn_reservations -= 1
                resumable_detach = (
                    transport_detached
                    and runtime_opened
                    and not runtime_closed
                    and session.state == DuplexSessionState.OPEN
                    and session.capabilities.supports_session_resume
                    and attachment_generation is not None
                    and resume_credential_delivered
                )
                if resumable_detach:

                    async def cancel_orphan_response_after_grace() -> None:
                        tasks = self._session_tasks.get(session.session_id)
                        current_session = self._registry.get(session.session_id)
                        if tasks is None or current_session is not session or session.state != DuplexSessionState.OPEN:
                            return
                        await tasks.cancel_append_tasks(response_bound_only=True)
                        await self._cancel_native_data_plane_stream(session)
                        await self._cancel_active_response(
                            session,
                            tasks.active_response_task,
                            emit_event,
                            reason="disconnect_grace_expired",
                            notify=False,
                        )
                        tasks.active_response_task = None

                    detached = await self._attachment_registry.detach(
                        session.session_id,
                        attachment_generation=attachment_generation,
                        on_grace_expired=cancel_orphan_response_after_grace,
                    )
                    touch_session = getattr(self._chat_service.engine_client, "touch_duplex_session_async", None)
                    if detached and callable(touch_session):
                        with suppress(Exception):
                            await touch_session(
                                session.session_id,
                                fence=DuplexFence(
                                    session.session_id,
                                    epoch=session.epoch,
                                    turn_id=session.turn_id,
                                    incarnation=session.incarnation,
                                ),
                                activity=DuplexLeaseActivity.DETACH,
                            )
                else:
                    begin_close(actor.close_reason or "disconnect")
                    await actor.cancel_append_tasks()
                    await self._cancel_native_data_plane_stream(session)
                    await self._cancel_active_response(
                        session,
                        actor.active_response_task,
                        emit_event,
                        reason="disconnect",
                        notify=False,
                    )
                    if runtime_opened and not runtime_closed and session.state != DuplexSessionState.CLOSED:
                        await self._close_runtime_session(session, reason="disconnect")
                    self._cleanup_duplex_session_state(session)
                    self._registry.close(session.session_id)
                    self._session_tasks.pop(session.session_id, None)
                    self._realtime_protocols.pop(session.session_id, None)
                    self._lease_generations.pop(session.session_id, None)
                    self._resync_required_sessions.discard(session.session_id)
                    with suppress(Exception):
                        await self._attachment_registry.close(session.session_id)
                    self._stop_lifecycle_listener_if_idle()
            await actor.close_writer()
            with suppress(Exception):
                await asyncio.wait_for(actor.output_queue.join(), timeout=2.0)
            if not writer_task.done():
                writer_task.cancel()
            with suppress(asyncio.CancelledError):
                await writer_task
