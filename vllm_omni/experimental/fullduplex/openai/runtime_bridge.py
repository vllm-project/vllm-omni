from __future__ import annotations

import asyncio
import base64
import inspect
from collections.abc import Mapping

import numpy as np
from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.engine.duplex_control_client import DuplexControlRequestError
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import duplex_data_plane_request_info
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexSession,
    DuplexSessionState,
)
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    coerce_int,
    payload_turn_id,
)

logger = init_logger(__name__)


class NativeRuntimeBridgeMixin:
    """Bridge serving sessions to native runtime control and data-plane APIs."""

    _NATIVE_RUNTIME_FENCE_CONTRACT = {
        "open_duplex_session_async": ("fence",),
        "append_duplex_input_async": ("fence",),
        "signal_duplex_turn_async": ("fence", "next_fence"),
        "close_duplex_session_async": ("fence",),
    }

    async def _open_runtime_session(self, session: DuplexSession, send_json) -> dict[str, object] | bool:
        contract_error = self._native_runtime_contract_error(session)
        if contract_error is not None:
            await send_json(
                {
                    "type": "error",
                    "code": "runtime_contract_invalid",
                    "error": contract_error,
                }
            )
            return False
        open_session = getattr(self._chat_service.engine_client, "open_duplex_session_async", None)
        if not callable(open_session):
            return True
        try:
            open_kwargs = {
                "session_mode": "duplex",
                "capabilities": session.capabilities.as_dict(),
                "session_config": session.config.as_dict(),
                "timeout": self._runtime_control_timeout_s(session),
            }
            if self._callable_accepts_keyword(open_session, "runtime_config"):
                open_kwargs["runtime_config"] = dict(session.runtime_config)
            if self._callable_accepts_keyword(open_session, "fence"):
                open_kwargs["fence"] = DuplexFence(
                    session.session_id,
                    epoch=session.epoch,
                    turn_id=session.turn_id,
                    incarnation=session.incarnation,
                )
            result = await open_session(session.session_id, **open_kwargs)
        except Exception as exc:
            if isinstance(exc, DuplexControlRequestError) and exc.code == "resource_exhausted":
                logger.info("Duplex runtime session admission rejected: %s", exc)
            else:
                logger.exception("Failed to open duplex runtime session: %s", exc)
            await self._send_runtime_error(send_json, "runtime_open_failed", exc, session=session)
            return False
        if (
            isinstance(result, dict)
            and session.capabilities.implementation_level == "model_native_duplex"
            and self._runtime_control_failed(result)
        ):
            await send_json(
                {
                    "type": "error",
                    "error": "Native duplex runtime is not available for this session",
                    "code": ("runtime_open_unsupported" if result.get("unsupported_count") else "runtime_open_failed"),
                    "runtime_control": self._redact_runtime_control_result(result),
                }
            )
            return False
        return result if isinstance(result, dict) else True

    async def _append_runtime_input(
        self,
        session: DuplexSession,
        payload: object,
        *,
        operation_id: str | None = None,
        final: bool,
        send_json,
        mode: str = "append_tokens",
        expected_epoch: int | None = None,
    ) -> tuple[bool, bool]:
        if not session.capabilities.supports_input_append:
            return True, False
        append_input = getattr(self._chat_service.engine_client, "append_duplex_input_async", None)
        if not callable(append_input):
            return True, False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return True, False
        try:
            append_kwargs = {
                "mode": mode,
                "payload": payload,
                "final": final,
                "timeout": self._runtime_control_timeout_s(session),
            }
            if operation_id is not None and self._callable_accepts_keyword(append_input, "operation_id"):
                append_kwargs["operation_id"] = operation_id
            if expected_epoch is not None and self._callable_accepts_keyword(append_input, "expected_epoch"):
                append_kwargs["expected_epoch"] = expected_epoch
            if self._callable_accepts_keyword(append_input, "fence"):
                payload_turn = payload_turn_id(payload)
                append_kwargs["fence"] = DuplexFence(
                    session.session_id,
                    epoch=session.epoch,
                    turn_id=(
                        payload_turn
                        if payload_turn is not None
                        else (
                            session.active_response_turn_id
                            if session.active_response_turn_id is not None
                            else session.turn_id
                        )
                    ),
                    incarnation=session.incarnation,
                )
            if self._callable_accepts_keyword(append_input, "collect_outputs"):
                append_kwargs["collect_outputs"] = False
            result = await append_input(session.session_id, **append_kwargs)
        except Exception as exc:
            logger.exception("Failed to append duplex runtime input: %s", exc)
            await self._send_runtime_error(send_json, "runtime_append_failed", exc, session=session)
            return False, False
        if isinstance(result, dict) and self._runtime_control_failed(result):
            await self._send_runtime_control_error(
                send_json,
                "runtime_append_failed",
                "Duplex runtime append failed",
                result,
                session=session,
            )
            return False, False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return True, False
        await self._send_runtime_control_if_needed(send_json, result, session=session)
        request_id, _ = self._data_plane_request_info(result) if isinstance(result, dict) else (None, None)
        if request_id is not None:
            self._serving_runtime_adapter.data_plane.begin_request(request_id)
        close_reason, emitted_response = await self._send_native_duplex_events(
            send_json,
            result,
            session=session,
            expected_epoch=expected_epoch,
        )
        # A resumable data-plane append remains active even when its persistent
        # drain task already exists. Treating only a newly-created drain as
        # activity clears active_request_id after later appends and prevents a
        # terminal TTS segment from scheduling the next model decision.
        emitted_response = emitted_response or request_id is not None
        if close_reason is None and await self._start_native_data_plane_stream_task(
            send_json,
            result,
            session=session,
            expected_epoch=expected_epoch,
        ):
            emitted_response = True
        if close_reason is not None:
            if not await self._close_runtime_session(session, reason=close_reason, send_json=send_json):
                return False, emitted_response
            session.close()
            await send_json(
                {
                    "type": "session.closed",
                    "session_id": session.session_id,
                    "reason": close_reason,
                }
            )
            return False, emitted_response
        return True, emitted_response

    @staticmethod
    def _callable_accepts_keyword(fn, name: str) -> bool:
        try:
            params = inspect.signature(fn).parameters.values()
        except (TypeError, ValueError):
            return True
        return any(param.kind == inspect.Parameter.VAR_KEYWORD or param.name == name for param in params)

    def _native_runtime_contract_error(self, session: DuplexSession) -> str | None:
        if session.capabilities.implementation_level != "model_native_duplex":
            return None
        engine_client = self._chat_service.engine_client
        invalid: list[str] = []
        for method_name, required_keywords in self._NATIVE_RUNTIME_FENCE_CONTRACT.items():
            method = getattr(engine_client, method_name, None)
            if not callable(method):
                invalid.append(f"{method_name} is missing")
                continue
            try:
                parameters = tuple(inspect.signature(method).parameters.values())
            except (TypeError, ValueError):
                invalid.append(f"{method_name} signature is not inspectable")
                continue
            accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
            declared = {parameter.name for parameter in parameters}
            missing = [name for name in required_keywords if not accepts_kwargs and name not in declared]
            if missing:
                invalid.append(f"{method_name} is missing keyword(s): {', '.join(missing)}")
        if not invalid:
            return None
        return "Native duplex runtime requires the fenced control contract: " + "; ".join(invalid)

    async def _start_native_data_plane_stream_task(
        self,
        send_json,
        result: object,
        *,
        session: DuplexSession,
        expected_epoch: int | None = None,
    ) -> bool:
        request_id, _ = self._data_plane_request_info(result)
        if request_id is None or self._data_plane_outputs_finished(result):
            return False
        session.bind_request(request_id)

        native = self._runtime_session_state(session)
        old_task = native.data_plane_task
        if old_task is not None and not old_task.done():
            if self._session_auto_responds(session):
                # One persistent drain per session, like the official worker's
                # single synchronous loop: the resumable data-plane request id
                # is stable, and cancel/restart on every append orphans any
                # decision that lands in the swap window.
                native.data_plane_restart_requested = True
                return False
            old_task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(old_task, return_exceptions=True), timeout=0.25)
            except asyncio.TimeoutError:
                pass

        async def _run() -> None:
            close_reason: str | None = None
            try:
                close_reason = await self._drain_native_data_plane_stream(
                    send_json,
                    result,
                    session=session,
                    expected_epoch=expected_epoch,
                )
                if close_reason is not None:
                    if await self._close_runtime_session(session, reason=close_reason, send_json=send_json):
                        session.close()
                        await send_json(
                            {
                                "type": "session.closed",
                                "session_id": session.session_id,
                                "reason": close_reason,
                            }
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Failed to drain duplex data-plane stream: %s", exc)
                await self._send_runtime_error(send_json, "runtime_data_plane_stream_failed", exc, session=session)
                if session.state != DuplexSessionState.CLOSED:
                    close_reason = "runtime_data_plane_stream_failed"
                    if await self._close_runtime_session(session, reason=close_reason, send_json=send_json):
                        session.close()
                        await send_json(
                            {
                                "type": "session.closed",
                                "session_id": session.session_id,
                                "reason": close_reason,
                            }
                        )
            finally:
                if native.data_plane_task is task:
                    native.data_plane_task = None
                restart_requested = (
                    native.data_plane_restart_requested
                    and close_reason is None
                    and session.state != DuplexSessionState.CLOSED
                    and session.active_request_id == request_id
                    and (expected_epoch is None or session.epoch == expected_epoch)
                )
                native.data_plane_restart_requested = False
                if close_reason is None and not self._session_auto_responds(session):
                    # Auto-respond sessions keep one resumable stage-1 stream
                    # whose audio accumulates across speak units; the offset
                    # must survive drain-task turnover or every unit re-sends
                    # the reply audio from the start.
                    self._serving_runtime_adapter.data_plane.close_stream(request_id)
            if restart_requested:
                await self._start_native_data_plane_stream_task(
                    send_json,
                    result,
                    session=session,
                    expected_epoch=expected_epoch,
                )
            elif close_reason is None:
                await self._maybe_continue_native_response(send_json, session=session, expected_epoch=expected_epoch)

        task = asyncio.create_task(_run())
        native.data_plane_task = task
        return True

    # One model unit (1 s at 16 kHz) of pcm_f32le silence, matching the
    # official full-duplex behavior where the microphone keeps streaming
    # silence while the assistant speaks; replies span multiple units.
    _NATIVE_SILENCE_UNIT_PAYLOAD_AUDIO = base64.b64encode(bytes(16000 * 4)).decode("ascii")
    _NATIVE_RESPONSE_MAX_CONTINUATION_UNITS = 8

    def _native_silence_unit_payload(self) -> dict[str, object]:
        return {
            "type": "audio",
            "audio": self._NATIVE_SILENCE_UNIT_PAYLOAD_AUDIO,
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        }

    def _native_response_continuations_remaining(self, session: DuplexSession, response_id: str) -> bool:
        if self._session_auto_responds(session):
            return True
        native = self._runtime_session_state(session)
        owner_id = f"response:{response_id}"
        count = native.continuation_units if native.continuation_owner_id == owner_id else 0
        return count < self._NATIVE_RESPONSE_MAX_CONTINUATION_UNITS

    def _native_silence_continuation_is_stale(
        self,
        session: DuplexSession,
        *,
        request_id: str,
        response_id: str | None,
        response_owned: bool,
        expected_epoch: int | None,
        expected_incarnation: int,
        expected_model_turn_id: int | None,
    ) -> bool:
        stale_common_owner = (
            session.state == DuplexSessionState.CLOSED
            or session.incarnation != expected_incarnation
            or session.active_request_id != request_id
            or (expected_epoch is not None and session.epoch != expected_epoch)
        )
        stale_response_owner = response_owned and session.active_response_id != response_id
        stale_model_turn_owner = not response_owned and (
            session.active_response_id is not None or session.turn_id != expected_model_turn_id
        )
        return stale_common_owner or stale_response_owner or stale_model_turn_owner

    async def _maybe_continue_native_response(
        self,
        send_json,
        *,
        session: DuplexSession,
        expected_epoch: int | None,
        expected_model_turn_id: int | None = None,
    ) -> None:
        """Give an active model turn another silence unit.

        A model turn can need another unit before it has emitted visible
        text/audio and therefore before a Realtime response exists. Bind that
        continuation to the model-turn identity; once a response exists, its
        response/turn ownership remains the continuation fence.
        """
        response_id = session.active_response_id
        native = self._runtime_session_state(session)
        if session.state == DuplexSessionState.CLOSED:
            native.clear_continuation()
            return
        request_id = session.active_request_id
        if request_id is None:
            native.clear_continuation()
            return
        if expected_epoch is not None and session.epoch != expected_epoch:
            return
        auto_response = self._session_auto_responds(session)
        response_owned = response_id is not None
        if response_owned:
            owner_id = f"response:{response_id}"
            payload_turn_id = (
                session.active_response_turn_id if session.active_response_turn_id is not None else session.turn_id
            )
        else:
            if not auto_response or expected_model_turn_id is None or session.turn_id != expected_model_turn_id:
                native.clear_continuation()
                return
            owner_id = f"model-turn:{expected_model_turn_id}"
            payload_turn_id = expected_model_turn_id
        count = native.continuation_units if native.continuation_owner_id == owner_id else 0
        if not auto_response and count >= self._NATIVE_RESPONSE_MAX_CONTINUATION_UNITS:
            return
        payload = self._native_silence_unit_payload()
        payload["duplex_turn_id"] = payload_turn_id

        scheduler = native.silence_continuation_scheduler
        if scheduler is None:
            return
        try:
            scheduled = await scheduler(
                payload,
                request_id=request_id,
                owner_id=owner_id,
                response_id=response_id,
                response_owned=response_owned,
                expected_epoch=expected_epoch,
                expected_incarnation=session.incarnation,
                expected_model_turn_id=expected_model_turn_id,
                send_json=send_json,
            )
        except Exception as exc:
            logger.exception("Failed to schedule duplex native response continuation: %s", exc)
            scheduled = False
        if scheduled:
            native.continuation_owner_id = owner_id
            native.continuation_units = count + 1

    async def _cancel_native_data_plane_stream(self, session: DuplexSession) -> bool:
        native = self._runtime_session_state(session)
        task = native.data_plane_task
        native.data_plane_task = None
        native.data_plane_restart_requested = False
        if task is None or task.done():
            return False
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=0.25)
        except asyncio.TimeoutError:
            # Keep barge-in/cancel responsive. The task still carries the old
            # expected_epoch and late model output is filtered by the writer.
            pass
        return True

    async def _signal_runtime_session(
        self,
        session: DuplexSession,
        event: str,
        send_json=None,
        *,
        fence: DuplexFence | None = None,
        next_fence: DuplexFence | None = None,
        session_config: dict[str, object] | None = None,
        runtime_config: dict[str, object] | None = None,
    ) -> bool:
        signal_turn = getattr(self._chat_service.engine_client, "signal_duplex_turn_async", None)
        if not callable(signal_turn):
            return True
        try:
            signal_kwargs = {
                "event": event,
                "timeout": self._runtime_control_timeout_s(session),
            }
            if self._callable_accepts_keyword(signal_turn, "fence"):
                signal_kwargs["fence"] = fence or DuplexFence(
                    session.session_id,
                    epoch=session.epoch,
                    turn_id=session.turn_id,
                    incarnation=session.incarnation,
                )
            if next_fence is not None and self._callable_accepts_keyword(signal_turn, "next_fence"):
                signal_kwargs["next_fence"] = next_fence
            if session_config is not None and self._callable_accepts_keyword(signal_turn, "session_config"):
                signal_kwargs["session_config"] = session_config
            if runtime_config is not None and self._callable_accepts_keyword(signal_turn, "runtime_config"):
                signal_kwargs["runtime_config"] = runtime_config
            result = await signal_turn(session.session_id, **signal_kwargs)
        except Exception as exc:
            logger.exception("Failed to signal duplex runtime session: %s", exc)
            if send_json is not None:
                await self._send_runtime_error(send_json, "runtime_signal_failed", exc, session=session)
            return False
        if isinstance(result, dict) and self._runtime_signal_failed(result):
            logger.warning(
                "Duplex runtime signal failed: session=%s event=%s result=%s",
                session.session_id,
                event,
                self._redact_runtime_control_result(result),
            )
            if send_json is not None:
                await self._send_runtime_control_error(
                    send_json,
                    "runtime_signal_failed",
                    "Duplex runtime signal failed",
                    result,
                    session=session,
                )
            return False
        await self._send_runtime_control_if_needed(send_json, result, session=session)
        return True

    async def _close_runtime_session(self, session: DuplexSession, *, reason: str, send_json=None) -> bool:
        close_session = getattr(self._chat_service.engine_client, "close_duplex_session_async", None)
        if not callable(close_session):
            return True
        try:
            close_kwargs = {
                "reason": reason,
                "timeout": self._runtime_control_timeout_s(session),
            }
            if self._callable_accepts_keyword(close_session, "fence"):
                close_kwargs["fence"] = DuplexFence(
                    session.session_id,
                    epoch=session.epoch,
                    turn_id=session.turn_id,
                    incarnation=session.incarnation,
                )
            result = await close_session(session.session_id, **close_kwargs)
        except Exception as exc:
            logger.exception("Failed to close duplex runtime session: %s", exc)
            if send_json is not None:
                await self._send_runtime_error(send_json, "runtime_close_failed", exc, session=session)
            return False
        if isinstance(result, dict) and self._runtime_control_failed(result):
            if send_json is not None:
                await self._send_runtime_control_error(
                    send_json,
                    "runtime_close_failed",
                    "Duplex runtime close failed",
                    result,
                    session=session,
                )
            return False
        await self._send_runtime_control_if_needed(send_json, result, session=session)
        return True

    @staticmethod
    def _runtime_control_timeout_s(session: DuplexSession) -> float:
        raw = session.config.extra_body.get("duplex_control_timeout_s") or session.config.extra_body.get(
            "runtime_control_timeout_s"
        )
        if isinstance(raw, int | float) and raw > 0:
            return float(raw)
        if session.capabilities.implementation_level == "model_native_duplex":
            return 60.0
        return 10.0

    async def _send_runtime_control_if_needed(
        self,
        send_json,
        result: object,
        *,
        session: DuplexSession,
    ) -> None:
        if send_json is None:
            return
        if not isinstance(result, dict):
            return
        if not result.get("unsupported_count") and not result.get("error_count"):
            return
        await send_json(
            {
                "type": "runtime.control",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "result": self._redact_runtime_control_result(result),
            }
        )

    @staticmethod
    def _runtime_control_failed(result: dict[str, object]) -> bool:
        if result.get("ok") is False:
            return True
        for key in ("unsupported_count", "error_count"):
            value = result.get(key)
            if isinstance(value, int | float) and value > 0:
                return True
        return False

    @classmethod
    def _runtime_signal_failed(cls, result: dict[str, object]) -> bool:
        error_count = result.get("error_count")
        if isinstance(error_count, int | float) and error_count > 0:
            return True
        if result.get("ok") is not False:
            return False
        return not cls._runtime_signal_has_data_plane_ack(result)

    @classmethod
    def _runtime_signal_has_data_plane_ack(cls, value: object) -> bool:
        if isinstance(value, dict):
            if value.get("data_plane_signal") is True and value.get("supported") is True:
                return True
            return any(cls._runtime_signal_has_data_plane_ack(child) for child in value.values())
        if isinstance(value, list | tuple):
            return any(cls._runtime_signal_has_data_plane_ack(item) for item in value)
        return False

    @classmethod
    def _redact_runtime_control_result(cls, value: object) -> object:
        if isinstance(value, dict):
            redacted = {
                key: cls._redact_runtime_control_result(child)
                for key, child in value.items()
                if key
                not in {
                    "stage_handoff",
                    "tts_handoff",
                    "omni_payload",
                    "tts_hidden_states",
                    "tts_token_ids",
                    "traceback",
                }
            }
            native_result = redacted.get("native_result")
            if isinstance(native_result, dict):
                if native_result.get("requires_stage_handoff") is True:
                    native_result.pop("requires_stage_handoff", None)
                if native_result.get("requires_tts_stage") is True:
                    native_result.pop("requires_tts_stage", None)
            return redacted
        if isinstance(value, list | tuple):
            return [cls._redact_runtime_control_result(item) for item in value]
        return value

    async def _send_native_duplex_events(
        self,
        send_json,
        result: object,
        *,
        session: DuplexSession,
        expected_epoch: int | None = None,
    ) -> tuple[str | None, bool]:
        if send_json is None:
            return None, False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return None, False
        close_reason: str | None = None
        emitted_response = False
        request_id, _ = self._data_plane_request_info(result)
        if self._serving_runtime_adapter.data_plane.is_terminal(request_id):
            return None, False
        if request_id is not None and session.active_request_id is None:
            session.bind_request(request_id)
        context = self._runtime_data_plane_context(session)
        for native_result in self._serving_runtime_adapter.data_plane.project(result, context=context):
            close_reason_for_result, did_emit = await self._send_one_native_duplex_event(
                send_json,
                native_result,
                session=session,
                expected_epoch=expected_epoch,
            )
            emitted_response = emitted_response or did_emit
            close_reason = close_reason or close_reason_for_result
            if expected_epoch is not None and session.epoch != expected_epoch:
                return None, emitted_response
        return close_reason, emitted_response

    async def _drain_native_data_plane_stream(
        self,
        send_json,
        result: object,
        *,
        session: DuplexSession,
        expected_epoch: int | None = None,
    ) -> str | None:
        request_id, response_stage_id = self._data_plane_request_info(result)
        if request_id is None:
            return None
        if self._data_plane_outputs_finished(result):
            return None
        collect_outputs = getattr(
            self._chat_service.engine_client,
            "collect_duplex_data_plane_outputs_async",
            None,
        )
        if not callable(collect_outputs):
            return None

        close_reason: str | None = None
        empty_polls = 0
        while close_reason is None:
            if self._serving_runtime_adapter.data_plane.is_terminal(request_id):
                return None
            if self._session_auto_responds(session):
                active_request_id = session.active_request_id
                if active_request_id is not None and active_request_id != request_id:
                    return None
            if expected_epoch is not None and session.epoch != expected_epoch:
                return None
            if session.state == DuplexSessionState.CLOSED:
                return None
            outputs = await collect_outputs(
                request_id,
                response_stage_id=response_stage_id,
                timeout=self._runtime_control_timeout_s(session),
            )
            if expected_epoch is not None and session.epoch != expected_epoch:
                return None
            if not outputs:
                # An empty poll means no output arrived within one control
                # window, not that the stream is over. Exiting here orphans a
                # decision that lands moments later: it would sit queued until
                # the NEXT append starts a fresh drain task, adding one full
                # chunk of latency to every model decision.
                empty_polls += 1
                if empty_polls >= 3:
                    return None
                continue
            empty_polls = 0
            drain_result = {
                "data_plane_outputs": outputs,
            }
            close_reason, emitted_response = await self._send_native_duplex_events(
                send_json,
                drain_result,
                session=session,
                expected_epoch=expected_epoch,
            )
            if (
                self._data_plane_outputs_finished(drain_result)
                and emitted_response
                and not self._session_auto_responds(session)
            ):
                # Auto-respond sessions are resumable: every segment ends
                # with finished=True but the stream continues with the next
                # audio chunk. Exiting here would make each decision wait for
                # the next append to start a fresh drain task (one full chunk
                # of added latency); keep draining until epoch change,
                # session close, or cancellation by a newer drain task.
                return close_reason
            # A batch without a client-visible event (e.g. the generic
            # stage-0 final-output message that precedes the listen/speak
            # decision in the same segment) must not terminate the drain;
            # the real decision is still in flight.
        return close_reason

    @staticmethod
    def _data_plane_outputs_finished(result: object) -> bool:
        if not isinstance(result, dict):
            return False
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list) or not outputs:
            return False
        return bool(getattr(outputs[-1], "finished", False))

    @staticmethod
    def _data_plane_request_info(result: object) -> tuple[str | None, int | None]:
        if not isinstance(result, dict):
            return None, None
        return duplex_data_plane_request_info(result)

    def _runtime_data_plane_context(self, session: DuplexSession) -> object:
        response_config = session.response_config
        return self._serving_runtime_adapter.data_plane_context(
            epoch=session.epoch,
            turn_id=session.turn_id,
            active_response_turn_id=session.active_response_turn_id,
            active_response_id=session.active_response_id,
            auto_responds=self._session_auto_responds(session),
            response_format=response_config.response_format,
            speed=response_config.speed,
            modalities=tuple(response_config.modalities),
        )

    async def _send_one_native_duplex_event(
        self,
        send_json,
        native_result: dict[str, object],
        *,
        session: DuplexSession,
        expected_epoch: int | None = None,
    ) -> tuple[str | None, bool]:
        close_reason: str | None = None
        emitted_response = False
        if expected_epoch is not None and session.epoch != expected_epoch:
            return close_reason, emitted_response
        data_plane_request_id = native_result.get("data_plane_request_id")
        if isinstance(data_plane_request_id, str) and self._serving_runtime_adapter.data_plane.is_terminal(
            data_plane_request_id
        ):
            return close_reason, emitted_response
        active_request_matches = session.active_request_id == data_plane_request_id or (
            self._session_auto_responds(session) and session.active_request_id is None
        )
        if isinstance(data_plane_request_id, str) and not active_request_matches:
            return close_reason, emitted_response
        if isinstance(native_result.get("error_code"), str):
            response_id = session.active_response_id
            await send_json(
                {
                    "type": "error",
                    "code": native_result.get("error_code"),
                    "session_id": session.session_id,
                    "epoch": session.epoch,
                    "error": str(native_result.get("error") or "Duplex native data-plane error"),
                }
            )
            if response_id is not None:
                session.end_response(commit_text=False)
                await send_json(
                    {
                        "type": "response.done",
                        "session_id": session.session_id,
                        "response_id": response_id,
                        "epoch": session.epoch,
                        "committed": False,
                        "status": "failed",
                        "status_details": {
                            "type": "failed",
                            "reason": native_result.get("error_code"),
                        },
                        "playback": session.playback.as_dict(),
                    }
                )
            return close_reason, True
        if native_result.get("requires_stage_handoff") is True or native_result.get("requires_tts_stage") is True:
            # Stage0 announces a talker handoff before Stage1 has produced
            # client-visible audio. A single client input stream may contain
            # several model turns, including terminal-only turns. Reserve the
            # protocol response only when Stage1 emits text/audio so empty
            # model turns cannot create empty responses or steal later audio.
            return close_reason, False
        is_listen = native_result.get("is_listen")
        model_turn_id = coerce_int(native_result.get("model_turn_id"))
        if native_result.get("is_buffering") is True or native_result.get("prefill_success") is False:
            if native_result.get("data_plane_request_id") == session.active_request_id:
                session.clear_request()
            payload = {
                "type": "response.listen",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "reason": native_result.get("reason") or "buffering",
                "model_listen": False,
                "buffering": True,
            }
            self._attach_native_runtime_metadata(payload, native_result)
            await send_json(payload)
            return close_reason, emitted_response
        if is_listen is True:
            await self._end_active_response_before_future_model_turn(
                send_json,
                session=session,
                model_turn_id=model_turn_id,
            )
            if (
                session.active_response_id is not None
                and model_turn_id is not None
                and not session.active_response_accepts_model_turn(model_turn_id)
            ):
                return close_reason, emitted_response
            non_terminal_auto_listen = (
                self._session_auto_responds(session)
                and session.active_response_id is not None
                and session.active_request_id is not None
                and native_result.get("end_of_turn") is not True
            )
            if non_terminal_auto_listen:
                await self._maybe_continue_native_response(
                    send_json,
                    session=session,
                    expected_epoch=expected_epoch,
                )
                return close_reason, emitted_response
            auto_response = self._session_auto_responds(session)
            if not auto_response and data_plane_request_id == session.active_request_id:
                session.clear_request()
            model_listen = native_result.get("model_listen")
            if not isinstance(model_listen, bool):
                model_listen = native_result.get("reason") in {None, "", "model_listen"}
            response_id = session.active_response_id
            if isinstance(data_plane_request_id, str) and not auto_response:
                self._serving_runtime_adapter.data_plane.mark_terminal(data_plane_request_id)
            emitted_response = True
            payload = {
                "type": "response.listen",
                "session_id": session.session_id,
                "epoch": session.epoch,
                "reason": native_result.get("reason") or "model_listen",
                "model_listen": model_listen,
            }
            self._attach_native_runtime_metadata(payload, native_result)
            await send_json(payload)
            if native_result.get("abort_data_plane_request") is True and isinstance(data_plane_request_id, str):
                await self._abort_request_background(
                    session,
                    data_plane_request_id,
                    send_json,
                    notify=False,
                )
            if response_id is not None:
                if not self._session_auto_responds(session) and self._native_response_continuations_remaining(
                    session, response_id
                ):
                    # The official model often listens for a silence beat or
                    # two before answering; keep the response open and give it
                    # the next silence unit as a decision point.
                    await self._maybe_continue_native_response(
                        send_json,
                        session=session,
                        expected_epoch=expected_epoch,
                    )
                    return close_reason, emitted_response
                session.end_response(commit_text=False, preserve_request=auto_response)
                await send_json(
                    {
                        "type": "response.done",
                        "session_id": session.session_id,
                        "response_id": response_id,
                        "epoch": session.epoch,
                        "committed": False,
                        "playback": session.playback.as_dict(),
                    }
                )
            return close_reason, emitted_response

        text = native_result.get("text")
        audio = native_result.get("audio_data", native_result.get("audio"))
        end_of_turn = bool(native_result.get("end_of_turn", False))
        has_text = isinstance(text, str) and bool(text)
        has_audio = isinstance(audio, str) and bool(audio)
        if not has_text and not has_audio and not end_of_turn:
            tts_segment_ended = (
                native_result.get("stage_role") == "tts" and native_result.get("abort_data_plane_request") is True
            )
            if (
                tts_segment_ended
                and self._session_auto_responds(session)
                and (model_turn_id is not None or session.active_response_id is not None)
            ):
                await self._maybe_continue_native_response(
                    send_json,
                    session=session,
                    expected_epoch=expected_epoch,
                    expected_model_turn_id=model_turn_id,
                )
            return close_reason, emitted_response
        if end_of_turn and not has_text and not has_audio and session.active_response_id is None:
            if isinstance(data_plane_request_id, str):
                if not self._session_auto_responds(session) and data_plane_request_id == session.active_request_id:
                    session.clear_request()
                if not self._session_auto_responds(session):
                    self._serving_runtime_adapter.data_plane.mark_terminal(data_plane_request_id)
            if model_turn_id is not None:
                session.complete_model_turn(model_turn_id)
            if self._session_auto_responds(session):
                self._runtime_session_state(session).clear_continuation()
                emitted_response = True
                payload = {
                    "type": "response.listen",
                    "session_id": session.session_id,
                    "epoch": session.epoch,
                    "reason": "model_turn_completed_without_output",
                    "model_listen": True,
                }
                self._attach_native_runtime_metadata(payload, native_result)
                await send_json(payload)
            return close_reason, emitted_response
        if session.active_response_id is None and model_turn_id is not None and model_turn_id < session.turn_id:
            # A continuation append can already be in flight when the prior
            # response reaches turn EOS.  Its late audio still carries the
            # completed model turn, so it must not reserve a second Realtime
            # response for that turn.
            return close_reason, emitted_response
        await self._end_active_response_before_future_model_turn(
            send_json,
            session=session,
            model_turn_id=model_turn_id,
        )
        if (
            session.active_response_id is not None
            and model_turn_id is not None
            and not session.active_response_accepts_model_turn(model_turn_id)
        ):
            return close_reason, emitted_response
        emitted_response = True
        response_created = False
        response_id = session.active_response_id
        if response_id is None:
            response_id = session.begin_response(turn_id=model_turn_id)
            response_created = True
            await send_json(
                self._response_created_payload(
                    session,
                    response_id,
                    epoch=session.epoch,
                )
            )
        response_stage_metrics = session.accumulate_response_stage_metrics(
            native_result.get("stage_metrics") if isinstance(native_result.get("stage_metrics"), Mapping) else None
        )
        if response_created:
            speak_payload = {
                "type": "response.speak",
                "session_id": session.session_id,
                "response_id": response_id,
                "epoch": session.epoch,
                "text": text if isinstance(text, str) else "",
                "end_of_turn": end_of_turn,
                "model_speak": True,
            }
            self._attach_native_runtime_metadata(
                speak_payload,
                native_result,
                stage_metrics=response_stage_metrics,
            )
            await send_json(speak_payload)
        previous_sent_ms = session.playback.sent_ms
        text_chars_before_append = len("".join(session.assistant_text_buffer))
        if isinstance(text, str):
            session.append_assistant_text(text)
        duration_ms = native_result.get("audio_duration_ms")
        text_chars = len("".join(session.assistant_text_buffer))
        mark_duration_ms = None
        mark_text_chars: int | None = text_chars
        if native_result.get("audio_text_mark") is False:
            mark_text_chars = None
        if isinstance(duration_ms, int | float):
            mark_duration_ms = int(duration_ms)
            if native_result.get("audio_duration_is_cumulative") is not True:
                mark_duration_ms += session.playback.sent_ms
        audio_text_marks = native_result.get("audio_text_marks")
        audio_text_marks = self._normalize_native_audio_text_marks(
            audio_text_marks if isinstance(audio_text_marks, list) else None,
            audio_offset_ms=(
                0
                if native_result.get("audio_text_marks_are_cumulative") is True
                or native_result.get("audio_duration_is_cumulative") is True
                else previous_sent_ms
            ),
            text_offset_chars=(
                0 if native_result.get("audio_text_marks_are_cumulative") is True else text_chars_before_append
            ),
        )
        session.mark_audio_sent(
            mark_duration_ms,
            text_chars=mark_text_chars if mark_duration_ms is not None else None,
            audio_text_marks=audio_text_marks,
        )
        payload = {
            "type": "response.output_audio.delta",
            "session_id": session.session_id,
            "response_id": response_id,
            "epoch": session.epoch,
            "text": text if isinstance(text, str) else "",
            "audio": audio if isinstance(audio, str) else "",
            "format": (
                native_result.get("audio_format")
                if isinstance(native_result.get("audio_format"), str)
                else session.response_config.response_format
            ),
            "end_of_turn": end_of_turn,
            "model_speak": True,
        }
        if mark_duration_ms is not None:
            payload["audio_duration_ms"] = mark_duration_ms
        if audio_text_marks:
            payload["audio_text_marks"] = audio_text_marks
        elif mark_duration_ms is not None and mark_text_chars is not None:
            payload["audio_text_marks"] = [
                {
                    "text_chars": max(0, int(mark_text_chars)),
                    "audio_end_ms": max(0, int(mark_duration_ms)),
                }
            ]
        payload["playback"] = session.playback.as_dict()
        sample_rate_hz = native_result.get("sample_rate_hz") or native_result.get("audio_sample_rate_hz")
        if isinstance(sample_rate_hz, int | float) and int(sample_rate_hz) > 0:
            payload["sample_rate_hz"] = int(sample_rate_hz)
        self._attach_native_runtime_metadata(
            payload,
            native_result,
            stage_metrics=response_stage_metrics,
        )
        await send_json(payload)
        if (
            not end_of_turn
            and native_result.get("stage_role") == "tts"
            and native_result.get("abort_data_plane_request") is True
            and self._session_auto_responds(session)
        ):
            await self._maybe_continue_native_response(
                send_json,
                session=session,
                expected_epoch=expected_epoch,
            )
        if end_of_turn:
            data_plane_request_id = native_result.get("data_plane_request_id")
            if isinstance(data_plane_request_id, str) and not self._session_auto_responds(session):
                self._serving_runtime_adapter.data_plane.close_stream(data_plane_request_id)
            if isinstance(data_plane_request_id, str):
                if not self._session_auto_responds(session) and data_plane_request_id == session.active_request_id:
                    session.clear_request()
                if not self._session_auto_responds(session):
                    self._serving_runtime_adapter.data_plane.mark_terminal(data_plane_request_id)
            should_commit = self._should_commit_response_to_history(session, response_id)
            committed_message = session.end_response(
                commit_text=should_commit,
                preserve_request=self._session_auto_responds(session),
            )
            model_turn_id = coerce_int(native_result.get("model_turn_id"))
            if model_turn_id is not None:
                session.complete_model_turn(model_turn_id)
            if should_commit:
                session.register_history_item(f"item_{response_id}", committed_message)
            await send_json(
                {
                    "type": "response.done",
                    "session_id": session.session_id,
                    "response_id": response_id,
                    "epoch": session.epoch,
                    "committed": committed_message is not None,
                    "playback": session.playback.as_dict(),
                }
            )
        return close_reason, emitted_response

    async def _end_active_response_before_future_model_turn(
        self,
        send_json,
        *,
        session: DuplexSession,
        model_turn_id: int | None,
    ) -> None:
        if not self._session_auto_responds(session):
            return
        response_id = session.active_response_id
        active_turn_id = session.active_response_turn_id
        if response_id is None or model_turn_id is None or active_turn_id is None:
            return
        if int(model_turn_id) <= int(active_turn_id):
            return
        session.complete_model_turn(int(model_turn_id) - 1)
        should_commit = self._should_commit_response_to_history(session, response_id)
        committed_message = session.end_response(
            commit_text=should_commit,
            preserve_request=True,
        )
        if should_commit:
            session.register_history_item(f"item_{response_id}", committed_message)
        await send_json(
            {
                "type": "response.done",
                "session_id": session.session_id,
                "response_id": response_id,
                "epoch": session.epoch,
                "committed": committed_message is not None,
                "playback": session.playback.as_dict(),
            }
        )

    @staticmethod
    def _normalize_native_audio_text_marks(
        audio_text_marks: list[object] | None,
        *,
        audio_offset_ms: int,
        text_offset_chars: int,
    ) -> list[dict[str, int]] | None:
        if not audio_text_marks:
            return None
        normalized: list[dict[str, int]] = []
        for raw_mark in audio_text_marks:
            if not isinstance(raw_mark, dict):
                continue
            raw_text_chars = raw_mark.get("text_chars")
            raw_audio_end_ms = raw_mark.get("audio_end_ms", raw_mark.get("audio_ms"))
            if not isinstance(raw_text_chars, int | float) or not isinstance(raw_audio_end_ms, int | float):
                continue
            normalized.append(
                {
                    "text_chars": max(0, int(raw_text_chars) + int(text_offset_chars)),
                    "audio_end_ms": max(0, int(raw_audio_end_ms) + int(audio_offset_ms)),
                }
            )
        return normalized or None

    def _cleanup_duplex_session_state(self, session: DuplexSession) -> None:
        session_id = session.session_id
        self._serving_runtime_adapter.remove_session_state(session_id)
        self._serving_runtime_adapter.data_plane.close_session(
            session_id,
            active_request_id=session.active_request_id,
        )

    def _encode_native_data_plane_audio(
        self,
        audio_data: object,
        sample_rate_hz: int,
        response_format: str,
        speed: float | None,
    ) -> str | None:
        if audio_data is None:
            return None
        try:
            import torch

            from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio

            if isinstance(audio_data, torch.Tensor):
                audio_tensor = audio_data.detach().cpu().float().numpy()
            else:
                audio_tensor = np.asarray(audio_data, dtype=np.float32)
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.reshape(-1)
            audio_response = self._chat_service.create_audio(
                CreateAudio(
                    audio_tensor=audio_tensor,
                    sample_rate=sample_rate_hz,
                    response_format=response_format,
                    speed=float(speed) if isinstance(speed, int | float) and speed > 0 else 1.0,
                    stream_format="audio",
                    base64_encode=True,
                )
            )
            return str(audio_response.audio_data)
        except Exception:
            logger.exception("Failed to encode duplex data-plane audio output")
            return None

    @staticmethod
    def _attach_native_runtime_metadata(
        payload: dict[str, object],
        native_result: dict[str, object],
        *,
        stage_metrics: Mapping[str, object] | None = None,
    ) -> None:
        metadata: dict[str, object] = {}
        runtime_impl = native_result.get("runtime_impl")
        if isinstance(runtime_impl, str) and runtime_impl:
            metadata["runtime_impl"] = runtime_impl
        owned_runtime = native_result.get("owned_runtime")
        if isinstance(owned_runtime, bool):
            metadata["owned_runtime"] = owned_runtime
        model_turn_id = coerce_int(native_result.get("model_turn_id"))
        if model_turn_id is not None:
            metadata["model_turn_id"] = model_turn_id
        for name in (
            "uses_model_runner_scheduler",
            "runner_kv_backed",
        ):
            value = native_result.get(name)
            if isinstance(value, bool):
                metadata[name] = value
        effective_stage_metrics = stage_metrics if stage_metrics is not None else native_result.get("stage_metrics")
        if isinstance(effective_stage_metrics, Mapping):
            metadata["stage_metrics"] = {
                str(stage_id): dict(values)
                for stage_id, values in effective_stage_metrics.items()
                if isinstance(values, Mapping)
            }
        if metadata:
            payload["vllm_omni"] = metadata

    async def _send_runtime_error(
        self,
        send_json,
        code: str,
        exc: Exception,
        *,
        session: DuplexSession | None = None,
    ) -> None:
        retryable = False
        message = str(exc)
        if isinstance(exc, DuplexControlRequestError):
            code = exc.code
            retryable = exc.retryable
            error_data = exc.result.get("error")
            if isinstance(error_data, dict) and isinstance(error_data.get("message"), str):
                message = error_data["message"]
        payload: dict[str, object] = {
            "type": "error",
            "code": code,
            "error": message,
            "retryable": retryable,
        }
        if session is not None:
            payload["session_id"] = session.session_id
            payload["epoch"] = session.epoch
        await send_json(payload)

    async def _send_runtime_control_error(
        self,
        send_json,
        code: str,
        message: str,
        result: dict[str, object],
        *,
        session: DuplexSession,
    ) -> None:
        await send_json(
            {
                "type": "error",
                "code": code,
                "error": message,
                "session_id": session.session_id,
                "epoch": session.epoch,
                "runtime_control": self._redact_runtime_control_result(result),
            }
        )
