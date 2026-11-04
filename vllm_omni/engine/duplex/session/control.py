# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Control of one duplex session: server/semantic VAD, and the events that reconfigure it.

Server turn detection and ``session.update`` are the same concern seen from two
sides. The detector is per-session mutable state, and ``session.update`` is the
event that replaces it mid-flight -- along with the session config, the runtime
config and the sampling policy, each of which has to be validated against a
candidate before the live session adopts it. The rest of ``turn.signal``
(conversation items, local turn markers) travels the same path and is small, so
it lives here rather than in a module of its own.

The one thing this does not own is ordering against the data plane: a
``session.update`` must not overtake an append still in flight, so the runner
supplies ``wait_for_append_tail``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy

from vllm.logger import init_logger

from vllm_omni.engine.duplex.config import DuplexConfigError, realtime_item_to_history_message
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.session import helpers
from vllm_omni.engine.duplex.session.context import DuplexSessionContext
from vllm_omni.engine.duplex.session.emitter import SessionEmitter, session_auto_responds
from vllm_omni.engine.duplex.session.lease import DuplexLeaseActivity
from vllm_omni.engine.duplex.session.model_channel import ModelChannel
from vllm_omni.engine.duplex.session.tool_calls import ToolCallRejection, ToolCallState
from vllm_omni.engine.duplex.turn_detection import (
    PendingTurnDetectionUpdate,
    SemanticVADUnavailableError,
    ServerVADUnavailableError,
    SileroVADBackendProvider,
    SmartTurnBackendProvider,
    TurnDetectionConfig,
    TurnDetectionResult,
    TurnDetector,
    apply_turn_detection_result,
)
from vllm_omni.protocol.duplex.events import error_event

logger = init_logger(__name__)


class SessionControl:
    """Server VAD and the control events that reconfigure one session."""

    def __init__(
        self,
        ctx: DuplexSessionContext,
        out: SessionEmitter,
        model: ModelChannel,
        *,
        wait_for_append_tail: Callable[[], Awaitable[bool]],
    ) -> None:
        self._ctx = ctx
        self._out = out
        self._model = model
        self._wait_for_append_tail = wait_for_append_tail
        self._config: TurnDetectionConfig | None = None
        self._detector: TurnDetector | None = None

    def init_turn_detection(self) -> None:
        turn_detection = self._ctx.session.config.extra_body.get("realtime_turn_detection")
        if not isinstance(turn_detection, dict) or turn_detection.get("type") not in ("server_vad", "semantic_vad"):
            self._config = None
            self._detector = None
            return
        try:
            config = TurnDetectionConfig.from_realtime(turn_detection)
            unsupported = self._semantic_vad_unsupported(config, self._ctx.session.config.extra_body)
            if unsupported is not None:
                raise ValueError(unsupported)
            self._config = config
            self._detector = config.build_detector(self._vad_backend_provider(), self._semantic_vad_backend_provider())
        except Exception as exc:
            logger.warning("Duplex session %s: turn detection disabled: %s", self._ctx.session.session_id, exc)
            self._config = None
            self._detector = None

    def _vad_backend_provider(self) -> SileroVADBackendProvider | None:
        """The engine-wide Silero backend, so one model serves every session.

        Held by the manager rather than built here: the ONNX backend is shared,
        and ``duplex_session.server_vad_model_path`` is a deploy-level setting.
        """
        return getattr(self._ctx.manager, "vad_backend_provider", None)

    def _semantic_vad_backend_provider(self) -> SmartTurnBackendProvider | None:
        """The engine-wide Smart Turn backend (``duplex_session.semantic_vad_model_path``)."""
        return getattr(self._ctx.manager, "semantic_vad_backend_provider", None)

    def _semantic_vad_unsupported(self, config: TurnDetectionConfig | None, extra_body: object) -> str | None:
        """``semantic_vad`` only decides turn ends, so it needs a session that commits turns.

        An auto-responding (native duplex) session never commits on the
        detector; there semantic pauses would only hold ``force_listen`` on the
        model for seconds at a time.
        """
        if config is None or config.type != "semantic_vad":
            return None
        if session_auto_responds(self._ctx.session.capabilities, extra_body):
            return "turn_detection.type='semantic_vad' requires a turn-based (committing) session"
        return None

    def reset_vad(self) -> None:
        """Drop the detector's speech state (a barge-in or a clear starts a new turn)."""
        if self._detector is not None:
            self._detector.reset()

    async def run_turn_detection(
        self, event: dict[str, object]
    ) -> list[tuple[dict[str, object], TurnDetectionResult | None]]:
        detector = self._detector
        if detector is None:
            return [(event, None)]
        audio = event.get("audio")
        if not isinstance(audio, str) or not audio:
            return [(event, None)]
        fmt = event.get("format") if isinstance(event.get("format"), str) else "pcm_f32le"
        sample_rate_hz = event.get("sample_rate_hz")
        try:
            segments = await self._ctx.services.offload(
                detector.process_segments,
                audio,
                fmt=fmt,
                sample_rate_hz=sample_rate_hz if isinstance(sample_rate_hz, int) else None,
            )
        except SemanticVADUnavailableError as exc:
            self._out.emit_error("semantic_vad_unavailable", str(exc))
            self._detector = None
            return [(event, None)]
        except ServerVADUnavailableError as exc:
            self._out.emit_error("server_vad_unavailable", str(exc))
            self._detector = None
            return [(event, None)]
        except ValueError as exc:
            self._out.emit_error("bad_audio", str(exc))
            return []
        detected: list[tuple[dict[str, object], TurnDetectionResult | None]] = []
        for index, segment in enumerate(segments):
            payload = dict(event, audio=segment.audio, format="pcm_f32le", sample_rate_hz=16000)
            if index:
                # Video frames and transcript hints belong to the original
                # append; splitting its audio must not duplicate them.
                payload.pop("video_frames", None)
                payload.pop("transcript", None)
            self._log_turn_detection(segment.result)
            apply_turn_detection_result(payload, segment.result)
            detected.append((payload, segment.result))
        return detected

    def _log_turn_detection(self, result: TurnDetectionResult) -> None:
        if (
            result.speech_started
            or result.speech_stopped
            or result.turn_complete_probability is not None
            or result.semantic_decision is not None
        ):
            logger.info(
                "Duplex VAD session=%s backend=%s started=%s stopped=%s active=%s probability=%.3f "
                "turn_complete=%s start_ms=%s end_ms=%s commit=%s semantic_decision=%s commit_reason=%s",
                self._ctx.session.session_id,
                result.backend,
                result.speech_started,
                result.speech_stopped,
                result.speech_active,
                result.speech_probability,
                result.turn_complete_probability,
                result.audio_start_ms,
                result.audio_end_ms,
                result.should_commit,
                result.semantic_decision,
                result.commit_reason,
            )

    # ------------------------------------------------------------------ #
    # Control events                                                     #
    # ------------------------------------------------------------------ #

    async def on_turn_signal(self, event: dict[str, object]) -> None:
        session = self._ctx.session
        turn_event = event.get("event")
        realtime_event_id = event.get("realtime_event_id")
        if not isinstance(turn_event, str):
            self._out.emit_error("bad_event", "turn.signal requires event")
            return
        if turn_event == "barge_in" and not session.capabilities.supports_barge_in:
            self._out.emit_events([helpers.barge_in_unsupported_error()])
            return
        if turn_event == "session.update":
            payload = event.get("payload")
            if not isinstance(payload, dict):
                self._out.emit_error(
                    "bad_event", "session.update requires a session payload", event_id=realtime_event_id
                )
                return
            await self.on_session_update(
                payload,
                realtime_event_id=realtime_event_id if isinstance(realtime_event_id, str) else None,
            )
            return
        if turn_event == "conversation.item.create":
            await self._on_conversation_item_create(event)
            return
        if turn_event == "conversation.item.delete":
            payload = event.get("payload")
            item_id = payload.get("item_id") if isinstance(payload, dict) else None
            deleted = session.delete_history_item(item_id) if isinstance(item_id, str) else False
            self._out.emit(
                {
                    "type": "conversation.item.deleted",
                    "session_id": session.session_id,
                    "item_id": item_id,
                    "deleted": deleted,
                }
            )
            return
        if turn_event == "conversation.item.truncate":
            payload = event.get("payload")
            item_id = payload.get("item_id") if isinstance(payload, dict) else None
            audio_end_ms = payload.get("audio_end_ms") if isinstance(payload, dict) else None
            if isinstance(item_id, str) and not session.history_item_has_audio(item_id):
                self._out.emit_error(
                    "bad_event",
                    "conversation.item.truncate requires an item that produced audio",
                    event_id=realtime_event_id,
                )
                return
            truncated = (
                session.truncate_history_item(
                    item_id,
                    audio_end_ms=int(audio_end_ms) if isinstance(audio_end_ms, int | float) else 0,
                    hard=True,
                )
                if isinstance(item_id, str)
                else False
            )
            self._out.emit(
                {
                    "type": "conversation.item.truncated",
                    "session_id": session.session_id,
                    "item_id": item_id,
                    "content_index": (payload.get("content_index", 0) if isinstance(payload, dict) else 0),
                    "audio_end_ms": audio_end_ms,
                    "truncated": truncated,
                }
            )
            return
        self._out.emit_events([session.signal_turn(turn_event, event)])

    async def on_session_update(self, payload: dict[str, object], *, realtime_event_id: str | None) -> None:
        session = self._ctx.session
        model_state = self._ctx.model_state
        # Staged detectors have not processed audio, so a rejected candidate
        # needs no cleanup. Upstream dropped PendingTurnDetectionUpdate.reject.
        pending_turn_detection: PendingTurnDetectionUpdate | None = None

        try:
            pending_turn_detection = PendingTurnDetectionUpdate.prepare(
                payload,
                backend_provider=self._vad_backend_provider(),
                semantic_backend_provider=self._semantic_vad_backend_provider(),
            )
        except Exception as exc:
            self._out.emit_error("unsupported_turn_detection", str(exc), event_id=realtime_event_id)
            return
        if not await self._wait_for_append_tail():
            self._out.emit_error(
                "session_update_aborted",
                "session.update was not applied because the preceding append failed",
                event_id=realtime_event_id,
            )
            return
        if (
            pending_turn_detection is not None
            and (
                pending_turn_detection.config != self._config
                or (pending_turn_detection.config is not None and self._detector is None)
            )
            and (
                model_state.input_since_commit
                or model_state.audio_buffer.has_pending()
                or model_state.audio_buffer.has_reserved()
            )
        ):
            # A fresh detector cannot reconstruct the turn/pause represented by
            # buffered audio, including recovery with identical settings after
            # backend failure. Refuse before adopting any config.
            self._out.emit_error(
                "unsupported_turn_detection",
                "session.update cannot replace or restart turn_detection while input is pending; "
                "commit or clear input first",
                event_id=realtime_event_id,
            )
            return
        try:
            self._ctx.plugin.validate_client_extra_body(payload.get("extra_body"))
        except DuplexRuntimeConfigError as exc:
            self._out.emit_error(exc.code, str(exc), event_id=realtime_event_id)
            return
        candidate_config = deepcopy(session.config)
        audio_started = session.playback.generated_ms > 0 or session.playback.sent_ms > 0
        try:
            candidate_config.apply_realtime_update(
                payload,
                session_id=session.session_id,
                audio_started=audio_started,
            )
        except DuplexConfigError as exc:
            self._out.emit_error(exc.code, str(exc) or "session.update was rejected", event_id=realtime_event_id)
            return
        except Exception as exc:
            # Still before the live session adopts the candidate. A stamped
            # error means nothing was applied, including a staged detector.
            self._out.emit_error("internal_error", str(exc), event_id=realtime_event_id)
            return
        # Also when the update only turns auto_response on under a live semantic_vad.
        unsupported = self._semantic_vad_unsupported(
            pending_turn_detection.config if pending_turn_detection is not None else self._config,
            candidate_config.extra_body,
        )
        if unsupported is not None:
            self._out.emit_error("unsupported_turn_detection", unsupported, event_id=realtime_event_id)
            return
        if candidate_config.instructions != session.config.instructions and model_state.context_locked:
            self._out.emit_error(
                "instructions_update_unsupported",
                "session.update cannot change instructions after the native duplex context is initialized",
                event_id=realtime_event_id,
            )
            return
        requests_audio = any(str(modality).lower() == "audio" for modality in candidate_config.modalities)
        if (
            requests_audio
            and "ref_audio_data" not in session.runtime_config
            and getattr(self._ctx.plugin, "requires_ref_audio", False)
        ):
            self._out.emit_error(
                "ref_audio_required", "Native duplex audio output requires ref_audio", event_id=realtime_event_id
            )
            return
        try:
            candidate_runtime_config = self._ctx.plugin.runtime_config_for_update(
                candidate_config,
                dict(session.runtime_config),
            )
            # Validate the sampling policy for the candidate before adopting it.
            self._ctx.plugin.configure_sampling_params(
                runtime_config=dict(candidate_runtime_config),
                defaults=tuple(self._ctx.stage_port.sampling_defaults()),
            )
        except (DuplexRuntimeConfigError, DuplexConfigError) as exc:
            self._out.emit_error(exc.code, str(exc), event_id=realtime_event_id)
            return
        except Exception as exc:
            self._out.emit_error("runtime_signal_failed", str(exc), event_id=realtime_event_id)
            return
        try:
            # Projection defaults do not depend on the live config. A failure
            # here is still a rejection: the session has not adopted anything.
            projector = self._out.require_projector()
            projector.apply_session_defaults(payload)
        except Exception as exc:
            self._out.emit_error("internal_error", str(exc), event_id=realtime_event_id)
            return

        # Past this point the projector has the new wire defaults. Do not emit
        # a stamped error: serving would keep the old format and rate.
        def adopted_session() -> dict[str, object]:
            # The object serving applies. This is the patch apply_session_defaults
            # already accepted. Copy so a later mutation cannot rewrite the event.
            public = dict(payload)
            public["id"] = session.session_id
            return public

        def without_turn_detection(public: dict[str, object]) -> dict[str, object]:
            # Shallow-copy the nested audio maps before dropping turn detection,
            # so the live payload the projector already read stays intact.
            public.pop("turn_detection", None)
            audio = public.get("audio")
            if isinstance(audio, dict):
                audio = dict(audio)
                audio_input = audio.get("input")
                if isinstance(audio_input, dict):
                    audio_input = dict(audio_input)
                    audio_input.pop("turn_detection", None)
                    audio["input"] = audio_input
                public["audio"] = audio
            return public

        previous_config = session.config
        runtime_adopted = False
        turn_committed = pending_turn_detection is None
        try:
            session.replace_config(candidate_config)
            session.replace_runtime_config(candidate_runtime_config)
            runtime_adopted = True
            try:
                session.touch_lease(DuplexLeaseActivity.SIGNAL)
            except Exception:
                pass
            if pending_turn_detection is not None:
                self._config, self._detector = pending_turn_detection.commit(self._detector)
                turn_committed = True
            try:
                public: dict[str, object] = session.as_public_dict()
            except Exception:
                logger.exception(
                    "session.update %s was adopted but its public snapshot failed",
                    realtime_event_id,
                )
                public = adopted_session()
            self._out.emit({"type": "session.updated", "session": public})
        except Exception:
            logger.exception(
                "session.update %s was adopted but publishing it failed",
                realtime_event_id,
            )
            # replace_config assigns before it can raise. Finish the runtime
            # config and the staged detector once that assignment happened.
            # Otherwise the echo would advertise a VAD config that is not live,
            # and the engine would keep the new config against the old runtime
            # config. If the runtime replace cannot be finished, put the
            # previous config back. A staged detector has not processed audio,
            # so dropping it is enough.
            if session.config is candidate_config and not runtime_adopted:
                try:
                    session.replace_runtime_config(candidate_runtime_config)
                    runtime_adopted = True
                except Exception:
                    logger.exception(
                        "session.update %s could not finish replacing its runtime config",
                        realtime_event_id,
                    )
                    try:
                        # Not replace_config: that would count the candidate's
                        # seed again and bump config_generation a second time.
                        session.restore_config(previous_config)
                    except Exception:
                        logger.exception(
                            "session.update %s could not restore its previous config",
                            realtime_event_id,
                        )
            if session.config is candidate_config:
                if pending_turn_detection is not None and not turn_committed:
                    try:
                        self._config, self._detector = pending_turn_detection.commit(self._detector)
                        turn_committed = True
                    except Exception:
                        logger.exception(
                            "session.update %s could not commit its turn detection",
                            realtime_event_id,
                        )
                public = adopted_session()
                if not turn_committed:
                    public = without_turn_detection(public)
            else:
                public = {"id": session.session_id}
            self._out.emit({"type": "session.updated", "session": public})

    async def _on_conversation_item_create(self, event: dict[str, object]) -> None:
        session = self._ctx.session
        payload = event.get("payload")
        item = payload.get("item") if isinstance(payload, dict) else None
        item_type = item.get("type") if isinstance(item, dict) else None
        if item_type == "function_call_output" and isinstance(item, dict):
            await self._on_function_call_output(item, event_id=event.get("realtime_event_id"))
            return
        item_payload = item if isinstance(item, dict) else None
        raw_parts = item_payload.get("content", []) if item_payload is not None else []
        parts: list[object] = [part for part in raw_parts] if isinstance(raw_parts, list) else []
        images = [p for p in parts if isinstance(p, dict) and p.get("type") == "input_image"]
        if images:
            if item_payload is None or item_payload.get("role") != "user":
                self._out.emit_error("invalid_image", "input_image is supported only in user messages")
                return
            if not session.capabilities.supports_image_input:
                self._out.emit_error("unsupported", "This model does not support input_image conversation items")
                return
            from vllm_omni.engine.duplex.realtime_commands import validate_realtime_video_frames

            urls: list[str] = []
            for part in images:
                url = part.get("image_url")
                if not isinstance(url, str) or not url.startswith(
                    ("data:image/jpeg;base64,", "data:image/png;base64,")
                ):
                    self._out.emit_error("invalid_image", "input_image requires a JPEG or PNG base64 data URL")
                    return
                error = validate_realtime_video_frames([url.split(",", 1)[1]], None)
                if error:
                    self._out.emit_error("invalid_image", error)
                    return
                urls.append(url)
            existing: list[str] = []
            for history_message in session.history:
                content = history_message.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict) or part.get("type") != "image_url":
                        continue
                    image_url = part.get("image_url")
                    nested_url = image_url.get("url") if isinstance(image_url, dict) else None
                    if isinstance(nested_url, str):
                        existing.append(nested_url)
            if len(existing) + len(urls) > 8 or sum(map(len, existing + urls)) > 4 * 1024 * 1024:
                self._out.emit_error("input_backpressure", "Image context exceeds 8 images or 4 MiB; delete old items")
                return
        message = realtime_item_to_history_message(item)
        item_id = item.get("id") if isinstance(item, dict) else None
        if message is not None:
            session.append_history_message(message)
            session.register_history_item(item_id if isinstance(item_id, str) else None, message)
            if message.get("role") == "user":
                # A later response.create may answer this without any audio.
                session.notify_new_user_item()
        self._out.emit(
            {
                "type": "conversation.item.created",
                "session_id": session.session_id,
                "item": item,
                "created": message is not None,
            }
        )

    async def _on_function_call_output(self, item: Mapping[str, object], *, event_id: object) -> None:
        """Admit one client tool result, or answer why it cannot be admitted.

        The session's tool-call ledger is the admission authority (see
        ``session/tool_calls.py``): a result is admitted only for a call this
        session exposed, that is still pending, and whose origin epoch the
        session has not cancelled since. Eligibility is checked before the
        append tail and again after it, and the plugin's candidate runtime
        config is built before anything is committed. Retryable preparation
        errors leave the call pending. Terminal plugin bookkeeping finishes
        before the success acknowledgement; failure there closes the session
        rather than reopening an already accepted call. Commit has no ``await``.
        """
        session = self._ctx.session
        ledger = self._ctx.tool_calls
        call_id = item.get("call_id")
        acceptance = ledger.begin_accept(call_id)
        if isinstance(acceptance, ToolCallRejection):
            self._reject_function_call_output(acceptance, call_id, event_id=event_id)
            return
        if not await self._wait_for_append_tail():
            # Nothing was applied and the call is still pending: the client may
            # send the same result again once the append path has recovered.
            self._out.emit_error(
                "commit_aborted",
                "the function_call_output was not applied because the preceding append failed",
                event_id=event_id,
            )
            return
        epoch = session.epoch
        rejection = ledger.revalidate(acceptance, epoch=epoch)
        if rejection is not None:
            self._reject_function_call_output(rejection, call_id, event_id=event_id)
            return
        call = ledger.record(acceptance.call_id)
        if call is None:
            self._out.emit_error(
                "commit_aborted", "the function-call context is no longer available", event_id=event_id
            )
            return
        try:
            # The client returns a result, not an authoritative function name
            # or argument list. Adaptation uses the call this session exposed.
            prepared_item = {key: value for key, value in item.items() if key not in {"name", "arguments"}}
            prepared_item["call_id"] = call.origin.call_id
            if call.origin.name is not None:
                prepared_item["name"] = call.origin.name
            if call.origin.arguments is not None:
                prepared_item["arguments"] = call.origin.arguments
            candidate_runtime_config = self._ctx.plugin.runtime_config_for_function_output(
                session.config,
                deepcopy(dict(session.runtime_config)),
                deepcopy(prepared_item),
                history_len=len(session.history),
            )
            # Detach returned plugin-owned objects before terminal cleanup can
            # mutate or release them, while refusal is still retryable.
            if candidate_runtime_config is not None:
                candidate_runtime_config = deepcopy(candidate_runtime_config)
        except DuplexRuntimeConfigError as exc:
            # The plugin refused the result; the call stays pending, so a
            # corrected result for the same call is still admissible.
            self._out.emit_error(exc.code, str(exc), event_id=event_id)
            return
        except Exception:
            logger.exception("Failed to prepare function-call output")
            self._out.emit_error("commit_aborted", "the function_call_output could not be prepared", event_id=event_id)
            return
        # Rendering the candidate is synchronous but not instant (a tool result
        # is tokenized here), so the deadline is rechecked *after* it and
        # immediately before the commit: a result that missed its deadline while
        # the plugin was rendering it is a timeout, and neither the candidate
        # nor the continuation may be installed for it.
        epoch = session.epoch
        expiry = ledger.revalidate(acceptance, epoch=epoch)
        if expiry is not None:
            self._reject_function_call_output(expiry, call_id, event_id=event_id)
            return
        # -- commit: no ``await`` past this point --
        if not ledger.commit_accept(acceptance, epoch=epoch):
            call = ledger.record(acceptance.call_id)
            if call is not None and call.is_pending:
                self._out.emit_error(
                    "commit_aborted", "the function_call_output could not be committed", event_id=event_id
                )
                return
            self._reject_function_call_output(ToolCallRejection.NOT_PENDING, call_id, event_id=event_id)
            return
        if not self._ctx.services.finalize_tool_call(acceptance.call_id):
            # Acceptance is terminal. A failed finalize closes the session and
            # must not acknowledge or continue an unusable prepared result.
            return
        # Read the continuation target inside the await-free commit, so it is
        # the turn the session is on when the result is applied -- not the turn
        # the call was emitted in. A model turn can complete while the client is
        # answering, and the ledger's job is eligibility (origin epoch plus
        # settlement), not liveness: the origin turn stays on the record as
        # identity, the response has to continue the *current* turn.
        current_model_turn_id = session.turn_id
        if candidate_runtime_config is not None:
            session.replace_runtime_config(candidate_runtime_config)
        self._out.emit(
            {
                "type": "conversation.item.created",
                "session_id": session.session_id,
                "item": dict(item),
                "created": True,
            }
        )
        # Tool-result admission does not finish an in-flight committed turn.
        # Its real terminal output owns response closure; the non-resumable
        # branch of maybe_continue_response is only for listen/final decisions.
        if session.capabilities.supports_core_resumable_request:
            self._ctx.services.spawn(
                self._model.maybe_continue_response(
                    expected_epoch=epoch,
                    expected_model_turn_id=current_model_turn_id,
                ),
                name="duplex-continue",
            )

    def _reject_function_call_output(self, reason: ToolCallRejection, call_id: object, *, event_id: object) -> None:
        """Report a refused result with the wire shape the public contract kept.

        Same code, ``event_id`` correlation and ``item.call_id`` /
        ``item.output`` param as the projection used to send; only the message
        is per-cause, because the causes are no longer one bucket.
        """
        settled = self._ctx.tool_calls.record(call_id)
        state = settled.state if settled is not None else None
        if reason is ToolCallRejection.UNKNOWN:
            message = "function_call_output requires the call_id of a completed function call"
            param = "item.call_id"
        elif reason is ToolCallRejection.EXPIRED or state is ToolCallState.TIMED_OUT:
            # ``EXPIRED`` is "the deadline had already passed when this arrived";
            # a timed-out record is the same fact one step later (the timer
            # settled it first). Both answer as a timeout, not as a cancellation.
            message = f"function_call_output arrived after tool_result_timeout_s for call_id {call_id}"
            param = "item.call_id"
        elif state is ToolCallState.ACCEPTED:
            message = f"function_call_output already exists for call_id {call_id}"
            param = "item.output"
        else:
            message = f"function_call_output for call_id {call_id} belongs to a cancelled call and will not be applied"
            param = "item.output"
        self._out.emit_events([error_event("invalid_function_call_output", message, event_id=event_id, param=param)])
