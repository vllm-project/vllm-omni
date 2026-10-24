# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Engine-owned VAD/commit turns with concurrent input and interruptible output.

Qwen3-Omni is a turn model: this plugin does not advertise native streaming KV
or model-driven turn decisions. Each committed utterance is a fresh request.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pybase64 as base64
from vllm.logger import init_logger
from vllm.multimodal.image import open_image
from vllm.sampling_params import RequestOutputKind

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.plugin import DuplexDataPlane, DuplexModelPlugin, DuplexRuntimeConfigError
from vllm_omni.model_executor.models.qwen3_omni.duplex.session import QwenDuplexSessionState
from vllm_omni.model_executor.models.qwen3_omni.duplex.tools_utils import (
    ThinkerToolCursor,
    chat_template_tools,
)
from vllm_omni.utils.speakers import (
    get_supported_speakers_from_hf_config,
    validate_requested_speaker,
)

logger = init_logger(__name__)

#: Mirrors ``limit_mm_per_prompt.image`` in ``deploy/qwen3_omni_duplex.yaml``.
#: The session already refuses a ninth stored image, but that budget lives in
#: the engine and this one in the deploy config; the prompt is the only place
#: that can keep them from drifting apart into a rejected submission.
MAX_PROMPT_IMAGES = 8
# Same window ``prepare_prompt_config`` keeps. Anchors older than this cannot
# still point at a message the next prompt will retain.
_TOOL_ANCHOR_WINDOW = 16


def _anchored_history_len(
    current: Mapping[str, object],
    item: Mapping[str, object],
    history_len: int | None,
) -> int:
    """Prefer the length recorded at publish over the length when the result arrives.

    A client ``function_call_output`` can carry ``_history_len``. That key is
    not a splice position.
    """
    call_id = item.get("call_id")
    anchors = current.get("qwen_call_anchors")
    if isinstance(call_id, str) and isinstance(anchors, Mapping):
        anchored = anchors.get(call_id)
        if isinstance(anchored, int):
            return anchored
    if isinstance(history_len, int):
        return history_len
    return 0


def _split_call_anchors(anchors: Mapping[str, object], *, history_len: int) -> tuple[dict[str, int], list[str]]:
    """Keep anchors inside the prompt window. The rest were published and expired."""
    floor = history_len - _TOOL_ANCHOR_WINDOW
    kept: dict[str, int] = {}
    expired: list[str] = []
    for key, value in anchors.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, int) and value > floor:
            kept[key] = value
        else:
            expired.append(key)
    return kept, expired


def _prune_call_anchors(anchors: Mapping[str, object], *, history_len: int) -> dict[str, int]:
    return _split_call_anchors(anchors, history_len=history_len)[0]


_EXPIRED_CALL_LIMIT = 64


def _remember_expired(updated: dict[str, object], expired_ids: list[str]) -> None:
    if not expired_ids:
        return
    current = updated.get("qwen_expired_call_ids")
    kept = [item for item in current if isinstance(item, str)] if isinstance(current, list) else []
    for call_id in expired_ids:
        if call_id not in kept:
            kept.append(call_id)
    if len(kept) > _EXPIRED_CALL_LIMIT:
        kept = kept[-_EXPIRED_CALL_LIMIT:]
    updated["qwen_expired_call_ids"] = kept


def _tool_turns(runtime_config: Mapping[str, object], key: str) -> list[dict[str, object]]:
    raw = runtime_config.get(key)
    if not isinstance(raw, list):
        return []
    return [turn for turn in raw if isinstance(turn, dict)]


def _call_ids_of(turns: object) -> list[str]:
    if not isinstance(turns, list):
        return []
    found: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        call_id = _tool_call_id(turn)
        if call_id and call_id not in found:
            found.append(call_id)
    return found


def _tool_call_id(turn: Mapping[str, object]) -> str | None:
    messages = turn.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        call_id = message.get("tool_call_id")
        if isinstance(call_id, str) and call_id:
            return call_id
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            nested = call.get("id")
            if isinstance(nested, str) and nested:
                return nested
    return None


def _with_response_owner(turn: dict[str, object], owner: object) -> dict[str, object]:
    if isinstance(owner, str) and owner:
        if turn.get("response_id") == owner:
            return turn
        return {**turn, "response_id": owner}
    if "response_id" not in turn:
        return turn
    return {key: value for key, value in turn.items() if key != "response_id"}


def _current_turns_by_call_id(runtime_config: Mapping[str, object]) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    # Waiting overwrites inflight. A failure puts the pair back there with no owner.
    for key in ("qwen_tool_turns_inflight", "qwen_tool_turns"):
        for turn in _tool_turns(runtime_config, key):
            call_id = _tool_call_id(turn)
            if call_id:
                found[call_id] = turn
    return found


def _live_tool_response_ids(runtime_config: Mapping[str, object]) -> set[str]:
    live: set[str] = set()
    for key in ("qwen_tool_turns", "qwen_tool_turns_inflight"):
        for turn in _tool_turns(runtime_config, key):
            owner = turn.get("response_id")
            if isinstance(owner, str) and owner:
                live.add(owner)
    return live


class QwenDataPlane(DuplexDataPlane):
    def __init__(self, encode_audio):
        self.encode_audio = encode_audio
        self.terminal: set[str] = set()
        self.retain_history_audio = False
        self._tool_cursors: dict[str, ThinkerToolCursor] = {}

    def begin_request(self, request_id):
        self.terminal.discard(request_id)
        self._tool_cursors.pop(request_id, None)

    def is_terminal(self, request_id):
        return request_id in self.terminal

    def mark_terminal(self, request_id):
        self.terminal.add(request_id)
        self._tool_cursors.pop(request_id, None)

    def close_stream(self, request_id):
        self._tool_cursors.pop(request_id, None)

    def close_session(self, session_id, *, active_request_id=None):
        self.terminal = {
            rid for rid in self.terminal if not duplex_resource_request_belongs_to_session(rid, session_id)
        }
        self._tool_cursors = {
            rid: cursor
            for rid, cursor in self._tool_cursors.items()
            if not duplex_resource_request_belongs_to_session(rid, session_id)
        }

    def project(self, result, *, context=None):
        if not isinstance(result, dict):
            return
        context = context or {}
        for output in result.get("data_plane_outputs", []):
            rid = output.request_id
            if self.is_terminal(rid):
                continue
            completion = next(iter(getattr(output, "outputs", ()) or ()), None)
            stage_id = getattr(output, "stage_id", 0)
            text = getattr(completion, "text", "") if stage_id == 0 else ""
            mm = getattr(output, "multimodal_output", None)
            if not isinstance(mm, Mapping) or "audio" not in mm:
                mm = getattr(completion, "multimodal_output", {}) or {}
            audio = mm.get("audio")
            encoded = None
            duration_ms = 0
            rate = mm.get("sr", mm.get("sample_rate", mm.get("sample_rate_hz", 24000)))
            if isinstance(rate, (list, tuple)):
                rate = next((value for value in rate if value is not None), 24000)
            rate = int(rate)
            if rate <= 0:
                raise ValueError("Invalid output audio sample rate")
            if audio is not None and "audio" in context.get("modalities", ()):
                chunks = audio if isinstance(audio, (list, tuple)) else [audio]
                arrays = [
                    np.asarray(
                        chunk.detach().float().cpu().numpy() if hasattr(chunk, "detach") else chunk, dtype=np.float32
                    ).reshape(-1)
                    for chunk in chunks
                ]
                audio = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)
                duration_ms = audio.size * 1000 / rate
                if audio.size:
                    encoded = self.encode_audio(
                        audio, rate, context.get("response_format", "pcm16"), context.get("speed")
                    )
            cursor = self._tool_cursors.setdefault(rid, ThinkerToolCursor())
            speakable = ""
            calls: list[dict[str, str]] = []
            if stage_id == 0 and isinstance(text, str) and text:
                piece, call = cursor.absorb(text)
                speakable += piece
                while call is not None:
                    calls.append(call)
                    piece, call = cursor.absorb("")
                    speakable += piece
            finished = bool(output.finished)
            if finished:
                speakable += cursor.finish()
            # Tool markup is not speech on the Thinker. Codec stages keep their
            # packets: talker audio is not aligned to the tag token.
            suppress_audio = stage_id == 0 and (bool(calls) or cursor.holding_open_call() or cursor.tool_turn)
            if suppress_audio:
                encoded = None
                duration_ms = 0
            model_turn_id = (
                context.get("active_response_turn_id")
                if context.get("active_response_turn_id") is not None
                else context.get("turn_id", 0)
            )

            def _event(
                *,
                event_text: str,
                event_audio: str,
                end_of_turn: bool,
                audio_complete: bool,
                function_call: dict[str, str] | None = None,
            ) -> dict[str, object]:
                payload: dict[str, object] = {
                    "supported": True,
                    "data_plane_request_id": rid,
                    "text": event_text,
                    "audio_data": event_audio,
                    "audio_format": context.get("response_format", "pcm16"),
                    "audio_duration_ms": duration_ms if event_audio else 0,
                    # Thinker text and codec chunks are independent streams. Only
                    # the final audio boundary can conservatively attest all text.
                    "audio_text_mark": False,
                    "text_requires_complete_audio": "audio" in context.get("modalities", ()),
                    "audio_complete": audio_complete,
                    "sample_rate_hz": rate,
                    "history_audio_pcm": audio.astype("<f4", copy=False).tobytes()
                    if event_audio and self.retain_history_audio
                    else None,
                    "history_audio_supported": context.get("speed") in (None, 1, 1.0),
                    "end_of_turn": end_of_turn,
                    "model_turn_id": model_turn_id,
                    "runtime_impl": "qwen3_commit",
                    "uses_model_runner_scheduler": True,
                    "runner_kv_backed": True,
                }
                if function_call is not None:
                    payload["qwen_function_call"] = function_call
                return payload

            if not suppress_audio:
                yield _event(
                    event_text=speakable if stage_id == 0 else (text or ""),
                    event_audio=encoded or "",
                    end_of_turn=finished,
                    audio_complete=finished and stage_id == 2,
                )
                continue
            if speakable:
                yield _event(event_text=speakable, event_audio="", end_of_turn=False, audio_complete=False)
            for call in calls:
                yield _event(
                    event_text="",
                    event_audio="",
                    end_of_turn=False,
                    audio_complete=False,
                    function_call=call,
                )
            if finished:
                yield _event(
                    event_text="",
                    event_audio="",
                    end_of_turn=True,
                    audio_complete=stage_id == 2,
                )


class Qwen3OmniDuplexPlugin(DuplexModelPlugin):
    plugin_id = "qwen3-omni"
    projects_intermediate_outputs = True

    def __init__(self, encode_audio):
        self.data_plane = QwenDataPlane(encode_audio)
        self.processor = None
        self._processor_lock = asyncio.Lock()
        self._history_calibrator = None
        # Checkpoint speaker names, loaded once from the stage-0 model config.
        # ``None`` means "not attempted" (no validation); an empty set means the
        # checkpoint declares no speaker_id mapping (names pass through unvalidated).
        self._supported_speakers: set[str] | None = None

    def history_calibrator(self, runtime_config):
        if runtime_config.history_asr_url is None and runtime_config.history_ms_per_token is None:
            return None
        if self._history_calibrator is None:
            from vllm_omni.engine.duplex.session.history_calibration import HistoryCalibrationPolicy

            if runtime_config.history_ms_per_token is not None:
                from vllm_omni.model_executor.models.qwen3_omni.duplex.history_rate import QwenTokenRateCalibration

                self._history_calibrator = HistoryCalibrationPolicy(
                    QwenTokenRateCalibration(self.processor.tokenizer, runtime_config.history_ms_per_token),
                    requires_audio=False,
                )
            else:
                from vllm_omni.model_executor.models.qwen3_omni.duplex.history_asr import QwenAsrCalibration

                self._history_calibrator = HistoryCalibrationPolicy(
                    QwenAsrCalibration(
                        runtime_config.history_asr_url,
                        runtime_config.history_asr_model,
                        runtime_config.history_calibration_timeout_ms / 1000,
                        max_concurrency=runtime_config.history_asr_max_concurrency,
                    )
                )
                self.data_plane.retain_history_audio = True
        return self._history_calibrator

    def capabilities(self, *, max_sessions):
        return DuplexCapabilities(
            supports_model_native_turn_policy=False,
            supports_input_append=True,
            supports_barge_in=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=True,
            supports_turn_commit_only=True,
            supports_core_resumable_request=False,
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=max_sessions > 1,
            supports_multi_session_same_replica=max_sessions > 1,
            supports_session_lease=True,
            supports_session_resume=True,
            session_admission_mode="engine_managed",
            supports_audio_truncate=True,
            supports_chat_completions=True,
            supports_image_input=True,
            # Frames arrive as input_image conversation items, never on an
            # audio append (QwenPcmBuffer.prepare_append refuses video_frames).
            optional_input_modalities=frozenset(),
            supports_text_only_turn=True,
            adapter_patterns=["turn_commit"],
            signal_sources=["client_event", "server_policy"],
            stage_handoff_transport="scheduler_data_plane",
            target_barge_in_latency_ms=None,
        )

    def create_session_state(self):
        return QwenDuplexSessionState()

    def validate_client_extra_body(self, extra_body):
        if not isinstance(extra_body, Mapping):
            return
        if extra_body.get("auto_response"):
            raise DuplexRuntimeConfigError(
                "Qwen requires server VAD or explicit commits; native auto_response is unsupported"
            )
        if extra_body.get("realtime_tool_choice") == "none":
            raise DuplexRuntimeConfigError(
                "Qwen duplex does not support tool_choice none",
                code="unsupported_tool_choice",
            )

    def _load_supported_speakers(self, model_config) -> None:
        if self._supported_speakers is not None or model_config is None:
            return
        try:
            # The stage-0 config carries the full Omni hf_config; the speaker
            # names live in ``talker_config.speaker_id``.
            self._supported_speakers = get_supported_speakers_from_hf_config(model_config.hf_config)
        except Exception:
            logger.warning("Qwen duplex could not read the checkpoint's speaker list", exc_info=True)
            self._supported_speakers = set()

    async def prepare_runtime_config(self, config, *, model_config):
        self.validate_client_extra_body(config.extra_body)
        if config.ref_audio:
            raise DuplexRuntimeConfigError("Qwen duplex does not support reference voice audio")
        self._load_supported_speakers(model_config)
        if model_config is None and self.processor is None:
            raise DuplexRuntimeConfigError("Qwen duplex requires the stage-0 model configuration")
        async with self._processor_lock:
            if self.processor is None:
                from transformers import AutoProcessor

                self.processor = await asyncio.to_thread(
                    AutoProcessor.from_pretrained,
                    model_config.model,
                    trust_remote_code=model_config.trust_remote_code,
                    revision=getattr(model_config, "revision", None),
                )
        return self.runtime_config_for_update(config, {})

    def runtime_config_for_update(self, config, current):
        self.validate_client_extra_body(config.extra_body)
        if config.ref_audio:
            raise DuplexRuntimeConfigError("Qwen duplex does not support reference voice audio")
        voice = config.voice
        try:
            voice = validate_requested_speaker(voice, self._supported_speakers or set())
        except ValueError as exc:
            raise DuplexRuntimeConfigError(str(exc), code="invalid_speaker") from exc
        updated: dict[str, object] = {
            "instructions": config.instructions,
            "initial_user_text": config.initial_user_text,
            "voice": voice,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        tools = config.extra_body.get("realtime_tools")
        if isinstance(tools, list):
            updated["qwen_tools"] = tools
        turns = current.get("qwen_tool_turns")
        if isinstance(turns, list):
            updated["qwen_tool_turns"] = turns
        inflight = current.get("qwen_tool_turns_inflight")
        if isinstance(inflight, list):
            updated["qwen_tool_turns_inflight"] = inflight
        response_id = current.get("qwen_tool_response_id")
        if isinstance(response_id, str) and response_id:
            updated["qwen_tool_response_id"] = response_id
        anchors = current.get("qwen_call_anchors")
        if isinstance(anchors, Mapping):
            updated["qwen_call_anchors"] = dict(anchors)
        expired = current.get("qwen_expired_call_ids")
        if isinstance(expired, list) and expired:
            updated["qwen_expired_call_ids"] = [item for item in expired if isinstance(item, str)]
        return updated

    def note_published_function_call(
        self,
        runtime_config: Mapping[str, object],
        *,
        call_id: str,
        history_len: int,
    ) -> dict[str, object]:
        """Remember the history length at ``function_call.done`` for this call."""
        updated = dict(runtime_config)
        anchors = updated.get("qwen_call_anchors")
        copied = dict(anchors) if isinstance(anchors, Mapping) else {}
        copied[call_id] = history_len
        kept, expired = _split_call_anchors(copied, history_len=history_len)
        updated["qwen_call_anchors"] = kept
        _remember_expired(updated, expired)
        return updated

    def has_pending_tool_followup(self, runtime_config: Mapping[str, object]) -> bool:
        turns = runtime_config.get("qwen_tool_turns")
        return isinstance(turns, list) and bool(turns)

    def extra_prompt_config(self, runtime_config: Mapping[str, object]) -> Mapping[str, object]:
        extra: dict[str, object] = {}
        turns = runtime_config.get("qwen_tool_turns")
        if isinstance(turns, list):
            extra["qwen_tool_turns"] = turns
        inflight = runtime_config.get("qwen_tool_turns_inflight")
        if isinstance(inflight, list) and inflight:
            extra["qwen_tool_turns_inflight"] = inflight
        return extra

    def runtime_patch_for_submitted_prompt(self, prompt: dict[str, object]) -> Mapping[str, object] | None:
        patch: dict[str, object] = {}
        if "_qwen_tool_turns_remaining" in prompt:
            patch["qwen_tool_turns"] = prompt.pop("_qwen_tool_turns_remaining")
        if "_qwen_tool_turns_inflight" in prompt:
            patch["qwen_tool_turns_inflight"] = prompt.pop("_qwen_tool_turns_inflight")
        if "_qwen_call_anchors" in prompt:
            patch["qwen_call_anchors"] = prompt.pop("_qwen_call_anchors")
        if "_qwen_expired_call_ids" in prompt:
            patch["qwen_expired_call_ids"] = prompt.pop("_qwen_expired_call_ids")
        if "_qwen_dropped_tool_call_ids" in prompt:
            # Seen by reconcile, then removed. Not session runtime.
            patch["_qwen_dropped_tool_call_ids"] = prompt.pop("_qwen_dropped_tool_call_ids")
        if "_qwen_newly_expired_call_ids" in prompt:
            # Seen by reconcile, then removed. Not session runtime.
            patch["_qwen_newly_expired_call_ids"] = prompt.pop("_qwen_newly_expired_call_ids")
        return patch or None

    def reconcile_submitted_runtime(
        self,
        current: Mapping[str, object],
        updated: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Drop a patched pair the ACK finished, and keep runtime accepted since the snapshot.

        ``plan_append`` snapshots runtime before it yields. An ACK in that window
        removes the pair. A failure puts the same pair back with no response id.
        Match ``tool_call_id`` so that failure is not treated as the ACK. A
        waiting result, anchor, or expired id accepted during the yield stays.
        This does not decide whether a result may be accepted. ``None`` means
        the patch already matches the live runtime.
        """
        current_by_call = _current_turns_by_call_id(current)
        live_response_ids = _live_tool_response_ids(current)
        result = dict(updated)
        changed = False
        dropped_raw = result.pop("_qwen_dropped_tool_call_ids", None)
        if dropped_raw is not None:
            changed = True
        dropped_ids = (
            {item for item in dropped_raw if isinstance(item, str)} if isinstance(dropped_raw, list) else set()
        )
        seen_call_ids: set[str] = set()
        for key in ("qwen_tool_turns", "qwen_tool_turns_inflight"):
            raw = result.get(key)
            if not isinstance(raw, list):
                continue
            kept: list[object] = []
            key_changed = False
            for turn in raw:
                if not isinstance(turn, dict):
                    kept.append(turn)
                    continue
                call_id = _tool_call_id(turn)
                owner = turn.get("response_id")
                if call_id:
                    seen_call_ids.add(call_id)
                    current_turn = current_by_call.get(call_id)
                    if current_turn is None and isinstance(owner, str) and owner:
                        key_changed = True
                        continue
                    if current_turn is not None:
                        adopted = _with_response_owner(turn, current_turn.get("response_id"))
                        if adopted is not turn:
                            key_changed = True
                        kept.append(adopted)
                        continue
                elif isinstance(owner, str) and owner and owner not in live_response_ids:
                    key_changed = True
                    continue
                kept.append(turn)
            if not key_changed:
                continue
            changed = True
            if kept:
                result[key] = kept
            else:
                result.pop(key, None)
        carried: list[dict[str, object]] = []
        for turn in _tool_turns(current, "qwen_tool_turns"):
            call_id = _tool_call_id(turn)
            if call_id and call_id not in seen_call_ids and call_id not in dropped_ids:
                carried.append(turn)
        if carried:
            changed = True
            waiting = result.get("qwen_tool_turns")
            base = list(waiting) if isinstance(waiting, list) else []
            result["qwen_tool_turns"] = [*base, *carried]
        patch_expired = result.get("qwen_expired_call_ids")
        expired_ids = (
            {item for item in patch_expired if isinstance(item, str)} if isinstance(patch_expired, list) else set()
        )
        current_anchors = current.get("qwen_call_anchors")
        patch_anchors = result.get("qwen_call_anchors")
        if isinstance(current_anchors, Mapping) or isinstance(patch_anchors, Mapping):
            # plan_append only removes anchors. Keep the live map, minus ids this
            # snapshot already expired, so a result accepted during the yield is
            # not given its old anchor back.
            source = current_anchors if isinstance(current_anchors, Mapping) else {}
            merged_anchors = {
                key: value
                for key, value in source.items()
                if isinstance(key, str) and isinstance(value, int) and key not in expired_ids
            }
            patch_kept = (
                {key: value for key, value in patch_anchors.items() if isinstance(key, str) and isinstance(value, int)}
                if isinstance(patch_anchors, Mapping)
                else {}
            )
            if merged_anchors != patch_kept:
                changed = True
                if merged_anchors:
                    result["qwen_call_anchors"] = merged_anchors
                else:
                    result.pop("qwen_call_anchors", None)
        current_expired = current.get("qwen_expired_call_ids")
        current_ids = (
            [item for item in current_expired if isinstance(item, str)] if isinstance(current_expired, list) else []
        )
        # The patch list is the snapshot window, including ids live may already
        # have evicted. Append only the ids this plan itself expired.
        newly_raw = result.pop("_qwen_newly_expired_call_ids", None)
        if newly_raw is not None:
            changed = True
        newly_ids = [item for item in newly_raw if isinstance(item, str)] if isinstance(newly_raw, list) else []
        if newly_ids or current_ids:
            expired_state: dict[str, object] = {"qwen_expired_call_ids": current_ids}
            before = result.get("qwen_expired_call_ids")
            _remember_expired(expired_state, newly_ids)
            after = expired_state.get("qwen_expired_call_ids")
            if after != before:
                changed = True
                if after:
                    result["qwen_expired_call_ids"] = after
                else:
                    result.pop("qwen_expired_call_ids", None)
        if changed:
            owner = result.get("qwen_tool_response_id")
            if isinstance(owner, str) and owner and owner not in _live_tool_response_ids(result):
                result.pop("qwen_tool_response_id", None)
        return result if changed else None

    def bind_inflight_response(
        self,
        runtime_config: Mapping[str, object],
        *,
        response_id: str | None,
    ) -> dict[str, object] | None:
        if not isinstance(response_id, str) or not response_id:
            return None
        inflight = runtime_config.get("qwen_tool_turns_inflight")
        if not isinstance(inflight, list):
            return None
        stamped: list[dict[str, object]] = []
        changed = False
        for turn in inflight:
            if not isinstance(turn, dict):
                continue
            owner = turn.get("response_id")
            if isinstance(owner, str) and owner:
                stamped.append(turn)
                continue
            stamped.append({**turn, "response_id": response_id})
            changed = True
        if not changed and runtime_config.get("qwen_tool_response_id") == response_id:
            return None
        updated = dict(runtime_config)
        updated["qwen_tool_turns_inflight"] = stamped
        updated["qwen_tool_response_id"] = response_id
        return updated

    def runtime_config_after_response(
        self,
        runtime_config: Mapping[str, object],
        *,
        committed: bool,
        awaiting_playback: bool = False,
        response_id: str | None = None,
        release_owner: bool = False,
    ) -> dict[str, object] | None:
        """Drop a submitted tool pair only when that response stored its text.

        The pair stays inflight while playback can still store the answer.
        ``committed`` is false for cancel, failure, or
        ``realtime_response_conversation`` set to ``none``; those put the pair
        back. A settlement for a different response leaves this pair alone.
        The pair is not in ``session.history``.
        """
        inflight_raw = runtime_config.get("qwen_tool_turns_inflight")
        inflight = inflight_raw if isinstance(inflight_raw, list) else []
        waiting_raw = runtime_config.get("qwen_tool_turns")
        waiting = [turn for turn in waiting_raw if isinstance(turn, dict)] if isinstance(waiting_raw, list) else []
        if awaiting_playback and not committed:
            return None
        owner = runtime_config.get("qwen_tool_response_id")

        def _owns(turn: object) -> bool:
            if not isinstance(turn, dict):
                return False
            turn_owner = turn.get("response_id")
            if isinstance(response_id, str) and isinstance(turn_owner, str):
                return turn_owner == response_id
            if isinstance(response_id, str) and isinstance(owner, str) and response_id != owner:
                return False
            return True

        owned = [turn for turn in inflight if _owns(turn)]
        rest = [turn for turn in inflight if isinstance(turn, dict) and not _owns(turn)]
        # Only a pair that already names this response. A new result has no id yet.
        tagged_waiting = [
            turn for turn in waiting if isinstance(response_id, str) and turn.get("response_id") == response_id
        ]
        if not owned and not (committed and tagged_waiting):
            return None
        updated = dict(runtime_config)
        if rest:
            updated["qwen_tool_turns_inflight"] = rest
        else:
            updated.pop("qwen_tool_turns_inflight", None)
            updated.pop("qwen_tool_response_id", None)
        if committed:
            updated["qwen_tool_turns"] = [turn for turn in waiting if turn not in tagged_waiting]
        elif owned:
            restored = []
            for turn in owned:
                if not isinstance(turn, dict):
                    continue
                if release_owner:
                    restored.append({key: value for key, value in turn.items() if key != "response_id"})
                else:
                    restored.append(turn)
            others = [turn for turn in waiting if turn not in tagged_waiting]
            updated["qwen_tool_turns"] = [*restored, *others]
        return updated

    def parse_function_call(self, model_result: Mapping[str, object]) -> dict[str, str] | None:
        raw = model_result.get("qwen_function_call")
        if not isinstance(raw, Mapping):
            return None
        if raw.get("invalid") == "1":
            return {"call_id": "invalid", "name": ""}
        call_id = raw.get("call_id")
        name = raw.get("name")
        arguments = raw.get("arguments", "")
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            return None
        if not isinstance(arguments, str):
            try:
                arguments = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError):
                # Same shape as an invalid tool tag: the session rejects the
                # call instead of publishing ``str(arguments)`` as JSON.
                return {"call_id": "invalid", "name": ""}
        return {"call_id": call_id, "name": name, "arguments": arguments}

    def runtime_config_for_function_output(
        self,
        config: DuplexSessionConfig,
        current: Mapping[str, object],
        item: Mapping[str, object],
        *,
        history_len: int | None = None,
    ) -> dict[str, object]:
        """Accept a tool result for the next Qwen prompt.

        Each accepted result is one ``qwen_tool_turns`` entry. Its
        ``history_len`` is the length stored when the call was published.
        ``history_len`` here is only the fallback for a result whose call was
        not published through this plugin. The next prompt splices that pair
        after the messages that existed then. A bare ``response.create`` opens
        the follow-up while those turns are still waiting.
        """
        del config
        call_id = item.get("call_id")
        output = item.get("output")
        name = item.get("name")
        arguments = item.get("arguments")
        if not isinstance(call_id, str) or not call_id:
            raise DuplexRuntimeConfigError(
                "function_call_output requires call_id",
                code="invalid_function_call_output",
            )
        if not isinstance(output, str):
            raise DuplexRuntimeConfigError(
                "function_call_output requires a string output",
                code="invalid_function_call_output",
            )
        if not isinstance(name, str) or not name or not isinstance(arguments, str):
            raise DuplexRuntimeConfigError(
                "function_call_output requires the session call name and arguments",
                code="invalid_function_call_output",
            )
        expired_ids = current.get("qwen_expired_call_ids")
        if isinstance(expired_ids, list) and call_id in expired_ids:
            raise DuplexRuntimeConfigError(
                "function_call_output is for a call outside the prompt window",
                code="invalid_function_call_output",
            )
        receipt_len = history_len if isinstance(history_len, int) else None
        history_len = _anchored_history_len(current, item, history_len)
        updated = dict(current)
        turns = [turn for turn in updated.get("qwen_tool_turns", []) if isinstance(turn, dict)]
        turns.append(
            {
                "history_len": history_len,
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": call_id, "content": output},
                ],
            }
        )
        updated["qwen_tool_turns"] = turns
        anchors = updated.get("qwen_call_anchors")
        if isinstance(anchors, Mapping):
            kept = {
                key: value
                for key, value in anchors.items()
                if isinstance(key, str) and isinstance(value, int) and key != call_id
            }
            expired: list[str] = []
            if receipt_len is not None:
                kept, expired = _split_call_anchors(kept, history_len=receipt_len)
            _remember_expired(updated, expired)
            if kept:
                updated["qwen_call_anchors"] = kept
            else:
                updated.pop("qwen_call_anchors", None)
        return updated

    def configure_sampling_params(self, *, runtime_config, defaults):
        params = copy.deepcopy(defaults)
        for value in params:
            if value is not None:
                value.output_kind = RequestOutputKind.DELTA
        if params and params[0] is not None:
            if runtime_config.get("temperature") is not None:
                params[0].temperature = float(runtime_config["temperature"])
            if isinstance(runtime_config.get("max_tokens"), int):
                params[0].max_tokens = runtime_config["max_tokens"]
        return params

    def prepare_prompt_config(self, config, *, state, payload):
        history = config.get("conversation", [])
        bind_from = 0
        lengths = []
        for key in ("qwen_tool_turns", "qwen_tool_turns_inflight"):
            raw_turns = config.get(key)
            if not isinstance(raw_turns, list):
                continue
            lengths.extend(
                turn.get("history_len")
                for turn in raw_turns
                if isinstance(turn, dict) and isinstance(turn.get("history_len"), int)
            )
        if lengths:
            # Audio committed after a tool result belongs to the new
            # question, not to every earlier user item still in history.
            bind_from = max(lengths)
        # The history accessor copies message dicts but preserves content identity.
        # Keep payloads privately, never put PCM into public conversation events.
        has_audio = bool(payload.get("audio"))
        # A committed audio item need not be last: camera/text items can arrive
        # before response.create. Search only the unanswered user turn so a new
        # payload can never overwrite an earlier, already answered utterance.
        current_contents = []
        if has_audio:
            for index, message in reversed(list(enumerate(history))):
                if index < bind_from or message.get("role") != "user":
                    break
                content = message.get("content")
                if isinstance(content, list) and any(part.get("type") == "audio_url" for part in content):
                    current_contents.append(content)
        current_is_in_history = bool(current_contents)
        if current_is_in_history:
            # The runner concatenates multiple commits made before a response.
            # Associate that single payload with all of its source audio items.
            state.audio_history = [
                (keys, value)
                for keys, value in state.audio_history
                if not any(key is content for key in keys for content in current_contents)
            ]
            state.audio_history.append((tuple(current_contents), payload))
        # Reserve one of the four audio slots for the actual submitted input,
        # including when deferred input has no trailing user history entry.
        history_limit = 4 if current_is_in_history or not has_audio else 3
        retained = state.audio_history[-history_limit:]
        while len(retained) > 1 and sum(len(value.get("audio", "")) for _, value in retained) > 8 * 1024 * 1024:
            retained.pop(0)
        state.audio_history = retained
        # Match the official demo's unit of history: consecutive user items
        # followed by their assistant reply(s). Prune whole turns, before
        # formatting, so a camera item cannot rescue an orphaned answer.
        turns = []
        for index, message in enumerate(history):
            role = message.get("role")
            if not turns or role == "system" or (role == "user" and turns[-1][1][-1].get("role") != "user"):
                turns.append((index, []))
            turns[-1][1].append(message)

        messages = []
        for index, turn in turns:
            # Keep the existing recent-message window, rounded outwards to a
            # complete turn. Never retain an answer whose input was cut off.
            if index + len(turn) <= len(history) - 16:
                continue
            if turn[0].get("role") == "assistant":
                continue
            resolved = []
            included_audio = set()
            for offset, message in enumerate(turn):
                src = index + offset
                content = message.get("content")
                if message.get("role") == "user" and isinstance(content, list):
                    if any(part.get("type") == "audio_url" for part in content):
                        audio = next((value for keys, value in retained if any(key is content for key in keys)), None)
                        if audio is None:
                            break
                        if id(audio) not in included_audio:
                            resolved.append({"role": "user", "audio_payload": audio, "_src": src})
                            included_audio.add(id(audio))
                        # Preserve text/images in mixed source items as well.
                        other_parts = [part for part in content if part.get("type") != "audio_url"]
                        if other_parts:
                            resolved.append({"role": "user", "content": other_parts, "_src": src})
                        continue
                copied = dict(message)
                copied["_src"] = src
                resolved.append(copied)
            else:
                messages.extend(resolved)
        if has_audio and not current_is_in_history:
            # History is context, not a replacement for the current request.
            # Re-encoding only history here makes Qwen answer an old question.
            messages.append({"role": "user", "audio_payload": payload, "_src": len(history)})
        return {**config, "qwen_messages": messages}

    @staticmethod
    def format_history(messages):
        """Merge user items as Qwen's official web_demo.format_history does.

        This is a model-input view, not a mutation of addressable session items.
        Empty assistant messages remain boundaries for interrupted/unheard turns.
        """
        formatted = []
        pending = []
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content")
                pending.extend([{"type": "text", "text": content}] if isinstance(content, str) else content or [])
                continue
            if pending:
                formatted.append({"role": "user", "content": pending})
                pending = []
            formatted.append(message)
        if pending:
            # The official demo puts media before the final turn's instruction.
            media = [part for part in pending if part.get("type") != "text"]
            text = [part for part in pending if part.get("type") == "text"]
            formatted.append({"role": "user", "content": media + text})
        return formatted

    @staticmethod
    def _trim_prompt_images(messages, images):
        """Drop the oldest images, with their placeholders, down to the prompt limit.

        The processor pairs ``{"type": "image"}`` parts with ``mm["image"]``
        positionally, and the placeholders read across *messages* in the same
        order as *images*, so a dropped image has to take its placeholder with
        it. Oldest first keeps what the client shared most recently.
        """
        excess = len(images) - MAX_PROMPT_IMAGES
        if excess <= 0:
            return messages, images
        logger.warning(
            "Qwen duplex prompt carries %d images; dropping the %d oldest to stay within %d",
            len(images),
            excess,
            MAX_PROMPT_IMAGES,
        )
        remaining = excess
        for message in messages:
            if remaining <= 0:
                break
            content = message.get("content")
            if not isinstance(content, list):
                continue
            kept = []
            for part in content:
                if remaining > 0 and isinstance(part, dict) and part.get("type") == "image":
                    remaining -= 1
                    continue
                kept.append(part)
            message["content"] = kept
        return [message for message in messages if message.get("content") != []], images[excess:]

    @staticmethod
    def _audio_content(payload):
        raw = base64.b64decode(payload["audio"], validate=True)
        audio = np.frombuffer(raw, dtype="<f4").copy()
        if not audio.size or not np.isfinite(audio).all():
            raise DuplexRuntimeConfigError("Audio must contain finite PCM samples")
        return [{"type": "audio"}], audio

    @staticmethod
    def _collect_tool_turns(raw: object) -> list[dict[str, object]]:
        turns: list[dict[str, object]] = []
        if not isinstance(raw, list):
            return turns
        for turn in raw:
            if not isinstance(turn, dict):
                continue
            history_len = turn.get("history_len")
            pair = turn.get("messages")
            if not isinstance(history_len, int) or not isinstance(pair, list):
                continue
            item: dict[str, object] = {"history_len": history_len, "messages": pair}
            response_id = turn.get("response_id")
            if isinstance(response_id, str) and response_id:
                item["response_id"] = response_id
            turns.append(item)
        return turns

    @staticmethod
    def _splice_tool_turns(messages, turns):
        """Insert each tool pair after the history it was accepted against.

        Returns ``(messages, remaining_turns, inflight_turns)``. Turns whose
        ``history_len`` is at or before the first kept message belonged to a
        user turn that already left the window and are dropped. Spliced turns
        leave ``remaining_turns``. A turn that already names a response keeps
        that id so the playback ACK can still drop it.
        """
        waiting = Qwen3OmniDuplexPlugin._collect_tool_turns(turns)
        if not waiting:
            cleaned = [{key: value for key, value in message.items() if key != "_src"} for message in messages]
            return cleaned, None, None
        sources = [message.get("_src") for message in messages if isinstance(message.get("_src"), int)]
        earliest = min(sources) if sources else None

        def _kept(turn: dict[str, object]) -> bool:
            history_len = turn.get("history_len")
            return earliest is None or not isinstance(history_len, int) or history_len > earliest

        waiting = [turn for turn in waiting if _kept(turn)]
        pending = sorted(waiting, key=lambda turn: turn["history_len"])
        spliced = []
        for message in messages:
            src = message.get("_src")
            src_index = src if isinstance(src, int) else 10**9
            while pending and pending[0]["history_len"] <= src_index:
                spliced.extend(pending.pop(0)["messages"])
            cleaned = {key: value for key, value in message.items() if key != "_src"}
            spliced.append(cleaned)
        for turn in pending:
            spliced.extend(turn["messages"])
        return spliced, [], waiting

    async def prepare_append_plan(self, **kwargs):
        # History mutation has already happened on the session loop. The worker
        # owns this snapshot and never reads or mutates live session state.
        snapshot = copy.deepcopy(kwargs)
        return await asyncio.to_thread(self.plan_append, **snapshot)

    def plan_append(
        self, *, request_id, fence, session_config, runtime_config, seq, turn_seq, payload, final, sampling_params
    ):
        if not final or not isinstance(payload, dict):
            raise DuplexRuntimeConfigError("Qwen generation requires a committed audio turn")
        has_audio = bool(payload.get("audio"))
        if not has_audio and payload.get("type") != "conversation":
            raise DuplexRuntimeConfigError("Qwen generation requires a committed audio turn")
        messages = []
        if runtime_config.get("instructions"):
            messages.append({"role": "system", "content": runtime_config["instructions"]})
        history = session_config.get("qwen_messages")
        if history is None and has_audio:
            # Nothing prepared the conversation, so the committed payload is
            # the whole turn. An empty prepared list is not that: it means the
            # preparation ran and found nothing to say.
            history = [{"role": "user", "audio_payload": payload}]
        if not history:
            raise DuplexRuntimeConfigError("Qwen generation requires committed audio or conversation history")
        # A pair still in flight belongs to a response whose answer is not in
        # history yet. Put it back on the waiting list so this prompt uses the
        # same splice. The response id stays, so that response's ACK can drop it.
        folded = [
            *self._collect_tool_turns(runtime_config.get("qwen_tool_turns_inflight")),
            *self._collect_tool_turns(session_config.get("qwen_tool_turns")),
        ]
        history, remaining_tool_turns, inflight_tool_turns = self._splice_tool_turns(list(history), folded)
        if has_audio and runtime_config.get("initial_user_text"):
            history = [*history, {"role": "user", "content": runtime_config["initial_user_text"]}]
        audios = []
        images = []
        rendered = []
        for message in history:
            if message.get("role") == "tool" or message.get("tool_calls"):
                rendered.append(message)
                continue
            if "audio_payload" not in message:
                content = message.get("content")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if part.get("type") == "image_url":
                            encoded = part["image_url"]["url"].split(",", 1)[1]
                            with open_image(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                                image.thumbnail((448, 448))
                                images.append(image.convert("RGB").copy())
                            parts.append({"type": "image"})
                        elif part.get("type") == "text":
                            parts.append(part)
                    rendered.append({"role": message["role"], "content": parts})
                else:
                    rendered.append(message)
                continue
            content, audio = self._audio_content(message["audio_payload"])
            audios.append((audio, 16000))
            rendered.append({"role": "user", "content": content})
        rendered, images = self._trim_prompt_images(rendered, images)
        messages.extend(self.format_history(rendered))
        tools = chat_template_tools(runtime_config.get("qwen_tools"))
        if not tools:
            extra_body = session_config.get("extra_body") if isinstance(session_config, Mapping) else None
            if isinstance(extra_body, Mapping):
                tools = chat_template_tools(extra_body.get("realtime_tools"))
        template_kwargs: dict[str, object] = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            template_kwargs["tools"] = tools
        prompt = self.processor.apply_chat_template(messages, **template_kwargs)
        mm = {"audio": audios} if audios else {}
        if images:
            mm["image"] = images
        logger.info(
            "Qwen duplex visual input request=%s images=%d sizes=%s audios=%d",
            request_id,
            len(images),
            [image.size for image in images],
            len(audios),
        )
        # Opt-in capture of the exact decoded images submitted to the processor.
        # Request IDs are hashed because they need not be safe path components.
        debug_dir = os.environ.get("VLLM_OMNI_QWEN_VISUAL_DEBUG_DIR")
        if debug_dir:
            try:
                target = Path(debug_dir) / hashlib.sha256(request_id.encode()).hexdigest()[:24]
                target.mkdir(parents=True, exist_ok=True)
                filenames = []
                for index, image in enumerate(images):
                    filename = f"image_{index:02d}.png"
                    image.save(target / filename)
                    filenames.append(filename)
                (target / "input.json").write_text(
                    json.dumps(
                        {"request_id": request_id, "images": filenames, "audio_count": len(audios), "prompt": prompt},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                logger.info("Qwen duplex visual input capture request=%s path=%s", request_id, target)
            except (OSError, ValueError):
                logger.exception("Failed to capture Qwen visual input request=%s", request_id)
        prompt_payload: dict[str, object] = {"prompt": prompt, "multi_modal_data": mm}
        if remaining_tool_turns is not None:
            # Private write-back for the session after a successful submit.
            prompt_payload["_qwen_tool_turns_remaining"] = remaining_tool_turns
        if inflight_tool_turns is not None:
            prompt_payload["_qwen_tool_turns_inflight"] = inflight_tool_turns
        kept_call_ids = set(_call_ids_of(remaining_tool_turns)) | set(_call_ids_of(inflight_tool_turns))
        dropped_call_ids = [call_id for call_id in _call_ids_of(folded) if call_id not in kept_call_ids]
        if dropped_call_ids:
            prompt_payload["_qwen_dropped_tool_call_ids"] = dropped_call_ids
        conversation = session_config.get("conversation")
        anchors = runtime_config.get("qwen_call_anchors")
        if isinstance(conversation, list) and isinstance(anchors, Mapping):
            # Drop anchors whose user turn has already left the prompt window.
            kept_anchors, expired_anchors = _split_call_anchors(anchors, history_len=len(conversation))
            prompt_payload["_qwen_call_anchors"] = kept_anchors
            if expired_anchors:
                expired_state: dict[str, object] = {
                    "qwen_expired_call_ids": runtime_config.get("qwen_expired_call_ids")
                }
                _remember_expired(expired_state, expired_anchors)
                prompt_payload["_qwen_expired_call_ids"] = expired_state["qwen_expired_call_ids"]
                # Reconcile appends these onto the live list. The full window above
                # still feeds anchor removal, but must not be appended again.
                prompt_payload["_qwen_newly_expired_call_ids"] = expired_anchors
        # The Talker picks the checkpoint speaker token from this field; the
        # chat-completions path sets it in serving_chat, so the duplex plan
        # must forward the session voice the same way (#8815). runtime_config
        # already carries the normalized voice; setdefault keeps any other
        # additional_information keys this plan may carry.
        voice = runtime_config.get("voice")
        if isinstance(voice, str) and voice.strip():
            prompt_payload.setdefault("additional_information", {})["speaker"] = [voice]
        return DuplexAppendPlan(prompt=prompt_payload)

    def decide_output(self, **kwargs):
        return None

    def data_plane_context(self, **kwargs):
        return kwargs
