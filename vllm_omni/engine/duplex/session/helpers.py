# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Small reads and payload builders over one duplex session.

None of these hold state or decide anything: they answer a question about the
session, or shape one of the wire payloads its events carry. They were methods
on the runner only because that is where the session reference lived, which
made every component that needed one a reason to reach back into the runner.

Functions, not a component, because there is nothing here to own.
"""

from __future__ import annotations

import binascii
from typing import TYPE_CHECKING

import pybase64 as base64

from vllm_omni.engine.duplex.config import DuplexPlaybackCommitPolicy
from vllm_omni.engine.duplex.contracts import DuplexFence, duplex_resource_request_id
from vllm_omni.engine.duplex.events import ErrorEvent, OverlapDecision, error_event

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vllm_omni.engine.duplex.session.context import DuplexSessionTasks
    from vllm_omni.engine.duplex.session.engine_session import DuplexCommittedInput, DuplexEngineSession


# --------------------------------------------------------------------------- #
# Questions about the session                                                 #
# --------------------------------------------------------------------------- #


def stage0_request_id(session: DuplexEngineSession, epoch: int) -> str:
    return duplex_resource_request_id(DuplexFence(session.session_id, epoch=epoch), "stage0")


def response_in_progress(session: DuplexEngineSession, tasks: DuplexSessionTasks) -> bool:
    """Whether the model still owns the turn.

    Broader than ``active_response_id``: audio already sent but not yet played
    back, and an append still in flight, both mean the turn is not free.
    """
    if session.active_response_id is not None:
        return True
    if assistant_playback_active(session):
        return True
    active_task = tasks.active_response_task
    if active_task is not None and not active_task.done():
        return True
    return tasks.has_response_bound_append_tasks()


def assistant_playback_active(session: DuplexEngineSession) -> bool:
    """Whether the client is still playing audio the session has sent."""
    return (
        session.config.playback_commit_policy == DuplexPlaybackCommitPolicy.ACK_ONLY.value
        and session.playback.sent_ms > session.playback.committed_ms
    )


def audio_payload_size_bytes(payload: Mapping[str, object]) -> int:
    audio = payload.get("audio") or payload.get("data")
    if not isinstance(audio, str):
        return 0
    try:
        return len(base64.b64decode(audio, validate=True))
    except (ValueError, binascii.Error):
        return 0


# --------------------------------------------------------------------------- #
# Turn transitions                                                            #
# --------------------------------------------------------------------------- #


def advance_barge_in_epoch(session: DuplexEngineSession) -> tuple[int, dict[str, int]]:
    """Start a new epoch for a barge-in, returning it with the playback it cut off."""
    old_playback = session.playback.as_dict()
    new_epoch = session.barge_in()
    session.clear_playback_cursor()
    return new_epoch, old_playback


def commit_played_response_history(
    session: DuplexEngineSession,
    response_id: str | None,
    committed_ms: int,
) -> None:
    """Truncate an interrupted response in history to what the client actually heard."""
    if not response_id or committed_ms < 0:
        return
    session.truncate_history_item(
        f"item_{response_id}",
        audio_end_ms=committed_ms,
        playback=session.playback_for_response(response_id),
    )


def commit_audio_input(
    session: DuplexEngineSession,
    *,
    realtime_item_id: object | None = None,
    transcript: object | None = None,
    turn_id: int | None = None,
) -> DuplexCommittedInput:
    clean_transcript = transcript.strip() if isinstance(transcript, str) else None
    committed = session.commit_audio_input(
        transcript=clean_transcript or None,
        turn_id=turn_id,
    )
    if isinstance(realtime_item_id, str) and realtime_item_id:
        session.register_history_item(realtime_item_id, committed.message)
    return committed


# --------------------------------------------------------------------------- #
# Payloads                                                                    #
# --------------------------------------------------------------------------- #


def barge_in_unsupported_error() -> ErrorEvent:
    return error_event("barge_in_unsupported", "Barge-in is not supported by this duplex model")


def overlap_decision_event(session: DuplexEngineSession, decision: dict[str, object]) -> OverlapDecision:
    details: dict[str, object] = {
        "type": "overlap.decision",
        "session_id": session.session_id,
        "epoch": session.epoch,
        "policy": session.config.overlap_policy,
        **decision,
    }
    action = decision.get("action")
    reason = decision.get("reason")
    return OverlapDecision(
        policy=session.config.overlap_policy,
        action=action if isinstance(action, str) else None,
        reason=reason if isinstance(reason, str) else None,
        details=details,
    )


def input_committed_payload(
    session: DuplexEngineSession,
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


def audio_committed_payload(
    session: DuplexEngineSession,
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
        message = {"role": "user", "content": [input_audio_part]}
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
