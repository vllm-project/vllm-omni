# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Applying one ``playback.ack`` to a duplex session.

The client reports how much of a response it has actually played; the session
uses that to commit or truncate the assistant item in conversation history, and
to release the response's playback state once it is fully played.

Extracted from ``DuplexSessionRunner`` because the only runner state it touched
was ``session``. It returns the events it produces rather than emitting them,
so the rules can be exercised without a runner.
"""

from __future__ import annotations

from vllm_omni.engine.duplex.events import DuplexEvent, PlaybackAcknowledged, error_event
from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.engine.duplex.session.lease import DuplexLeaseActivity


def apply_playback_ack(
    session: DuplexEngineSession,
    event: dict[str, object],
    *,
    require_audio_complete: bool = False,
) -> list[DuplexEvent]:
    """Apply an ACK, retaining incomplete replies when calibration is enabled."""
    played_ms = event.get("played_ms", event.get("audio_ms", 0))
    committed_ms = event.get("committed_ms")
    if not isinstance(played_ms, int | float):
        return [error_event("bad_event", "playback.ack requires played_ms")]
    try:
        session.touch_lease(DuplexLeaseActivity.PLAYBACK_ACK)
    except Exception:
        pass
    committed_cursor = int(committed_ms) if isinstance(committed_ms, int | float) else int(played_ms)
    raw_item_id = event.get("item_id")
    item_id: str | None = raw_item_id if isinstance(raw_item_id, str) and raw_item_id else None
    response_id = event.get("response_id")
    response_id = response_id if isinstance(response_id, str) and response_id else None
    if item_id is None:
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
    if (
        response_id is None
        and item_id is None
        and not session.pending_history_item_ids
        and session.last_assistant_full_message is not None
    ):
        # Resolve the legacy single-history-item target before any validation
        # or cursor mutation, just like explicit, pending, and active targets.
        assistant_item_ids = [
            known_item_id
            for known_item_id, message in session.history_item_ids.items()
            if message.get("role") == "assistant"
        ]
        # Imported context with an arbitrary ID cannot identify a response.
        # Leave it unselected so an ordinary ID-less ACK remains progress-only.
        if len(assistant_item_ids) == 1 and assistant_item_ids[0].startswith("item_"):
            item_id = assistant_item_ids[0]
            response_id = item_id.removeprefix("item_")
    audio_only = response_id is not None and session.has_audio_only_response(response_id)
    history_is_late = False
    if response_id is not None:
        response_item_id = f"item_{response_id}"
        if item_id is None:
            item_id = response_item_id
        elif item_id != response_item_id:
            return [error_event("playback_item_mismatch", "playback.ack item_id must match item_<response_id>.")]
        if not session.has_assistant_response_item(response_id, item_id):
            return [
                error_event("playback_item_not_found", f"No assistant response item is registered for {response_id}.")
            ]
        history_is_late = session.playback_ack_is_too_late(response_id, item_id)
        if history_is_late and not audio_only:
            return [
                error_event("playback_ack_too_late", "playback.ack arrived after a later user input was committed.")
            ]
    elif item_id is not None:
        return [error_event("playback_item_not_found", "playback.ack requires a response-owned assistant item.")]
    hard_truncate = event.get("truncate") is True
    if hard_truncate and (item_id is None or not session.history_item_has_audio(item_id)):
        return [error_event("bad_event", "playback.ack truncate requires one identifiable item that produced audio")]
    if item_id is not None and not history_is_late:
        session.reserve_history_item(item_id)
    if hard_truncate:
        playback = session.acknowledge_playback(int(played_ms), committed_cursor, response_id=response_id)
        playback = session.truncate_playback_commit(committed_cursor, response_id=response_id)
    else:
        playback = session.acknowledge_playback(int(played_ms), committed_cursor, response_id=response_id)
    committed_history = False
    if isinstance(item_id, str) and item_id:
        if audio_only:
            # Keep an acknowledged turn boundary, but never append an older
            # answer after a later input. Empty text does not settle tool output.
            if not history_is_late:
                session.register_history_item(item_id, {"role": "assistant", "content": ""})
            if hard_truncate:
                session.truncate_history_item(item_id, audio_end_ms=committed_cursor, playback=playback, hard=True)
        else:
            if (
                item_id == f"item_{response_id}"
                and item_id not in session.history_item_ids
                and item_id not in session.pending_history_item_ids
            ):
                session.register_history_item(item_id, None)
            committed_history = session.truncate_history_item(
                item_id,
                audio_end_ms=committed_cursor,
                playback=playback,
                hard=hard_truncate,
            )
    events: list[DuplexEvent] = [
        PlaybackAcknowledged(
            details={
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
    ]
    session.finalize_response_playback(response_id, require_audio_complete=require_audio_complete)
    return events
