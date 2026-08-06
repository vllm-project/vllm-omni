from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommitAction(str, Enum):
    """How one native audio commit participates in response scheduling."""

    START_AUTO_RESPONSE = "start_auto_response"
    DEFER_ACTIVE_RESPONSE = "defer_active_response"
    COMMIT_ONLY = "commit_only"


@dataclass(frozen=True, slots=True)
class CommitSnapshot:
    auto_responds: bool
    speech_since_commit: bool
    active_response_id: str | None
    overlap_speech_ms: int
    native_response_in_progress: bool
    playback_active: bool


def decide_commit_action(snapshot: CommitSnapshot) -> CommitAction:
    """Choose response scheduling from an immutable session snapshot."""
    if snapshot.auto_responds:
        if snapshot.speech_since_commit:
            return CommitAction.START_AUTO_RESPONSE
        return CommitAction.COMMIT_ONLY
    if snapshot.native_response_in_progress and snapshot.overlap_speech_ms > 0:
        return CommitAction.DEFER_ACTIVE_RESPONSE
    if snapshot.playback_active and snapshot.overlap_speech_ms > 0:
        return CommitAction.DEFER_ACTIVE_RESPONSE
    return CommitAction.COMMIT_ONLY


__all__ = ["CommitAction", "CommitSnapshot", "decide_commit_action"]
