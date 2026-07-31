from __future__ import annotations

import pytest

from vllm_omni.experimental.fullduplex.openai.commit_policy import (
    CommitAction,
    CommitSnapshot,
    decide_commit_action,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("case", "snapshot", "expected"),
    [
        (
            "first idle commit",
            CommitSnapshot(
                auto_responds=True,
                speech_since_commit=True,
                active_response_id=None,
                overlap_speech_ms=0,
                native_response_in_progress=False,
                playback_active=False,
            ),
            CommitAction.START_AUTO_RESPONSE,
        ),
        (
            "native auto-response remains model-owned during playback",
            CommitSnapshot(
                auto_responds=True,
                speech_since_commit=True,
                active_response_id=None,
                overlap_speech_ms=800,
                native_response_in_progress=True,
                playback_active=True,
            ),
            CommitAction.START_AUTO_RESPONSE,
        ),
        (
            "native auto-response commit without speech",
            CommitSnapshot(
                auto_responds=True,
                speech_since_commit=False,
                active_response_id=None,
                overlap_speech_ms=0,
                native_response_in_progress=True,
                playback_active=False,
            ),
            CommitAction.COMMIT_ONLY,
        ),
        (
            "non-auto response overlap",
            CommitSnapshot(
                auto_responds=False,
                speech_since_commit=True,
                active_response_id=None,
                overlap_speech_ms=800,
                native_response_in_progress=True,
                playback_active=False,
            ),
            CommitAction.DEFER_ACTIVE_RESPONSE,
        ),
        (
            "non-auto playback overlap",
            CommitSnapshot(
                auto_responds=False,
                speech_since_commit=True,
                active_response_id=None,
                overlap_speech_ms=800,
                native_response_in_progress=False,
                playback_active=True,
            ),
            CommitAction.DEFER_ACTIVE_RESPONSE,
        ),
        (
            "non-auto commit without overlap",
            CommitSnapshot(
                auto_responds=False,
                speech_since_commit=True,
                active_response_id=None,
                overlap_speech_ms=0,
                native_response_in_progress=False,
                playback_active=False,
            ),
            CommitAction.COMMIT_ONLY,
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_decide_commit_action_covers_realtime_lifecycle_sequences(
    case: str,
    snapshot: CommitSnapshot,
    expected: CommitAction,
) -> None:
    del case

    assert decide_commit_action(snapshot) is expected
