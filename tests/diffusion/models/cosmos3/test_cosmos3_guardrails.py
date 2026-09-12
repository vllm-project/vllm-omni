# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the real Cosmos3 guardrail adapter.

These live apart from ``test_cosmos3_pipeline.py`` because that module installs
an autouse fixture replacing ``guardrails`` in ``sys.modules`` with a stub, so a
test there can never reach the code below.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.cosmos3 import guardrails

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_check_video_safety_is_a_no_op_when_no_guardrail_is_loaded() -> None:
    frames = torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8)

    assert guardrails.check_video_safety(frames) is frames


def test_check_video_safety_skips_conversions_for_display_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """uint8 channel-last frames are already the guardrail's own format.

    The float path has to denormalize, scale, round and permute on the way in and
    undo all of it on the way out. Frames that arrive display-ready skip both.
    """
    seen: list[np.ndarray] = []

    def _guardrail(frames: np.ndarray) -> np.ndarray:
        seen.append(frames)
        return frames

    monkeypatch.setattr(guardrails, "_video_guardrail", _guardrail)
    frames = torch.arange(2 * 4 * 4 * 3, dtype=torch.uint8).reshape(1, 2, 4, 4, 3)

    checked = guardrails.check_video_safety(frames)

    assert len(seen) == 1
    assert seen[0].dtype == np.uint8
    assert seen[0].shape == (2, 4, 4, 3)
    assert checked.dtype == torch.uint8
    assert torch.equal(checked, frames)


def test_check_video_safety_accepts_unbatched_display_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guardrails, "_video_guardrail", lambda frames: frames)
    frames = torch.arange(2 * 4 * 4 * 3, dtype=torch.uint8).reshape(2, 4, 4, 3)

    checked = guardrails.check_video_safety(frames)

    assert checked.shape == frames.shape
    assert torch.equal(checked, frames)


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (torch.zeros(1, 2, 4, 4, 4, dtype=torch.uint8), "display-frame guardrails expect"),
        (torch.zeros(2, 2, 4, 4, 3, dtype=torch.uint8), "one video per request"),
    ],
)
def test_check_video_safety_validates_display_frame_contract(
    monkeypatch: pytest.MonkeyPatch,
    frames: torch.Tensor,
    message: str,
) -> None:
    monkeypatch.setattr(guardrails, "_video_guardrail", lambda value: value)

    with pytest.raises(ValueError, match=message):
        guardrails.check_video_safety(frames)


def test_check_video_safety_still_round_trips_the_vae_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers that pass float [-1, 1] must get float [-1, 1] back."""
    captured: list[np.ndarray] = []

    def _guardrail(frames: np.ndarray) -> np.ndarray:
        captured.append(frames)
        return frames

    monkeypatch.setattr(guardrails, "_video_guardrail", _guardrail)
    video = torch.zeros(1, 3, 2, 4, 4)

    checked = guardrails.check_video_safety(video)

    # The guardrail sees uint8 channel-last either way; only the wrapping differs.
    assert captured[0].dtype == np.uint8
    assert captured[0].shape == (2, 4, 4, 3)
    assert checked.shape == video.shape
    assert checked.dtype == torch.float32
    torch.testing.assert_close(checked, video, atol=1 / 127.5, rtol=0)
