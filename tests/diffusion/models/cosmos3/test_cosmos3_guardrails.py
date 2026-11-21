# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for the real Cosmos3 guardrail adapter.

These live apart from ``test_cosmos3_pipeline.py`` because that module installs
an autouse fixture replacing ``guardrails`` in ``sys.modules`` with a stub, so a
test there can never reach the code below.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# RetinaFace (via cosmos_guardrail) disables autograd at import time. Restore
# the caller's grad mode so collection does not affect unrelated tests.
with torch.set_grad_enabled(torch.is_grad_enabled()):
    from vllm_omni.diffusion.models.cosmos3 import guardrails

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("modify_frames", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_float_guardrails_preserve_input_and_modified_output_bytes(
    monkeypatch: pytest.MonkeyPatch, modify_frames: bool, dtype: torch.dtype, device: str
) -> None:
    from diffusers.video_processor import VideoProcessor

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Include bf16 rounding boundaries, saturation, and both ends of the range.
    values = torch.tensor([-1.4, -1, -0.50390625, 0, 0.50390625, 1, 1.4], dtype=dtype)
    video = values.view(1, 1, 7, 1, 1).expand(1, 3, 7, 2, 2).clone().to(device)
    original = video.clone()
    captured: list[np.ndarray] = []

    def check(frames: np.ndarray) -> np.ndarray:
        captured.append(frames.copy())
        if modify_frames:
            # Exercise the adapter's handling of modified guardrail output,
            # including every possible byte through the old float round trip.
            return np.arange(256 * 3, dtype=np.int64).astype(np.uint8).reshape(1, 16, 16, 3)
        return frames

    monkeypatch.setattr(guardrails, "_video_guardrail", check)
    checked = guardrails.check_video_safety(video)
    float_video = video[0].detach().cpu().float().clamp(-1, 1)
    expected_input = ((float_video * 0.5 + 0.5).permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    np.testing.assert_array_equal(captured[0], expected_input)
    expected_output = (
        np.arange(256 * 3, dtype=np.int64).astype(np.uint8).reshape(1, 16, 16, 3) if modify_frames else expected_input
    )
    displayed = VideoProcessor(vae_scale_factor=16).postprocess_video(checked, output_type="np")
    np.testing.assert_array_equal(np.round(displayed[0] * 255).astype(np.uint8), expected_output)
    assert checked.dtype == torch.float32
    assert checked.device == video.device
    assert torch.equal(video, original)


def test_check_video_safety_is_a_no_op_when_no_guardrail_is_loaded() -> None:
    frames = torch.zeros(1, 3, 2, 4, 4)

    assert guardrails.check_video_safety(frames) is frames


def test_check_video_safety_still_round_trips_the_vae_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers that pass float [-1, 1] must get float [-1, 1] back."""
    captured: list[np.ndarray] = []

    def _guardrail(frames: np.ndarray) -> np.ndarray:
        captured.append(frames)
        return frames

    monkeypatch.setattr(guardrails, "_video_guardrail", _guardrail)
    video = torch.zeros(1, 3, 2, 4, 4)

    checked = guardrails.check_video_safety(video)

    # The classifier receives display bytes; the adapter returns normalized floats.
    assert captured[0].dtype == np.uint8
    assert captured[0].shape == (2, 4, 4, 3)
    assert checked.shape == video.shape
    assert checked.dtype == torch.float32
    torch.testing.assert_close(checked, video, atol=1 / 127.5, rtol=0)


@pytest.mark.parametrize(("batch_size", "num_frames"), [(2, 1), (3, 4)], ids=["t2i_n2", "video_b3_t4"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_check_video_safety_checks_every_sample_of_a_batch(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, num_frames: int, dtype: torch.dtype, device: str
) -> None:
    """Every sample of a [B, C, T, H, W] batch is checked once, in order (#10648)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Give every (sample, frame) its own value so a dropped, duplicated or
    # swapped sample changes the bytes the guardrail receives.
    values = torch.linspace(-1, 1, batch_size * num_frames).to(dtype).view(batch_size, 1, num_frames, 1, 1)
    video = values.expand(batch_size, 3, num_frames, 4, 4).clone().to(device)
    original = video.clone()
    captured: list[np.ndarray] = []

    def check(frames: np.ndarray) -> np.ndarray:
        captured.append(frames.copy())
        # Stamp the call order so each output sample traces back to one call.
        return np.full_like(frames, 50 * len(captured))

    monkeypatch.setattr(guardrails, "_video_guardrail", check)
    checked = guardrails.check_video_safety(video)

    assert len(captured) == batch_size
    assert len({frames.tobytes() for frames in captured}) == batch_size
    for index, frames in enumerate(captured):
        sample = video[index].detach().cpu().float().clamp(-1, 1)
        expected_input = ((sample * 0.5 + 0.5).permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
        np.testing.assert_array_equal(frames, expected_input)
    assert checked.shape == video.shape
    assert checked.dtype == torch.float32
    assert checked.device == video.device
    for index in range(batch_size):
        expected = torch.full(video.shape[1:], 50 * (index + 1), dtype=torch.float32) / 127.5 - 1.0
        torch.testing.assert_close(checked[index], expected.to(video.device), rtol=0, atol=0)
    assert torch.equal(video, original)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_check_video_safety_checks_a_4d_video_once(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    """A single [C, T, H, W] video is checked once and keeps its 4-D shape."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    video = torch.linspace(-1, 1, 3).view(1, 3, 1, 1).expand(3, 3, 4, 4).clone().to(device)
    captured: list[np.ndarray] = []

    def check(frames: np.ndarray) -> np.ndarray:
        captured.append(frames.copy())
        return frames

    monkeypatch.setattr(guardrails, "_video_guardrail", check)
    checked = guardrails.check_video_safety(video)

    assert len(captured) == 1
    assert captured[0].shape == (3, 4, 4, 3)
    assert checked.shape == (3, 3, 4, 4)
    assert checked.dtype == torch.float32
    assert checked.device == video.device
    torch.testing.assert_close(checked, video, atol=1 / 127.5, rtol=0)
