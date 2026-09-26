# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The sharded clip budget scales with device memory from its calibration."""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.models.seedvr2 import video
from vllm_omni.diffusion.models.seedvr2.video import (
    CALIBRATED_SHARDED_CLIP_PIXELS,
    VALIDATED_SHARDED_CLIP_PIXELS,
    scaled_clip_pixels,
    sharded_budget,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
MIB = 1024**2
FRAME = 2560 * 1472
RTX_5090 = 32607 * MIB
B300 = 274113 * MIB


def test_the_calibrated_device_keeps_the_calibrated_budget() -> None:
    assert scaled_clip_pixels(FRAME, RTX_5090) == CALIBRATED_SHARDED_CLIP_PIXELS


@pytest.mark.parametrize("memory", [None, 8 * 1024 * MIB, 24 * 1024 * MIB])
def test_smaller_or_unknown_devices_never_admit_less(memory: int | None) -> None:
    assert scaled_clip_pixels(FRAME, memory) == CALIBRATED_SHARDED_CLIP_PIXELS


def test_more_memory_admits_more_up_to_the_validated_clip() -> None:
    budgets = [scaled_clip_pixels(FRAME, gib * 1024 * MIB) for gib in (48, 80, 141, 192, 268, 1024)]
    assert budgets == sorted(budgets)
    assert budgets[0] > CALIBRATED_SHARDED_CLIP_PIXELS
    assert budgets[-1] == VALIDATED_SHARDED_CLIP_PIXELS


def test_a_b300_admits_the_validated_clip() -> None:
    # 1536x2688 is the 2x upscale of a 768x1344 source.
    assert scaled_clip_pixels(1536 * 2688, B300) == VALIDATED_SHARDED_CLIP_PIXELS
    assert VALIDATED_SHARDED_CLIP_PIXELS // (1536 * 2688) == 513


def test_the_budget_follows_the_smallest_visible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS", raising=False)
    monkeypatch.setattr(video, "_device_memory", lambda: RTX_5090)
    assert sharded_budget() == (FRAME, CALIBRATED_SHARDED_CLIP_PIXELS)
    monkeypatch.setattr(video, "_device_memory", lambda: B300)
    assert sharded_budget()[1] == scaled_clip_pixels(FRAME, B300)


def test_an_explicit_budget_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS", "12345")
    monkeypatch.setattr(video, "_device_memory", lambda: B300)
    assert sharded_budget()[1] == 12345
