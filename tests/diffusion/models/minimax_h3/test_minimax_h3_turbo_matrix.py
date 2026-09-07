# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contract coverage for the LightX2V MiniMax-H3 Turbo artifact matrix."""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.models.minimax_h3.lora import (
    _TURBO_AUDIO_SHIFT,
    _TURBO_VIDEO_SHIFT_544P,
    _TURBO_VIDEO_SHIFT_768P,
    TurboSpec,
    parse_turbo_filename,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# The eight Diffusers-layout artifacts published at lightx2v/Minimax-h3-Turbo,
# with the contract each one implies. The ComfyUI export of an artifact is a
# fused-QKV re-packing of the same weights and is not served.
PUBLISHED = [
    ("minimax_h3_fl2v_turbo_4step_v0.1.safetensors", "fl2v", 4, _TURBO_VIDEO_SHIFT_544P),
    ("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors", "fl2v", 4, _TURBO_VIDEO_SHIFT_768P),
    ("minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors", "fl2v", 4, _TURBO_VIDEO_SHIFT_768P),
    ("minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors", "fl2v", 4, _TURBO_VIDEO_SHIFT_768P),
    ("minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors", "fl2v", 8, _TURBO_VIDEO_SHIFT_544P),
    ("minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors", "fl2v", 8, _TURBO_VIDEO_SHIFT_768P),
    ("minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors", "ref2v", 4, _TURBO_VIDEO_SHIFT_544P),
    ("minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors", "ref2v", 8, _TURBO_VIDEO_SHIFT_768P),
]


@pytest.mark.parametrize(("filename", "task", "steps", "video_shift"), PUBLISHED)
def test_every_published_artifact_is_recognised(filename: str, task: str, steps: int, video_shift: float) -> None:
    spec = parse_turbo_filename(filename)
    assert spec is not None, filename
    assert spec.filename == filename
    assert spec.task_family == task
    assert spec.denoise_steps == steps
    assert spec.video_shift == video_shift
    assert spec.audio_shift == _TURBO_AUDIO_SHIFT
    # Until the loader reads metadata, alpha is LightX2V's reference default.
    assert spec.rank == 128
    assert spec.alpha == 8.0


def test_all_eight_artifacts_are_covered() -> None:
    assert len(PUBLISHED) == 8
    assert len({name for name, *_ in PUBLISHED}) == 8


@pytest.mark.parametrize(
    "filename",
    [
        "adapter_model.safetensors",
        "minimax_h3_fl2v_turbo_v1.0_768p_bf16.safetensors",
        "minimax_h3_t2v_turbo_4step_v1.0.safetensors",
        "minimax_h3_fl2v_turbo_0step_v1.0.safetensors",
        # The ComfyUI exports are a different tensor layout, not a variant of
        # the supported one; the loader refuses them by name.
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    ],
)
def test_unrelated_names_are_rejected(filename: str) -> None:
    assert parse_turbo_filename(filename) is None


def _spec(filename: str) -> TurboSpec:
    spec = parse_turbo_filename(filename)
    assert spec is not None
    return spec


def test_sigma_points_follow_the_step_count() -> None:
    assert _spec("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors").sigma_points == 5
    assert _spec("minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors").sigma_points == 9


def test_task_family_decides_supported_tasks() -> None:
    fl2v = _spec("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors")
    ref2v = _spec("minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors")
    assert fl2v.supported_tasks == frozenset({"t2va", "fl2va"})
    assert ref2v.supported_tasks == frozenset({"ref2va"})
