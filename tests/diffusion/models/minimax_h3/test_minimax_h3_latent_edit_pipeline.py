# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pipeline-level MiniMax H3 latent initialization tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_path_backed_source_audio_is_duration_capped_while_loading(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3

    expected = (torch.ones(2, 32), 32000)
    calls = []

    def fake_load(path, *, duration_seconds=None):
        calls.append((path, duration_seconds))
        return expected

    monkeypatch.setattr(pipeline_minimax_h3, "load_audio_file", fake_load)

    actual = pipeline_minimax_h3._load_audio(
        "large-source.wav",
        duration_seconds=107 / 24,
    )

    assert actual is expected
    assert calls == [("large-source.wav", 107 / 24)]


def test_source_audio_waveform_is_trimmed_but_not_padded_before_encode():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _trim_audio_waveform

    short = torch.arange(6, dtype=torch.float32)
    fitted, sample_rate = _trim_audio_waveform(short, 4, duration_seconds=2.0)
    torch.testing.assert_close(fitted, short)
    assert sample_rate == 4

    long = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    fitted, sample_rate = _trim_audio_waveform(long, 2, duration_seconds=2.0)
    torch.testing.assert_close(fitted, long[:, :4])
    assert sample_rate == 2


def test_source_audio_latent_fit_is_channel_major_and_regenerates_missing_tail():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _fit_audio_edit_rows

    # Two channels, two source frames, one latent feature. Flattening is
    # channel-major: ch0(t0,t1), then ch1(t0,t1).
    source = torch.tensor([[1.0], [2.0], [11.0], [12.0]])
    mask = torch.zeros(8)
    restore_mask = torch.full((8,), 0.25)
    fitted, fitted_mask, fitted_restore_mask = _fit_audio_edit_rows(
        source,
        2,
        target_audio_t=4,
        mask_rows=mask,
        restore_mask_rows=restore_mask,
    )
    torch.testing.assert_close(
        fitted[:, 0],
        torch.tensor([1.0, 2.0, 0.0, 0.0, 11.0, 12.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        fitted_mask,
        torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
    )
    torch.testing.assert_close(
        fitted_restore_mask,
        torch.tensor([0.25, 0.25, 1.0, 1.0, 0.25, 0.25, 1.0, 1.0]),
    )

    longer = torch.arange(12, dtype=torch.float32).reshape(12, 1)
    fitted, fitted_mask, fitted_restore_mask = _fit_audio_edit_rows(
        longer,
        6,
        target_audio_t=3,
        mask_rows=torch.full((6,), 0.5),
    )
    torch.testing.assert_close(fitted[:, 0], torch.tensor([0.0, 1.0, 2.0, 6.0, 7.0, 8.0]))
    torch.testing.assert_close(fitted_mask, torch.full((6,), 0.5))
    torch.testing.assert_close(fitted_restore_mask, torch.full((6,), 0.5))


def test_prepare_latent_edit_inputs_validates_masks_and_source_dependencies(tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.errors import OmniClientError

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    common = dict(
        width=96,
        height=64,
        num_frames=22,
        latent_t=2,
        audio_t=3,
        duration_seconds=22 / 24,
        workdir=str(tmp_path),
    )

    with pytest.raises(OmniClientError, match="finite"):
        pipeline._prepare_latent_edit_inputs(
            {"video_noise_mask": float("nan")},
            **common,
        )
    with pytest.raises(OmniClientError, match="requires source_video"):
        pipeline._prepare_latent_edit_inputs(
            {"video_noise_mask": 0.0},
            **common,
        )
    with pytest.raises(OmniClientError, match="requires source_audio or source_video"):
        pipeline._prepare_latent_edit_inputs(
            {"audio_noise_mask": 0.0},
            **common,
        )
    with pytest.raises(OmniClientError, match="requires at least one"):
        pipeline._prepare_latent_edit_inputs(
            {"source_video": "source.mp4"},
            **common,
        )

    # Only exact all-one masks are a no-op: the model mask for 0.999 rounds to
    # one, but its raw output mask still restores a small source fraction.
    with pytest.raises(OmniClientError, match="requires source_video"):
        pipeline._prepare_latent_edit_inputs(
            {"video_noise_mask": 0.999},
            **common,
        )
    result = pipeline._prepare_latent_edit_inputs(
        {"video_noise_mask": 1.0, "audio_noise_mask": 1.0},
        **common,
    )
    assert all(value is None for value in result.values())


@pytest.mark.parametrize("source_audio", [None, "override.wav"])
def test_prepare_latent_edit_inputs_routes_video_and_audio_sources(
    tmp_path,
    source_audio,
):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    video_clean = torch.full((12, 96), 3.0)
    audio_clean = torch.full((6, 32), 4.0)
    calls: dict[str, Any] = {}

    def prepare_video(value, **kwargs):
        calls["prepare"] = (value, kwargs)
        return {
            "original_path": "source.mov",
            "prepared_path": str(tmp_path / "prepared.mp4"),
        }

    def encode_video(prepared, **kwargs):
        calls["video"] = (prepared, kwargs)
        return video_clean

    def encode_audio(value, **kwargs):
        calls["audio"] = (value, kwargs)
        return (
            audio_clean,
            kwargs["mask_rows"].clone(),
            kwargs["restore_mask_rows"].clone(),
        )

    pipeline._prepare_edit_video = prepare_video
    pipeline._encode_edit_video_source = encode_video
    pipeline._encode_edit_audio_source = encode_audio
    result = pipeline._prepare_latent_edit_inputs(
        {
            "source_video": "source.mov",
            "source_audio": source_audio,
            "video_noise_mask": 0.0,
            "audio_noise_mask": [0.0, 0.5, 1.0],
        },
        width=96,
        height=64,
        num_frames=22,
        latent_t=2,
        audio_t=3,
        duration_seconds=22 / 24,
        workdir=str(tmp_path),
    )

    assert calls["prepare"] == (
        "source.mov",
        {
            "target_width": 96,
            "target_height": 64,
            "target_frame_count": 22,
            "workdir": str(tmp_path / "edit_video"),
        },
    )
    audio_value, audio_kwargs = calls["audio"]
    assert audio_value == ("source.mov" if source_audio is None else source_audio)
    assert audio_kwargs["source_is_video"] is (source_audio is None)
    assert audio_kwargs["duration_seconds"] == pytest.approx(22 / 24)
    torch.testing.assert_close(result["video_edit_clean_rows"], video_clean)
    torch.testing.assert_close(result["video_edit_mask_rows"], torch.zeros(12))
    torch.testing.assert_close(result["video_edit_restore_mask_rows"], torch.zeros(12))
    torch.testing.assert_close(result["audio_edit_clean_rows"], audio_clean)
    torch.testing.assert_close(
        result["audio_edit_mask_rows"],
        torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0]),
    )
    torch.testing.assert_close(
        result["audio_edit_restore_mask_rows"],
        torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0]),
    )


def _build_inputs(*, video_mask: torch.Tensor, audio_mask: torch.Tensor):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    latent_t, latent_h, latent_w, audio_t = 2, 4, 6, 3
    video_clean = torch.linspace(-1.0, 1.0, latent_t * (latent_h // 2) * (latent_w // 2) * 96).reshape(-1, 96)
    audio_clean = torch.linspace(-0.5, 0.5, audio_t * 2 * 32).reshape(-1, 32)
    inputs = pipeline._build_denoise_inputs(
        task="t2va",
        text_embeddings=torch.zeros(2, 4),
        text_tags=torch.ones(2, dtype=torch.long),
        seed=17,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        num_frames=22,
        num_steps=3,
        video_shift=12.0,
        audio_shift=3.0,
        base_schedule=None,
        visual_condition=None,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
        video_edit_clean_rows=video_clean,
        video_edit_mask_rows=video_mask,
        video_edit_restore_mask_rows=video_mask,
        audio_edit_clean_rows=audio_clean,
        audio_edit_mask_rows=audio_mask,
        audio_edit_restore_mask_rows=audio_mask,
    )
    return pipeline, inputs, video_clean, audio_clean


def test_build_denoise_inputs_uses_h3_source_anchors_and_all_generate_fast_path():
    pipeline, inputs, video_clean, audio_clean = _build_inputs(
        video_mask=torch.zeros(12),
        audio_mask=torch.zeros(6),
    )
    video_edit = inputs["video_edit"]
    audio_edit = inputs["audio_edit"]
    assert video_edit is not None and audio_edit is not None
    initial_video, _ = pipeline._initial_noise(
        seed=17,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
    )
    torch.testing.assert_close(video_edit.clean_rows, video_clean)
    torch.testing.assert_close(video_edit.anchor_rows, 0.999 * video_clean + 0.001 * initial_video)
    torch.testing.assert_close(audio_edit.clean_rows, audio_clean)
    torch.testing.assert_close(audio_edit.anchor_rows, audio_clean)

    _, legacy, _, _ = _build_inputs(
        video_mask=torch.ones(12),
        audio_mask=torch.ones(6),
    )
    assert legacy["video_edit"] is None
    assert legacy["audio_edit"] is None
