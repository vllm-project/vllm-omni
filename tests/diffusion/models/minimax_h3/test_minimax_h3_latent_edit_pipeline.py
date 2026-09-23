# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pipeline-level MiniMax H3 latent initialization tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_prepare_edit_video_clones_the_last_frame_with_lossless_rgb(monkeypatch, tmp_path):
    from vllm_omni.model_executor.models.minimax_h3 import reference_video

    source = tmp_path / "source.webm"
    commands: list[list[str]] = []
    monkeypatch.setattr(
        reference_video,
        "_probe_video",
        lambda _path: {"frame_count": 12, "audio_codecs": ("opus",)},
    )
    monkeypatch.setattr(
        reference_video.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command),
    )

    prepared = reference_video.prepare_edit_video(
        source,
        target_width=448,
        target_height=256,
        target_frame_count=73,
        workdir=str(tmp_path / "prepared"),
    )

    assert prepared["input_has_audio"] is True
    assert prepared["frame_count"] == 73
    command = commands[0]
    assert command[command.index("-vf") + 1].endswith("tpad=stop_mode=clone:stop=-1")
    assert command[command.index("-c:v") + 1] == "libx264rgb"
    assert command[command.index("-pix_fmt") + 1] == "rgb24"


@pytest.mark.parametrize("source_audio", [None, "override.wav"])
def test_encoder_preparation_routes_edit_sources_without_qwen_inputs(
    monkeypatch,
    tmp_path,
    source_audio,
):
    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing

    latent_t, audio_t = 7, 37
    video_clean = torch.full((42, 96), 3.0)
    audio_source = torch.zeros(4, 32)
    audio_source[:, 0] = torch.tensor([1.0, 2.0, 11.0, 12.0])
    waveform = torch.zeros(3_200)
    frames = torch.zeros(22, 64, 96, 3, dtype=torch.uint8).numpy()
    calls: dict[str, Any] = {}

    def prepare_video(value, **kwargs):
        calls["prepare"] = (value, kwargs)
        return {
            "original_path": "source.mov",
            "prepared_path": str(tmp_path / "prepared.mp4"),
            "input_has_audio": True,
        }

    def load_video_audio(value, **kwargs):
        calls["video_audio"] = (value, kwargs)
        return waveform, 32_000

    def load_audio_file(value):
        calls["audio_file"] = value
        return waveform, 32_000

    class VideoVAE:
        def encode_video(self, value):
            calls["video_encode_shape"] = value.shape
            return video_clean.clone(), (latent_t, 4, 6)

    class AudioVAE:
        def encode_waveform(self, value, sample_rate):
            calls["audio_encode"] = (value.shape, sample_rate)
            return audio_source.clone(), 2

    monkeypatch.setattr(processing, "resolve_minimax_h3_shape", lambda *_args: (64, 96, 22, latent_t, audio_t))
    monkeypatch.setattr(processing, "prepare_edit_video", prepare_video)
    monkeypatch.setattr(processing, "load_video_frames", lambda _path: frames)
    monkeypatch.setattr(processing, "load_video_audio", load_video_audio)
    monkeypatch.setattr(processing, "load_audio_file", load_audio_file)

    prepared = processing.prepare_encoder_inputs(
        {
            "prompt": "edit the clip",
            "multi_modal_data": {
                "source_video": "source.mov",
                "source_audio": source_audio,
                "video_noise_mask": torch.zeros(1, 1, latent_t, 4, 6),
                "audio_noise_mask": torch.tensor([0.0, 0.5, *([0.0] * (audio_t - 2))]).reshape(1, 1, -1),
            },
        },
        SimpleNamespace(extra_args={"task": "t2va"}),
    )
    media = type(prepared.media).from_mm_tensors(
        prepared.media.to_mm_tensors(),
        prepared.media.to_metadata(),
    )
    conditioning = processing.encode_media(
        media,
        video_vae=VideoVAE(),
        audio_vae=AudioVAE(),
        emit_conditioning=True,
    )

    assert prepared.images == []
    assert prepared.qwen_videos == []
    assert prepared.condition_labels == []
    assert prepared.media.images == ()
    assert prepared.media.videos == ()
    assert calls["prepare"][0] == "source.mov"
    assert calls["prepare"][1]["target_width"] == 96
    assert calls["prepare"][1]["target_height"] == 64
    assert calls["prepare"][1]["target_frame_count"] == 22
    if source_audio is None:
        assert calls["video_audio"] == (
            "source.mov",
            {"duration_seconds": pytest.approx(22 / 24)},
        )
        assert "audio_file" not in calls
    else:
        assert calls["audio_file"] == "override.wav"
        assert "video_audio" not in calls

    assert conditioning is not None
    torch.testing.assert_close(conditioning.video_edit_clean_rows, video_clean)
    assert conditioning.video_edit_mask.shape == (latent_t, 4, 6)
    torch.testing.assert_close(conditioning.video_edit_mask, torch.zeros(latent_t, 4, 6))
    assert conditioning.audio_edit_source_t == 2
    assert conditioning.audio_edit_mask.shape == (2, audio_t)
    torch.testing.assert_close(
        conditioning.audio_edit_clean_rows[:, 0],
        torch.tensor([1.0, 2.0, *([0.0] * (audio_t - 2)), 11.0, 12.0, *([0.0] * (audio_t - 2))]),
    )
    torch.testing.assert_close(
        conditioning.audio_edit_mask,
        torch.tensor([0.0, 0.5, *([0.0] * (audio_t - 2))]).repeat(2, 1),
    )

    from vllm_omni.model_executor.models.minimax_h3.conditioning import (
        MiniMaxH3EncoderConditioning,
        MiniMaxH3TextConditioning,
    )

    combined = MiniMaxH3EncoderConditioning.from_components(
        MiniMaxH3TextConditioning(
            torch.zeros(1, 5120, dtype=torch.bfloat16),
            torch.ones(1, dtype=torch.int64),
        ),
        conditioning,
    )
    round_trip = MiniMaxH3EncoderConditioning.from_omni_payload(combined.to_omni_payload())
    torch.testing.assert_close(round_trip.video_edit_clean_rows, video_clean)
    torch.testing.assert_close(round_trip.video_edit_mask, torch.zeros(latent_t, 4, 6))
    torch.testing.assert_close(round_trip.audio_edit_clean_rows, conditioning.audio_edit_clean_rows)
    torch.testing.assert_close(
        round_trip.audio_edit_mask,
        torch.tensor([0.0, 0.5, *([0.0] * (audio_t - 2))]).repeat(2, 1),
    )
    assert round_trip.audio_edit_source_t == 2

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    pipeline.od_config = SimpleNamespace(step_execution=False)
    pipeline._fasth3 = None
    pipeline._active_turbo_spec = lambda _sampling: None
    pipeline._has_active_native_lora = lambda _sampling: False
    pipeline._resolve_task = lambda task, **_kwargs: task
    pipeline._resolve_sigma_positions = lambda _task, _sampling: (None, 3)
    pipeline._quality_policy = SimpleNamespace(resolve=lambda **_kwargs: SimpleNamespace(cache_dit=None))
    pipeline._cache_dit_runtime = SimpleNamespace(prepare=lambda _plan: None)
    context = pipeline._prepare_encoder_conditioning_inputs(
        round_trip,
        SimpleNamespace(extra_args={}, quality=None, seed=17, num_outputs_per_prompt=1),
    )
    torch.testing.assert_close(context["video_edit_clean_rows"], video_clean)
    torch.testing.assert_close(context["video_edit_mask_rows"], torch.zeros(42))
    assert context["video_edit_restore_mask_rows"].shape == (42, 96)
    torch.testing.assert_close(
        context["audio_edit_mask_rows"],
        torch.tensor([0.0, 0.5, *([1.0] * (audio_t - 2))]).repeat(2),
    )
    torch.testing.assert_close(context["audio_edit_mask_rows"], context["audio_edit_restore_mask_rows"])


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


def test_build_denoise_inputs_uses_h3_source_anchors():
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
