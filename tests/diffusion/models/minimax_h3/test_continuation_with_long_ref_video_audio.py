# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing
from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def prepared(monkeypatch):
    frames = np.zeros((552, 32, 32, 3), dtype=np.uint8)
    monkeypatch.setattr(
        processing,
        "prepare_reference_videos",
        lambda *a, **k: [
            {
                "prepared_path": "prepared.mp4",
                "original_path": "source.mp4",
                "input_has_audio": True,
                "duration_seconds": 23.0,
            }
        ],
    )
    monkeypatch.setattr(processing, "load_video_frames", lambda *a: frames)
    monkeypatch.setattr(
        processing,
        "sample_reference_video_frames",
        lambda *a, **k: {
            "frames": frames[:2],
            "block_timestamps": [0.0, 1.0],
        },
    )
    monkeypatch.setattr(processing, "load_video_audio", lambda *a, **k: (torch.zeros(1, 2300), 100))

    def prepare(extra):
        return processing.prepare_encoder_inputs(
            {"prompt": "test", "multi_modal_data": {"video": ["source.mp4"]}},
            OmniDiffusionSamplingParams(width=64, height=64, extra_args={"task": "ref2va", "duration": 23, **extra}),
        ).media

    return prepare


def test_long_soundtrack_survives_transport_and_encoding(prepared):
    media = prepared(
        {"long_video": True, "long_video_mode": "continuation", "continuation_slice_reference_video": True}
    )
    media = MiniMaxH3EncoderMediaInput.from_mm_tensors(media.to_mm_tensors(), media.to_metadata())
    audio_vae = Mock()
    audio_vae.encode_waveform.return_value = (torch.ones(1840, 32), 920)
    video_vae = Mock()
    video_vae.encode_video.return_value = (torch.ones(1, 96), (1, 2, 2))
    result = processing.encode_media(media, video_vae=video_vae, audio_vae=audio_vae, emit_conditioning=True)
    assert result.ref_blocks[0]["ref_audio_t"] == 920
    assert result.audio_condition_lengths == (920,)


def test_long_full_keeps_standalone_limit_before_codecs(prepared):
    media = prepared({"long_video": True, "long_video_mode": "full"})
    media = replace(media, audios=((torch.zeros(1, 2300), 100),))
    with pytest.raises(OmniClientError, match=r"\[2, 15\]"):
        processing.encode_media(media, video_vae=None, audio_vae=None, emit_conditioning=False)


def test_short_terminal_soundtrack_survives_transport_and_encoding():
    from vllm_omni.diffusion.models.minimax_h3.continuation import plan_continuation_windows

    frames = torch.zeros(360, 32, 32, 3, dtype=torch.uint8)
    waveform = torch.zeros(1, 36000)
    media = MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=32,
        width=32,
        num_frames=362,
        latent_t=107,
        audio_t=603,
        videos=(frames,),
        video_audios=((waveform, 2400),),
    )
    prepared = processing.PreparedEncoderInputs("test", media, [], [], [], [("video", 1)])
    window = plan_continuation_windows(362, 345, 22)[-1]
    sliced = processing.slice_continuation_reference(prepared, window.start, window.end).media
    sliced = MiniMaxH3EncoderMediaInput.from_mm_tensors(sliced.to_mm_tensors(), sliced.to_metadata())
    assert sliced.video_audios[0][0].shape[-1] == 3700
    audio_vae = Mock()
    audio_vae.encode_waveform.return_value = (torch.ones(124, 32), 62)
    video_vae = Mock()
    video_vae.encode_video.return_value = (torch.ones(1, 96), (1, 2, 2))
    result = processing.encode_media(sliced, video_vae=video_vae, audio_vae=audio_vae, emit_conditioning=True)
    assert result.audio_condition_lengths == (62,)
    with pytest.raises(OmniClientError, match=r"\[2, 15\]"):
        processing.encode_media(
            replace(sliced, audios=((torch.zeros(1, 2400), 2400),)),
            video_vae=video_vae,
            audio_vae=audio_vae,
            emit_conditioning=True,
        )
    with pytest.raises(OmniClientError, match="positive"):
        processing.encode_media(
            replace(sliced, video_audios=((torch.zeros(1, 0), 2400),)),
            video_vae=video_vae,
            audio_vae=audio_vae,
            emit_conditioning=True,
        )


@pytest.mark.parametrize("allow_long_video", [False, True])
def test_long_reference_requires_explicit_opt_in(monkeypatch, tmp_path, allow_long_video):
    from vllm_omni.model_executor.models.minimax_h3 import reference_video

    monkeypatch.setattr(
        reference_video,
        "_probe_video",
        lambda _: {
            "width": 1280,
            "height": 720,
            "fps": 24,
            "duration": 24,
            "file_size": 1000,
            "format_names": ("mp4",),
            "video_codec": "h264",
            "audio_codecs": (),
        },
    )
    monkeypatch.setattr(reference_video, "_transcode_reference_video", lambda *a, **k: "prepared.mp4")
    if allow_long_video:
        result = reference_video.prepare_reference_videos(
            ["source.mp4"],
            target_frame_count=583,
            workdir=str(tmp_path),
            allow_long_video=True,
        )
        assert result[0]["duration_seconds"] == 24
    else:
        with pytest.raises(OmniClientError, match="15"):
            reference_video.prepare_reference_videos(["source.mp4"], target_frame_count=583, workdir=str(tmp_path))
