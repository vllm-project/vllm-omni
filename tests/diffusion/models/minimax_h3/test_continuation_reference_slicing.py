# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.continuation import (
    diffuse_continuation,
    plan_continuation_windows,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import (
    PreparedEncoderInputs,
    slice_continuation_reference,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_reference_windows_slice_semantics_video_and_soundtrack_together():
    frames = torch.arange(576).remainder(256).to(torch.uint8)[:, None, None, None].expand(-1, 2, 2, 3)
    audio = torch.arange(4800)[None]
    image = torch.zeros(2, 2, 3, dtype=torch.uint8)
    media = MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=32,
        width=32,
        num_frames=583,
        latent_t=172,
        audio_t=972,
        images=(image,),
        videos=(frames,),
        video_audios=((audio, 200),),
    )
    prepared = PreparedEncoderInputs("swap", media, [], [], [], [("image", 1), ("video", 1)])
    windows = plan_continuation_windows(583, 345, 22)
    assert [(w.start, w.end) for w in windows] == [(0, 345), (323, 583)]
    for window in windows:
        sliced = slice_continuation_reference(prepared, window.start, window.end)
        torch.testing.assert_close(sliced.media.videos[0], frames[window.start : window.end])
        sampled = sliced.qwen_videos[0][0]
        np.testing.assert_array_equal(sampled, frames[window.start : window.end : 12].numpy())
        start, end = round(window.start * 200 / 24), round(window.end * 200 / 24)
        torch.testing.assert_close(sliced.media.video_audios[0][0], audio[..., start:end])
        assert sliced.media.images[0] is image
        assert sliced.condition_labels == prepared.condition_labels
        assert sliced.media.num_frames == 583
    with pytest.raises(OmniClientError, match="does not cover"):
        slice_continuation_reference(prepared, 600, 700)
    silent = replace(prepared, media=replace(media, video_audios=(None,)))
    assert slice_continuation_reference(silent, 323, 583).media.video_audios == (None,)


def test_each_window_receives_its_reference_before_tail_guide_is_appended():
    calls: list[dict[str, object]] = []
    events = []
    conditions = [
        dict(
            text_embeddings=torch.zeros(2 + i, 8),
            text_tags=torch.ones(2 + i, dtype=torch.long),
            visual_condition=torch.full((1, 96), 10.0 + i),
            visual_condition_shapes=[(1, 2, 2)],
            audio_condition=torch.full((2, 32), 20.0 + i),
            audio_condition_lengths=[1],
            ref_blocks=[{"kind": "image", "latent_h": 2, "latent_w": 2}],
        )
        for i in range(2)
    ]

    def encode_window(inputs, index):
        events.append(("encode", index))
        condition = conditions[index]
        return SimpleNamespace(
            hidden_states=condition["text_embeddings"],
            token_tags=condition["text_tags"],
            **{name: value for name, value in condition.items() if name not in {"text_embeddings", "text_tags"}},
        )

    def sample(**kw):
        index = len(calls)
        events.append(("sample", index))
        calls.append(kw)
        assert torch.all(kw["visual_condition"][0] == 10 + index)
        assert torch.all(kw["audio_condition"][:2] == 20 + index)
        if index:
            assert torch.all(kw["visual_condition"][1:] == 1)
            assert kw["ref_blocks"][-1]["kind"] == "latent_guide"
        return (torch.full((1, 24, kw["latent_t"], 2, 2), float(index + 1)), torch.zeros(2, 32, kw["audio_t"]))

    video, audio = diffuse_continuation(
        sample,
        dict(
            num_frames=583,
            latent_t=172,
            latent_h=2,
            latent_w=2,
            audio_t=972,
            **conditions[0],
        ),
        window_frames=345,
        overlap_frames=22,
        window_conditioning=(encode_window(None, index) for index in range(1, 2)),
        media_time_origin=3,
    )
    assert events == [("sample", 0), ("encode", 1), ("sample", 1)]
    assert len(calls) == 2
    assert video.shape[2] == 172 and audio.shape[-1] == 972
    assert torch.all(video[:, :, :102] == 1)
    assert calls[1]["temporal_offset"] == pytest.approx(323 * 40 / 24)
    assert calls[0]["media_time_origin"] == calls[1]["media_time_origin"] == 3


def test_pipeline_reencodes_both_condition_paths_for_each_reference_window(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaConditioning

    frames = torch.arange(576).remainder(256).to(torch.uint8)[:, None, None, None].expand(-1, 2, 2, 3)
    media = MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=32,
        width=32,
        num_frames=583,
        latent_t=172,
        audio_t=972,
        videos=(frames,),
        video_audios=(None,),
    )
    prepared = PreparedEncoderInputs("swap", media, [], [], [], [("video", 1)])
    monkeypatch.setattr(module, "prepare_encoder_inputs", lambda *args, **kwargs: prepared)
    pipeline = object.__new__(module.MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.load_text_encoder = pipeline.load_vae_encoder = True
    pipeline.device = torch.device("cpu")
    pipeline.od_config = None
    pipeline._active_turbo_spec = lambda sampling: None
    pipeline._has_active_native_lora = lambda sampling: False
    pipeline._resolve_task = lambda *args, **kwargs: "ref2va"
    semantic_frames, vae_frames = [], []

    from torch._subclasses.fake_tensor import FakeTensorMode

    def encode_text(value, *, processor_inputs=None):
        semantic_frames.append(value.qwen_videos[0][0])
        with FakeTensorMode():
            hidden = torch.zeros(processor_inputs[0].numel(), 5120, dtype=torch.bfloat16, device="cuda")
        tags = torch.zeros(processor_inputs[0].numel(), dtype=torch.long)
        return hidden, tags

    def encode_media(value):
        vae_frames.append(value.videos[0])
        return MiniMaxH3EncoderMediaConditioning(
            task=value.task,
            height=value.height,
            width=value.width,
            num_frames=value.num_frames,
            latent_t=value.latent_t,
            audio_t=value.audio_t,
        )

    prefix_lengths = iter((3, 5))
    pipeline._prepare_prompt_inputs = lambda value: (torch.zeros(next(prefix_lengths), dtype=torch.long), None, {})
    pipeline.encode_prompt = encode_text
    pipeline._encode_local_media = encode_media
    pipeline._prepare_encoder_conditioning_inputs = lambda value, sampling: {
        "num_frames": value.num_frames,
        "task": value.task,
    }
    sampling = OmniDiffusionSamplingParams(
        extra_args={
            "task": "ref2va",
            "long_video": True,
            "continuation_slice_reference_video": True,
            "continuation_window_frames": 345,
        }
    )
    context = pipeline._prepare_request_inputs({"prompt": "swap"}, sampling)
    # Preparing the request must not encode later windows.
    assert len(semantic_frames) == len(vae_frames) == 1
    inputs = context["continuation_reference_inputs"]
    condition = pipeline._encode_continuation_window(inputs, 1)
    assert condition.hidden_states.device.type == "cuda"
    assert condition.token_tags.device.type == "cpu"
    for index, (start, end) in enumerate([(0, 345), (323, 583)]):
        torch.testing.assert_close(vae_frames[index], frames[start:end])
        np.testing.assert_array_equal(semantic_frames[index], frames[start:end:12].numpy())
