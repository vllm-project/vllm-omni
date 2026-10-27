# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 timeline guides: placement, preparation, encoding and context."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3 import timeline_guides as guides

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# 22 output frames on a 96x64 canvas: latent (7, 4, 6), 37 audio positions.
_HEIGHT, _WIDTH, _FRAMES, _LATENT_T, _AUDIO_T = 64, 96, 22, 7, 37


def test_guide_placement_math():
    # Clip lengths round down to the 17k + 5 cadence; shorter clips keep one frame.
    for source, expected in [(1, 1), (4, 1), (5, 5), (21, 5), (22, 22), (38, 22), (39, 39)]:
        assert guides.normalize_visual_frame_count(source) == expected
    # Pixel starts on a 124-frame output; negative indices count back from the end.
    for index, length, expected in [(36, 1, 36), (-1, 1, 123), (-22, 22, 102), (-124, 22, 0), (0, 124, 0)]:
        assert guides.resolve_guide_start(index, 124, length) == expected
    # 207 audio positions; a pixel frame spans 5/3 of them from a fractional origin.
    for start, expected in [(0, 207), (1, 205), (36, 147), (102, 37), (123, 2)]:
        assert guides.guide_audio_limit(124, start) == expected


def _two_tone(width: int, height: int) -> Image.Image:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, : width // 2] = (255, 0, 0)
    array[:, width // 2 :] = (0, 0, 255)
    return Image.fromarray(array)


def _prepare(value):
    return guides.prepare_timeline_guides(value, width=_WIDTH, height=_HEIGHT, num_frames=_FRAMES)


def test_prepare_places_stills_clips_and_audio_in_order():
    clip = np.full((30, 32, 48, 3), 7, dtype=np.uint8)
    waveform = torch.zeros(1, 3200)
    prepared = _prepare(
        [
            {"frame_index": -1, "image": _two_tone(400, 100)},
            {"frame_index": -22, "video": clip},
            {"frame_index": 21, "audio": (waveform, 16000)},
            {"frame_index": 3, "image": _two_tone(96, 64), "audio": {"waveform": waveform, "sample_rate": 16000}},
        ]
    )

    assert [guide.start for guide in prepared] == [21, 0, 21, 3]
    assert [guide.is_clip for guide in prepared] == [False, True, False, False]
    still = prepared[0].frames
    assert still.dtype == torch.uint8 and tuple(still.shape) == (1, _HEIGHT, _WIDTH, 3)
    # Center crop keeps the middle of a 4:1 source: both halves remain visible.
    assert still[0, :, 0, 0].min() == 255 and still[0, :, -1, 2].min() == 255
    # A 30-frame clip is normalized down to the 17k + 5 cadence.
    assert tuple(prepared[1].frames.shape) == (22, _HEIGHT, _WIDTH, 3)
    assert prepared[2].frames is None and prepared[2].audio[1] == 16000
    assert prepared[2].audio_limit == guides.guide_audio_limit(_FRAMES, 21) == 2
    assert prepared[3].audio_limit == guides.guide_audio_limit(_FRAMES, 3)
    assert prepared[1].audio is None and prepared[1].audio_limit == 0
    assert _prepare(None) == ()


def _media(**kwargs):
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput

    return MiniMaxH3EncoderMediaInput(
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=_FRAMES,
        latent_t=_LATENT_T,
        audio_t=_AUDIO_T,
        **kwargs,
    )


def _guide_media(task="t2va", **kwargs):
    prepared = _prepare(
        [
            {"frame_index": 7, "image": _two_tone(96, 64), "audio": (torch.ones(2, 800), 8000)},
            {"frame_index": -5, "video": np.zeros((5, 64, 96, 3), dtype=np.uint8)},
            {"frame_index": 0, "audio": (torch.ones(1, 800), 8000)},
        ]
    )
    return _media(task=task, timeline_guides=prepared, **kwargs)


def test_media_input_round_trips_timeline_guides():
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput

    media = _guide_media(images=(torch.zeros(32, 32, 3, dtype=torch.uint8),), task="ref2va")
    restored = MiniMaxH3EncoderMediaInput.from_mm_tensors(media.to_mm_tensors(), media.to_metadata())

    assert len(restored.images) == 1
    assert len(restored.timeline_guides) == 3
    for original, copy in zip(media.timeline_guides, restored.timeline_guides, strict=True):
        assert (copy.start, copy.is_clip, copy.audio_limit) == (original.start, original.is_clip, original.audio_limit)
        assert (copy.frames is None) == (original.frames is None)
        if original.frames is not None:
            assert torch.equal(copy.frames, original.frames)
        assert (copy.audio is None) == (original.audio is None)
        if original.audio is not None:
            assert torch.equal(copy.audio[0], original.audio[0]) and copy.audio[1] == original.audio[1]


class _VideoVAE:
    def __init__(self):
        self.calls = []

    def encode_image(self, image):
        self.calls.append(("image", image.size))
        rows = (image.height // 32) * (image.width // 32)
        return torch.full((rows, 96), float(len(self.calls)))

    def encode_video(self, frames):
        self.calls.append(("video", frames.shape))
        return torch.full((2 * 6, 96), float(len(self.calls))), (2, 4, 6)


class _AudioVAE:
    def encode_waveform(self, waveform, sample_rate):
        length = 5
        # Channel-major rows: channel 0 is 0..4, channel 1 is 100..104.
        rows = torch.cat([torch.arange(length), 100 + torch.arange(length)]).float()[:, None].repeat(1, 32)
        return rows, length


def test_encode_media_builds_ordered_guide_blocks_and_crops_audio_per_channel():
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media

    video_vae = _VideoVAE()
    conditioning = encode_media(
        _guide_media(images=(torch.zeros(32, 32, 3, dtype=torch.uint8),), task="ref2va"),
        video_vae=video_vae,
        audio_vae=_AudioVAE(),
        emit_conditioning=True,
    )

    # The reference image encodes first, then the guide still and clip.
    assert [call[0] for call in video_vae.calls] == ["image", "image", "video"]
    assert conditioning.visual_condition_shapes == ((1, 2, 2),)
    assert conditioning.guide_blocks == (
        {"frame_index": 7, "kind": "video_audio", "latent_t": 1, "latent_h": 4, "latent_w": 6, "ref_audio_t": 5},
        {"frame_index": 17, "kind": "video", "latent_t": 2, "latent_h": 4, "latent_w": 6, "ref_audio_t": 0},
        {"frame_index": 0, "kind": "audio", "ref_audio_t": 5},
    )
    assert conditioning.guide_visual_condition.shape == (18, 96)
    assert conditioning.guide_visual_condition[:6].unique().tolist() == [2.0]
    assert conditioning.guide_visual_condition[6:].unique().tolist() == [3.0]
    assert conditioning.guide_audio_condition.shape == (20, 32)


def test_encode_media_crops_guide_audio_to_the_remaining_timeline():
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media

    media = _media(task="t2va", timeline_guides=_prepare([{"frame_index": 21, "audio": (torch.ones(1, 800), 8000)}]))
    conditioning = encode_media(media, video_vae=None, audio_vae=_AudioVAE(), emit_conditioning=True)

    assert conditioning.guide_blocks == ({"frame_index": 21, "kind": "audio", "ref_audio_t": 2},)
    assert conditioning.guide_audio_condition[:, 0].tolist() == [0.0, 1.0, 100.0, 101.0]
    assert conditioning.guide_visual_condition is None


def test_guide_frames_join_the_video_vae_collective_on_non_leader_ranks():
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media

    video_vae = _VideoVAE()
    assert encode_media(_guide_media(), video_vae=video_vae, audio_vae=None, emit_conditioning=False) is None
    assert [call[0] for call in video_vae.calls] == ["image", "video"]


def _guide_pipeline():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "combined"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace()
    pipeline._base_schedule_by_partition = {"t2va": None, "fl2va": None, "ref2va": None}
    pipeline.load_text_encoder = False
    pipeline.load_vae_encoder = False
    pipeline._quality_policy = Mock()
    pipeline._quality_policy.resolve.return_value = SimpleNamespace(cache_dit=None)
    pipeline._cache_dit_runtime = SimpleNamespace(prepare=lambda spec: None)
    pipeline._turbo_lora_specs = {}
    pipeline._native_distilled_lora_adapter_ids = set()
    pipeline._lora_sigma_schedules = {}
    return pipeline


def _conditioning(media):
    from vllm_omni.model_executor.models.minimax_h3.conditioning import (
        MiniMaxH3EncoderConditioning,
        MiniMaxH3TextConditioning,
    )
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media

    encoded = encode_media(media, video_vae=_VideoVAE(), audio_vae=_AudioVAE(), emit_conditioning=True)
    return MiniMaxH3EncoderConditioning.from_components(
        MiniMaxH3TextConditioning(torch.ones(3, 5120, dtype=torch.bfloat16), torch.ones(3, dtype=torch.long)),
        encoded,
    )


def _sampling(**extra):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionSamplingParams(quality="lossless", fps=24, num_inference_steps=4, extra_args=extra)


def test_context_packs_guides_ahead_of_references():
    conditioning = _conditioning(_guide_media(images=(torch.zeros(64, 32, 3, dtype=torch.uint8),), task="ref2va"))
    context = _guide_pipeline()._prepare_encoder_conditioning_inputs(conditioning, _sampling())

    assert context["guide_blocks"] == [dict(block) for block in conditioning.guide_blocks]
    assert context["ref_blocks"] == [{"kind": "image", "latent_h": 4, "latent_w": 2}]
    assert context["visual_condition_shapes"] == [(1, 4, 6), (2, 4, 6), (1, 4, 2)]
    assert context["audio_condition_lengths"] == [5, 5]
    torch.testing.assert_close(
        context["visual_condition"],
        torch.cat([conditioning.guide_visual_condition, conditioning.visual_condition]),
    )
    torch.testing.assert_close(context["audio_condition"], conditioning.guide_audio_condition)

    inputs = _guide_pipeline()._build_denoise_inputs(
        **{
            **_guide_pipeline()._denoise_kwargs(context),
            "text_embeddings": context["text_embeddings"].float(),
        }
    )
    branch = inputs["branch"]
    assert int((~branch.update_mask).sum()) == context["visual_condition"].shape[0]
    assert int((~branch.audio_update_mask).sum()) == context["audio_condition"].shape[0]


def test_fl2va_keyframes_become_trailing_image_guides():
    keyframes = (torch.zeros(64, 96, 3, dtype=torch.uint8), torch.zeros(64, 96, 3, dtype=torch.uint8))
    media = _media(
        task="fl2va",
        images=keyframes,
        keyframe_frame_indices=(0, -1),
        timeline_guides=_prepare([{"frame_index": 9, "image": _two_tone(96, 64)}]),
    )
    context = _guide_pipeline()._prepare_encoder_conditioning_inputs(_conditioning(media), _sampling())

    assert [(block["kind"], block["frame_index"]) for block in context["guide_blocks"]] == [
        ("image", 9),
        ("image", 0),
        ("image", 21),
    ]
    assert context["ref_blocks"] is None
    assert context["visual_condition"].shape == (18, 96)


def test_unguided_context_is_unchanged():
    media = _media(task="ref2va", images=(torch.zeros(64, 32, 3, dtype=torch.uint8),))
    context = _guide_pipeline()._prepare_encoder_conditioning_inputs(_conditioning(media), _sampling())

    assert context["guide_blocks"] is None
    assert context["ref_blocks"] == [{"kind": "image", "latent_h": 4, "latent_w": 2}]
    assert context["visual_condition_shapes"] == [(1, 4, 2)]


def test_guides_are_rejected_across_a_separate_encoder_stage():
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_executor.stage_input_processors.minimax_h3 import prepare_encoder_prompt

    with pytest.raises(ValueError, match="separate encoder stage"):
        _conditioning(_guide_media()).to_omni_payload()
    prompt = {"prompt": "x", "multi_modal_data": {"timeline_guides": [{"frame_index": 0, "image": "x.png"}]}}
    with pytest.raises(OmniClientError, match="media encoder in the diffusion stage"):
        prepare_encoder_prompt(prompt, [OmniDiffusionSamplingParams()])


def test_guided_ref2va_references_are_scaled_down_to_the_output_area(monkeypatch):
    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing
    from vllm_omni.model_executor.models.minimax_h3.preprocessing import resolve_minimax_h3_reference_image_shape

    image = Image.new("RGB", (1600, 800))
    assert resolve_minimax_h3_reference_image_shape(image) == (1600, 800)
    assert resolve_minimax_h3_reference_image_shape(image, max_area=1344 * 768) == (1440, 704)
    assert resolve_minimax_h3_reference_image_shape(Image.new("RGB", (512, 256)), max_area=1344 * 768) == (512, 256)

    monkeypatch.setattr(processing, "resolve_minimax_h3_shape", lambda *_: (768, 1344, 124, 37, 207))
    sampling = SimpleNamespace(extra_args={"task": "ref2va"})
    unguided = {"prompt": "x", "multi_modal_data": {"image": [image]}}
    assert processing.prepare_encoder_inputs(unguided, sampling).images[0].size == (1600, 800)
    guide = {"frame_index": 0, "image": Image.new("RGB", (1344, 768))}
    guided = {"prompt": "x", "multi_modal_data": {"image": [image], "timeline_guides": [guide]}}
    prepared = processing.prepare_encoder_inputs(guided, sampling)
    assert prepared.images[0].size == (1440, 704)
    assert prepared.condition_labels == [("image", 1)]
    assert len(prepared.media.timeline_guides) == 1
