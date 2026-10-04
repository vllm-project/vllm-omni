# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise exact output geometry and real CPU MP4 encoding without weights."""

import io
from contextlib import nullcontext

import av
import numpy as np
import pytest
import torch

from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import (
    resolve_minimax_h3_exact_output_size,
    resolve_minimax_h3_shape,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("height,width,canvas", [(1080, 1920, (1088, 1920)), (1920, 1080, (1920, 1088))])
def test_exact_canvas_rounds_up_and_default_retains_round_down(height, width, canvas):
    sampling = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        extra_args={"exact_output_size": True, "aspect_ratio": "16:9" if width > height else "9:16"},
    )
    assert resolve_minimax_h3_shape("t2va", sampling, None)[:2] == canvas
    assert resolve_minimax_h3_exact_output_size(sampling) == (height, width)
    sampling.extra_args.pop("exact_output_size")
    assert resolve_minimax_h3_shape("t2va", sampling, None)[:2] == (height // 32 * 32, width // 32 * 32)


@pytest.mark.parametrize(
    "height,width,extra",
    [
        (None, 1920, {}),
        (1081, 1920, {}),
        (True, 1920, {}),
        (32, 1920, {}),
        (1080, 1920, {"latent_refine": {}}),
        (1080, 1920, {"latent_upscale": True}),
        (1080, 1920, {"long_video_mode": "continuation"}),
        (1080, 1920, {"exact_output_size": "true"}),
    ],
)
def test_exact_canvas_rejects_unsupported_requests(height, width, extra):
    sampling = OmniDiffusionSamplingParams(height=height, width=width, extra_args={"exact_output_size": True, **extra})
    with pytest.raises(OmniClientError):
        resolve_minimax_h3_exact_output_size(sampling)


@pytest.mark.parametrize("height,width", [(1080, 1920), (1920, 1080)])
def test_tensor_and_chunked_mp4_use_the_same_center_crop(height, width):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        MiniMaxH3Pipeline,
        _minimax_h3_output_canvas,
        _minimax_h3_output_crop_kwargs,
    )
    from vllm_omni.diffusion.utils.chunked_video import ChunkedVideoMP4Session

    canvas_h, canvas_w = (height + 31) // 32 * 32, (width + 31) // 32 * 32
    shape = {"height": canvas_h, "width": canvas_w, "output_height": height, "output_width": width}
    offset = _minimax_h3_output_crop_kwargs(shape)["crop_offset"]
    assert _minimax_h3_output_canvas(shape, None) == (height, width)
    assert offset == ((canvas_h - height) // 2, (canvas_w - width) // 2)
    # Spatial ramp detects a top-left crop even when dimensions are correct.
    video = torch.linspace(0, 1, canvas_h * canvas_w).reshape(1, 1, 1, canvas_h, canvas_w)
    video = video.expand(1, 3, 4, -1, -1).contiguous()
    audio = torch.zeros(1, 2, 5334)

    class VideoVAE:
        decoder_component = None

        def decode_latent(self, latent):
            return latent

    class AudioVAE:
        def decode_latent(self, latent):
            return latent

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.video_vae = VideoVAE()
    pipeline.audio_vae = AudioVAE()
    pipeline._release_stage_cache = lambda: None
    pipeline._component_on_device = lambda component: nullcontext()
    pipeline._offload_model_cpu_stage_output = lambda result: result
    cropped, returned_audio = pipeline.decode(video, audio, height=height, width=width, crop_offset=offset)
    top, left = offset
    torch.testing.assert_close(cropped, video[..., top : top + height, left : left + width])
    assert returned_audio is audio

    options = {"preset": "ultrafast", "crf": "0"}
    kwargs = dict(
        value_range=(0.0, 1.0),
        fps=24,
        transfer_slots=0,
        audio_waveforms=[audio[0].numpy()],
        audio_sample_rate=32000,
        video_codec_options=options,
    )
    session = ChunkedVideoMP4Session(**kwargs, crop=(height, width), crop_offset=offset)
    reference = ChunkedVideoMP4Session(**kwargs)
    for chunk in video.split(2, dim=2):
        session.push(chunk)
    reference.push(cropped)
    actual_mp4 = session.finish()[0]
    reference_mp4 = reference.finish()[0]
    with av.open(io.BytesIO(actual_mp4)) as container:
        assert (container.streams.video[0].height, container.streams.video[0].width) == (height, width)
        assert container.streams.audio[0].codec_context.layout.name == "stereo"
        actual = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    with av.open(io.BytesIO(reference_mp4)) as container:
        expected = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    assert actual.shape == (4, height, width, 3)
    np.testing.assert_array_equal(actual, expected)


def test_crop_offset_rejects_an_out_of_bounds_rectangle():
    from vllm_omni.diffusion.utils.chunked_video import ChunkedVideoMP4Session

    session = ChunkedVideoMP4Session(value_range=(0.0, 1.0), fps=24, crop=(8, 16), crop_offset=(1, 0))
    with pytest.raises(ValueError, match="exceeds"):
        session.push(torch.zeros(1, 3, 1, 8, 16))
    session.abort()


@pytest.mark.parametrize("wrong_canvas,edited", [(True, False), (False, True)])
def test_external_encoder_rejects_wrong_canvas_or_edit_rows(wrong_canvas, edited):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderConditioning

    conditioning = MiniMaxH3EncoderConditioning(
        hidden_states=torch.zeros(1, 5120),
        token_tags=torch.zeros(1, dtype=torch.long),
        task="t2va",
        height=1056 if wrong_canvas else 1088,
        width=1920,
        num_frames=97,
        latent_t=13,
        audio_t=100,
        video_edit_clean_rows=torch.zeros(1, 96) if edited else None,
    )
    pipeline = object.__new__(MiniMaxH3Pipeline)
    sampling = OmniDiffusionSamplingParams(height=1080, width=1920, extra_args={"exact_output_size": True})
    with pytest.raises(OmniClientError, match="encoder canvas" if wrong_canvas else "editing"):
        pipeline._prepare_encoder_conditioning_inputs(conditioning, sampling)


@pytest.mark.parametrize("feature", ["latent_upscale", "latent_refine"])
def test_exact_canvas_rejects_server_default_resize(feature):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderConditioning

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.od_config = OmniDiffusionConfig(additional_config={feature: True})
    conditioning = MiniMaxH3EncoderConditioning(
        hidden_states=torch.zeros(1, 5120),
        token_tags=torch.zeros(1, dtype=torch.long),
        task="t2va",
        height=1088,
        width=1920,
        num_frames=97,
        latent_t=13,
        audio_t=100,
    )
    sampling = OmniDiffusionSamplingParams(height=1080, width=1920, extra_args={"exact_output_size": True})
    with pytest.raises(OmniClientError, match=feature):
        pipeline._prepare_encoder_conditioning_inputs(conditioning, sampling)


def test_exact_canvas_allows_explicitly_disabled_resize():
    sampling = OmniDiffusionSamplingParams(
        height=1080,
        width=1920,
        extra_args={"exact_output_size": True, "latent_upscale": False, "latent_refine": False},
    )
    assert resolve_minimax_h3_exact_output_size(sampling) == (1080, 1920)
