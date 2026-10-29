# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from PIL import Image

from vllm_omni.diffusion.models.flux2.pipeline_flux2 import (
    Flux2ImageProcessor,
    Flux2Pipeline,
)
from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import (
    Flux2ImageProcessor as Flux2KleinImageProcessor,
)
from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import (
    Flux2KleinPipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(params=[Flux2Pipeline, Flux2KleinPipeline], ids=["dev", "klein"])
def pipeline(request):
    # Size validation needs only the VAE scale, not model weights or devices.
    pipe = object.__new__(request.param)
    pipe.vae_scale_factor = 8
    return pipe


@pytest.fixture(params=[Flux2ImageProcessor, Flux2KleinImageProcessor], ids=["dev", "klein"])
def image_processor(request):
    return request.param(vae_scale_factor=16)


@pytest.mark.parametrize(
    ("height", "width", "dimension"),
    [(1, 64, "height"), (64, 15, "width"), (0, 64, "height"), (64, -1, "width")],
)
def test_rejects_output_dimension_that_would_empty_the_grid(pipeline, height, width, dimension):
    with pytest.raises(ValueError, match=rf"{dimension}.*at least 16.*got"):
        pipeline.check_inputs(prompt="a landscape", height=height, width=width)


@pytest.mark.parametrize(("scale", "minimum"), [(1, 2), (8, 16), (16, 32)])
@pytest.mark.parametrize("dimension", ["height", "width"])
def test_minimum_output_dimension_follows_vae_scale(pipeline, scale, minimum, dimension):
    pipeline.vae_scale_factor = scale
    dimensions = {"height": 64, "width": 64}
    dimensions[dimension] = minimum - 1
    with pytest.raises(ValueError, match=rf"{dimension}.*at least {minimum}"):
        pipeline.check_inputs(prompt="a landscape", **dimensions)

    dimensions[dimension] = minimum
    pipeline.check_inputs(prompt="a landscape", **dimensions)


@pytest.mark.parametrize(
    ("height", "width"),
    [(16, 16), (205, 333), (None, None), (None, 16), (16, None)],
)
def test_accepts_nonempty_or_default_output_dimensions(pipeline, height, width):
    pipeline.check_inputs(prompt="a landscape", height=height, width=width)


@pytest.mark.parametrize(
    ("height", "width", "dimension"),
    [(16, 1, "width"), (0, 16, "height"), (16, 0, "width")],
)
def test_forward_rejects_small_output_before_prompt_encoding(pipeline, monkeypatch, height, width, dimension):
    def unexpected_encode_prompt(**kwargs):
        pytest.fail("Invalid output dimensions must be rejected before prompt encoding")

    monkeypatch.setattr(pipeline, "encode_prompt", unexpected_encode_prompt)
    request = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="a landscape",
                sampling_params=OmniDiffusionSamplingParams(width=width, height=height, num_inference_steps=4),
                request_id="invalid-output-size",
            )
        ]
    )

    with pytest.raises(ValueError, match=rf"{dimension}.*at least 16"):
        pipeline.forward(request)


@pytest.mark.parametrize("dimension", ["height", "width"])
def test_dev_forward_does_not_replace_zero_with_fallback_size(dimension):
    pipe = object.__new__(Flux2Pipeline)
    pipe.vae_scale_factor = 8
    dimensions = {"height": 64, "width": 64}
    dimensions[dimension] = 0
    request = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="a landscape",
                sampling_params=OmniDiffusionSamplingParams(**dimensions),
                request_id="zero-output-with-fallback",
            )
        ]
    )

    with pytest.raises(ValueError, match=rf"{dimension}.*at least 16"):
        pipe.forward(request, height=64, width=64)


@pytest.mark.parametrize("dimension", ["height", "width"])
def test_dev_forward_validates_fallback_size_when_request_omits_it(dimension):
    pipe = object.__new__(Flux2Pipeline)
    pipe.vae_scale_factor = 8
    dimensions = {"height": 64, "width": 64}
    dimensions[dimension] = 1
    request = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="a landscape",
                sampling_params=OmniDiffusionSamplingParams(height=None, width=None),
                request_id="omitted-output-with-fallback",
            )
        ]
    )

    with pytest.raises(ValueError, match=rf"{dimension}.*at least 16"):
        pipe.forward(request, **dimensions)


class _StopAfterDimensionCaptureError(Exception):
    pass


@pytest.mark.parametrize(
    ("request_height", "request_width", "fallback_height", "fallback_width", "expected_height", "expected_width"),
    [
        (None, None, 64, 96, 64, 96),
        (None, 333, 64, 96, 64, 333),
        (205, None, 64, 96, 205, 96),
        (205, 333, 64, 96, 205, 333),
        (0, None, 64, 96, 0, 96),
        (None, 0, 64, 96, 64, 0),
    ],
)
def test_dev_forward_applies_fallback_only_for_none(
    monkeypatch, request_height, request_width, fallback_height, fallback_width, expected_height, expected_width
):
    """Check forwarded values, not just whether validation raises."""
    pipe = object.__new__(Flux2Pipeline)
    pipe.vae_scale_factor = 8
    captured = {}

    def capture_check_inputs(**kwargs):
        captured.update(kwargs)
        raise _StopAfterDimensionCaptureError

    monkeypatch.setattr(pipe, "check_inputs", capture_check_inputs)
    request = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="a landscape",
                sampling_params=OmniDiffusionSamplingParams(height=request_height, width=request_width),
                request_id="dimension-resolution",
            )
        ]
    )

    with pytest.raises(_StopAfterDimensionCaptureError):
        pipe.forward(request, height=fallback_height, width=fallback_width)

    assert (captured["height"], captured["width"]) == (expected_height, expected_width)


@pytest.mark.parametrize("size", [(15, 64), (64, 15)])
def test_rejects_condition_image_with_small_side(image_processor, size):
    with pytest.raises(ValueError, match="Image too small"):
        image_processor.check_image_input(Image.new("RGB", size))


@pytest.mark.parametrize("size", [(64, 64), (1600, 900), (8192, 1024)])
def test_accepted_condition_image_keeps_nonempty_aligned_dimensions(image_processor, size):
    image = Image.new("RGB", size)
    image_processor.check_image_input(image)
    image = image_processor._resize_if_exceeds_area(image)
    tensor = image_processor.preprocess(image, resize_mode="crop")

    assert tensor.shape[-2] >= 16
    assert tensor.shape[-1] >= 16
    assert tensor.shape[-2] % 16 == 0
    assert tensor.shape[-1] % 16 == 0
