# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-stage image inputs must build the reference context.

The reference ``interleave_inference`` resizes an input image for the VAE with
``ImageTransform(1024, 512, 16)`` and, from that image, for the ViT with
``ImageTransform(980, 224, 14)``; both keep the aspect ratio. Understanding
requests (``understanding_output=True``) encode the ViT only; the VAE tokens
belong to image-generation requests (img2img).
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image
from pytest_mock import MockerFixture

from vllm_omni.diffusion.models.bagel.image_transforms import to_tensor
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _ImageContextBuiltError(Exception):
    pass


def _pipeline(mocker: MockerFixture) -> BagelPipeline:
    pipeline = object.__new__(BagelPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.od_config = mocker.Mock(dtype=torch.float32)
    pipeline.tokenizer = mocker.sentinel.tokenizer
    pipeline.new_token_ids = {}
    pipeline.language_model = mocker.Mock(vocab_size=10)
    pipeline.image_processor = mocker.Mock()
    pipeline.vae = mocker.Mock()

    bagel = mocker.MagicMock()
    bagel.max_latent_size = 64
    bagel.latent_downsample = 16
    bagel.vit_patch_size = 14
    bagel.vit_max_num_patch_per_side = 70
    bagel.config.llm_config.num_hidden_layers = 1
    bagel.prepare_vae_images.return_value = ({}, [3], [1])
    bagel.prepare_vit_images.return_value = ({}, [5], [2])
    bagel.prepare_prompts.side_effect = _ImageContextBuiltError
    pipeline.bagel = bagel
    return pipeline


@pytest.mark.parametrize(
    ("modalities", "image_key", "vae_updates"),
    [(["text"], "image", 0), (["img2img"], "img2img", 1)],
    ids=["understanding", "img2img"],
)
def test_understanding_encodes_the_vit_only(
    mocker: MockerFixture,
    modalities: list[str],
    image_key: str,
    vae_updates: int,
) -> None:
    pipeline = _pipeline(mocker)
    prompt = {
        "prompt": "Describe this image in detail.",
        "modalities": modalities,
        "multi_modal_data": {image_key: Image.new("RGB", (256, 256))},
    }
    request = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=OmniDiffusionSamplingParams(),
                request_id="test",
            )
        ]
    )

    with pytest.raises(_ImageContextBuiltError):
        pipeline.forward(request)

    assert pipeline.bagel.prepare_vae_images.call_count == vae_updates
    assert pipeline.bagel.forward_cache_update_vae.call_count == vae_updates
    pipeline.bagel.prepare_vit_images.assert_called_once()
    pipeline.bagel.forward_cache_update_vit.assert_called_once()


def _run_until_image_context(
    pipeline: BagelPipeline, modalities: list[str], image_key: str, image: Image.Image
) -> None:
    prompt = {"prompt": "edit", "modalities": modalities, "multi_modal_data": {image_key: image}}
    request = DiffusionRequestBatch(
        requests=[OmniDiffusionRequest(prompt=prompt, sampling_params=OmniDiffusionSamplingParams(), request_id="test")]
    )
    with pytest.raises(_ImageContextBuiltError):
        pipeline.forward(request)


@pytest.mark.parametrize(
    ("source", "vae", "vit"),
    [((800, 1024), (800, 1024), (770, 980)), ((478, 640), (512, 688), (518, 686))],
    ids=["portrait", "short-edge-below-512"],
)
def test_img2img_uses_the_reference_aspect_preserving_sizes(
    mocker: MockerFixture,
    source: tuple[int, int],
    vae: tuple[int, int],
    vit: tuple[int, int],
) -> None:
    pipeline = _pipeline(mocker)

    _run_until_image_context(pipeline, ["img2img"], "img2img", Image.new("RGB", source, (40, 80, 120)))

    vae_call = pipeline.bagel.prepare_vae_images.call_args.kwargs
    assert vae_call["images"][0].size == vae
    assert vae_call["transforms"] is to_tensor
    vit_call = pipeline.bagel.prepare_vit_images.call_args.kwargs
    assert vit_call["images"][0].size == vae
    assert tuple(vit_call["transforms"](vit_call["images"][0]).shape) == (3, vit[1], vit[0])


def test_understanding_uses_the_reference_vit_size(mocker: MockerFixture) -> None:
    pipeline = _pipeline(mocker)

    _run_until_image_context(pipeline, ["text"], "image", Image.new("RGB", (500, 663)))

    vit_call = pipeline.bagel.prepare_vit_images.call_args.kwargs
    assert tuple(vit_call["transforms"](vit_call["images"][0]).shape) == (3, 672, 518)
