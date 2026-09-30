# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-stage image understanding must build the reference context.

The reference ``interleave_inference(understanding_output=True)`` encodes the
input image with the ViT only; the VAE tokens belong to image-generation
requests (img2img).
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image
from pytest_mock import MockerFixture

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
    bagel.max_latent_size = 32
    bagel.latent_downsample = 8
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
