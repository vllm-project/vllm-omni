# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for GLM-Image VAE patch parallelism support."""

import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_glm_image_pipeline_uses_distributed_vae_class():
    from vllm_omni.diffusion.models.glm_image import pipeline_glm_image as glm_pipeline_module

    assert glm_pipeline_module.DistributedAutoencoderKL is DistributedAutoencoderKL
    assert issubclass(DistributedAutoencoderKL, DistributedVaeMixin)


def test_glm_image_pipeline_loads_distributed_vae(mocker):
    from vllm_omni.diffusion.models.glm_image.pipeline_glm_image import GlmImagePipeline

    mock_vae = mocker.MagicMock()
    mock_vae.config.block_out_channels = [128, 256, 512, 512]
    mock_vae.eval.return_value = mock_vae
    mock_vae.to.return_value = mock_vae

    mock_scheduler = mocker.MagicMock()
    mock_text_encoder = mocker.MagicMock()
    mock_text_encoder.eval.return_value = mock_text_encoder
    mock_text_encoder.to.return_value = mock_text_encoder
    mock_tokenizer = mocker.MagicMock()
    mock_transformer = mocker.MagicMock()
    mock_transformer.patch_size = 2

    mocker.patch(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.DistributedAutoencoderKL.from_pretrained",
        return_value=mock_vae,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.FlowMatchEulerDiscreteScheduler.from_pretrained",
        return_value=mock_scheduler,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.T5EncoderModel.from_pretrained",
        return_value=mock_text_encoder,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.ByT5Tokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    mocker.patch(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.GlmImageTransformer2DModel",
        return_value=mock_transformer,
    )
    mocker.patch("vllm_omni.diffusion.models.glm_image.pipeline_glm_image.os.path.exists", return_value=True)
    mocker.patch(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.download_weights_from_hf_specific",
        return_value="/tmp/glm-image",
    )

    od_config = mocker.MagicMock()
    od_config.model = "/tmp/glm-image"
    od_config.revision = None
    od_config.parallel_config = mocker.MagicMock()
    od_config.quantization_config = None
    od_config.enable_diffusion_pipeline_profiler = False

    pipeline = GlmImagePipeline(od_config=od_config)

    assert pipeline.vae is mock_vae
    DistributedAutoencoderKL.from_pretrained.assert_called_once_with(
        "/tmp/glm-image",
        subfolder="vae",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
