# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint reads must use the same revision as the model weights."""

import pytest
import torch
from transformers import PretrainedConfig

from vllm_omni.diffusion.data import DiffusionKVCacheMode, OmniDiffusionConfig
from vllm_omni.diffusion.models.hunyuan_image3 import pipeline_hunyuan_image3 as pipeline
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import TokenizerWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(params=[("org/model", "commit-a"), ("org/model", None), ("/local/checkpoint", None)])
def checkpoint(request, mocker):
    model, revision = request.param
    config = mocker.Mock(spec=OmniDiffusionConfig)
    config.model = model
    config.revision = revision
    config.weight_sharing = None
    config.diffusion_kv_mode = DiffusionKVCacheMode.PAGED_SCHEDULER
    # Pipeline startup resolves the default attention schedule (subscripts its
    # entries); an unset optional config must be None, not an auto-created Mock.
    config.diffusion_attention_schedule = None
    hf_config = PretrainedConfig(vae_downsample_factor=[8, 8], patch_size=2, vae={})
    get_config = mocker.patch.object(pipeline, "get_config", return_value=hf_config)
    generation = mocker.patch.object(pipeline.GenerationConfig, "from_pretrained")
    tokenizer = mocker.patch(
        "vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer.AutoTokenizer.from_pretrained"
    )
    return config, get_config, generation, tokenizer


def assert_checkpoint_reads(config, get_config, generation, tokenizer):
    get_config.assert_called_once_with(config.model, trust_remote_code=True, revision=config.revision)
    generation.assert_called_once_with(config.model, revision=config.revision)
    tokenizer.assert_called_once_with(config.model, revision=config.revision)


def test_preprocessor_uses_checkpoint_revision(checkpoint, mocker):
    config, get_config, generation, tokenizer = checkpoint
    mocker.patch.object(pipeline, "HunyuanImage3ImageProcessor")
    assert callable(pipeline.get_hunyuan_image_3_pre_process_func(config))
    assert_checkpoint_reads(config, get_config, generation, tokenizer)


def test_pipeline_uses_checkpoint_revision(checkpoint, mocker):
    config, get_config, generation, tokenizer = checkpoint

    # Stop after all checkpoint reads; no GPU model or VAE is constructed.
    class CheckpointReadsCompleteError(Exception):
        pass

    mocker.patch.object(
        pipeline.HunyuanImage3PreTrainedModel,
        "__init__",
        lambda self, config: torch.nn.Module.__init__(self),
    )
    mocker.patch.object(pipeline, "HunyuanImage3Model", return_value=torch.nn.Identity())
    mocker.patch(
        "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuan."
        "DistributedAutoencoderKLHunyuan.from_config",
        return_value=torch.nn.Identity(),
    )
    mocker.patch.object(pipeline, "HunyuanImage3ImageProcessor", side_effect=CheckpointReadsCompleteError)
    model = pipeline.HunyuanImage3Pipeline.__new__(pipeline.HunyuanImage3Pipeline)
    with pytest.raises(CheckpointReadsCompleteError):
        model.__init__(config)
    assert model.weights_sources[0].revision == config.revision
    assert_checkpoint_reads(config, get_config, generation, tokenizer)


def test_tokenizer_object_does_not_reload(mocker):
    load = mocker.patch(
        "vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer.AutoTokenizer.from_pretrained"
    )
    tokenizer = mocker.Mock()
    assert TokenizerWrapper(tokenizer).tokenizer is tokenizer
    load.assert_not_called()
