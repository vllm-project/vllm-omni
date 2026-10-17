# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
    Cosmos3LanguageModel,
    _Cosmos3StagedEmbedding,
)
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.config import OffloadStrategy
from vllm_omni.diffusion.offloader.plan_resolver import resolve_offload_plan
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_super_embedding_resolves_distributed_offload(dtype):
    class Pipeline(nn.Module):
        _dit_modules = ["transformer.language_model", "transformer"]
        _encoder_modules = []
        _vae_modules = []
        _resident_modules = []

    class Transformer(nn.Module):
        _layerwise_offload_blocks_attrs = ["gen_layers"]

    with torch.device("meta"):
        language_model = Cosmos3LanguageModel(
            hidden_size=5120,
            intermediate_size=16,
            num_hidden_layers=0,
            num_attention_heads=40,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
            rms_norm_eps=1e-6,
            rope_theta=5_000_000,
            mrope_section=[24, 20, 20],
        ).to(dtype=dtype)
        language_model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        pipeline = Pipeline()
        pipeline.transformer = Transformer()
        pipeline.transformer.language_model = language_model
        pipeline.transformer.gen_layers = nn.ModuleList([nn.Identity(), nn.Identity()])

    config = OffloadConfig(strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE)
    resolve_offload_plan(pipeline, config)
    assert language_model.embed_tokens._stager is None
    # The same topology with the original embedding reproduces the report.
    with torch.device("meta"):
        language_model.embed_tokens = nn.Embedding(151936, 5120, dtype=dtype)
    with pytest.raises(ValueError, match="language_model.embed_tokens"):
        resolve_offload_plan(pipeline, config)


@pytest.mark.cpu
def test_embedding_preserves_unstaged_forward_and_checkpoint_keys():
    reference = nn.Embedding(16, 8)
    embedding = _Cosmos3StagedEmbedding(16, 8)
    embedding.load_state_dict(reference.state_dict(), strict=True)
    tokens = torch.tensor([[0, 3, 15]])
    torch.testing.assert_close(embedding(tokens), reference(tokens), rtol=0, atol=0)
    assert list(embedding.state_dict()) == ["weight"]
    assert embedding._stager is None


@pytest.mark.cuda
@pytest.mark.parametrize("raise_in_forward", [False, True])
def test_staged_embedding_reuses_checkpoint_and_recovers_after_forward(monkeypatch, raise_in_forward):
    if not current_omni_platform.is_cuda() or current_omni_platform.get_device_count() == 0:
        pytest.skip("CUDA required")
    device = current_omni_platform.get_torch_device(0)
    embedding = _Cosmos3StagedEmbedding(16, 8).to(dtype=torch.bfloat16)
    checkpoint = torch.arange(128, dtype=torch.bfloat16).reshape(16, 8)
    embedding.load_state_dict({"weight": checkpoint}, strict=True)
    tokens = torch.tensor([[0, 3, 15]], device=device)
    expected = checkpoint[tokens.cpu()].to(device)
    embedding.offload_to_cpu()
    stager = embedding._stager
    assert stager is not None
    original_embedding = torch.nn.functional.embedding

    def fail(*args, **kwargs):
        raise ValueError("injected embedding failure")

    if raise_in_forward:
        monkeypatch.setattr(torch.nn.functional, "embedding", fail)
        with torch.no_grad(), pytest.raises(ValueError, match="injected embedding failure"):
            embedding(tokens)
        assert embedding.weight.device.type == "cpu"
        assert not stager.loaded
        monkeypatch.setattr(torch.nn.functional, "embedding", original_embedding)

    for _ in range(2):
        with torch.no_grad():
            actual = embedding(tokens)
        assert actual.device == device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert embedding.weight.device.type == "cpu"
        assert not stager.loaded
        assert embedding._stager is stager
        torch.testing.assert_close(embedding.weight, checkpoint, rtol=0, atol=0)
