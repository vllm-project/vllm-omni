# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_pipeline():
    pipeline = ZImagePipeline.__new__(ZImagePipeline)
    nn.Module.__init__(pipeline)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=True,
    )
    pipeline.text_encoder = Qwen3ForCausalLM(config)
    pipeline.transformer = nn.Linear(4, 4)
    pipeline.vae = nn.Linear(4, 4)
    return pipeline


@pytest.mark.parametrize("bare_encoder", [False, True])
def test_load_original_and_bare_encoder_weights(bare_encoder):
    source = _make_pipeline()
    target = _make_pipeline()
    encoder = source.text_encoder.model if bare_encoder else source.text_encoder
    weights = [(f"text_encoder.{name}", tensor) for name, tensor in encoder.state_dict().items()]
    for component in ("transformer", "vae"):
        weights.extend(
            (f"{component}.{name}", tensor) for name, tensor in getattr(source, component).state_dict().items()
        )

    loaded = target.load_weights(iter(weights))

    for name, expected in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], expected, rtol=0, atol=0)
    assert "text_encoder.model.embed_tokens.weight" in loaded
    assert "text_encoder.model.layers.0.self_attn.q_proj.weight" in loaded
    assert "text_encoder.model.norm.weight" in loaded
    assert "transformer.weight" in loaded
    assert "vae.weight" in loaded


def test_load_weights_without_text_encoder():
    pipeline = _make_pipeline()
    pipeline.text_encoder = None
    weight = torch.ones_like(pipeline.transformer.weight)

    loaded = pipeline.load_weights(iter([("transformer.weight", weight)]))

    assert "transformer.weight" in loaded
    torch.testing.assert_close(pipeline.transformer.weight, weight)


def test_unknown_encoder_weights_still_rejected():
    pipeline = _make_pipeline()
    with pytest.raises(ValueError, match="text_encoder.unexpected"):
        pipeline.load_weights(iter([("text_encoder.unexpected.weight", torch.ones(1))]))
