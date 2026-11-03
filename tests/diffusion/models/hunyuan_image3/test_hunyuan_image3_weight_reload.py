# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)

from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import HunyuanImage3Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _pipeline():
    pipeline = object.__new__(HunyuanImage3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.hf_config = SimpleNamespace(tie_word_embeddings=False)
    pipeline.model = nn.Module()
    pipeline.model.wte = nn.Embedding(8, 4)
    pipeline.model.ln_f = nn.LayerNorm(4)
    pipeline.transformer = pipeline.model
    pipeline.patch_embed = nn.Linear(4, 4)
    pipeline.final_layer = nn.Module()
    pipeline.final_layer.model = nn.Linear(4, 2)
    pipeline.lm_head = nn.Linear(4, 8, bias=False)
    pipeline.vae = nn.Linear(2, 2)
    pipeline.vision_model = nn.Linear(4, 4)
    pipeline.register_buffer("cache_marker", torch.tensor([3.0]), persistent=False)
    return pipeline


@pytest.mark.parametrize("head", ["patch_embed", "final_layer.model", "lm_head"])
def test_repeated_partial_reload_preserves_untouched_weights_and_storage(head):
    pipeline = _pipeline()
    originals = dict(pipeline.named_parameters())
    initial = {name: param.detach().clone() for name, param in originals.items()}
    storage = {name: param.data_ptr() for name, param in originals.items()}
    marker = pipeline.cache_marker
    targets = {name for name in originals if name.startswith(("model.", head + "."))}
    with torch.device("cpu"):
        record_metadata_for_reloading(pipeline)

    for version in (1, 2):
        updates = {name: torch.full_like(initial[name], float(version)) for name in targets}
        initialize_layerwise_reload(pipeline)
        assert pipeline.patch_embed.weight.is_meta
        for name, value in updates.items():
            assert pipeline.load_weights([(name, value)]) == {name}
        finalize_layerwise_reload(pipeline, SimpleNamespace(dtype=torch.float32))

        for name, param in pipeline.named_parameters():
            torch.testing.assert_close(param, updates.get(name, initial[name]))
            assert param is originals[name]
            assert param.data_ptr() == storage[name]
        assert pipeline.transformer is pipeline.model
        assert pipeline.cache_marker is marker
        torch.testing.assert_close(marker, torch.tensor([3.0]))


def test_tied_output_head_is_skipped_during_reload():
    pipeline = _pipeline()
    pipeline.hf_config.tie_word_embeddings = True
    initial = pipeline.lm_head.weight.detach().clone()
    with torch.device("cpu"):
        record_metadata_for_reloading(pipeline)
    initialize_layerwise_reload(pipeline)
    assert pipeline.load_weights([("lm_head.weight", torch.ones_like(initial))]) == set()
    finalize_layerwise_reload(pipeline, SimpleNamespace(dtype=torch.float32))
    torch.testing.assert_close(pipeline.lm_head.weight, initial)
