# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.models.helios.conditioning_cache import HeliosConditioningCache
from vllm_omni.diffusion.models.helios.helios_transformer import HeliosTransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def conditioning_model():
    model = HeliosTransformer3DModel.__new__(HeliosTransformer3DModel)
    torch.nn.Module.__init__(model)
    model.condition_embedder = torch.nn.Module()
    model.condition_embedder.text_embedder = torch.nn.Linear(4, 4)
    block = torch.nn.Module()
    block.attn2 = torch.nn.Module()
    block.attn2.project_kv = Mock(side_effect=lambda hidden: (hidden.unsqueeze(2), hidden.unsqueeze(2)))
    model.blocks = torch.nn.ModuleList([block])
    model._conditioning_owner = object()
    model._conditioning_revision = 0
    model._conditioning_dependencies = ()
    model._conditioning_versions = ()
    model.eval()
    return model


@pytest.mark.parametrize("enabled", [False, True])
def test_cross_attention_precompute_respects_offload_setting(conditioning_model, enabled):
    model = conditioning_model
    model.cache_cross_attention = enabled
    hidden = torch.randn(1, 2, 4)

    with torch.no_grad(), HeliosConditioningCache() as cache:
        projected, first = model._prepare_conditioning(hidden, cache, "positive")
        _, second = model._prepare_conditioning(hidden, cache, "positive")

    project_kv = model.blocks[0].attn2.project_kv
    if enabled:
        assert first is not None and second is not None
        assert first[0][0] is second[0][0]
        assert first[0][1] is second[0][1]
        project_kv.assert_called_once_with(projected)
    else:
        assert first is second is None
        project_kv.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_projected_prompt_cache_respects_offload_setting(conditioning_model, enabled):
    model = conditioning_model
    model.cache_cross_attention = enabled
    hidden = torch.randn(1, 2, 4)
    project = Mock(return_value=None)
    handle = model.condition_embedder.text_embedder.register_forward_hook(project)
    try:
        with torch.no_grad(), HeliosConditioningCache() as cache:
            model._prepare_conditioning(hidden, cache, "positive")
            model._prepare_conditioning(hidden, cache, "positive")
    finally:
        handle.remove()

    assert project.call_count == (1 if enabled else 2)
