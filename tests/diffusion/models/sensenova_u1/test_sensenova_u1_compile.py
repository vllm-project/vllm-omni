# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache, DynamicLayer

from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    SenseNovaU1DecoderLayer,
    SenseNovaU1Model,
    _ensure_preallocated_cache_layers,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _CacheUpdatingAttention(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_layer: DynamicLayer | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if cache_layer is not None:
            cache_layer.update(hidden_states, hidden_states + 1)
        return hidden_states * 2


class _CacheDiTLikeWrapper(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return hidden_states


def _build_tiny_sensenova_model(num_layers: int = 2) -> SenseNovaU1Model:
    layers = []
    for layer_idx in range(num_layers):
        layer = SenseNovaU1DecoderLayer.__new__(SenseNovaU1DecoderLayer)
        nn.Module.__init__(layer)
        layer.self_attn = _CacheUpdatingAttention(layer_idx)
        layer.input_layernorm = nn.Identity()
        layer.post_attention_layernorm = nn.Identity()
        layer.mlp = nn.Identity()
        layer.attention_type = "full_attention"
        layers.append(layer)

    model = SenseNovaU1Model.__new__(SenseNovaU1Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=num_layers)
    model.layers = nn.ModuleList(layers)
    model.norm = nn.Identity()
    return model


def test_sensenova_declares_decoder_layers_for_regional_compile() -> None:
    assert SenseNovaU1Model._repeated_blocks == ["SenseNovaU1DecoderLayer"]


def test_regionally_compiled_decoder_matches_eager_cache_updates() -> None:
    eager_model = _build_tiny_sensenova_model()
    compiled_model = _build_tiny_sensenova_model()
    regionally_compile(compiled_model, backend="eager", fullgraph=True, dynamic=True)
    compiled_model.layers = nn.ModuleList([_CacheDiTLikeWrapper(compiled_model.layers)])

    eager_cache = DynamicCache()
    compiled_cache = DynamicCache()

    for seq_len in (2, 1):
        inputs_embeds = torch.arange(seq_len * 4, dtype=torch.float32).reshape(1, seq_len, 4)
        indexes = torch.zeros(3, seq_len, dtype=torch.long)

        eager_output = eager_model(
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            past_key_values=eager_cache,
            use_cache=True,
        )
        compiled_output = compiled_model(
            inputs_embeds=inputs_embeds,
            indexes=indexes,
            past_key_values=compiled_cache,
            use_cache=True,
        )

        torch.testing.assert_close(compiled_output.last_hidden_state, eager_output.last_hidden_state)
        assert len(compiled_cache.layers) == len(eager_cache.layers) == 2
        for compiled_layer, eager_layer in zip(compiled_cache.layers, eager_cache.layers):
            torch.testing.assert_close(compiled_layer.keys, eager_layer.keys)
            torch.testing.assert_close(compiled_layer.values, eager_layer.values)


def test_preallocated_cache_does_not_grow_during_layer_updates() -> None:
    cache = DynamicCache()
    _ensure_preallocated_cache_layers(cache, num_layers=3)

    assert len(cache.layers) == 3
    assert all(isinstance(layer, DynamicLayer) for layer in cache.layers)
    assert cache.layer_class_to_replicate is None

    key = torch.ones(1, 1, 2, 4)
    value = torch.full_like(key, 2)
    updated_key, updated_value = cache.layers[2].update(key, value)

    assert len(cache.layers) == 3
    torch.testing.assert_close(updated_key, key)
    torch.testing.assert_close(updated_value, value)


def test_preallocation_preserves_existing_cache_layers() -> None:
    cache = DynamicCache()
    _ensure_preallocated_cache_layers(cache, num_layers=2)
    first_layer = cache.layers[0]

    _ensure_preallocated_cache_layers(cache, num_layers=4)

    assert len(cache.layers) == 4
    assert cache.layers[0] is first_layer
