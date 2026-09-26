# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for Pi-family shared backbone composition."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.pi.common import attention, backbone

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _TestSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 2
        self.num_key_value_groups = 1
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _TestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(torch.tanh(self.up_proj(hidden_states)))


class _TestPrefixLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = _TestSelfAttention()
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = _TestMLP()


class _TestLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_TestPrefixLayer()])
        self.norm = nn.LayerNorm(4)

    def rotary_emb(self, value, position_ids):
        batch_size, _, sequence_length, head_dim = value.shape
        shape = (batch_size, sequence_length, head_dim)
        return torch.zeros(shape, dtype=value.dtype), torch.ones(shape, dtype=value.dtype)


def _test_paligemma():
    torch.manual_seed(7)
    language_model = _TestLanguageModel()
    return SimpleNamespace(model=SimpleNamespace(language_model=language_model))


@pytest.mark.parametrize("num_views", [1, 2, 3])
def test_multimodal_prefix_preserves_camera_order_and_masks_missing_slots(num_views, monkeypatch):
    batch_size, image_tokens, language_tokens, width = 2, 2, 3, 1
    images = [torch.full((batch_size, image_tokens, width), float(index + 1)) for index in range(num_views)]
    image_masks = [torch.tensor([True, index != 1]) for index in range(num_views)]
    language = torch.full((batch_size, language_tokens, width), 9.0)
    language_masks = torch.tensor([[True, True, False], [True, False, False]])
    monkeypatch.setattr(backbone, "embed_image", lambda _paligemma, image: image)
    monkeypatch.setattr(backbone, "embed_language_tokens", lambda _paligemma, tokens: tokens)

    embeddings, padding_masks, attention_markers = backbone.embed_multimodal_prefix(
        images,
        image_masks,
        language,
        language_masks,
        paligemma=object(),
    )

    assert torch.equal(embeddings, torch.cat([*images, language], dim=1))
    expected_image_masks = [mask[:, None].expand(batch_size, image_tokens) for mask in image_masks]
    assert torch.equal(padding_masks, torch.cat([*expected_image_masks, language_masks], dim=1))
    assert attention_markers.dtype == torch.bool
    assert not attention_markers.any()


@pytest.mark.parametrize(
    "images,image_masks,error",
    [
        ([torch.empty(1)], [], "same number of views"),
        ([torch.empty(1), torch.empty(1)], [torch.empty(1), torch.empty(1)], "Expected exactly 3 image views"),
    ],
)
def test_fixed_camera_layout_is_validated_when_requested(images, image_masks, error):
    with pytest.raises(ValueError, match=error):
        backbone.embed_multimodal_prefix(
            images,
            image_masks,
            torch.empty(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            paligemma=object(),
            expected_num_views=3,
        )


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("gemma_2b", backbone.GemmaVariantConfig(2048, 18, 16384, 8, 1, 256)),
        ("gemma_300m", backbone.GemmaVariantConfig(1024, 18, 4096, 8, 1, 256)),
    ],
)
def test_gemma_variant_dimensions_match_openpi(variant, expected):
    assert backbone.get_gemma_config(variant) == expected


def test_unknown_gemma_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown variant"):
        backbone.get_gemma_config("gemma_7b")


def test_build_backbones_preserves_hf_configuration_and_disables_expert_tokens(monkeypatch):
    built = {}

    class FakePaliGemma:
        def __init__(self, config):
            built["paligemma_config"] = config

    class FakeGemma:
        def __init__(self, config):
            built["expert_config"] = config
            self.model = SimpleNamespace(embed_tokens=object())

    monkeypatch.setattr(backbone, "PaliGemmaForConditionalGeneration", FakePaliGemma)
    monkeypatch.setattr(backbone, "GemmaForCausalLM", FakeGemma)

    vlm = backbone.GemmaVariantConfig(32, 2, 64, 4, 1, 8)
    expert = backbone.GemmaVariantConfig(16, 3, 48, 2, 1, 8)
    _, built_expert = backbone.build_backbones(vlm, expert)

    pali_config = built["paligemma_config"]
    assert pali_config.image_token_index == 257152
    assert pali_config.text_config.hidden_size == 32
    assert pali_config.text_config.intermediate_size == 64
    assert pali_config.text_config.num_hidden_layers == 2
    assert pali_config.text_config.num_attention_heads == 4
    assert pali_config.text_config.num_key_value_heads == 1
    assert pali_config.text_config.head_dim == 8
    assert pali_config.vision_config.intermediate_size == 4304
    assert pali_config.vision_config.projection_dim == 2048

    expert_config = built["expert_config"]
    assert expert_config.hidden_size == 16
    assert expert_config.intermediate_size == 48
    assert expert_config.num_hidden_layers == 3
    assert expert_config.num_attention_heads == 2
    assert expert_config.num_key_value_heads == 1
    assert expert_config.head_dim == 8
    assert built_expert.model.embed_tokens is None


def test_embed_image_uses_explicit_vision_tower_then_projector():
    pixel_values = torch.tensor([[[[1.0]]]])
    paligemma = SimpleNamespace(
        model=SimpleNamespace(
            vision_tower=lambda pixels: SimpleNamespace(last_hidden_state=pixels + 2.0),
            multi_modal_projector=lambda features: features * 3.0,
        )
    )

    assert torch.equal(backbone.embed_image(paligemma, pixel_values), torch.tensor([[[[9.0]]]]))


def test_embed_language_tokens_applies_legacy_scale_once():
    embed_tokens = nn.Embedding(4, 4)
    nn.init.ones_(embed_tokens.weight)
    paligemma = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(embed_tokens=embed_tokens)))

    actual = backbone.embed_language_tokens(paligemma, torch.tensor([[0, 1]]))

    assert torch.equal(actual, torch.full((1, 2, 4), 2.0))


def test_embed_language_tokens_does_not_repeat_self_scaling():
    class SelfScalingEmbedding(nn.Module):
        embed_scale = 2.0

        def forward(self, tokens):
            return torch.full((*tokens.shape, 4), 3.0)

    paligemma = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(embed_tokens=SelfScalingEmbedding()))
    )

    actual = backbone.embed_language_tokens(paligemma, torch.tensor([[0, 1]]))

    assert torch.equal(actual, torch.full((1, 2, 4), 3.0))


def test_execute_prefix_layer_matches_reference_and_returns_post_rope_kv():
    paligemma = _test_paligemma()
    layer = paligemma.model.language_model.layers[0]
    hidden_states = torch.randn(2, 3, 4)
    position_ids = torch.arange(3)[None, :].expand(2, -1)
    attention_mask = torch.zeros(2, 1, 3, 3)

    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    hidden_shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2)
    key = layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2)
    value = layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2)
    cos, sin = paligemma.model.language_model.rotary_emb(value, position_ids)
    query, expected_key = backbone.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
    attended = attention.eager_attention(
        query,
        expected_key,
        value,
        attention_mask,
        num_kv_groups=layer.self_attn.num_key_value_groups,
        scaling=layer.self_attn.head_dim**-0.5,
    )
    attended = attended.transpose(1, 2).reshape(2, 3, 4)
    expected_hidden = layer.self_attn.o_proj(attended) + residual
    expected_hidden = layer.mlp(layer.post_attention_layernorm(expected_hidden)) + expected_hidden

    actual_hidden, (actual_key, actual_value) = backbone.execute_prefix_layer(
        0,
        hidden_states,
        attention_mask,
        position_ids,
        paligemma,
    )

    assert torch.equal(actual_hidden, expected_hidden)
    assert torch.equal(actual_key, expected_key)
    assert torch.equal(actual_value, value)


def test_execute_prefix_layer_uses_variant_dtype_alignment_policy():
    paligemma = _test_paligemma()
    layer = paligemma.model.language_model.layers[0]
    aligned_modules = []

    def record_alignment(tensor, module):
        aligned_modules.append(module)
        return tensor

    backbone.execute_prefix_layer(
        0,
        torch.randn(1, 2, 4),
        torch.zeros(1, 1, 2, 2),
        torch.arange(2)[None, :],
        paligemma,
        align_module_input=record_alignment,
    )

    assert aligned_modules == [layer.self_attn.q_proj, layer.self_attn.o_proj, layer.mlp.up_proj]


def test_execute_prefix_runs_all_layers_applies_final_norm_and_orders_kv():
    paligemma = _test_paligemma()
    language_model = paligemma.model.language_model
    language_model.layers.append(_TestPrefixLayer())
    hidden_states = torch.randn(1, 2, 4)
    attention_mask = torch.zeros(1, 1, 2, 2)
    position_ids = torch.arange(2)[None, :]

    expected_hidden = hidden_states
    expected_kv = []
    for layer_idx in range(2):
        expected_hidden, layer_kv = backbone.execute_prefix_layer(
            layer_idx,
            expected_hidden,
            attention_mask,
            position_ids,
            paligemma,
        )
        expected_kv.append(layer_kv)
    expected_hidden = language_model.norm(expected_hidden)

    actual_hidden, actual_kv = backbone.execute_prefix(
        hidden_states,
        attention_mask,
        position_ids,
        paligemma,
    )

    assert torch.equal(actual_hidden, expected_hidden)
    assert len(actual_kv) == 2
    for actual, expected in zip(actual_kv, expected_kv):
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
