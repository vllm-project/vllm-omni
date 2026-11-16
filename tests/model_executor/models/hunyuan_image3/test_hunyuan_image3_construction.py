# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Construction contracts with CPU modules in place of accelerator kernels."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import CompilationConfig

from vllm_omni.model_executor.models.hunyuan_image3 import _hunyuan_v1_vendored as upstream
from vllm_omni.model_executor.models.hunyuan_image3 import hunyuan_image3 as model

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def components(monkeypatch):
    registrations = []

    class MLP(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.weight = nn.Parameter(torch.ones(2))

        def forward(self, x):
            return x * self.weight

    class Attention(MLP):
        def forward(self, *, hidden_states, **kwargs):
            return super().forward(hidden_states), None

    class CrossAttention(Attention):
        pass

    class ImageMoE(MLP):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            registrations.append(self)

    class Norm(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))

        def forward(self, x, residual=None):
            if residual is None:
                return x * self.weight
            return (x + residual) * self.weight, x + residual

    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    for name, value in dict(
        HunYuanAttention=Attention, HunYuanCrossAttention=CrossAttention, HunYuanMLP=MLP, RMSNorm=Norm
    ).items():
        monkeypatch.setattr(upstream, name, value)
    monkeypatch.setattr(model, "HunyuanImage3SparseMoeBlock", ImageMoE)
    monkeypatch.setattr(upstream, "HunYuanSparseMoeBlock", MLP)
    return SimpleNamespace(registrations=registrations, moe=ImageMoE, norm=Norm)


def config(**overrides):
    return SimpleNamespace(
        **(
            dict(
                hidden_size=2,
                intermediate_size=4,
                num_attention_heads=1,
                rms_norm_eps=1e-6,
                num_experts=2,
                hidden_act="silu",
                num_hidden_layers=3,
                vocab_size=8,
                tie_word_embeddings=False,
                use_cla=True,
                cla_share_factor=2,
            )
            | overrides
        )
    )


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("layer_id", [0, 1])
def test_decoder_matches_upstream_structure_and_forward(components, moe, layer_id):
    cfg = config(num_experts=2 if moe else 1, intermediate_size=[4, 6])
    quant = object()
    kwargs = dict(
        config=cfg,
        cache_config=object(),
        quant_config=quant,
        prefix=f"model.layers.{layer_id}",
        layer_id=layer_id,
        enable_eplb=False,
    )
    reference = upstream.HunYuanDecoderLayer(**kwargs)
    if moe:
        reference.mlp = components.moe(**reference.mlp.kwargs)
    components.registrations.clear()
    actual = model.HunyuanImage3DecoderLayer(**kwargs)
    assert len(components.registrations) == int(moe)
    assert type(actual.self_attn) is type(reference.self_attn)
    assert actual.self_attn.kwargs == reference.self_attn.kwargs
    assert actual.mlp.kwargs == reference.mlp.kwargs
    actual.load_state_dict(reference.state_dict(), strict=True)
    x = torch.tensor([[1.0, 2.0]])
    expected = reference(torch.tensor([0]), x, None)
    result = actual(torch.tensor([0]), x, None)
    for a, b in zip(result[:2], expected[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("layer_id", [0, 1])
def test_npu_decoder_selects_backend_during_construction(components, monkeypatch, layer_id):
    from vllm.platforms import current_platform

    from vllm_omni.platforms.npu.models import hunyuan_image3 as npu_model

    backend = object()
    monkeypatch.setattr(current_platform, "device_type", "npu")
    monkeypatch.setattr(npu_model, "get_image_prefix_backend", lambda: backend)
    layer = model.HunyuanImage3DecoderLayer(config=config(), layer_id=layer_id)
    assert layer.self_attn.kwargs["attn_backend"] is backend
    assert len(components.registrations) == 1


@pytest.mark.parametrize(
    "first,last,tied", [(True, True, False), (False, True, False), (False, True, True), (False, False, False)]
)
def test_final_moe_exists_before_offload_and_pp_layout(components, monkeypatch, first, last, tied):
    cfg = config(tie_word_embeddings=tied)
    vc = SimpleNamespace(
        compilation_config=CompilationConfig(mode=0),
        model_config=SimpleNamespace(hf_config=cfg),
        cache_config=object(),
        quant_config=object(),
        parallel_config=SimpleNamespace(enable_eplb=False, eplb_config=SimpleNamespace(num_redundant_experts=0)),
    )
    monkeypatch.setattr(upstream, "get_pp_group", lambda: SimpleNamespace(is_first_rank=first, is_last_rank=last))
    monkeypatch.setattr(upstream, "VocabParallelEmbedding", lambda *a, **kw: nn.Embedding(8, 2))

    def make_layers(count, factory, prefix):
        layers = []
        for idx in range(count):
            layer = factory(prefix=f"{prefix}.{idx}")
            # This is where upstream make_layers wraps the final layer for offloading.
            assert isinstance(layer.mlp, components.moe)
            assert len(components.registrations) == idx + 1
            layers.append(layer)
        return 0, count, nn.ModuleList(layers)

    monkeypatch.setattr(upstream, "make_layers", make_layers)
    actual = model.HunyuanModel(vllm_config=vc, prefix="model")
    assert components.registrations == [layer.mlp for layer in actual.layers]
    assert isinstance(actual.embed_tokens, nn.Embedding) == (first or (last and tied))
    assert isinstance(actual.norm, components.norm) == last
    assert actual.quant_config is vc.quant_config
    assert all(layer.mlp.kwargs["quant_config"] is vc.quant_config for layer in actual.layers)
