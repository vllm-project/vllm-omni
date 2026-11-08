# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for SenseNova-U1-A3B MoE config, layer dispatch and weight loading.

FusedMoE numerics live in ``test_sensenova_u1_moe_cuda.py`` so the ready
``core_model and cpu`` lane and the ``core_model and cuda`` model lane collect
disjoint items.
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    _is_moe,
    _is_sparse_und_layer,
)
from vllm_omni.transformers_utils.configs.sensenova_u1 import (
    SenseNovaU1Config,
    SenseNovaU1LLMConfig,
    SenseNovaU1MoELLMConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_A3B_LLM = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "gen_num_experts": 32,
    "num_experts_per_tok": 8,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "num_hidden_layers": 48,
    "head_dim": 128,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "vocab_size": 151936,
    "max_position_embeddings": 262144,
    "max_position_embeddings_hw": 10000,
    "rope_theta": 10000000,
    "rope_theta_hw": 10000.0,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "norm_topk_prob": True,
}


def _tiny_moe_llm(**overrides) -> SenseNovaU1MoELLMConfig:
    kwargs = dict(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        # A3B sets this explicitly; the Qwen3MoeConfig default is False.
        norm_topk_prob=True,
    )
    kwargs.update(overrides)
    return SenseNovaU1MoELLMConfig(**kwargs)


def _tiny_dense_llm() -> SenseNovaU1LLMConfig:
    return SenseNovaU1LLMConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4)


def test_dense_checkpoint_still_uses_qwen3_config():
    cfg = SenseNovaU1Config(llm_config={"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2})
    assert isinstance(cfg.llm_config, SenseNovaU1LLMConfig)
    assert not _is_moe(cfg.llm_config)


def test_a3b_dict_selects_moe_config_and_gen_defaults():
    cfg = SenseNovaU1Config(llm_config=_A3B_LLM)
    assert isinstance(cfg.llm_config, SenseNovaU1MoELLMConfig)
    llm = cfg.llm_config
    assert llm.num_experts == 128
    assert llm.gen_num_experts == 32
    assert llm.gen_num_experts_per_tok == 8
    assert llm.gen_moe_intermediate_size == 768
    assert llm.moe_intermediate_size == 768
    assert len(llm.layer_types) == 48
    assert llm.layer_types == ["full_attention"] * 48
    assert llm.rope_theta_hw == 10000.0


def test_gen_moe_knobs_fall_back_to_und_path():
    llm = dict(_A3B_LLM)
    llm.pop("gen_num_experts")
    cfg = SenseNovaU1Config(llm_config=llm)
    assert cfg.llm_config.gen_num_experts == 128
    assert cfg.llm_config.gen_num_experts_per_tok == 8
    assert cfg.llm_config.gen_moe_intermediate_size == 768


def test_sparse_und_layer_matches_qwen3_moe_step():
    config = _tiny_moe_llm(num_hidden_layers=3, mlp_only_layers=[0], decoder_sparse_step=2)
    assert not _is_sparse_und_layer(config, 0)  # mlp_only
    assert _is_sparse_und_layer(config, 1)  # (1+1) % 2 == 0
    assert not _is_sparse_und_layer(config, 2)
    assert not _is_sparse_und_layer(_tiny_dense_llm(), 0)


class _Routed(nn.Module):
    def __init__(self):
        super().__init__()
        self.w13_weight = nn.Parameter(torch.zeros(2, 8, 4))
        self.w2_weight = nn.Parameter(torch.zeros(2, 4, 8))


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.routed_experts = _Routed()


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = nn.Linear(4, 16, bias=False)
        self.mlp.experts = _Experts()
        self.mlp_mot_gen = nn.Module()
        self.mlp_mot_gen.experts = _Experts()


class _MoeStub(SenseNovaU1Pipeline):
    """Only the parameter tree matters here, so skip the real __init__."""

    def __init__(self, llm_cfg: SenseNovaU1LLMConfig | SenseNovaU1MoELLMConfig):
        nn.Module.__init__(self)
        self.llm_cfg = llm_cfg
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([_Layer()])


def test_load_weights_routes_expert_gate_proj_to_fused_w13():
    """A3B shards are ``experts.{i}.gate_proj``; they must not hit dense gate_up."""
    pipe = _MoeStub(_tiny_moe_llm(num_experts=2, gen_num_experts=2, num_experts_per_tok=1))
    layer = pipe.language_model.model.layers[0]
    with torch.no_grad():
        layer.mlp.gate_up_proj.weight.zero_()
        layer.mlp.experts.routed_experts.w13_weight.zero_()

    calls: list[tuple] = []

    def _loader(param, loaded, name, shard_id=None, expert_id=None, return_success=True):
        calls.append((name, shard_id, expert_id, tuple(loaded.shape)))
        return True

    layer.mlp.experts.routed_experts.w13_weight.weight_loader = _loader
    layer.mlp_mot_gen.experts.routed_experts.w13_weight.weight_loader = _loader

    def _stacked_loader(param, loaded_weight, shard_id):
        rows = loaded_weight.shape[0]
        if shard_id == 0:
            param.data[:rows].copy_(loaded_weight)
        elif shard_id == 1:
            param.data[rows:].copy_(loaded_weight)
        else:
            raise AssertionError(shard_id)

    layer.mlp.gate_up_proj.weight.weight_loader = _stacked_loader

    hidden, inter = 4, 8
    loaded = pipe.load_weights(
        [
            (
                "language_model.model.layers.0.mlp.experts.0.gate_proj.weight",
                torch.ones(inter, hidden),
            ),
            (
                "language_model.model.layers.0.mlp.experts.0.up_proj.weight",
                torch.full((inter, hidden), 2.0),
            ),
            (
                "language_model.model.layers.0.mlp.gate_proj.weight",
                torch.full((inter, hidden), 3.0),
            ),
        ]
    )

    assert torch.equal(layer.mlp.gate_up_proj.weight[:inter], torch.full((inter, hidden), 3.0))
    assert torch.equal(layer.mlp.gate_up_proj.weight[inter:], torch.zeros(inter, hidden))
    assert any(c[1] == "w1" and c[2] == 0 for c in calls)
    assert any(c[1] == "w3" and c[2] == 0 for c in calls)
    assert any(name.endswith("mlp.experts.routed_experts.w13_weight") for name, *_ in calls)
    assert "language_model.model.layers.0.mlp.experts.routed_experts.w13_weight" in loaded
    assert "language_model.model.layers.0.mlp.gate_up_proj.weight" in loaded


def test_load_weights_skips_stacked_mapping_on_expert_names():
    """Without the skip, ``.gate_proj`` would rewrite expert keys to gate_up_proj."""
    pipe = _MoeStub(_tiny_dense_llm())
    layer = pipe.language_model.model.layers[0]
    with torch.no_grad():
        layer.mlp.gate_up_proj.weight.fill_(1.0)

    pipe.load_weights(
        [
            (
                "language_model.model.layers.0.mlp.experts.0.gate_proj.weight",
                torch.full((8, 4), 9.0),
            ),
        ]
    )
    assert torch.equal(layer.mlp.gate_up_proj.weight, torch.ones(16, 4))
