# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint scalar scales must survive packed projection remapping."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.hunyuan_image3 import hunyuan_image3

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _Load:
    tensor: torch.Tensor
    shard: str | int
    expert: int | None


class _Parameter:
    def __init__(self):
        self.loads: list[_Load] = []

    def weight_loader(self, param, tensor, shard_or_name, *, shard_id=None, expert_id=None, return_success=False):
        self.loads.append(_Load(tensor.clone(), shard_or_name if shard_id is None else shard_id, expert_id))
        return True


class _Model:
    load_weights = hunyuan_image3.HunyuanModel.load_weights
    _split_qkv_weight = hunyuan_image3.HunyuanModel._split_qkv_weight

    def __init__(self, name: str):
        self.config = SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            hidden_size=8,
            use_cla=False,
            tie_word_embeddings=False,
        )
        self.name = name
        self.param = _Parameter()

    def named_parameters(self):
        return [(self.name, self.param)]

    def get_expert_mapping(self):
        return [
            ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
            ("experts.w13_", "experts.0.up_proj.", 0, "w3"),
        ]

    def _get_expert_weights_remapping(self):
        return {"gate_proj": ("gate_and_up_proj", 1, 2), "up_proj": ("gate_and_up_proj", 0, 2)}


@pytest.fixture(autouse=True)
def local_parameters(monkeypatch):
    monkeypatch.setattr(hunyuan_image3, "is_pp_missing_parameter", lambda name, model: False)


@pytest.mark.parametrize("shape", [(), (1,)])
@pytest.mark.parametrize("scale", ["input_scale", "weight_scale", "weight_scale_2"])
@pytest.mark.parametrize(
    "source,target,shards",
    [
        ("self_attn.qkv_proj.", "self_attn.qkv_proj.", ["q", "k", "v"]),
        ("mlp.gate_and_up_proj.", "mlp.gate_up_proj.", [1, 0]),
        ("mlp.experts.0.gate_and_up_proj.", "mlp.experts.w13_", ["w1", "w3"]),
    ],
)
def test_packed_projection_broadcasts_scalar_scale(shape, scale, source, target, shards):
    target_name = "layers.0." + target + scale
    model = _Model(target_name)
    value = torch.full(shape, 0.125)
    assert model.load_weights([("layers.0." + source + scale, value)]) == {target_name}
    assert [load.shard for load in model.param.loads] == shards
    for load in model.param.loads:
        torch.testing.assert_close(load.tensor, value, rtol=0, atol=0)
        assert load.expert == (0 if "experts" in target else None)


def test_qkv_weight_keeps_interleaved_head_order():
    name = "layers.0.self_attn.qkv_proj.weight"
    model = _Model(name)
    weight = torch.arange(16 * 8).reshape(16, 8)
    assert model.load_weights([(name, weight)]) == {name}
    expected_rows = ([0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 12, 13], [6, 7, 14, 15])
    for load, rows in zip(model.param.loads, expected_rows):
        torch.testing.assert_close(load.tensor, weight[rows], rtol=0, atol=0)


@pytest.mark.parametrize("expert", [False, True])
def test_gate_up_weight_keeps_checkpoint_up_then_gate_order(expert):
    source = "layers.0.mlp." + ("experts.0." if expert else "") + "gate_and_up_proj.weight"
    target = "layers.0.mlp." + ("experts.w13_weight" if expert else "gate_up_proj.weight")
    model = _Model(target)
    weight = torch.arange(8 * 8).reshape(8, 8)
    assert model.load_weights([(source, weight)]) == {target}
    expected = [weight[4:], weight[:4]] if expert else [weight[:4], weight[4:]]
    for load, tensor in zip(model.param.loads, expected):
        torch.testing.assert_close(load.tensor, tensor, rtol=0, atol=0)
