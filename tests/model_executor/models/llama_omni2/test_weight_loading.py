# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.model_executor.models.llama_omni2.llama_omni2_talker import (
    TALKER_WEIGHTS_MAPPER,
)
from vllm_omni.model_executor.models.llama_omni2.llama_omni2_thinker import (
    THINKER_WEIGHTS_MAPPER,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "llama_omni2_0_5b_weight_keys.txt"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("mapper", "source", "destination", "shard_id"),
    (
        (
            THINKER_WEIGHTS_MAPPER,
            "model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.qkv_proj.weight",
            "q",
        ),
        (
            THINKER_WEIGHTS_MAPPER,
            "model.layers.0.mlp.gate_proj.weight",
            "language_model.model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            TALKER_WEIGHTS_MAPPER,
            "speech_generator.model.model.layers.0.self_attn.v_proj.weight",
            "language_model.model.layers.0.self_attn.qkv_proj.weight",
            "v",
        ),
        (
            TALKER_WEIGHTS_MAPPER,
            "speech_generator.model.model.layers.0.mlp.up_proj.weight",
            "language_model.model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
    ),
)
def test_qwen2_weights_map_to_packed_vllm_parameters(
    mapper,
    source,
    destination,
    shard_id,
):
    assert mapper._map_name_with_shard(source) == (destination, shard_id)


def test_real_checkpoint_keys_have_exactly_one_stage_owner():
    keys = _FIXTURE.read_text().splitlines()
    owners = {}

    for key in keys:
        thinker = THINKER_WEIGHTS_MAPPER._map_name_with_shard(key)
        talker = TALKER_WEIGHTS_MAPPER._map_name_with_shard(key)
        assert (thinker is None) != (talker is None), key
        owners[key] = "thinker" if thinker is not None else "talker"

    assert len(keys) == 1079
    assert sum(owner == "thinker" for owner in owners.values()) == 782
    assert sum(owner == "talker" for owner in owners.values()) == 297


class _RecordingParameter(nn.Parameter):
    pass


class _PackedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = _RecordingParameter(torch.zeros(1))
        self.calls = []

    def load_weights(self, weights):
        loaded = set()
        for name, loaded_weight in weights:
            assert name == "weight"
            self.calls.append(
                (
                    loaded_weight.clone(),
                    getattr(loaded_weight, "shard_id"),
                )
            )
            loaded.add(name)
        return loaded


class _TinyMappedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([nn.Module()])
        layer = self.language_model.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj = _PackedModule()


def test_auto_loader_passes_qkv_shard_ids_to_parameter_loader():
    model = _TinyMappedModel()
    loader = AutoWeightsLoader(model)

    loaded = loader.load_weights(
        [
            (
                "model.layers.0.self_attn.q_proj.weight",
                torch.tensor([1.0]),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                torch.tensor([2.0]),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                torch.tensor([3.0]),
            ),
        ],
        mapper=THINKER_WEIGHTS_MAPPER,
    )

    packed = model.language_model.model.layers[0].self_attn.qkv_proj
    assert [shard_id for _, shard_id in packed.calls] == ["q", "k", "v"]
    assert loaded == {"language_model.model.layers.0.self_attn.qkv_proj.weight"}


@pytest.fixture
def tp2_rank(monkeypatch):
    rank = {"value": 0}
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        lambda: rank["value"],
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: rank["value"],
    )
    return rank


def _with_shard_id(tensor: torch.Tensor, shard_id):
    tensor.shard_id = shard_id
    return tensor


def test_tp2_qkv_and_gate_up_load_only_local_shards(tp2_rank):
    q_weight = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
    k_weight = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8) + 1000
    v_weight = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8) + 2000
    gate_weight = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
    up_weight = gate_weight + 3000
    qkv_by_rank = []
    gate_up_by_rank = []

    for rank in (0, 1):
        tp2_rank["value"] = rank
        qkv = QKVParallelLinear(
            hidden_size=8,
            head_size=2,
            total_num_heads=4,
            total_num_kv_heads=2,
            bias=False,
        )
        list(
            qkv.load_weights(
                [
                    ("weight", _with_shard_id(q_weight.clone(), "q")),
                    ("weight", _with_shard_id(k_weight.clone(), "k")),
                    ("weight", _with_shard_id(v_weight.clone(), "v")),
                ]
            )
        )
        qkv_by_rank.append(qkv.weight.detach().clone())

        gate_up = MergedColumnParallelLinear(
            input_size=8,
            output_sizes=[8, 8],
            bias=False,
        )
        list(
            gate_up.load_weights(
                [
                    ("weight", _with_shard_id(gate_weight.clone(), 0)),
                    ("weight", _with_shard_id(up_weight.clone(), 1)),
                ]
            )
        )
        gate_up_by_rank.append(gate_up.weight.detach().clone())

    assert qkv_by_rank[0].shape == qkv_by_rank[1].shape == (8, 8)
    assert torch.equal(qkv_by_rank[0][:4], q_weight[:4])
    assert torch.equal(qkv_by_rank[1][:4], q_weight[4:])
    assert torch.equal(qkv_by_rank[0][4:6], k_weight[:2])
    assert torch.equal(qkv_by_rank[1][4:6], k_weight[2:])
    assert torch.equal(qkv_by_rank[0][6:], v_weight[:2])
    assert torch.equal(qkv_by_rank[1][6:], v_weight[2:])

    assert gate_up_by_rank[0].shape == gate_up_by_rank[1].shape == (8, 8)
    assert torch.equal(gate_up_by_rank[0][:4], gate_weight[:4])
    assert torch.equal(gate_up_by_rank[1][:4], gate_weight[4:])
    assert torch.equal(gate_up_by_rank[0][4:], up_weight[:4])
    assert torch.equal(gate_up_by_rank[1][4:], up_weight[4:])


def test_tp2_talker_projection_linears_load_column_and_row_shards(tp2_rank):
    column_weight = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
    row_weight = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
    columns = []
    rows = []

    for rank in (0, 1):
        tp2_rank["value"] = rank
        column = ColumnParallelLinear(
            input_size=8,
            output_size=16,
            bias=False,
            gather_output=False,
        )
        column.weight.weight_loader(column.weight, column_weight)
        columns.append(column.weight.detach().clone())

        row = RowParallelLinear(
            input_size=16,
            output_size=8,
            bias=False,
            input_is_parallel=True,
        )
        row.weight.weight_loader(row.weight, row_weight)
        rows.append(row.weight.detach().clone())

    assert columns[0].shape == columns[1].shape == (8, 8)
    assert torch.equal(columns[0], column_weight[:8])
    assert torch.equal(columns[1], column_weight[8:])
    assert rows[0].shape == rows[1].shape == (8, 8)
    assert torch.equal(rows[0], row_weight[:, :8])
    assert torch.equal(rows[1], row_weight[:, 8:])
