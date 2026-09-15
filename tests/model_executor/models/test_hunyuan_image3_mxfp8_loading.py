# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
    HunyuanModel,
)


def test_split_qkv_weight_supports_mxfp8_group_scales():
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
        )
    )
    row_ids = torch.arange(48 * 128).reshape(-1, 1)
    group_scales = row_ids.expand(-1, 128)

    split_scales = HunyuanModel._split_qkv_weight(model, group_scales)

    grouped_rows = row_ids.reshape(8, 6, 128)
    expected_rows = torch.cat(
        (
            grouped_rows[:, :4].reshape(-1),
            grouped_rows[:, 4].reshape(-1),
            grouped_rows[:, 5].reshape(-1),
        )
    )
    assert split_scales.shape == group_scales.shape
    torch.testing.assert_close(split_scales[:, 0], expected_rows)


def test_load_weights_splits_packed_mxfp8_qkv_scales(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3._get_cla_factor",
        lambda _config: 1,
    )
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3.is_pp_missing_parameter",
        lambda _name, _model: False,
    )

    row_ids = torch.arange(48 * 128, dtype=torch.float32).reshape(-1, 1)
    packed_scale = row_ids.expand(-1, 128)
    param_name = "layers.0.self_attn.qkv_proj.weight_scale"
    param = torch.nn.Parameter(torch.empty_like(packed_scale), requires_grad=False)
    shard_offsets = {
        "q": 0,
        "k": 32 * 128,
        "v": 40 * 128,
    }
    loaded_shard_ids = []

    def weight_loader(target_param, loaded_weight, shard_id):
        offset = shard_offsets[shard_id]
        target_param.data[offset : offset + loaded_weight.shape[0]].copy_(loaded_weight)
        loaded_shard_ids.append(shard_id)

    param.weight_loader = weight_loader
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            tie_word_embeddings=False,
        ),
        named_parameters=lambda: [(param_name, param)],
        get_expert_mapping=lambda: ([], {}),
    )
    model._split_qkv_weight = lambda tensor: HunyuanModel._split_qkv_weight(model, tensor)

    loaded_params = HunyuanModel.load_weights(model, [(param_name, packed_scale)])

    grouped_rows = row_ids.reshape(8, 6, 128)
    expected_shards = {
        "q": grouped_rows[:, :4].reshape(-1, 1).expand(-1, 128),
        "k": grouped_rows[:, 4].reshape(-1, 1).expand(-1, 128),
        "v": grouped_rows[:, 5].reshape(-1, 1).expand(-1, 128),
    }
    assert loaded_params == {param_name}
    assert loaded_shard_ids == ["q", "k", "v"]
    for shard_id, expected in expected_shards.items():
        offset = shard_offsets[shard_id]
        torch.testing.assert_close(param[offset : offset + expected.shape[0]], expected)
