# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.e2e.offline_inference.llama_omni2.verify_tp_shards import (
    build_expected_local_parameters,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_build_expected_local_parameters_matches_qwen2_tp_layout():
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(8 * 8).view(8, 8),
        "model.layers.0.self_attn.k_proj.weight": torch.arange(4 * 8).view(4, 8) + 1000,
        "model.layers.0.self_attn.v_proj.weight": torch.arange(4 * 8).view(4, 8) + 2000,
        "model.layers.0.self_attn.o_proj.weight": torch.arange(8 * 8).view(8, 8) + 3000,
        "model.layers.0.mlp.gate_proj.weight": torch.arange(8 * 8).view(8, 8) + 4000,
        "model.layers.0.mlp.up_proj.weight": torch.arange(8 * 8).view(8, 8) + 5000,
        "model.layers.0.mlp.down_proj.weight": torch.arange(8 * 8).view(8, 8) + 6000,
    }

    rank_one = build_expected_local_parameters(
        tensors,
        source_prefix="model.layers.0.",
        tp_rank=1,
        tp_world_size=2,
    )

    assert torch.equal(
        rank_one["language_model.model.layers.0.self_attn.qkv_proj.weight"],
        torch.cat(
            [
                tensors["model.layers.0.self_attn.q_proj.weight"][4:],
                tensors["model.layers.0.self_attn.k_proj.weight"][2:],
                tensors["model.layers.0.self_attn.v_proj.weight"][2:],
            ]
        ),
    )
    assert torch.equal(
        rank_one["language_model.model.layers.0.self_attn.o_proj.weight"],
        tensors["model.layers.0.self_attn.o_proj.weight"][:, 4:],
    )
    assert torch.equal(
        rank_one["language_model.model.layers.0.mlp.gate_up_proj.weight"],
        torch.cat(
            [
                tensors["model.layers.0.mlp.gate_proj.weight"][4:],
                tensors["model.layers.0.mlp.up_proj.weight"][4:],
            ]
        ),
    )
    assert torch.equal(
        rank_one["language_model.model.layers.0.mlp.down_proj.weight"],
        tensors["model.layers.0.mlp.down_proj.weight"][:, 4:],
    )
