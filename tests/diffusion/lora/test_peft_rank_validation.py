# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate physical factors without rejecting PEFT's per-module patterns."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.lora.utils import validate_generic_peft_ranks

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _checkpoint(path, *, rank=8, b_rank=None, rank_pattern=None, alpha_pattern=None, use_rslora=False):
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "lora_alpha": 8,
                "target_modules": ["q_proj", "k_proj"],
                "rank_pattern": rank_pattern or {},
                "alpha_pattern": alpha_pattern or {},
                "use_rslora": use_rslora,
            }
        )
    )
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(rank, 16),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.full((32, b_rank or rank), 2.0),
            "base_model.model.layers.0.self_attn.k_proj.lora_A.weight": torch.ones(8, 16),
            "base_model.model.layers.0.self_attn.k_proj.lora_B.weight": torch.full((32, 8), 2.0),
        },
        path / "adapter_model.safetensors",
    )


def _load(path, *, optimize=True):
    manager = object.__new__(DiffusionLoRAManager)
    manager._expected_lora_modules = {"q_proj", "k_proj"}
    manager.pipeline = SimpleNamespace()
    manager.dtype = torch.float32
    manager._scale_in_epilogue = not optimize
    return manager._load_adapter(LoRARequest("rank-pattern", 1, str(path)))


@pytest.mark.parametrize(
    "rank,rank_pattern,alpha_pattern,use_rslora,expected_scale",
    [
        (4, {"q_proj": 4}, {"q_proj": 16}, False, 4.0),
        (8, {}, {"q_proj": 16}, False, 2.0),
        (4, {"q_proj": 4}, {"q_proj": 16}, True, 8.0),
    ],
)
def test_loader_preserves_pattern_rank_and_folds_scaling(
    tmp_path, rank, rank_pattern, alpha_pattern, use_rslora, expected_scale
):
    _checkpoint(
        tmp_path,
        rank=rank,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        use_rslora=use_rslora,
    )
    raw, _ = _load(tmp_path, optimize=False)
    q = raw.get_lora("layers.0.self_attn.q_proj")
    assert q.rank == rank
    assert q.scaling == expected_scale
    torch.testing.assert_close(q.lora_b, torch.full((32, rank), 2.0), rtol=0, atol=0)

    optimized, _ = _load(tmp_path)
    q = optimized.get_lora("layers.0.self_attn.q_proj")
    assert q.rank == rank
    assert q.scaling == 1
    torch.testing.assert_close(q.lora_b, torch.full((32, rank), 2.0 * expected_scale), rtol=0, atol=0)


def test_uniform_rank_without_patterns_is_accepted(tmp_path):
    _checkpoint(tmp_path)
    model, _ = _load(tmp_path)
    assert all(layer.rank == 8 for layer in model.loras.values())


@pytest.mark.parametrize("a_rank,b_rank", [(24, 24), (8, 4)])
def test_loader_rejects_factors_that_disagree_with_resolved_rank(tmp_path, a_rank, b_rank):
    _checkpoint(tmp_path, rank=a_rank, b_rank=b_rank)
    with pytest.raises(ValueError, match="resolved PEFT ranks"):
        _load(tmp_path)


def test_guard_uses_resolved_rank_even_after_module_name_mapping():
    helper = PEFTHelper.from_dict(
        {"r": 8, "lora_alpha": 8, "target_modules": ["q_proj"], "rank_pattern": {"q_proj": 4}}
    )
    layer = LoRALayerWeights.from_config("mapped.query", helper, "layers.0.self_attn.q_proj")
    layer.lora_a = torch.ones(4, 16)
    layer.lora_b = torch.ones(32, 4)
    validate_generic_peft_ranks({"mapped.query": layer})
    assert layer.rank == 4


def test_guard_leaves_non_2d_moe_layout_to_existing_loader():
    layer = LoRALayerWeights("layers.0.experts", 4, 8, torch.ones(2, 4, 16), torch.ones(2, 32, 4))
    validate_generic_peft_ranks({layer.module_name: layer})
