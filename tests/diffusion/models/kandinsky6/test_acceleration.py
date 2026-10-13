# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU checks that Kandinsky 6 acceleration hooks are actually wired."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.kandinsky6.cache_accel import K6_PRO_MAG_RATIOS, K6StepCache
from vllm_omni.diffusion.models.kandinsky6.kandinsky6_transformer import Kandinsky6Transformer3DModel
from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import _shard_loaded_weight
from vllm_omni.diffusion.registry import _NO_CACHE_ACCELERATION

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_shard_loaded_weight_narrows_column_parallel_output():
    class _Owner:
        tp_rank = 1

        def weight_loader(self, param, loaded):
            del param, loaded

    param = nn.Parameter(torch.empty(2, 4))
    param.output_dim = 0
    param.weight_loader = _Owner().weight_loader
    loaded = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    shard = _shard_loaded_weight(param, loaded)

    assert tuple(shard.shape) == (2, 4)
    torch.testing.assert_close(shard, loaded[2:4])


def test_row_bias_is_not_sharded_when_shapes_match():
    class _Owner:
        tp_rank = 1

        def weight_loader(self, param, loaded):
            del param, loaded

    param = nn.Parameter(torch.empty(4))
    param.output_dim = 0
    param.weight_loader = _Owner().weight_loader
    loaded = torch.arange(4, dtype=torch.float32)

    shard = _shard_loaded_weight(param, loaded)

    torch.testing.assert_close(shard, loaded)


def test_hsdp_predicate_matches_visual_blocks_only_when_indexed():
    assert Kandinsky6Transformer3DModel._is_transformer_block("visual_transformer_blocks.3", nn.Identity())
    assert Kandinsky6Transformer3DModel._is_transformer_block("video_text_transformer_blocks.0", nn.Identity())
    assert not Kandinsky6Transformer3DModel._is_transformer_block("visual_embeddings", nn.Identity())
    assert Kandinsky6Transformer3DModel._hsdp_shard_conditions


def test_sp_plan_shards_visual_tokens_and_gathers_before_the_head():
    plan = Kandinsky6Transformer3DModel._sp_plan
    assert "_sp_visual_shard" in plan
    assert "_sp_visual_rope" in plan
    assert plan["_sp_visual_gather"].gather_dim == 1
    assert Kandinsky6Transformer3DModel._magcache_block_attrs == ("visual_transformer_blocks",)


def test_k6_is_not_on_the_no_cache_list_and_enablers_are_registered():
    import vllm_omni.diffusion.cache.cachedit  # noqa: F401  registers enablers
    from vllm_omni.diffusion.cache.cachedit.backend import CUSTOM_DIT_ENABLERS
    from vllm_omni.diffusion.cache.teacache.backend import CUSTOM_TEACACHE_ENABLERS

    assert "Kandinsky6TI2VAPipeline" not in _NO_CACHE_ACCELERATION
    assert "Kandinsky6TI2VAPipeline" in CUSTOM_TEACACHE_ENABLERS
    assert "Kandinsky6TI2VAPipeline" in CUSTOM_DIT_ENABLERS
    assert len(K6_PRO_MAG_RATIOS) == 100


def test_step_cache_skips_only_after_a_real_step():
    cache = K6StepCache(
        kind="mag",
        threshold=0.24,
        max_skip_steps=3,
        retention_ratio=0.0,
        mag_ratios=(1.0, 1.0),
    )
    assert cache.should_skip(0) is False
    cache.store(torch.zeros(1), torch.zeros(1))
    assert cache.should_skip(1) is True
