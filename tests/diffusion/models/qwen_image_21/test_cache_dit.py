# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU integration checks using Cache-DiT's actual adapters and context lifecycle."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.cache.cachedit import CUSTOM_DIT_ENABLERS, CacheDiTBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.models.qwen_image_21.cache_dit import enable_qwen_image21_cache

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


class RecordingBlock(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(index + 1)))
        self.calls = []

    def forward(self, hidden_states, *, kv_cache, cache_branch, **kwargs):
        self.calls.append((cache_branch, kv_cache))
        return hidden_states + self.weight * kv_cache[cache_branch]


class FakePipeline:
    def __init__(self):
        self.transformer = nn.Module()
        self.transformer.transformer_blocks = nn.ModuleList([RecordingBlock(i) for i in range(4)])
        self.transformer.causal_condition = True
        self.transformer._decode_graph_manager = Mock()
        self.transformer.enable_cuda_graph_decode = True


def config(**kwargs):
    return DiffusionCacheConfig(max_warmup_steps=1, residual_diff_threshold=100.0, **kwargs)


def caches(scale=1):
    return [{"cond": float(scale * (i + 1)), "uncond": float(-scale * (i + 1))} for i in range(4)]


def decode(pipeline, branch, kv):
    x = torch.ones(1, 2, 4)
    return pipeline.transformer._cache_dit_decode_proxies[branch](x, kv_cache=kv, cache_branch=branch)


def expected(branch, kv):
    return torch.ones(1, 2, 4) + sum((i + 1) * layer[branch] for i, layer in enumerate(kv))


def test_layer_kv_routing_branch_isolation_and_refresh():
    pipeline = FakePipeline()
    result = enable_qwen_image21_cache(pipeline, config())
    blocks = pipeline.transformer.transformer_blocks
    for scale in (1, 7):
        kv = caches(scale)
        result.refresh(pipeline, 10, verbose=False)
        for block in blocks:
            block.calls.clear()
        for _ in range(4):
            for branch in ("cond", "uncond"):
                assert torch.equal(decode(pipeline, branch, kv), expected(branch, kv))
        # Actual middle-block skips prove that branch separation was exercised
        # with active caches, rather than only testing uncached warmup passes.
        assert len(blocks[-1].calls) < len(blocks[0].calls)
        for index, block in enumerate(blocks):
            assert {branch for branch, _ in block.calls} == {"cond", "uncond"}
            assert all(actual is kv[index] for _, actual in block.calls)


def test_state_dict_and_graph_lifecycle_unchanged_weights():
    pipeline = FakePipeline()
    transformer = pipeline.transformer
    graph = transformer._decode_graph_manager
    state = {k: v.clone() for k, v in transformer.state_dict().items()}
    parameters = list(transformer.parameters())
    enable_qwen_image21_cache(pipeline, config())
    graph.clear.assert_called_once_with()
    assert transformer._decode_graph_manager is None
    assert transformer.enable_cuda_graph_decode is False
    assert list(transformer.state_dict()) == list(state)
    assert all(torch.equal(v, state[k]) for k, v in transformer.state_dict().items())
    assert all(a is b for a, b in zip(parameters, transformer.parameters(), strict=True))


def test_backend_disable_restores_uncached_execution(monkeypatch):
    monkeypatch.setitem(CUSTOM_DIT_ENABLERS, "FakePipeline", enable_qwen_image21_cache)
    pipeline = FakePipeline()
    backend = CacheDiTBackend(config())
    backend.enable(pipeline)
    backend.refresh(pipeline, 10, verbose=False)
    for _ in range(4):
        decode(pipeline, "cond", caches())
    backend.disable(pipeline)
    assert not backend.enabled
    assert not hasattr(pipeline, "_cache_dit_targets")
    for block in pipeline.transformer.transformer_blocks:
        block.calls.clear()
    # Changing KV data after disable must execute every layer and not reuse a
    # previous request's cached residual, even if decode proxies still exist.
    for scale in (3, 5):
        kv = caches(scale)
        assert torch.equal(decode(pipeline, "cond", kv), expected("cond", kv))
    assert all(len(block.calls) == 2 for block in pipeline.transformer.transformer_blocks)
    backend.enable(pipeline)
    backend.refresh(pipeline, 10, verbose=False)
    assert torch.equal(decode(pipeline, "cond", caches(9)), expected("cond", caches(9)))
    backend.disable(pipeline)


def test_rejects_noncausal_and_scm():
    pipeline = FakePipeline()
    pipeline.transformer.causal_condition = False
    with pytest.raises(ValueError, match="causal"):
        enable_qwen_image21_cache(pipeline, config())
    pipeline.transformer.causal_condition = True
    with pytest.raises(ValueError, match="SCM"):
        enable_qwen_image21_cache(pipeline, config(scm_steps_mask_policy="fast"))
    assert not hasattr(pipeline.transformer, "_cache_dit_decode_proxies")


def test_backend_registration_controls_support():
    from vllm_omni.diffusion.cache.teacache.extractors import get_extractor
    from vllm_omni.diffusion.registry import _NO_CACHE_ACCELERATION

    assert "QwenImage21Pipeline" not in _NO_CACHE_ACCELERATION
    assert "QwenImage21Pipeline" in CUSTOM_DIT_ENABLERS
    with pytest.raises(ValueError, match="Unknown model type"):
        get_extractor("QwenImage21Transformer2DModel")
