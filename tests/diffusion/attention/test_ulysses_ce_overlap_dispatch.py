# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention import layer
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel import ulysses_ce_overlap
from vllm_omni.diffusion.kv_offload_config import REFERENCE_ISOLATION

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _CudaTensor:
    """A CPU tensor that reports ``is_cuda`` so eligibility can be checked without a GPU."""

    is_cuda = True

    def __init__(self, tensor):
        self._tensor = tensor

    def __getattr__(self, name):
        return getattr(self._tensor, name)


@pytest.fixture(autouse=True)
def _nvidia_cuda(monkeypatch):
    monkeypatch.setattr(layer, "_is_nvidia_cuda", lambda: True)


def _case(**overrides):
    attention = SimpleNamespace(use_ring=False, _explicit_cross_contract=False)
    strategy = SimpleNamespace(
        name="ulysses",
        ce_overlap_chunks=2,
        _ulysses_a2a_permute=True,
        _scatter_idx=2,
        _gather_idx=1,
        _sp_group=SimpleNamespace(ring_world_size=1, ulysses_world_size=2),
    )
    tensor = torch.empty(1, 8, 8, 16)
    args = dict(query=_CudaTensor(tensor), key=_CudaTensor(tensor), value=_CudaTensor(tensor))
    args["attn_metadata"] = AttentionMetadata()
    args["use_paged_attention"] = False
    for name, value in overrides.items():
        target = attention if hasattr(attention, name) else strategy if hasattr(strategy, name) else None
        if target is None:
            args[name] = value
        else:
            setattr(target, name, value)
    return attention, strategy, args


@pytest.mark.parametrize(
    "overrides",
    [
        {"ce_overlap_chunks": 0},
        {"name": "ring"},
        {"use_ring": True},
        {"_explicit_cross_contract": True},
        {"_ulysses_a2a_permute": False},
        {"_scatter_idx": 1},
        {"_sp_group": SimpleNamespace(ring_world_size=2, ulysses_world_size=2)},
        {"_sp_group": SimpleNamespace(ring_world_size=1, ulysses_world_size=1)},
        {"use_paged_attention": True},
        {"query": torch.empty(1, 8, 8, 16)},
        {"key": _CudaTensor(torch.empty(1, 8, 4, 16))},
        {"value": _CudaTensor(torch.empty(1, 8, 16, 16)[:, :, ::2])},
        {"query": _CudaTensor(torch.empty(1, 8, 6, 16))},
        {"attn_metadata": AttentionMetadata(attn_mask=torch.ones(1, 8, dtype=torch.bool))},
        {"attn_metadata": AttentionMetadata(joint_query=torch.empty(1, 2, 8, 16))},
        {"attn_metadata": AttentionMetadata(extra={"gate_compress": torch.empty(1)})},
        {"attn_metadata": AttentionMetadata(extra={REFERENCE_ISOLATION: (None, None)})},
    ],
)
def test_ineligible_layers_keep_the_standard_path(monkeypatch, overrides):
    monkeypatch.setattr(layer, "get_ulysses_mode", lambda default: "strict")
    attention, strategy, args = _case(**overrides)
    assert layer.Attention._ulysses_ce_overlap_chunks(attention, strategy, **args) == 0


def test_non_strict_mode_keeps_the_standard_path(monkeypatch):
    monkeypatch.setattr(layer, "get_ulysses_mode", lambda default: "advanced_uaa")
    attention, strategy, args = _case()
    assert layer.Attention._ulysses_ce_overlap_chunks(attention, strategy, **args) == 0


@pytest.mark.parametrize("platform_is_cuda, hip", [(False, None), (True, "6.4")])
def test_rocm_keeps_the_standard_path(monkeypatch, platform_is_cuda, hip):
    """ROCm tensors report ``is_cuda``; the copy-engine overlap must not load the NVIDIA driver there."""
    monkeypatch.undo()
    monkeypatch.setattr(layer, "get_ulysses_mode", lambda default: "strict")
    monkeypatch.setattr(layer.current_omni_platform, "is_cuda", lambda: platform_is_cuda)
    monkeypatch.setattr(layer.torch.version, "hip", hip)
    attention, strategy, args = _case()
    assert layer.Attention._ulysses_ce_overlap_chunks(attention, strategy, **args) == 0


def test_eligible_layer_runs_local_attention_per_chunk(monkeypatch):
    monkeypatch.setattr(layer, "get_ulysses_mode", lambda default: "strict")
    attention, strategy, args = _case()
    assert layer.Attention._ulysses_ce_overlap_chunks(attention, strategy, **args) == 2

    metadata = AttentionMetadata()
    calls = []
    attention._with_kv_cache_dtype = lambda m: m
    attention._run_local_attention = lambda q, k, v, m: calls.append(m) or q

    def runner(pg, query, key, value, attn_fn, *, chunks):
        assert pg == "pg" and chunks == 2
        return attn_fn(query, key, value)

    monkeypatch.setattr(ulysses_ce_overlap, "run_ulysses_ce_overlap", runner)
    strategy._ulysses_pg = "pg"
    query = torch.randn(1, 8, 8, 16)
    output = layer.Attention._forward_ulysses_ce_overlap(attention, strategy, query, query, query, metadata, 2)
    assert output is query and calls == [metadata]
