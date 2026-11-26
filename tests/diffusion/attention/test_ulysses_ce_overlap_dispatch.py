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
    attention = SimpleNamespace(use_ring=False, _explicit_cross_contract=False, _has_custom_attention=False)
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
        {"_has_custom_attention": True},
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


class _Impl:
    flashinfer_backend = "fa2"


def _verifying_layer(monkeypatch, *, match=True, fail=None):
    """A layer whose overlap and standard paths are fakes; records which ran and every all_reduce."""
    from vllm_omni.diffusion.attention.parallel import ulysses_ce_overlap

    monkeypatch.setattr(ulysses_ce_overlap, "VERDICTS", {})
    calls = []
    monkeypatch.setattr(layer.torch.distributed, "all_reduce", lambda t, op, group: calls.append("all_reduce"))
    reference = torch.randn(1, 8, 8, 16)
    attention = SimpleNamespace(impl=_Impl())
    strategy = SimpleNamespace(_ulysses_pg=SimpleNamespace(group_name="pg"))

    def call():
        query = torch.empty(1, 8, 8, 16)
        return layer.Attention._ulysses_ce_overlap_or_none(attention, strategy, query, query, query, None, 2)

    def overlap(strategy, q, k, v, m, chunks):
        calls.append("overlap")
        if fail == "overlap":
            raise RuntimeError("overlap failed")
        return reference.clone() if match else reference + 1

    def standard(q, k, v, m):
        # The reference run re-enters the dispatch; the pending verdict sends it down the standard exchange.
        if layer.Attention._ulysses_ce_overlap_or_none(attention, strategy, q, k, v, m, 2) is not None:
            raise AssertionError("the reference run must take the standard exchange")
        calls.append("standard")
        if fail == "standard":
            raise RuntimeError("standard failed")
        return reference

    attention._forward_ulysses_ce_overlap = overlap
    attention._forward_impl = standard
    attention.effective_attention = lambda: (attention.impl, None, None)
    return attention, call, calls, reference, ulysses_ce_overlap.VERDICTS


@pytest.mark.parametrize("match", [True, False])
def test_overlap_is_verified_once_then_reused(monkeypatch, match):
    """The first call at a shape runs both paths; later calls follow the cached verdict without verifying again."""
    _, call, calls, reference, _ = _verifying_layer(monkeypatch, match=match)
    assert call() is reference
    assert calls == ["overlap", "standard", "all_reduce"]
    calls.clear()
    second = call()
    if match:
        assert second is not None and calls == ["overlap"]
    else:
        assert second is None and calls == []


def test_verdict_is_keyed_by_attention_kernel(monkeypatch):
    """A scheduled switch to another kernel at the same shape is verified on its own."""
    attention, call, calls, _, _ = _verifying_layer(monkeypatch)
    call()
    attention.impl = SimpleNamespace(flashinfer_backend="cute-dsl")
    calls.clear()
    call()
    assert calls == ["overlap", "standard", "all_reduce"]


@pytest.mark.parametrize("fail", ["overlap", "standard"])
def test_failed_verification_leaves_no_verdict(monkeypatch, fail):
    """A failure on one rank must not leave a verdict that sends it down a different path than its peers later."""
    _, call, _, _, verdicts = _verifying_layer(monkeypatch, fail=fail)
    with pytest.raises(RuntimeError):
        call()
    assert verdicts == {}
