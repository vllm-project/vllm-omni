# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _strategy():
    group = SimpleNamespace(ulysses_group=object(), ulysses_world_size=2, ulysses_rank=0, ring_world_size=1)
    strategy = UlyssesParallelAttention(group, 2, 1, False)
    strategy._ulysses_a2a_permute = True
    return strategy


def test_exchange_keeps_gate_with_query_layout(monkeypatch):
    strategy = _strategy()
    calls = []

    def scatter(tensor, slot):
        calls.append(slot)
        return tensor.repeat_interleave(2, dim=1)[:, :, :2].contiguous()

    monkeypatch.setattr(strategy, "_scatter_heads", scatter)
    q = torch.randn(1, 3, 4, 8)
    metadata = AttentionMetadata(extra={"gate_compress": q.clone()})
    query, key, value, metadata, ctx = strategy.pre_attention(q, q, q, metadata)
    assert calls == ["q", "k", "v", "g"]
    assert query.shape == key.shape == value.shape == metadata.extra["gate_compress"].shape == (1, 6, 2, 8)
    assert not ctx.use_uaa


def test_backend_selection_keeps_optional_dependency_lazy(monkeypatch):
    from vllm_omni.diffusion.distributed import flashinfer_ulysses

    def forbidden():
        pytest.fail("default Ulysses must not inspect optional FlashInfer dependencies")

    monkeypatch.setattr(flashinfer_ulysses, "ensure_flashinfer_pcie_available", forbidden)
    monkeypatch.setenv("VLLM_OMNI_ULYSSES_A2A_BACKEND", "flashinfer-pcie")
    assert _strategy()._ulysses_a2a_backend == "symmetric"


def test_strict_validation_precedes_exchange(monkeypatch):
    strategy = _strategy()
    monkeypatch.setattr(strategy, "_exchange_qkv", lambda *args: pytest.fail("invalid heads reached transport"))
    q = torch.zeros(1, 3, 3, 8)
    with pytest.raises(ValueError, match="head_cnt divisible"):
        strategy.pre_attention(q, q, q, None)
