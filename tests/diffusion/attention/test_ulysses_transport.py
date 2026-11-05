# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.parallel import ulysses
from vllm_omni.diffusion.distributed import a2a_permute, ce_ulysses

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("backend,compiling", [("ce", False), ("ce", True), (None, False)])
def test_selected_transport_handles_qkv_and_output(monkeypatch, backend, compiling):
    sp = SimpleNamespace(ulysses_group=SimpleNamespace(group_name="subgroup"), ulysses_world_size=2, ring_world_size=1)
    initialized: list[object] = []
    exchanges: list[int] = []
    monkeypatch.setattr(ce_ulysses, "initialize_ulysses_groups", initialized.append)
    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", lambda: initialized.append("lsa"))
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda default: "strict")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    monkeypatch.setattr(ulysses.dist, "get_world_size", lambda pg: 2)
    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", lambda *args: pytest.fail("Unexpected NCCL fallback"))

    slots: list[int | None] = []

    def exchange(x, group_name, world_size, mode, slot=None):
        assert group_name == "subgroup" and world_size == 2
        exchanges.append(mode)
        slots.append(slot)
        b, s, h, d = x.shape
        return x.new_empty((b, s * 2, h // 2, d) if mode == 0 else (b, s // 2, h * 2, d))

    monkeypatch.setattr(ce_ulysses, "ulysses_all_to_all_4d", exchange)
    monkeypatch.setattr(a2a_permute, "ulysses_qkv_fwd", lambda x, g, w: exchange(x, g, w, 0))
    monkeypatch.setattr(a2a_permute, "ulysses_o_rev", lambda x, g, w: exchange(x, g, w, 1))
    strategy = ulysses.UlyssesParallelAttention(sp, 2, 1, False, ulysses_a2a_permute=True, ulysses_a2a_backend=backend)
    q = torch.empty(1, 16, 8, 64)
    q, k, v, _, ctx = strategy.pre_attention(q, q, q, None)
    result = strategy.post_attention(q, ctx)
    assert initialized == [sp if backend == "ce" else "lsa"]
    assert exchanges == [0, 0, 0, 1]
    expected_slots: list[int | None] = [-1] * 4 if compiling else [0, 1, 2, 3]
    assert slots == (expected_slots if backend == "ce" else [None] * 4)
    assert result.shape == (1, 16, 8, 64)


@pytest.mark.parametrize("scatter,gather,size", [(1, 2, 2), (2, 1, 1)])
def test_ce_rejects_unsupported_layout_before_initialization(scatter, gather, size):
    sp = SimpleNamespace(ulysses_group=None, ulysses_world_size=size)
    with pytest.raises(ValueError, match="strict 4D"):
        ulysses.UlyssesParallelAttention(sp, scatter, gather, False, ulysses_a2a_backend="ce")
