# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.parallel import ulysses
from vllm_omni.diffusion.distributed import a2a_permute

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("backend,legacy_flag", [("ce", False), (None, True), (None, False), ("nccl", True)])
def test_selected_transport_handles_qkv_and_output(monkeypatch, backend, legacy_flag):
    sp = SimpleNamespace(ulysses_group=SimpleNamespace(group_name="subgroup"), ulysses_world_size=2, ring_world_size=1)
    use_ce = backend == "ce" or (backend is None and legacy_flag)
    initialized: list[bool] = []
    exchanges: list[int] = []
    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", lambda: initialized.append(True))
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda default: "strict")
    monkeypatch.setattr(ulysses.dist, "get_world_size", lambda pg: 2)

    def exchange(x, group_name, world_size, mode):
        assert group_name == "subgroup" and world_size == 2
        exchanges.append(mode)
        b, s, h, d = x.shape
        return x.new_empty((b, s * 2, h // 2, d) if mode == 0 else (b, s // 2, h * 2, d))

    def nccl_exchange(pg, x, scatter, gather, sync):
        assert not use_ce and pg is sp.ulysses_group
        return exchange(x, pg.group_name, 2, 0 if scatter == 2 else 1)

    monkeypatch.setattr(ulysses.SeqAllToAll4D, "apply", nccl_exchange)
    monkeypatch.setattr(a2a_permute, "ulysses_o_rev", lambda x, g, w: exchange(x, g, w, 1))

    def exchange_qkv(q, k, v, group_name, world_size):
        assert group_name == "subgroup" and world_size == 2 and use_ce
        exchanges.append(2)
        return q.new_empty(q.numel() + k.numel() + v.numel())

    monkeypatch.setattr(a2a_permute, "ulysses_qkv_ce", exchange_qkv)
    strategy = ulysses.UlyssesParallelAttention(
        sp, 2, 1, False, ulysses_a2a_permute=legacy_flag, ulysses_a2a_backend=backend
    )
    q = torch.empty(1, 16, 8, 64)
    q, k, v, _, ctx = strategy.pre_attention(q, q, q, None)
    result = strategy.post_attention(q, ctx)
    assert initialized == ([True] if use_ce else [])
    assert exchanges == ([2, 1] if use_ce else [0, 0, 0, 1])
    assert result.shape == (1, 16, 8, 64)


@pytest.mark.parametrize("scatter,gather,size", [(1, 2, 2), (2, 1, 1)])
def test_ce_rejects_unsupported_layout_before_initialization(monkeypatch, scatter, gather, size):
    sp = SimpleNamespace(ulysses_group=None, ulysses_world_size=size)
    monkeypatch.setattr(
        a2a_permute, "ensure_a2a_permute_available", lambda: pytest.fail("Unexpected CE initialization")
    )
    with pytest.raises(ValueError, match="strict 4D"):
        ulysses.UlyssesParallelAttention(sp, scatter, gather, False, ulysses_a2a_backend="ce")
    strategy = ulysses.UlyssesParallelAttention(sp, scatter, gather, False, ulysses_a2a_permute=True)
    assert not strategy._use_ce
