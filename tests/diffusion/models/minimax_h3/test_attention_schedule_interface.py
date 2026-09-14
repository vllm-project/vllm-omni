# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.attention import overlap, parallel, qkv_overlap, schedule

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_split_v_is_replaced_before_transport(monkeypatch):
    strategy = object.__new__(parallel.H3UlyssesAttention)
    strategy._ulysses_pg = object()
    order = []
    query, key, placeholder, produced_v = (torch.tensor(i) for i in range(4))

    def scatter(tensor, slot):
        order.append(slot)
        if slot == "v":
            assert tensor is produced_v
        return tensor

    monkeypatch.setattr(strategy, "_scatter_heads", scatter)
    monkeypatch.setattr(schedule, "after_q", lambda *args: order.append("prepare_q"))
    monkeypatch.setattr(qkv_overlap, "after_q", lambda: order.append("submit_v"))

    def before_v(q, k, v, metadata, group):
        assert q is query and k is key and v is placeholder
        order.append("join_v")
        return produced_v

    monkeypatch.setattr(overlap, "before_v", before_v)
    token = overlap.ACTIVE.set({})
    try:
        _, _, result = strategy._exchange_qkv(query, key, placeholder, None)
    finally:
        overlap.ACTIVE.reset(token)
    assert result is produced_v
    assert order == ["q", "prepare_q", "submit_v", "k", "join_v", "v"]


def test_default_path_uses_generic_exchange(monkeypatch):
    strategy = object.__new__(parallel.H3UlyssesAttention)
    monkeypatch.setattr(strategy, "_scatter_heads", lambda tensor, slot: tensor)
    monkeypatch.setattr(schedule, "after_q", lambda *args: pytest.fail("inactive schedule ran"))
    q, k, v = (torch.tensor(i) for i in range(3))
    assert strategy._exchange_qkv(q, k, v, None) == (q, k, v)
