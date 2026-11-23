# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.vdnh3_attn import VDNAttentionImpl
from vllm_omni.diffusion.models.minimax_h3 import head_overlap

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def collective(monkeypatch):
    group = object()
    strategy = SimpleNamespace(
        name="ulysses", _ulysses_pg=group, _scatter_idx=2, _gather_idx=1, _use_sync=False, _ulysses_a2a_permute=False
    )
    impl = object.__new__(head_overlap.FlashAttentionImpl)
    impl._kernel_variant = "aiter"
    impl.causal = impl.is_cross_attn = impl.fa_deterministic = False
    impl.softmax_scale = 128**-0.5
    layer = SimpleNamespace(
        effective_attention=lambda: (impl, None, None),
        _get_active_parallel_strategy=lambda: strategy,
        _effective_kv_cache_dtype=lambda: None,
    )
    owner = SimpleNamespace(
        attention=layer, num_heads=56, num_kv_heads=56, head_dim=128, to_gate_compress=None, vsa_sparsity=None, vdn=None
    )
    blocks = [SimpleNamespace(attn=owner)]
    monkeypatch.setattr(head_overlap, "get_sp_group", lambda: SimpleNamespace(ulysses_group=group))
    monkeypatch.setattr(head_overlap.dist, "get_world_size", lambda _: 4)
    monkeypatch.setattr(head_overlap, "get_ulysses_mode", lambda *, default: "strict")
    monkeypatch.setattr(head_overlap, "prepare_head_fmha", lambda device: object())
    monkeypatch.setattr(torch.cpu, "is_current_stream_capturing", lambda: False, raising=False)

    def prepare(*, bounds=(0, 37749, 37760), cut=37749, peer_cut=37749, expected_vote=37749, grad=False):
        votes = []

        def all_reduce(vote, *, op, group):
            assert op == head_overlap.dist.ReduceOp.MIN and group is strategy._ulysses_pg
            votes.append(vote.tolist())
            # These two entries are the MIN of the other three ranks. A rank
            # that cannot participate sets the first entry to zero.
            vote.copy_(torch.minimum(vote, vote.new_tensor([peer_cut, -peer_cut])))

        monkeypatch.setattr(head_overlap.dist, "all_reduce", all_reduce)
        hidden = torch.empty((9440, 1), dtype=torch.bfloat16)
        with torch.set_grad_enabled(grad):
            accepted = head_overlap.prepare_overlap(blocks, hidden, torch.tensor(bounds, dtype=torch.int32), cut)
        assert votes == [[expected_vote, -expected_vote]]
        return accepted

    return SimpleNamespace(prepare=prepare, layer=layer, strategy=strategy, impl=impl, blocks=blocks)


@pytest.mark.parametrize(("peer_cut", "accepted"), [(37749, True), (0, False), (37746, False), (37757, False)])
def test_all_four_ranks_must_agree_on_the_same_cut(collective, peer_cut, accepted):
    assert collective.prepare(peer_cut=peer_cut) is accepted


@pytest.mark.parametrize(
    ("bounds", "cut"), [((1, 37749, 37760), 37749), ((0, 37750, 37760), 37749), ((0, 37752, 37760), 37752)]
)
def test_unqualified_packed_bounds_veto_the_exchange(collective, bounds, cut):
    assert not collective.prepare(bounds=bounds, cut=cut, expected_vote=0)


def test_gradients_veto_the_exchange_even_with_detached_hidden(collective):
    assert not collective.prepare(grad=True, expected_vote=0)


def test_unavailable_native_kernel_vetoes_the_exchange(collective, monkeypatch):
    monkeypatch.setattr(head_overlap, "prepare_head_fmha", lambda device: None)
    assert not collective.prepare(expected_vote=0)


def test_only_strict_ulysses_can_use_split_head_packets(collective, monkeypatch):
    monkeypatch.setattr(head_overlap, "get_ulysses_mode", lambda *, default: "advanced_uaa")
    assert not collective.prepare(expected_vote=0)


def test_sync_ulysses_vetoes_the_exchange(collective):
    collective.strategy._use_sync = True
    assert not collective.prepare(expected_vote=0)


def test_other_flash_kernel_vetoes_the_exchange(collective):
    collective.impl._kernel_variant = "fa4"
    assert not collective.prepare(expected_vote=0)


def test_scheduled_vdn_implementation_keeps_its_window_attention(collective):
    vdn = object.__new__(VDNAttentionImpl)
    collective.layer.effective_attention = lambda: (vdn, None, None)
    assert not collective.prepare(expected_vote=0)


def test_kv_quantized_attention_keeps_the_configured_backend(collective):
    collective.layer._effective_kv_cache_dtype = lambda: "fp8"
    assert not collective.prepare(expected_vote=0)


def test_cached_block_wrapper_keeps_its_own_forward(collective):
    collective.blocks[:] = [SimpleNamespace(transformer_blocks=list(collective.blocks))]
    assert not collective.prepare(expected_vote=0)
