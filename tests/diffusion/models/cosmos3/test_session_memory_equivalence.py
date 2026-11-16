# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence and session-isolation tests for Cosmos3StateAdapter (RFC #4480).

These drive the adapter directly with tiny CPU tensors -- no model, no GPU --
and assert it stores/returns the per-branch UND K/V exactly and isolates
durable state between sessions. They do not cover the transformer's mutable
load-to-forward handoff.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.experimental.world_models.adapters.state_cosmos3_adapter import (
    Cosmos3StateAdapter,
)
from vllm_omni.experimental.world_models.session_state import SessionStateManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LAYERS = 3
SEQ, DIM = 3, 4


def _fake_cached_kv(seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """A list of per-layer (K, V); contents seeded so branches/sessions differ."""
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(1, SEQ, DIM, generator=g), torch.randn(1, SEQ, DIM, generator=g)) for _ in range(LAYERS)]


def _adapter(session_id: str, manager: SessionStateManager | None = None) -> Cosmos3StateAdapter:
    # `manager is not None`, not `manager or ...`: an empty SessionStateManager is
    # falsy (it defines __len__), so `or` would silently swap in a fresh manager.
    return Cosmos3StateAdapter(session_id, manager if manager is not None else SessionStateManager())


def _assert_kv_equal(got: list | None, expected: list) -> None:
    assert got is not None and len(got) == len(expected) == LAYERS
    for (k, v), (ke, ve) in zip(got, expected):
        torch.testing.assert_close(k, ke)
        torch.testing.assert_close(v, ve)


class _FakeTransformer:
    cached_kv = None
    cached_freqs_gen = None


# ----------------------------- equivalence -----------------------------


def test_branch_starts_uninitialized() -> None:
    a = _adapter("s0")
    for is_neg in (False, True):
        assert a.is_branch_initialized(is_neg) is False
        assert a.get_branch_kv(is_neg) is None


def test_set_then_get_roundtrip() -> None:
    a = _adapter("s0")
    kv = _fake_cached_kv(1)
    a.set_branch_kv(False, kv)
    assert a.is_branch_initialized(False) is True
    _assert_kv_equal(a.get_branch_kv(False), kv)


def test_cond_uncond_branches_isolated() -> None:
    a = _adapter("s0")
    kv_pos, kv_neg = _fake_cached_kv(1), _fake_cached_kv(2)
    a.set_branch_kv(False, kv_pos)
    a.set_branch_kv(True, kv_neg)
    _assert_kv_equal(a.get_branch_kv(False), kv_pos)
    _assert_kv_equal(a.get_branch_kv(True), kv_neg)


def test_encode_once_capture_is_noop_when_initialized() -> None:
    a = _adapter("s0")
    kv = _fake_cached_kv(1)
    a.set_branch_kv(False, kv)
    tr = _FakeTransformer()
    tr.cached_kv = _fake_cached_kv(999)  # different content
    a.capture_from_transformer(tr, False)
    _assert_kv_equal(a.get_branch_kv(False), kv)  # still the first one


def test_capture_writes_when_uninitialized() -> None:
    a = _adapter("s0")
    tr = _FakeTransformer()
    tr.cached_kv = _fake_cached_kv(5)
    tr.cached_freqs_gen = (torch.ones(2), torch.zeros(2))
    a.capture_from_transformer(tr, False)
    _assert_kv_equal(a.get_branch_kv(False), _fake_cached_kv(5))


def test_capture_without_freqs_does_not_partially_initialize() -> None:
    a = _adapter("s0")
    tr = _FakeTransformer()
    tr.cached_kv = _fake_cached_kv(5)
    tr.cached_freqs_gen = None
    with pytest.raises(RuntimeError, match="did not produce GEN RoPE frequencies"):
        a.capture_from_transformer(tr, False)
    assert a.is_branch_initialized(False) is False


def test_load_into_transformer_without_freqs_sets_kv() -> None:
    a = _adapter("s0")
    kv = _fake_cached_kv(1)
    a.set_branch_kv(False, kv)
    tr = _FakeTransformer()
    a.load_into_transformer(tr, False)
    _assert_kv_equal(tr.cached_kv, kv)
    assert tr.cached_freqs_gen is None


def test_set_then_get_preserves_per_layer_values() -> None:
    # The stored value must reflect the tensors passed in (per-layer).
    a = _adapter("s0")
    kv = _fake_cached_kv(3)
    a.set_branch_kv(False, kv)
    _assert_kv_equal(a.get_branch_kv(False), kv)


def test_load_into_transformer_restores_kv_and_freqs_gen() -> None:
    a = _adapter("s0")
    tr = _FakeTransformer()
    tr.cached_kv = _fake_cached_kv(3)
    tr.cached_freqs_gen = (torch.ones(2), torch.zeros(2))
    a.capture_from_transformer(tr, False)
    tr.cached_kv = None
    tr.cached_freqs_gen = None
    a.load_into_transformer(tr, False)
    _assert_kv_equal(tr.cached_kv, _fake_cached_kv(3))
    assert tr.cached_freqs_gen is not None
    torch.testing.assert_close(tr.cached_freqs_gen[0], torch.ones(2))


def test_partial_branch_raises() -> None:
    a = _adapter("s0")
    a.set_branch_kv(False, _fake_cached_kv(1))
    a._session.get("und_kv/1").reset()  # type: ignore[union-attr]
    with pytest.raises(RuntimeError, match="partially initialized"):
        a.get_branch_kv(False)


# ----------------------------- reset -----------------------------


def test_reset_clears_both_branches() -> None:
    a = _adapter("s0")
    a.set_branch_kv(False, _fake_cached_kv(1))
    a.set_branch_kv(True, _fake_cached_kv(2))
    a.reset()
    for is_neg in (False, True):
        assert a.is_branch_initialized(is_neg) is False
        assert a.get_branch_kv(is_neg) is None


# ----------------------- concurrency isolation (core) -----------------------


def test_two_sessions_do_not_clobber() -> None:
    manager = SessionStateManager()
    a, b = _adapter("A", manager), _adapter("B", manager)
    kv_a, kv_b = _fake_cached_kv(10), _fake_cached_kv(20)
    a.set_branch_kv(False, kv_a)
    assert b.is_branch_initialized(False) is False  # B not seeing A's state
    b.set_branch_kv(False, kv_b)
    _assert_kv_equal(a.get_branch_kv(False), kv_a)
    _assert_kv_equal(b.get_branch_kv(False), kv_b)


def test_fresh_adapter_same_session_sees_prior_state() -> None:
    manager = SessionStateManager()
    kv = _fake_cached_kv(7)
    _adapter("s", manager).set_branch_kv(False, kv)
    again = Cosmos3StateAdapter("s", manager)
    _assert_kv_equal(again.get_branch_kv(False), kv)


def test_adapter_survives_session_lru_eviction() -> None:
    # An in-use adapter keeps its state even after the manager evicts its
    # session id from the lookup table (LRU bounded by max_sessions).
    manager = SessionStateManager(max_sessions=2)
    a = _adapter("active", manager)
    kv = _fake_cached_kv(11)
    a.set_branch_kv(False, kv)
    for i in range(3):
        _adapter(f"other{i}", manager)
    assert "active" not in manager
    _assert_kv_equal(a.get_branch_kv(False), kv)
