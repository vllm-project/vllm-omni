# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-forward permutation must preserve exact transport bytes and layouts."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def load_module():
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_parallel.py"
    spec = importlib.util.spec_from_file_location("vision_permutation_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_bytes(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert left.stride() == right.stride()
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


@pytest.mark.parametrize("size", [2, 4, 8, 16])
@pytest.mark.parametrize("lengths", [(4,), (4, 12, 0, 20), (0, 0), (8160, 8160, 880, 660, 660, 7656)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_permutation_matches_reference_and_caches_only_metadata(monkeypatch, size, lengths, dtype):
    m = load_module()
    plan = m.build_vision_plan(lengths, size)
    monkeypatch.setattr(m.dist, "get_rank", lambda group=None: size - 1)
    monkeypatch.setattr(m.dist, "get_world_size", lambda group=None: size)
    transport = m.VisionAllToAll(plan, use_permutation=True)
    inverse, source, padding = transport._permutation_for(torch.device("cpu"))
    identities = tuple(id(x) for x in (inverse, source, padding))
    total = sum(lengths)
    for offset in (0, 7):
        full = (torch.arange(total * 6).reshape(total, 3, 2).transpose(1, 2) + offset).to(dtype)
        if total:
            full[0, 0] = torch.tensor([-0.0, float("inf"), float("nan")], dtype=dtype)
        expected = m.pack_tokens(full, plan)
        actual = m.pack_tokens(full, plan, permutation=source, padding_mask=padding)
        assert_bytes(expected, actual)
        received = actual.clone()
        for rank, ids in enumerate(plan.rank_token_indices):
            received[rank, len(ids) :] = float("nan")
        expected = m.unpack_tokens(received, plan)
        actual = m.unpack_tokens(received, plan, permutation=inverse)
        assert_bytes(expected, actual)
        assert identities == tuple(id(x) for x in transport._permutation_for(torch.device("cpu")))


def test_permutation_transport_uses_actual_group_and_preserves_payload(monkeypatch):
    m = load_module()
    plan = m.build_vision_plan((4, 12, 20), 8)
    group = object()
    monkeypatch.setattr(m.dist, "get_rank", lambda group=None: 7)
    monkeypatch.setattr(m.dist, "get_world_size", lambda group=None: 8)
    calls = []

    def exchange(received, packed, *, group):
        calls.append((group, packed.numel() * packed.element_size()))
        received.copy_(packed)

    monkeypatch.setattr(m.dist, "all_to_all_single", exchange)
    monkeypatch.setattr(m.dist, "all_reduce", lambda *args, **kwargs: None)
    baseline = m.VisionAllToAll(plan, group=group)
    candidate = m.VisionAllToAll(plan, group=group, use_permutation=True, verify_permutation=True)
    count = len(plan.rank_token_indices[7])
    for offset in (0, 3):
        local = torch.arange(count * 16 * 3).reshape(count, 16, 3).float() + offset
        full = torch.arange(36 * 2 * 3).reshape(36, 2, 3).float() + offset
        assert_bytes(baseline.to_heads(local), candidate.to_heads(local))
        assert_bytes(baseline.to_tokens(full), candidate.to_tokens(full))
    assert baseline.calls == candidate.calls == 4
    assert baseline.sent_bytes == candidate.sent_bytes
    assert all(g is group for g, _ in calls)
    assert len(candidate._verified_layouts) == 2


def test_first_shape_check_rejects_a_corrupt_permutation(monkeypatch):
    m = load_module()
    plan = m.build_vision_plan((32,), 8)
    monkeypatch.setattr(m.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(m.dist, "get_world_size", lambda group=None: 8)
    monkeypatch.setattr(m.dist, "all_reduce", lambda *args, **kwargs: None)
    candidate = m.VisionAllToAll(plan, use_permutation=True, verify_permutation=True)
    inverse, _, _ = candidate._permutation_for(torch.device("cpu"))
    inverse[0], inverse[1] = inverse[1].clone(), inverse[0].clone()
    monkeypatch.setattr(candidate, "exchange", lambda x: torch.arange(x.numel()).reshape(x.shape).to(x.dtype))
    with pytest.raises(RuntimeError, match="changed bytes"):
        candidate.to_heads(torch.ones(4, 16, 3))


def test_permutation_flag_is_opt_in_and_rejects_unknown_values(monkeypatch):
    m = load_module()
    monkeypatch.delenv("VLLM_OMNI_H3_VISION_PERMUTE", raising=False)
    assert m.vision_permutation_enabled() is False
    for value, expected in [("0", False), ("1", True)]:
        monkeypatch.setenv("VLLM_OMNI_H3_VISION_PERMUTE", value)
        assert m.vision_permutation_enabled() is expected
    monkeypatch.setenv("VLLM_OMNI_H3_VISION_PERMUTE", "yes")
    with pytest.raises(ValueError):
        m.vision_permutation_enabled()


def test_verification_coordinates_cached_shapes_and_peer_failures(monkeypatch):
    m = load_module()
    plan = m.build_vision_plan((32,), 8)
    group = object()
    monkeypatch.setattr(m.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(m.dist, "get_world_size", lambda group=None: 8)
    candidate = m.VisionAllToAll(plan, group=group, use_permutation=True, verify_permutation=True)
    full = torch.arange(32 * 2 * 3).reshape(32, 2, 3).float()
    packed = m.pack_tokens(full, plan)
    calls = []

    def reduce(status, *, op, group):
        calls.append((op, group))
        if len(calls) == 3:
            status.fill_(1)  # Another rank found a mismatch on this call.

    monkeypatch.setattr(m.dist, "all_reduce", reduce)
    candidate._verify_layout("pack", full, packed)
    candidate._verify_layout("pack", full, packed)
    with pytest.raises(RuntimeError, match="changed bytes"):
        candidate._verify_layout("pack", full, packed)
    assert calls == [(m.dist.ReduceOp.MAX, group)] * 3
