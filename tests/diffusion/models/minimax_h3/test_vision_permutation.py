# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The SP8 transport permutation must preserve exact bytes and layouts."""

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


def reference_pack_tokens(full, plan):
    capacity = max(1, max(map(len, plan.rank_token_indices)))
    output = full.new_zeros((len(plan.rank_token_indices), capacity, *full.shape[1:]))
    for rank, indices in enumerate(plan.rank_token_indices):
        ids = torch.tensor(indices, dtype=torch.long, device=full.device)
        output[rank, : len(indices)] = full.index_select(0, ids)
    return output


def reference_unpack_tokens(received, plan):
    output = received.new_empty((sum(plan.segment_lengths), *received.shape[2:]))
    for rank, indices in enumerate(plan.rank_token_indices):
        ids = torch.tensor(indices, dtype=torch.long, device=received.device)
        output.index_copy_(0, ids, received[rank, : len(indices)])
    return output


@pytest.mark.parametrize("size", [2, 4, 8, 16])
@pytest.mark.parametrize("lengths", [(4,), (4, 12, 0, 20), (0, 0), (8160, 8160, 880, 660, 660, 7656)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_permutation_matches_reference_and_caches_only_metadata(monkeypatch, size, lengths, dtype):
    m = load_module()
    plan = m.build_vision_plan(lengths, size)
    monkeypatch.setattr(m.dist, "get_rank", lambda group=None: size - 1)
    monkeypatch.setattr(m.dist, "get_world_size", lambda group=None: size)
    transport = m.VisionAllToAll(plan)
    inverse, source, padding = transport._permutation_for(torch.device("cpu"))
    identities = tuple(id(x) for x in (inverse, source, padding))
    total = sum(lengths)
    for offset in (0, 7):
        full = (torch.arange(total * 6).reshape(total, 3, 2).transpose(1, 2) + offset).to(dtype)
        if total:
            full[0, 0] = torch.tensor([-0.0, float("inf"), float("nan")], dtype=dtype)
        expected = reference_pack_tokens(full, plan)
        actual = m.pack_tokens(full, plan, source, padding_mask=padding)
        assert_bytes(expected, actual)
        received = actual.clone()
        for rank, ids in enumerate(plan.rank_token_indices):
            received[rank, len(ids) :] = float("nan")
        expected = reference_unpack_tokens(received, plan)
        actual = m.unpack_tokens(received, inverse)
        assert_bytes(expected, actual)
        assert identities == tuple(id(x) for x in transport._permutation_for(torch.device("cpu")))


def test_transport_uses_actual_group_and_preserves_payload(monkeypatch):
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
    transport = m.VisionAllToAll(plan, group=group)
    count = len(plan.rank_token_indices[7])
    for offset in (0, 3):
        local = torch.arange(count * 16 * 3).reshape(count, 16, 3).float() + offset
        full = torch.arange(36 * 2 * 3).reshape(36, 2, 3).float() + offset
        expected_heads = reference_unpack_tokens(m.pack_heads(local, 8, transport.capacity), plan)
        expected_tokens = m.unpack_heads(reference_pack_tokens(full, plan), count)
        assert_bytes(expected_heads, transport.to_heads(local))
        assert_bytes(expected_tokens, transport.to_tokens(full))
    assert transport.calls == 4
    assert transport.sent_bytes == sum(size for _, size in calls)
    assert all(seen_group is group for seen_group, _ in calls)
