# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Token index reuse must preserve the existing transport layouts exactly."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("size", [2, 4, 8, 16])
@pytest.mark.parametrize("lengths", [(4, 12, 0, 20), (4,)])
def test_transport_reuses_only_indices_and_preserves_values(monkeypatch, size, lengths):
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_parallel.py"
    spec = importlib.util.spec_from_file_location("vision_cached_transport_test", path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    plan = m.build_vision_plan(lengths, size)
    rank = size - 1  # Includes completely empty local shards.
    monkeypatch.setattr(m.dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(m.dist, "get_world_size", lambda group=None: size)
    transport = m.VisionAllToAll(plan)
    monkeypatch.setattr(transport, "exchange", lambda packed: packed)
    count = len(plan.rank_token_indices[rank])
    local = torch.arange(count * 16 * 2).reshape(count, 16, 2).float()
    full = torch.arange(sum(lengths) * (16 // size) * 2).reshape(sum(lengths), 16 // size, 2).float()
    expected_heads = m.unpack_tokens(m.pack_heads(local, size, transport.capacity), plan)
    expected_tokens = m.unpack_heads(m.pack_tokens(full, plan), count)
    original_tensor = torch.tensor
    constructions = []

    def counted_tensor(*args, **kwargs):
        constructions.append(args[0])
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(m.torch, "tensor", counted_tensor)
    for _ in range(2):
        # Different data prevents accidentally caching activations or results.
        for offset in (0, 1, 2):
            actual = transport.to_heads(local + offset)
            expected = expected_heads.clone()
            for source, ids in enumerate(plan.rank_token_indices):
                expected[list(ids)] += offset if count else 0
                if len(ids) > count:
                    expected[list(ids)[count:]] -= offset if count else 0
            assert torch.equal(actual, expected)
            assert actual.stride() == expected_heads.stride()
        assert torch.equal(transport.to_tokens(full), expected_tokens)
    assert len(constructions) == size
