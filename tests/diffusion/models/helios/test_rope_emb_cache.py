# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.helios.helios_transformer import HeliosRotaryPosEmbed

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def rope():
    return HeliosRotaryPosEmbed(rope_dim=(44, 42, 42), theta=10000.0)


def test_cached_result_is_bit_exact_with_recompute(rope):
    device = "cpu"
    idx = torch.arange(2, 11).unsqueeze(0)
    first = rope(frame_indices=idx, height=30, width=52, device=device)
    second = rope(frame_indices=idx, height=30, width=52, device=device)
    assert torch.equal(first, second)

    rope._rope_emb_cache.clear()  # force a full recompute
    third = rope(frame_indices=idx, height=30, width=52, device=device)
    assert torch.equal(first, third)


def test_cache_hit_returns_same_tensor(rope):
    idx = torch.arange(0, 9).unsqueeze(0)
    a = rope(frame_indices=idx, height=30, width=52, device="cpu")
    b = rope(frame_indices=idx, height=30, width=52, device="cpu")
    assert a is b


def test_lru_keeps_hot_history_keys(rope):
    """History entries are hit every denoise step and must not be evicted
    by the stream of one-off per-chunk entries."""
    history = [
        torch.arange(0, 2).unsqueeze(0),
        torch.arange(0, 16).unsqueeze(0),
    ]
    for k in range(30):
        rope(frame_indices=torch.arange(100 + k, 105 + k).unsqueeze(0), height=30, width=52, device="cpu")
        for h in history:
            rope(frame_indices=h, height=30, width=52, device="cpu")
    keys = [indices for indices, *_ in rope._rope_emb_cache]
    assert ((0, 1),) in keys
    assert (tuple(range(16)),) in keys
    assert len(keys) == 16


def test_different_inputs_get_distinct_entries(rope):
    a = rope(frame_indices=torch.arange(0, 9).unsqueeze(0), height=30, width=52, device="cpu")
    b = rope(frame_indices=torch.arange(1, 10).unsqueeze(0), height=30, width=52, device="cpu")
    c = rope(frame_indices=torch.arange(0, 9).unsqueeze(0), height=32, width=52, device="cpu")
    assert not torch.equal(a, b)
    assert not torch.equal(a, c)


def test_equal_flat_indices_with_different_batch_shapes_do_not_collide(rope):
    indices = torch.arange(4)
    single = rope(indices.reshape(1, 4), 2, 2, "cpu")
    batched = rope(indices.reshape(2, 2), 2, 2, "cpu")
    assert single.shape[0] == 1
    assert batched.shape[0] == 2


def test_module_move_clears_cached_embeddings(rope):
    indices = torch.arange(2).unsqueeze(0)
    first = rope(indices, 2, 2, "cpu")
    rope.to("cpu")
    assert rope(indices, 2, 2, "cpu") is not first
