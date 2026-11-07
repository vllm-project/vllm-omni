# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Keep the attention reference regression runnable without importing Triton.
_path = Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/phonon/mimi_attention.py"
_spec = importlib.util.spec_from_file_location("phonon_mimi_attention", _path)
_attention = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_attention)


def ring_attention_step(query, key, value, key_pool, value_pool, slots, positions):
    """Simple incremental reference used to validate chunked ring attention."""
    window = key_pool.shape[1]
    write_indices = positions.remainder(window)
    key_pool[slots, write_indices] = key
    value_pool[slots, write_indices] = value
    physical_indices = torch.arange(window, device=positions.device)
    ages = (positions[:, None] - physical_indices).remainder(window)
    valid = ages <= positions[:, None]
    keys = key_pool[slots].transpose(1, 2)
    values = value_pool[slots].transpose(1, 2)
    scores = (query.unsqueeze(2) * keys).sum(-1) * query.shape[-1] ** -0.5
    scores = scores.masked_fill(~valid[:, None], float("-inf"))
    probabilities = scores.float().softmax(-1).to(values.dtype)
    return (probabilities.unsqueeze(-1) * values).sum(2)


def test_ring_wrap_and_slot_reuse():
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn(13, 2, 3, 4, generator=generator)
    values = torch.randn(13, 2, 3, 4, generator=generator)
    queries = torch.randn(13, 2, 3, 4, generator=generator)
    pool = torch.zeros(4, 5, 3, 4)
    value_pool = torch.zeros_like(pool)
    slots = torch.tensor([3, 1])
    for position in range(13):
        actual = ring_attention_step(
            queries[position],
            keys[position],
            values[position],
            pool,
            value_pool,
            slots,
            torch.full((2,), position),
        )
        start = max(0, position - 4)
        expected = F.scaled_dot_product_attention(
            queries[position].unsqueeze(2),
            keys[start : position + 1].permute(1, 2, 0, 3),
            values[start : position + 1].permute(1, 2, 0, 3),
        ).squeeze(2)
        torch.testing.assert_close(actual, expected)
    # A reused slot at position zero must mask every old ring entry.
    actual = ring_attention_step(
        queries[0], keys[0], values[0], pool, value_pool, slots, torch.zeros(2, dtype=torch.long)
    )
    torch.testing.assert_close(actual, values[0])
    assert pool[0].count_nonzero() == 0


def test_chunk_attention_matches_incremental_ring_and_wrap():
    generator = torch.Generator().manual_seed(71)
    slots = torch.tensor([2, 0])
    key_pool = torch.zeros(3, 5, 2, 4)
    value_pool = torch.zeros_like(key_pool)
    reference_keys = key_pool.clone()
    reference_values = value_pool.clone()
    position = 0
    for frames in (1, 3, 8, 2):
        query, key, value = [torch.randn(2, frames, 2, 4, generator=generator) for _ in range(3)]
        expected = torch.stack(
            [
                ring_attention_step(
                    query[:, frame],
                    key[:, frame],
                    value[:, frame],
                    reference_keys,
                    reference_values,
                    slots,
                    torch.full((2,), position + frame),
                )
                for frame in range(frames)
            ],
            1,
        )
        actual = _attention.ring_attention_chunk(
            query, key, value, key_pool, value_pool, slots, torch.full((2,), position)
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(key_pool, reference_keys)
        torch.testing.assert_close(value_pool, reference_values)
        position += frames
    assert key_pool[1].count_nonzero() == 0


@pytest.mark.parametrize("capacity", [8, 9])
def test_ring_headroom_preserves_window_across_chunks_and_wrap(capacity):
    generator = torch.Generator().manual_seed(97)
    slots = torch.tensor([2, 0])
    keys, values = torch.zeros(3, capacity, 2, 4), torch.zeros(3, capacity, 2, 4)
    reference_keys, reference_values = torch.zeros(3, 5, 2, 4), torch.zeros(3, 5, 2, 4)
    position = 0
    for frames in (3, 4, 4, 9, 1):
        q, k, v = [torch.randn(2, frames, 2, 4, generator=generator) for _ in range(3)]
        expected = torch.stack(
            [
                ring_attention_step(
                    q[:, t], k[:, t], v[:, t], reference_keys, reference_values, slots, torch.full((2,), position + t)
                )
                for t in range(frames)
            ],
            1,
        )
        actual = _attention.ring_attention_chunk(q, k, v, keys, values, slots, torch.full((2,), position), window=5)
        torch.testing.assert_close(actual, expected)
        position += frames
        absolute = torch.arange(max(0, position - 5), position)
        torch.testing.assert_close(
            keys[slots[:, None], absolute % capacity], reference_keys[slots[:, None], absolute % 5]
        )
        torch.testing.assert_close(
            values[slots[:, None], absolute % capacity], reference_values[slots[:, None], absolute % 5]
        )
    assert keys[1].count_nonzero() == 0
