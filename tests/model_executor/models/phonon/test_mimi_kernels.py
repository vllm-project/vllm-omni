# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Keep the reference regression runnable without importing the GPU runtime.
_path = Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/phonon/mimi_kernels.py"
_spec = importlib.util.spec_from_file_location("phonon_mimi_kernels", _path)
_kernels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kernels)


def test_ring_wrap_and_slot_reuse():
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn(13, 2, 3, 4, generator=generator)
    values = torch.randn(13, 2, 3, 4, generator=generator)
    queries = torch.randn(13, 2, 3, 4, generator=generator)
    pool = torch.zeros(4, 5, 3, 4)
    value_pool = torch.zeros_like(pool)
    slots = torch.tensor([3, 1])
    for position in range(13):
        actual = _kernels.ring_attention_step(
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
    actual = _kernels.ring_attention_step(
        queries[0], keys[0], values[0], pool, value_pool, slots, torch.zeros(2, dtype=torch.long)
    )
    torch.testing.assert_close(actual, values[0])
    assert pool[0].count_nonzero() == 0


def test_convolution_chunking_and_history_aliasing():
    generator = torch.Generator().manual_seed(29)
    inputs = torch.randn(2, 3, 17, generator=generator)
    weight = torch.randn(4, 3, 3, generator=generator)
    bias = torch.randn(4, generator=generator)
    history = torch.zeros(4, 3, 4)
    slots = torch.tensor([2, 0])
    outputs = [
        _kernels.causal_conv_step(chunk, weight, bias, history, slots, dilation=2)
        for chunk in inputs.split([1, 3, 2, 11], dim=-1)
    ]
    scratch_history = torch.zeros_like(history)
    scratch_outputs = []
    for chunk in inputs.split([1, 3, 2, 11], dim=-1):
        scratch = torch.empty(2, 3, 4 + chunk.shape[-1])
        scratch_outputs.append(
            _kernels.causal_conv_step(
                chunk,
                weight,
                bias,
                scratch_history,
                slots,
                dilation=2,
                scratch=scratch,
            )
        )
    torch.testing.assert_close(torch.cat(scratch_outputs, -1), torch.cat(outputs, -1))
    torch.testing.assert_close(scratch_history, history)
    expected = F.conv1d(F.pad(inputs, (4, 0)), weight, bias, dilation=2)
    torch.testing.assert_close(torch.cat(outputs, -1), expected)
    torch.testing.assert_close(history[slots], inputs[..., -4:])
    assert history[1].count_nonzero() == 0


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
                _kernels.ring_attention_step(
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
        actual = _kernels.ring_attention_chunk(
            query, key, value, key_pool, value_pool, slots, torch.full((2,), position)
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(key_pool, reference_keys)
        torch.testing.assert_close(value_pool, reference_values)
        position += frames
    assert key_pool[1].count_nonzero() == 0
