# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU memory accounting tests for the Omni tensor prefix cache."""

import pytest
import torch

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_memory_stats_account_for_hidden_and_multimodal_tensors():
    cache = OmniTensorPrefixCache(
        num_blocks=10,
        block_size=4,
        hidden_size=2,
        hs_dtype=torch.float32,
    )
    cache.maybe_init_missing_mm_cache_keys(
        {
            "foo": torch.zeros(15, 3, dtype=torch.float16),
            "bar": torch.zeros(15, 5, dtype=torch.float32),
        },
        seq_len=15,
    )

    stats = cache.memory_stats()
    hidden_bytes = cache.hidden_states_cache.numel() * cache.hidden_states_cache.element_size()
    mm_bytes = {key: tensor.numel() * tensor.element_size() for key, tensor in cache.mm_outputs_cache.items()}
    all_tensors = [
        cache.hidden_states_cache,
        *cache.mm_outputs_cache.values(),
    ]
    expected_pinned_bytes = sum(tensor.numel() * tensor.element_size() for tensor in all_tensors if tensor.is_pinned())

    assert stats["hidden_states_bytes"] == hidden_bytes
    assert stats["mm_cache_bytes"] == mm_bytes
    assert stats["mm_cache_bytes_total"] == sum(mm_bytes.values())
    assert stats["static_cache_bytes"] == hidden_bytes + sum(mm_bytes.values())
    assert stats["pending_write_bytes"] == 0
    assert stats["total_cpu_bytes"] == stats["static_cache_bytes"]
    assert stats["pinned_bytes"] == expected_pinned_bytes
