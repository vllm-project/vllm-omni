# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU memory accounting tests for the Omni prefix cache."""

import pytest
import torch

from vllm_omni.core.prefix_cache import HIDDEN_KEY, OmniPrefixCacheManager, PrefixCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_memory_stats_account_for_hidden_and_multimodal_tensors():
    config = PrefixCacheConfig(
        num_blocks=10,
        block_size=4,
    )
    cache = OmniPrefixCacheManager(config, object(), eager=True)
    cache._pool.ensure_key(HIDDEN_KEY, torch.float32, 2)
    cache._pool.ensure_key("foo", torch.float16, 3)
    cache._pool.ensure_key("bar", torch.float32, 5)

    stats = cache.memory_stats()
    num_slots = config.num_blocks * config.block_size
    hidden_bytes = num_slots * 2 * torch.tensor([], dtype=torch.float32).element_size()
    mm_bytes = {
        "foo": num_slots * 3 * torch.tensor([], dtype=torch.float16).element_size(),
        "bar": num_slots * 5 * torch.tensor([], dtype=torch.float32).element_size(),
    }
    static_cache_bytes = hidden_bytes + sum(mm_bytes.values())

    assert stats["hidden_size"] == 2
    assert stats["hidden_dtype"] == "torch.float32"
    assert stats["hidden_states_bytes"] == hidden_bytes
    assert stats["mm_cache_bytes"] == mm_bytes
    assert stats["mm_cache_bytes_total"] == sum(mm_bytes.values())
    assert stats["static_cache_bytes"] == static_cache_bytes
    assert stats["pending_write_bytes"] == 0
    assert stats["total_cpu_bytes"] == stats["static_cache_bytes"]
    assert stats["pinned_bytes"] == (static_cache_bytes if torch.cuda.is_available() else 0)


def test_memory_stats_account_for_reusable_staging_buffers():
    config = PrefixCacheConfig(
        num_blocks=10,
        block_size=4,
        staging_depth=2,
        staging_capacity_tokens=4,
    )
    cache = OmniPrefixCacheManager(config, object(), eager=True)
    cache._controller._staging_pool.views(
        slot=0,
        key=HIDDEN_KEY,
        n=2,
        width=3,
        dtype=torch.float32,
        pin=False,
    )

    stats = cache.memory_stats()
    staging_bytes = config.staging_depth * config.staging_capacity_tokens * 3 * torch.float32.itemsize
    assert stats["static_cache_bytes"] == 0
    assert stats["pending_write_bytes"] == staging_bytes
    assert stats["total_cpu_bytes"] == staging_bytes
    assert stats["pinned_bytes"] == 0
