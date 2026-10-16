# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The decode allocate_slots fast path returns exactly what the full path returns."""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.fast_allocate import install_decode_allocate_fast_path

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BLOCK = 16


def _manager(num_blocks=40, enable_caching=False, groups=1):
    spec = FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float16)
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"layer{g}"], spec) for g in range(groups)],
    )
    return KVCacheManager(
        config, max_model_len=256, scheduler_block_size=BLOCK, hash_block_size=BLOCK, enable_caching=enable_caching
    )


def _request(rid, prompt_len):
    req = Request(rid, list(range(prompt_len)), SamplingParams(max_tokens=200), None)
    return req


def _ids(blocks):
    return None if blocks is None else blocks.get_block_ids()


def test_fast_path_matches_full_allocation_through_block_boundaries():
    ref, fast = _manager(), _manager()
    assert install_decode_allocate_fast_path(fast)
    reqs = {name: (_request("a", 13), _request("a", 13)) for name in ("a",)}
    reqs["b"] = (_request("b", 30), _request("b", 30))
    for r_ref, r_fast in reqs.values():
        assert _ids(ref.allocate_slots(r_ref, r_ref.num_tokens)) == _ids(fast.allocate_slots(r_fast, r_fast.num_tokens))
        for r in (r_ref, r_fast):
            r.status = RequestStatus.RUNNING
            r.num_computed_tokens = r.num_tokens
    for step in range(60):
        for r_ref, r_fast in reqs.values():
            lookahead = 3 if step % 7 == 0 else 0
            a = ref.allocate_slots(r_ref, 1, num_lookahead_tokens=lookahead)
            b = fast.allocate_slots(r_fast, 1, num_lookahead_tokens=lookahead)
            assert _ids(a) == _ids(b), step
            for r in (r_ref, r_fast):
                r.num_computed_tokens += 1
            assert ref.block_pool.get_num_free_blocks() == fast.block_pool.get_num_free_blocks()
    # Reservation larger than the free pool refuses even a request that needs no block.
    r_ref, r_fast = reqs["a"]
    reserve = ref.block_pool.get_num_free_blocks() + 1
    assert ref.allocate_slots(r_ref, 1, reserved_blocks=reserve) is None
    assert fast.allocate_slots(r_fast, 1, reserved_blocks=reserve) is None


def test_fast_path_only_installs_where_it_is_exact():
    assert not install_decode_allocate_fast_path(_manager(enable_caching=True))
    assert not install_decode_allocate_fast_path(_manager(groups=2))
