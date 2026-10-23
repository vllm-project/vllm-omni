# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The decode allocate_slots fast path returns exactly what the full path returns."""

import pytest
import torch
import vllm.v1.core.single_type_kv_cache_manager as native_kv_managers
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.fast_allocate import install_decode_allocate_fast_path

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BLOCK = 16


def _manager(num_blocks=40, enable_caching=False, groups=1):
    # vLLM lazily registers built-in KV specs. Another test may already have
    # registered an out-of-tree spec, so the registry can be non-empty before
    # FullAttentionSpec has a manager. Initialize the native mappings first.
    native_kv_managers.register_all_kvcache_specs(None)
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


@pytest.mark.parametrize("skip_zeroing_group_ids", [(), (0,)])
def test_fast_path_matches_full_allocation_through_block_boundaries(skip_zeroing_group_ids):
    ref, fast = _manager(), _manager()
    assert install_decode_allocate_fast_path(fast)
    reqs = {name: (_request("a", 13), _request("a", 13)) for name in ("a",)}
    reqs["b"] = (_request("b", 30), _request("b", 30))
    for r_ref, r_fast in reqs.values():
        assert _ids(ref.allocate_slots(r_ref, r_ref.num_tokens, skip_zeroing_group_ids=skip_zeroing_group_ids)) == _ids(
            fast.allocate_slots(r_fast, r_fast.num_tokens, skip_zeroing_group_ids=skip_zeroing_group_ids)
        )
        for r in (r_ref, r_fast):
            r.status = RequestStatus.RUNNING
            r.num_computed_tokens = r.num_tokens
    for step in range(60):
        for r_ref, r_fast in reqs.values():
            lookahead = 3 if step % 7 == 0 else 0
            a = ref.allocate_slots(
                r_ref, 1, num_lookahead_tokens=lookahead, skip_zeroing_group_ids=skip_zeroing_group_ids
            )
            b = fast.allocate_slots(
                r_fast, 1, num_lookahead_tokens=lookahead, skip_zeroing_group_ids=skip_zeroing_group_ids
            )
            assert _ids(a) == _ids(b), step
            for r in (r_ref, r_fast):
                r.num_computed_tokens += 1
            assert ref.block_pool.get_num_free_blocks() == fast.block_pool.get_num_free_blocks()
    # Reservation larger than the free pool refuses even a request that needs no block.
    r_ref, r_fast = reqs["a"]
    reserve = ref.block_pool.get_num_free_blocks() + 1
    assert ref.allocate_slots(r_ref, 1, reserved_blocks=reserve) is None
    assert fast.allocate_slots(r_fast, 1, reserved_blocks=reserve) is None


def test_foreign_registry_still_builds_full_attention_manager():
    """A prior out-of-tree spec must not hide FullAttentionSpec from ``_manager``.

    ``KVCacheSpecRegistry._ensure_registered`` returns once the registry is
    non-empty, which is the weekly ``No manager registered for KVCacheSpec
    <...FullAttentionSpec>`` failure. ``_manager`` calls
    ``register_all_kvcache_specs`` anyway, so the same construction succeeds.
    The process registry is restored afterwards.
    """
    from dataclasses import dataclass

    import vllm.v1.kv_cache_spec_registry as kv_spec_registry
    from vllm.v1.kv_cache_interface import KVCacheSpec

    @dataclass(frozen=True)
    class _ForeignKVCacheSpec(KVCacheSpec):
        """Occupies the registry without registering a native attention spec."""

    class _ForeignKVManager:
        """Never constructed. Its registration only makes the registry non-empty."""

    spec_registry = kv_spec_registry._REGISTRY_KVCACHESPEC_LIST
    role_registry = kv_spec_registry._REGISTRY_ROLE_MANAGERS
    saved_specs = dict(spec_registry)
    saved_roles = dict(role_registry)
    spec = FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float16)
    config = KVCacheConfig(
        num_blocks=40,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer0"], spec)],
    )

    def construct_without_native_registration():
        return KVCacheManager(
            config,
            max_model_len=256,
            scheduler_block_size=BLOCK,
            hash_block_size=BLOCK,
            enable_caching=False,
        )

    try:
        spec_registry.clear()
        kv_spec_registry.KVCacheSpecRegistry.register(_ForeignKVCacheSpec, _ForeignKVManager)
        kv_spec_registry.KVCacheSpecRegistry._ensure_registered(None)
        assert FullAttentionSpec not in spec_registry

        with pytest.raises(
            AssertionError,
            match=r"No manager registered for KVCacheSpec <class '.*FullAttentionSpec'>",
        ):
            construct_without_native_registration()

        manager = _manager()
        assert manager.coordinator.single_type_managers
        assert FullAttentionSpec in spec_registry
    finally:
        spec_registry.clear()
        spec_registry.update(saved_specs)
        role_registry.clear()
        role_registry.update(saved_roles)


def test_fast_path_only_installs_where_it_is_exact():
    assert not install_decode_allocate_fast_path(_manager(enable_caching=True))
    assert not install_decode_allocate_fast_path(_manager(groups=2))


@pytest.mark.parametrize("skip_zeroing_group_ids", [(), (0,)])
def test_external_load_blocks_preserve_zeroing_policy(skip_zeroing_group_ids):
    manager = _manager()
    assert install_decode_allocate_fast_path(manager)
    group = manager.coordinator.single_type_managers[0]
    # Enable allocation recording to observe the upstream async-load policy.
    group._record_new_block_ids = True
    request = _request("external", 32)
    blocks = manager.allocate_slots(
        request,
        0,
        num_external_computed_tokens=32,
        delay_cache_blocks=True,
        skip_zeroing_group_ids=skip_zeroing_group_ids,
    )
    assert blocks is not None
    external_ids = [block.block_id for block in group.req_to_blocks[request.request_id]]
    assert len(external_ids) == 2
    assert group.take_new_block_ids() == ([] if skip_zeroing_group_ids else external_ids)
