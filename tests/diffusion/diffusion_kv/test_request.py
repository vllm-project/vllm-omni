# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import vllm.v1.core.single_type_kv_cache_manager as native_kv_managers
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.request import RequestStatus

from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVContext, DiffusionKVRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

BLOCK_SIZE = 4


def _request(**overrides) -> DiffusionKVRequest:
    values = dict(
        request_id="req/diffusion-kv/0",
        sequence_id=0,
        prefix_len=4,
        target_len=4,
        seq_len=8,
    )
    values.update(overrides)
    return DiffusionKVRequest(**values)


def test_request_exposes_native_request_and_diffusion_semantics() -> None:
    request = _request()

    assert request.sequence_id == 0
    assert request.prefix_len == 4
    assert request.target_len == 4
    assert request.num_tokens == request.seq_len == 8
    assert request.num_prompt_tokens == 4
    assert request.num_computed_tokens == 0
    assert request.block_hashes == []
    assert request.kv_contexts == ()
    assert request.skip_reading_prefix_cache is True
    assert request.shared_prefix_boundary == 0
    assert request.status is RequestStatus.WAITING


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sequence_id": -1}, "sequence_id"),
        ({"prefix_len": -1}, "prefix_len"),
        ({"target_len": 0}, "target_len"),
        ({"seq_len": 0}, "seq_len"),
        ({"prefix_len": 5, "target_len": 4, "seq_len": 8}, "must not exceed"),
    ],
)
def test_request_rejects_invalid_lengths(overrides, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _request(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"context_id": ""}, "context_id"),
        ({"cache_role": ""}, "cache_role"),
        ({"num_tokens": 0}, "num_tokens"),
    ],
)
def test_context_rejects_invalid_identity_or_length(overrides, message: str) -> None:
    values = dict(context_id="text", cache_role="cross.text", num_tokens=8)
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        DiffusionKVContext(**values)


def test_request_owns_independent_kv_contexts() -> None:
    text = DiffusionKVContext(context_id="text", cache_role="cross.text", num_tokens=8)
    image = DiffusionKVContext(context_id="image", cache_role="cross.image", num_tokens=4)

    request = _request(kv_contexts=(text, image))

    assert request.kv_contexts == (text, image)


def test_request_rejects_duplicate_context_ids() -> None:
    contexts = (
        DiffusionKVContext(context_id="condition", cache_role="cross.text", num_tokens=8),
        DiffusionKVContext(context_id="condition", cache_role="cross.image", num_tokens=4),
    )

    with pytest.raises(ValueError, match="unique context_id"):
        _request(kv_contexts=contexts)


def test_request_rejects_non_context_values() -> None:
    with pytest.raises(TypeError, match="DiffusionKVContext"):
        _request(kv_contexts=(object(),))


def _manager(*, enable_caching: bool) -> KVCacheManager:
    # vLLM 0.26 lazily registers its built-in KV specs. Another test may have
    # already registered an out-of-tree spec, making the registry non-empty
    # before FullAttentionSpec is registered. Initialize the native mappings
    # explicitly; vLLM 0.25 has no registration hook and needs no setup.
    register_all = getattr(native_kv_managers, "register_all_kvcache_specs", None)
    if register_all is not None:
        register_all(None)

    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
    )
    group = KVCacheGroupSpec(layer_names=["layer0"], kv_cache_spec=spec)
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes * 8, shared_by=["layer0"])],
        kv_cache_groups=[group],
    )
    return KVCacheManager(
        config,
        max_model_len=64,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=enable_caching,
    )


def test_empty_hash_request_conforms_to_native_request_local_allocation() -> None:
    manager = _manager(enable_caching=True)
    free_before = manager.block_pool.get_num_free_blocks()
    request = _request()

    # vLLM 0.26 adds a shared-prefix-boundary return value.
    computed_blocks, num_computed_tokens, *_ = manager.get_computed_blocks(request)
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        request,
        num_new_tokens=request.num_tokens,
        new_computed_blocks=computed_blocks,
        delay_cache_blocks=True,
        full_sequence_must_fit=True,
    )

    assert blocks is not None
    assert len(manager.get_block_ids(request.request_id)[0]) == 2
    # Empty hashes intentionally mean request-local residency only. Calling
    # native cache_blocks() would require a canonical hash for the full prefix
    # block and is therefore outside this request's contract.
    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == free_before


def test_hashed_request_conforms_to_native_delayed_prefix_commit() -> None:
    manager = _manager(enable_caching=True)
    prefix_hash = BlockHash(b"prefix".ljust(32, b"\0"))
    target_hash = BlockHash(b"target".ljust(32, b"\0"))
    request = _request(block_hashes=[prefix_hash, target_hash])

    computed_blocks, num_computed_tokens, *_ = manager.get_computed_blocks(request)
    assert num_computed_tokens == 0
    blocks = manager.allocate_slots(
        request,
        num_new_tokens=request.num_tokens,
        new_computed_blocks=computed_blocks,
        delay_cache_blocks=True,
        full_sequence_must_fit=True,
    )

    assert blocks is not None
    manager.cache_blocks(request, request.prefix_len)
    manager.free(request)

    cached_request = _request(
        request_id="req/diffusion-kv/cached",
        block_hashes=[prefix_hash, target_hash],
    )
    cached_blocks, num_computed_tokens, *_ = manager.get_computed_blocks(cached_request)

    # Only the stable prefix was published. The dynamic target remains a miss.
    assert num_computed_tokens == cached_request.prefix_len
    assert len(cached_blocks.blocks[0]) == 1
    cached_allocation = manager.allocate_slots(
        cached_request,
        num_new_tokens=cached_request.num_tokens - num_computed_tokens,
        num_new_computed_tokens=num_computed_tokens,
        new_computed_blocks=cached_blocks,
        delay_cache_blocks=True,
        full_sequence_must_fit=True,
    )
    assert cached_allocation is not None
    manager.free(cached_request)
