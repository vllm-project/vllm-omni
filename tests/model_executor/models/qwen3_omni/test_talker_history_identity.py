# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local Talker generated blocks use owner identity, not RVQ history."""

from types import SimpleNamespace

import pytest
import torch
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_block_hash_extra_keys,
    get_request_block_hasher,
    hash_block_tokens,
    init_none_hash,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import RequestStatus

from vllm_omni.core.prefix_cache.adapter import PrefixCacheRequestOwner
from vllm_omni.core.sched.input_finalization import install_request_input, prepare_request_input
from vllm_omni.model_executor.models.qwen3_omni.talker_history import get_talker_request_block_hasher
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def initialize_hash():
    init_none_hash(sha256)


def _request(req_id="a", prompt_len=4, hash_size=4, *, owner=None, salt="conditioning", **kwargs):
    request = OmniRequest(
        request_id=req_id,
        prompt_token_ids=list(range(100, 100 + prompt_len)),
        sampling_params=SamplingParams(max_tokens=16, ignore_eos=True),
        pooling_params=None,
        block_hasher=get_talker_request_block_hasher(hash_size, sha256),
        cache_salt=salt,
        **kwargs,
    )
    request._omni_prefix_cache_owner = owner or PrefixCacheRequestOwner(1)
    return request


def _append_primary(request, token=7, count=1):
    for _ in range(count):
        request.append_output_token_ids(token)


def _manager(hash_size=4):
    register_all_kvcache_specs(SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="none")))
    spec = FullAttentionSpec(block_size=hash_size, num_kv_heads=1, head_size=1, dtype=torch.float32)
    config = KVCacheConfig(num_blocks=24, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)])
    return KVCacheManager(config, max_model_len=64, scheduler_block_size=hash_size, hash_block_size=hash_size)


@pytest.mark.parametrize("prompt_len", range(1, 13))
def test_pure_prefill_preserves_native_salt_and_lora_hashes(prompt_len):
    request = _request(prompt_len=prompt_len, lora_request=LoRARequest("adapter", 1, "/immutable/adapter"))
    original = list(request.block_hashes)
    request.block_hashes = []
    assert get_request_block_hasher(4, sha256)(request) == original


@pytest.mark.parametrize("hash_size", [2, 4])
def test_native_kv_publication_isolates_generated_blocks_by_owner(hash_size):
    manager = _manager(hash_size)
    producer = _request(hash_size=hash_size, owner=PrefixCacheRequestOwner(1))
    consumer = _request("different-request-id", hash_size=hash_size, owner=PrefixCacheRequestOwner(987))
    _append_primary(producer, count=hash_size)
    _append_primary(consumer, count=hash_size)

    assert list(producer.all_token_ids) == list(consumer.all_token_ids)
    assert manager.allocate_slots(producer, num_new_tokens=2 * hash_size) is not None
    assert manager.get_computed_blocks(consumer)[1] == producer.num_prompt_tokens


@pytest.mark.parametrize("hash_size", [2, 4])
def test_same_owner_generated_resume_hits_even_with_private_request_id(hash_size):
    manager = _manager(hash_size)
    owner = PrefixCacheRequestOwner(44, 3)
    producer = _request(hash_size=hash_size, owner=owner)
    replay = _request("different-request-id", hash_size=hash_size, owner=owner)
    _append_primary(producer, count=hash_size + 1)
    _append_primary(replay, count=hash_size + 1)

    assert list(producer.all_token_ids) == list(replay.all_token_ids)
    assert manager.allocate_slots(producer, num_new_tokens=producer.num_tokens) is not None
    expected_hit = (producer.num_tokens - 1) // hash_size * hash_size
    assert manager.get_computed_blocks(replay)[1] == expected_hit


def test_mixed_block_uses_native_parent_extras_and_owner_identity():
    request = _request(prompt_len=3, lora_request=LoRARequest("adapter", 1, "/immutable/adapter"))
    _append_primary(request, count=5)

    parent = None
    expected = []
    start = 0
    for end in (4, 8):
        extras, _ = generate_block_hash_extra_keys(request, start, end, 0 if start == 0 else -1)
        extras = (extras or ()) + (
            (
                "qwen3-omni.generated-owner.v1",
                (
                    request._omni_prefix_cache_owner.admission_id,
                    request._omni_prefix_cache_owner.generation,
                ),
            ),
        )
        parent = hash_block_tokens(sha256, parent, request.all_token_ids[start:end], extras)
        expected.append(parent)
        start = end
    assert request.block_hashes == expected


def test_generated_hash_does_not_depend_on_scheduler_rvq_history_storage():
    owner = PrefixCacheRequestOwner(12, 2)
    producer = _request(prompt_len=4, owner=owner)
    replay = _request("replay", prompt_len=4, owner=owner)
    _append_primary(producer, count=4)
    _append_primary(replay, count=4)
    assert not hasattr(producer, "talker_codec_inputs")
    assert producer.block_hashes == replay.block_hashes


def test_reused_request_id_new_admission_isolates_generated_blocks_even_for_same_primary():
    first = _request("same-id", owner=PrefixCacheRequestOwner(1))
    second = _request("same-id", owner=PrefixCacheRequestOwner(2))
    _append_primary(first, count=4)
    _append_primary(second, count=4)
    assert len(first.block_hashes) == len(second.block_hashes) == 2
    assert first.block_hashes[0] == second.block_hashes[0]
    assert first.block_hashes[1] != second.block_hashes[1]


def test_missing_owner_cannot_produce_a_generated_hash():
    request = _request(prompt_len=3)
    delattr(request, "_omni_prefix_cache_owner")
    with pytest.raises(ValueError, match="scheduler-owned"):
        request.append_output_token_ids(7)
    assert not request.block_hashes


def test_replacement_changes_generated_owner_without_old_history_transport():
    request = _request(prompt_len=4, owner=PrefixCacheRequestOwner(1))
    _append_primary(request, count=4)
    old_hashes = list(request.block_hashes)
    candidate = prepare_request_input(
        request,
        prompt_token_ids=[10, 11, 12, 13],
        mm_features=[],
        cache_salt="new",
        sampling_params=request.sampling_params,
    )
    install_request_input(request, candidate)
    request._omni_prefix_cache_owner = PrefixCacheRequestOwner(1, 1)
    assert not hasattr(request, "talker_codec_inputs")
    _append_primary(request, count=4)
    assert request.block_hashes != old_hashes


@pytest.mark.parametrize("scheduler_cls", [Scheduler, AsyncScheduler])
def test_native_output_update_hashes_owner_before_publication(scheduler_cls):
    manager = _manager()
    request = _request()
    request.status = RequestStatus.RUNNING
    assert manager.allocate_slots(request, num_new_tokens=4) is not None
    request.num_computed_tokens = 4
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.max_model_len = 64
    scheduler.kv_cache_manager = manager

    for _ in range(4):
        request.num_output_placeholders = int(scheduler_cls is AsyncScheduler)
        assert scheduler._update_request_with_output(request, [7]) == ([7], False)
    assert len(request.block_hashes) == 2
