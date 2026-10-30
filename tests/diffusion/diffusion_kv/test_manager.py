# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import vllm.v1.core.single_type_kv_cache_manager as native_kv_managers
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from tests.helpers.kv_layout import build_kv_cache_tensor
from vllm_omni.diffusion.diffusion_kv.manager import DiffusionKVAdmissionError, DiffusionKVCacheManager
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVContext, DiffusionKVRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

BLOCK_SIZE = 4


def _config(num_blocks: int) -> KVCacheConfig:
    native_kv_managers.register_all_kvcache_specs(None)
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[build_kv_cache_tensor(spec, num_blocks, ["layer0"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer0"], kv_cache_spec=spec)],
    )


def _request(
    public_id: str,
    sequence_id: int,
    *,
    prefix_len: int = 4,
    target_len: int = 4,
    seq_len: int = 8,
    cache_token_ids=(),
    mm_features=(),
    kv_contexts=(),
) -> DiffusionKVRequest:
    return DiffusionKVRequest(
        f"{public_id}/diffusion-kv/{sequence_id}",
        sequence_id=sequence_id,
        prefix_len=prefix_len,
        target_len=target_len,
        seq_len=seq_len,
        cache_token_ids=cache_token_ids,
        mm_features=mm_features,
        kv_contexts=kv_contexts,
    )


def _manager(
    num_blocks: int,
    *,
    max_model_len: int = 64,
    enable_prefix_caching: bool = False,
) -> DiffusionKVCacheManager:
    return DiffusionKVCacheManager(
        _config(num_blocks),
        max_model_len=max_model_len,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_prefix_caching=enable_prefix_caching,
    )


def test_successful_request_publishes_prefix_for_a_warm_hit() -> None:
    manager = _manager(8, enable_prefix_caching=True)
    cold = _request("cold", 0, cache_token_ids=range(4))

    cold_metadata = manager.reserve_request("cold", (cold,))
    assert cold_metadata is not None
    assert cold_metadata.sequences[0].cached_prefix_len == 0
    cold_prefix_block = cold_metadata.sequences[0].block_ids[0][0]
    manager.publish_request("cold")
    manager.free_request("cold")

    warm = _request("warm", 0, cache_token_ids=range(4))
    warm_metadata = manager.reserve_request("warm", (warm,))

    assert warm_metadata is not None
    assert warm_metadata.sequences[0].cached_prefix_len == 4
    assert warm_metadata.sequences[0].block_ids[0][0] == cold_prefix_block
    # Metadata contains both the shared hit and the newly allocated target.
    assert len(warm_metadata.sequences[0].block_ids[0]) == 2

    manager.free_request("warm")
    assert warm.num_computed_tokens == 0
    assert warm.shared_prefix_boundary == 0


def test_prefix_hash_root_is_stable_across_manager_instances() -> None:
    first_manager = _manager(8, enable_prefix_caching=True)
    second_manager = _manager(8, enable_prefix_caching=True)
    first = _request("first", 0, cache_token_ids=range(4))
    second = _request("second", 0, cache_token_ids=range(4))

    first_manager._prepare_block_hashes((first,))
    second_manager._prepare_block_hashes((second,))

    assert first.block_hashes == second.block_hashes


def test_warm_hit_fits_at_the_same_capacity_as_the_cold_request() -> None:
    manager = _manager(3, enable_prefix_caching=True)
    cold = _request("cold", 0, cache_token_ids=range(4))
    assert manager.reserve_request("cold", (cold,)) is not None
    manager.publish_request("cold")
    manager.free_request("cold")

    warm = _request("warm", 0, cache_token_ids=range(4))
    metadata = manager.reserve_request("warm", (warm,))

    assert metadata is not None
    assert metadata.sequences[0].cached_prefix_len == 4


def test_unpublished_request_does_not_become_a_prefix_hit() -> None:
    manager = _manager(8, enable_prefix_caching=True)
    failed = _request("failed", 0, cache_token_ids=range(4))
    assert manager.reserve_request("failed", (failed,)) is not None
    manager.free_request("failed")

    retry = _request("retry", 0, cache_token_ids=range(4))
    metadata = manager.reserve_request("retry", (retry,))

    assert metadata is not None
    assert metadata.sequences[0].cached_prefix_len == 0


def test_partial_prefix_hit_recomputes_from_the_first_changed_block() -> None:
    manager = _manager(10, enable_prefix_caching=True)
    cold = _request(
        "cold",
        0,
        prefix_len=8,
        seq_len=12,
        cache_token_ids=range(8),
    )
    assert manager.reserve_request("cold", (cold,)) is not None
    manager.publish_request("cold")
    manager.free_request("cold")

    warm = _request(
        "warm",
        0,
        prefix_len=8,
        seq_len=12,
        cache_token_ids=(*range(4), 100, 101, 102, 103),
    )
    metadata = manager.reserve_request("warm", (warm,))

    assert metadata is not None
    assert metadata.sequences[0].cached_prefix_len == 4
    assert len(metadata.sequences[0].block_ids[0]) == 3


def test_cfg_rows_use_the_minimum_warm_hit_as_one_execution_boundary() -> None:
    manager = _manager(16, enable_prefix_caching=True)
    for public_id, tokens in (("seed-a", range(8)), ("seed-b", (*range(4), 20, 21, 22, 23))):
        request = _request(public_id, 0, prefix_len=8, seq_len=12, cache_token_ids=tokens)
        assert manager.reserve_request(public_id, (request,)) is not None
        manager.publish_request(public_id)
        manager.free_request(public_id)

    requests = (
        _request("warm", 0, prefix_len=8, seq_len=12, cache_token_ids=range(8)),
        _request(
            "warm",
            1,
            prefix_len=8,
            seq_len=12,
            cache_token_ids=(*range(4), 30, 31, 32, 33),
        ),
    )
    metadata = manager.reserve_request("warm", requests)

    assert metadata is not None
    assert [sequence.cached_prefix_len for sequence in metadata.sequences] == [4, 4]


def test_reserve_and_free_multi_cfg_request() -> None:
    manager = _manager(8)
    free_before = manager.native_manager.block_pool.get_num_free_blocks()
    requests = (_request("public", 0), _request("public", 1))

    metadata = manager.reserve_request("public", requests)

    assert metadata is not None
    assert metadata.request_id == "public"
    assert metadata.allocation_generation == 1
    assert [sequence.sequence_id for sequence in metadata.sequences] == [0, 1]
    assert [sequence.prefix_len for sequence in metadata.sequences] == [4, 4]
    assert [sequence.target_len for sequence in metadata.sequences] == [4, 4]
    assert [sequence.seq_len for sequence in metadata.sequences] == [8, 8]
    assert [len(sequence.block_ids[0]) for sequence in metadata.sequences] == [2, 2]
    assert all(sequence.context_ids == () for sequence in metadata.sequences)
    assert metadata.contexts == ()
    assert metadata == manager.get_metadata("public")
    assert manager.native_manager.block_pool.get_num_free_blocks() == free_before - 4

    manager.free_request("public")
    assert manager.native_manager.block_pool.get_num_free_blocks() == free_before


def test_impossible_cfg_allocation_rolls_back_and_fails_fast() -> None:
    manager = _manager(3)
    free_before = manager.native_manager.block_pool.get_num_free_blocks()

    with pytest.raises(DiffusionKVAdmissionError, match="cannot fit even when the block pool is empty"):
        manager.reserve_request(
            "public",
            (_request("public", 0), _request("public", 1)),
        )

    assert manager.has_request("public") is False
    assert manager.native_manager.block_pool.get_num_free_blocks() == free_before


def test_impossible_cfg_allocation_fails_fast_while_pool_is_busy() -> None:
    manager = _manager(5)
    assert manager.reserve_request("running", (_request("running", 0),)) is not None
    free_before = manager.native_manager.block_pool.get_num_free_blocks()

    with pytest.raises(DiffusionKVAdmissionError, match="required_blocks=6, available_blocks=4"):
        manager.reserve_request(
            "impossible",
            tuple(_request("impossible", sequence_id) for sequence_id in range(3)),
        )

    assert manager.has_request("impossible") is False
    assert manager.native_manager.block_pool.get_num_free_blocks() == free_before


def test_temporary_capacity_pressure_returns_none() -> None:
    manager = _manager(3)
    assert manager.reserve_request("running", (_request("running", 0),)) is not None

    allocation = manager.reserve_request("waiting", (_request("waiting", 0),))

    assert allocation is None
    assert manager.has_request("waiting") is False


def test_partial_cfg_rollback_can_retry_after_capacity_is_released() -> None:
    manager = _manager(5)
    assert manager.reserve_request("running", (_request("running", 0),)) is not None
    waiting = (_request("waiting", 0), _request("waiting", 1))

    assert manager.reserve_request("waiting", waiting) is None
    assert manager.has_request("waiting") is False

    manager.free_request("running")
    metadata = manager.reserve_request("waiting", waiting)

    assert metadata is not None
    assert [len(sequence.block_ids[0]) for sequence in metadata.sequences] == [2, 2]


def test_reallocated_request_gets_a_new_allocation_generation() -> None:
    manager = _manager(3)
    request = (_request("public", 0),)

    first = manager.reserve_request("public", request)
    assert first is not None
    manager.free_request("public")
    second = manager.reserve_request("public", request)

    assert second is not None
    assert second.allocation_generation == first.allocation_generation + 1


def test_cfg_allocation_exception_rolls_back(monkeypatch) -> None:
    manager = _manager(8)
    free_before = manager.native_manager.block_pool.get_num_free_blocks()
    native_allocate = manager.native_manager.allocate_slots
    num_calls = 0

    def fail_second_allocation(*args, **kwargs):
        nonlocal num_calls
        num_calls += 1
        if num_calls == 2:
            raise RuntimeError("injected allocation failure")
        return native_allocate(*args, **kwargs)

    monkeypatch.setattr(manager.native_manager, "allocate_slots", fail_second_allocation)

    with pytest.raises(RuntimeError, match="injected allocation failure"):
        manager.reserve_request(
            "public",
            (_request("public", 0), _request("public", 1)),
        )

    assert manager.has_request("public") is False
    assert manager.native_manager.block_pool.get_num_free_blocks() == free_before


def test_rejects_independent_context_until_role_routing_is_implemented() -> None:
    manager = _manager(8)
    context = DiffusionKVContext(context_id="text", cache_role="cross.text", num_tokens=8)

    with pytest.raises(DiffusionKVAdmissionError, match="DiffusionKVContext"):
        manager.reserve_request(
            "public",
            (_request("public", 0, kv_contexts=(context,)),),
        )


def test_rejects_sequence_longer_than_admission_bound() -> None:
    manager = _manager(8, max_model_len=8)

    with pytest.raises(DiffusionKVAdmissionError, match="exceeds max_model_len"):
        manager.reserve_request("public", (_request("public", 0, seq_len=9),))


@pytest.mark.parametrize(
    "requests",
    [
        (_request("public", 1),),
        (_request("public", 1), _request("public", 0)),
        (_request("public", 0), _request("public", 2)),
    ],
)
def test_rejects_ambiguous_sequence_order(requests) -> None:
    manager = _manager(8)

    with pytest.raises(ValueError, match="ordered by contiguous sequence_id"):
        manager.reserve_request("public", requests)


def test_close_releases_every_public_request() -> None:
    manager = _manager(8)
    free_before = manager.native_manager.block_pool.get_num_free_blocks()
    assert manager.reserve_request("first", (_request("first", 0),)) is not None
    assert manager.reserve_request("second", (_request("second", 0),)) is not None

    manager.close()

    assert manager.native_manager.block_pool.get_num_free_blocks() == free_before
    assert manager.has_request("first") is False
    assert manager.has_request("second") is False
