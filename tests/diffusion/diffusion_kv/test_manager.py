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
    atomic_cache_spans=(),
    cache_dependencies=(),
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
        atomic_cache_spans=atomic_cache_spans,
        cache_dependencies=cache_dependencies,
        kv_contexts=kv_contexts,
    )


def _manager(
    num_blocks: int,
    *,
    max_model_len: int = 64,
    enable_prefix_caching: bool = False,
    cache_granularity: int = BLOCK_SIZE,
) -> DiffusionKVCacheManager:
    return DiffusionKVCacheManager(
        _config(num_blocks),
        max_model_len=max_model_len,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=cache_granularity,
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


@pytest.mark.parametrize("num_rows", [1, 2])
def test_lru_evicts_many_distinct_prefixes_without_exhausting_pool(num_rows: int) -> None:
    blocks_per_row = 4
    manager = _manager(num_rows * blocks_per_row + 1, enable_prefix_caching=True)
    free_capacity = manager.native_manager.block_pool.get_num_free_blocks()

    for generation in range(128):
        public_id = f"request-{generation}"
        rows = tuple(
            _request(
                public_id,
                sequence_id,
                prefix_len=12,
                seq_len=16,
                cache_token_ids=tuple(generation * 10_000 + sequence_id * 100 + token for token in range(12)),
            )
            for sequence_id in range(num_rows)
        )
        metadata = manager.reserve_request(public_id, rows)
        assert metadata is not None
        assert all(sequence.cached_prefix_len == 0 for sequence in metadata.sequences)
        manager.publish_request(public_id)
        manager.free_request(public_id)

        assert manager.native_manager.block_pool.get_num_free_blocks() == free_capacity
        assert manager._cow_retained_blocks == {}
        assert (
            len(manager.native_manager.block_pool.cached_block_hashes_by_block)
            <= manager.native_manager.block_pool.num_gpu_blocks
        )


def test_lru_reclaims_partial_cow_aliases_and_retained_blocks() -> None:
    # Native caching reserves one null block; six configured blocks leave the
    # five physical blocks required by four request pages plus one COW source.
    manager = _manager(6, enable_prefix_caching=True, cache_granularity=1)
    free_capacity = manager.native_manager.block_pool.get_num_free_blocks()

    cold = _request(
        "cold",
        0,
        prefix_len=12,
        seq_len=16,
        cache_token_ids=range(12),
        atomic_cache_spans=((1, 5), (7, 11)),
        cache_dependencies=((1, "image-1"), (7, "image-2-0")),
    )
    assert manager.reserve_request("cold", (cold,)) is not None
    manager.publish_request("cold")
    manager.free_request("cold")

    for generation in range(1, 128):
        public_id = f"changed-{generation}"
        changed = _request(
            public_id,
            0,
            prefix_len=12,
            seq_len=16,
            cache_token_ids=range(12),
            atomic_cache_spans=((1, 5), (7, 11)),
            cache_dependencies=((1, "image-1"), (7, f"image-2-{generation}")),
        )
        metadata = manager.reserve_request(public_id, (changed,))
        assert metadata is not None
        assert metadata.sequences[0].cached_prefix_len == 7
        assert len(metadata.block_copies) == 1
        manager.complete_block_copies(public_id)
        manager.publish_request(public_id)
        manager.free_request(public_id)

        assert manager.native_manager.block_pool.get_num_free_blocks() == free_capacity
        assert manager._cow_retained_blocks == {}
        assert all(
            cache_manager.new_block_ids == []
            for cache_manager in manager.native_manager.coordinator.single_type_managers
        )
        cached_hash_count = sum(
            len(hashes) for hashes in manager.native_manager.block_pool.cached_block_hashes_by_block.values()
        )
        assert cached_hash_count <= manager.native_manager.block_pool.num_gpu_blocks * 5

    # Abort/free before Worker completion must also release both retained COW
    # endpoints instead of permanently pinning pages.
    aborted = _request(
        "aborted",
        0,
        prefix_len=12,
        seq_len=16,
        cache_token_ids=range(12),
        atomic_cache_spans=((1, 5), (7, 11)),
        cache_dependencies=((1, "image-1"), (7, "image-2-aborted")),
    )
    aborted_metadata = manager.reserve_request("aborted", (aborted,))
    assert aborted_metadata is not None and aborted_metadata.block_copies
    manager.free_request("aborted")
    assert manager.native_manager.block_pool.get_num_free_blocks() == free_capacity
    assert manager._cow_retained_blocks == {}
    assert all(
        cache_manager.new_block_ids == [] for cache_manager in manager.native_manager.coordinator.single_type_managers
    )


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


def test_atomic_dependency_hit_uses_exact_boundary_with_copy_on_write() -> None:
    manager = _manager(12, enable_prefix_caching=True, cache_granularity=1)
    cold = _request(
        "cold",
        0,
        prefix_len=12,
        seq_len=16,
        cache_token_ids=range(12),
        atomic_cache_spans=((1, 5), (7, 11)),
        cache_dependencies=((1, "image-1"), (7, "image-2-old")),
    )
    assert manager.reserve_request("cold", (cold,)) is not None
    manager.publish_request("cold")
    manager.free_request("cold")

    warm = _request(
        "warm",
        0,
        prefix_len=12,
        seq_len=16,
        cache_token_ids=range(12),
        atomic_cache_spans=((1, 5), (7, 11)),
        cache_dependencies=((1, "image-1"), (7, "image-2-new")),
    )
    metadata = manager.reserve_request("warm", (warm,))

    assert metadata is not None
    assert metadata.sequences[0].cached_prefix_len == 7
    assert len(metadata.block_copies) == 1
    src_block_id, dst_block_id = metadata.block_copies[0]
    assert src_block_id != dst_block_id
    assert metadata.sequences[0].block_ids[0][1] == dst_block_id
    manager.complete_block_copies("warm")
    manager.publish_request("warm")
    manager.free_request("warm")

    for public_id, second_dependency in (("old-again", "image-2-old"), ("new-again", "image-2-new")):
        repeated = _request(
            public_id,
            0,
            prefix_len=12,
            seq_len=16,
            cache_token_ids=range(12),
            atomic_cache_spans=((1, 5), (7, 11)),
            cache_dependencies=((1, "image-1"), (7, second_dependency)),
        )
        repeated_metadata = manager.reserve_request(public_id, (repeated,))
        assert repeated_metadata is not None
        assert repeated_metadata.sequences[0].cached_prefix_len == 12
        assert repeated_metadata.block_copies == ()
        manager.free_request(public_id)


def test_cfg_rows_clip_partial_hits_before_taking_common_boundary() -> None:
    manager = _manager(24, enable_prefix_caching=True, cache_granularity=1)
    for public_id, dependency in (("seed-a", "second-a"), ("seed-b", "second-b")):
        cold = _request(
            public_id,
            0,
            prefix_len=12,
            seq_len=16,
            cache_token_ids=range(12),
            atomic_cache_spans=((1, 5), (7, 11)),
            cache_dependencies=((1, "first"), (7, dependency)),
        )
        assert manager.reserve_request(public_id, (cold,)) is not None
        manager.publish_request(public_id)
        manager.free_request(public_id)

    rows = (
        _request(
            "warm",
            0,
            prefix_len=12,
            seq_len=16,
            cache_token_ids=range(12),
            atomic_cache_spans=((1, 5), (7, 11)),
            cache_dependencies=((1, "first"), (7, "second-a")),
        ),
        _request(
            "warm",
            1,
            prefix_len=12,
            seq_len=16,
            cache_token_ids=range(12),
            atomic_cache_spans=((1, 5), (7, 11)),
            cache_dependencies=((1, "first"), (7, "changed")),
        ),
    )
    metadata = manager.reserve_request("warm", rows)

    assert metadata is not None
    assert [sequence.cached_prefix_len for sequence in metadata.sequences] == [7, 7]
    assert len(metadata.block_copies) == 2


@pytest.mark.parametrize("num_images", [2, 3, 4, 5])
def test_last_changed_atomic_span_reuses_every_preceding_span(num_images: int) -> None:
    manager = _manager(24, max_model_len=64, enable_prefix_caching=True, cache_granularity=1)
    all_spans = ((1, 5), (7, 13), (15, 19), (21, 29), (31, 37))
    spans = all_spans[:num_images]
    cold_dependencies = tuple((start, f"image-{idx}") for idx, (start, _) in enumerate(spans))
    cold = _request(
        "cold",
        0,
        prefix_len=40,
        target_len=4,
        seq_len=44,
        cache_token_ids=range(40),
        atomic_cache_spans=spans,
        cache_dependencies=cold_dependencies,
    )
    assert manager.reserve_request("cold", (cold,)) is not None
    manager.publish_request("cold")
    manager.free_request("cold")

    changed_dependencies = (*cold_dependencies[:-1], (spans[-1][0], "changed-last-image"))
    warm = _request(
        "warm",
        0,
        prefix_len=40,
        target_len=4,
        seq_len=44,
        cache_token_ids=range(40),
        atomic_cache_spans=spans,
        cache_dependencies=changed_dependencies,
    )
    metadata = manager.reserve_request("warm", (warm,))

    assert metadata is not None
    assert metadata.sequences[0].cached_prefix_len == spans[-1][0]
    assert len(metadata.block_copies) == 1


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
