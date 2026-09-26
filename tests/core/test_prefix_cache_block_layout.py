# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare adapter layouts with vLLM's real GPU slot-mapping kernel."""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, SlidingWindowSpec
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind,
    PrefixCacheRequestEvent,
    PrefixCacheSchedulerAdapter,
    PrefixCacheWriteLayout,
)
from vllm_omni.core.prefix_cache.group_view import (
    FullAttentionGroupView,
    check_prefix_cache_kv_groups,
    get_prefix_cache_group_view,
)
from vllm_omni.core.prefix_cache.interface import HIDDEN_KEY, OmniPrefixCacheUnmatchError, PrefixCacheConfig
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


def _assert_manager_layout(layout: PrefixCacheWriteLayout, expected: torch.Tensor, block_size: int) -> None:
    assert layout.slots_cpu is not None
    assert torch.equal(layout.slots_cpu, expected)
    manager = OmniPrefixCacheManager(PrefixCacheConfig(num_blocks=32, block_size=block_size), eager=True)
    manager.new_step_starts(
        tuple(
            PrefixCacheRequestEvent(
                write.req_id, PrefixCacheEventKind.STARTED, scheduled_tokens=write.row_end - write.row_start
            )
            for write in layout.writes
        )
    )
    hidden = torch.arange(layout.total_rows, dtype=torch.float32).unsqueeze(1)
    step_id = manager.save_outputs(
        hidden, {}, num_tokens_unpadded=layout.total_rows, num_tokens_padded=layout.total_rows, write_layout=layout
    )
    assert torch.equal(manager._pool.rows(HIDDEN_KEY, expected), hidden)
    outputs = manager.materialize(step_id, [write.req_id for write in layout.writes])
    for write in layout.writes:
        assert torch.equal(outputs.hidden_states[write.req_id], hidden[write.row_start : write.row_end])


@pytest.mark.parametrize("allocator_size,kernel_size", [(16, 16), (32, 16), (128, 16), (128, 32), (128, 64)])
@pytest.mark.parametrize("start,count", [(0, 1), (13, 7), (15, 18), (31, 3), (127, 2)])
def test_adapter_matches_real_block_table(allocator_size: int, kernel_size: int, start: int, count: int) -> None:
    device = torch.device("cuda:0")
    batch = InputBatch(
        max_num_reqs=2,
        max_model_len=256,
        max_num_batched_tokens=64,
        device=device,
        vocab_size=256,
        block_sizes=[allocator_size],
        kernel_block_sizes=[kernel_size],
        max_num_blocks_per_req=[16],
    )
    for req_id, computed, blocks in [
        ("a", 0, [9, 3, 7, 1, 8, 2, 6, 4, 5]),
        ("b", start, [19, 13, 17, 11, 18, 12, 16, 14, 15]),
    ]:
        batch.add_request(
            CachedRequestState(
                req_id=req_id,
                prompt_token_ids=list(range(144)),
                mm_features=[],
                sampling_params=SamplingParams(temperature=0),
                generator=None,
                block_ids=(blocks,),
                num_computed_tokens=computed,
                output_token_ids=[],
            )
        )
    batch.swap_states(0, 1)
    table = batch.block_table[0]
    table.commit_block_table(2)
    positions = torch.cat((torch.arange(start, start + count), torch.arange(3))).to(device)
    query_start = torch.tensor([0, count, count + 3], dtype=torch.int32, device=device)
    table.compute_slot_mapping(2, query_start, positions)
    expected = table.slot_mapping.gpu[: count + 3].cpu()

    layout = PrefixCacheSchedulerAdapter().build_write_layout(
        FullAttentionGroupView(batch, allocator_size), num_scheduled_tokens={"a": 3, "b": count}
    )
    assert layout.total_rows == count + 3
    assert [(w.req_id, w.row_start, w.row_end) for w in layout.writes] == [("b", 0, count), ("a", count, count + 3)]
    assert layout.slots_cpu is not None
    assert torch.equal(layout.slots_cpu[layout.writes[0].row_start : layout.writes[0].row_end], expected[:count])
    assert torch.equal(layout.slots_cpu[layout.writes[1].row_start : layout.writes[1].row_end], expected[count:])
    _assert_manager_layout(layout, expected, allocator_size)


def test_hybrid_group_uses_dense_table_after_sliding_recycling() -> None:
    block_size = 128
    batch = InputBatch(
        max_num_reqs=1,
        max_model_len=512,
        max_num_batched_tokens=16,
        device=torch.device("cuda:0"),
        vocab_size=256,
        block_sizes=[block_size, block_size],
        kernel_block_sizes=[block_size, 64],
        max_num_blocks_per_req=[16, 16],
    )
    batch.add_request(
        CachedRequestState(
            req_id="a",
            prompt_token_ids=list(range(256)),
            mm_features=[],
            sampling_params=SamplingParams(temperature=0),
            generator=None,
            block_ids=([1, 1, 1], [9, 3, 7]),
            num_computed_tokens=127,
            output_token_ids=[],
        )
    )
    groups = [
        KVCacheGroupSpec(
            ["sliding"],
            SlidingWindowSpec(block_size=128, num_kv_heads=1, head_size=64, dtype=torch.float16, sliding_window=128),
        ),
        KVCacheGroupSpec(
            ["full"], FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=64, dtype=torch.float16)
        ),
    ]
    view = get_prefix_cache_group_view(batch, block_size, groups)
    assert view is not None and view.group_id == 1
    batch.block_table.commit_block_table(1)
    positions = torch.arange(127, 135, device="cuda:0")
    query_start = torch.tensor([0, 8], dtype=torch.int32, device="cuda:0")
    batch.block_table.compute_slot_mapping(1, query_start, positions)
    expected = batch.block_table[1].slot_mapping.gpu[:8].cpu()
    layout = PrefixCacheSchedulerAdapter().build_write_layout(view, num_scheduled_tokens={"a": 8})
    _assert_manager_layout(layout, expected, block_size)


def test_sliding_only_has_no_stable_output_group() -> None:
    group = KVCacheGroupSpec(
        ["sliding"],
        SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float16, sliding_window=64),
    )
    with pytest.raises(OmniPrefixCacheUnmatchError, match="requires a full-attention KV group"):
        check_prefix_cache_kv_groups([group])
