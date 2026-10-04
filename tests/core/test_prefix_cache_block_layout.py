# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare adapter layouts with vLLM's real GPU slot-mapping kernel."""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_omni.core.prefix_cache.adapter import (
    PrefixCacheEventKind,
    PrefixCacheRequestEvent,
    PrefixCacheSchedulerAdapter,
)
from vllm_omni.core.prefix_cache.group_view import FullAttentionGroupView
from vllm_omni.core.prefix_cache.interface import HIDDEN_KEY, PrefixCacheConfig
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


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

    manager = OmniPrefixCacheManager(PrefixCacheConfig(num_blocks=32, block_size=allocator_size), eager=True)
    manager.new_step_starts(
        (
            PrefixCacheRequestEvent("b", PrefixCacheEventKind.STARTED, scheduled_tokens=count),
            PrefixCacheRequestEvent("a", PrefixCacheEventKind.STARTED, scheduled_tokens=3),
        )
    )
    hidden = torch.arange(count + 3, dtype=torch.float32).unsqueeze(1)
    step_id = manager.save_outputs(
        hidden, {}, num_tokens_unpadded=count + 3, num_tokens_padded=count + 3, write_layout=layout
    )
    assert torch.equal(manager._pool.rows(HIDDEN_KEY, expected), hidden)
    output = manager.materialize(step_id, ["b", "a"])
    assert torch.equal(output.hidden_states["b"], hidden[:count])
    assert torch.equal(output.hidden_states["a"], hidden[count:])
