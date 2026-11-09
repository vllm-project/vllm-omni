# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Duplex's allocation window must not mask its dense attention history."""

import pytest
import torch
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.worker.gpu.block_table import BlockTables

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
    duplex_window_geometry,
    duplex_window_spec,
)

pytestmark = pytest.mark.core_model


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("computed", [8121, 8200, 16384])
def test_duplex_keeps_history_beyond_allocation_window(computed: int) -> None:
    geometry = duplex_window_geometry(
        prefix_tokens=96,
        window_tokens=6000,
        block_size=16,
        max_model_len=40960,
        high_watermark_tokens=8000,
    )
    ordinary = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=8112,
    )
    duplex = duplex_window_spec(geometry, ordinary)
    # The override changes attention visibility, not the admission budget.
    assert duplex.max_admission_blocks_per_request(128, 40960) == ordinary.max_admission_blocks_per_request(128, 40960)
    tables = BlockTables(
        block_sizes=[16, 16],
        max_num_reqs=2,
        max_num_batched_tokens=128,
        max_num_blocks_per_group=[2560, 2560],
        device=torch.device("cuda"),
        sliding_window_reach=[duplex.num_retained_tokens, ordinary.num_retained_tokens],
    )
    count = (computed + 1 + 15) // 16
    blocks = list(range(1, count + 1))
    tables.append_block_ids(1, (blocks, blocks), overwrite=True)
    tables.apply_staged_writes()
    slots = torch.tensor([1], device="cuda", dtype=torch.int32)
    positions = torch.tensor([0, computed], device="cuda", dtype=torch.int32)
    gathered = tables.gather_block_tables(slots, 1, num_computed_tokens=positions)
    assert gathered[0][0, :count].tolist() == blocks
    null_count = max(0, computed - ordinary.num_retained_tokens) // 16
    assert gathered[1][0, :count].tolist() == [0] * null_count + blocks[null_count:]

    # After the manager compacts/reanchors a session, the worker must read
    # exactly the replacement table, including its retained sink pages.
    compacted = blocks[:6] + blocks[-4:]
    tables.append_block_ids(1, (compacted, compacted), overwrite=True)
    tables.apply_staged_writes()
    positions.fill_(len(compacted) * 16 - 1)
    gathered = tables.gather_block_tables(slots, 1, num_computed_tokens=positions)
    assert gathered[0][0, : len(compacted)].tolist() == compacted
