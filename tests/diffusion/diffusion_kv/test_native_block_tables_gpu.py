# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from importlib import import_module, util

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.gpu]


def test_native_block_tables_slot_mapping_handles_non_aligned_tail() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the native BlockTables slot-mapping smoke test")
    module_name = "vllm.v1.worker.gpu.block_table"
    try:
        module_spec = util.find_spec(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is None or not module_name.startswith(f"{exc.name}."):
            raise
        pytest.skip("installed vLLM does not provide GPU BlockTables")
    if module_spec is None:
        pytest.skip("installed vLLM does not provide GPU BlockTables")
    block_table_module = import_module(module_name)
    block_tables_cls = getattr(block_table_module, "BlockTables", None)
    if block_tables_cls is None or not hasattr(block_tables_cls, "compute_slot_mappings"):
        pytest.skip("installed vLLM does not provide the required BlockTables API")

    device = torch.device("cuda", torch.accelerator.current_device_index())
    block_tables = block_tables_cls(
        block_sizes=[4],
        max_num_reqs=1,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[4],
        device=device,
        kernel_block_sizes=[4],
    )
    block_tables.append_block_ids(0, ([1, 2, 3],), overwrite=True)
    block_tables.apply_staged_writes()
    idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 9], dtype=torch.int32, device=device)
    positions = torch.arange(9, dtype=torch.int64, device=device)

    slot_mappings = block_tables.compute_slot_mappings(
        idx_mapping,
        query_start_loc,
        positions,
        num_tokens_padded=9,
    )
    torch.accelerator.synchronize(device)

    assert slot_mappings.cpu().tolist() == [[4, 5, 6, 7, 8, 9, 10, 11, 12]]
