# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CellComputeBatch member contract and compatibility organization."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.cell_batch import (
    CellComputeBatch,
    CellMemberRef,
    organize_cell_batches,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_organize_groups_same_cell_and_splits_by_max_batch_size():
    cell = (0, 0, 0)
    tasks = [("A", cell), ("B", cell), ("C", cell)]
    kvs = [["ka"], ["kb"], ["kc"]]
    batches = organize_cell_batches(
        tasks,
        kvs,
        compatibility_of=lambda req, c: c,
        max_batch_size=2,
    )
    assert len(batches) == 2
    assert batches[0].request_ids == ("A", "B")
    assert batches[0].kv_views == ("ka", "kb")
    assert batches[0].task_indices == (0, 1)
    assert batches[1].request_ids == ("C",)
    assert batches[1].size == 1


def test_organize_keeps_incompatible_cells_separate_and_preserves_task_order():
    tasks = [("A", (0, 0, 0)), ("B", (1, 0, 0)), ("C", (0, 0, 0))]
    kvs = [object(), object(), object()]
    batches = organize_cell_batches(
        tasks,
        kvs,
        compatibility_of=lambda req, c: c,
        max_batch_size=4,
    )
    assert [b.request_ids for b in batches] == [("A", "C"), ("B",)]
    assert batches[0].task_indices == (0, 2)
    split = batches[0].split_outputs(
        {"latent": torch.stack([torch.full((1,), 1.0), torch.full((1,), 3.0)])}
    )
    assert split[0]["latent"].item() == 1.0
    assert split[1]["latent"].item() == 3.0


def test_cell_compute_batch_rejects_mixed_cells():
    with pytest.raises(ValueError, match="share one cell"):
        CellComputeBatch(
            members=(
                CellMemberRef("A", (0, 0, 0), 0),
                CellMemberRef("B", (1, 0, 0), 1),
            ),
            kv_views=(None, None),
        )
