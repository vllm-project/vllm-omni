# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generic Batch organization and member/activation mapping."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.batch import (
    ComputeBatch,
    MemberRef,
    organize_batches,
    slice_activation_for_task,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_organize_groups_and_splits_by_max_batch_size():
    cell = (0, 0, 0)
    tasks = [("A", cell), ("B", cell), ("C", cell)]
    kvs = [["ka"], ["kb"], ["kc"]]
    batches = organize_batches(
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


def test_organize_respects_compatibility_key_order_and_split_outputs():
    """Custom keys group A/C together; task order and row→member mapping are preserved."""
    cell = (0, 0, 0)
    tasks = [("A", cell), ("B", cell), ("C", cell)]
    keys = {"A": (cell, 1.0), "B": (cell, 8.0), "C": (cell, 1.0)}
    batches = organize_batches(
        tasks,
        [None, None, None],
        compatibility_of=lambda req, c: keys[req],
        max_batch_size=4,
    )
    assert [b.request_ids for b in batches] == [("A", "C"), ("B",)]
    assert batches[0].task_indices == (0, 2)
    split = batches[0].split_outputs({"latent": torch.stack([torch.full((1,), 1.0), torch.full((1,), 3.0)])})
    assert split[0]["latent"].item() == 1.0
    assert split[1]["latent"].item() == 3.0


def test_compute_batch_rejects_mixed_blocks():
    with pytest.raises(ValueError, match="share one block"):
        ComputeBatch(
            members=(
                MemberRef("A", chunk=0, step=0, block=0, task_index=0),
                MemberRef("B", chunk=1, step=0, block=1, task_index=1),
            ),
            kv_views=(None, None),
        )


def test_slice_activation_maps_packed_rows_by_task_index():
    """PP / split batches must slice the shared payload; never re-pop per member."""
    packed = {
        "latent": torch.stack(
            [
                torch.full((2,), 1.0),
                torch.full((2,), 2.0),
                torch.full((2,), 3.0),
            ]
        ),
        "hidden_states": torch.stack(
            [
                torch.full((4,), 10.0),
                torch.full((4,), 20.0),
                torch.full((4,), 30.0),
            ]
        ),
    }
    part = slice_activation_for_task(packed, task_index=2, task_count=3)
    assert part is not None
    assert torch.allclose(part["latent"], torch.full((1, 2), 3.0))
    assert torch.allclose(part["hidden_states"], torch.full((1, 4), 30.0))
    assert slice_activation_for_task(None, task_index=0, task_count=1) is None


@pytest.mark.parametrize("shape", [(), (1, 4), (3, 4)])
def test_activation_rejects_missing_or_extra_member_rows(shape):
    with pytest.raises(ValueError, match="one row per member"):
        slice_activation_for_task({"latent": torch.zeros(shape)}, task_index=0, task_count=2)
    batch = ComputeBatch(
        members=(MemberRef("A", 0, 0, 0, 0), MemberRef("B", 0, 0, 0, 1)),
        kv_views=(None, None),
    )
    with pytest.raises(ValueError, match="one row per member"):
        batch.split_outputs({"latent": torch.zeros(shape)})


@pytest.mark.parametrize("contexts", [[], ["a", "b"]])
def test_organize_rejects_ambiguous_kv_views(contexts):
    with pytest.raises(ValueError, match="exactly one KV view"):
        organize_batches([("A", (0, 0, 0))], [contexts], compatibility_of=lambda *_: 0, max_batch_size=1)
