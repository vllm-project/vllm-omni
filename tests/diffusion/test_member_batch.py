# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the shared AR diffusion member and batch boundary."""

import pytest
import torch

from vllm_omni.diffusion.distributed.member_batch import (
    BatchMember,
    ChunkStepPlan,
    ExecutionBatch,
    execute_member_batch,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_chunk_step_plan_preserves_grain_and_validates_bounds():
    plan = ChunkStepPlan(chunk=3, steps=2)
    assert plan.num_slots == 2
    assert plan.task(0, 0) == (3, 0)
    assert plan.task(1, 0) == (3, 1)
    assert plan.task(2, 0) is None
    with pytest.raises(ValueError, match="one logical execution position"):
        plan.task(0, 1)
    with pytest.raises(ValueError, match="positive steps"):
        ChunkStepPlan(chunk=0, steps=0)


def test_batch_splits_packed_rows_by_member_and_preserves_order():
    kv_a, kv_b = object(), object()
    batch = ExecutionBatch(
        (
            BatchMember("a", (0, 2), "input-a", kv_a, rows=2),
            BatchMember("b", (1, 0), "input-b", kv_b, rows=1),
        )
    )
    packed = torch.tensor([[10.0], [11.0], [20.0]])
    actual = batch.split_rows(packed)
    assert actual["a"].flatten().tolist() == [10.0, 11.0]
    assert actual["b"].flatten().tolist() == [20.0]
    assert batch.kv_views == (kv_a, kv_b)
    assert batch.request_ids == ("a", "b")


def test_batch_rejects_duplicate_members_mismatched_rows_and_missed_outputs():
    member = BatchMember("a", (0,), None, None)
    with pytest.raises(ValueError, match="unique non-empty"):
        ExecutionBatch((member, member))
    batch = ExecutionBatch((member,))
    with pytest.raises(ValueError, match="requires 1 result rows"):
        batch.split_rows(torch.empty((0, 4)))
    with pytest.raises(ValueError, match="produced 0 outputs"):
        batch.map_outputs(())


def test_batch_publish_happens_after_successful_compute_only():
    batch = ExecutionBatch((BatchMember("a", (0,), "in", "kv"),))
    events = []

    def compute(actual):
        assert actual is batch
        events.append("compute")
        return "result"

    def publish(actual, result):
        assert actual is batch
        assert result == "result"
        events.append("publish")

    assert execute_member_batch(batch, compute, publish) == "result"
    assert events == ["compute", "publish"]

    def fail(_actual):
        raise RuntimeError("compute failed")

    with pytest.raises(RuntimeError, match="compute failed"):
        execute_member_batch(batch, fail, publish)
    assert events == ["compute", "publish"]
