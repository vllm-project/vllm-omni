# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The AR-Diffusion execution envelope: identity, ownership and attribution."""

from __future__ import annotations

import pytest

from vllm_omni.experimental.ar_diffusion.batch import (
    ARDiffusionBatch,
    ARDiffusionExecutionMode,
    ARDiffusionRequestRecord,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_a_request_mode_cell_call_owns_one_record() -> None:
    batch = ARDiffusionBatch.for_noisy(["session-a"], session_ids=["session-a"])

    assert batch.mode is ARDiffusionExecutionMode.NOISY_CONTINUOUS
    assert batch.request_ids == ("session-a",)
    assert batch.record.session_id == "session-a"
    assert batch.record.owned_kv_ids == ("session-a",)


def test_a_cell_batch_wave_owns_the_records_it_schedules() -> None:
    """A wave may schedule several requests; each stays independently identified."""
    batch = ARDiffusionBatch.for_noisy(["session-a", "request-7"], session_ids=["session-a", None])

    assert batch.request_ids == ("session-a", "request-7")
    assert [record.session_id for record in batch.records] == ["session-a", None]
    with pytest.raises(ValueError, match="exactly one request record"):
        _ = batch.record


def test_a_paged_batch_carries_several_records() -> None:
    batch = ARDiffusionBatch.for_paged(["request-1", "request-2"], session_ids=["s1", None])

    assert batch.mode is ARDiffusionExecutionMode.PAGED_CONTINUOUS
    assert batch.request_ids == ("request-1", "request-2")


def test_duplicate_or_empty_batches_are_refused() -> None:
    with pytest.raises(ValueError, match="unique"):
        ARDiffusionBatch.for_paged(["request-1", "request-1"])
    with pytest.raises(ValueError, match="at least one"):
        ARDiffusionBatch.for_noisy([])
    with pytest.raises(ValueError, match="align"):
        ARDiffusionBatch.for_paged(["request-1", "request-2"], session_ids=["s1"])


def test_payload_and_output_attribution_reject_unknown_requests() -> None:
    batch = ARDiffusionBatch.for_noisy(["session-a"])

    batch.set_strategy_payload("session-a", {"cell_plan": "plan"})
    batch.set_output("session-a", "output")
    assert batch.records[0].strategy_payload == {"cell_plan": "plan"}
    assert batch.outputs == {"session-a": "output"}

    with pytest.raises(KeyError, match="request-b"):
        batch.set_strategy_payload("request-b", object())
    with pytest.raises(KeyError, match="request-b"):
        batch.set_output("request-b", object())


def test_the_cell_context_refuses_a_request_outside_the_envelope() -> None:
    """The executor only runs requests the call owns."""
    from types import SimpleNamespace

    from vllm_omni.experimental.ar_diffusion.cell_executor import (
        ARDiffusionCellContext,
        CellRunSpec,
        CellTopology,
    )

    ctx = ARDiffusionCellContext(
        spec=CellRunSpec(topology=CellTopology(step_replicas=1, layer_groups=1), rank=0),
        kv=SimpleNamespace(),
        batch=ARDiffusionBatch.for_noisy(["session-a"]),
    )

    with pytest.raises(ValueError, match="not part of the AR-Diffusion batch"):
        ctx.enqueue("request-b", SimpleNamespace(), chunk_tokens=4)


def test_a_request_record_needs_an_identity() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ARDiffusionRequestRecord("")
