# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Input readiness is shared without consuming connector state."""

import pytest

from tests.core.sched.test_generation_scheduler_atomic_chunk import FakeAdapter
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.outputs import OmniConnectorOutput, SchedulingMetadataUpdate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(params=[OmniARScheduler, OmniGenerationScheduler])
def scheduler(request):
    s = request.param.__new__(request.param)
    s.chunk_transfer_adapter = None
    s.input_coordinator = None
    s._native_data_plane = False
    return s


def test_legacy_readiness_is_a_snapshot(scheduler):
    adapter = scheduler.chunk_transfer_adapter = FakeAdapter()
    assert scheduler._get_coalescing_ready_request_ids() == set()
    adapter.requests_with_ready_chunks.add("ready")
    ready = scheduler._get_coalescing_ready_request_ids()
    assert ready == {"ready"}
    ready.clear()
    assert adapter.requests_with_ready_chunks == {"ready"}


@pytest.mark.parametrize("source", ["none", "sender", "stage_zero", "full_payload"])
def test_unsupported_sources_are_distinct_from_empty(scheduler, source):
    if source == "sender":
        scheduler.chunk_transfer_adapter = FakeAdapter(receives_chunks=False)
    elif source in ("stage_zero", "full_payload"):
        scheduler._native_data_plane = True
        scheduler.input_coordinator = OmniSchedulingCoordinator(
            stage_id=0 if source == "stage_zero" else 1,
            async_chunk=source != "full_payload",
        )
    assert scheduler._get_coalescing_ready_request_ids() is None


def test_native_readiness_includes_unapplied_terminal_notifications(scheduler):
    scheduler._native_data_plane = True
    coordinator = scheduler.input_coordinator = OmniSchedulingCoordinator(stage_id=1, async_chunk=True)
    assert scheduler._get_coalescing_ready_request_ids() == set()
    coordinator.requests_with_ready_chunks.add("ready")
    coordinator.finished_requests.add("finished")
    coordinator.input_terminal_req_ids.add("terminal")
    output = OmniConnectorOutput(
        chunk_ready_req_ids={"new", "ready"},
        chunk_finished_req_ids={"new_finished"},
        request_metadata={
            "new_terminal": SchedulingMetadataUpdate(input_terminal=True),
            "other": SchedulingMetadataUpdate(input_terminal=False),
        },
    )
    assert scheduler._get_coalescing_ready_request_ids(iter([output])) == {
        "ready",
        "finished",
        "terminal",
        "new",
        "new_finished",
        "new_terminal",
    }
    assert coordinator.requests_with_ready_chunks == {"ready"}
    assert coordinator.finished_requests == {"finished"}
    assert coordinator.input_terminal_req_ids == {"terminal"}
    assert output.chunk_ready_req_ids == {"new", "ready"}
    assert output.chunk_finished_req_ids == {"new_finished"}
