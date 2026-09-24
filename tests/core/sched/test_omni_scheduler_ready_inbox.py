# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drain ready events once, merging live requests and filtering cancellations."""

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.outputs import OmniConnectorOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ready_inbox_coalesces_live_events_and_drops_cancelled(mocker):
    scheduler = OmniSchedulerMixin()
    scheduler.requests, scheduler.waiting, scheduler.running = {"r": object()}, [], []
    coordinator = SimpleNamespace(
        _async_chunk=True, update_request_metadata=mocker.Mock(), process_pending_chunks=mocker.Mock()
    )
    scheduler.input_coordinator = coordinator
    scheduler._init_omni_connector_output_inbox()
    scheduler.enqueue_omni_connector_output(OmniConnectorOutput(chunk_ready_req_ids={"r"}))
    scheduler.enqueue_omni_connector_output(
        OmniConnectorOutput(
            chunk_ready_req_ids={"r", "aborted"},
            chunk_finished_req_ids={"r", "aborted"},
            request_metadata={"r": {"decode_token_end": 2}, "aborted": {"decode_token_end": 99}},
        )
    )
    scheduler._consume_pending_connector_output(model_mode="ar")
    coordinator.update_request_metadata.assert_called_once_with(
        scheduler.requests, {"r": {"decode_token_end": 2}}, model_mode="ar"
    )
    coordinator.process_pending_chunks.assert_called_once_with([], [], {"r"}, {"r"})
    scheduler._consume_pending_connector_output(model_mode="ar")
    assert coordinator.update_request_metadata.call_count == 1
    coordinator.process_pending_chunks.assert_called_with([], [], set(), set())
