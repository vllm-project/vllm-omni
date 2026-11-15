# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A stage-local private result must not acquire a downstream delivery ticket."""

import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("async_chunk", [False, True])
@pytest.mark.parametrize(
    "stage_id,final_stage,has_downstream", [(0, 0, False), (0, 1, True), (1, 1, False), (0, None, True)]
)
def test_stage_local_result_skips_transport_and_releases_lifecycle(
    stage_id, final_stage, has_downstream, serialized, async_chunk
):
    plane = object.__new__(OmniRunnerDataPlane)
    plane._stage_id = stage_id
    plane._can_send = True
    plane._async_chunk = async_chunk
    plane._native_output_lock = threading.RLock()
    plane._native_send_lock = threading.Lock()
    plane._native_output_error_lock = threading.Lock()
    plane._native_output_error = None
    plane._native_outputs_in_flight = defaultdict(int)
    plane._native_terminal_pending = set()
    plane._put_req_chunk = defaultdict(int)
    plane._pending_full_payload_send = {}
    plane._custom_process_func = SimpleNamespace(_requires_full_token_history=False)
    info = {"omni_final_stage_id": final_stage} if final_stage is not None else {}
    plane._native_requests = {}
    plane.register_request(
        SimpleNamespace(
            req_id="private",
            external_req_id="private",
            prompt_token_ids=[1, 2],
            additional_information=serialize_additional_information(info) if serialized else info,
        )
    )
    plane.reserve_outputs(["private"])
    # Only external transport and connector cleanup are replaced. Native output
    # accounting, terminal ordering and state release execute unchanged.
    plane.send_chunks = Mock(side_effect=lambda entries, **kwargs: len(entries))
    plane.cleanup_finished_request = Mock()

    plane.complete_outputs(
        req_ids=["private"],
        inter_stage_outputs=[{"hidden_states": {"output": "synthetic"}}],
        sampled_token_ids=[[3]],
    )
    assert "private" in plane._native_requests
    if not has_downstream:
        plane.send_chunks.assert_not_called()
    else:
        assert plane.send_chunks.call_count == int(async_chunk)

    plane.request_terminal({"private"})
    assert "private" not in plane._native_requests
    assert not plane._native_outputs_in_flight
    assert not plane._native_terminal_pending
    plane.cleanup_finished_request.assert_called_once_with("private")
    assert plane.send_chunks.call_count == ((2 if async_chunk else 1) if has_downstream else 0)
