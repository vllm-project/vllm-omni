# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""V1 coalescing retains batch rows while still processing idle-step cleanup."""

import numpy as np
import pytest
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_omni.core.sched.output import OmniSchedulerOutput
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def runner(mocker):
    r = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    r.requests = {
        req_id: CachedRequestState(
            req_id=req_id,
            prompt_token_ids=[1] * 128,
            mm_features=[],
            sampling_params=None,
            generator=None,
            block_ids=([0],),
            num_computed_tokens=128,
            output_token_ids=[2],
        )
        for req_id in ("live", "done")
    }
    batch = r.input_batch = mocker.Mock(spec=InputBatch)
    batch.req_id_to_index = {"done": 0, "live": 1}
    batch.prev_req_id_to_index = batch.req_id_to_index.copy()
    batch.num_computed_tokens_cpu = np.array([128, 128])
    batch.remove_request.side_effect = lambda req_id: batch.req_id_to_index.pop(req_id, None)

    def condense():
        # Mirror row compaction while retaining the previous GPU batch's map.
        batch.req_id_to_index = {req_id: i for i, req_id in enumerate(batch.req_id_to_index)}

    batch.condense.side_effect = condense
    r.model_intermediate_buffer = {"done": {}}
    r.num_prompt_logprobs = {"done": 1}
    r.encoder_cache = {"old-mm": object()}
    r.late_interaction_runner = mocker.Mock()
    r._omni_connector_initialized = True
    r.cleanup_finished_request = mocker.Mock()
    mocker.patch.object(r, "_drop_hs_pending_state")
    mocker.patch.object(r, "_may_reorder_batch")
    r.speculative_config = None
    r.use_async_spec_decode = True
    r.use_async_scheduling = True
    r.prev_num_draft_tokens = mocker.Mock(np=np.array([0, 2]))
    mocker.patch("vllm_omni.worker.gpu_model_runner.get_pp_group").return_value.is_last_rank = True
    return r


@pytest.mark.parametrize("preserve", [False, True])
def test_idle_step_retains_only_coalesced_rows_and_cleans_finished_requests(mocker, preserve):
    r = runner(mocker)
    output = OmniSchedulerOutput.make_empty()
    output.preserve_v1_batch = preserve
    output.finished_req_ids = {"done"}
    output.free_encoder_mm_hashes = ["old-mm"]
    assert r._update_states(output) is None
    assert r.input_batch.req_id_to_index == ({"live": 0} if preserve else {})
    assert "done" not in r.requests
    assert "done" not in r.model_intermediate_buffer
    assert "done" not in r.num_prompt_logprobs
    assert not r.encoder_cache
    r.cleanup_finished_request.assert_called_once_with("done")
    r.input_batch.add_request.assert_not_called()
    r.input_batch.refresh_metadata.assert_called_once()
    if preserve:
        assert r.prev_num_draft_tokens.np.tolist() == [0, 2]
        r._may_reorder_batch.assert_not_called()
    else:
        assert r.prev_num_draft_tokens.np.tolist() == [0, 0]


@pytest.mark.parametrize("marker_on_execution", [False, True])
def test_retained_row_resumes_async_spec_corrections_without_history_payload(mocker, marker_on_execution):
    r = runner(mocker)
    r.requests["live"].prev_num_draft_len = 2
    idle = OmniSchedulerOutput.make_empty()
    idle.preserve_v1_batch = True
    idle.finished_req_ids = {"done"}
    r._update_states(idle)
    another_idle = OmniSchedulerOutput.make_empty()
    another_idle.preserve_v1_batch = True
    r._update_states(another_idle)
    output = OmniSchedulerOutput.make_empty()
    output.preserve_v1_batch = marker_on_execution
    output.num_scheduled_tokens = {"live": 1}
    output.total_num_scheduled_tokens = 1
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["live"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[130],
        num_output_tokens=[3],
    )
    correction = r._update_states(output)
    r.input_batch.add_request.assert_not_called()
    assert r.requests["live"].num_computed_tokens == 130
    mocker.patch.object(r, "_get_valid_sampled_token_count", return_value=[1, 2])
    assert correction is not None
    correction()
    assert r.requests["live"].num_computed_tokens == 129
    assert r.input_batch.num_computed_tokens_cpu[0] == 129
    r._may_reorder_batch.assert_called_once_with(output)
