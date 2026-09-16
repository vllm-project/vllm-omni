# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""v0.28 prefill-stats finalization in the Omni schedulers (#6606 review).

Upstream v0.28 finalizes prefill statistics with
``PrefillStats.finalize(kv_cache_manager.estimate_cached_tokens(request))``
before emitting them, which computes ``num_cache_creation_tokens`` (tokens
computed and written to the prefix cache). Both Omni schedulers previously
emitted ``take_prefill_stats()`` raw, so creation usage was always 0.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request
# before Request / RequestStatus are bound in this module.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_request() -> Request:
    request = Request(
        request_id="req-prefill-stats",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    request.status = RequestStatus.RUNNING
    # A scheduled prefill: 3 prompt tokens, 1 from the local prefix cache.
    request.prefill_stats.set(
        num_prompt_tokens=3,
        num_local_cached_tokens=1,
        num_external_cached_tokens=0,
    )
    return request


def _make_sched(request: Request) -> MagicMock:
    sched = MagicMock()
    sched.requests = {request.request_id: request}
    sched.perf_metrics = None
    sched.defer_block_free = False
    sched.structured_output_manager.should_advance.return_value = False
    sched._process_kv_transfer_trigger.return_value = False
    sched._maybe_decode_pooling_output.return_value = None
    sched._handle_stopped_request.return_value = True
    # The whole 3-token prompt ends up cached: 3 - 1 pre-cached = 2 created.
    sched.kv_cache_manager.estimate_cached_tokens.return_value = 3

    def _free_and_invalidate(req):
        # Lifecycle guard: freeing releases the KV blocks, after which the
        # manager reports 0 cached tokens. If finalize ran after
        # _free_request (the reviewed bug), creation count would compute
        # against 0 and the assertions below would see 0, not 2.
        sched.kv_cache_manager.estimate_cached_tokens.return_value = 0
        return (None, None)

    sched._free_request.side_effect = _free_and_invalidate
    sched.chunk_transfer_adapter = None
    sched.finished_req_ids_dict = None
    sched._new_prompt_len_snapshot = {}
    return sched


def _frame(request: Request):
    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {request.request_id: 1}
    scheduler_output.total_num_scheduled_tokens = 1
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = MagicMock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = [[42]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {request.request_id: 0}
    model_runner_output.routed_experts = None
    return scheduler_output, model_runner_output


def test_ar_scheduler_finalizes_prefill_stats_before_emitting() -> None:
    request = _make_request()
    sched = _make_sched(request)
    sched._update_request_with_output.return_value = ([42], True)

    engine_core_outputs = OmniARScheduler.update_from_output(sched, *_frame(request))

    (eco,) = engine_core_outputs[request.client_index].outputs
    assert eco.prefill_stats is not None
    assert eco.prefill_stats.num_cached_tokens == 1
    assert eco.prefill_stats.num_cache_creation_tokens == 2
    sched.kv_cache_manager.estimate_cached_tokens.assert_called_once_with(request)


def test_generation_scheduler_finalizes_prefill_stats_before_emitting() -> None:
    request = _make_request()
    request.num_computed_tokens = len(request.prompt_token_ids)
    sched = _make_sched(request)

    engine_core_outputs = OmniGenerationScheduler.update_from_output(sched, *_frame(request))

    (eco,) = engine_core_outputs[request.client_index].outputs
    assert eco.prefill_stats is not None
    assert eco.prefill_stats.num_cached_tokens == 1
    assert eco.prefill_stats.num_cache_creation_tokens == 2
    sched.kv_cache_manager.estimate_cached_tokens.assert_called_once_with(request)
