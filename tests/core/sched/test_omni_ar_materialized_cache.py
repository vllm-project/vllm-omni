# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Materialized AR outputs must publish their completed prefix-cache blocks."""

import pytest

# Import Omni before Request so the request type is the runtime's patched type.
# isort: off
import vllm_omni  # noqa: F401
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus
# isort: on

from tests.core.sched.test_fast_allocate import BLOCK, _manager
from tests.core.sched.test_omni_ar_scheduler_logprobs import _make_scheduler_stub
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler
from vllm_omni.outputs import OmniModelRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(request_id, tokens):
    return Request(
        request_id,
        list(tokens),
        SamplingParams(max_tokens=200, stop_token_ids=[99]),
        None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _prepared(*, trailing_inflight=0, sync_spec=False):
    init_none_hash(sha256)
    manager = _manager(enable_caching=True)
    request = _request("source", range(31))
    request.status = RequestStatus.RUNNING
    # Async allocation can run before the previous sampled token is delivered:
    # reserve two blocks, but the known 31-token sequence hashes only one.
    scheduled = 3 if sync_spec else 1
    computed = 34 if sync_spec else 32 + trailing_inflight
    assert manager.allocate_slots(request, computed) is not None
    request.append_output_token_ids(7)
    request.num_computed_tokens = computed
    request.num_in_flight_tokens = scheduled + trailing_inflight
    request.num_output_placeholders = 0 if sync_spec else 1 + trailing_inflight
    request.prefill_stats = None
    cls = OmniARScheduler if sync_spec else OmniARAsyncScheduler
    stub = _make_scheduler_stub([request])
    scheduler = cls.__new__(cls)
    scheduler.__dict__.update(stub.__dict__)
    scheduler.kv_cache_manager = manager
    scheduler.max_model_len = 256
    scheduler.num_spec_tokens = 2 if sync_spec else 0
    scheduler._process_kv_transfer_trigger = lambda *args: False
    scheduler._handle_stopped_request = lambda *args: True
    before_free = []

    def free_request(req):
        before_free.append(_hit(manager, list(req.all_token_ids)))
        manager.free(req)
        scheduler.requests.pop(req.request_id, None)
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.finished_req_ids_dict[req.client_index].add(req.request_id)
        return None, None

    scheduler._free_request = free_request
    return scheduler, request, manager, before_free


def _hit(manager, tokens):
    query = _request("lookup", tokens)
    return manager.get_computed_blocks(query)[1]


def _frame(*, general=False, token=8, sync_spec=False):
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {"source": 3 if sync_spec else 1}
    scheduler_output.total_num_scheduled_tokens = 3 if sync_spec else 1
    scheduler_output.scheduled_spec_decode_tokens = {"source": [9, 10]} if sync_spec else {}
    runner_output = OmniModelRunnerOutput(
        req_ids=["source"],
        req_id_to_index={"source": 0},
        sampled_token_ids=[[token]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[None] if general else None,
    )
    return scheduler_output, runner_output


@pytest.mark.parametrize("general", [False, True], ids=["fast", "general"])
@pytest.mark.parametrize("trailing_inflight", [0, 1], ids=["settled", "one_step_inflight"])
def test_materialized_async_output_exposes_completed_prefix(mocker, general, trailing_inflight):
    scheduler, request, manager, _ = _prepared(trailing_inflight=trailing_inflight)
    lookup = [*request.all_token_ids, 8]
    assert _hit(manager, lookup) == BLOCK

    publication = mocker.spy(manager, "cache_blocks")
    scheduler.update_from_output(*_frame(general=general))

    publication.assert_called_once_with(request, 2 * BLOCK)
    assert request.num_computed_tokens - request.num_in_flight_tokens == 2 * BLOCK
    assert _hit(manager, lookup) == 2 * BLOCK


@pytest.mark.parametrize("general", [False, True], ids=["fast", "general"])
def test_stop_publishes_completed_prefix_before_free(general):
    scheduler, request, manager, before_free = _prepared()
    lookup = [*request.all_token_ids, 99]
    assert _hit(manager, lookup) == BLOCK

    scheduler.update_from_output(*_frame(general=general, token=99))

    assert before_free == [2 * BLOCK]
    assert _hit(manager, lookup) == 2 * BLOCK


@pytest.mark.parametrize("preempted", [False, True], ids=["discarded", "preempted"])
def test_stale_output_does_not_publish_unconfirmed_prefix(preempted):
    scheduler, request, manager, _ = _prepared()
    request.num_output_placeholders = 0
    request.num_stale_output_tokens = 1
    request.drop_stale_output = not preempted
    if preempted:
        request.status = RequestStatus.PREEMPTED
    lookup = [*request.all_token_ids, 8]
    assert _hit(manager, lookup) == BLOCK

    scheduler.update_from_output(*_frame())

    assert _hit(manager, lookup) == BLOCK


def test_sync_spec_output_exposes_only_verified_prefix():
    scheduler, request, manager, _ = _prepared(sync_spec=True)
    lookup = [*request.all_token_ids, 8]
    assert _hit(manager, lookup) == BLOCK

    scheduler.update_from_output(*_frame(sync_spec=True))

    assert request.num_computed_tokens == 2 * BLOCK
    assert _hit(manager, lookup) == 2 * BLOCK
