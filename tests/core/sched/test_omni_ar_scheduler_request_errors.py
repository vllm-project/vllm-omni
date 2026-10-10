# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
from vllm.v1.engine import FinishReason
from vllm.v1.request import RequestStatus

from tests.core.sched.test_omni_ar_scheduler_logprobs import (
    _bind_request_lifecycle,
    _make_scheduler_stub,
    _Request,
)
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.outputs import OmniModelRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_terminal_transaction_error_frees_failed_binding_and_preserves_unrelated_request():
    bad, good = _Request("failed"), _Request("other")
    good.sampling_params.num_logprobs = None
    scheduler = _make_scheduler_stub([bad, good])
    updates = []

    def accept(request, ids):
        updates.append(request.request_id)
        return ids, False

    def handle_stopped(request):
        assert request.status == RequestStatus.FINISHED_ERROR
        assert request.resumable is False
        return True

    _bind_request_lifecycle(scheduler, update_request=accept, handle_stopped=handle_stopped)
    # A failure must suppress even accidentally retained token/multimodal
    # content from the partially completed transaction.
    runner = OmniModelRunnerOutput(
        req_ids=["failed", "other"],
        req_id_to_index={"failed": 0, "other": 1},
        sampled_token_ids=[[99], [8]],
        multimodal_outputs=[{"stale": True}, None],
        inter_stage_outputs=[{"stale": True}, None],
        request_errors={"failed": "Lychee merge transaction aborted; rebuild required"},
    )
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"failed": 1, "other": 1}, scheduled_spec_decode_tokens={}, num_invalid_spec_tokens=0
    )
    outputs = OmniARScheduler.update_from_output(scheduler, scheduled, runner)
    by_id = {item.request_id: item for item in outputs[0].outputs}
    assert updates == ["other"]
    assert by_id["failed"].finish_reason == FinishReason.ERROR
    assert by_id["failed"].new_token_ids == []
    assert by_id["failed"].multimodal_output is None
    assert "rebuild required" in by_id["failed"].stop_reason
    assert "failed" not in scheduler.requests
    assert scheduler.requests["other"] is good
    assert good.status == RequestStatus.RUNNING
    assert by_id["other"].new_token_ids == [8]


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_native_capacity_pressure_finishes_victim_without_recompute_and_keeps_other_session(policy):
    from unittest.mock import Mock

    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
    from vllm.v1.core.sched.scheduler import PauseState
    from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler

    bad, good = _Request("capacity-victim"), _Request("other-session")
    for index, request in enumerate((good, bad)):
        request.num_tokens_with_spec = 1
        request.num_tokens = 1
        request.num_prompt_tokens = 1
        request.max_tokens = 10
        request.is_prefill_chunk = False
        request.next_decode_eligible_step = 0
        request.priority = index
        request.arrival_time = index
        request.spec_token_ids = []
        request.mm_features = []
    good.sampling_params.num_logprobs = None
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.__dict__.update(_make_scheduler_stub([good, bad]).__dict__)
    scheduler.policy = SchedulingPolicy(policy)
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.kv_holding_waiting = create_request_queue(scheduler.policy)
    scheduler.vllm_config.model_config.supports_native_preemption = False
    scheduler.vllm_config.speculative_config = None
    scheduler.current_step = 0
    scheduler.scheduler_config = SimpleNamespace(max_num_batched_tokens=2, long_prefill_token_threshold=0)
    scheduler.max_num_scheduled_tokens = 2
    scheduler.max_num_running_reqs = 2
    scheduler.max_num_active_reqs = 2
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.max_model_len = 100
    scheduler.num_sampled_tokens_per_step = 1
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.prefill_capacity_bound = False
    scheduler.adaptive_long_prefill_threshold = False
    scheduler.need_mamba_block_aligned_split = False
    scheduler.num_lookahead_tokens = 0
    scheduler.num_prefill_lookahead = 0
    scheduler.num_spec_tokens = 0
    scheduler.dynamic_sd_lookup = None
    scheduler.defer_block_free = False
    scheduler.log_stats = False
    scheduler.lora_config = None
    scheduler.ec_connector = None
    scheduler.requires_kv_delivery = False
    scheduler.use_v2_model_runner = True
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    scheduler.reset_preempted_req_ids = set()
    scheduler._get_new_block_ids_to_zero = lambda: None
    scheduler._reserve_prefill_lookahead = lambda _request, _computed, count: count
    scheduler._request_blocks_can_be_freed = lambda _request: True
    scheduler.get_request_counts = lambda: (len(scheduler.running), len(scheduler.waiting))
    scheduler._make_cached_request_data = lambda *args: None

    def after_schedule(output):
        for request_id, count in output.num_scheduled_tokens.items():
            scheduler.requests[request_id].num_in_flight_tokens += count

    scheduler._update_after_schedule = after_schedule
    freed = []

    def free(request):
        assert request.status == RequestStatus.FINISHED_ERROR
        assert request.resumable is False
        freed.append(request.request_id)
        scheduler.requests.pop(request.request_id)
        scheduler.finished_req_ids.add(request.request_id)
        scheduler.finished_req_ids_dict[request.client_index].add(request.request_id)
        return None, None

    scheduler._free_request = free
    cache = Mock()
    cache.allocate_slots.side_effect = lambda request, *args, **kwargs: object() if freed else None
    cache.get_num_common_prefix_blocks.return_value = []
    cache.take_boundary_state_offloads.return_value = {}
    cache.take_kv_cache_block_copies.return_value = ([], [])
    scheduler.kv_cache_manager = cache
    scheduler.encoder_cache_manager = Mock()
    scheduler.encoder_cache_manager.get_freed_mm_hashes.return_value = []
    scheduler.encoder_cache_manager.get_manager_metadata.return_value = None
    # Exercise the real fixed-base allocation/preemption decision, not a
    # synthetic direct call to the policy hook. Freeing the victim allows the
    # other session to keep its retained KV and schedule normally.
    scheduled = VLLMScheduler.schedule(scheduler)
    assert scheduled.num_scheduled_tokens == {"other-session": 1}
    assert freed == ["capacity-victim"]
    assert bad.num_computed_tokens == 0
    assert bad.status == RequestStatus.FINISHED_ERROR
    assert bad.resumable is False
    assert list(scheduler.waiting) == []
    assert scheduler.running == [good]
    assert scheduled.finished_req_ids == {"capacity-victim"}
    assert scheduled.preempted_req_ids == set()
    assert cache.allocate_slots.call_count == 2

    # No recompute output is published for the terminal victim. The explicit
    # error is delivered alongside the unaffected session's normal output.
    scheduler._update_request_with_output = lambda request, ids: (ids, False)
    scheduler._process_kv_transfer_trigger = lambda request, ids: False
    runner = OmniModelRunnerOutput(
        req_ids=["other-session"], req_id_to_index={"other-session": 0}, sampled_token_ids=[[8]]
    )
    outputs = scheduler.update_from_output(scheduled, runner)
    by_id = {item.request_id: item for item in outputs[0].outputs}
    assert by_id["capacity-victim"].finish_reason == FinishReason.ERROR
    assert "fresh binding" in by_id["capacity-victim"].stop_reason
    assert by_id["capacity-victim"].new_token_ids == []
    assert by_id["other-session"].new_token_ids == [8]
    assert scheduler.requests["other-session"] is good


def test_stage_without_preemption_rejects_prefix_cache_before_scheduler_startup():
    config = SimpleNamespace(
        model_config=SimpleNamespace(supports_native_preemption=False),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
    )
    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        OmniARScheduler(config)


def test_terminal_error_never_transfers_unsafe_kv():
    request = _Request("failed")
    request.status = RequestStatus.FINISHED_ERROR
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.requests = {"failed": request}
    scheduler._get_omni_kv_config_value = lambda key, default=None: True
    assert scheduler._should_transfer_kv_for_request("failed") is False
