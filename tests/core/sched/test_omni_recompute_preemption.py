# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capacity preemption through the production scheduler and real KV allocator.

Only model configuration, runner output and connector I/O are synthetic. No
schedule, finish, allocation or terminal-output method is replaced.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CacheConfig, SchedulerConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_omni.core.sched import omni_ar_scheduler
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.preemption_reserve import PreemptionReserve

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler(
    mode="fail",
    requires_kv_delivery=False,
    auxiliary=False,
    block_size=16,
    num_blocks=2,
    max_model_len=64,
    max_num_seqs=2,
    max_num_batched_tokens=64,
    enable_prefix_caching=False,
    sliding_window=None,
):
    config = VllmConfig()
    # Avoid downloading model weights/config; all scheduler/cache objects are real.
    config.model_config = SimpleNamespace(
        multimodal_config=None,
        uses_mrope=False,
        is_encoder_decoder=False,
        max_model_len=max_model_len,
        runner_type="generate",
        is_diffusion=False,
        supports_multimodal_inputs=False,
        return_sampling_mask=False,
        recompute_preemption=mode,
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        is_encoder_decoder=False,
    )
    config.aux_output_config.enable_return_routed_experts = auxiliary
    config.cache_config = CacheConfig(block_size=block_size, enable_prefix_caching=enable_prefix_caching)
    # By default: one null block and one usable block, enough only for prefill.
    config.cache_config.num_gpu_blocks = num_blocks
    # Match EngineCore initialization before constructing the scheduler. Custom
    # specs registered during test collection can suppress lazy built-in registration.
    register_all_kvcache_specs(config)
    spec = dict(block_size=block_size, num_kv_heads=1, head_size=8, dtype=torch.float16)
    kv_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(**spec)
                if sliding_window is None
                else SlidingWindowSpec(**spec, sliding_window=sliding_window),
            )
        ],
    )
    scheduler = OmniARScheduler(config, kv_config, Mock(spec=StructuredOutputManager), block_size)
    if requires_kv_delivery:
        connector = Mock(spec=KVConnectorBase_V1)
        connector.requires_kv_delivery = True
        connector.has_pending_block_frees.return_value = False
        connector.get_num_new_matched_tokens.return_value = (0, False)
        connector.request_finished.return_value = (False, None)
        connector.build_connector_meta.return_value = None
        connector.take_events.return_value = ()
        connector.get_kv_connector_stats.return_value = None
        scheduler.connector = connector
        scheduler.requires_kv_delivery = connector.requires_kv_delivery
    return scheduler


def _output(request=None):
    ids = [] if request is None else [request.request_id]
    return ModelRunnerOutput(
        req_ids=ids,
        # req_id_to_index was dropped upstream; the pin derives it from req_ids.
        sampled_token_ids=[] if request is None else [[1]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _prefill(scheduler, prompt_len=16):
    request = Request("req-preempt", list(range(prompt_len)), SamplingParams(max_tokens=32), None, client_index=3)
    scheduler.add_request(request)
    scheduled = scheduler.schedule()
    assert scheduled.num_scheduled_tokens == {request.request_id: prompt_len}
    scheduler.update_from_output(scheduled, _output(request))
    assert request.status == RequestStatus.RUNNING
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    return request


@pytest.mark.parametrize("requires_kv_delivery", [False, True])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_capacity_preemption_emits_one_error_and_frees_state(requires_kv_delivery, auxiliary):
    scheduler = _scheduler(requires_kv_delivery=requires_kv_delivery, auxiliary=auxiliary)
    request = _prefill(scheduler)
    scheduled = scheduler.schedule()
    outputs = scheduler.update_from_output(scheduled, _output())

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert not scheduler.running and not scheduler.waiting and not scheduler.kv_holding_waiting
    assert request not in scheduler.deferred_waiting
    assert request not in scheduler._inflight_prefills
    assert not any(scheduler.kv_cache_manager.get_block_ids(request.request_id))
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 1
    assert scheduled.finished_req_ids == {request.request_id}
    assert scheduled.num_scheduled_tokens == {}
    [terminal] = outputs[3].outputs
    assert terminal.finish_reason == FinishReason.ERROR
    assert terminal.new_token_ids == []
    assert "cannot replay" in terminal.stop_reason and "truncated" in terminal.stop_reason
    assert outputs[3].finished_requests == {request.request_id}
    if requires_kv_delivery:
        scheduler.connector.request_finished.assert_called_once()

    if auxiliary:
        assert request.request_id not in scheduler.aux_output_connector._sent_hash_counts
        assert scheduled.aux_output_connector_metadata.finished_requests == (request.request_id,)

    # A late repeat and subsequent empty engine step must not emit ERROR/ABORT twice.
    scheduler._preempt_request(request, 1.0)
    next_step = scheduler.schedule()
    if auxiliary:
        assert next_step.aux_output_connector_metadata.finished_requests == ()
    assert scheduler.update_from_output(next_step, _output()) == {}


@pytest.mark.parametrize("requires_kv_delivery", [False, True])
def test_allow_mode_capacity_preemption_requeues(requires_kv_delivery):
    scheduler = _scheduler("allow", requires_kv_delivery)
    request = _prefill(scheduler)
    scheduled = scheduler.schedule()
    assert request.status == RequestStatus.PREEMPTED
    assert request in scheduler.waiting
    assert request.num_computed_tokens == 0
    assert request.drop_stale_output is requires_kv_delivery
    assert scheduler.update_from_output(scheduled, _output()) == {}


@pytest.mark.parametrize("requires_kv_delivery", [False, True])
def test_explicit_prefix_cache_reset_preserves_resume(requires_kv_delivery):
    scheduler = _scheduler(requires_kv_delivery=requires_kv_delivery)
    # The prompt plus its first sampled token must still fit for reset replay.
    request = _prefill(scheduler, prompt_len=15)
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert request.status == RequestStatus.PREEMPTED
    assert request in scheduler.waiting
    assert request.drop_stale_output
    assert not scheduler._resetting_prefix_cache
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 1
    # The resumed prefill fills the only block, then capacity must fail again:
    # the reset exception cannot leak into a subsequent scheduler iteration.
    scheduled = scheduler.schedule()
    scheduler.update_from_output(scheduled, _output(request))
    outputs = scheduler.update_from_output(scheduler.schedule(), _output())
    [terminal] = outputs[3].outputs
    assert terminal.finish_reason == FinishReason.ERROR


def test_prefix_cache_reset_scope_is_cleared_on_error(monkeypatch):
    scheduler = _scheduler()
    _prefill(scheduler)
    monkeypatch.setattr(scheduler.kv_cache_manager, "reset_prefix_cache", lambda: False)
    with pytest.raises(RuntimeError, match="Failed to reset KV cache"):
        scheduler.reset_prefix_cache(reset_running_requests=True)
    assert not scheduler._resetting_prefix_cache


# --- Admission reserve for stages that cannot replay (vllm_omni/core/sched/preemption_reserve.py)


def _reserve_scheduler(monkeypatch, reserve=True, **kwargs):
    if not reserve:  # control: the same scheduler without the admission reserve
        monkeypatch.setattr(omni_ar_scheduler, "install_preemption_reserve", lambda *args: None)
    kwargs.setdefault("max_model_len", 256)
    kwargs.setdefault("max_num_seqs", 8)
    kwargs.setdefault("max_num_batched_tokens", 256)
    return _scheduler(**kwargs)


def _request(rid, prompt_len, max_tokens, block_hasher=None, first_token=2):
    request = Request(
        rid,
        list(range(first_token, first_token + prompt_len)),
        SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        None,
        block_hasher=block_hasher,
    )
    request.additional_information = None
    return request


def _run(scheduler, requests, steps=200):
    """Drive schedule / update_from_output, sampling one token once a request's prompt is computed."""
    for request in requests:
        scheduler.add_request(request)
    for _ in range(steps):
        _step(scheduler, scheduler.schedule())
        if not scheduler.requests:
            return
    raise AssertionError("requests did not finish")


def _step(scheduler, scheduled):
    """Apply one step's output: one sampled token per request whose prompt is computed."""
    ids = list(scheduled.num_scheduled_tokens)
    sampled = [
        [1] if scheduler.requests[rid].num_computed_tokens >= scheduler.requests[rid].num_tokens else [] for rid in ids
    ]
    scheduler.update_from_output(
        scheduled,
        ModelRunnerOutput(
            req_ids=ids,
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


@pytest.mark.parametrize("reserve", [True, False])
def test_reserve_queues_requests_that_would_be_preempted(monkeypatch, reserve):
    # 5 usable blocks; each request needs 1 block of prompt and 2 of growth.
    scheduler = _reserve_scheduler(monkeypatch, reserve, num_blocks=6)
    requests = [_request(f"r{i}", 16, 32) for i in range(3)]
    _run(scheduler, requests)
    statuses = [r.status for r in requests]
    if reserve:
        assert statuses == [RequestStatus.FINISHED_LENGTH_CAPPED] * 3
        assert not scheduler._preemption_reserve.admitted
    else:
        assert RequestStatus.FINISHED_ERROR in statuses


@pytest.mark.parametrize("reserve", [True, False])
def test_reserve_covers_the_rest_of_a_chunked_prompt(monkeypatch, reserve):
    # 48-token prompts in 32-token chunks; 6 usable blocks, 4 per request at completion.
    scheduler = _reserve_scheduler(monkeypatch, reserve, num_blocks=7, max_num_batched_tokens=32)
    requests = [_request("a", 48, 16), _request("b", 48, 16)]
    _run(scheduler, requests)
    statuses = [r.status for r in requests]
    if reserve:
        assert statuses == [RequestStatus.FINISHED_LENGTH_CAPPED] * 2
    else:
        assert RequestStatus.FINISHED_ERROR in statuses


def test_reserve_admits_requests_that_fit_together_in_one_step(monkeypatch):
    # 10 usable blocks; each request needs 4 blocks of prompt and 1 of growth, so both fit at once.
    # Blocks allocated earlier in the step must not be counted again as still needed.
    scheduler = _reserve_scheduler(monkeypatch, num_blocks=11)
    for request in (_request("a", 64, 16), _request("b", 64, 16)):
        scheduler.add_request(request)
    assert set(scheduler.schedule().num_scheduled_tokens) == {"a", "b"}


def test_reserve_counts_held_blocks_and_adds_the_callers_reservation(monkeypatch):
    scheduler = _reserve_scheduler(monkeypatch, num_blocks=101, max_model_len=512, max_num_batched_tokens=512)
    manager, reserve = scheduler.kv_cache_manager, scheduler._preemption_reserve
    a = _request("a", 160, 320)
    assert manager.allocate_slots(a, 160) is not None  # empty stage: admitted without a reserve
    # a: 480 tokens at completion = 30 blocks, holds 10.
    assert reserve.reserved_blocks() == 20
    assert manager.block_pool.get_num_free_blocks() == 90
    # b needs 10 blocks now; the reserve is 20 for a plus 20 for the rest of b.
    assert manager.allocate_slots(_request("b", 160, 320), 160, reserved_blocks=41) is None
    assert manager.allocate_slots(_request("b", 160, 320), 160, reserved_blocks=40) is not None


def test_reserve_learns_growth_from_normal_finishes_and_clears_on_abort(monkeypatch):
    scheduler = _reserve_scheduler(monkeypatch, num_blocks=12)
    reserve = scheduler._preemption_reserve
    a, b = _request("a", 16, 32), _request("b", 16, 32)
    scheduler.add_request(a)
    scheduler.add_request(b)
    scheduler.schedule()
    assert set(reserve.admitted) == {"a", "b"}
    scheduler.finish_requests(["a", "b"], RequestStatus.FINISHED_ABORTED)
    assert not reserve.admitted and reserve.max_growth is None  # aborts are not growth samples
    _run(scheduler, [_request("c", 16, 8)])
    assert reserve.max_growth == 8
    assert reserve.growth(_request("d", 16, 4)) == 4  # capped by max_tokens


def test_reserve_growth_beyond_the_estimate_can_still_be_preempted(monkeypatch):
    # The estimate is learned: after a 2-token request, 48-token requests outgrow their reserve.
    scheduler = _reserve_scheduler(monkeypatch, num_blocks=6)
    reserve = scheduler._preemption_reserve
    _run(scheduler, [_request("short", 16, 2)])
    assert reserve.max_growth == 2
    requests = [_request(f"l{i}", 16, 48) for i in range(3)]
    _run(scheduler, requests)
    assert RequestStatus.FINISHED_ERROR in [r.status for r in requests]
    assert not reserve.admitted  # the failed request is released through _free_request too


def test_reserve_counts_prefix_cache_hits_as_covered(monkeypatch):
    init_none_hash(sha256)
    hasher = get_request_block_hasher(16, sha256)
    scheduler = _reserve_scheduler(monkeypatch, num_blocks=12, enable_prefix_caching=True)
    reserve = scheduler._preemption_reserve
    _run(scheduler, [_request("warm", 48, 1, hasher)])
    requests = [_request("a", 48, 16, hasher), _request("b", 48, 16, hasher)]
    for request in requests:
        scheduler.add_request(request)
    scheduled = scheduler.schedule()
    assert scheduled.num_scheduled_tokens == {"a": 16, "b": 16}  # 32 of 48 prompt tokens hit the cache
    _step(scheduler, scheduled)
    _run(scheduler, [])
    assert [r.status for r in requests] == [RequestStatus.FINISHED_LENGTH_CAPPED] * 2
    assert not reserve.admitted


def test_reserve_passes_resumable_sessions_through_untracked(monkeypatch):
    scheduler = _reserve_scheduler(monkeypatch, num_blocks=101, max_model_len=512, max_num_batched_tokens=512)
    manager, reserve = scheduler.kv_cache_manager, scheduler._preemption_reserve
    manager.allocate_slots(_request("a", 160, 320), 160)
    session = _request("s", 160, 1)
    session.resumable = True
    # 90 free - 10 for s: only fits without a reserve for it.
    assert manager.allocate_slots(session, 160, reserved_blocks=80) is not None
    assert "s" not in reserve.admitted


@pytest.mark.parametrize("num_blocks, admitted", [(25, True), (24, False)])
def test_reserve_counts_an_in_flight_prefill_once_with_an_async_kv_load(monkeypatch, num_blocks, admitted):
    # A is in chunked prefill (160-token prompt, 16 tokens per step); B then loads 32 of its 48 tokens
    # asynchronously. When B is considered, A holds 2 blocks. The scheduler reserves A's unfinished prompt
    # (10 - 2 = 8 blocks); the admission reserve expects A to need 20 - 2 = 18, so only 10 more are added for
    # A, plus 2 for B's own growth. B needs 2 blocks for its loaded tokens: 8 + 10 + 2 + 2 = 22 free, i.e. 25
    # blocks with the null block and A's 2. Adding both reservations for A would need 30 free (33 blocks).
    scheduler = _reserve_scheduler(monkeypatch, num_blocks=num_blocks, max_model_len=512, max_num_seqs=4)
    scheduler.scheduler_config.long_prefill_token_threshold = 16  # applies while B is also waiting
    lookups = []

    def matched_tokens(request, _):
        if request.request_id != "b":
            return 0, False
        lookups.append(request.request_id)
        return (None, False) if len(lookups) == 1 else (32, True)  # B's lookup resolves in the second step

    scheduler.connector = _async_load_connector(matched_tokens)
    a, b = _request("a", 160, 160), _request("b", 48, 16, first_token=1000)
    scheduler.add_request(a)
    scheduler.add_request(b)
    _step(scheduler, scheduler.schedule())
    assert a.num_computed_tokens < a.num_prompt_tokens and b.status == RequestStatus.WAITING
    scheduler.schedule()
    assert b.status == (RequestStatus.WAITING_FOR_REMOTE_KVS if admitted else RequestStatus.WAITING)


def _async_load_connector(matched_tokens):
    connector = Mock(spec=KVConnectorBase_V1)
    connector.requires_kv_delivery = False
    connector.has_pending_block_frees.return_value = False
    connector.get_num_new_matched_tokens.side_effect = matched_tokens
    connector.get_loaded_kv_cache_group_ids.return_value = ()
    connector.request_finished.return_value = (False, None)
    connector.build_connector_meta.return_value = None
    connector.take_events.return_value = ()
    connector.get_kv_connector_stats.return_value = None
    return connector


@pytest.mark.parametrize("num_blocks, admitted", [(25, True), (24, False)])
def test_reserve_counts_a_parked_async_kv_load_once(monkeypatch, num_blocks, admitted):
    # A and B both load asynchronously. A (160-token prompt, 32 tokens loaded) is parked first and holds
    # 2 blocks; the scheduler caches its reservation as parked (10 - 2 = 8 blocks), not as an in-flight
    # prefill. The admission reserve expects A to need 20 - 2 = 18, so only 10 more are added for A, plus 2
    # for B's own growth. B needs 2 blocks for its loaded tokens: 8 + 10 + 2 + 2 = 22 free, i.e. 25 blocks
    # with the null block and A's 2. Adding both reservations for A would need 30 free (33 blocks).
    scheduler = _reserve_scheduler(monkeypatch, num_blocks=num_blocks, max_model_len=512, max_num_seqs=4)
    loaded = {"a": 32, "b": 32}
    scheduler.connector = _async_load_connector(lambda request, _: (loaded[request.request_id], True))
    a, b = _request("a", 160, 160), _request("b", 48, 16, first_token=1000)
    scheduler.add_request(a)
    scheduler.add_request(b)
    scheduler.schedule()
    assert a.status == RequestStatus.WAITING_FOR_REMOTE_KVS and scheduler._parked_reservations[a] == 8
    assert b.status == (RequestStatus.WAITING_FOR_REMOTE_KVS if admitted else RequestStatus.WAITING)


@pytest.mark.parametrize(
    "scheduler_blocks, caller_blocks, added",
    [
        (8, 9, 10),  # the caller covers 8 of the 18 blocks A needs: add the other 10
        (20, 21, 0),  # the caller covers more than A needs (speculative-decode slots): add nothing
        (8, 3, 15),  # a caller reservation smaller than its in-flight terms is never undercut
        (0, 5, 18),  # A is not one of the caller's in-flight requests
    ],
)
def test_reserve_adds_only_what_the_callers_reservation_does_not_cover(scheduler_blocks, caller_blocks, added):
    # A (160-token prompt, max_tokens 160) is expected to hold 20 blocks and holds 2, so it needs 18.
    reserve = PreemptionReserve(16, 512, {"a": [None, None]}, lambda request: scheduler_blocks)
    reserve.admitted["a"] = (_request("a", 160, 160), 160)
    assert reserve.reserved_blocks(caller_blocks) == added
    assert reserve.reserved_blocks() == 18


@pytest.mark.parametrize("mode, sliding_window", [("allow", None), ("fail", 32)])
def test_reserve_is_installed_only_for_fail_mode_full_attention(mode, sliding_window):
    scheduler = _scheduler(mode, num_blocks=6, sliding_window=sliding_window)
    assert scheduler._preemption_reserve is None
