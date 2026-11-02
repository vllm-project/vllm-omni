# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR execution coalescing uses a bounded gate, separate from admission."""

import pytest
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

from tests.core.sched.test_generation_scheduler_atomic_chunk import FakeAdapter, _make_request
from tests.core.sched.test_generation_scheduler_init import _config
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler, OmniARScheduler, VLLMScheduler
from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.outputs import OmniConnectorOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _init_policies(s):
    extra = s._get_scheduler_extra_config()
    s._init_prefill_admission_policy(extra)
    s._init_generation_coalescing(extra)


def scheduler(*, native=False, extras=None, population=4, async_scheduling=False):
    scheduler_cls = OmniARAsyncScheduler if async_scheduling else OmniARScheduler
    s = scheduler_cls.__new__(scheduler_cls)
    s.vllm_config = _config(native=native, extras=extras)
    s.max_num_running_reqs = 128
    s._native_data_plane = native
    s.chunk_transfer_adapter = None if native else FakeAdapter()
    s.input_coordinator = OmniSchedulingCoordinator(stage_id=1, async_chunk=True) if native else None
    s._pause_state = PauseState.UNPAUSED
    s.requests = {str(i): _make_request(str(i), 8) for i in range(population)}
    s.running = []
    s.policy = SchedulingPolicy.FCFS
    s.waiting = create_request_queue(s.policy)
    s.kv_holding_waiting = create_request_queue(s.policy)
    s.deferred_waiting = set()
    s.use_v2_model_runner = native
    s.max_num_active_reqs = 128
    s.max_num_scheduled_tokens = 128
    _init_policies(s)
    return s


def knobs(**overrides):
    return {"generation_min_batch_size": 4, "generation_max_wait_ms": 12, **overrides}


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("policy", ["generation", "admission", "conflict"])
def test_constructor_initializes_both_policies(monkeypatch, mocker, native, policy):
    def upstream_init(self, config):
        self.vllm_config = config
        self.max_num_running_reqs = config.scheduler_config.max_num_seqs
        self.kv_cache_manager = mocker.Mock(spec=KVCacheManager)

    monkeypatch.setattr(VLLMScheduler, "__init__", upstream_init)
    mocker.patch("vllm_omni.core.sched.omni_ar_scheduler.install_decode_allocate_fast_path")
    mocker.patch("vllm_omni.core.sched.omni_scheduler_mixin.OmniChunkTransferAdapter", return_value=FakeAdapter())
    extras = knobs() if policy != "admission" else {}
    if policy != "generation":
        extras.update(ar_prefill_admission_min_batch=4, ar_prefill_admission_max_wait_ms=12)
    config = _config(native=native, extras=extras)
    if policy == "conflict":
        with pytest.raises(ValueError, match="cannot be enabled together"):
            OmniARScheduler(config)
    else:
        s = OmniARScheduler(config)
        assert s._generation_min_batch_size == (4 if policy == "generation" else 1)
        assert s._prefill_admission_min_batch == (4 if policy == "admission" else 0)


@pytest.mark.parametrize("native", [False, True])
def test_configuration_supports_both_runners_and_additional_config(native):
    s = scheduler(native=native)
    s.vllm_config.additional_config = knobs()
    _init_policies(s)
    assert s._generation_min_batch_size == 4
    assert s._generation_max_wait_s == 0.012
    assert s._generation_always_wait_target_batch is False


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("generation_min_batch_size", 0, "max_num_seqs"),
        ("generation_min_batch_size", 129, "max_num_seqs"),
        ("generation_max_wait_ms", -1, "finite and nonnegative"),
        ("generation_max_wait_ms", float("nan"), "finite and nonnegative"),
        ("generation_max_wait_ms", float("inf"), "finite and nonnegative"),
        ("generation_always_wait_target_batch", "false", "boolean"),
        ("generation_coalescing_policy", "idle_wait", "fixed"),
    ],
)
def test_rejects_invalid_configuration(key, value, message):
    with pytest.raises(ValueError, match=message):
        scheduler(extras=knobs(**{key: value}))


@pytest.mark.parametrize("max_wait_ms", [0, 12])
def test_batch_size_upper_bound_applies_even_when_waiting_is_disabled(max_wait_ms):
    with pytest.raises(ValueError, match="max_num_seqs"):
        scheduler(extras=knobs(generation_min_batch_size=129, generation_max_wait_ms=max_wait_ms))
    s = scheduler(extras=knobs(generation_min_batch_size=128, generation_max_wait_ms=max_wait_ms))
    assert s._generation_min_batch_size == 128
    assert s._generation_batch_deadline is None


@pytest.mark.parametrize(
    "old_knobs",
    [
        {"ar_prefill_admission_min_batch": 4, "ar_prefill_admission_max_wait_ms": 12},
        {"ar_prefill_admission_max_wait_ms": 12},
        {"ar_prefill_admission_min_running": 2},
    ],
)
def test_rejects_combining_admission_and_execution_coalescing(old_knobs):
    with pytest.raises(ValueError, match="cannot be enabled together"):
        scheduler(extras=knobs(**old_knobs))


def test_deadline_does_not_restart_on_partial_arrivals(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("vllm_omni.core.sched.omni_ar_scheduler.time.monotonic", lambda: clock[0])
    s = scheduler(extras=knobs())
    assert s._should_defer_ar_generation({"0"})
    clock[0] += 0.006
    assert s._should_defer_ar_generation({"0", "1"})
    assert s._generation_batch_deadline == pytest.approx(10.012)
    clock[0] += 0.006
    assert not s._should_defer_ar_generation({"0", "1"})
    assert s._generation_batch_deadline is None


@pytest.mark.parametrize(
    "always_wait,population,ready,wait",
    [
        (False, 4, 1, True),
        (False, 4, 4, False),
        (False, 3, 1, False),
        (True, 3, 1, True),
        (True, 3, 3, False),
        (True, 4, 0, False),
    ],
)
def test_batch_target_policy(always_wait, population, ready, wait):
    s = scheduler(extras=knobs(generation_always_wait_target_batch=always_wait), population=population)
    assert s._should_defer_ar_generation({str(i) for i in range(ready)}) is wait


@pytest.mark.parametrize("native", [False, True])
def test_candidates_combine_input_prefill_and_continuation_without_duplicates(native):
    s = scheduler(native=native, extras=knobs(), population=7)
    receiver = s.input_coordinator if native else s.chunk_transfer_adapter
    receiver.requests_with_ready_chunks.update({"0", "1", "4", "5", "gone"})
    # The same stream appears in the input set and waiting queue.
    s.waiting.add_request(s.requests["0"])
    for i in (1, 2, 3):
        request = s.requests[str(i)]
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = request.num_prompt_tokens
        s.running.append(request)
    # A normal continuation and an async continuation both have token work.
    s.requests["1"].append_output_token_ids(1)
    s.requests["2"].num_output_placeholders = 1
    s.requests["2"].num_in_flight_tokens = 1
    # Running request 3 has no work; 4 is parked; 5 is finished.
    s.requests["4"].status = RequestStatus.WAITING_FOR_CHUNK
    s.requests["5"].status = RequestStatus.FINISHED_ABORTED
    s.requests["6"].status = RequestStatus.PREEMPTED
    s.kv_holding_waiting.add_request(s.requests["6"])
    assert s._get_ar_coalescing_ready_request_ids() == {"0", "1", "2", "6"}
    assert receiver.requests_with_ready_chunks == {"0", "1", "4", "5", "gone"}


def prepare_schedule_boundary(monkeypatch, s, *, fail=False):
    """Keep the real AR gate; isolate upstream allocation and output packaging."""
    for name in (
        "_process_pending_omni_inputs",
        "_resync_streaming_input_counter",
        "_postprocess_omni_schedule_output",
    ):
        monkeypatch.setattr(s, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(s, "get_finished_requests_needing_kv_transfer", lambda: {})
    monkeypatch.setattr(s, "_wrap_omni_scheduler_output", lambda output, **kwargs: output)
    budgets = []

    def upstream_schedule(self, throttle_prefills=False):
        budgets.append(self.max_num_scheduled_tokens)
        if fail:
            raise RuntimeError("allocation failed")
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = {"finished"}
        if self.max_num_scheduled_tokens:
            output.num_scheduled_tokens = {"0": 8}
            output.total_num_scheduled_tokens = 8
        return output

    monkeypatch.setattr(VLLMScheduler, "schedule", upstream_schedule)
    return budgets


@pytest.mark.parametrize("native", [False, True])
def test_schedule_defers_before_allocation_and_preserves_control_output(monkeypatch, native):
    clock = [10.0]
    monkeypatch.setattr("vllm_omni.core.sched.omni_ar_scheduler.time.monotonic", lambda: clock[0])
    s = scheduler(native=native, extras=knobs())
    s.waiting.add_request(s.requests["0"])
    budgets = prepare_schedule_boundary(monkeypatch, s)
    assert s.schedule().finished_req_ids == {"finished"}
    assert budgets == [0]
    assert s.max_num_scheduled_tokens == 128
    assert s.requests["0"].num_computed_tokens == 0
    assert list(s.waiting) == [s.requests["0"]]
    clock[0] += 0.012
    assert s.schedule().num_scheduled_tokens == {"0": 8}
    assert budgets == [0, 128]


def test_schedule_restores_budget_on_upstream_exception(monkeypatch):
    s = scheduler(extras=knobs())
    s.waiting.add_request(s.requests["0"])
    prepare_schedule_boundary(monkeypatch, s, fail=True)
    with pytest.raises(RuntimeError, match="allocation failed"):
        s.schedule()
    assert s.max_num_scheduled_tokens == 128


@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 2)])
def test_rejects_unsupported_topology(tp, pp):
    s = scheduler()
    s.vllm_config.parallel_config.tensor_parallel_size = tp
    s.vllm_config.parallel_config.pipeline_parallel_size = pp
    s.vllm_config.additional_config = knobs()
    with pytest.raises(ValueError, match="TP1/PP1"):
        _init_policies(s)


@pytest.mark.parametrize("native", [False, True])
def test_unsupported_input_source_disables_waiting(native):
    s = scheduler(native=native)
    s.vllm_config.additional_config = knobs()
    s.chunk_transfer_adapter = None
    s.input_coordinator = OmniSchedulingCoordinator(stage_id=0, async_chunk=True) if native else None
    _init_policies(s)
    assert s._generation_max_wait_s == 0


@pytest.mark.parametrize("pause", [PauseState.PAUSED_ALL, PauseState.PAUSED_NEW])
def test_pause_clears_coalescing_deadline(pause):
    s = scheduler(extras=knobs())
    assert s._should_defer_ar_generation({"0"})
    s._pause_state = pause
    assert not s._should_defer_ar_generation({"0"})
    assert s._generation_batch_deadline is None


def test_empty_candidates_clear_deadline(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("vllm_omni.core.sched.omni_ar_scheduler.time.monotonic", lambda: clock[0])
    s = scheduler(extras=knobs())
    assert s._should_defer_ar_generation({"0"})
    clock[0] += 0.006
    assert not s._should_defer_ar_generation(set())
    assert s._generation_batch_deadline is None
    assert s._should_defer_ar_generation({"1"})
    assert s._generation_batch_deadline == pytest.approx(10.018)


@pytest.mark.parametrize("extras", [None, knobs(generation_min_batch_size=1), knobs(generation_max_wait_ms=0)])
def test_disabled_coalescing_keeps_normal_budget(monkeypatch, extras):
    s = scheduler(extras=extras)
    s.waiting.add_request(s.requests["0"])
    budgets = prepare_schedule_boundary(monkeypatch, s)
    s.schedule()
    assert budgets == [128]


@pytest.mark.parametrize("native", [False, True])
def test_new_ready_requests_release_before_deadline(monkeypatch, native):
    clock = [10.0]
    monkeypatch.setattr("vllm_omni.core.sched.omni_ar_scheduler.time.monotonic", lambda: clock[0])
    s = scheduler(native=native, extras=knobs())
    s.waiting.add_request(s.requests["0"])
    budgets = prepare_schedule_boundary(monkeypatch, s)
    s.schedule()
    clock[0] += 0.003
    for i in range(1, 4):
        s.waiting.add_request(s.requests[str(i)])
    s.schedule()
    assert budgets == [0, 128]
    assert s._generation_batch_deadline is None
    assert clock[0] == 10.003


@pytest.mark.parametrize("native", [False, True])
def test_cancellation_during_wait_releases_remaining_request(monkeypatch, native):
    s = scheduler(native=native, extras=knobs())
    s.waiting.add_request(s.requests["0"])
    s.waiting.add_request(s.requests["1"])
    budgets = prepare_schedule_boundary(monkeypatch, s)
    s.schedule()
    aborted = s.requests.pop("1")
    aborted.status = RequestStatus.FINISHED_ABORTED
    s.schedule()
    assert budgets == [0, 128]
    assert aborted not in s.waiting
    assert s._generation_batch_deadline is None


def test_native_notifications_update_readiness_before_gate(monkeypatch):
    s = scheduler(native=True, extras=knobs())
    s._init_omni_connector_output_inbox()
    s._latest_omni_connector_output = None
    s._outputs_awaiting_stage_payload = {}
    for request in s.requests.values():
        s.waiting.add_request(request)
    process_inputs = s._process_pending_omni_inputs
    budgets = prepare_schedule_boundary(monkeypatch, s)
    monkeypatch.setattr(s, "_process_pending_omni_inputs", process_inputs)
    s.enqueue_omni_connector_output(OmniConnectorOutput(chunk_ready_req_ids={"0"}))
    s.schedule()
    assert budgets == [0]
    assert s.input_coordinator.requests_with_ready_chunks == {"0"}
    assert s.requests["0"].num_computed_tokens == 0
    assert s.requests["1"].status == RequestStatus.WAITING_FOR_CHUNK
    s.enqueue_omni_connector_output(OmniConnectorOutput(chunk_ready_req_ids={"1", "2"}, chunk_finished_req_ids={"3"}))
    s.schedule()
    assert budgets == [0, 128]
    assert s._generation_batch_deadline is None


def test_orchestrator_readiness_override_enables_ar_gate(monkeypatch):
    s = scheduler()
    ready: set[str] = set()
    monkeypatch.setattr(s, "_get_coalescing_ready_request_ids", lambda: ready.copy())
    s.chunk_transfer_adapter = None
    s.vllm_config.additional_config = knobs()
    _init_policies(s)
    assert s._generation_max_wait_s == 0.012
    s.requests["0"].status = RequestStatus.RUNNING
    s.requests["0"].num_computed_tokens = s.requests["0"].num_prompt_tokens
    s.running.append(s.requests["0"])
    budgets = prepare_schedule_boundary(monkeypatch, s)
    ready.add("0")
    s.schedule()
    assert budgets == [0]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_real_upstream_empty_step_preserves_tokens_and_emits_finished_ids(
    monkeypatch, mocker, native, async_scheduling
):
    clock = [10.0]
    monkeypatch.setattr("vllm_omni.core.sched.omni_ar_scheduler.time.monotonic", lambda: clock[0])
    s = scheduler(native=native, extras=knobs(), async_scheduling=async_scheduling)
    s.waiting.add_request(s.requests["0"])
    # Isolate input transport and output packaging, retaining upstream schedule
    # and accounting. Cache allocations must never run on a deferred step.
    monkeypatch.setattr(s, "_process_pending_omni_inputs", lambda model_mode: None)
    monkeypatch.setattr(s, "_postprocess_omni_schedule_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(s, "get_finished_requests_needing_kv_transfer", lambda: {})
    monkeypatch.setattr(s, "_wrap_omni_scheduler_output", lambda output, **kwargs: output)
    s.current_step = 0
    s.vllm_config.speculative_config = None
    s.scheduler_config = s.vllm_config.scheduler_config
    s.scheduler_config.max_num_batched_tokens = 128
    s.scheduler_config.long_prefill_token_threshold = 0
    s.adaptive_long_prefill_threshold = False
    s.prefill_capacity_bound = False
    s.max_num_encoder_input_tokens = 0
    s.num_waiting_for_streaming_input = 0
    s.lora_config = None
    s.prev_step_scheduled_req_ids = set()
    s.finished_req_ids = {"finished"}
    s.reset_preempted_req_ids = set()
    s.num_spec_tokens = 0
    s.dynamic_sd_lookup = None
    s.log_stats = False
    s.defer_block_free = False
    s.connector = s.aux_output_connector = s.ec_connector = None
    s.needs_kv_cache_zeroing = False
    cache = mocker.Mock(spec=KVCacheManager)
    cache.take_boundary_state_offloads.return_value = {}
    cache.take_kv_cache_block_copies.return_value = ([], [])
    cache.take_new_block_ids.return_value = []
    s.kv_cache_manager = cache
    encoder = mocker.Mock(spec=EncoderCacheManager)
    encoder.get_freed_mm_hashes.return_value = []
    encoder.get_manager_metadata.return_value = None
    s.encoder_cache_manager = encoder
    output = s.schedule()
    assert output.num_scheduled_tokens == {}
    assert output.finished_req_ids == {"finished"}
    assert s.finished_req_ids == set()
    cache.allocate_slots.assert_not_called()
    assert s.requests["0"].num_computed_tokens == s.requests["0"].num_in_flight_tokens == 0
    assert s.max_num_scheduled_tokens == 128
    # On expiry the real upstream scheduler resumes allocation/accounting.
    request = s.requests["0"]
    s.waiting.remove_requests([request])
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_prompt_tokens
    request.append_output_token_ids(1)
    s.running.append(request)
    s.max_model_len = 256
    s.num_sampled_tokens_per_step = 1
    s.num_prefill_lookahead = 1
    s.num_lookahead_tokens = 0
    s.need_mamba_block_aligned_split = False
    s.use_pp = False
    s.decode_stagger = 1
    s._inflight_prefills = set()
    cache.allocate_slots.return_value = KVCacheBlocks(blocks=())
    clock[0] += 0.012
    output = s.schedule()
    assert output.num_scheduled_tokens == {"0": 1}
    assert request.num_computed_tokens == request.num_prompt_tokens + 1
    assert request.num_in_flight_tokens == 1
    assert request.num_output_placeholders == int(async_scheduling)
    cache.allocate_slots.assert_called_once()
