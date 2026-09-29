# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-payload completion, failure, preemption and request-ID reuse."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# isort: off
import vllm_omni  # noqa: F401 - install Request patches before importing vLLM
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import Request, RequestStatus
from vllm.config import CacheConfig, DeviceConfig, VllmConfig
from vllm_omni.config.model import OmniModelConfig
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(params=[OmniARScheduler, OmniGenerationScheduler], ids=["ar", "generation"])
def scheduler(request):
    sched = request.param.__new__(request.param)
    sched.vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    sched.requests = {}
    sched.running = []
    sched.waiting = create_request_queue(SchedulingPolicy.FCFS)
    sched.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)
    sched.num_waiting_for_streaming_input = 0
    sched.chunk_transfer_adapter = None
    sched.input_coordinator = None
    sched.connector = None
    sched.ec_connector = None
    sched._inflight_prefills = set()
    sched.encoder_cache_manager = MagicMock()
    sched._free_request_blocks = MagicMock()
    sched.finished_req_ids = set()
    sched.finished_req_ids_dict = None
    sched._new_prompt_len_snapshot = {}
    sched._omits_kv_transfer_cache = {}
    sched._kv_wait_start_ts = {}
    return sched


def add_request(scheduler, req_id, *, waiting=False):
    req = Request(
        request_id=req_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )
    scheduler.requests[req_id] = req
    if waiting:
        scheduler.waiting.add_request(req)
    else:
        req.status = RequestStatus.RUNNING
        scheduler.running.append(req)
    return req


def take_output(scheduler):
    output = SchedulerOutput.make_empty()
    output.finished_req_ids = scheduler.finished_req_ids
    scheduler.finished_req_ids = set()
    return scheduler._wrap_omni_scheduler_output(output)


@pytest.mark.parametrize("waiting", [False, True], ids=["running", "waiting"])
def test_abort_metadata_is_consumed_once(scheduler, waiting):
    add_request(scheduler, "cancel", waiting=waiting)
    add_request(scheduler, "done")
    scheduler.finish_requests("done", RequestStatus.FINISHED_STOPPED)
    scheduler.finish_requests(iter(["cancel", "done", "missing"]), RequestStatus.FINISHED_ABORTED)
    output = take_output(scheduler)
    assert output.finished_req_ids == {"cancel", "done"}
    assert output.discarded_req_ids == {"cancel"}
    assert take_output(scheduler).discarded_req_ids == set()

    add_request(scheduler, "cancel")
    scheduler.finish_requests("cancel", RequestStatus.FINISHED_LENGTH_CAPPED)
    assert take_output(scheduler).discarded_req_ids == set()


class StopBeforeForwardError(Exception):
    pass


@pytest.fixture(params=[GPUARModelRunner, GPUGenerationModelRunner], ids=["ar-worker", "generation-worker"])
def runner(request):
    host = request.param.__new__(request.param)
    model_config = OmniModelConfig.__new__(OmniModelConfig)
    model_config.stage_connector_config = None
    model_config.async_chunk = False
    model_config.worker_type = "ar"
    model_config.custom_process_next_stage_input_func = None
    host.init_omni_connectors(model_config)
    host.requests = {}
    host.execute_model_state = None
    host.routed_experts_initialized = False
    host._warmup_state_cleared = True
    host.omni_prefix_cache = None
    host.kv_transfer_manager = MagicMock()
    host.kv_caches = []
    host.cache_config = CacheConfig(block_size=16)
    host.speculative_config = None
    host.synchronize_input_prep = nullcontext
    host.model_config = model_config
    host._update_states = MagicMock(side_effect=StopBeforeForwardError)
    host.send_full_payload_outputs = MagicMock()
    try:
        yield host
    finally:
        host.shutdown_omni_connectors()


def test_execute_discards_abort_without_flushing_other_pending_requests(scheduler, runner):
    for req_id in ("cancel", "done", "active"):
        req = add_request(scheduler, req_id)
        runner.requests[req_id] = req
        runner.accumulate_full_payload_output(req_id, {"codes": torch.tensor([[1], [2]])}, req)
    scheduler.finish_requests("cancel", RequestStatus.FINISHED_ABORTED)
    scheduler.finish_requests("done", RequestStatus.FINISHED_STOPPED)
    output = take_output(scheduler)

    with pytest.raises(StopBeforeForwardError):
        runner.execute_model(output)
    sent = runner.send_full_payload_outputs.call_args.kwargs["outputs"]
    assert set(sent) == {"done"}
    assert torch.equal(sent["done"][0]["codes"], torch.tensor([[1], [2]]))
    assert set(runner._pending_full_payload_send) == {"active"}

    # Cleanup cannot resurrect the discarded payload; a reused ID starts fresh.
    runner.cleanup_finished_request("cancel")
    runner.send_full_payload_outputs.reset_mock()
    req = add_request(scheduler, "cancel")
    runner.requests["cancel"] = req
    runner.accumulate_full_payload_output("cancel", {"codes": torch.tensor([[9]])}, req)
    scheduler.finish_requests("cancel", RequestStatus.FINISHED_STOPPED)
    with pytest.raises(StopBeforeForwardError):
        runner.execute_model(take_output(scheduler))
    sent = runner.send_full_payload_outputs.call_args.kwargs["outputs"]
    assert set(sent) == {"cancel"}
    assert torch.equal(sent["cancel"][0]["codes"], torch.tensor([[9]]))
    assert set(runner._pending_full_payload_send) == {"active"}


@pytest.mark.parametrize("aborted", [False, True])
def test_cleanup_fallback_discards_only_aborted_payload(runner, aborted):
    req = Request(
        request_id="cancel", prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=8), pooling_params=None
    )
    runner.accumulate_full_payload_output("cancel", {"codes": torch.tensor([[1]])}, req, token_range=(0, 1))
    runner.cleanup_finished_request("cancel", discard_payload=aborted)
    assert not runner._full_payload_token_ends
    assert not runner._pending_full_payload_send
    assert runner.send_full_payload_outputs.call_count == (0 if aborted else 1)


def test_abort_all_ignores_already_completed_requests(scheduler):
    add_request(scheduler, "running")
    add_request(scheduler, "waiting", waiting=True)
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    assert take_output(scheduler).discarded_req_ids == {"running", "waiting"}
    assert take_output(scheduler).discarded_req_ids == set()


def test_update_states_discards_aborted_payload_before_removing_request(scheduler, runner):
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    req = add_request(scheduler, "cancel")
    runner.requests["cancel"] = req
    runner.accumulate_full_payload_output("cancel", {"codes": torch.tensor([[1]])}, req)
    scheduler.finish_requests("cancel", RequestStatus.FINISHED_ABORTED)
    runner.model_intermediate_buffer = {}
    runner.num_prompt_logprobs = {}
    runner.late_interaction_runner = SimpleNamespace(
        on_requests_finished=MagicMock(side_effect=StopBeforeForwardError),
    )
    with pytest.raises(StopBeforeForwardError):
        OmniGPUModelRunner._update_states(runner, take_output(scheduler))
    assert not runner._pending_full_payload_send
    assert not runner.requests
    runner.send_full_payload_outputs.assert_not_called()


@pytest.mark.parametrize(
    "status", [RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR, RequestStatus.FINISHED_IGNORED]
)
def test_failed_request_discards_payload(scheduler, runner, status):
    req = add_request(scheduler, "failed")
    runner.requests["failed"] = req
    runner.accumulate_full_payload_output("failed", {"hidden": torch.ones(3, 2)}, req, token_range=(0, 3))
    # Errors can originate in update_from_output, bypassing finish_requests.
    scheduler.running.remove(req)
    req.status = status
    scheduler._free_request(req)
    with pytest.raises(StopBeforeForwardError):
        runner.execute_model(take_output(scheduler))
    runner.send_full_payload_outputs.assert_not_called()
    assert not runner._pending_full_payload_send
    assert not runner._full_payload_token_ends
    replacement = add_request(scheduler, "failed")
    runner.accumulate_full_payload_output("failed", {"hidden": torch.zeros(1, 2)}, replacement, token_range=(0, 1))
    payload, _ = runner._materialize_full_payload_entry(runner._pending_full_payload_send["failed"])
    assert torch.equal(payload["hidden"], torch.zeros(1, 2))


def test_scheduler_preemption_replay_preserves_rows_and_other_requests(scheduler, runner):
    req = add_request(scheduler, "preempt")
    other = add_request(scheduler, "other")
    runner.requests.update(scheduler.requests)
    original = torch.arange(10).reshape(5, 2)
    runner.accumulate_full_payload_output("preempt", {"hidden": original}, req, token_range=(0, 5))
    runner.accumulate_full_payload_output("other", {"hidden": torch.full((3, 2), 99)}, other, token_range=(0, 3))
    req.append_output_token_ids([4, 5, 6])
    req.num_computed_tokens = 5
    scheduler.log_stats = False
    scheduler.reset_preempted_req_ids = set()
    scheduler.running.remove(req)
    scheduler._preempt_request(req, timestamp=0)
    assert req.status == RequestStatus.PREEMPTED
    assert req.num_computed_tokens == 0
    assert not scheduler.finished_req_ids

    # Recompute in different chunk sizes, crossing the previous high-water mark.
    for start, end in ((0, 2), (2, 4), (4, 7)):
        replay = torch.full((end - start, 2), -1)
        if end == 7:
            replay[-2:] = torch.tensor([[10, 11], [12, 13]])
        runner.accumulate_full_payload_output("preempt", {"hidden": replay}, req, token_range=(start, end))
    payload, _ = runner._materialize_full_payload_entry(runner._pending_full_payload_send["preempt"])
    assert torch.equal(payload["hidden"], torch.arange(14).reshape(7, 2))
    payload, _ = runner._materialize_full_payload_entry(runner._pending_full_payload_send["other"])
    assert torch.equal(payload["hidden"], torch.full((3, 2), 99))

    runner.flush_full_payload_outputs({"preempt"})
    assert "preempt" not in runner._full_payload_token_ends
    runner.cleanup_finished_request("preempt")
    runner.accumulate_full_payload_output("preempt", {"hidden": torch.ones(1, 2)}, req, token_range=(0, 1))
    payload, _ = runner._materialize_full_payload_entry(runner._pending_full_payload_send["preempt"])
    assert payload["hidden"].shape == (1, 2)


def test_replay_preserves_snapshot_metadata_and_handles_recovered_prefix(runner):
    req = None
    runner._full_payload_replace_keys_cached = frozenset({"snapshot"})
    runner.accumulate_full_payload_output(
        "r", {"hidden": torch.ones(4, 2), "snapshot": torch.ones(2, 3), "meta": 4}, req, token_range=(0, 4)
    )
    runner.accumulate_full_payload_output(
        "r", {"hidden": torch.zeros(2, 2), "snapshot": torch.zeros(1, 3), "meta": 2}, req, token_range=(0, 2)
    )
    # Cache recovery emits the whole prefix plus the newly executed suffix.
    runner.accumulate_full_payload_output(
        "r",
        {"hidden": torch.full((6, 2), 2), "snapshot": torch.full((3, 3), 2), "meta": 6},
        req,
        token_range=(4, 6),
        prefix_cache_keys=frozenset({"hidden"}),
    )
    payload, _ = runner._materialize_full_payload_entry(runner._pending_full_payload_send["r"])
    assert torch.equal(payload["hidden"], torch.cat([torch.ones(4, 2), torch.full((2, 2), 2)]))
    assert torch.equal(payload["snapshot"], torch.full((3, 3), 2))
    assert payload["meta"] == 6


def test_ambiguous_partial_replay_fails_without_advancing_positions(runner):
    runner.accumulate_full_payload_output("r", {"audio": torch.ones(20, 2)}, None, token_range=(0, 4))
    runner.accumulate_full_payload_output("r", {"audio": torch.zeros(10, 2)}, None, token_range=(0, 2))
    with pytest.raises(ValueError, match="do not align with token range"):
        runner.accumulate_full_payload_output("r", {"audio": torch.zeros(30, 2)}, None, token_range=(2, 6))
    assert runner._full_payload_token_ends["r"] == {"audio": 4}
    payload, _ = runner._materialize_full_payload_entry(runner._pending_full_payload_send["r"])
    assert torch.equal(payload["audio"], torch.ones(20, 2))


@pytest.mark.parametrize("previous_end", [2, 3], ids=["new-delta", "partial-replay"])
def test_scaled_delta_is_not_inferred_as_cached_prefix(runner, previous_end):
    original = torch.ones(previous_end * 2, 2)
    runner.accumulate_full_payload_output("r", {"audio": original}, None, token_range=(0, previous_end))
    delta = torch.full((4, 2), 2)
    if previous_end == 3:
        with pytest.raises(ValueError, match="do not align with token range"):
            runner.accumulate_full_payload_output("r", {"audio": delta}, None, token_range=(2, 4))
        expected = original
    else:
        runner.accumulate_full_payload_output("r", {"audio": delta}, None, token_range=(2, 4))
        expected = torch.cat([original, delta])
    payload, _ = runner._materialize_full_payload_entry(runner._pending_full_payload_send["r"])
    assert torch.equal(payload["audio"], expected)
    assert runner._full_payload_token_ends["r"]["audio"] == (3 if previous_end == 3 else 4)
