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
    sched.deferred_waiting = set()
    sched.kv_holding_waiting = create_request_queue(SchedulingPolicy.FCFS)
    sched.waiting = create_request_queue(SchedulingPolicy.FCFS)
    sched.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)
    sched.num_waiting_for_streaming_input = 0
    sched.chunk_transfer_adapter = None
    sched.input_coordinator = None
    sched._kv_fetch_stages = {}
    sched.connector = None
    sched.ec_connector = None
    sched.aux_output_connector = None
    sched._inflight_prefills = set()
    sched.encoder_cache_manager = MagicMock()
    sched._free_request_blocks = MagicMock()
    sched.finished_req_ids = set()
    sched.finished_req_ids_dict = None
    sched._new_prompt_len_snapshot = {}
    sched._omits_kv_transfer_cache = {}
    sched._kv_wait_start_ts = {}
    sched._outputs_awaiting_stage_payload = {}
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
    host._hs_pending_buffer = {}
    host._hs_saved_boundary = {}
    host._hs_buffer_origin = {}
    host._hs_mm_features = {}
    host.requests = {}
    host.execute_model_state = None
    host.routed_experts_initialized = False
    host._prev_step_idle = False
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
    runner.accumulate_full_payload_output("cancel", {"codes": torch.tensor([[1]])}, req)
    runner.cleanup_finished_request("cancel", discard_payload=aborted)
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
    runner.accumulate_full_payload_output("failed", {"hidden": torch.ones(3, 2)}, req)
    # Errors can originate in update_from_output, bypassing finish_requests.
    scheduler.running.remove(req)
    req.status = status
    scheduler._free_request(req)
    with pytest.raises(StopBeforeForwardError):
        runner.execute_model(take_output(scheduler))
    runner.send_full_payload_outputs.assert_not_called()
    assert not runner._pending_full_payload_send
    replacement = add_request(scheduler, "failed")
    runner.accumulate_full_payload_output("failed", {"hidden": torch.zeros(1, 2)}, replacement)
    payload, _ = runner._materialize_full_payload_entry(runner._pending_full_payload_send["failed"])
    assert torch.equal(payload["hidden"], torch.zeros(1, 2))


def test_qwen3_accumulates_token_rows_and_replaces_fixed_embeddings(runner):
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import thinker2talker_full_payload

    runner._custom_process_func = thinker2talker_full_payload
    constants = {key: torch.ones(1, 1, 4) for key in ("embed.tts_bos", "embed.tts_eos", "embed.tts_pad")}
    runner.accumulate_full_payload_output("r", {"hidden": torch.ones(3, 4), **constants}, None)
    runner.accumulate_full_payload_output("r", {"hidden": torch.zeros(2, 4), **constants}, None)
    runner.flush_full_payload_outputs({"r"})
    payload, _ = runner.send_full_payload_outputs.call_args.kwargs["outputs"]["r"]
    assert torch.equal(payload["hidden"], torch.cat([torch.ones(3, 4), torch.zeros(2, 4)]))
    for key, value in constants.items():
        assert torch.equal(payload[key], value)


def test_resume_trims_dropped_token_tail_and_preserves_non_token_fields(runner):
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import thinker2talker_full_payload

    runner._custom_process_func = thinker2talker_full_payload
    original = torch.arange(12).reshape(3, 4)
    constant = torch.ones(1, 1, 4)
    runner.accumulate_full_payload_output(
        "r", {"hidden": original, "hidden_states.layer_7": original, "audio": torch.ones(6, 2)}, None
    )
    runner.accumulate_full_payload_output(
        "r",
        {
            "hidden": torch.full((2, 4), -1),
            "hidden_states.layer_7": torch.full((2, 4), -1),
            "embed.tts_bos": constant,
            "audio": torch.zeros(4, 2),
            "meta": 5,
        },
        None,
    )
    runner.accumulate_full_payload_output("other", {"hidden": torch.ones(4, 4)}, None)
    assert runner.resume_full_payload_output("r", 3) is None
    runner.accumulate_full_payload_output(
        "r", {"hidden": torch.full((1, 4), 99), "hidden_states.layer_7": torch.full((1, 4), 99)}, None
    )
    runner.flush_full_payload_outputs({"r", "other"})
    sent = runner.send_full_payload_outputs.call_args.kwargs["outputs"]
    expected = torch.cat([original, torch.full((1, 4), 99)])
    assert torch.equal(sent["r"][0]["hidden"], expected)
    assert torch.equal(sent["r"][0]["hidden_states.layer_7"], expected)
    assert torch.equal(sent["r"][0]["embed.tts_bos"], constant)
    assert torch.equal(sent["r"][0]["audio"], torch.cat([torch.ones(6, 2), torch.zeros(4, 2)]))
    assert sent["r"][0]["meta"] == 5
    assert torch.equal(sent["other"][0]["hidden"], torch.ones(4, 4))
