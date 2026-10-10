# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Admission must align collective schedules across sharded-weight ranks."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.executor.ray_executor import RayDiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import DiffusionRequestStatus, RequestScheduler
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.sched.request_scheduler import build_request_batch_sampling_params_key
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(params=["text", "empty_text", "embeddings"])
def negative_inputs(request):
    if request.param == "text":
        return {"negative_prompt": "blurry"}
    if request.param == "empty_text":
        return {"negative_prompt": ""}
    return {
        "negative_prompt_embeds": torch.zeros(1, 2, 4),
        "negative_prompt_embeds_mask": torch.ones(1, 2, dtype=torch.bool),
    }


def _make_request(request_id, negative_inputs=None, true_cfg_scale=4.0):
    return OmniDiffusionRequest(
        request_id=request_id,
        prompt={"prompt": f"prompt {request_id}", **(negative_inputs or {})},
        sampling_params=OmniDiffusionSamplingParams(
            guidance_scale=1.0,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=2,
        ),
    )


def _make_wave(*requests):
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[NewRequestData(request_id=req.request_id, req=req) for req in requests],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=len(requests),
        num_waiting_reqs=0,
    )


@pytest.fixture(params=[MultiprocDiffusionExecutor, RayDiffusionExecutor], ids=["multiproc", "ray"])
def executor(request):
    executor = object.__new__(request.param)
    executor._closed = False
    executor._is_failed = False
    executor._result_mq = Mock()
    executor._broadcast_mq = Mock()
    executor.collective_rpc = Mock()
    return executor


def _make_wave_config(mode):
    config = SimpleNamespace(
        max_num_seqs=2,
        step_execution=False,
        parallel_config=SimpleNamespace(data_parallel_size=2, hsdp_data_parallel=mode == "hsdp"),
    )
    if mode != "hsdp":
        component = "text_encoder" if mode == "dlo_text_encoder" else "dit"
        config.diffusion_offload_config = {
            "mode": "layer",
            "components": [component],
            "layer_options": {component: {"weight_transfer": "allgather"}},
        }
    return config


@pytest.fixture(params=["hsdp", "dlo", "dlo_text_encoder"])
def wave_config(request):
    return _make_wave_config(request.param)


@pytest.mark.parametrize("true_cfg_scale", [None, 4.0])
def test_negative_inputs_change_qwen_cfg_admission_key(negative_inputs, true_cfg_scale):
    positive = _make_request("positive", true_cfg_scale=true_cfg_scale)
    guided = _make_request("guided", negative_inputs, true_cfg_scale)

    assert positive.sampling_params.do_classifier_free_guidance is False
    assert guided.sampling_params.do_classifier_free_guidance is False
    assert build_request_batch_sampling_params_key(positive) != build_request_batch_sampling_params_key(guided)


@pytest.mark.parametrize("true_cfg_scale", [None, 4.0])
def test_scheduler_separates_mixed_qwen_cfg_requests(negative_inputs, true_cfg_scale):
    scheduler = RequestScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2))
    scheduler.add_request(_make_request("positive", true_cfg_scale=true_cfg_scale))
    scheduler.add_request(_make_request("guided", negative_inputs, true_cfg_scale))

    first = scheduler.schedule()

    assert first.scheduled_request_ids == ["positive"]
    assert first.num_waiting_reqs == 1
    scheduler.update_from_output(
        first,
        RunnerOutput(request_id="positive", finished=True, result=DiffusionOutput(output="done")),
    )
    second = scheduler.schedule()
    assert second.scheduled_request_ids == ["guided"]


@pytest.mark.parametrize("true_cfg_scale", [None, 4.0])
def test_executors_reject_mixed_qwen_cfg_wave(executor, wave_config, negative_inputs, true_cfg_scale):
    executor.od_config = wave_config
    wave = _make_wave(
        _make_request("positive", true_cfg_scale=true_cfg_scale),
        _make_request("guided", negative_inputs, true_cfg_scale),
    )

    with pytest.raises(ValueError, match="compatible shape, CFG"):
        executor.execute_request(wave)

    executor.collective_rpc.assert_not_called()


def test_matching_negative_inputs_run_after_rejected_wave(executor, wave_config, negative_inputs):
    executor.od_config = wave_config
    invalid = _make_wave(_make_request("positive"), _make_request("guided", negative_inputs))
    with pytest.raises(ValueError, match="compatible shape, CFG"):
        executor.execute_request(invalid)
    executor.collective_rpc.assert_not_called()

    wave = _make_wave(_make_request("first", negative_inputs), _make_request("second", negative_inputs))
    results = [DiffusionOutput(output="first"), DiffusionOutput(output="second")]
    executor.collective_rpc.return_value = (
        [{"dp_rank": rank, "output": out} for rank, out in enumerate(results)]
        if isinstance(executor, RayDiffusionExecutor)
        else results
    )

    output = executor.execute_request(wave)

    assert [item.result for item in output.runner_outputs] == results
    executor.collective_rpc.assert_called_once()


def test_negative_prompt_contents_are_request_local():
    first = _make_request("first", {"negative_prompt": "blurry"})
    second = _make_request("second", {"negative_prompt": "grainy"})

    assert build_request_batch_sampling_params_key(first) == build_request_batch_sampling_params_key(second)


def test_absent_and_none_negative_inputs_are_compatible():
    absent = _make_request("absent")
    explicit_none = _make_request(
        "none",
        {"negative_prompt": None, "negative_prompt_embeds": None, "negative_prompt_embeds_mask": None},
    )

    assert build_request_batch_sampling_params_key(absent) == build_request_batch_sampling_params_key(explicit_none)


def test_negative_pooled_embeds_change_admission_key():
    # Flux and HiDream require pooled negative embeddings to enable true CFG.
    embeds = {"negative_prompt_embeds": torch.zeros(1, 2, 4)}
    without_pooled = _make_request("without_pooled", embeds)
    with_pooled = _make_request(
        "with_pooled",
        {**embeds, "negative_pooled_prompt_embeds": torch.zeros(1, 4)},
    )

    assert build_request_batch_sampling_params_key(without_pooled) != build_request_batch_sampling_params_key(
        with_pooled
    )


@pytest.mark.parametrize(
    ("first_extra_args", "second_extra_args"),
    [
        ({"use_resolution_template": True}, {"use_resolution_template": False}),
        ({"custom": {"enabled": True}}, {"custom": {"enabled": False}}),
        (None, {}),
    ],
)
def test_dp_extra_args_mismatch_runs_in_separate_waves(executor, wave_config, first_extra_args, second_extra_args):
    wave_config.max_num_seqs = 2
    executor.od_config = wave_config
    scheduler = RequestScheduler()
    scheduler.initialize(wave_config)
    first = _make_request("first")
    second = _make_request("second")
    first.sampling_params.extra_args = first_extra_args
    second.sampling_params.extra_args = second_extra_args
    scheduler.add_request(first)
    scheduler.add_request(second)

    # These requests differ only in fields outside the ordinary batching key.
    assert build_request_batch_sampling_params_key(first) == build_request_batch_sampling_params_key(second)
    with pytest.raises(ValueError, match="extra_args"):
        executor.execute_request(_make_wave(first, second))
    executor.collective_rpc.assert_not_called()

    for request_id, waiting_count in (("first", 1), ("second", 0)):
        wave = scheduler.schedule()
        assert wave.scheduled_request_ids == [request_id]
        assert wave.num_waiting_reqs == waiting_count
        if waiting_count:
            assert scheduler.get_request_state("second").status == DiffusionRequestStatus.WAITING

        executor.collective_rpc.return_value = DiffusionOutput(output=request_id)
        output = executor.execute_batch(wave)
        assert scheduler.update_from_output(wave, output) == {request_id}
        assert scheduler.get_request_state(request_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        scheduler.pop_request_state(request_id)

    assert executor.collective_rpc.call_count == 2


@pytest.mark.parametrize(
    ("first_extra_args", "second_extra_args"),
    [
        ({"use_resolution_template": True}, {"use_resolution_template": True}),
        (
            {"use_resolution_template": True, "custom": {"a": 1, "b": 2}},
            {"custom": {"b": 2, "a": 1}, "use_resolution_template": True},
        ),
        (None, None),
        ({}, {}),
    ],
)
def test_dp_matching_extra_args_share_wave(executor, wave_config, first_extra_args, second_extra_args):
    wave_config.max_num_seqs = 2
    executor.od_config = wave_config
    scheduler = RequestScheduler()
    scheduler.initialize(wave_config)
    for request_id, extra_args in (("first", first_extra_args), ("second", second_extra_args)):
        request = _make_request(request_id)
        request.sampling_params.extra_args = extra_args
        scheduler.add_request(request)

    wave = scheduler.schedule()
    assert wave.scheduled_request_ids == ["first", "second"]
    assert wave.num_waiting_reqs == 0
    results = [DiffusionOutput(output="first"), DiffusionOutput(output="second")]
    executor.collective_rpc.return_value = (
        [{"dp_rank": rank, "output": out} for rank, out in enumerate(results)]
        if isinstance(executor, RayDiffusionExecutor)
        else results
    )

    output = executor.execute_batch(wave)
    assert scheduler.update_from_output(wave, output) == {"first", "second"}
    for request_id in ("first", "second"):
        assert scheduler.get_request_state(request_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
    executor.collective_rpc.assert_called_once()


def test_ordinary_batching_keeps_extra_args_request_local():
    scheduler = RequestScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2, parallel_config=SimpleNamespace(data_parallel_size=2)))
    for request_id, enabled in (("first", True), ("second", False)):
        request = _make_request(request_id)
        request.sampling_params.extra_args = {"use_resolution_template": enabled}
        scheduler.add_request(request)

    assert scheduler.schedule().scheduled_request_ids == ["first", "second"]


@pytest.fixture(params=["tuple_key", "mixed_keys", "nested_tuple_key", "circular_value"])
def unserializable_extra_args(request):
    if request.param == "tuple_key":
        return {("a", "b"): 1}, TypeError, "keys must be"
    if request.param == "mixed_keys":
        return {1: "x", "a": "y"}, TypeError, "not supported between"
    if request.param == "nested_tuple_key":
        return {"custom": {("a", "b"): 1}}, TypeError, "keys must be"
    circular = {}
    circular["self"] = circular
    return circular, ValueError, "Circular reference"


def test_dp_invalid_extra_args_only_fail_submission(executor, wave_config, unserializable_extra_args):
    wave_config.max_num_seqs = 2
    executor.od_config = wave_config
    scheduler = RequestScheduler()
    scheduler.initialize(wave_config)
    engine = object.__new__(DiffusionEngine)
    engine.scheduler = scheduler
    engine.executor = executor
    engine._cv = threading.Condition()
    engine._closed = False
    engine._out_streams = {}
    engine._request_cancellations = Mock()

    engine._add_prepared_request(_make_request("first"))
    invalid = _make_request("invalid")
    invalid.sampling_params.extra_args, error_type, message = unserializable_extra_args

    with pytest.raises(error_type, match=message):
        engine._add_prepared_request(invalid)

    assert not engine._closed
    assert scheduler.get_request_state("invalid") is None
    assert scheduler.num_waiting_requests() == 1
    assert scheduler.num_running_requests() == 0
    assert set(engine._out_streams) == {"first"}
    engine._request_cancellations.finish.assert_called_once_with("invalid")
    assert invalid.cancellation_signal is None
    executor.collective_rpc.assert_not_called()

    # The engine can admit more requests and execute the existing queue.
    engine._add_prepared_request(_make_request("later"))
    wave = scheduler.schedule()
    assert wave.scheduled_request_ids == ["first", "later"]
    assert wave.num_waiting_reqs == 0
    results = [DiffusionOutput(output="first"), DiffusionOutput(output="later")]
    executor.collective_rpc.return_value = (
        [{"dp_rank": rank, "output": out} for rank, out in enumerate(results)]
        if isinstance(executor, RayDiffusionExecutor)
        else results
    )

    output = executor.execute_batch(wave)
    assert scheduler.update_from_output(wave, output) == {"first", "later"}
    for request_id in ("first", "later"):
        assert scheduler.get_request_state(request_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
    executor.collective_rpc.assert_called_once()


def test_ordinary_batching_does_not_serialize_extra_args(unserializable_extra_args):
    scheduler = RequestScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2, parallel_config=SimpleNamespace(data_parallel_size=2)))
    for request_id in ("first", "second"):
        request = _make_request(request_id)
        request.sampling_params.extra_args = unserializable_extra_args[0]
        scheduler.add_request(request)

    assert scheduler.schedule().scheduled_request_ids == ["first", "second"]


def _make_encoded_request(request_id, *, positive_embeds, negative_embeds=False):
    request = _make_request(request_id)
    if positive_embeds:
        request.prompt = {
            "prompt_embeds": torch.zeros(1, 2, 4),
            "prompt_embeds_mask": torch.ones(1, 2, dtype=torch.bool),
        }
    if negative_embeds:
        request.prompt.update(
            negative_prompt_embeds=torch.zeros(1, 2, 4),
            negative_prompt_embeds_mask=torch.ones(1, 2, dtype=torch.bool),
        )
    return request


def _complete_scheduled_wave(scheduler, executor, wave):
    request_ids = wave.scheduled_request_ids
    results = [DiffusionOutput(output=request_id) for request_id in request_ids]
    if len(results) == 1:
        executor.collective_rpc.return_value = results[0]
    elif isinstance(executor, RayDiffusionExecutor):
        executor.collective_rpc.return_value = [{"dp_rank": rank, "output": out} for rank, out in enumerate(results)]
    else:
        executor.collective_rpc.return_value = results
    output = executor.execute_batch(wave)
    assert [item.result.output for item in output.runner_outputs] == request_ids
    assert scheduler.update_from_output(wave, output) == set(request_ids)
    for request_id in request_ids:
        assert scheduler.get_request_state(request_id).status == DiffusionRequestStatus.FINISHED_COMPLETED
        scheduler.pop_request_state(request_id)


@pytest.mark.parametrize("embeddings_first", [False, True])
def test_text_encoder_allgather_schedules_mixed_inputs_separately(executor, embeddings_first):
    config = _make_wave_config("dlo_text_encoder")
    executor.od_config = config
    scheduler = RequestScheduler()
    scheduler.initialize(config)
    first = _make_encoded_request("first", positive_embeds=embeddings_first)
    second = _make_encoded_request("second", positive_embeds=not embeddings_first)
    scheduler.add_request(first)
    scheduler.add_request(second)

    with pytest.raises(ValueError, match="same positive/negative prompt embedding fields"):
        executor.execute_batch(_make_wave(first, second))
    executor.collective_rpc.assert_not_called()

    for request_id, waiting in (("first", 1), ("second", 0)):
        wave = scheduler.schedule()
        assert wave.scheduled_request_ids == [request_id]
        assert wave.num_waiting_reqs == waiting
        _complete_scheduled_wave(scheduler, executor, wave)
    assert executor.collective_rpc.call_count == 2


@pytest.mark.parametrize("positive_embeds", [False, True])
@pytest.mark.parametrize("negative_embeds", [False, True])
def test_text_encoder_allgather_batches_matching_input_paths(executor, positive_embeds, negative_embeds):
    config = _make_wave_config("dlo_text_encoder")
    executor.od_config = config
    scheduler = RequestScheduler()
    scheduler.initialize(config)
    for request_id in ("first", "second"):
        scheduler.add_request(
            _make_encoded_request(request_id, positive_embeds=positive_embeds, negative_embeds=negative_embeds)
        )

    wave = scheduler.schedule()
    assert wave.scheduled_request_ids == ["first", "second"]
    _complete_scheduled_wave(scheduler, executor, wave)
    executor.collective_rpc.assert_called_once()


@pytest.mark.parametrize("mode", ["hsdp", "dlo"])
def test_dp_without_text_encoder_allgather_batches_mixed_inputs(executor, mode):
    config = _make_wave_config(mode)
    executor.od_config = config
    scheduler = RequestScheduler()
    scheduler.initialize(config)
    scheduler.add_request(_make_encoded_request("text", positive_embeds=False))
    scheduler.add_request(_make_encoded_request("embeds", positive_embeds=True))

    wave = scheduler.schedule()
    assert wave.scheduled_request_ids == ["text", "embeds"]
    _complete_scheduled_wave(scheduler, executor, wave)
    executor.collective_rpc.assert_called_once()


@pytest.mark.parametrize("empty_prompt", [None, "", [], (), {}, {"prompt": ""}])
@pytest.mark.parametrize("order", ["empty_first", "empty_last", "both_empty"])
def test_dp_empty_prompts_run_alone(executor, wave_config, empty_prompt, order):
    executor.od_config = wave_config
    scheduler = RequestScheduler()
    scheduler.initialize(wave_config)
    first, second = _make_request("first"), _make_request("second")
    if order in {"empty_first", "both_empty"}:
        first.prompt = empty_prompt
    if order in {"empty_last", "both_empty"}:
        second.prompt = empty_prompt
    scheduler.add_request(first)
    scheduler.add_request(second)

    with pytest.raises(ValueError, match="non-empty prompt"):
        executor.execute_batch(_make_wave(first, second))
    executor.collective_rpc.assert_not_called()

    for request_id, waiting in (("first", 1), ("second", 0)):
        wave = scheduler.schedule()
        assert wave.scheduled_request_ids == [request_id]
        assert wave.num_waiting_reqs == waiting
        _complete_scheduled_wave(scheduler, executor, wave)
    assert executor.collective_rpc.call_count == 2


@pytest.mark.parametrize(
    "prompt",
    [
        [1, 2],
        {"prompt_token_ids": [1, 2]},
        {"prompt_ids": [1, 2]},
        {"prompt": "", "prompt_embeds": torch.zeros(1, 2, 4)},
    ],
)
def test_dp_batches_nonempty_token_and_embedding_inputs(executor, wave_config, prompt):
    executor.od_config = wave_config
    scheduler = RequestScheduler()
    scheduler.initialize(wave_config)
    for request_id in ("first", "second"):
        request = _make_request(request_id)
        request.prompt = prompt
        scheduler.add_request(request)

    wave = scheduler.schedule()
    assert wave.scheduled_request_ids == ["first", "second"]
    _complete_scheduled_wave(scheduler, executor, wave)
    executor.collective_rpc.assert_called_once()


@pytest.mark.parametrize("input_kind", ["empty", "embeddings"])
def test_ordinary_batching_keeps_prompt_paths_request_local(input_kind):
    scheduler = RequestScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2, parallel_config=SimpleNamespace(data_parallel_size=2)))
    first = _make_encoded_request("first", positive_embeds=input_kind == "embeddings")
    if input_kind == "empty":
        first.prompt = ""
    scheduler.add_request(first)
    scheduler.add_request(_make_request("second"))

    assert scheduler.schedule().scheduled_request_ids == ["first", "second"]
