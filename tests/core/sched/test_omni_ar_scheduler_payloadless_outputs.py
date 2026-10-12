# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.stats import IterationStats, LoRARequestStates, RequestStateStats
from vllm.v1.request import Request, RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler, _holds_payloadless_outputs
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.outputs import OmniModelRunnerOutput

# isort: on

from tests.core.sched.test_omni_ar_scheduler_logprobs import _bind_request_lifecycle, _make_scheduler_stub
from tests.core.sched.test_omni_ar_scheduler_streaming import _make_request

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_AUDIO = {"model_outputs": torch.zeros(4), "sr": torch.tensor(24000)}


@pytest.fixture
def audio_scheduler():
    request = _make_request()
    request.sampling_params.detokenize = False
    request.status = RequestStatus.RUNNING
    request.prefill_stats.set(num_prompt_tokens=3, num_local_cached_tokens=0, num_external_cached_tokens=0)
    sched = _make_scheduler_stub([request])
    sched._hold_payloadless_outputs = True
    sched._held_token_ids = {}
    sched._pending_input_timeout_outputs = {}
    sched._streaming_context_overflow = {}
    sched.kv_cache_manager.estimate_cached_tokens = lambda req: 0
    sched.vllm_config.model_config.engine_output_type = "audio"
    return request, sched


def _step(mocker, sched, request: Request, token: int | None, mm_output, *, stop: bool = False, nan_count: int = 0):
    scheduler_output = mocker.Mock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {request.request_id: 1}
    scheduler_output.total_num_scheduled_tokens = 1
    scheduler_output.scheduled_spec_decode_tokens = {}

    model_runner_output = OmniModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[]] if token is None else [[token]],
        multimodal_outputs=[mm_output],
        num_nans_in_logits={request.request_id: nan_count},
    )

    def _update(req, new_token_ids, **_kwargs):
        req.append_output_token_ids(new_token_ids)
        if stop:
            req.status = RequestStatus.FINISHED_STOPPED
        return new_token_ids, stop

    _bind_request_lifecycle(sched, update_request=_update)
    sched._update_request_with_output = _update
    outputs = OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)
    return list(outputs[request.client_index].outputs) if request.client_index in outputs else []


def test_token_only_steps_ride_on_the_next_audio_output(mocker, audio_scheduler) -> None:
    request, sched = audio_scheduler

    (first,) = _step(mocker, sched, request, 10, {})
    assert first.new_token_ids == [10]
    assert first.prefill_stats is not None
    assert not first.is_coalesced

    assert _step(mocker, sched, request, 11, {}) == []
    assert _step(mocker, sched, request, None, None) == []
    assert _step(mocker, sched, request, 12, {}) == []

    (audio,) = _step(mocker, sched, request, 13, _AUDIO)
    assert audio.new_token_ids == [11, 12, 13]
    assert audio.multimodal_output is _AUDIO
    assert audio.is_coalesced

    assert _step(mocker, sched, request, 14, {}) == []
    (last,) = _step(mocker, sched, request, 15, {}, stop=True)
    assert last.new_token_ids == [14, 15]
    assert last.finish_reason is not None
    assert last.is_coalesced
    assert sched._held_token_ids == {}


def test_nan_diagnostic_survives_a_later_zero_count(mocker, monkeypatch, audio_scheduler) -> None:
    monkeypatch.setenv("VLLM_COMPUTE_NANS_IN_LOGITS", "1")
    request, sched = audio_scheduler
    stats, request_stats, lora_states = IterationStats(), RequestStateStats(), LoRARequestStates()
    emitted = []
    for step in range(7):
        for output in _step(mocker, sched, request, step, {}, stop=step == 6, nan_count=3 if step == 5 else 0):
            stats.update_from_output(output, 100.0 + step * 0.01, step == 0, request_stats, lora_states, None)
            emitted.append(output)
    assert [output.num_nans_in_logits for output in emitted] == [0, 3, 0]
    assert request_stats.is_corrupted
    assert request_stats.num_generation_tokens == 7


@pytest.mark.parametrize(
    ("field", "value"),
    [("final_output", False), ("engine_output_type", "text"), ("detokenize", True), ("logprobs", 1)],
)
def test_every_step_is_emitted_when_tokens_are_client_visible(
    mocker, monkeypatch, audio_scheduler, field, value
) -> None:
    import vllm_omni.core.sched.omni_ar_scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "_slice_sampled_logprobs", lambda *args: object())
    request, sched = audio_scheduler
    config = sched.vllm_config.model_config
    target = request.sampling_params if field in ("detokenize", "logprobs") else config
    setattr(target, field, value)
    sched._hold_payloadless_outputs = _holds_payloadless_outputs(config)
    emitted = []
    for token in (10, 11, 12):
        emitted += _step(mocker, sched, request, token, {})
    assert [eco.new_token_ids for eco in emitted] == [[10], [11], [12]]


@pytest.mark.parametrize("termination", ["kv_failure", "grammar_error", "abort", "input_timeout", "context_overflow"])
def test_external_termination_flushes_pending_tokens_after_request_removal(
    mocker, audio_scheduler, termination
) -> None:
    request, sched = audio_scheduler
    _step(mocker, sched, request, 10, {})
    assert _step(mocker, sched, request, 11, {}) == []

    def finish_requests(request_ids, status):
        request.status = status
        sched.requests.clear()
        sched.finished_req_ids_dict = {request.client_index: {request.request_id}}
        return [request]

    sched.finish_requests = finish_requests
    expected = FinishReason.ERROR
    if termination == "kv_failure":
        sched.recompute_kv_load_failures = False

        def finish_kv_failure(failed_ids, outputs):
            return OmniSchedulerMixin._handle_failed_kv_load_outputs(sched, {request.request_id}, outputs)

        sched._handle_failed_kv_load_outputs = finish_kv_failure
    elif termination == "grammar_error":
        sched.grammar_compile_error_reqs = {request.request_id}
    elif termination == "input_timeout":
        OmniSchedulerMixin._finish_input_timeout_requests(sched, {request.request_id})
    elif termination == "context_overflow":
        sched._streaming_context_overflow[request.request_id] = (request.client_index, "context overflow")
        sched.finish_requests({request.request_id}, RequestStatus.FINISHED_ERROR)
    else:
        expected = FinishReason.ABORT
        sched.finish_requests({request.request_id}, RequestStatus.FINISHED_ABORTED)

    (terminal,) = _step(mocker, sched, request, None, None)
    assert terminal.finish_reason == expected
    assert terminal.new_token_ids == [11]
    assert terminal.is_coalesced
    assert sched._held_token_ids == {}
    assert sched.requests == {}
