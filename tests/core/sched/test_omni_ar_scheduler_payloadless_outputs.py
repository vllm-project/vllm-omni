# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Final audio stages fold token-only decode steps into the next audio output."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler, _holds_payloadless_outputs
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_AUDIO = {"model_outputs": torch.zeros(4), "sr": torch.tensor(24000)}


@dataclass
class _StageOutputConfig:
    """The OmniModelConfig fields the fold decision reads."""

    final_output: bool
    engine_output_type: str | None


class _PlainModelConfig:
    """A vLLM ModelConfig carries neither Omni stage field."""


@pytest.mark.parametrize(
    ("model_config", "expected"),
    [
        (_StageOutputConfig(final_output=True, engine_output_type="audio"), True),
        (_StageOutputConfig(final_output=True, engine_output_type="text"), False),
        (_StageOutputConfig(final_output=False, engine_output_type="audio"), False),
        (_PlainModelConfig(), False),
    ],
)
def test_only_final_audio_stages_fold_token_only_steps(model_config, expected) -> None:
    assert _holds_payloadless_outputs(model_config) is expected


def _make_request(**sampling) -> Request:
    request = Request(
        request_id="req-audio",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=16, **sampling),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    request.status = RequestStatus.RUNNING
    request.prefill_stats.set(num_prompt_tokens=3, num_local_cached_tokens=0, num_external_cached_tokens=0)
    return request


def _make_sched(mocker, request: Request, *, hold: bool):
    sched = mocker.MagicMock()
    sched._hold_payloadless_outputs = hold
    sched._held_token_ids = {}
    sched._pending_input_timeout_outputs = {}
    sched._streaming_context_overflow = {}
    sched._attach_finished_request_sets = OmniSchedulerMixin._attach_finished_request_sets.__get__(sched)
    sched._emit_streaming_context_overflow_outputs = OmniARScheduler._emit_streaming_context_overflow_outputs.__get__(
        sched
    )
    sched.requests = {request.request_id: request}
    sched.perf_metrics = None
    sched.defer_block_free = False
    sched._process_kv_transfer_trigger.return_value = False
    sched._reject_invalid_grammar_tokens.return_value = False
    sched._maybe_decode_pooling_output.return_value = None
    sched._handle_stopped_request.return_value = True
    sched._free_request.return_value = (None, None)
    sched.kv_cache_manager.estimate_cached_tokens.return_value = 0
    sched.chunk_transfer_adapter = None
    sched.finished_req_ids_dict = None
    sched._new_prompt_len_snapshot = {}
    sched.vllm_config.model_config = _StageOutputConfig(final_output=True, engine_output_type="audio")
    return sched


def _step(mocker, sched, request: Request, token: int | None, mm_output, *, stop: bool = False):
    scheduler_output = mocker.Mock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {request.request_id: 1}
    scheduler_output.total_num_scheduled_tokens = 1
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = mocker.Mock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = [[]] if token is None else [[token]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.prompt_token_id_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.multimodal_outputs = [mm_output]
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {request.request_id: 0}
    model_runner_output.routed_experts = None
    model_runner_output.aux_output_connector_output = None

    def _update(req, new_token_ids, **_kwargs):
        if stop:
            req.status = RequestStatus.FINISHED_STOPPED
        return new_token_ids, stop

    sched._update_request_with_output.side_effect = _update
    outputs = OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)
    if request.client_index not in outputs:
        return []
    return list(outputs[request.client_index].outputs)


def test_token_only_steps_ride_on_the_next_audio_output(mocker) -> None:
    request = _make_request(detokenize=False)
    sched = _make_sched(mocker, request, hold=True)

    # The first output always goes out: it carries the prefill statistics.
    (first,) = _step(mocker, sched, request, 10, {})
    assert first.new_token_ids == [10]
    assert first.prefill_stats is not None

    assert _step(mocker, sched, request, 11, {}) == []
    assert _step(mocker, sched, request, 12, {}) == []

    (audio,) = _step(mocker, sched, request, 13, _AUDIO)
    assert audio.new_token_ids == [11, 12, 13]
    assert audio.multimodal_output is _AUDIO

    assert _step(mocker, sched, request, 14, {}) == []
    (last,) = _step(mocker, sched, request, 15, {}, stop=True)
    assert last.new_token_ids == [14, 15]
    assert last.finish_reason is not None


@pytest.mark.parametrize(
    ("hold", "sampling"),
    [
        (False, {"detokenize": False}),  # not a final audio stage
        (True, {"detokenize": True}),  # text is visible to the client
        (True, {"detokenize": False, "logprobs": 1}),  # per-token logprobs requested
    ],
)
def test_every_step_is_emitted_when_tokens_are_client_visible(mocker, monkeypatch, hold, sampling) -> None:
    import vllm_omni.core.sched.omni_ar_scheduler as scheduler_module

    # Logprob rows come from the runner; only emission is checked here.
    monkeypatch.setattr(scheduler_module, "_slice_sampled_logprobs", lambda *args: object())
    request = _make_request(**sampling)
    sched = _make_sched(mocker, request, hold=hold)

    emitted = []
    for token in (10, 11, 12):
        emitted += _step(mocker, sched, request, token, {})
    assert [eco.new_token_ids for eco in emitted] == [[10], [11], [12]]


def test_partial_prefill_preserves_pending_tokens(mocker) -> None:
    request = _make_request(detokenize=False)
    sched = _make_sched(mocker, request, hold=True)
    _step(mocker, sched, request, 10, {})
    assert _step(mocker, sched, request, 11, {}) == []
    assert _step(mocker, sched, request, None, None) == []
    (audio,) = _step(mocker, sched, request, 12, _AUDIO)
    assert audio.new_token_ids == [11, 12]
    assert sched._held_token_ids == {}


@pytest.mark.parametrize("termination", ["kv_failure", "grammar_error", "abort", "input_timeout", "context_overflow"])
def test_external_termination_flushes_pending_tokens_after_request_removal(mocker, termination) -> None:
    request = _make_request(detokenize=False)
    sched = _make_sched(mocker, request, hold=True)
    _step(mocker, sched, request, 10, {})
    assert _step(mocker, sched, request, 11, {}) == []
    assert _step(mocker, sched, request, 12, {}) == []

    def finish_requests(request_ids, status):
        request.status = status
        sched.requests.clear()
        sched.finished_req_ids_dict = {request.client_index: {request.request_id}}
        return [request]

    sched.finish_requests.side_effect = finish_requests
    expected = FinishReason.ERROR
    if termination == "kv_failure":
        sched.recompute_kv_load_failures = False

        def finish_kv_failure(failed_ids, outputs):
            return OmniSchedulerMixin._handle_failed_kv_load_outputs(sched, {request.request_id}, outputs)

        sched._handle_failed_kv_load_outputs.side_effect = finish_kv_failure
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
    assert terminal.new_token_ids == [11, 12]
    assert sched._held_token_ids == {}
    assert sched.requests == {}
