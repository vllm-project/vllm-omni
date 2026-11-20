# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR detachment keeps the native request alive until audio or cancellation."""

import pytest
import torch
from vllm import SamplingParams
from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import FinishReason
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.output import OmniSchedulerOutput
from vllm_omni.model_executor.models.yue2.runtime import (
    Yue2ForCausalLM,
    Yue2ModelRunnerOutput,
    Yue2SchedulerBase,
)
from vllm_omni.model_executor.models.yue2.yue2 import (
    MUSIC_END,
    _RequestState,
    _RowConstants,
    _SynthesisJob,
    _SynthesisQueue,
)
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_v2_is_rejected_before_backbone_initialization(mocker):
    config = mocker.Mock(
        spec=VllmConfig,
        parallel_config=mocker.Mock(spec=ParallelConfig, tensor_parallel_size=1, pipeline_parallel_size=1),
        scheduler_config=mocker.Mock(spec=SchedulerConfig, async_scheduling=False),
        model_config=mocker.Mock(spec=ModelConfig, use_v2_model_runner=True),
        speculative_config=None,
    )
    initialize = mocker.patch("vllm_omni.model_executor.models.yue2.runtime.Yue2Backbone.__init__", return_value=None)
    with pytest.raises(ValueError, match="YuE2 requires model runner v1 without speculative decoding"):
        Yue2ForCausalLM(vllm_config=config)
    initialize.assert_not_called()


def test_b1_output_owns_audio_after_job_release(mocker):
    audio = torch.arange(12, dtype=torch.float32).reshape(2, 6)[:, ::2]
    state = _RequestState(
        "song",
        mocker.Mock(spec=_RowConstants),
        job=mocker.Mock(spec=_SynthesisJob, done=lambda: True, finish=lambda: (audio, False)),
        truncated=True,
    )
    model = mocker.Mock(
        spec=Yue2ForCausalLM,
        _synthesis=mocker.Mock(spec=_SynthesisQueue, pump=lambda: None),
        _states={"song": state},
        _decode_t0={"song": 0},
    )
    payload = Yue2ForCausalLM.poll_audio(model)["song"]
    torch.testing.assert_close(payload["audio"], audio)
    assert payload["audio"].is_contiguous() and payload["audio"]._base is None
    assert payload["audio"].untyped_storage().data_ptr() != audio.untyped_storage().data_ptr()
    audio.fill_(-1)
    assert payload["audio"].tolist() == [[0, 2, 4], [6, 8, 10]]
    assert payload["sr"].item() == 48000 and payload["meta.truncated"].item() == 1
    assert payload["meta.error"].item() == 0 and state.job is None and not model._decode_t0
    assert Yue2ForCausalLM.poll_audio(model) == {}


@pytest.mark.parametrize("outcome", ["success", "error", "abort"])
def test_detach_and_delayed_completion_with_coalesced_tokens(mocker, outcome):
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

    from tests.core.sched.test_omni_ar_scheduler_logprobs import _bind_request_lifecycle, _make_scheduler_stub

    request = OmniRequest("song", [1, 2, 3], SamplingParams(max_tokens=20, detokenize=False), None)
    request.status = RequestStatus.RUNNING
    request.prefill_stats.set(num_prompt_tokens=3, num_local_cached_tokens=0, num_external_cached_tokens=0)
    scheduler = object.__new__(Yue2SchedulerBase)
    scheduler.__dict__.update(vars(_make_scheduler_stub([request])))
    scheduler.async_kv_waiting = create_request_queue(SchedulingPolicy.FCFS)
    _bind_request_lifecycle(scheduler, update_request=lambda r, tokens: (tokens, False))
    del scheduler._update_request_with_output
    scheduler.max_model_len = 4096
    scheduler.num_spec_tokens = 0
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.num_waiting_for_streaming_input = 0
    scheduler._hold_payloadless_outputs = True
    scheduler._held_token_ids = {}
    scheduler._pending_input_timeout_outputs = {}
    scheduler._streaming_context_overflow = {}
    scheduler.kv_cache_manager.estimate_cached_tokens = lambda req: 0
    scheduler.vllm_config.model_config.engine_output_type = "audio"
    scheduler._inflight_prefills = {request}
    scheduler._free_request_blocks = mocker.Mock()
    scheduler.encoder_cache_manager = mocker.Mock(spec=EncoderCacheManager)

    def step(token=None, *, started=False, payload=None):
        scheduled = {"song": 1} if token is not None else {}
        schedule = mocker.Mock(
            spec=OmniSchedulerOutput,
            num_scheduled_tokens=scheduled,
            total_num_scheduled_tokens=len(scheduled),
            scheduled_spec_decode_tokens={},
        )
        output = Yue2ModelRunnerOutput(
            req_ids=["song"] if scheduled else [],
            sampled_token_ids=[[token]] if scheduled else [],
            multimodal_outputs=[{}] if scheduled else [],
            acoustic_started=["song"] if started else [],
            acoustic_ready={"song": payload} if payload is not None else {},
        )
        result = scheduler.update_from_output(schedule, output)
        return [item for batch in result.values() for item in batch.outputs]

    (first,) = step(151853)
    assert first.new_token_ids == [151853]
    assert step(151854) == []
    assert scheduler._held_token_ids == {"song": [151854]}
    assert step(MUSIC_END, started=True) == []
    scheduler._free_request_blocks.assert_called_once_with(request)
    assert not scheduler.running and not scheduler._inflight_prefills
    assert scheduler.requests["song"] is request and scheduler.get_num_unfinished_requests() == 1
    assert step() == []
    if outcome == "abort":
        request.status = RequestStatus.FINISHED_ABORTED
        scheduler.requests.clear()
        assert step(payload={"audio": torch.ones(2, 3), "meta.error": torch.tensor(False)}) == []
        assert scheduler.get_num_unfinished_requests() == 0
        return
    failed = outcome == "error"
    payload = {"audio": torch.ones(2, 3), "meta.error": torch.tensor(failed)}
    (final,) = step(payload=payload)
    assert final.new_token_ids == [151854, MUSIC_END]
    assert final.is_coalesced
    assert final.finish_reason == (FinishReason.ERROR if failed else FinishReason.STOP)
    assert final.multimodal_output is payload
    assert not scheduler._held_token_ids and not scheduler.requests
    assert scheduler.get_num_unfinished_requests() == 0
    assert step(payload=payload) == []
