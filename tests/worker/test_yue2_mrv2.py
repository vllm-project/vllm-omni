# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""V2 batch binding and the AR-to-acoustic lifecycle for YuE2."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.v1.outputs import AsyncModelRunnerOutput, SamplerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.yue2.model_state import Yue2ModelState, Yue2Sampler
from vllm_omni.model_executor.models.yue2.runtime import Yue2ForCausalLM, Yue2SchedulerOutput
from vllm_omni.model_executor.models.yue2.worker import Yue2ModelRunner
from vllm_omni.model_executor.models.yue2.yue2 import CODEC_OFFSET, MUSIC_END
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def batch():
    inputs = InputBuffers(2, 4, torch.device("cpu"))
    batch = InputBatch.make_dummy(2, 4, inputs)
    batch.req_ids = ["second", "first"]
    batch.idx_mapping_np = np.array([3, 1])
    batch.idx_mapping = torch.tensor([3, 1])
    batch.num_computed_tokens_np = np.array([10, 0])
    batch.num_computed_prefill_tokens_np = np.array([3, 0])
    batch.prefill_len_np = np.array([3, 5])
    batch.is_prefilling_np = np.array([False, True])
    return batch


@pytest.fixture
def state(mocker):
    state = object.__new__(Yue2ModelState)
    state.model = mocker.Mock(spec=Yue2ForCausalLM)
    state.intermediate_buffer = OmniIntermediateBuffer(4)
    for slot, rid, seed in ((1, "first", 11), (3, "second", 22)):
        state.intermediate_buffer.buffers[slot] = {
            "req_id": rid,
            "sampling_params": SamplingParams(extra_args={"yue2_seed": seed}),
        }
    return state


def test_prepare_inputs_binds_slots_and_discards_partial_recompute(state, batch, mocker):
    base = mocker.patch.object(OmniModelState, "prepare_inputs", return_value={"positions": batch.positions})
    request_state = mocker.Mock(spec=RequestState)
    result = state.prepare_inputs(batch, request_state)
    assert result is base.return_value
    args = state.model.prepare_runner_inputs.call_args.kwargs
    assert args["req_ids"] == ["second", "first"]
    assert args["sampling_extra_args"] == [{"yue2_seed": 22}, {"yue2_seed": 11}]
    assert args["num_computed_tokens"].tolist() == [10, 0]
    assert args["discard_mask"].tolist() == [False, True]
    # Completing recompute becomes eligible without changing slot order.
    batch.num_computed_prefill_tokens_np[1] = 3
    state.prepare_inputs(batch, request_state)
    assert state.model.prepare_runner_inputs.call_args.kwargs["discard_mask"].tolist() == [False, False]


def test_v2_sampler_uses_model_tokens_and_keeps_prefill_counts(state, batch, mocker):
    base = mocker.Mock(spec=Sampler)
    sampler, rejection = state.custom_sampler(base)
    assert isinstance(sampler, Yue2Sampler) and rejection is None
    counts, rejected = torch.tensor([1, 0]), torch.tensor([0, 0])
    mocker.patch(
        "vllm_omni.model_executor.models.yue2.model_state.get_num_sampled_and_rejected",
        return_value=(counts, rejected),
    )
    tokens = torch.tensor([[MUSIC_END], [CODEC_OFFSET]], dtype=torch.int32)
    state.model.sample.return_value = SamplerOutput(tokens, None)
    hidden = torch.arange(16).reshape(4, 4)
    stock = mocker.Mock(side_effect=AssertionError("YuE2 must own real-request sampling"))
    request_state = mocker.Mock(spec=RequestState, prefill_len=mocker.Mock(gpu=torch.tensor([0, 5, 0, 3])))
    result = sampler.sample_step(hidden, batch, request_state, None, stock)
    torch.testing.assert_close(state.model.compute_logits.call_args.args[0], hidden[[1, 3]])
    assert result.sampler_output.sampled_token_ids is tokens
    assert result.num_sampled is counts and result.num_rejected is rejected
    assert not result.include_hidden_states
    stock.assert_not_called()


def test_warmup_uses_stock_sampler_without_creating_song_state(state, batch, mocker):
    batch.req_ids = ["_warmup_0", "_v2_mixed_warmup_1"]
    mocker.patch.object(OmniModelState, "prepare_inputs", return_value={})
    request_state = mocker.Mock(spec=RequestState)
    state.prepare_inputs(batch, request_state)
    state.model.prepare_runner_inputs.assert_not_called()
    expected = (object(), torch.ones(2), torch.zeros(2))
    stock = mocker.Mock(return_value=expected)
    result = Yue2Sampler(mocker.Mock(spec=Sampler), state).sample_step(
        torch.zeros(4, 2), batch, request_state, None, stock
    )
    assert result.sampler_output is expected[0]
    state.model.sample.assert_not_called()


@pytest.mark.parametrize("grammar,drafts", [(object(), 0), (None, 1)])
def test_unsupported_sampling_fails_explicitly(state, batch, mocker, grammar, drafts):
    batch.num_draft_tokens = drafts
    with pytest.raises(ValueError, match="grammar or speculative"):
        Yue2Sampler(mocker.Mock(spec=Sampler), state).sample_step(
            torch.zeros(4, 2), batch, mocker.Mock(spec=RequestState), grammar, mocker.Mock()
        )
    state.model.sample.assert_not_called()


def _schedule():
    return replace(
        Yue2SchedulerOutput.make_empty(),
        finished_req_ids={"aborted"},
        acoustic_req_ids={"synthesizing", "aborted"},
        replaced_req_ids={"abc-to-semantic"},
    )


def test_detachment_does_not_cancel_audio_and_phase_replacement_does(mocker):
    runner = mocker.Mock(spec=Yue2ModelRunner)
    finish = mocker.patch.object(OmniARModelRunner, "finish_requests")
    schedule = _schedule()
    Yue2ModelRunner.finish_requests(runner, schedule)
    actual = finish.call_args.args[0]
    assert actual.finished_req_ids == {"aborted", "abc-to-semantic"}
    assert schedule.finished_req_ids == {"aborted"}  # scheduler-owned input is unchanged
    assert {call.args[0] for call in runner._remove_request.call_args_list} == {"aborted", "synthesizing"}
    # A later abort still reaches model cleanup after the native slot is gone.
    Yue2ModelRunner.finish_requests(runner, replace(schedule, finished_req_ids={"synthesizing"}))
    assert "synthesizing" in finish.call_args.args[0].finished_req_ids


@pytest.mark.parametrize("asynchronous", [False, True])
def test_audio_completion_preserves_metadata_and_drains_started_once(mocker, asynchronous):
    audio = {"audio": torch.ones(2, 8), "sr": torch.tensor(48000), "meta.truncated": torch.tensor(1)}
    model = mocker.Mock(spec=Yue2ForCausalLM, acoustic_started=["song"])
    model.poll_audio.return_value = {"song": audio}
    runner = mocker.Mock(spec=Yue2ModelRunner, model=model)
    output = OmniModelRunnerOutput(req_ids=["song"], sampled_token_ids=[[MUSIC_END]])
    output._async_chunk = False
    carrier = mocker.Mock(spec=AsyncModelRunnerOutput, get_output=lambda: output) if asynchronous else output
    result = Yue2ModelRunner._with_audio(runner, carrier)
    assert result.req_ids == ["song"] and result.sampled_token_ids == [[MUSIC_END]]
    assert result.acoustic_started == ["song"] and not model.acoustic_started
    assert result.acoustic_ready["song"] is audio
    assert Yue2ModelRunner._with_audio(runner, output).acoustic_started == []


def test_idle_execution_polls_audio_without_a_sampling_step(mocker):
    runner = object.__new__(Yue2ModelRunner)
    runner.model = mocker.Mock(spec=Yue2ForCausalLM, acoustic_started=[])
    runner.model.poll_audio.return_value = {"song": {"audio": torch.ones(2, 8)}}
    mocker.patch.object(
        OmniARModelRunner, "execute_model", return_value=OmniModelRunnerOutput(req_ids=[], sampled_token_ids=[])
    )
    result = runner.execute_model(_schedule())
    assert result.acoustic_ready["song"]["audio"].shape == (2, 8)
