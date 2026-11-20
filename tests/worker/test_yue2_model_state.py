# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2 sampling through native MRv2 inputs and the standard sample dispatch."""

import gc
import weakref
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.yue2 import model_state as ms
from vllm_omni.model_executor.models.yue2.runtime import Yue2ForCausalLM
from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner
from vllm_omni.worker_v2.omni_sampler import sample_with_output

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def state(mocker):
    state = object.__new__(ms.Yue2ModelState)
    state.model = mocker.Mock(spec=Yue2ForCausalLM)
    state.has_preprocess = False
    state.rope_state = None
    state._static_inputs_embeds = None
    state.intermediate_buffer = OmniIntermediateBuffer(4)
    return state


@pytest.fixture
def req_states(mocker):
    return mocker.Mock(
        spec=RequestState,
        req_id_to_index={"a": 0, "b": 2},
        prefill_len=SimpleNamespace(gpu=torch.tensor([4, 0, 4, 0])),
    )


def _batch(ids=("b", "a"), slots=(2, 0)):
    buffers = InputBuffers(4, 8, torch.device("cpu"))
    return replace(
        InputBatch.make_dummy(2, 4, buffers, is_padding=False),
        req_ids=list(ids),
        idx_mapping_np=np.array(slots),
        idx_mapping=torch.tensor(slots),
        expanded_idx_mapping=torch.tensor(slots),
        seq_lens=torch.tensor([3, 7], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([3, 7], dtype=torch.int32),
        num_computed_tokens_np=np.array([1, 5]),
        num_computed_prefill_tokens_np=np.array([1, 5]),
        prefill_len_np=np.array([4, 4]),
        is_prefilling_np=np.array([True, False]),
        has_prefill=True,
    )


def _admit(state, mocker, slot, rid, **args):
    params = SamplingParams(extra_args=args)
    data = mocker.Mock(
        spec=NewRequestData,
        req_id=rid,
        sampling_params=params,
        mm_features=[],
        prompt_embeds=None,
    )
    state.intermediate_buffer.add_request(slot, data)
    return params


def test_factory_and_tensor_forward_contract(mocker):
    model = object.__new__(Yue2ForCausalLM)
    torch.nn.Module.__init__(model)
    factory = mocker.patch.object(ms, "Yue2ModelState")
    config, cache, device = object(), object(), torch.device("cpu")
    assert init_omni_model_state(config, model, cache, device) is factory.return_value
    factory.assert_called_once_with(config, model, cache, device)
    assert model.mrv2_forward_returns_tensor


def test_binding_reorder_recompute_and_slot_reuse(state, req_states, mocker):
    first = _admit(state, mocker, 0, "a", yue2_seed=17)
    second = _admit(state, mocker, 2, "b", yue2_seed=23)
    batch = _batch()
    state.prepare_inputs(batch, req_states)
    bound = state.model.prepare_runner_inputs.call_args.kwargs
    assert bound["req_ids"] == ["b", "a"]
    assert bound["sampling_extra_args"] == [second.extra_args, first.extra_args]
    assert bound["sampling_extra_args"][0] is second.extra_args
    assert bound["num_computed_tokens"].tolist() == [1, 5]
    assert bound["num_scheduled_tokens"].tolist() == [2, 2]
    assert bound["discard_mask"].tolist() == [True, False]
    assert bound["input_ids"] is batch.input_ids
    state.intermediate_buffer.remove_request(2)
    replacement = _admit(state, mocker, 2, "c", yue2_seed=29)
    req_states.req_id_to_index.pop("b")
    req_states.req_id_to_index["c"] = 2
    resumed = replace(batch, req_ids=["c", "a"])
    state.prepare_inputs(resumed, req_states)
    assert state.model.prepare_runner_inputs.call_args.kwargs["sampling_extra_args"][0] is replacement.extra_args
    assert "b" not in state.intermediate_buffer.req_id_to_index


@pytest.mark.parametrize("kind", ["capture", "profile", "kernel", "mixed-attention"])
def test_dummy_and_warmup_do_not_enter_request_sampler(state, req_states, mocker, kind):
    _admit(state, mocker, 0, "a", yue2_seed=17)
    _admit(state, mocker, 2, "b", yue2_seed=23)
    state.prepare_inputs(_batch(), req_states)
    state.model.reset_mock()
    buffers = InputBuffers(4, 8, torch.device("cpu"))
    batch = InputBatch.make_dummy(2, 4, buffers, is_padding=kind != "profile")
    if kind in ("kernel", "mixed-attention"):
        prefix = "_warmup_" if kind == "kernel" else "_v2_mixed_warmup_"
        batch = replace(batch, req_ids=[f"{prefix}{i}" for i in range(2)])
        for i, rid in enumerate(batch.req_ids):
            _admit(state, mocker, i, rid)
        req_states.req_id_to_index = {rid: i for i, rid in enumerate(batch.req_ids)}
    state.prepare_inputs(batch, req_states)
    hidden = torch.zeros(4, 3)
    logits = state.compute_sample_logits(hidden)
    state.model.compute_prompt_logits.assert_called_once_with(hidden)
    base = mocker.Mock(spec=Sampler)
    sampler, rejection = state.custom_sampler(base)
    assert rejection is None
    assert sampler(logits, batch) is base.return_value
    state.model.prepare_runner_inputs.assert_not_called()
    state.model.sample.assert_not_called()
    state.model.compute_logits.assert_not_called()


def test_native_sample_dispatch_and_counts(state, req_states, mocker):
    _admit(state, mocker, 0, "a")
    _admit(state, mocker, 2, "b")
    batch = _batch()
    state.prepare_inputs(batch, req_states)
    base = mocker.Mock(spec=Sampler, req_states=req_states)
    sampler, _ = state.custom_sampler(base)
    counts, rejected = torch.tensor([0, 1], dtype=torch.int32), torch.zeros(2, dtype=torch.int32)
    # The native Triton count kernel is GPU-only; verify its exact inputs here.
    count = mocker.patch.object(ms, "get_num_sampled_and_rejected", return_value=(counts, rejected))
    tokens = torch.tensor([[0], [151853]], dtype=torch.int32)
    state.model.sample.return_value = tokens
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    runner = SimpleNamespace(batch_sharder=None, model_state=state, sampler=sampler, rejection_sampler=None)
    output = sample_with_output(
        sampler,
        lambda *args: GPUModelRunner.sample(runner, *args),
        hidden,
        batch,
        req_states,
        None,
    )
    torch.testing.assert_close(state.model.compute_logits.call_args.args[0], hidden[batch.logits_indices])
    state.model.sample.assert_called_once_with(state.model.compute_logits.return_value)
    assert output.sampler_output.sampled_token_ids is tokens
    assert output.num_sampled is counts and output.num_rejected is rejected
    ones, seq, cu, mapping, prefill = count.call_args.args
    assert ones.tolist() == [1, 1] and ones.dtype == torch.int32
    assert seq is batch.seq_lens and cu is batch.cu_num_logits and mapping is batch.idx_mapping
    assert prefill is req_states.prefill_len.gpu
    base.assert_not_called()


def test_sampler_profile_does_not_reuse_last_real_batch(state, req_states, mocker):
    _admit(state, mocker, 0, "a")
    _admit(state, mocker, 2, "b")
    batch = _batch()
    state.prepare_inputs(batch, req_states)
    base = mocker.Mock(spec=Sampler)
    sampler, _ = state.custom_sampler(base)
    runner = object.__new__(OmniGPUModelRunner)
    runner._kstep_frames = 1
    runner.model = state.model
    runner.model_state = state
    runner.sampler = sampler
    runner.input_buffers = InputBuffers(4, 8, torch.device("cpu"))
    mocker.patch("vllm_omni.worker_v2.omni_model_runner.current_omni_platform.is_cpu", return_value=True)
    hidden = torch.zeros(2, 3)
    runner._dummy_sampler_run(hidden)
    state.model.compute_prompt_logits.assert_called_once_with(hidden)
    state.model.compute_logits.assert_not_called()
    state.model.sample.assert_not_called()
    assert base.call_count == 1
    assert base.call_args.args[1] is not batch
    assert state._sampling_batch is batch


@pytest.mark.parametrize("error", ["slot", "params", "missing-buffers", "speculative", "grammar"])
def test_invalid_real_batch_fails_before_sampling(state, req_states, mocker, error):
    _admit(state, mocker, 0, "a")
    _admit(state, mocker, 2, "b")
    batch = _batch()
    if error == "slot":
        batch = replace(batch, idx_mapping_np=np.array([0, 2]))
    elif error == "params":
        state.intermediate_buffer.buffers[2].pop("sampling_params")
    elif error == "missing-buffers":
        state.intermediate_buffer.remove_request(0)
        state.intermediate_buffer.remove_request(2)
    elif error == "speculative":
        batch = replace(batch, num_draft_tokens=1)
    else:
        batch = replace(batch, has_structured_output_reqs=True)
    with pytest.raises((ValueError, KeyError)):
        state.prepare_inputs(batch, req_states)
    state.model.prepare_runner_inputs.assert_not_called()


def test_sampler_does_not_retain_state_or_model():
    state = object.__new__(ms.Yue2ModelState)
    state.model = torch.nn.Linear(2, 2)
    sampler, _ = state.custom_sampler(object())
    state_ref, model_ref = weakref.ref(state), weakref.ref(state.model)
    del state
    gc.collect()
    assert state_ref() is None and model_ref() is None
    assert sampler.base_sampler is not None


def test_prompt_logits_ignore_request_phase(mocker):
    model = object.__new__(Yue2ForCausalLM)
    torch.nn.Module.__init__(model)
    model.lm_head = torch.nn.Linear(2, 3)
    model.logits_processor = mocker.Mock()
    model._step_rows = [("semantic", 10, 1)]
    hidden = torch.ones(2, 2)
    assert model.compute_prompt_logits(hidden) is model.logits_processor.return_value
    model.logits_processor.assert_called_once_with(model.lm_head, hidden)
