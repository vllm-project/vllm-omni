# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_omni.model_executor.output_snapshot import PackedOutputSnapshot
from vllm_omni.worker_v2.model_states.lychee_model_state import LycheeModelState
from vllm_omni.worker_v2.model_states.lychee_sampler import LycheeSampler
from vllm_omni.worker_v2.omni_sampler import OmniSamplingContext, sample_with_output

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_model_state_registers_lychee_sampler_through_current_interface():
    state = object.__new__(LycheeModelState)
    base = SimpleNamespace()
    sampler, rejection = state.custom_sampler(base)
    assert isinstance(sampler, LycheeSampler)
    assert sampler.base_sampler is base and sampler.model_state is state
    assert rejection is None


def test_registered_sampler_publishes_constrained_text_and_owned_tick_payload():
    constrained = torch.tensor([[99], [17]], dtype=torch.int32)
    counts, rejected = torch.tensor([1, 0]), torch.tensor([0, 0])
    standard_output = SamplerOutput(torch.tensor([[7], [17]]), None, None, counts, rejected)
    payload = {"lychee_text_token_ids": torch.tensor([99, -1])}
    mm = {"lychee_stoken_hidden": torch.zeros(2, 4), "lychee_control_hidden": torch.ones(2, 4)}
    events = []

    @contextmanager
    def forward_context():
        events.append("enter")
        yield
        events.append("exit")

    def continuation(**kwargs):
        assert events == ["enter"]
        assert kwargs["sampled_token_ids"] is constrained
        assert kwargs["multimodal_outputs"] is mm
        return payload

    state = SimpleNamespace(
        constrain_primary_sample=Mock(return_value=constrained),
        continue_after_primary_sample=Mock(side_effect=continuation),
        mark_primary_continuation_failed=Mock(),
    )
    sampler = LycheeSampler(SimpleNamespace(), state)
    hidden, req_states, grammar = torch.zeros(2, 4), object(), object()
    batch = SimpleNamespace(num_reqs=2, query_start_loc_np=np.array([0, 1, 2]))
    standard = Mock(return_value=(standard_output, counts, rejected))
    context = OmniSamplingContext(batch, forward_context, mm)
    with sampler.set_sampling_context(context):
        result = sample_with_output(sampler, standard, hidden, batch, req_states, grammar)
    standard.assert_called_once_with(hidden, batch, grammar)
    assert result.sampler_output.sampled_token_ids is constrained
    assert result.num_sampled is counts and result.num_rejected is rejected
    assert isinstance(result.multimodal_outputs, PackedOutputSnapshot)
    assert result.multimodal_outputs["lychee_text_token_ids"].tolist() == [99, -1]
    copied = result.multimodal_outputs.copy_to_cpu(lambda slab: slab.clone())
    result.multimodal_outputs.mark_copy_started()
    result.multimodal_outputs.bind_copy_event(object())
    assert result.include_hidden_states is False and result.owns_multimodal_outputs is True
    finalized = result.finalize_multimodal(copied, counts.tolist())
    assert finalized.inter_stage[0]["lychee_text_token_ids"].tolist() == [99]
    assert finalized.client[0]["lychee_text_token_ids"].tolist() == [99]
    assert finalized.inter_stage[1] is None and finalized.client[1] is None
    assert events == ["enter", "exit"]
    assert sampler._sampling_context is None
    state.mark_primary_continuation_failed.assert_not_called()


def test_registered_sampler_rejects_missing_forward_context():
    sampler = LycheeSampler(SimpleNamespace(), SimpleNamespace())
    standard = Mock()
    with pytest.raises(RuntimeError, match="current forward context"):
        sample_with_output(sampler, standard, torch.zeros(1, 2), object(), object(), None)
    standard.assert_not_called()


def test_poison_failure_preserves_original_sampling_error():
    state = SimpleNamespace(mark_primary_continuation_failed=Mock(side_effect=RuntimeError("poison failed")))
    sampler = LycheeSampler(SimpleNamespace(), state)
    with pytest.raises(RuntimeError, match="sampling failed"):
        with sampler.set_sampling_context(OmniSamplingContext(object(), nullcontext)):
            raise RuntimeError("sampling failed")
    assert sampler._sampling_context is None


def test_lychee_factory_uses_generic_model_owned_dispatch(monkeypatch):
    from vllm_omni.model_executor.models.lychee_fd.modeling_lychee import LycheeFullDuplexForConditionalGeneration
    from vllm_omni.worker_v2.model_states import init_omni_model_state, lychee_model_state

    model = object.__new__(LycheeFullDuplexForConditionalGeneration)
    torch.nn.Module.__init__(model)
    expected = object.__new__(LycheeModelState)
    constructor = Mock(return_value=expected)
    monkeypatch.setattr(lychee_model_state, "LycheeModelState", constructor)
    config, cache, device = object(), object(), torch.device("cpu")
    assert init_omni_model_state(config, model, cache, device) is expected
    constructor.assert_called_once_with(config, model, cache, device)


def test_primary_logits_constraints_run_before_native_base_sampler():
    constrained = torch.tensor([[float("-inf"), 2.0]])
    state = SimpleNamespace(constrain_primary_logits=Mock(return_value=constrained))
    base = Mock(return_value=object())
    sampler = LycheeSampler(base, state)
    batch = object()
    logits = torch.ones(1, 2)
    assert sampler(logits, batch) is base.return_value
    state.constrain_primary_logits.assert_called_once_with(logits, batch)
    base.assert_called_once_with(constrained, batch)
