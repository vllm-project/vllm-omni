# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generic model opt-in contracts; no model checkpoints required."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_ar_model_runner import _merge_payload_trees

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_custom_sampler_receives_default_sampler_and_returns_wrapper():
    sampler, wrapped, state = object(), object(), object.__new__(OmniModelState)
    seen = []

    def customize(default):
        seen.append(default)
        return wrapped, None

    state.model = SimpleNamespace(mrv2_custom_sampler=customize)
    assert state.custom_sampler(sampler) == (wrapped, None)
    assert seen == [sampler]


def test_device_output_hook_receives_live_batch_and_request_state():
    state = object.__new__(OmniModelState)
    state.have_multimodal_outputs = True
    batch, requests, buffers = object(), object(), [{"req_id": "r"}]
    hidden = torch.ones(1, 2)
    seen = []

    def make_output(value, **kwargs):
        seen.append((value, kwargs))
        return OmniOutput(text_hidden_states=value, multimodal_outputs={"codes": {"audio": value}})

    state.model = SimpleNamespace(make_omni_output_mrv2=make_output)
    state.intermediate_buffer = SimpleNamespace(gather=lambda value: buffers)
    result_hidden, payload = state.postprocess_model_output(hidden, batch, requests)
    assert result_hidden is hidden
    assert payload["codes"]["audio"] is hidden
    assert seen == [(hidden, {"input_batch": batch, "req_states": requests, "model_intermediate_buffer": buffers})]


def test_sampled_embeddings_merge_preserves_sibling_fields_and_input():
    prefill, sampled = torch.ones(2, 3), torch.zeros(1, 3)
    base = {"embed": {"prefill": prefill}, "meta": {"finished": False}}
    merged = _merge_payload_trees(base, {"embed": {"sampled": [sampled]}})
    assert merged["embed"]["prefill"] is prefill
    assert merged["embed"]["sampled"] == [sampled]
    assert "sampled" not in base["embed"]
    assert merged["meta"] == {"finished": False}


def test_custom_sampler_falls_back_to_parent(monkeypatch):
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    state = object.__new__(OmniModelState)
    state.model = SimpleNamespace()
    sampler, marker = object(), object()
    monkeypatch.setattr(DefaultModelState, "custom_sampler", lambda self, original: (original, marker))
    assert state.custom_sampler(sampler) == (sampler, marker)


def test_logits_vocab_is_adopted_before_parent_sampler_initialization(monkeypatch):
    import vllm.v1.worker.gpu.model_runner as upstream

    from vllm_omni.worker_v2 import omni_model_runner as module

    runner = object.__new__(module.OmniGPUModelRunner)
    runner.vocab_size = 100
    runner.req_states = SimpleNamespace(vocab_size=100)
    runner.model_config = SimpleNamespace(model_arch_config=SimpleNamespace(vocab_size=100))
    model = SimpleNamespace(logits_vocab_size=32)
    original_initializer = upstream.init_model_state
    sentinel = object()
    monkeypatch.setattr(module, "init_omni_model_state", lambda *args: sentinel)

    def load(parent):
        assert upstream.init_model_state(None, model, None, None) is sentinel
        assert parent.vocab_size == parent.req_states.vocab_size == 32
        assert parent.model_config.model_arch_config.vocab_size == 32
        raise RuntimeError("stop before sampler allocation")

    monkeypatch.setattr(upstream.GPUModelRunner, "load_model", load)
    with pytest.raises(RuntimeError, match="stop before sampler allocation"):
        runner.load_model()
    assert upstream.init_model_state is original_initializer
