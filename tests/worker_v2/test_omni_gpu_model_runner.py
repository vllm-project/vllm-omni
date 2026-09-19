# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MRV2 admission, capture, dispatch and request lifecycle contracts."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm import SamplingParams
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_runner():
    """Create an OmniGPUModelRunner without calling __init__."""
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = MagicMock()
    runner.req_states = SimpleNamespace(req_id_to_index={"r1": 0, "r2": 1})
    runner.execute_model_state = None
    return runner


def test_add_requests_empty_admission_and_stop_id_sanitization():
    runner = _make_runner()
    runner.sampler = Sampler.__new__(Sampler)
    with patch.object(GPUModelRunner, "add_requests", return_value=None) as parent:
        runner.add_requests(SchedulerOutput.make_empty())
        parent.assert_not_called()

        # New requests always go upstream; narrow logits heads sanitize stop ids.
        sampling_params = SamplingParams(min_tokens=2, stop_token_ids=[2150])
        sampling_params.update_from_generation_config({}, 151645)
        runner.model = SimpleNamespace(logits_processor=SimpleNamespace(vocab_size=3072))
        output = SchedulerOutput.make_empty()
        output.scheduled_new_reqs = [SimpleNamespace(sampling_params=sampling_params)]
        runner.add_requests(output)
        parent.assert_called_once_with(output)
    assert sampling_params.all_stop_token_ids == {2150}
    assert sampling_params.eos_token_id == 151645


def test_prepare_native_data_plane_terminal_abort_split_and_warmup_skip():
    runner = _make_runner()
    plane = SimpleNamespace(
        register_request=MagicMock(),
        register_receivers=MagicMock(),
        request_terminal=MagicMock(),
        abort_requests=MagicMock(),
    )
    runner._omni_data_plane = plane
    new_req = SimpleNamespace(req_id="r1")
    warmup_req = SimpleNamespace(req_id="_warmup_0_")
    handle = SimpleNamespace(request_id="r2", external_req_id="ext-r2")
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[new_req, warmup_req],
        pending_input_registrations=[handle],
        data_plane_terminal_req_ids={"r0"},
        finished_req_ids={"r0", "aborted"},
    )

    runner._prepare_native_data_plane(scheduler_output)

    plane.register_request.assert_called_once_with(new_req)  # warmup skipped
    plane.register_receivers.assert_called_once_with([handle])
    plane.request_terminal.assert_called_once_with({"r0"})
    plane.abort_requests.assert_called_once_with({"aborted"})


@pytest.mark.parametrize("output_form", ["tuple", "omni"])
def test_capture_model_unwraps_exclude_full_and_capture_mtp(output_form):
    runner = object.__new__(OmniGPUModelRunner)
    hidden = torch.ones(1, 2)

    def original_forward():
        if output_form == "tuple":
            return hidden, {"layers": {}}
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

    runner.model = SimpleNamespace(forward=original_forward)
    runner._model_returns_tuple = True
    runner._exclude_full_graph = True
    runner.use_aux_hidden_state_outputs = False
    piecewise = SimpleNamespace(cg_mode=CUDAGraphMode.PIECEWISE)
    full = SimpleNamespace(cg_mode=CUDAGraphMode.FULL)
    runner.cudagraph_manager = SimpleNamespace(
        _capture_descs={CUDAGraphMode.PIECEWISE: [piecewise], CUDAGraphMode.FULL: [full]},
        _candidates={(1, 0): [piecewise, full]},
    )
    runner.model_state = SimpleNamespace(capture_talker_mtp_graphs=MagicMock())
    runner._dispatch_mtp_batch_descriptor = MagicMock(return_value="desc")

    def assert_unwrapped(_self):
        assert torch.equal(runner.model.forward(), hidden)  # forward unwrapped during capture
        return 3

    with patch.object(GPUModelRunner, "capture_model", assert_unwrapped):
        assert runner.capture_model() == 3

    assert runner.model.forward is original_forward  # restored after capture
    assert runner.cudagraph_manager._capture_descs == {CUDAGraphMode.PIECEWISE: [piecewise]}
    runner.model_state.capture_talker_mtp_graphs.assert_called_once_with(runner._dispatch_mtp_batch_descriptor)


@pytest.mark.parametrize(
    "parallel_config,match",
    [
        (dict(pipeline_parallel_size=2, prefill_context_parallel_size=1), "pipeline parallel"),
        (dict(pipeline_parallel_size=1, prefill_context_parallel_size=2), "prefill context parallelism"),
    ],
)
def test_mrv2_rejects_parallel_modes_at_startup(parallel_config, match):
    runner = object.__new__(OmniGPUModelRunner)
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(**parallel_config))
    with pytest.raises(NotImplementedError, match=match):
        runner._validate_parallel_support()


@pytest.mark.parametrize(
    "architectures,expect_omni",
    [(["LlamaForCausalLM"], False), (["Qwen3TTSTalkerForConditionalGeneration"], True)],
)
def test_init_model_state_factory_dispatches_omni_only(monkeypatch, architectures, expect_omni):
    upstream = MagicMock(return_value=object())
    monkeypatch.setattr("vllm_omni.worker_v2.model_states._upstream_init_model_state", upstream)
    monkeypatch.setattr(OmniModelState, "__init__", lambda *args: None)
    cfg = SimpleNamespace(model_config=SimpleNamespace(architectures=architectures))

    state = init_omni_model_state(cfg, SimpleNamespace(), None, torch.device("cpu"))

    if expect_omni:
        assert isinstance(state, OmniModelState)
        upstream.assert_not_called()
    else:
        upstream.assert_called_once()
        assert state is upstream.return_value


def test_finish_requests_notifies_model_and_cleans_only_known_slots(monkeypatch):
    runner = _make_runner()
    calls = []
    runner.model = SimpleNamespace(on_requests_finished=lambda ids: calls.append(set(ids)))
    runner.model_state = MagicMock()
    monkeypatch.setattr(GPUModelRunner, "finish_requests", lambda *args: None)
    runner.finish_requests(SimpleNamespace(finished_req_ids={"released"}, preempted_req_ids={"r1"}))
    assert calls == [{"released"}]
    assert sorted(c.args[0] for c in runner.model_state.remove_request.call_args_list) == [0]
