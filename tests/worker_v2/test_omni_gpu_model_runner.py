# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MRV2 admission, capture, dispatch and request lifecycle contracts."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm import SamplingParams
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.config.model import OmniModelConfig
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
    runner.model_config = object.__new__(OmniModelConfig)
    runner.model_config.async_chunk = True
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


def test_full_payload_receive_is_polled_without_scheduled_tokens(mocker):
    runner = object.__new__(OmniGPUModelRunner)
    runner.model_config = object.__new__(OmniModelConfig)
    runner.model_config.async_chunk = False
    plane = mocker.Mock()
    runner._omni_data_plane = plane
    scheduler_output = SchedulerOutput.make_empty()

    runner._prepare_native_data_plane(scheduler_output)

    plane.recv_full_payload_inputs.assert_called_once_with(scheduler_output)


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
    runner.model_state = SimpleNamespace(capture_mtp_graphs=MagicMock())
    runner._dispatch_mtp_batch_descriptor = MagicMock(return_value="desc")

    def assert_unwrapped(_self):
        assert torch.equal(runner.model.forward(), hidden)  # forward unwrapped during capture
        return 3

    with patch.object(GPUModelRunner, "capture_model", assert_unwrapped):
        assert runner.capture_model() == 3

    assert runner.model.forward is original_forward  # restored after capture
    assert runner.cudagraph_manager._capture_descs == {CUDAGraphMode.PIECEWISE: [piecewise]}
    runner.model_state.capture_mtp_graphs.assert_called_once_with(runner._dispatch_mtp_batch_descriptor)


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
    "flag,expect_omni",
    [(None, False), ("has_preprocess", True), ("has_postprocess", True), ("have_multimodal_outputs", True)],
)
def test_init_model_state_factory_dispatches_omni_only(monkeypatch, flag, expect_omni):
    upstream = MagicMock(return_value=object())
    monkeypatch.setattr("vllm_omni.worker_v2.model_states._upstream_init_model_state", upstream)
    monkeypatch.setattr(OmniModelState, "__init__", lambda *args: None)
    cfg = SimpleNamespace(model_config=SimpleNamespace(architectures=["CustomModel"]))
    model = SimpleNamespace(**({flag: True} if flag else {}))
    state = init_omni_model_state(cfg, model, None, torch.device("cpu"))

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


@pytest.mark.parametrize("stage,declared", [("thinker", False), ("custom_ar", True)])
def test_capture_contract_uses_model_declaration(stage, declared):
    runner = _make_runner()
    runner.model = SimpleNamespace(model_stage=stage, _returns_tuple=declared)
    runner._configure_cudagraph_output_contract()
    assert runner._model_returns_tuple is declared
    assert runner._exclude_full_graph is declared


@pytest.mark.parametrize("runner_kind", ["gpu", "ar", "generation"])
def test_dummy_forward_uses_upstream_execution_state(runner_kind, monkeypatch):
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import ExecuteModelState

    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
    from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

    runner_cls = {"gpu": OmniGPUModelRunner, "ar": OmniARModelRunner, "generation": OmniGenerationModelRunner}[
        runner_kind
    ]
    runner = object.__new__(runner_cls)
    hidden = torch.ones(1, 2)
    runner.model = MagicMock(
        return_value=OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
        if runner_kind == "generation"
        else hidden
    )
    runner._dummy_hidden = hidden
    runner.model_config = SimpleNamespace()
    runner.vllm_config = SimpleNamespace()
    runner.req_states = SimpleNamespace()
    runner.model_state = MagicMock()
    runner.model_state.prepare_inputs.return_value = {}
    runner._omni_data_plane = object()
    runner.supports_mm_inputs = False
    runner.lora_config = None
    runner.is_encoder_decoder = False
    runner.eplb = MagicMock()
    runner.kv_connector = MagicMock()
    runner.input_buffers = object()
    runner.kv_cache_config = object()
    runner.attn_groups = []
    input_batch = SimpleNamespace(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        num_tokens=1,
        num_tokens_after_padding=1,
        is_padding=None,
    )
    runner.prepare_dummy_attn = MagicMock(return_value=((), torch.empty(0)))
    runner.gather_batch_req_state = MagicMock(return_value=(None, 1))
    batch_desc = SimpleNamespace(
        cg_mode=CUDAGraphMode.NONE, num_reqs=1, num_tokens=1, num_active_loras=0, max_query_len=1
    )
    runner._dispatch_batch_descriptor = MagicMock(return_value=(batch_desc, None))
    monkeypatch.setattr(InputBatch, "make_dummy", lambda *args, **kwargs: input_batch)
    monkeypatch.setattr("vllm_omni.worker_v2.omni_model_runner.build_slot_mappings_by_layer", lambda *args: {})
    for module in ("omni_model_runner", "omni_generation_model_runner"):
        monkeypatch.setattr(f"vllm_omni.worker_v2.{module}.set_forward_context", lambda *args, **kwargs: nullcontext())
    scheduled = SchedulerOutput.make_empty()
    scheduled.num_scheduled_tokens = {"_dummy_req_0": 1}
    scheduled.total_num_scheduled_tokens = 1

    # Upstream _dummy_run always supplies valid_dummy_state_slots, and the
    # result must use the real upstream constructor rather than a mocked state.
    assert runner.execute_model(scheduled, dummy_run=True, valid_dummy_state_slots=True) is None
    assert isinstance(runner.execute_model_state, ExecuteModelState)
    assert runner.execute_model_state.input_batch is input_batch
    assert runner.execute_model_state.cudagraph_stats is None
    if runner_kind != "generation":
        runner.prepare_dummy_attn.assert_called_once_with(input_batch, True)
