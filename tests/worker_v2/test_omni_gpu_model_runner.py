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
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
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


def test_duplex_kv_reanchor_runs_after_block_table_writes():
    # Same model hook as the V1 AR runner: reanchor reads the committed block tables.
    runner = object.__new__(OmniGPUModelRunner)
    order = []
    for name in ("_prepare_native_data_plane", "finish_requests", "free_states", "add_requests", "update_requests"):
        setattr(runner, name, lambda *_args: None)
    runner._sync_native_data_plane_payloads = lambda *_args: None
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: order.append("block_tables"))
    runner.model = SimpleNamespace(
        apply_duplex_kv_reanchor=lambda r, scheduler_output: order.append(("reanchor", r, scheduler_output))
    )
    runner.aux_output_connector = None
    output = object()
    runner.kv_connector = SimpleNamespace(no_forward=lambda _s: output)
    runner._merge_ec_connector_no_forward = lambda _s, value: value
    runner._attach_native_data_plane_signals = lambda value: value
    scheduler_output = SchedulerOutput.make_empty()

    assert runner.execute_model(scheduler_output) is output
    assert order == ["block_tables", ("reanchor", runner, scheduler_output)]


@pytest.mark.parametrize("output_form", ["tuple", "omni"])
@pytest.mark.parametrize("profile_only", [False, True])
def test_capture_model_unwraps_exclude_full_and_capture_mtp(output_form, profile_only):
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

    def assert_unwrapped(_self, *, profile_only: bool = False):
        assert torch.equal(runner.model.forward(), hidden)  # forward unwrapped during capture
        assert profile_only is expected_profile_only
        return 3

    expected_profile_only = profile_only
    with patch.object(GPUModelRunner, "capture_model", assert_unwrapped):
        assert runner.capture_model(profile_only=profile_only) == 3

    assert runner.model.forward is original_forward  # restored after capture
    assert runner.cudagraph_manager._capture_descs == {CUDAGraphMode.PIECEWISE: [piecewise]}
    if profile_only:
        runner.model_state.capture_mtp_graphs.assert_not_called()
    else:
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


@pytest.mark.parametrize("omni_stage", [False, True])
@pytest.mark.parametrize("specialized", [False, True])
def test_auxiliary_default_state_has_omni_lifecycle_but_preserves_upstream_states(monkeypatch, omni_stage, specialized):
    class SpecializedState(DefaultModelState):
        pass

    state_cls = SpecializedState if specialized else DefaultModelState
    upstream_state = object.__new__(state_cls)
    monkeypatch.setattr("vllm_omni.worker_v2.model_states._upstream_init_model_state", lambda *_args: upstream_state)
    monkeypatch.setattr(OmniModelState, "__init__", lambda *_args: None)
    model_config = object.__new__(OmniModelConfig) if omni_stage else SimpleNamespace()
    state = init_omni_model_state(
        SimpleNamespace(model_config=model_config), SimpleNamespace(), None, torch.device("cpu")
    )
    if omni_stage and not specialized:
        assert isinstance(state, OmniModelState)
        assert callable(state.run_preprocess)
        assert callable(state.postprocess_model_output)
    else:
        assert state is upstream_state


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
    assert runner._full_graph_aux_outputs is False


def _aux_output(num_tokens):
    hidden = torch.arange(num_tokens * 2, dtype=torch.float32).reshape(num_tokens, 2)
    return hidden, {"hidden_states": {"layers": {0: hidden + 1, 24: hidden + 2}}}


def test_full_graph_aux_contract_keeps_full_and_round_trips_leaves():
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(_returns_tuple=True, supports_mrv2_full_graph_aux_outputs=True)
    runner._configure_cudagraph_output_contract()
    assert runner._full_graph_aux_outputs and not runner._exclude_full_graph

    calls = []

    def forward(num_tokens=3):
        calls.append(num_tokens)
        return _aux_output(num_tokens)

    runner.model.forward = forward
    original_forward = runner.model.forward
    runner.use_aux_hidden_state_outputs = False
    full = SimpleNamespace(cg_mode=CUDAGraphMode.FULL)
    runner.cudagraph_manager = SimpleNamespace(_capture_descs={CUDAGraphMode.FULL: [full]}, _candidates={})
    runner.model_state = SimpleNamespace()
    captured = {}

    def capture(_self, *, profile_only=False):
        # The graph manager stores (hidden, leaves) in its aux buffers.
        assert runner.use_aux_hidden_state_outputs is True
        hidden, leaves = runner.model.forward(4)
        captured["hidden"], captured["leaves"] = hidden, leaves
        assert [tuple(leaf.shape) for leaf in leaves] == [(4, 2), (4, 2)]
        runner.model.forward(2)  # smaller capture, same structure
        return 1

    with patch.object(GPUModelRunner, "capture_model", capture):
        assert runner.capture_model() == 1
    assert runner.model.forward is original_forward and runner.use_aux_hidden_state_outputs is False
    assert runner.cudagraph_manager._capture_descs == {CUDAGraphMode.FULL: [full]}  # FULL kept

    hidden, aux = runner._split_fullgraph_output((captured["hidden"], captured["leaves"]))
    reference_hidden, reference_aux = _aux_output(4)
    assert torch.equal(hidden, reference_hidden)
    assert set(aux["hidden_states"]["layers"]) == {0, 24}
    assert torch.equal(aux["hidden_states"]["layers"][24], reference_aux["hidden_states"]["layers"][24])


@pytest.mark.parametrize(
    "aux",
    [
        {},  # no leaves
        {"layers": {0: torch.zeros(5, 2)}},  # leaf not on the token axis
        {"layers": {0: 1.0}},  # non-tensor leaf
    ],
)
def test_full_graph_aux_contract_rejects_invalid_outputs(aux):
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(_returns_tuple=True, supports_mrv2_full_graph_aux_outputs=True)
    runner._configure_cudagraph_output_contract()
    with pytest.raises(RuntimeError, match="supports_mrv2_full_graph_aux_outputs"):
        runner._flatten_capture_aux_output((torch.zeros(3, 2), aux))


def test_full_graph_aux_contract_rejects_structure_change():
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(_returns_tuple=True, supports_mrv2_full_graph_aux_outputs=True)
    runner._configure_cudagraph_output_contract()
    runner._flatten_capture_aux_output((torch.zeros(3, 2), {"layers": {0: torch.zeros(3, 2)}}))
    with pytest.raises(RuntimeError, match="structure changed"):
        runner._flatten_capture_aux_output((torch.zeros(3, 2), {"layers": {1: torch.zeros(3, 2)}}))


def test_full_graph_output_without_contract_is_hidden_only():
    runner = object.__new__(OmniGPUModelRunner)
    hidden = torch.ones(2, 2)
    assert runner._split_fullgraph_output(hidden) == (hidden, None)


@pytest.mark.parametrize("runner_kind", ["gpu", "ar", "generation"])
@pytest.mark.parametrize("randomize_inputs", [False, True])
@pytest.mark.parametrize("is_profile", [False, True])
def test_dummy_forward_uses_upstream_execution_state(runner_kind, randomize_inputs, is_profile, monkeypatch):
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
    runner.model.requires_request_ids = False
    runner.model_config = SimpleNamespace()
    runner.vllm_config = SimpleNamespace()
    runner.req_states = SimpleNamespace()
    runner.model_state = MagicMock()
    runner.model_state.prepare_inputs.return_value = {}
    runner._omni_data_plane = object()
    runner.supports_mm_inputs = False
    runner.vocab_size = 32
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
    make_dummy = MagicMock(return_value=input_batch)
    monkeypatch.setattr(InputBatch, "make_dummy", make_dummy)
    randomized = []

    def randomize(tensor, low, high):
        randomized.append((low, high))
        return tensor.fill_(7)

    monkeypatch.setattr(torch.Tensor, "random_", randomize)
    monkeypatch.setattr("vllm_omni.worker_v2.omni_model_runner.build_slot_mappings_by_layer", lambda *args: {})
    for module in ("omni_model_runner", "omni_generation_model_runner"):
        monkeypatch.setattr(f"vllm_omni.worker_v2.{module}.set_forward_context", lambda *args, **kwargs: nullcontext())
    scheduled = SchedulerOutput.make_empty()
    scheduled.num_scheduled_tokens = {"_dummy_req_0": 1}
    scheduled.total_num_scheduled_tokens = 1

    # Upstream _dummy_run always supplies valid_dummy_state_slots, and the
    # result must use the real upstream constructor rather than a mocked state.
    assert (
        runner.execute_model(
            scheduled,
            dummy_run=True,
            valid_dummy_state_slots=True,
            randomize_inputs=randomize_inputs,
            is_profile=is_profile,
        )
        is None
    )
    assert make_dummy.call_args.kwargs["is_padding"] is not is_profile
    should_randomize = randomize_inputs and runner_kind != "generation"
    assert randomized == ([(0, 32)] if should_randomize else [])
    assert input_batch.input_ids.tolist() == ([7] if should_randomize else [1])
    assert isinstance(runner.execute_model_state, ExecuteModelState)
    assert runner.execute_model_state.input_batch is input_batch
    assert runner.execute_model_state.cudagraph_stats is None
    if runner_kind != "generation":
        runner.prepare_dummy_attn.assert_called_once_with(input_batch, True)
