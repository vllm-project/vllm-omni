"""Unit tests for OmniGPUModelRunner v2 overrides."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm import SamplingParams
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner, _needs_capture_tensor_unwrap

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    def __init__(self, idx_mapping=None):
        self.idx_mapping_np = idx_mapping or []


def _make_runner():
    """Create an OmniGPUModelRunner without calling __init__."""
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = MagicMock()
    runner.req_states = SimpleNamespace(req_id_to_index={"r1": 0, "r2": 1})
    runner.execute_model_state = None
    return runner


def test_native_data_plane_does_not_inject_legacy_forward_inputs():
    runner = _make_runner()
    runner._omni_data_plane = object()
    runner.sampler = object()
    input_batch = SimpleNamespace(
        sampling_metadata=object(),
        logits_indices=torch.tensor([0]),
    )
    model_inputs = {"input_ids": torch.tensor([1])}

    runner._add_legacy_forward_inputs(model_inputs, input_batch)

    assert set(model_inputs) == {"input_ids"}


def test_legacy_data_plane_injects_required_forward_inputs():
    runner = _make_runner()
    runner._omni_data_plane = None
    runner.sampler = object()
    sampling_metadata = object()
    logits_indices = torch.tensor([0])
    input_batch = SimpleNamespace(
        sampling_metadata=sampling_metadata,
        logits_indices=logits_indices,
    )
    model_inputs = {"input_ids": torch.tensor([1])}

    runner._add_legacy_forward_inputs(model_inputs, input_batch)

    assert model_inputs["sampling_metadata"] is sampling_metadata
    assert model_inputs["logits_index"] is logits_indices
    assert model_inputs["sampler"] is runner.sampler


def test_get_mm_embeddings_uses_vllm_025_request_state_contract():
    runner = _make_runner()
    scheduled_encoder_inputs = {"r1": [0]}
    input_batch = object()
    expected = object()
    runner.model_state = SimpleNamespace(get_mm_embeddings=MagicMock(return_value=expected))

    result = runner._get_mm_embeddings(
        scheduled_encoder_inputs,
        input_batch,
    )

    assert result is expected
    runner.model_state.get_mm_embeddings.assert_called_once_with(
        scheduled_encoder_inputs,
        input_batch,
        runner.req_states,
    )


def test_prepare_mm_inputs_uses_dummy_embeddings_without_encoder_cache():
    runner = _make_runner()
    input_ids = object()
    dummy_embeddings = object()
    runner.supports_mm_inputs = True
    runner.is_first_pp_rank = True
    runner.model = SimpleNamespace(requires_raw_input_tokens=False)
    runner.model_state = SimpleNamespace(
        dummy_inputs_embeds=MagicMock(return_value=dummy_embeddings),
        get_mm_embeddings=MagicMock(),
    )
    input_batch = SimpleNamespace(
        input_ids=input_ids,
        num_tokens_after_padding=8,
    )

    result_ids, result_embeddings, ec_output = runner._prepare_mm_inputs(
        SimpleNamespace(scheduled_encoder_inputs={"_dummy_req_0": [0]}),
        input_batch,
        dummy_run=True,
    )

    assert result_ids is None
    assert result_embeddings is dummy_embeddings
    assert ec_output is None
    runner.model_state.dummy_inputs_embeds.assert_called_once_with(8)
    runner.model_state.get_mm_embeddings.assert_not_called()


def test_prepare_mm_inputs_preserves_raw_tokens_for_real_request():
    runner = _make_runner()
    input_ids = object()
    embeddings = object()
    runner.supports_mm_inputs = True
    runner.is_first_pp_rank = True
    runner.lora_config = None
    runner.model = SimpleNamespace(requires_raw_input_tokens=True)
    runner.model_state = SimpleNamespace(
        dummy_inputs_embeds=MagicMock(),
        get_mm_embeddings=MagicMock(return_value=embeddings),
    )
    input_batch = SimpleNamespace(input_ids=input_ids)
    scheduled_encoder_inputs = {"r1": [0]}
    expected_ec_output = object()
    runner.ec_connector = SimpleNamespace(maybe_get_output=MagicMock(return_value=nullcontext(expected_ec_output)))

    result_ids, result_embeddings, ec_output = runner._prepare_mm_inputs(
        SimpleNamespace(scheduled_encoder_inputs=scheduled_encoder_inputs),
        input_batch,
        dummy_run=False,
    )

    assert result_ids is input_ids
    assert result_embeddings is embeddings
    assert ec_output is expected_ec_output
    runner.model_state.get_mm_embeddings.assert_called_once_with(
        scheduled_encoder_inputs,
        input_batch,
        runner.req_states,
    )
    runner.model_state.dummy_inputs_embeds.assert_not_called()


def test_add_requests_sanitizes_stop_ids_for_narrow_logits_head():
    runner = _make_runner()
    runner.model = SimpleNamespace(logits_processor=SimpleNamespace(vocab_size=3072))
    sampling_params = SamplingParams(min_tokens=2, stop_token_ids=[2150])
    sampling_params.update_from_generation_config({}, 151645)
    scheduler_output = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(sampling_params=sampling_params)])

    with patch.object(GPUModelRunner, "add_requests", return_value=None) as upstream:
        runner.add_requests(scheduler_output)

    upstream.assert_called_once_with(scheduler_output)
    assert sampling_params.all_stop_token_ids == {2150}
    assert sampling_params.stop_token_ids == [2150]
    assert sampling_params.eos_token_id == 151645


def test_finish_requests_calls_remove_for_finished():
    runner = _make_runner()
    mock_state = MagicMock()
    runner.model_state = mock_state

    sched_output = SimpleNamespace(
        finished_req_ids={"r1"},
        preempted_req_ids=set(),
    )

    with patch.object(type(runner).__bases__[0], "finish_requests", return_value=None):
        runner.finish_requests(sched_output)

    mock_state.remove_request.assert_called_once_with(0)


def test_finish_requests_calls_remove_for_preempted():
    runner = _make_runner()
    mock_state = MagicMock()
    runner.model_state = mock_state

    sched_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"r2"},
    )

    with patch.object(type(runner).__bases__[0], "finish_requests", return_value=None):
        runner.finish_requests(sched_output)

    mock_state.remove_request.assert_called_once_with(1)


def test_finish_requests_ignores_unknown_req_ids():
    runner = _make_runner()
    mock_state = MagicMock()
    runner.model_state = mock_state

    sched_output = SimpleNamespace(
        finished_req_ids={"unknown"},
        preempted_req_ids=set(),
    )

    with patch.object(type(runner).__bases__[0], "finish_requests", return_value=None):
        runner.finish_requests(sched_output)

    mock_state.remove_request.assert_not_called()


def test_finish_requests_handles_both_finished_and_preempted():
    runner = _make_runner()
    mock_state = MagicMock()
    runner.model_state = mock_state

    sched_output = SimpleNamespace(
        finished_req_ids={"r1"},
        preempted_req_ids={"r2"},
    )

    with patch.object(type(runner).__bases__[0], "finish_requests", return_value=None):
        runner.finish_requests(sched_output)

    assert mock_state.remove_request.call_count == 2


def test_update_requests_preserves_cached_gpu_resident_side_state():
    runner = _make_runner()
    gpu_keys = {("hidden_states", "last"), ("hidden_states", "trailing_text")}
    runner.model = SimpleNamespace(gpu_resident_buffer_keys=gpu_keys)
    update_calls = []
    runner.model_state = SimpleNamespace(
        intermediate_buffer=SimpleNamespace(
            update=lambda req_idx, updates, gpu_resident_keys=None: update_calls.append(
                (req_idx, updates, gpu_resident_keys)
            )
        )
    )
    hidden = torch.randn(4)
    sched_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            additional_information={
                "r1": {
                    "hidden_states": {"last": hidden},
                }
            }
        )
    )

    with patch.object(type(runner).__bases__[0], "update_requests", return_value=None):
        runner.update_requests(sched_output)

    assert update_calls == [(0, {"hidden_states": {"last": hidden}}, gpu_keys)]


def test_prepare_native_data_plane_separates_natural_terminals_from_aborts():
    runner = _make_runner()
    plane = SimpleNamespace(
        register_request=MagicMock(),
        register_receivers=MagicMock(),
        request_terminal=MagicMock(),
        abort_requests=MagicMock(),
    )
    runner._omni_data_plane = plane
    new_req = SimpleNamespace(req_id="r1")
    handle = SimpleNamespace(request_id="r2", external_req_id="external-r2")
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[new_req],
        pending_input_registrations=[handle],
        data_plane_terminal_req_ids={"r0"},
        finished_req_ids={"r0", "aborted"},
    )

    runner._prepare_native_data_plane(scheduler_output)

    plane.register_request.assert_called_once_with(new_req)
    plane.register_receivers.assert_called_once_with([handle])
    plane.request_terminal.assert_called_once_with({"r0"})
    plane.abort_requests.assert_called_once_with({"aborted"})


def test_prepare_native_data_plane_ignores_upstream_warmup_requests():
    runner = _make_runner()
    plane = SimpleNamespace(
        register_request=MagicMock(),
        register_receivers=MagicMock(),
        request_terminal=MagicMock(),
        abort_requests=MagicMock(),
    )
    runner._omni_data_plane = plane
    warmup_req = SimpleNamespace(req_id="_warmup_0_")
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[warmup_req],
        pending_input_registrations=[],
        data_plane_terminal_req_ids=set(),
        finished_req_ids=set(),
    )

    runner._prepare_native_data_plane(scheduler_output)

    plane.register_request.assert_not_called()


def test_finalize_native_data_plane_sends_inter_stage_payload_without_engine_bridge():
    runner = _make_runner()
    connector_output = SimpleNamespace(chunk_ready_req_ids={"ready"})
    plane = SimpleNamespace(
        enqueue_outputs=MagicMock(),
        get_omni_connector_output=MagicMock(return_value=connector_output),
    )
    runner._omni_data_plane = plane
    output = SimpleNamespace(
        req_ids=["r1"],
        inter_stage_outputs=[{"codes.audio": "gpu-tensor"}],
        sampled_token_ids=[[21]],
        omni_connector_output=None,
    )

    result = runner._finalize_native_data_plane_output(output)

    assert result is output
    plane.enqueue_outputs.assert_called_once_with(
        req_ids=["r1"],
        inter_stage_outputs=[{"codes.audio": "gpu-tensor"}],
        sampled_token_ids=[[21]],
    )
    assert output.inter_stage_outputs is None
    assert output.omni_connector_output is connector_output


def test_reserve_native_data_plane_outputs_tracks_deferred_batch():
    runner = _make_runner()
    plane = SimpleNamespace(reserve_outputs=MagicMock())
    runner._omni_data_plane = plane

    runner._reserve_native_data_plane_outputs(["r1", "r2"])

    plane.reserve_outputs.assert_called_once_with(["r1", "r2"])


def test_thinker_stage_needs_capture_tensor_unwrap():
    assert _needs_capture_tensor_unwrap(SimpleNamespace(model_stage="thinker"))
    assert not _needs_capture_tensor_unwrap(SimpleNamespace(model_stage="talker"))


def test_configure_cudagraph_output_contract_enables_explicit_aux_support():
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(
        _returns_tuple=True,
        supports_mrv2_full_graph_aux_outputs=True,
    )
    runner.use_aux_hidden_state_outputs = False

    runner._configure_cudagraph_output_contract()

    assert runner._model_returns_tuple
    assert runner._supports_full_graph_aux_outputs
    assert runner.use_aux_hidden_state_outputs
    assert not runner._exclude_full_graph
    assert runner._aux_output_tree_spec is None


def test_configure_cudagraph_output_contract_keeps_unsupported_tuple_safe():
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(_returns_tuple=True)
    runner.use_aux_hidden_state_outputs = False

    runner._configure_cudagraph_output_contract()

    assert runner._model_returns_tuple
    assert not runner._supports_full_graph_aux_outputs
    assert not runner.use_aux_hidden_state_outputs
    assert runner._exclude_full_graph


def test_full_graph_capture_flattens_and_replay_restores_aux_output_tree():
    runner = object.__new__(OmniGPUModelRunner)
    runner._supports_full_graph_aux_outputs = True
    runner._aux_output_tree_spec = None
    hidden = torch.ones(2, 4)
    layer_0 = torch.full((2, 4), 3.0)
    layer_24 = torch.full((2, 4), 7.0)
    aux = {"hidden_states": {"layers": {0: layer_0, 24: layer_24}}}

    capture_output = runner._prepare_cudagraph_capture_output(
        (hidden, aux),
        CUDAGraphMode.FULL,
    )

    assert capture_output[0] is hidden
    assert capture_output[1] == [layer_0, layer_24]
    replay_hidden, replay_aux = runner._unpack_full_graph_output(capture_output)
    assert replay_hidden is hidden
    assert replay_aux["hidden_states"]["layers"][0] is layer_0
    assert replay_aux["hidden_states"]["layers"][24] is layer_24


def test_piecewise_capture_unwraps_main_tensor_without_recording_aux_schema():
    runner = object.__new__(OmniGPUModelRunner)
    runner._supports_full_graph_aux_outputs = True
    runner._aux_output_tree_spec = None
    hidden = torch.ones(2, 4)
    aux = {"hidden_states": {"layers": {0: torch.ones(2, 4)}}}

    capture_output = runner._prepare_cudagraph_capture_output(
        (hidden, aux),
        CUDAGraphMode.PIECEWISE,
    )

    assert capture_output is hidden
    assert runner._aux_output_tree_spec is None


def test_piecewise_warmup_preserves_native_aux_output_pair():
    runner = object.__new__(OmniGPUModelRunner)
    runner._supports_full_graph_aux_outputs = True
    runner._aux_output_tree_spec = None
    hidden = torch.ones(2, 4)
    layer_0 = torch.full((2, 4), 3.0)
    aux = {"hidden_states": {"layers": {0: layer_0}}}

    capture_output = runner._prepare_cudagraph_capture_output(
        (hidden, aux),
        CUDAGraphMode.NONE,
    )

    # vLLM warms up every capture descriptor in NONE mode. When auxiliary
    # outputs are enabled, its graph manager still unpacks a strict pair.
    assert capture_output == (hidden, [layer_0])
    assert runner._aux_output_tree_spec is not None


def test_full_graph_capture_rejects_non_tensor_aux_leaves():
    runner = object.__new__(OmniGPUModelRunner)
    runner._supports_full_graph_aux_outputs = True
    runner._aux_output_tree_spec = None
    hidden = torch.ones(2, 4)

    with pytest.raises(TypeError, match="tensor-only"):
        runner._prepare_cudagraph_capture_output(
            (hidden, {"hidden_states": None}),
            CUDAGraphMode.FULL,
        )


def test_capture_model_unwraps_tuple_outputs():
    runner = object.__new__(OmniGPUModelRunner)
    hidden = torch.ones(1, 2)

    def original_forward():
        return hidden, {"layers": {}}

    runner.model = SimpleNamespace(forward=original_forward)
    runner._model_returns_tuple = True
    runner._exclude_full_graph = False

    def capture_model(_self):
        assert torch.equal(runner.model.forward(), hidden)
        return 3

    with patch.object(type(runner).__bases__[0], "capture_model", capture_model):
        assert runner.capture_model() == 3

    assert runner.model.forward is original_forward


def test_capture_model_unwraps_omni_outputs():
    runner = object.__new__(OmniGPUModelRunner)
    hidden = torch.ones(1, 2)

    def original_forward():
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

    runner.model = SimpleNamespace(forward=original_forward)
    runner._model_returns_tuple = True
    runner._exclude_full_graph = False

    def capture_model(_self):
        assert torch.equal(runner.model.forward(), hidden)
        return 5

    with patch.object(type(runner).__bases__[0], "capture_model", capture_model):
        assert runner.capture_model() == 5

    assert runner.model.forward is original_forward


def test_capture_model_excludes_full_graph_without_assuming_candidate_descriptors():
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(forward=lambda: torch.ones(1, 2))
    runner._model_returns_tuple = False
    runner._exclude_full_graph = True
    piecewise = SimpleNamespace(cg_mode=CUDAGraphMode.PIECEWISE)
    full = SimpleNamespace(cg_mode=CUDAGraphMode.FULL)
    manager = SimpleNamespace(
        _capture_descs={
            CUDAGraphMode.FULL: [full],
            CUDAGraphMode.PIECEWISE: [piecewise],
        },
        _candidates=[
            [1, 2, 4],
            [piecewise, full],
        ],
    )
    runner.cudagraph_manager = manager

    with patch.object(type(runner).__bases__[0], "capture_model", return_value=7):
        assert runner.capture_model() == 7

    assert CUDAGraphMode.FULL not in manager._capture_descs
    assert manager._candidates[0] == [1, 2, 4]
    assert manager._candidates[1] == [piecewise]


def test_capture_model_excludes_full_graph_when_candidates_are_dict():
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(forward=lambda: torch.ones(1, 2))
    runner._model_returns_tuple = False
    runner._exclude_full_graph = True
    piecewise = SimpleNamespace(cg_mode=CUDAGraphMode.PIECEWISE)
    full = SimpleNamespace(cg_mode=CUDAGraphMode.FULL)
    manager = SimpleNamespace(
        _capture_descs={
            CUDAGraphMode.FULL: [full],
            CUDAGraphMode.PIECEWISE: [piecewise],
        },
        _candidates={
            CUDAGraphMode.FULL: [full],
            CUDAGraphMode.PIECEWISE: [piecewise, full],
            "sizes": [1, 2, 4],
        },
    )
    runner.cudagraph_manager = manager

    with patch.object(type(runner).__bases__[0], "capture_model", return_value=9):
        assert runner.capture_model() == 9

    assert CUDAGraphMode.FULL not in manager._capture_descs
    assert CUDAGraphMode.FULL not in manager._candidates
    assert manager._candidates[CUDAGraphMode.PIECEWISE] == [piecewise]
    assert manager._candidates["sizes"] == [1, 2, 4]


def test_capture_model_fails_closed_when_full_graph_exclusion_api_drifted():
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(forward=lambda: torch.ones(1, 2))
    runner._model_returns_tuple = False
    runner._exclude_full_graph = True
    runner.cudagraph_manager = SimpleNamespace()

    with (
        patch.object(type(runner).__bases__[0], "capture_model") as capture_model,
        pytest.raises(RuntimeError, match="cannot safely exclude FULL"),
    ):
        runner.capture_model()

    capture_model.assert_not_called()


def test_capture_model_captures_talker_mtp_graphs_after_main_capture():
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(forward=lambda: torch.ones(1, 2))
    runner._model_returns_tuple = False
    runner._exclude_full_graph = False
    runner.model_state = SimpleNamespace(capture_talker_mtp_graphs=MagicMock())
    runner._dispatch_mtp_batch_descriptor = MagicMock(return_value="desc")

    with patch.object(type(runner).__bases__[0], "capture_model", return_value=11):
        assert runner.capture_model() == 11

    runner.model_state.capture_talker_mtp_graphs.assert_called_once_with(runner._dispatch_mtp_batch_descriptor)


def test_dispatch_batch_descriptor_passes_lora_count_to_cudagraph_manager():
    runner = object.__new__(OmniGPUModelRunner)
    batch_desc = SimpleNamespace(num_tokens=8, num_reqs=2)
    runner.cudagraph_manager = SimpleNamespace(dispatch=MagicMock(return_value=batch_desc))
    runner.dp_size = 1
    runner.dp_rank = 0

    assert runner._dispatch_batch_descriptor(
        num_reqs=2,
        num_toks=8,
        uniform_tok_count=4,
        num_active_loras=0,
        use_eager=False,
        max_query_len=4,
    ) == (batch_desc, None)

    runner.cudagraph_manager.dispatch.assert_called_once_with(2, 8, 4, num_active_loras=0, max_query_len=4)


def test_dispatch_batch_descriptor_passes_lora_count_to_dp_sync():
    runner = object.__new__(OmniGPUModelRunner)
    batch_desc = SimpleNamespace(num_tokens=8, num_reqs=2)
    runner.cudagraph_manager = SimpleNamespace(dispatch=MagicMock(return_value=batch_desc))
    runner.dp_size = 2
    runner.dp_rank = 1

    with patch(
        "vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding",
        return_value=("synced", "tokens"),
    ) as sync:
        assert runner._dispatch_batch_descriptor(
            num_reqs=2,
            num_toks=8,
            uniform_tok_count=4,
            num_active_loras=3,
            use_eager=False,
            max_query_len=4,
        ) == ("synced", "tokens")

    runner.cudagraph_manager.dispatch.assert_called_once_with(2, 8, 4, num_active_loras=3, max_query_len=4)
    sync.assert_called_once_with(
        runner.cudagraph_manager,
        batch_desc,
        8,
        2,
        4,
        2,
        1,
        num_active_loras=3,
        max_query_len=4,
    )


def test_mrv2_rejects_pipeline_parallel_at_runner_startup():
    runner = object.__new__(OmniGPUModelRunner)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )

    with pytest.raises(NotImplementedError, match="pipeline parallel"):
        runner._validate_parallel_support()


def test_mrv2_rejects_prefill_context_parallel_at_runner_startup():
    runner = object.__new__(OmniGPUModelRunner)
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=2,
        ),
    )

    with pytest.raises(NotImplementedError, match="prefill context parallelism"):
        runner._validate_parallel_support()


def test_mtp_descriptor_dispatch_is_local_and_uses_captured_bucket():
    runner = object.__new__(OmniGPUModelRunner)
    runner.scheduler_config = SimpleNamespace(max_num_seqs=6)
    runner.model_state = SimpleNamespace(
        _get_talker_mtp_capture_sizes=MagicMock(return_value=[4, 2, 1]),
    )
    expected = SimpleNamespace(cg_mode=CUDAGraphMode.FULL, num_tokens=4)
    runner.cudagraph_manager = SimpleNamespace(
        dispatch=MagicMock(return_value=expected),
    )
    runner.dp_size = 2

    result = runner._dispatch_mtp_batch_descriptor(3)

    assert result is expected
    runner.cudagraph_manager.dispatch.assert_called_once_with(4, 4, 1, 0)


def test_mtp_descriptor_falls_back_to_eager_when_no_bucket_was_captured():
    runner = object.__new__(OmniGPUModelRunner)
    runner.scheduler_config = SimpleNamespace(max_num_seqs=6)
    runner.model_state = SimpleNamespace(
        _get_talker_mtp_capture_sizes=MagicMock(return_value=[4, 2, 1]),
    )
    runner.cudagraph_manager = SimpleNamespace(dispatch=MagicMock())

    result = runner._dispatch_mtp_batch_descriptor(6)

    assert result.cg_mode == CUDAGraphMode.NONE
    assert result.num_reqs == 6
    assert result.num_tokens == 6
    runner.cudagraph_manager.dispatch.assert_not_called()
