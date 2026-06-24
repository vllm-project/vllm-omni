from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.gpu_ar_model_runner import (
    ExecuteModelState,
    GPUARModelRunner,
    OmniAsyncGPUModelRunnerOutput,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_runner(engine_output_type: str | None, downstream_req_ids: set[str]) -> GPUARModelRunner:
    runner = object.__new__(GPUARModelRunner)
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(engine_output_type=engine_output_type),
    )
    runner._request_needs_downstream_stage_payload = lambda rid: rid in downstream_req_ids
    return runner


def test_resolve_pooler_payload_req_ids_audio_terminal_stage_keeps_payload():
    runner = _make_runner(engine_output_type="audio", downstream_req_ids=set())

    engine_output_type, payload_req_ids = GPUARModelRunner._resolve_pooler_payload_req_ids(runner, ["r1", "r2"])

    assert engine_output_type == "audio"
    assert payload_req_ids == ["r1", "r2"]


def test_resolve_pooler_payload_req_ids_text_terminal_stage_drops_payload():
    runner = _make_runner(engine_output_type="text", downstream_req_ids=set())

    engine_output_type, payload_req_ids = GPUARModelRunner._resolve_pooler_payload_req_ids(runner, ["r1", "r2"])

    assert engine_output_type == "text"
    assert payload_req_ids == []


def test_resolve_pooler_payload_req_ids_downstream_stage_uses_filtered_requests():
    runner = _make_runner(engine_output_type="latent", downstream_req_ids={"r2"})

    engine_output_type, payload_req_ids = GPUARModelRunner._resolve_pooler_payload_req_ids(runner, ["r1", "r2", "r3"])

    assert engine_output_type == "latent"
    assert payload_req_ids == ["r2"]


def test_sparse_mm_req_ids_requires_sparse_audio_marker():
    assert GPUARModelRunner._sparse_mm_req_ids({"meta": {"req_id": ["r1"]}}) is None
    assert GPUARModelRunner._sparse_mm_req_ids({"meta.req_id": ["r1"]}) is None

    assert GPUARModelRunner._sparse_mm_req_ids({"meta": {"req_id": ["r1"], "sparse_audio": ["1"]}}) == ["r1"]
    assert GPUARModelRunner._sparse_mm_req_ids({"meta.req_id": ["r1"], "meta.sparse_audio": ["1"]}) == ["r1"]


def test_omni_async_gpu_model_runner_output_builds_lazily_once():
    async_output = object.__new__(OmniAsyncGPUModelRunnerOutput)
    calls = []
    sync_calls = []

    def builder():
        calls.append("build")
        return OmniModelRunnerOutput(req_ids=["r1"], req_id_to_index={"r1": 0})

    async_output._model_runner_output = None
    async_output._model_runner_output_builder = builder
    async_output._invalid_req_indices = []
    async_output.sampled_token_ids_cpu = torch.tensor([[7]], dtype=torch.long)
    async_output.async_copy_ready_event = SimpleNamespace(synchronize=lambda: sync_calls.append("sync"))
    async_output._sampled_token_ids = torch.tensor([[7]], dtype=torch.long)
    async_output._logprobs_tensors = None
    async_output._logprobs_tensors_cpu = None
    async_output._routed_experts = None
    async_output._routed_experts_cpu = None
    async_output.vocab_size = 10

    output = async_output.get_output()

    assert calls == ["build"]
    assert sync_calls == ["sync"]
    assert async_output._model_runner_output_builder is None
    assert output.req_ids == ["r1"]
    assert output.sampled_token_ids == [[7]]


def test_omni_async_gpu_model_runner_output_reraises_background_exception():
    async_output = object.__new__(OmniAsyncGPUModelRunnerOutput)
    joined = []

    class FakeThread:
        def join(self):
            joined.append("join")

    async_output._background_thread = FakeThread()
    async_output._background_exception = RuntimeError("background failed")

    with pytest.raises(RuntimeError, match="background failed"):
        async_output.get_output()

    assert joined == ["join"]
    assert async_output._background_thread is None


def _make_async_output_runner(engine_output_type: str = "audio"):
    runner = object.__new__(GPUARModelRunner)
    model_config = SimpleNamespace(
        engine_output_type=engine_output_type,
        async_chunk=True,
        enable_return_routed_experts=False,
    )
    runner.vllm_config = SimpleNamespace(model_config=model_config)
    runner.model_config = model_config
    runner.omni_prefix_cache = None
    runner.requests = {"r1": object(), "r2": object()}
    runner.supports_mm_inputs = False
    runner.routed_experts_initialized = False
    runner.model = SimpleNamespace(has_postprocess=False)
    runner.model_intermediate_buffer = {}
    runner.input_batch = SimpleNamespace(
        req_ids=["mutated"],
        req_id_to_index={"mutated": 0},
    )
    return runner


def test_execute_model_registers_pending_connector_registrations(monkeypatch):
    runner = object.__new__(GPUARModelRunner)
    runner.execute_model_state = None
    runner.routed_experts_initialized = False
    runner._warmup_state_cleared = True
    runner.model = SimpleNamespace()
    runner.omni_prefix_cache = None
    runner.kv_caches = []
    runner.cache_config = SimpleNamespace(block_size=16, cache_dtype="auto")
    runner._resolve_global_request_id = lambda req_id: req_id
    runner.kv_transfer_manager = SimpleNamespace(handle_finished_requests_kv_transfer=lambda **kwargs: [])
    runner._omni_connector = object()
    runner.recv_full_payload_inputs = lambda scheduler_output: None
    runner._pending_full_payload_send = {}
    runner.requests = {}
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.speculative_config = None
    runner.synchronize_input_prep = nullcontext
    runner._update_states = lambda scheduler_output: None
    runner.parallel_config = SimpleNamespace(distributed_executor_backend="mp", data_parallel_size=1)
    runner.vllm_config = SimpleNamespace()
    runner.attach_omni_connector_output = lambda output: output
    registered = []
    request = SimpleNamespace(request_id="r1")
    runner.register_chunk_recv = registered.append

    monkeypatch.setattr("vllm_omni.worker.gpu_ar_model_runner.has_kv_transfer_group", lambda: False)
    monkeypatch.setattr("vllm_omni.worker.gpu_ar_model_runner.has_ec_transfer", lambda: False)
    monkeypatch.setattr(
        "vllm_omni.worker.gpu_ar_model_runner.uses_async_chunk_model_runner_transport",
        lambda model_config: False,
    )

    scheduler_output = SimpleNamespace(
        pending_connector_registrations=[request],
        finished_requests_needing_kv_transfer={},
        finished_req_ids=set(),
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        num_scheduled_tokens={},
        kv_connector_metadata=None,
    )

    GPUARModelRunner.execute_model(runner, scheduler_output)

    assert registered == [request]


def test_build_omni_output_uses_snapshots_and_connector_after_accumulation(monkeypatch):
    runner = _make_async_output_runner()
    events = []

    monkeypatch.setattr(
        GPUARModelRunner,
        "_resolve_pooler_payload_req_ids",
        lambda self, req_ids: ("audio", req_ids),
    )
    monkeypatch.setattr(GPUARModelRunner, "_should_accumulate_full_payload_output", lambda self: True)
    monkeypatch.setattr(
        GPUARModelRunner,
        "accumulate_full_payload_output",
        lambda self, rid, payload, request: events.append(f"accumulate:{rid}"),
    )
    monkeypatch.setattr(
        GPUARModelRunner,
        "get_omni_connector_output",
        lambda self: events.append("connector") or "connector-output",
    )

    output = GPUARModelRunner._build_omni_model_runner_output_from_snapshot(
        runner,
        scheduler_output=SimpleNamespace(
            total_num_scheduled_tokens=3,
            num_scheduled_tokens={"r1": 1, "r2": 2},
        ),
        hidden_states=torch.tensor([[1.0], [2.0], [3.0]]),
        staged_hidden_states_cpu=None,
        multimodal_outputs={"foo": torch.tensor([10.0, 20.0, 30.0])},
        req_ids_output_copy=["r1", "r2"],
        req_id_to_index_output_copy={"r1": 0, "r2": 1},
        valid_sampled_token_ids=[[101], [102]],
        logprobs_lists=None,
        prompt_logprobs_dict={},
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        kv_extracted_req_ids=["r2"],
        seq_len=3,
        num_scheduled_tokens_np=torch.tensor([1, 2], dtype=torch.int32).numpy(),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.long),
    )

    assert output.req_ids == ["r1", "r2"]
    assert torch.equal(output.multimodal_outputs[0]["hidden"], torch.tensor([[1.0]]))
    assert torch.equal(output.multimodal_outputs[1]["hidden"], torch.tensor([[2.0], [3.0]]))
    assert output.kv_extracted_req_ids == ["r2"]
    assert output.omni_connector_output == "connector-output"
    assert events == ["accumulate:r1", "accumulate:r2", "connector"]


def test_build_omni_output_copies_hidden_for_partial_downstream_batch(monkeypatch):
    runner = _make_async_output_runner(engine_output_type="latent")

    monkeypatch.setattr(
        GPUARModelRunner,
        "_resolve_pooler_payload_req_ids",
        lambda self, req_ids: ("latent", ["r2"]),
    )
    monkeypatch.setattr(GPUARModelRunner, "_should_accumulate_full_payload_output", lambda self: False)
    monkeypatch.setattr(GPUARModelRunner, "get_omni_connector_output", lambda self: None)
    monkeypatch.setattr(GPUARModelRunner, "_process_additional_information_updates", lambda *args, **kwargs: None)

    output = GPUARModelRunner._build_omni_model_runner_output_from_snapshot(
        runner,
        scheduler_output=SimpleNamespace(
            total_num_scheduled_tokens=6,
            num_scheduled_tokens={"r1": 1, "r2": 2, "r3": 3},
        ),
        hidden_states=torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        staged_hidden_states_cpu=None,
        multimodal_outputs={},
        req_ids_output_copy=["r1", "r2", "r3"],
        req_id_to_index_output_copy={"r1": 0, "r2": 1, "r3": 2},
        valid_sampled_token_ids=[[], [], []],
        logprobs_lists=None,
        prompt_logprobs_dict={},
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        kv_extracted_req_ids=None,
        seq_len=6,
        num_scheduled_tokens_np=np.array([1, 2, 3], dtype=np.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.long),
    )

    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs[0] == {}
    assert torch.equal(output.multimodal_outputs[1]["hidden"], torch.tensor([[2.0], [3.0]]))
    assert output.multimodal_outputs[2] == {}


def test_process_additional_information_uses_snapshot_request_order(monkeypatch):
    runner = _make_async_output_runner()
    seen = []

    class PostprocessModel:
        has_postprocess = True

        def postprocess(self, hidden_states, **kwargs):
            seen.append(hidden_states.clone())
            return {}

    runner.model = PostprocessModel()
    runner.model_intermediate_buffer = {"r1": {}, "r2": {}}

    monkeypatch.setattr(
        GPUARModelRunner,
        "_resolve_pooler_payload_req_ids",
        lambda self, req_ids: ("audio", req_ids),
    )
    monkeypatch.setattr(GPUARModelRunner, "_should_accumulate_full_payload_output", lambda self: False)
    monkeypatch.setattr(GPUARModelRunner, "get_omni_connector_output", lambda self: None)
    monkeypatch.setattr(GPUARModelRunner, "_update_intermediate_buffer", lambda *args, **kwargs: None)

    GPUARModelRunner._build_omni_model_runner_output_from_snapshot(
        runner,
        scheduler_output=SimpleNamespace(
            total_num_scheduled_tokens=3,
            num_scheduled_tokens={"r1": 1, "r2": 2},
        ),
        hidden_states=torch.tensor([[1.0], [2.0], [3.0]]),
        staged_hidden_states_cpu=None,
        multimodal_outputs={},
        req_ids_output_copy=["r1", "r2"],
        req_id_to_index_output_copy={"r1": 0, "r2": 1},
        valid_sampled_token_ids=[[], []],
        logprobs_lists=None,
        prompt_logprobs_dict={},
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        kv_extracted_req_ids=None,
        seq_len=3,
        num_scheduled_tokens_np=torch.tensor([1, 2], dtype=torch.int32).numpy(),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.long),
    )

    assert len(seen) == 2
    assert torch.equal(seen[0], torch.tensor([[1.0]]))
    assert torch.equal(seen[1], torch.tensor([[2.0], [3.0]]))


def test_async_omni_output_guard_requires_safe_conditions():
    runner = _make_async_output_runner()
    runner.use_async_scheduling = True
    runner.speculative_config = None
    runner.model.use_async_omni_output = True

    assert GPUARModelRunner._should_use_async_omni_output(runner)

    runner.omni_prefix_cache = object()
    assert not GPUARModelRunner._should_use_async_omni_output(runner)

    runner.omni_prefix_cache = None
    runner.model.has_postprocess = True
    assert not GPUARModelRunner._should_use_async_omni_output(runner)

    runner.model.eager_omni_postprocess_before_async_output = True
    assert GPUARModelRunner._should_use_async_omni_output(runner)


def test_build_omni_output_skips_hidden_when_model_opts_out(monkeypatch):
    runner = _make_async_output_runner(engine_output_type="latent")
    runner.model.omni_pooler_payload_include_hidden = False

    monkeypatch.setattr(
        GPUARModelRunner,
        "_resolve_pooler_payload_req_ids",
        lambda self, req_ids: ("latent", req_ids),
    )
    monkeypatch.setattr(GPUARModelRunner, "_should_accumulate_full_payload_output", lambda self: False)
    monkeypatch.setattr(GPUARModelRunner, "get_omni_connector_output", lambda self: None)
    monkeypatch.setattr(GPUARModelRunner, "_process_additional_information_updates", lambda *args, **kwargs: None)

    output = GPUARModelRunner._build_omni_model_runner_output_from_snapshot(
        runner,
        scheduler_output=SimpleNamespace(
            total_num_scheduled_tokens=2,
            num_scheduled_tokens={"r1": 2},
        ),
        hidden_states=torch.tensor([[1.0], [2.0]]),
        staged_hidden_states_cpu=None,
        multimodal_outputs={"codes": {"audio": torch.tensor([[7, 8], [9, 10]], dtype=torch.long)}},
        req_ids_output_copy=["r1"],
        req_id_to_index_output_copy={"r1": 0},
        valid_sampled_token_ids=[[101]],
        logprobs_lists=None,
        prompt_logprobs_dict={},
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        kv_extracted_req_ids=None,
        seq_len=2,
        num_scheduled_tokens_np=np.array([2], dtype=np.int32),
        query_start_loc_cpu=torch.tensor([0], dtype=torch.long),
    )

    assert output.multimodal_outputs is not None
    assert len(output.multimodal_outputs) == 1
    assert "hidden" not in output.multimodal_outputs[0]
    assert torch.equal(output.multimodal_outputs[0]["codes.audio"], torch.tensor([[7, 8], [9, 10]], dtype=torch.long))


def test_async_snapshot_payload_omits_hidden_when_model_opts_out():
    runner = _make_async_output_runner()
    runner.model.omni_pooler_payload_include_hidden = False

    payload = GPUARModelRunner._build_omni_async_snapshot_payload(
        runner,
        hidden_states=torch.tensor([[1.0], [2.0]]),
        staged_hidden_states_cpu=torch.tensor([[3.0]]),
        multimodal_outputs={"codes": {"audio": torch.tensor([[1]], dtype=torch.long)}},
    )

    assert set(payload.keys()) == {"multimodal_outputs"}
    assert payload["multimodal_outputs"]["codes"]["audio"].tolist() == [[1]]


@pytest.mark.parametrize("query_start_loc_attr", ["method", "tensor_attr"])
def test_sample_tokens_tail_only_prefix_cache_uses_staged_cpu_hidden_states(monkeypatch, query_start_loc_attr):
    runner = object.__new__(GPUARModelRunner)
    runner.execute_model_state = ExecuteModelState(
        SimpleNamespace(
            total_num_scheduled_tokens=3,
            num_scheduled_tokens={"r1": 1, "r2": 2},
        ),
        None,
        None,
        None,
        torch.zeros((3, 2), dtype=torch.float32),
        torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        None,
        None,
        None,
        None,
        {},
        None,
    )
    runner.kv_connector_output = None
    runner.input_batch = SimpleNamespace(
        req_ids=["r1", "r2"],
        req_id_to_index={"r1": 0, "r2": 1},
        sampling_metadata=SimpleNamespace(no_penalties=True),
        vocab_size=10,
        num_tokens_no_spec=None,
    )
    query_start_loc = torch.tensor([0, 1], dtype=torch.long)
    if query_start_loc_attr == "method":
        runner.query_start_loc = query_start_loc
    else:
        runner.query_start_loc = SimpleNamespace(cpu=query_start_loc)
    runner.omni_prefix_cache = object()
    runner.speculative_config = None
    runner.routed_experts_initialized = False
    runner.requests = {}
    runner.supports_mm_inputs = False
    runner.use_async_scheduling = False
    runner._omni_num_scheduled_tokens_np = None
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(engine_output_type="audio"),
    )

    monkeypatch.setattr(
        GPUARModelRunner, "_sample", lambda self, logits, spec_decode_metadata: SimpleNamespace(sampled_token_ids=[])
    )
    monkeypatch.setattr(GPUARModelRunner, "_update_states_after_model_execute", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        GPUARModelRunner,
        "_bookkeeping_sync",
        lambda *args, **kwargs: (
            0,
            None,
            [],
            None,
            ["r1", "r2"],
            {"r1": 0, "r2": 1},
            [],
        ),
    )
    monkeypatch.setattr(GPUARModelRunner, "eplb_step", lambda self: None)
    monkeypatch.setattr(GPUARModelRunner, "_resolve_pooler_payload_req_ids", lambda self, req_ids: ("audio", req_ids))
    monkeypatch.setattr(GPUARModelRunner, "_deferred_prefix_cache_mm_keys", lambda self: set())
    monkeypatch.setattr(GPUARModelRunner, "_model_needs_full_prefix_hidden_states", lambda self: False)
    monkeypatch.setattr(
        GPUARModelRunner,
        "_maybe_get_combined_prefix_cache_tensors",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(GPUARModelRunner, "_process_additional_information_updates", lambda *args, **kwargs: None)
    monkeypatch.setattr(GPUARModelRunner, "_should_accumulate_full_payload_output", lambda self: False)
    monkeypatch.setattr(GPUARModelRunner, "get_omni_connector_output", lambda self: None)

    output = GPUARModelRunner.sample_tokens(runner, grammar_output=None)

    assert torch.equal(output.multimodal_outputs[0]["hidden"], torch.tensor([[1.0, 10.0]]))
    assert torch.equal(
        output.multimodal_outputs[1]["hidden"],
        torch.tensor([[2.0, 20.0], [3.0, 30.0]]),
    )


def test_build_omni_output_falls_back_to_mm_cpu_without_prefix_merge(monkeypatch):
    """Tail-only prefix-cache models still need per-step mm passthrough (e.g. codes.audio)."""
    runner = _make_async_output_runner(engine_output_type="latent")
    runner.omni_prefix_cache = object()

    monkeypatch.setattr(
        GPUARModelRunner,
        "_resolve_pooler_payload_req_ids",
        lambda self, req_ids: ("latent", req_ids),
    )
    monkeypatch.setattr(GPUARModelRunner, "_model_needs_full_prefix_hidden_states", lambda self: False)
    monkeypatch.setattr(GPUARModelRunner, "_deferred_prefix_cache_mm_keys", lambda self: {"codes.audio"})
    monkeypatch.setattr(GPUARModelRunner, "_stage_deferred_prefix_cache_mm_outputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        GPUARModelRunner,
        "_prepare_prefix_cache_pooler_payload_sources",
        lambda *args, **kwargs: (None, None, None),
    )
    monkeypatch.setattr(GPUARModelRunner, "_process_additional_information_updates", lambda *args, **kwargs: None)
    monkeypatch.setattr(GPUARModelRunner, "_should_accumulate_full_payload_output", lambda self: False)
    monkeypatch.setattr(GPUARModelRunner, "get_omni_connector_output", lambda self: None)

    codes = torch.tensor([[11.0, 12.0], [21.0, 22.0]], dtype=torch.float32)
    output = GPUARModelRunner._build_omni_model_runner_output_from_snapshot(
        runner,
        scheduler_output=SimpleNamespace(
            total_num_scheduled_tokens=2,
            num_scheduled_tokens={"r1": 1, "r2": 1},
        ),
        hidden_states=torch.tensor([[1.0], [2.0]]),
        staged_hidden_states_cpu=None,
        multimodal_outputs={"codes.audio": codes},
        req_ids_output_copy=["r1", "r2"],
        req_id_to_index_output_copy={"r1": 0, "r2": 1},
        valid_sampled_token_ids=[[], []],
        logprobs_lists=None,
        prompt_logprobs_dict={},
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        kv_extracted_req_ids=None,
        seq_len=2,
        num_scheduled_tokens_np=np.array([1, 1], dtype=np.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.long),
    )

    assert torch.equal(output.multimodal_outputs[0]["codes.audio"], codes[0:1])
    assert torch.equal(output.multimodal_outputs[1]["codes.audio"], codes[1:2])


@pytest.mark.parametrize(
    ("additional_information", "expected"),
    [
        ({"omni_final_stage_id": 0}, False),
        ({"omni_final_stage_id": 2}, True),
        ({}, True),
    ],
)
def test_request_needs_downstream_stage_payload_respects_final_stage_id(monkeypatch, additional_information, expected):
    runner = object.__new__(GPUARModelRunner)
    runner.model_config = SimpleNamespace()
    runner.model_intermediate_buffer = {}
    runner.requests = {
        "r1": SimpleNamespace(additional_information_cpu=additional_information),
    }
    runner._downstream_payload_cache = {}
    runner._custom_process_func = object()

    monkeypatch.setattr(
        "vllm_omni.worker.gpu_ar_model_runner.uses_async_chunk_model_runner_transport",
        lambda model_config: True,
    )

    assert GPUARModelRunner._request_needs_downstream_stage_payload(runner, "r1") is expected
    assert runner._downstream_payload_cache["r1"] is expected


def test_request_needs_downstream_stage_payload_keeps_runner_transport_producer(monkeypatch):
    runner = object.__new__(GPUARModelRunner)
    runner.model_config = SimpleNamespace()
    runner.model_intermediate_buffer = {}
    runner.requests = {
        "r1": SimpleNamespace(additional_information_cpu={"omni_final_stage_id": 1}),
    }
    runner._downstream_payload_cache = {}
    runner._custom_process_func = object()

    monkeypatch.setattr(
        "vllm_omni.worker.gpu_ar_model_runner.uses_async_chunk_model_runner_transport",
        lambda model_config: True,
    )

    assert GPUARModelRunner._request_needs_downstream_stage_payload(runner, "r1")


def test_finish_sentinel_enqueues_when_first_payload_is_cached():
    runner = object.__new__(GPUARModelRunner)
    runner._request_ids_mapping = {"local-1": "ext-1"}
    runner._put_req_chunk = {}
    runner._send_side_request_payload = {"ext-1": {"embed": {"prefill": object()}}}
    runner._pending_streaming_prefills = {}
    runner._send_side_request_snapshot = {
        "ext-1": SimpleNamespace(
            request_id="local-1",
            req_id="local-1",
            external_req_id="ext-1",
            is_finished=lambda: False,
        )
    }
    sent = []
    runner.enqueue_finish_sentinel = lambda request, ext_id: sent.append((request, ext_id)) or True

    GPUARModelRunner._send_async_chunk_finish_sentinels(runner, {"local-1"})

    assert sent == [(runner._send_side_request_snapshot["ext-1"], "ext-1")]
    assert runner._send_side_request_snapshot["ext-1"].is_finished() is True


def test_finish_sentinel_enqueues_bare_terminal_when_downstream_waits():
    runner = object.__new__(GPUARModelRunner)
    runner._request_ids_mapping = {"local-1": "ext-1"}
    runner._put_req_chunk = {}
    runner._send_side_request_payload = {}
    runner._pending_streaming_prefills = {}
    runner._send_side_request_snapshot = {}
    runner._downstream_payload_cache = {"local-1": True}
    runner.model_intermediate_buffer = {}
    runner.requests = {}
    sent = []

    def enqueue(request, ext_id):
        sent.append((request, ext_id, request.is_finished()))
        return True

    runner.enqueue_finish_sentinel = enqueue

    GPUARModelRunner._send_async_chunk_finish_sentinels(runner, {"local-1"})

    assert len(sent) == 1
    request, ext_id, is_finished = sent[0]
    assert ext_id == "ext-1"
    assert request.request_id == "local-1"
    assert request.external_req_id == "ext-1"
    assert is_finished is True


def test_segment_terminal_waits_for_talker_eos():
    runner = object.__new__(GPUARModelRunner)
    runner.model_config = SimpleNamespace(model_stage="talker")
    runner.model_intermediate_buffer = {
        "waiting": {
            "meta": {
                "is_segment_finished": torch.tensor(True),
                "eos_emitted": torch.tensor(False),
            }
        },
        "segment": {
            "meta": {
                "is_segment_finished": torch.tensor(True),
                "eos_emitted": torch.tensor(True),
                "finished": torch.tensor(False),
            }
        },
    }

    assert not GPUARModelRunner._async_chunk_segment_terminal_ready(runner, "waiting")
    assert GPUARModelRunner._async_chunk_segment_terminal_ready(runner, "segment")


def test_segment_terminal_non_talker_does_not_wait_for_eos():
    runner = object.__new__(GPUARModelRunner)
    runner.model_config = SimpleNamespace(model_stage="thinker")
    runner.model_intermediate_buffer = {
        "local-1": {
            "meta": {
                "is_segment_finished": torch.tensor(True),
                "eos_emitted": torch.tensor(False),
            }
        }
    }

    assert GPUARModelRunner._async_chunk_segment_terminal_ready(runner, "local-1")


def test_segment_terminal_enqueue_is_idempotent():
    runner = object.__new__(GPUARModelRunner)
    runner.model_intermediate_buffer = {
        "local-1": {
            "meta": {
                "is_segment_finished": torch.tensor(True),
                "eos_emitted": torch.tensor(True),
            }
        }
    }
    runner._request_ids_mapping = {"local-1": "ext-1"}
    runner._send_side_request_snapshot = {
        "ext-1": SimpleNamespace(
            request_id="local-1",
            req_id="local-1",
            external_req_id="ext-1",
            is_finished=lambda: False,
        )
    }
    runner._async_chunk_segment_terminal_sent = set()
    sent = []

    def enqueue(request, ext_id, *, is_segment_finished=False):
        sent.append((request, ext_id, is_segment_finished))
        return True

    runner.enqueue_finish_sentinel = enqueue

    GPUARModelRunner._send_async_chunk_segment_sentinels(runner, {"local-1"})
    GPUARModelRunner._send_async_chunk_segment_sentinels(runner, {"local-1"})

    assert len(sent) == 1
    assert sent[0][1:] == ("ext-1", True)


def test_segment_terminal_guard_clears_when_new_decode_resets_eos():
    runner = object.__new__(GPUARModelRunner)
    runner._async_chunk_segment_terminal_sent = {"ext-1"}
    runner.model_intermediate_buffer = {"local-1": {"meta": {"eos_emitted": torch.tensor(False)}}}

    assert not GPUARModelRunner._should_skip_async_chunk_payload_after_segment_terminal(runner, "local-1", "ext-1")
    assert "ext-1" not in runner._async_chunk_segment_terminal_sent


def test_build_omni_output_runner_transport_sends_nested_payload(monkeypatch):
    runner = _make_async_output_runner(engine_output_type="latent")
    runner._custom_process_func = object()
    runner.model_config.model_stage = "thinker"
    runner._request_ids_mapping = {"r1": "ext-r1"}
    runner._send_side_request_snapshot = {}
    runner.model_intermediate_buffer = {}
    runner.requests = {
        "r1": SimpleNamespace(
            req_id="r1",
            external_req_id="ext-r1",
            prompt_token_ids=[1],
            output_token_ids=[2],
            additional_information_cpu={"omni_final_stage_id": 1},
        )
    }
    sent = []

    def send_chunk(request, pooling_output):
        sent.append((request, pooling_output))
        return True

    runner.send_chunk = send_chunk
    runner._snapshot_request_for_send = lambda request, ext_id: SimpleNamespace(
        request_id=request.request_id,
        req_id=request.req_id,
        external_req_id=ext_id,
        is_finished=request.is_finished,
    )

    monkeypatch.setattr(
        "vllm_omni.worker.gpu_ar_model_runner.uses_async_chunk_model_runner_transport",
        lambda model_config: True,
    )
    monkeypatch.setattr(
        GPUARModelRunner,
        "_resolve_pooler_payload_req_ids",
        lambda self, req_ids: ("latent", req_ids),
    )
    monkeypatch.setattr(GPUARModelRunner, "_should_accumulate_full_payload_output", lambda self: False)
    monkeypatch.setattr(GPUARModelRunner, "get_omni_connector_output", lambda self: None)
    monkeypatch.setattr(GPUARModelRunner, "_process_additional_information_updates", lambda *args, **kwargs: None)

    output = GPUARModelRunner._build_omni_model_runner_output_from_snapshot(
        runner,
        scheduler_output=SimpleNamespace(
            total_num_scheduled_tokens=1,
            num_scheduled_tokens={"r1": 1},
        ),
        hidden_states=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        staged_hidden_states_cpu=None,
        multimodal_outputs={"codes": {"audio": torch.tensor([[7, 8]], dtype=torch.long)}},
        req_ids_output_copy=["r1"],
        req_id_to_index_output_copy={"r1": 0},
        valid_sampled_token_ids=[[]],
        logprobs_lists=None,
        prompt_logprobs_dict={},
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        kv_extracted_req_ids=None,
        seq_len=1,
        num_scheduled_tokens_np=np.array([1], dtype=np.int32),
        query_start_loc_cpu=torch.tensor([0], dtype=torch.long),
    )

    assert output.multimodal_outputs is None
    assert len(sent) == 1
    request, payload = sent[0]
    assert request.external_req_id == "ext-r1"
    assert request.is_finished() is False
    assert torch.equal(payload["hidden"], torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert "codes.audio" not in payload
    assert torch.equal(payload["codes"]["audio"], torch.tensor([[7, 8]], dtype=torch.long))
    assert runner._send_side_request_snapshot["ext-r1"].is_finished() is False
