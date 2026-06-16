from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.gpu_ar_model_runner import ExecuteModelState, GPUARModelRunner

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
