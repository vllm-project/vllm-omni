from collections.abc import Sequence
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
from vllm_omni.worker.runner_assisted_metadata import RunnerAssistedFullAttentionMetadataRequest

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


def test_runner_assisted_full_attention_metadata_request_is_opt_in():
    runner = object.__new__(GPUARModelRunner)
    runner.model = object()
    runner.scheduler_config = SimpleNamespace(max_num_seqs=16)

    request = runner._get_runner_assisted_full_attention_metadata_request(
        req_ids=["r1", "r2"],
        num_reqs=2,
        num_reqs_padded=4,
        num_scheduled_tokens_np=np.array([1, 1], dtype=np.int32),
        num_computed_tokens_cpu=np.array([5, 6], dtype=np.int32),
        max_num_scheduled_tokens=1,
    )

    assert request is None


def test_runner_assisted_full_attention_metadata_request_and_context_hooks():
    calls = []

    class Model:
        def get_runner_assisted_full_attention_metadata_request(
            self,
            *,
            req_ids: Sequence[str],
            num_reqs: int,
            num_scheduled_tokens: Sequence[int],
            num_computed_tokens: Sequence[int],
            max_num_scheduled_tokens: int,
        ) -> RunnerAssistedFullAttentionMetadataRequest:
            calls.append(
                (
                    "request",
                    {
                        "req_ids": list(req_ids),
                        "num_reqs": num_reqs,
                        "num_scheduled_tokens": [int(n) for n in num_scheduled_tokens],
                        "num_computed_tokens": [int(n) for n in num_computed_tokens],
                        "max_num_scheduled_tokens": max_num_scheduled_tokens,
                    },
                )
            )
            return RunnerAssistedFullAttentionMetadataRequest(
                num_reqs_padded=12,
                for_cudagraph_capture=True,
            )

        def set_runner_assisted_full_attention_metadata_context(
            self,
            *,
            enabled: bool,
            num_reqs: int = 0,
        ) -> None:
            calls.append(("context", {"enabled": enabled, "num_reqs": num_reqs}))

    runner = object.__new__(GPUARModelRunner)
    runner.model = Model()
    runner.scheduler_config = SimpleNamespace(max_num_seqs=8)

    request = runner._get_runner_assisted_full_attention_metadata_request(
        req_ids=["r1", "r2"],
        num_reqs=2,
        num_reqs_padded=4,
        num_scheduled_tokens_np=np.array([1, 1], dtype=np.int32),
        num_computed_tokens_cpu=np.array([5, 6], dtype=np.int32),
        max_num_scheduled_tokens=1,
    )
    context_enabled = runner._set_runner_assisted_full_attention_metadata_context(
        enabled=True,
        num_reqs=2,
    )
    context_disabled = runner._set_runner_assisted_full_attention_metadata_context(enabled=False)

    assert request == RunnerAssistedFullAttentionMetadataRequest(
        num_reqs_padded=8,
        for_cudagraph_capture=True,
    )
    assert context_enabled
    assert context_disabled
    assert calls == [
        (
            "request",
            {
                "req_ids": ["r1", "r2"],
                "num_reqs": 2,
                "num_scheduled_tokens": [1, 1],
                "num_computed_tokens": [5, 6],
                "max_num_scheduled_tokens": 1,
            },
        ),
        ("context", {"enabled": True, "num_reqs": 2}),
        ("context", {"enabled": False, "num_reqs": 0}),
    ]


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
    runner._async_chunk = True
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
        num_scheduled_tokens_np=torch.tensor([1, 2], dtype=torch.int32).numpy(),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.long),
    )

    assert output.req_ids == ["r1", "r2"]
    assert output.inter_stage_outputs is not None
    assert torch.equal(output.inter_stage_outputs[0]["hidden"], torch.tensor([[1.0]]))
    assert torch.equal(output.inter_stage_outputs[1]["hidden"], torch.tensor([[2.0], [3.0]]))
    assert output.multimodal_outputs is None
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
        num_scheduled_tokens_np=np.array([1, 2, 3], dtype=np.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.long),
    )

    assert output.inter_stage_outputs is not None
    assert output.inter_stage_outputs[0] is None
    assert torch.equal(output.inter_stage_outputs[1]["hidden"], torch.tensor([[2.0], [3.0]]))
    assert output.inter_stage_outputs[2] is None
    assert output.multimodal_outputs is None


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
        num_scheduled_tokens_np=np.array([2], dtype=np.int32),
        query_start_loc_cpu=torch.tensor([0], dtype=torch.long),
    )

    assert output.inter_stage_outputs is not None
    assert len(output.inter_stage_outputs) == 1
    assert "hidden" not in output.inter_stage_outputs[0]
    assert torch.equal(output.inter_stage_outputs[0]["codes.audio"], torch.tensor([[7, 8], [9, 10]], dtype=torch.long))
    assert output.multimodal_outputs is None


def test_build_omni_output_splits_mm_by_scheduled_tokens_when_hidden_is_tail_only(monkeypatch):
    runner = _make_async_output_runner(engine_output_type="latent")
    runner.model.omni_pooler_payload_include_hidden = False
    runner._async_chunk = False
    runner.requests = {"r1": object(), "r2": object(), "r3": object()}

    monkeypatch.setattr(
        GPUARModelRunner,
        "_resolve_pooler_payload_req_ids",
        lambda self, req_ids: ("latent", req_ids),
    )
    monkeypatch.setattr(GPUARModelRunner, "_should_accumulate_full_payload_output", lambda self: False)
    monkeypatch.setattr(GPUARModelRunner, "get_omni_connector_output", lambda self: None)
    monkeypatch.setattr(GPUARModelRunner, "_process_additional_information_updates", lambda *args, **kwargs: None)

    codes = torch.arange(48, dtype=torch.long).reshape(3, 16)
    output = GPUARModelRunner._build_omni_model_runner_output_from_snapshot(
        runner,
        scheduler_output=SimpleNamespace(
            total_num_scheduled_tokens=3,
            num_scheduled_tokens={"r1": 1, "r2": 1, "r3": 1},
        ),
        hidden_states=torch.tensor([[1.0]]),
        staged_hidden_states_cpu=None,
        multimodal_outputs={"codes": {"audio": codes}},
        req_ids_output_copy=["r1", "r2", "r3"],
        req_id_to_index_output_copy={"r1": 0, "r2": 1, "r3": 2},
        valid_sampled_token_ids=[[101], [102], [103]],
        logprobs_lists=None,
        prompt_logprobs_dict={},
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        kv_extracted_req_ids=None,
        num_scheduled_tokens_np=np.array([1, 1, 1], dtype=np.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.long),
    )

    assert output.inter_stage_outputs is not None
    assert torch.equal(output.inter_stage_outputs[0]["codes.audio"], codes[0:1])
    assert torch.equal(output.inter_stage_outputs[1]["codes.audio"], codes[1:2])
    assert torch.equal(output.inter_stage_outputs[2]["codes.audio"], codes[2:3])


def test_build_omni_output_splits_mm_by_hidden_len_when_scheduled_is_padded(monkeypatch):
    """Thinker mm rows align to hidden_states.shape[0], not padded scheduled count."""
    runner = _make_async_output_runner(engine_output_type="latent")
    runner.model.omni_pooler_payload_include_hidden = False
    runner._async_chunk = False
    runner.requests = {"r1": object(), "r2": object(), "r3": object()}

    monkeypatch.setattr(
        GPUARModelRunner,
        "_resolve_pooler_payload_req_ids",
        lambda self, req_ids: ("latent", req_ids),
    )
    monkeypatch.setattr(GPUARModelRunner, "_should_accumulate_full_payload_output", lambda self: False)
    monkeypatch.setattr(GPUARModelRunner, "get_omni_connector_output", lambda self: None)
    monkeypatch.setattr(GPUARModelRunner, "_process_additional_information_updates", lambda *args, **kwargs: None)

    layers = torch.arange(96, dtype=torch.long).reshape(3, 32)
    output = GPUARModelRunner._build_omni_model_runner_output_from_snapshot(
        runner,
        scheduler_output=SimpleNamespace(
            total_num_scheduled_tokens=5,
            num_scheduled_tokens={"r1": 1, "r2": 1, "r3": 1},
        ),
        hidden_states=torch.randn(3, 4),
        staged_hidden_states_cpu=None,
        multimodal_outputs={"hidden_states": {"layers": {0: layers}}},
        req_ids_output_copy=["r1", "r2", "r3"],
        req_id_to_index_output_copy={"r1": 0, "r2": 1, "r3": 2},
        valid_sampled_token_ids=[[101], [102], [103]],
        logprobs_lists=None,
        prompt_logprobs_dict={},
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        kv_extracted_req_ids=None,
        num_scheduled_tokens_np=np.array([1, 1, 1], dtype=np.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.long),
    )

    assert output.inter_stage_outputs is not None
    assert torch.equal(output.inter_stage_outputs[0]["hidden_states.layer_0"], layers[0:1])
    assert torch.equal(output.inter_stage_outputs[1]["hidden_states.layer_0"], layers[1:2])
    assert torch.equal(output.inter_stage_outputs[2]["hidden_states.layer_0"], layers[2:3])


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


def test_runner_assisted_full_attention_metadata_refresh_pads_buffers():
    class QueryStartLoc:
        def __init__(self):
            self.np = np.full(5, -1, dtype=np.int32)
            self.copied = False

        def copy_to_gpu(self):
            self.copied = True

    class BlockTable:
        def __init__(self):
            self.commits = []

        def commit_block_table(self, num_reqs_padded):
            self.commits.append(num_reqs_padded)

    runner = object.__new__(GPUARModelRunner)
    block_table = BlockTable()
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=torch.tensor([10, 20, 99, 99], dtype=torch.int32),
        block_table=block_table,
    )
    runner.optimistic_seq_lens_cpu = torch.zeros(4, dtype=torch.int32)
    runner.seq_lens = torch.empty(4, dtype=torch.int32)
    runner.query_pos = SimpleNamespace(np=np.empty(4, dtype=np.int32))
    runner.query_start_loc = QueryStartLoc()
    runner._get_cumsum_and_arange = lambda scheduled, _query_pos: np.cumsum(scheduled, dtype=np.int32)

    runner._refresh_runner_assisted_full_attention_metadata_buffers(
        num_reqs=2,
        num_reqs_padded=4,
        num_scheduled_tokens_np=np.array([1, 2], dtype=np.int32),
    )

    assert runner.optimistic_seq_lens_cpu.tolist() == [11, 22, 0, 0]
    assert runner.seq_lens.tolist() == [11, 22, 0, 0]
    assert runner.query_start_loc.np.tolist() == [0, 1, 3, 3, 3]
    assert runner.query_start_loc.copied
    assert block_table.commits == [4]


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
    runner._async_chunk = False

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

    # Non-async-chunk now ships the full payload to the next stage, so
    # inter_stage_outputs mirrors multimodal_outputs (PR #4792).
    assert output.inter_stage_outputs is not None
    assert output.multimodal_outputs is not None
    assert torch.equal(output.inter_stage_outputs[0]["hidden"], output.multimodal_outputs[0]["hidden"])
    assert torch.equal(output.inter_stage_outputs[1]["hidden"], output.multimodal_outputs[1]["hidden"])
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
        num_scheduled_tokens_np=np.array([1, 1], dtype=np.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.long),
    )

    assert torch.equal(output.inter_stage_outputs[0]["codes.audio"], codes[0:1])
    assert torch.equal(output.inter_stage_outputs[1]["codes.audio"], codes[1:2])
    assert output.multimodal_outputs is None
