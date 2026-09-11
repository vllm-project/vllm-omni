# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.duplex_sampling import DuplexSamplingRow
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.input_history import DuplexPromptHistory
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration
from vllm_omni.model_executor.models.output_templates import ModelInputError
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model_with_prepared_units():
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.get_input_embeddings = lambda ids: ids.float().unsqueeze(-1) * 10

    def prepare(*args, seq, **kwargs):
        values = [101, 102] if seq == 1 else [201, 202]
        return {
            "success": True,
            "inputs_embeds": torch.tensor(values).float().unsqueeze(-1),
            "input_token_ids": [11, 12] if seq == 1 else [21, 22],
        }

    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(
        sessions={("sid", 0): SimpleNamespace()},
        _decode_audio_payload=lambda payload: [1.0],
        _decode_video_frames_payload=lambda payload: [],
        _stage_prefill_embeddings_only=prepare,
    )
    return model


def _preprocess(model, *, seq, ids, offset, prompt):
    return model.preprocess(
        torch.tensor(ids),
        request_id="physical-0",
        duplex_token_offset=offset,
        duplex_prompt_len=len(prompt),
        duplex_scheduler_prompt_token_ids=prompt,
        duplex={"data_plane": True, "session_id": "sid", "incarnation": 0, "epoch": 0, "seq": seq, "payload": {}},
    )


def test_preemption_recompute_preserves_old_audio_and_generated_token_gap():
    model = _model_with_prepared_units()
    _preprocess(model, seq=1, ids=[0, 0], offset=0, prompt=[0, 0])
    _, tail, _ = _preprocess(model, seq=2, ids=[0, 0], offset=3, prompt=[0, 0, 7, 0, 0])
    assert tail[:, 0].tolist() == [201, 202]
    history = model._minicpmo45_duplex_input_histories["physical-0"]
    before = history.num_bytes
    ids, recomputed, result = _preprocess(model, seq=2, ids=[0, 0, 7, 0, 0], offset=0, prompt=[0, 0, 7, 0, 0])
    assert ids.tolist() == [11, 12, 7, 21, 22]
    assert recomputed[:, 0].tolist() == [101, 102, 70, 201, 202]
    assert result["duplex"]["duplex_prompt_token_ids"] == ids.tolist()
    assert history.num_bytes == before
    assert len(history.spans) == 2
    assert all(span.embeddings.device.type == "cpu" for span in history.spans)


@pytest.mark.parametrize("offset", [0, 1])
def test_partial_live_prefill_slices_the_prepared_unit(offset):
    model = _model_with_prepared_units()
    ids, embeddings, _ = _preprocess(model, seq=1, ids=[0], offset=offset, prompt=[0, 0])
    assert ids.tolist() == [11 + offset]
    assert embeddings[:, 0].tolist() == [101 + offset]


def test_live_prefill_output_does_not_alias_cached_prepared_embeddings():
    model = _model_with_prepared_units()
    prepared = torch.tensor([[101.0], [102.0]])
    model._minicpmo45_duplex_data_plane_helper._stage_prefill_embeddings_only = lambda *a, **kw: {
        "success": True,
        "inputs_embeds": prepared,
        "input_token_ids": [11, 12],
    }
    _, output, _ = _preprocess(model, seq=1, ids=[0, 0], offset=0, prompt=[0, 0])
    output.zero_()
    assert prepared[:, 0].tolist() == [101, 102]
    _, retried, _ = _preprocess(model, seq=1, ids=[0], offset=1, prompt=[0, 0])
    assert retried[:, 0].tolist() == [102]


@pytest.mark.parametrize("corrupt,stage", [(False, "llm"), (True, "llm"), (True, "tts")])
def test_embedding_oracle_checks_real_preprocess_outputs(monkeypatch, corrupt, stage):
    monkeypatch.delenv("VLLM_OMNI_TEST_NATIVE_INPUT_FAULT", raising=False)
    path = Path(__file__).resolve().parents[1] / "dfx/reliability/fault_injection/native_input_safety/sitecustomize.py"
    patch = runpy.run_path(str(path))["_patch"]
    patch.__globals__["_MODE"] = "embedding_oracle"
    model = _model_with_prepared_units()
    original = model.preprocess

    class OracleModel:
        model_stage = stage

        def preprocess(self, input_ids, **kwargs):
            result = original(input_ids, **kwargs)
            if corrupt:
                result[1].add_(1)
            return result

        def __getattr__(self, name):
            return getattr(model, name)

    patch(SimpleNamespace(MiniCPMO45OmniForConditionalGeneration=OracleModel))
    if corrupt and stage == "llm":
        with pytest.raises(AssertionError):
            _preprocess(OracleModel(), seq=1, ids=[0, 0], offset=0, prompt=[0, 0])
    else:
        _preprocess(OracleModel(), seq=1, ids=[0, 0], offset=0, prompt=[0, 0])


def test_failed_preprocess_uses_v028_embedding_interface_without_a_template():
    class Model:
        def __init__(self):
            self.calls = []

        def preprocess(self, *, input_ids, input_embeds, request_id):
            self.calls.append(request_id)
            if request_id == "bad":
                raise ModelInputError("native_model_state_recompute_unsupported")
            return input_ids, self.embed_input_ids(input_ids), {}

        def embed_input_ids(self, input_ids):
            return input_ids.float().unsqueeze(-1).expand(-1, 2)

    model = Model()
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner.model = model
    errors: dict[str, str] = {}
    for _ in range(2):
        _, embeddings, payload = runner._preprocess_request(
            "bad", torch.tensor([1]), None, {"request_id": "bad"}, errors
        )
        assert embeddings.shape == (1, 2)
        assert torch.count_nonzero(embeddings) == 0
        assert payload == {}
    _, healthy, _ = runner._preprocess_request("good", torch.tensor([2]), None, {"request_id": "good"}, errors)
    assert healthy.tolist() == [[2.0, 2.0]]
    assert model.calls == ["bad", "good"]
    assert errors == {"bad": "native_model_state_recompute_unsupported"}


def test_input_history_is_bounded_and_owns_a_copy():
    history = DuplexPromptHistory(max_tokens=4, max_bytes=8)
    original = torch.tensor([[1.0], [2.0]])
    history.append(prompt_len=2, embeddings=original, token_ids=[1, 2], identity=(1,))
    original.zero_()
    assert history.spans[0].embeddings[:, 0].tolist() == [1.0, 2.0]
    with pytest.raises(ModelInputError, match="byte limit"):
        history.append(prompt_len=3, embeddings=torch.ones(1, 1), token_ids=[3], identity=(2,))
    with pytest.raises(ModelInputError, match="context limit"):
        history.append(prompt_len=5, embeddings=torch.ones(1, 1), token_ids=[3], identity=(2,))
    assert history.num_bytes == 8
    assert len(history.spans) == 1


def test_input_history_rejects_missing_prefix_instead_of_padding_it():
    history = DuplexPromptHistory(max_tokens=8)
    with pytest.raises(ModelInputError, match="missing prefix"):
        history.append(prompt_len=4, embeddings=torch.ones(2, 1), token_ids=[1, 2], identity=(1,))
    assert not history.spans


def test_input_history_rejects_surplus_slots_in_current_unit():
    history = DuplexPromptHistory(max_tokens=8)
    history.append(prompt_len=2, embeddings=torch.ones(2, 1), token_ids=[1, 2], identity=(1,))
    with pytest.raises(ModelInputError, match="reserved=3, prepared=1"):
        history.append(prompt_len=5, embeddings=torch.ones(1, 1), token_ids=[3], identity=(2,), expected_start=2)
    assert len(history.spans) == 1


@pytest.mark.parametrize("stage,include_hidden", [("llm", True), ("tts", False)])
def test_only_thinker_handoff_transports_hidden_states(stage, include_hidden):
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = stage
    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.model = model
    assert runner._model_omni_pooler_payload_include_hidden() is include_hidden


def test_terminal_cleanup_releases_only_the_finished_physical_history():
    model = _model_with_prepared_units()
    _preprocess(model, seq=1, ids=[0, 0], offset=0, prompt=[0, 0])
    peer = DuplexPromptHistory(max_tokens=8)
    model._minicpmo45_duplex_input_histories["peer"] = peer
    model.model = SimpleNamespace()
    model.on_requests_finished({"physical-0"})
    assert model._minicpmo45_duplex_input_histories == {"peer": peer}


def test_failed_encoder_preparation_raises_request_local_error():
    model = _model_with_prepared_units()
    model._minicpmo45_duplex_data_plane_helper._stage_prefill_embeddings_only = lambda *a, **kw: {
        "success": False,
        "reason": "encoder emitted no stable frames",
    }
    with pytest.raises(ModelInputError, match="native_duplex_prefill_failed"):
        _preprocess(model, seq=1, ids=[0, 0], offset=0, prompt=[0, 0])
    assert not getattr(model, "_minicpmo45_duplex_input_histories", {})


def test_runner_latches_preparation_failure_without_failing_peer():
    calls = []

    def prepare(*, input_ids, input_embeds, request_id):
        calls.append(request_id)
        if request_id == "bad":
            raise ModelInputError("native_duplex_prefill_failed: empty encoder output")
        return input_ids, input_embeds, {}

    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(preprocess=prepare)
    errors: dict[str, str] = {}
    for _ in range(2):
        _, embeds, _ = runner._preprocess_request(
            "bad", torch.tensor([1]), torch.ones(1, 2), {"request_id": "bad"}, errors
        )
        assert torch.count_nonzero(embeds) == 0
    _, good, _ = runner._preprocess_request("good", torch.tensor([2]), torch.ones(1, 2), {"request_id": "good"}, errors)
    assert calls == ["bad", "good"]
    assert set(errors) == {"bad"}
    assert torch.equal(good, torch.ones(1, 2))


def test_failed_input_slot_mask_preserves_peer_and_all_group_views():
    runner = GPUARModelRunner.__new__(GPUARModelRunner)
    runner.query_start_loc = SimpleNamespace(cpu=torch.tensor([0, 2, 5]))
    slots = {0: torch.arange(6), 1: torch.arange(6) + 10}
    layer_view = slots[0][1:4]
    runner._mask_failed_input_kv_slots({"bad": "failed"}, ["bad", "good"], slots)
    assert slots[0].tolist() == [-1, -1, 2, 3, 4, 5]
    assert slots[1].tolist() == [-1, -1, 12, 13, 14, 15]
    assert layer_view.tolist() == [-1, 2, 3]


def test_sampling_snapshot_refresh_preserves_rng_until_seed_changes():
    runner = OmniGPUModelRunner.__new__(OmniGPUModelRunner)
    runner.device = torch.device("cpu")
    generator = torch.Generator().manual_seed(42)
    request = SimpleNamespace(sampling_params=SamplingParams(seed=42), generator=generator)
    runner.requests = {"req": request}
    replacement = SamplingParams(seed=42, temperature=0.8, max_tokens=7)
    runner._update_native_sampling_params("req", replacement)
    assert request.sampling_params is replacement
    assert request.generator is generator
    runner._update_native_sampling_params("req", SamplingParams(seed=43))
    assert request.generator is not generator
    assert request.generator.initial_seed() == 43


def test_duplex_sampling_rows_skip_partial_prefills_and_failed_inputs():
    import numpy as np

    from vllm_omni.model_executor.duplex_sampling import DuplexSamplingHelper

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["prefill", "decode"]),
        requests={},
        discard_request_mask=SimpleNamespace(np=np.array([True, False])),
        model_intermediate_buffer={rid: {"duplex": {"data_plane": True}} for rid in ("prefill", "decode")},
    )
    helper = DuplexSamplingHelper()
    helper.active_request_ids = {"prefill", "decode"}
    assert [row.sampling_enabled for row in helper.rows(runner)] == [False, True]
    runner._omni_failed_input_requests = {"decode": "failed"}
    assert [row.sampling_enabled for row in helper.rows(runner)] == [False, False]


def test_native_sampler_does_not_advance_discarded_prefill_policy(monkeypatch):
    model = _model_with_prepared_units()
    model._minicpmo45_active_duplex_rows = [0, 1]
    model._minicpmo45_discarded_duplex_rows = {0}
    monkeypatch.setattr(model, "_minicpmo45_native_duplex_token_ids", lambda: {"unit_token_id": 1})
    sampled, recorded = [], []

    def sample(logits, metadata, *, row_idx, token_ids):
        sampled.append(row_idx)
        return 2

    monkeypatch.setattr(model, "_sample_minicpmo45_native_duplex_row", sample)
    monkeypatch.setattr(model, "_record_minicpmo45_duplex_terminator", lambda row, *args: recorded.append(row))
    result = model.sample(torch.zeros(2, 4), SimpleNamespace())
    assert result.sampled_token_ids.tolist() == [[0], [2]]
    assert sampled == recorded == [1]


def test_mixed_chat_batch_preserves_native_policy_and_request_rng(monkeypatch):
    model = _model_with_prepared_units()
    model._minicpmo45_active_duplex_rows = [0]
    native_rng = torch.Generator().manual_seed(42)
    chat_rng = torch.Generator().manual_seed(43)
    native_before = native_rng.get_state().clone()
    chat_before = chat_rng.get_state().clone()
    # Only fields used by the fake standard sampler are needed in this L1
    # contract check, but use a dataclass so replace() has real semantics.
    metadata = SamplingMetadata.__new__(SamplingMetadata)
    for name in SamplingMetadata.__dataclass_fields__:
        setattr(metadata, name, None)
    metadata.generators = {0: native_rng, 1: chat_rng}
    sampled_rows = []
    logits = torch.zeros(2, 8)
    logits[0, 2] = 10.0

    def standard_sample(values, meta):
        # vLLM may modify logits in place and advance every row's generator.
        values.zero_()
        for generator in meta.generators.values():
            torch.rand((), generator=generator)
        return SamplerOutput(sampled_token_ids=torch.tensor([[7], [6]], dtype=torch.int32), logprobs_tensors=None)

    def native_sample(values, meta, *, row_idx, token_ids):
        sampled_rows.append(row_idx)
        assert values[0, 2] == 10.0
        assert meta.generators[0] is native_rng
        return 2

    model.__dict__["sampler"] = standard_sample
    monkeypatch.setattr(model, "_minicpmo45_native_duplex_token_ids", lambda: {"unit_token_id": 1})
    monkeypatch.setattr(model, "_sample_minicpmo45_native_duplex_row", native_sample)
    monkeypatch.setattr(model, "_record_minicpmo45_duplex_terminator", lambda *a: None)
    result = model.sample(logits, metadata)
    assert result is not None, "mixed chat batch bypassed MiniCPM native policy"
    assert sampled_rows == [0]
    assert result.sampled_token_ids.tolist() == [[2], [6]]
    assert torch.equal(native_rng.get_state(), native_before)
    assert not torch.equal(chat_rng.get_state(), chat_before)
    assert logits[0, 2] == 10.0


def test_greedy_native_row_never_randomizes_boundary_in_mixed_batch(monkeypatch):
    model = _model_with_prepared_units()
    metadata = SimpleNamespace(
        all_greedy=False,
        temperature=torch.tensor([0.0, 0.8]),
        top_k=torch.tensor([-1, 25]),
        top_p=torch.tensor([1.0, 0.85]),
        output_token_ids=[[], []],
        generators={},
    )
    token_ids = {"unit_token_id": 1, "listen_token_id": 2, "chunk_eos_token_id": 3, "tts_bos_token_id": 4}
    logits = torch.zeros(1, 8)
    logits[0, 6] = 20.0
    monkeypatch.setattr(model, "_maybe_cut_minicpmo45_native_duplex_text_chunk", lambda sampled, *a: sampled)

    def no_random(*args, **kwargs):
        raise AssertionError("greedy request used multinomial for a chunk boundary")

    monkeypatch.setattr(torch, "multinomial", no_random)
    assert model._sample_minicpmo45_native_duplex_row(logits, metadata, row_idx=0, token_ids=token_ids) == 6


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_native_sampling_uses_cpu_request_snapshot_without_gpu_scalar_reads(monkeypatch, temperature):
    from vllm_omni.model_executor.duplex_sampling import DuplexSamplingHelper

    model = _model_with_prepared_units()
    token_ids = {"unit_token_id": 1, "listen_token_id": 2, "chunk_eos_token_id": 3}
    model._minicpmo45_native_duplex_token_ids_cache = token_ids
    params = SamplingParams(temperature=temperature, top_k=5, top_p=0.85)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["req"]),
        requests={"req": SimpleNamespace(sampling_params=params)},
        model_intermediate_buffer={"req": {"duplex": {"data_plane": True}}},
    )
    helper = DuplexSamplingHelper()
    helper.active_request_ids = {"req"}
    rows = helper.rows(runner)
    assert (rows[0].temperature, rows[0].top_k, rows[0].top_p) == (
        params.temperature,
        params.top_k,
        params.top_p,
    )

    def no_gpu_scalar(*args, **kwargs):
        raise AssertionError("native sampling read a GPU metadata scalar")

    monkeypatch.setattr(model, "_sampling_metadata_value", no_gpu_scalar)
    monkeypatch.setattr(model, "_maybe_cut_minicpmo45_native_duplex_text_chunk", lambda sampled, *a: sampled)
    metadata = SimpleNamespace(all_greedy=False, output_token_ids=[[]], generators={})
    logits = torch.full((1, 8), -100.0)
    logits[0, 6] = 100.0
    model.prepare_duplex_sampling(logits, metadata, rows)
    assert model._sample_minicpmo45_native_duplex_row(logits, metadata, row_idx=0, token_ids=token_ids) == 6
    # A later append may update sampling on the same physical request.
    runner.requests["req"].sampling_params = SamplingParams(temperature=0.6, top_k=3, top_p=0.9)
    refreshed = helper.rows(runner)
    assert rows[0].temperature == params.temperature
    assert refreshed[0].temperature == 0.6
    model.prepare_duplex_sampling(logits, metadata, ())
    assert model._minicpmo45_duplex_row_sampling == {}


@pytest.mark.parametrize("async_output", [False, True])
def test_failed_input_sample_mask_handles_deferred_async_tokens(async_output):
    tokens = [] if async_output else [[11], [12], [13]]
    invalid = [0]
    OmniGPUModelRunner._suppress_failed_input_samples(
        {"bad": "invalid embeddings", "gone": "finished"},
        {"prefill": 0, "bad": 1, "good": 2},
        tokens,
        invalid,
    )
    assert invalid == [0, 1]
    assert tokens == ([] if async_output else [[11], [], [13]])


@pytest.mark.parametrize("terminator", [2, 3, 4, 6])
def test_native_async_lookahead_cannot_overwrite_unit_terminator(monkeypatch, terminator):
    model = _model_with_prepared_units()
    model._minicpmo45_native_duplex_token_ids_cache = {
        "unit_token_id": 1,
        "listen_token_id": 2,
        "chunk_eos_token_id": 3,
        "chunk_tts_eos_token_id": 4,
        "turn_eos_token_id": 5,
    }
    state = model._minicpmo45_duplex_data_plane_helper.sessions[("sid", 0)]
    if terminator == 6:
        model.config = SimpleNamespace(gander_unit8=True)
        model._minicpmo45_duplex_data_plane_helper.sessions[("req", 0)] = state
        model._minicpmo45_native_duplex_token_ids_cache["interrupt_token_id"] = 6
    row = DuplexSamplingRow(0, "req", "sid", 0, 1, {"is_speech": True}, 20)
    metadata = SimpleNamespace()
    calls = []

    def sample(*args, **kwargs):
        calls.append(1)
        return terminator

    monkeypatch.setattr(model, "_sample_minicpmo45_native_duplex_row", sample)
    model.prepare_duplex_sampling(torch.zeros(1, 8), metadata, (row,))
    assert model.sample(torch.zeros(1, 8), metadata).sampled_token_ids.item() == terminator
    before = vars(state).copy()
    # This speculative frame is later discarded by the scheduler, but the
    # worker runs it before the scheduler sees the previous stop token.
    model.prepare_duplex_sampling(torch.zeros(1, 8), metadata, (row,))
    model.sample(torch.zeros(1, 8), metadata)
    assert len(calls) == 1
    assert vars(state) == before
    assert state.pending_terminator_token == terminator
    # A new unit on the same physical request must be allowed to sample.
    model.prepare_duplex_sampling(torch.zeros(1, 8), metadata, (replace(row, seq=2),))
    model.sample(torch.zeros(1, 8), metadata)
    assert len(calls) == 2
    model.model = SimpleNamespace()
    model.on_requests_finished({"req"})
    assert not model._minicpmo45_closed_duplex_units


def test_turn_eos_does_not_fence_required_following_chunk_eos(monkeypatch):
    model = _model_with_prepared_units()
    model._minicpmo45_native_duplex_token_ids_cache = {
        "unit_token_id": 1,
        "listen_token_id": 2,
        "chunk_eos_token_id": 3,
        "turn_eos_token_id": 5,
    }
    row = DuplexSamplingRow(0, "req", "sid", 0, 1, {"is_speech": False}, 20)
    samples = iter([5, 3])
    monkeypatch.setattr(model, "_sample_minicpmo45_native_duplex_row", lambda *a, **kw: next(samples))
    for expected in (5, 3):
        model.prepare_duplex_sampling(torch.zeros(1, 8), SimpleNamespace(), (row,))
        assert model.sample(torch.zeros(1, 8), SimpleNamespace()).sampled_token_ids.item() == expected


@pytest.mark.parametrize("in_flight", [0, 8])
def test_preemption_fault_exercises_sync_and_async_in_flight_requests(monkeypatch, in_flight):
    monkeypatch.delenv("VLLM_OMNI_TEST_NATIVE_INPUT_FAULT", raising=False)
    fault_path = (
        Path(__file__).resolve().parents[1] / "dfx/reliability/fault_injection/native_input_safety/sitecustomize.py"
    )
    patch = runpy.run_path(str(fault_path))["_patch"]
    patch.__globals__["_MODE"] = "preempt"
    request = SimpleNamespace(
        request_id="req",
        streaming_prompt_continuous=True,
        model_intermediate_buffer={"duplex": {"seq": 2}},
        num_computed_tokens=89,
        num_in_flight_tokens=in_flight,
        num_preemptions=0,
    )

    class Scheduler:
        def __init__(self):
            self.running = [request]

        def schedule(self):
            return "scheduled"

        def _preempt_request(self, req, timestamp, *, drop_stale_output):
            assert drop_stale_output is True
            assert req is request
            assert timestamp > 0
            req.num_computed_tokens = 0
            req.num_preemptions += 1

    patch(SimpleNamespace(OmniARScheduler=Scheduler))
    scheduler = Scheduler()
    assert scheduler.schedule() == "scheduled"
    assert request.num_preemptions == 1
    assert request.num_computed_tokens == 0
    assert scheduler.schedule() == "scheduled"
    assert request.num_preemptions == 1


@pytest.mark.parametrize(
    "path",
    [
        "vllm_omni/worker/gpu_ar_model_runner.py",
        "vllm_omni/platforms/npu/worker/npu_ar_model_runner.py",
    ],
)
def test_ar_platforms_wire_input_failure_mask_and_error_carrier(path):
    # NPU dependencies are not available on CUDA CI. Verify the common hooks
    # are wired without claiming a native Ascend execution result.
    tree = ast.parse((Path(__file__).resolve().parents[2] / path).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attributes = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
    assert {"_mask_failed_input_kv_slots", "_mask_failed_input_logits"} <= attributes
    outputs = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "OmniModelRunnerOutput"]
    assert any("model_input_errors" in {kw.arg for kw in node.keywords} for node in outputs)
