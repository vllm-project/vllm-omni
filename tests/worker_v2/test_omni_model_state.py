# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniModelState: mixed-batch reorder, state isolation/slot reuse, MTP
graph/eager paths with per-request seed independence, async snapshot ownership."""

from collections import namedtuple
from contextlib import nullcontext
from dataclasses import replace
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.output_templates import OmniOutput, OwnedBatchTensor
from vllm_omni.worker_v2.model_states.mtp_model_state import FrameBatch
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState, _make_safe_get_rope

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_prepare_attn_forwards_release_model_specific_metadata():
    state = object.__new__(OmniModelState)
    metadata = object()
    output = object()
    batch = SimpleNamespace()
    with patch.object(DefaultModelState, "prepare_attn", return_value=output) as prepare:
        assert state.prepare_attn(batch, None, (), None, [], None, model_specific_attn_metadata=metadata) is output
    assert prepare.call_args.kwargs["model_specific_attn_metadata"] is metadata


class _DummyInputBatch:
    is_prefilling_np: np.ndarray
    req_ids: list[str]

    input_ids: SimpleNamespace
    query_start_loc: torch.Tensor

    def __init__(self, indices, *, num_computed_tokens_cpu=None):
        self.idx_mapping_np = indices
        self.num_reqs = len(indices)
        self.num_scheduled_tokens = [1] * len(indices)
        self.query_start_loc_np = list(range(len(indices)))
        if num_computed_tokens_cpu is not None:
            self.num_computed_tokens_cpu = np.array(num_computed_tokens_cpu, dtype=np.int32)


def _make_state(max_num_reqs=4, has_preprocess=False, has_postprocess=False, have_multimodal_outputs=False):
    state = object.__new__(OmniModelState)
    model = MagicMock()
    model.stream_decoder = None
    model.has_preprocess = has_preprocess
    model.has_postprocess = has_postprocess
    model.have_multimodal_outputs = have_multimodal_outputs
    model.gpu_resident_buffer_keys = set()
    model.batched_gpu_staging_keys = set()
    model.mtp_accepts_per_row_generators = False
    model.mtp_accepts_req_infos = False
    model.mtp_output_key = ("codes", "audio")
    model.mtp_validity_key = None
    model.mtp_sampling_params = {}
    model.get_mtp_seed = lambda params: (getattr(params, "extra_args", None) or {}).get("test_seed")
    model.preprocess_batch_mrv2 = None
    model.preprocess_prefill_rows_mrv2 = None
    model.preprocess_prefill_updates_owned = False
    model.preprocess_decode_batch_mrv2 = None
    model.preprocess_decode_batch = None
    model.postprocess_batch_mrv2 = None
    model.eager_decode_settled = None
    model.eager_frames_need_postprocess = True
    state.model = model
    state.scheduler_config = SimpleNamespace(max_num_seqs=max_num_reqs)
    state.vllm_config = SimpleNamespace(model_config=SimpleNamespace(subtalker_sampling_params={}))
    state.has_preprocess = has_preprocess
    state.has_postprocess = has_postprocess
    state.have_multimodal_outputs = have_multimodal_outputs

    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

    state.intermediate_buffer = OmniIntermediateBuffer(max_num_reqs)
    state._static_inputs_embeds = None
    from vllm_omni.worker_v2.model_states.mtp_model_state import MTPState

    state._embed_dim = 0
    state.device = torch.device("cpu")
    state.model.has_talker_mtp = False
    state._mtp_state = MTPState(state)
    # Individual tests install their model hook after constructing the fixture.
    state._mtp_state._raw_mtp = None
    return state


def _add(state, req_id="r1", idx=0, **req_kwargs):
    with patch.object(DefaultModelState, "add_request", return_value=None):
        state.add_request(idx, SimpleNamespace(req_id=req_id, mm_features=[], **req_kwargs))


def _fill_buffers(state, *req_ids):
    for idx, req_id in enumerate(req_ids):
        state.intermediate_buffer.buffers[idx] = {"req_id": req_id}


def _seeded(seed):
    return SimpleNamespace(extra_args={"test_seed": seed}, seed=None)


def test_preprocess_forwards_request_seed_to_model() -> None:
    state = _make_state(has_preprocess=True)
    state.intermediate_buffer.buffers[0] = {
        "req_id": "seeded",
        "sampling_params": SimpleNamespace(seed=42, max_tokens=128),
    }
    captured = []

    def preprocess(input_ids, input_embeds, **info):
        captured.append(info)
        return input_ids, input_embeds, {}

    state.model.preprocess = preprocess
    state.run_preprocess(
        _DummyInputBatch([0], num_computed_tokens_cpu=[0]),
        {"input_ids": torch.tensor([1]), "inputs_embeds": torch.zeros(1, 4)},
        SimpleNamespace(prompt_len=np.array([1], dtype=np.int32), num_computed_tokens=None),
    )
    assert captured[0]["_omni_seed"] == 42
    assert captured[0]["_omni_max_tokens"] == 128


def _init_static(state, bsz, dim=3):
    state._mtp_state._mtp_input_ids = torch.zeros(bsz, dtype=torch.long)
    state._mtp_state._mtp_input_embeds = torch.zeros((bsz, dim))
    state._mtp_state._mtp_hidden = torch.zeros((bsz, dim))
    state._mtp_state._mtp_text_step = torch.zeros((bsz, dim))
    state._mtp_state._mtp_offsets = torch.zeros(bsz, dtype=torch.long)


@pytest.fixture(autouse=True)
def _fwd_ctx():
    with patch("vllm.forward_context.set_forward_context", return_value=nullcontext()):
        yield


@pytest.mark.parametrize(
    "req_id,expect_validity",
    [("_warmup_0_", True), ("_v2_mixed_warmup_decode_", True), ("_v2_mixed_warmup_prefill_", True), ("r1", False)],
)
def test_add_request_declared_validity_only_for_warmup(req_id, expect_validity):
    # Real requests must not be masked with a fabricated validity key.
    state = _make_state()
    state.model.mtp_validity_key = ("meta", "codec_frame_valid")
    _add(state, req_id, 0)
    buf = state.intermediate_buffer.buffers[0]
    assert ("meta" in buf) is expect_validity
    if expect_validity:
        assert buf["meta"]["codec_frame_valid"].item() is False


@pytest.mark.parametrize("req_id", ["_warmup_0_", "_v2_mixed_warmup_decode_", "_v2_mixed_warmup_prefill_"])
def test_warmup_skips_model_preprocess_and_postprocess(req_id):
    state = _make_state(has_preprocess=True, has_postprocess=True)
    _fill_buffers(state, req_id)
    state.model.preprocess.side_effect = AssertionError("warmup has no model conditioning")
    state.model.postprocess.side_effect = AssertionError("warmup must not update model state")
    state.model.postprocess_batch_mrv2 = MagicMock(side_effect=AssertionError("warmup must not use batch postprocess"))
    batch = _DummyInputBatch([0], num_computed_tokens_cpu=[0])
    inputs = {"input_ids": torch.tensor([1]), "inputs_embeds": torch.zeros(1, 4)}

    state.run_preprocess(batch, inputs)
    state.run_postprocess(torch.zeros(1, 4), batch)

    state.model.preprocess.assert_not_called()
    state.model.postprocess.assert_not_called()
    state.model.postprocess_batch_mrv2.assert_not_called()
    assert state.intermediate_buffer.buffers[0] == {"req_id": req_id}


@pytest.mark.parametrize("mode", ["index", "req_id", "unknown"])
def test_remove_request_state_isolation(mode):
    state = _make_state()
    _add(state, "r1", 0)
    state._mtp_state._mtp_generators["r1"] = torch.Generator(device="cpu")
    _add(state, "r2", 1)
    eager = state._mtp_state
    eager.embeds = torch.zeros(4, 2)
    eager.ready = {0: "r1", 1: "r2"}
    eager.mark_settled(0, "r1")
    eager.mark_settled(1, "r2")
    if mode == "unknown":
        state.remove_request("missing")
        assert state.intermediate_buffer.buffers[0]["req_id"] == "r1"
        assert "r1" in state._mtp_state._mtp_generators
        return
    state.remove_request(0 if mode == "index" else "r1")
    assert state.intermediate_buffer.buffers[0] == {}
    assert "r1" not in state._mtp_state._mtp_generators  # recycled slot inherits no seed stream

    assert eager.ready == {1: "r2"}
    assert eager.settled.tolist() == [None, "r2", None, None]
    _add(state, "replacement", 0)
    assert eager.settled[0] is None and 0 not in eager.ready


def test_output_spans_follow_reordered_mixed_batch():
    # batch=[2, 0]: prefill (3 tokens) reordered ahead of decode (1 token).
    state = _make_state(have_multimodal_outputs=True)
    batch = _DummyInputBatch([2, 0])
    batch.req_ids = ["prefill", "decode"]
    batch.num_scheduled_tokens = [3, 1]
    batch.query_start_loc_np = [0, 3]
    state.intermediate_buffer.buffers[2] = {"req_id": "prefill"}
    state.intermediate_buffer.buffers[0] = {"req_id": "decode"}
    seen = {}
    state.model.make_omni_output = lambda hidden, **kwargs: (
        seen.update(kwargs) or OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
    )
    state.postprocess_model_output(torch.zeros(4, 2), batch, SimpleNamespace())
    assert seen["request_token_spans"] == [(0, 3), (3, 4)]
    assert [info["req_id"] for info in seen["model_intermediate_buffer"]] == ["prefill", "decode"]


@pytest.mark.parametrize("hook_name", ["preprocess_decode_batch_mrv2", "preprocess_decode_batch", "legacy"])
def test_decode_cohort_uses_batch_hook(hook_name):
    state = _make_state(max_num_reqs=2, has_preprocess=True)
    state.intermediate_buffer.buffers[0] = {"req_id": "r1", "meta": {"step": 1}}
    state.intermediate_buffer.buffers[1] = {"req_id": "r2", "meta": {"step": 2}}
    batch_calls = []

    def preprocess_decode_batch(*, input_ids, input_embeds, req_infos):
        batch_calls.append(list(req_infos))
        return (
            input_ids + 10,
            input_embeds + 20,
            torch.tensor([[31.0, 32.0], [41.0, 42.0]]),
            torch.tensor([[51.0, 52.0], [61.0, 62.0]]),
            [{"meta": {"step": 11}}, {"meta": {"step": 12}}],
        )

    if hook_name == "legacy":
        # Legacy V1 hook without input_embeds is wrapped by the adapter.
        state.model.preprocess_decode_batch = lambda *, input_ids, req_infos: preprocess_decode_batch(
            input_ids=input_ids, input_embeds=model_inputs["inputs_embeds"], req_infos=req_infos
        )
    else:
        setattr(state.model, hook_name, preprocess_decode_batch)
        if hook_name == "preprocess_decode_batch_mrv2":
            state.model.preprocess_decode_batch = MagicMock(
                side_effect=AssertionError("MRV2 hook must take precedence")
            )
    state.model.preprocess = MagicMock(side_effect=AssertionError("decode requests must use the batch hook"))
    seen_text_steps = []

    def capture(batches, embeds, batch, text_steps):
        assert batches == [(0, 0), (1, 1)]
        seen_text_steps.append(text_steps)

    state._has_mtp = True
    state._mtp_state.apply_frame_embeddings = capture
    model_inputs = {
        "input_ids": torch.tensor([101, 202], dtype=torch.long),
        "inputs_embeds": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }
    state.run_preprocess(
        _DummyInputBatch([0, 1], num_computed_tokens_cpu=[1, 1]),
        model_inputs,
        SimpleNamespace(prompt_len=np.array([1, 1], dtype=np.int32)),
    )
    state.model.preprocess.assert_not_called()
    assert [info["req_id"] for info in batch_calls[0]] == ["r1", "r2"]
    assert torch.equal(model_inputs["input_ids"], torch.tensor([111, 212]))
    assert state.intermediate_buffer.buffers[0]["meta"]["step"] == 11
    assert torch.equal(seen_text_steps[0], torch.tensor([[51.0, 52.0], [61.0, 62.0]]))


def test_static_decode_embeddings_refresh_from_input_ids():
    # FULL-graph replay reads the static buffer: it must hold fresh embeddings.
    state = _make_state(has_preprocess=True)
    static_embeds = torch.zeros(1, 4)
    state._static_inputs_embeds = static_embeds
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    state.model.embed_input_ids = lambda input_ids: expected.to(input_ids.device)
    state.model.preprocess = lambda input_ids, input_embeds, **_info: (input_ids, input_embeds, {})
    state.intermediate_buffer.buffers[0] = {"req_id": "r1"}
    state.run_preprocess(
        _DummyInputBatch([0]),
        {"input_ids": torch.tensor([1049], dtype=torch.long), "inputs_embeds": static_embeds[:1]},
    )
    assert torch.equal(static_embeds, expected)
    original = torch.tensor([[1.0, 2.0]])
    assert OmniModelState._preprocess_result_needs_writeback(original, original) is False
    assert OmniModelState._preprocess_result_needs_writeback(original, original.view_as(original)) is True


@pytest.mark.parametrize(
    "declared,mm_inputs,encoder_dim,reuses",
    [(True, True, 4, True), (False, True, 4, False), (True, False, 4, False), (True, True, 6, False)],
)
def test_preprocess_model_reuses_encoder_buffer_only_when_declared(
    monkeypatch, declared, mm_inputs, encoder_dim, reuses
):
    def default_init(self, vllm_config, model, encoder_cache, device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.device = device
        self.max_num_reqs = 2
        self.max_num_tokens = 8
        self.dtype = torch.float32
        self.supports_mm_inputs = encoder_cache is not None
        if self.supports_mm_inputs:
            self.encoder_runner = SimpleNamespace(inputs_embeds=torch.zeros(8, encoder_dim))

    class Model(torch.nn.Module):
        has_preprocess = True
        preprocess_keeps_mm_inputs = declared

        def embed_input_ids(self, input_ids):
            return torch.zeros(input_ids.shape[0], 4)

    monkeypatch.setattr(DefaultModelState, "__init__", default_init)
    config = SimpleNamespace(model_config=SimpleNamespace(), scheduler_config=SimpleNamespace(max_num_seqs=2))
    state = OmniModelState(config, Model(), object() if mm_inputs else None, torch.device("cpu"))
    assert state.preprocess_keeps_mm_inputs is reuses
    assert state._static_inputs_embeds.shape == (8, 4)
    encoder_runner = getattr(state, "encoder_runner", None)
    assert (encoder_runner is not None and state._static_inputs_embeds is encoder_runner.inputs_embeds) is reuses


def test_preprocess_on_encoder_embeddings_keeps_media_rows():
    """With native multimodal inputs kept, the encoder buffer already holds this step's rows."""
    state = _make_state(has_preprocess=True)
    encoder_embeds = torch.tensor([[7.0, 7.0], [1.0, 1.0]])  # row 0: image features, row 1: text
    state._static_inputs_embeds = encoder_embeds
    state.preprocess_keeps_mm_inputs = True
    state.model.embed_input_ids = lambda input_ids: torch.zeros(input_ids.shape[0], 2)
    duplex_embeds = torch.tensor([[3.0, 3.0]])

    def preprocess(input_ids, input_embeds, **info):
        # A duplex append replaces its own row; a chat row passes through.
        return input_ids, duplex_embeds if info["req_id"] == "duplex" else input_embeds, {}

    state.model.preprocess = preprocess
    _fill_buffers(state, "chat", "duplex")
    state.run_preprocess(
        _DummyInputBatch([0, 1]),
        {"input_ids": torch.tensor([151667, 42], dtype=torch.long), "inputs_embeds": encoder_embeds[:2]},
    )
    assert torch.equal(encoder_embeds, torch.tensor([[7.0, 7.0], [3.0, 3.0]]))


def test_moss_local_decode_runs_depth_predictor_and_routes_eos(mocker):
    """Exercise MRV2 dispatch through the real Local hook, output and logits."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import MossTTSLocalDepthTransformer
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSLocalTalkerForGeneration,
    )

    model = MossTTSLocalTalkerForGeneration.__new__(MossTTSLocalTalkerForGeneration)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(8, 4, _weight=torch.zeros(8, 4))
    model.hidden_size = 4
    model.n_vq = 2
    model.audio_pad_token_id = 8
    model.audio_assistant_slot_token_id = 2
    model.im_end_token_id = 3
    model.text_vocab_size = 8
    model.talker_mtp_output_key = ("audio_codes", "current")
    model.talker_mtp_graph_safe = False
    model.gpu_resident_buffer_keys = set()
    model.audio_lm_heads = model.audio_embeddings = model.local_text_lm_head = None
    model._audio_embed = lambda codes: codes[:, :1].expand(-1, 4).float()
    frame = mocker.Mock(return_value=(torch.tensor([True, False]), torch.tensor([[1, 2], [3, 4]])))
    model.local_transformer = mocker.Mock(spec=MossTTSLocalDepthTransformer, generate_frame=frame)

    state = _make_state(max_num_reqs=2, has_preprocess=True, have_multimodal_outputs=True)
    state.model = model
    for idx in range(2):
        state.intermediate_buffer.buffers[idx] = {
            "req_id": f"r{idx}",
            "audio_state": {"is_stopping": False},
            "hidden_states": {"last": torch.full((4,), float(idx + 1))},
        }
    batch = _DummyInputBatch([1, 0], num_computed_tokens_cpu=[1, 1])
    batch.req_ids = ["r1", "r0"]
    inputs = {"input_ids": torch.tensor([2, 2]), "inputs_embeds": torch.zeros(2, 4)}
    request_state = mocker.Mock(spec=RequestState, prompt_len=np.array([1, 1]), num_computed_tokens=None)
    infos = [state.intermediate_buffer.buffers[i] for i in [1, 0]]
    ids, base, hidden, control, _ = model.preprocess_decode_batch(
        input_ids=inputs["input_ids"],
        req_infos=infos,
    )
    new_embeds, codes = model.talker_mtp(ids, base, hidden, control)
    inputs["inputs_embeds"] = new_embeds
    for slot, row in zip([1, 0], codes):
        state.intermediate_buffer.buffers[slot]["audio_codes"] = {"current": row.unsqueeze(0)}

    frame.assert_called_once()
    torch.testing.assert_close(frame.call_args.args[0], torch.tensor([[2.0] * 4, [1.0] * 4]))
    assert frame.call_args.kwargs["temperature"] == 1.7
    assert frame.call_args.kwargs["top_k"] == 25
    assert frame.call_args.kwargs["top_p"] == 0.8
    torch.testing.assert_close(inputs["inputs_embeds"], torch.tensor([[1.0] * 4, [0.0] * 4]))
    _, payload = state.postprocess_model_output(torch.zeros(2, 4), batch, request_state)
    assert [codes.tolist() for codes in payload["codes"]["audio"]] == [[[1, 2]], [[8, 8]]]
    assert model.compute_logits(torch.zeros(2, 4)).argmax(-1).tolist() == [2, 3]
    # An explicit local seed must reach the request-owned MRV2 generator.
    params = SamplingParams(extra_args={"tts_local_seed": 17}, seed=99)
    assert state._mtp_state._get_mtp_generator("seeded", params, torch.device("cpu")).initial_seed() == 17


@pytest.mark.parametrize("owned", [False, True])
def test_batched_postprocess_gpu_snapshot_writeback(owned):
    # batch=[1, 0]: last-token indices follow the reordered batch, the scalar
    # hook is not called, and both rows share one snapshot storage so a later
    # graph replay cannot overwrite published rows. Owned producer batches
    # skip the extra snapshot while borrowed graph output still gets one.
    state = _make_state(max_num_reqs=2, has_postprocess=True)
    state.model.gpu_resident_buffer_keys = {("hidden_states", "last")}
    _fill_buffers(state, "r0", "r1")
    state.model.postprocess = MagicMock(side_effect=AssertionError("batch hook must replace scalar postprocess"))
    calls = []
    produced = []

    def postprocess_batch(*, hidden_states, last_token_indices):
        calls.append(last_token_indices.clone())
        gathered = hidden_states.index_select(0, last_token_indices)
        produced.append(gathered)
        if owned:
            return (("hidden_states", "last"), OwnedBatchTensor(gathered))
        return (("hidden_states", "last"), gathered)

    state.model.postprocess_batch_mrv2 = postprocess_batch
    batch = _DummyInputBatch([1, 0])
    batch.num_scheduled_tokens = [2, 2]
    batch.query_start_loc_np = [0, 2]
    batch.query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    state.run_postprocess(hidden, batch)

    state.model.postprocess.assert_not_called()
    assert torch.equal(calls[0], torch.tensor([1, 3], dtype=torch.int32))
    first = state.intermediate_buffer.buffers[0]["hidden_states"]["last"]
    second = state.intermediate_buffer.buffers[1]["hidden_states"]["last"]
    assert torch.equal(first, hidden[3])
    assert torch.equal(second, hidden[1])
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    if owned:
        # Owned output: rows view into the producer batch, no extra snapshot.
        assert first.untyped_storage().data_ptr() == produced[0].untyped_storage().data_ptr()
    else:
        # Borrowed/graph output: the snapshot owns its storage, not the producer.
        assert first.untyped_storage().data_ptr() != produced[0].untyped_storage().data_ptr()


def test_seed_independence_resolve_once_and_sampling_kwargs():
    state = _make_state(max_num_reqs=2)
    # vLLM sampling seed must not produce a talker generator.
    cpu = torch.device("cpu")
    assert state._mtp_state._get_mtp_generator("r1", SimpleNamespace(extra_args={}, seed=42), cpu) is None
    state._mtp_state._mtp_generators.clear()  # The following rows are new requests.
    # Same model-local seed reproduces identical uniforms regardless of batch makeup.
    state._mtp_state._mtp_sample_uniforms = torch.empty((2, 2, 4))
    assert torch.equal(
        state._mtp_state._prepare_mtp_sample_uniforms([torch.Generator().manual_seed(11)], 1).clone(),
        state._mtp_state._prepare_mtp_sample_uniforms([torch.Generator().manual_seed(11)], 1),
    )
    # Per-request generators are resolved once per step and ride sampling kwargs.
    state.model.mtp_sampling_params = {"do_sample": True, "temperature": 0.7}
    state.model.mtp_accepts_per_row_generators = True
    _init_static(state, 2)
    state.intermediate_buffer.buffers[0] = {"req_id": "r0", "sampling_params": _seeded(11)}
    state.intermediate_buffer.buffers[1] = {"req_id": "r1", "sampling_params": _seeded(22)}
    resolver = MagicMock(wraps=state._mtp_state._get_mtp_generator)
    state._mtp_state._get_mtp_generator = resolver

    def mtp(input_ids, input_embeds, last_hidden, text_step, **kwargs):
        assert (kwargs["do_sample"], kwargs["temperature"]) == (True, 0.7)
        assert all(isinstance(g, torch.Generator) for g in kwargs["generators"])
        return input_embeds + 10, torch.tensor([[1, 2, 3], [4, 5, 6]])

    state.model.mtp = mtp
    _, codes = state._mtp_state._mtp_forward(
        [0, 1], torch.tensor([101, 202]), torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3)
    )
    assert resolver.call_count == 2
    assert torch.equal(codes, torch.tensor([[1, 2, 3], [4, 5, 6]]))


def test_seeded_mtp_bypasses_outer_graph_runner():
    state = _make_state(max_num_reqs=2)
    raw_calls = []

    def mtp(*args, **kwargs):
        raw_calls.append(kwargs)
        return args[1], args[0].reshape(-1, 1)

    state.model.mtp = mtp
    state._mtp_state._mtp_runner = MagicMock(side_effect=AssertionError("seeded sampling must not replay the graph"))
    generators = [torch.Generator().manual_seed(11), torch.Generator().manual_seed(22)]
    state._mtp_state._call_mtp_runner(
        torch.tensor([101, 202]), torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3), generators=generators
    )
    assert raw_calls == [{"generators": generators}]
    state._mtp_state._mtp_runner.assert_not_called()


def test_mtp_forward_uses_dispatched_graph_descriptor():
    state = _make_state(max_num_reqs=4)

    class _FakeGraphRunner:
        def __call__(self, input_ids, input_embeds, last_hidden, text_step, **kwargs):
            assert input_ids.shape[0] == 4  # graph runs the padded descriptor size
            return input_embeds + 2, torch.arange(12, dtype=torch.long).reshape(4, 3)

    state._mtp_state._mtp_runner = _FakeGraphRunner()
    _init_static(state, 4)
    _fill_buffers(state, "r0", "r1")
    graph_desc = SimpleNamespace(cg_mode="FULL", num_tokens=4)
    set_ctx = MagicMock(return_value=nullcontext())
    with (
        patch(
            "vllm_omni.worker_v2.model_states.mtp_model_state.current_omni_platform.get_graph_wrapper_cls",
            return_value=_FakeGraphRunner,
        ),
        patch("vllm.forward_context.set_forward_context", set_ctx),
    ):
        _, codes = state._mtp_state._mtp_forward(
            [0, 1],
            torch.tensor([101, 202]),
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            lambda bsz: graph_desc,
        )
    _, kwargs = set_ctx.call_args
    assert kwargs["batch_descriptor"] is graph_desc
    assert kwargs["num_tokens"] == 4
    assert torch.equal(codes[:2], torch.tensor([[0, 1, 2], [3, 4, 5]]))


def test_steady_unseeded_mtp_replays_the_resolved_graph_entry():
    state = _make_state(max_num_reqs=4)
    _init_static(state, 4)
    _fill_buffers(state, "r0", "r1")
    state._mtp_state._mtp_sample_uniforms = torch.zeros(4, 2, 5)
    state._mtp_state._mtp_generators = {"r0": None, "r1": None, "r2": torch.Generator().manual_seed(3)}
    state._mtp_state._mtp_seeded = {"r2"}
    descriptor = namedtuple("descriptor", "cg_mode num_tokens")
    graph_desc = descriptor(CUDAGraphMode.FULL, 4)  # hashable, as vLLM's
    output = (torch.ones(4, 3), torch.arange(8).reshape(4, 2))
    entry = SimpleNamespace(cudagraph=MagicMock(), output=output)

    class _FakeGraphRunner:
        is_debugging_mode = False

        def __init__(self):
            self.concrete_cudagraph_entries = {graph_desc: entry}
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            entry.cudagraph.replay()
            return entry.output

    runner = state._mtp_state._mtp_runner = _FakeGraphRunner()
    inputs = (
        state._mtp_state._mtp_input_ids[:2],
        state._mtp_state._mtp_input_embeds[:2],
        state._mtp_state._mtp_hidden[:2],
        state._mtp_state._mtp_text_step[:2],
    )
    with patch(
        "vllm_omni.worker_v2.model_states.mtp_model_state.current_omni_platform.get_graph_wrapper_cls",
        return_value=_FakeGraphRunner,
    ):
        emb, codes = state._mtp_state._mtp_forward([0, 1], *inputs, lambda bsz: graph_desc, req_ids=["r0", "r1"])
        assert emb is output[0] and codes is output[1]
        assert runner.calls == 1 and state._mtp_state._mtp_replays == {2: (4, entry)}
        state._mtp_state._mtp_sample_uniforms.zero_()
        # Same size, unseeded rows: the entry replays without the wrapper; the
        # padded rows draw fresh noise as the wrapper path does.
        assert state._mtp_state._mtp_forward([0, 1], *inputs, lambda bsz: graph_desc, req_ids=["r0", "r1"]) is output
        assert runner.calls == 1 and entry.cudagraph.replay.call_count == 2
        assert (state._mtp_state._mtp_sample_uniforms > 0).all()
        # A row new this step resolves its generator in place and keeps the replay.
        _fill_buffers(state, "r0", "r3")
        state._mtp_state._mtp_forward([0, 1], *inputs, lambda bsz: graph_desc, req_ids=["r0", "r3"])
        assert runner.calls == 1 and state._mtp_state._mtp_generators["r3"] is None
        # A seeded row takes the wrapper path again.
        _fill_buffers(state, "r0", "r2")
        state._mtp_state._mtp_sample_uniforms = torch.zeros(4, 2, 5)
        state._mtp_state._mtp_forward([0, 1], *inputs, lambda bsz: graph_desc, req_ids=["r0", "r2"])
        assert runner.calls == 2


def test_rope_shim_propagates_type_error():
    # vLLM rope API drift (TypeError) must not be swallowed by the shim.
    def broken_get_rope(*_args, **_kwargs):
        raise TypeError("vLLM rope API drift")

    with pytest.raises(TypeError, match="rope API drift"):
        _make_safe_get_rope(broken_get_rope)(SimpleNamespace(uses_mrope=False), object())


def test_rope_shim_constructs_sequential_mrope_state():
    from vllm.platforms.cpu import CpuPlatform
    from vllm.v1.worker.gpu.mm.rope import RopeState, get_rope_state

    model = torch.nn.Module()
    config = SimpleNamespace(uses_mrope=True, mrope_num_dims=3)
    # Exercise the upstream constructor, including its argument contract, while
    # keeping the backing buffers on CPU for this unit test.
    with (
        patch("vllm.platforms.current_platform", CpuPlatform()),
        patch("vllm.v1.worker.gpu.buffer_utils.is_uva_available", return_value=False),
    ):
        rope = _make_safe_get_rope(get_rope_state)(
            config,
            model,
            max_num_reqs=2,
            max_num_tokens=8,
            max_model_len=16,
            device=torch.device("cpu"),
        )

    assert isinstance(rope, RopeState)
    assert rope.get_positions(4).shape == (3, 4)
    positions, delta = model.get_mrope_input_positions([11, 22, 33, 44], [])
    torch.testing.assert_close(torch.as_tensor(np.asarray(positions)).long(), torch.arange(4).expand(3, -1))
    assert delta == 0
    rope.init_prefill_positions(1, model, [11, 22, 33, 44], [])
    assert rope.prefill_delta.np[1] == 0
    assert rope.prefill_positions._staged_write_indices == [3, 4, 5]
    assert rope.prefill_positions._staged_write_contents == [0, 1, 2, 3] * 3


@pytest.mark.parametrize("output_key", [None, ("only_one",)])
def test_mtp_requires_model_declared_output_key(output_key):
    state = _make_state()
    model = SimpleNamespace(mtp=lambda: None, mtp_output_key=output_key)
    with pytest.raises(TypeError, match="must declare mtp_output_key"):
        state._mtp_state._init_mtp_runner(model)


_EAGER_DIM = 3
_CODEBOOK = 2048
_EOS = 2150


class _EagerBatch(_DummyInputBatch):
    def __init__(self, spans, indices=None):
        indices = list(range(len(spans))) if indices is None else indices
        super().__init__(indices)
        starts = [0]
        for n_tok in spans:
            starts.append(starts[-1] + n_tok)
        self.num_scheduled_tokens = list(spans)
        self.query_start_loc_np = starts[:-1]
        self.query_start_loc = torch.tensor(starts, dtype=torch.int32)


def _make_mtp_state(max_num_reqs=4):
    state = _make_state(max_num_reqs=max_num_reqs, has_preprocess=True)
    _init_static(state, max_num_reqs, dim=_EAGER_DIM)
    state._has_mtp = True
    state._mtp_state.embeds = torch.zeros((max_num_reqs, _EAGER_DIM))
    state._decode_preprocess = None
    state._mtp_state._mtp_sample_uniforms = None
    state.vllm_config.cache_config = SimpleNamespace(enable_prefix_caching=False)
    model = state.model
    model.first_frame_decoder = None
    model.mtp_frame_valid = lambda layer0: (layer0 >= 0) & (layer0 < _CODEBOOK)
    model.embed_input_ids = lambda ids: ids.float().reshape(-1, 1, 1).expand(-1, 1, _EAGER_DIM)
    model.mtp_calls = []

    def mtp(input_ids, input_embeds, last_hidden, text_step, **kwargs):
        model.mtp_calls.append((input_ids.clone(), last_hidden.clone(), text_step.clone()))
        codes = torch.stack([input_ids, input_ids + 1, input_ids + 2], dim=1)
        # Frame embedding sum plus the text step, like the Qwen3 talkers.
        return input_embeds + 100 + text_step, codes

    model.mtp = mtp
    return state


def _eager_outputs(num_tokens):
    return {
        "codes": {"audio": torch.zeros((num_tokens, 3), dtype=torch.long)},
        "meta": {"codec_frame_valid": torch.zeros((num_tokens,), dtype=torch.int8)},
    }


def test_eager_mtp_publishes_frames_in_the_step_that_sampled_cb0():
    state = _make_mtp_state()
    _fill_buffers(state, "prefill", "decode", "ended")
    # prefill: final 3-token chunk; decode: input CB0 42; ended: input CB0 was
    # EOS (async scheduling ran the finished request once more).
    batch = _EagerBatch([3, 1, 1])
    input_ids = torch.tensor([0, 0, 0, 42, _EOS])
    state._mtp_state.recorded = (
        batch,
        FrameBatch([(0, 0, "prefill", True), (1, 1, "decode", False), (2, 2, "ended", False)]),
        input_ids,
    )
    text_hidden = torch.arange(5 * _EAGER_DIM, dtype=torch.float32).reshape(5, _EAGER_DIM)
    sampled = torch.tensor([[7], [_EOS], [9]])
    outputs = _eager_outputs(5)
    original_hidden = text_hidden.clone()

    state.post_sample(batch, text_hidden, sampled, outputs)

    torch.testing.assert_close(text_hidden, original_hidden)
    ids, hidden, text_step = state.model.mtp_calls[0]
    assert ids.tolist() == [7, _EOS, 9]
    # The frame's hidden is this step's last-token hidden of each span.
    assert torch.equal(hidden, text_hidden[[2, 3, 4]])
    # The next step adds its own text step.
    assert not text_step.any()
    codes = outputs["codes"]["audio"]
    assert codes[2].tolist() == [7, 8, 9]
    assert codes[3].tolist() == [_EOS, _EOS + 1, _EOS + 2]
    assert codes[4].tolist() == [9, 10, 11]
    assert not codes[:2].any()
    # Valid only when both the sampled and (for decode) the input CB0 are codec ids.
    assert outputs["meta"]["codec_frame_valid"].tolist() == [0, 0, 1, 0, 0]
    assert state._mtp_state.ready == {0: "prefill", 1: "decode", 2: "ended"}
    assert torch.equal(state._mtp_state.embeds[0], torch.full((_EAGER_DIM,), 107.0))
    assert state._mtp_state.recorded is None


def test_eager_mtp_ignores_rows_recorded_for_another_batch():
    state = _make_mtp_state()
    _fill_buffers(state, "r0")
    state._mtp_state.recorded = (_EagerBatch([1]), FrameBatch([(0, 0, "r0", False)]), torch.tensor([5]))
    outputs = _eager_outputs(1)
    state.post_sample(_EagerBatch([1]), torch.zeros(1, _EAGER_DIM), torch.tensor([[3]]), outputs)
    assert state.model.mtp_calls == []
    assert not outputs["codes"]["audio"].any()


def test_eager_mtp_requires_retained_frame_outputs():
    state = _make_mtp_state()
    _fill_buffers(state, "r0")
    batch = _EagerBatch([1])
    state._mtp_state.recorded = (batch, FrameBatch([(0, 0, "r0", False)]), torch.tensor([5]))
    with pytest.raises(RuntimeError, match="requires retained codes.audio"):
        state.post_sample(batch, torch.zeros(1, _EAGER_DIM), torch.tensor([[3]]), {"codes": {"audio": None}})


def test_eager_decode_input_is_previous_frame_plus_text_step():
    state = _make_mtp_state()
    _fill_buffers(state, "r0", "r1")
    state._mtp_state.ready = {0: "r0", 1: "r1"}
    state._mtp_state.embeds[0] = 5.0
    state._mtp_state.embeds[1] = 7.0
    embeds = torch.zeros((2, _EAGER_DIM))
    batches = [(0, 0), (1, 1)]
    text_steps = torch.stack([torch.full((_EAGER_DIM,), 2.0), torch.ones(_EAGER_DIM)])
    state._mtp_state.apply_frame_embeddings(batches, embeds, _EagerBatch([1, 1]), text_steps)
    assert embeds.tolist() == [[7.0] * _EAGER_DIM, [8.0] * _EAGER_DIM]
    # The deferred MTP must not run for an eager decode row.
    assert state.model.mtp_calls == []


def test_eager_decode_without_a_frame_fails_loudly():
    state = _make_mtp_state()
    _fill_buffers(state, "r0")
    batches = [(0, 0)]
    with pytest.raises(RuntimeError, match="frame missing"):
        state._mtp_state.apply_frame_embeddings(
            batches, torch.zeros((1, _EAGER_DIM)), _EagerBatch([1]), torch.zeros(1, _EAGER_DIM)
        )


@pytest.mark.parametrize("has_stream_decoder", [True, False])
def test_run_preprocess_records_rows_that_keep_a_sample(has_stream_decoder):
    state = _make_mtp_state()
    if not has_stream_decoder:
        # Eager-MTP talkers without an in-Talker codec (Qwen3-Omni) do not
        # define the attribute at all.
        del state.model.stream_decoder
    _fill_buffers(state, "chunk", "final", "decode")
    state._mtp_state.ready = {2: "decode"}
    state._mtp_state.embeds[2] = 4.0
    text_step = torch.ones(_EAGER_DIM)

    def preprocess(input_ids, input_embeds, **info):
        updates = {"mtp_inputs": (torch.zeros(_EAGER_DIM), text_step)} if input_ids.shape[0] == 1 else {}
        return input_ids, input_embeds, updates

    state.model.preprocess = preprocess
    batch = _EagerBatch([3, 3, 1])
    model_inputs = {"input_ids": torch.zeros(7, dtype=torch.long), "inputs_embeds": torch.zeros((7, _EAGER_DIM))}
    req_states = SimpleNamespace(
        prompt_len=np.array([10, 10, 10], dtype=np.int32),
        num_computed_tokens=np.array([0, 7, 10], dtype=np.int32),
    )
    state.run_preprocess(batch, model_inputs, req_states)

    recorded_batch, frame_batch, _ids = state._mtp_state.recorded
    entries = frame_batch.entries
    assert recorded_batch is batch
    # A non-final prefill chunk samples nothing that is kept.
    assert entries == [(1, 1, "final", True), (2, 2, "decode", False)]
    assert model_inputs["inputs_embeds"][6].tolist() == [5.0] * _EAGER_DIM
    assert state.model.mtp_calls == []


def test_prefill_rows_written_for_the_batch_skip_per_request_preprocess():
    state = _make_mtp_state()
    _fill_buffers(state, "batched", "serial", "decode")
    state._mtp_state.ready = {2: "decode"}
    state._mtp_state.embeds[2] = 4.0
    seen_rows: list[list[tuple[int, int, str]]] = []
    preprocessed: list[str] = []

    def preprocess_prefill_rows_mrv2(*, entries, input_ids, input_embeds):
        seen_rows.append([(start, n_tok, info["req_id"]) for start, n_tok, info in entries])
        start, n_tok, _info = entries[0]
        input_embeds[start : start + n_tok] = 9.0
        return [{"meta": {"talker_prefill_offset": n_tok}}, None]

    state.model.preprocess_prefill_rows_mrv2 = preprocess_prefill_rows_mrv2

    def preprocess(input_ids, input_embeds, **info):
        preprocessed.append(info["req_id"])
        updates = {"mtp_inputs": (torch.zeros(_EAGER_DIM), torch.ones(_EAGER_DIM))} if input_ids.shape[0] == 1 else {}
        return input_ids, input_embeds, updates

    state.model.preprocess = preprocess
    batch = _EagerBatch([3, 2, 1])
    model_inputs = {"input_ids": torch.zeros(6, dtype=torch.long), "inputs_embeds": torch.zeros((6, _EAGER_DIM))}
    req_states = SimpleNamespace(
        prompt_len=np.array([3, 4, 10], dtype=np.int32),
        num_computed_tokens=np.array([0, 0, 10], dtype=np.int32),
    )
    state.run_preprocess(batch, model_inputs, req_states)

    assert seen_rows == [[(0, 3, "batched"), (3, 2, "serial")]]
    assert preprocessed == ["serial", "decode"]
    assert model_inputs["inputs_embeds"][:3].eq(9.0).all()
    assert state.intermediate_buffer.buffers[0]["meta"] == {"talker_prefill_offset": 3}
    # The batch-written row still keeps its sample (its prompt ends in this chunk).
    assert [entry[2] for entry in state._mtp_state.recorded[1].entries] == ["batched", "decode"]


@pytest.mark.parametrize("first_was_valid", [False, True])
def test_first_audio_requirement_survives_eos_only_if_audio_was_queued(first_was_valid):
    state = _make_mtp_state()
    _fill_buffers(state, "r0")
    state._mtp_state.first_audio_requests.add("r0")
    state._mtp_state._first_audio_valid = torch.tensor([first_was_valid, False, False, False])
    batch = _EagerBatch([1])
    state._mtp_state.recorded = (batch, FrameBatch([(0, 0, "r0", False)]), torch.tensor([5]))
    outputs = _eager_outputs(1)
    outputs["meta"]["first_audio"] = torch.zeros(1, dtype=torch.bool)
    state.post_sample(batch, torch.zeros(1, _EAGER_DIM), torch.tensor([[_EOS]]), outputs)
    assert outputs["meta"]["first_audio"].tolist() == [first_was_valid]


@pytest.mark.parametrize("accepted", [[], ["r1"], ["r0", "r1"]])
def test_first_audio_marker_requires_accepted_delivery(monkeypatch, accepted):
    from contextlib import nullcontext

    state = _make_mtp_state()
    _fill_buffers(state, "r0", "r1")
    state.model.first_frame_decoder = SimpleNamespace(
        sample_rate=24000, decode=lambda codes: torch.ones(codes.shape[0], 2)
    )
    state._mtp_state.first_audio_stream = SimpleNamespace(wait_stream=lambda stream: None)
    state._mtp_state.first_audio_sender = SimpleNamespace(submit=lambda ids, pcm, sr, valid: accepted)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    batch = _EagerBatch([1, 1])
    state._mtp_state.recorded = (
        batch,
        FrameBatch([(0, 0, "r0", True), (1, 1, "r1", True)]),
        torch.zeros(2, dtype=torch.long),
    )
    outputs = _eager_outputs(2)
    outputs["meta"]["first_audio"] = torch.zeros(2, dtype=torch.bool)

    state.post_sample(batch, torch.zeros(2, _EAGER_DIM), torch.tensor([[7], [8]]), outputs)

    assert state._mtp_state.first_audio_requests == set(accepted)
    # A missing route must leave the normal codec path responsible for frame 0;
    # otherwise it skips the frame and the orchestrator waits forever for it.
    assert outputs["meta"]["first_audio"].tolist() == ["r0" in accepted, "r1" in accepted]


class _FakeEvent:
    def record(self, stream=None) -> None:
        self.stream = stream


def test_publish_sampled_embeddings_for_rows_whose_sample_is_kept(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    state = _make_state()
    state.model.publishes_sampled_embeddings = True
    state.model.embed_input_ids = lambda ids: ids.to(torch.float32).reshape(-1, 1).repeat(1, 3)
    # Rows: final prefill chunk, decode, non-final prefill chunk, one-chunk prefill.
    batch = SimpleNamespace(
        num_reqs=4,
        num_scheduled_tokens=[4, 1, 2, 6],
        num_computed_prefill_tokens_np=np.array([2, 9, 0, 0], dtype=np.int32),
        prefill_len_np=np.array([6, 9, 5, 6], dtype=np.int32),
        is_prefilling_np=np.array([True, False, True, True]),
    )
    extra, _done = state.publish_sampled_embeddings(batch, torch.tensor([[11], [12], [13], [14]]))
    sampled = extra["embed"]["sampled"]
    # A non-final prefill chunk's sample is discarded, so it publishes nothing.
    assert [tuple(row.shape) for row in sampled] == [(1, 3), (1, 3), (0,), (1, 3)]
    assert [row.tolist() for row in sampled if row.numel()] == [[[v] * 3] for v in (11.0, 12.0, 14.0)]

    batch.is_prefilling_np[:] = True
    batch.num_computed_prefill_tokens_np[:] = 0
    batch.num_scheduled_tokens = [1] * 4
    extra, _done = state.publish_sampled_embeddings(batch, torch.tensor([[11], [12], [13], [14]]))
    assert all(row.numel() == 0 for row in extra["embed"]["sampled"])

    # Speculative steps sample several tokens per row: not published.
    assert state.publish_sampled_embeddings(batch, torch.tensor([[11, 1], [12, 1], [13, 1], [14, 1]])) is None


def test_publish_sampled_embeddings_is_opt_in() -> None:
    state = _make_state()
    state.model.publishes_sampled_embeddings = False
    assert state.publish_sampled_embeddings(SimpleNamespace(num_reqs=1), torch.tensor([[1]])) is None


def test_split_settled_rows_matches_per_row_check():
    batch = _DummyInputBatch(np.array([3, 0, 2, 1, 4]))
    batch.req_ids = ["c", "a", "x", "b", "e"]
    batch.num_scheduled_tokens = [1, 1, 1, 4, 1]
    batch.query_start_loc_np = [0, 1, 2, 3, 7]
    req_indices = batch.idx_mapping_np.tolist()
    # Slot 2 still names a finished request, slot 1 schedules a prefill
    # chunk, slot 4 is not settled and slot 7 is not scheduled.
    settled = np.array(["a", "b", "old", "c", None, None, None, "z"], dtype=object)
    split = OmniModelState._split_settled_rows(batch, req_indices, settled)
    assert split is not None
    settled_rows, remaining = split
    assert settled_rows == [(0, 3, 0, "c"), (1, 0, 1, "a")]
    assert remaining == [(2, 2), (3, 1), (4, 4)]
    assert (
        OmniModelState._split_settled_rows(batch, req_indices, np.array([None, None, "old", None, None], dtype=object))
        is None
    )


def test_settled_row_split_preserves_mixed_batch_and_pending_replay(monkeypatch):
    """Exercise run_preprocess, including the scalar fallback during replay."""

    def prepare():
        state = _make_mtp_state()
        _fill_buffers(state, "prefill", "decode")
        state._mtp_state.mark_settled(1, "decode")
        state._mtp_state.ready = {1: "decode"}
        state._mtp_state.embeds[1].fill_(7)
        state.model.eager_settled_text_step = lambda: torch.ones(_EAGER_DIM)
        state.model.eager_decode_settled = lambda info: False
        state.model.preprocess = lambda input_ids, input_embeds, **info: (input_ids, input_embeds, {})
        batch = _EagerBatch([3, 1])
        batch.req_ids = ["prefill", "decode"]
        inputs = {"input_ids": torch.zeros(4, dtype=torch.long), "inputs_embeds": torch.zeros(4, _EAGER_DIM)}
        return state, batch, inputs

    vector, batch, inputs = prepare()
    vector.run_preprocess(batch, inputs)
    reference, ref_batch, ref_inputs = prepare()
    monkeypatch.setattr(reference, "_split_settled_rows", lambda *args: None)
    reference.run_preprocess(ref_batch, ref_inputs)
    torch.testing.assert_close(inputs["inputs_embeds"], ref_inputs["inputs_embeds"], rtol=0, atol=0)
    assert vector._mtp_state.recorded[1].entries == reference._mtp_state.recorded[1].entries
    assert inputs["inputs_embeds"][3].tolist() == [8.0] * _EAGER_DIM

    replay, replay_batch, replay_inputs = prepare()
    replay._mtp_state._restore_audio["prefill"] = [torch.ones(1)]
    assert replay._mtp_state.has_pending_replay()
    monkeypatch.setattr(replay, "_split_settled_rows", MagicMock(side_effect=AssertionError("replay must skip split")))
    replay.run_preprocess(replay_batch, replay_inputs)
    torch.testing.assert_close(replay_inputs["inputs_embeds"], ref_inputs["inputs_embeds"], rtol=0, atol=0)


@pytest.mark.parametrize("prefilling", [False, True])
def test_identity_preprocess_skips_only_decode_rows(prefilling):
    state = _make_state(has_preprocess=True)
    state._decode_preprocess_is_identity = True
    state._decode_preprocess = None
    state.intermediate_buffer.buffers[0] = {"req_id": "r1"}
    seen = []

    def preprocess(input_ids, input_embeds, **info):
        seen.append(info["req_id"])
        return input_ids, input_embeds, {}

    state.model.preprocess = preprocess
    batch = _DummyInputBatch([0])
    batch.is_prefilling_np = np.array([prefilling])
    embeds = torch.ones(1, 4)
    state.run_preprocess(batch, {"input_ids": torch.tensor([1]), "inputs_embeds": embeds})
    assert seen == (["r1"] if prefilling else [])
    assert torch.equal(embeds, torch.ones(1, 4))


def test_mm_embeddings_exclude_zero_length_graph_padding_rows():
    state = object.__new__(OmniModelState)
    state.supports_mm_inputs = True
    state.mm_pruner = None
    state.prompt_embeds_state = None
    state.encoder_runner = MagicMock(spec=EncoderRunner, inputs_embeds=torch.zeros(8, 2))
    state.execute_mm_encoder = lambda _: None
    state.gather_mm_embeddings = lambda _: ([torch.ones(1, 2)], torch.ones(5, dtype=torch.bool))

    def embed(input_ids, *, query_start_loc, multimodal_embeddings, is_multimodal):
        assert query_start_loc == [0, 2, 5]
        return input_ids[:, None].expand(-1, 2).float()

    state.model = MagicMock(supports_embed_input_ids_query_start_loc=True, embed_input_ids=embed)
    buffers = InputBuffers(4, 8, torch.device("cpu"))
    batch = replace(
        InputBatch.make_dummy(2, 5, buffers),
        num_reqs=2,
        num_reqs_after_padding=4,
        num_tokens=5,
        num_tokens_after_padding=8,
        input_ids=torch.arange(8),
        query_start_loc_np=np.array([0, 2, 5, 5, 5], dtype=np.int32),
    )
    result = state.prepare_inputs_embeds({}, batch, MagicMock(spec=RequestState))
    torch.testing.assert_close(result[:5], torch.arange(5).float()[:, None].expand(-1, 2))
    assert result.shape == (8, 2)


def _prefill_mask_batch(counts, prefill_lens, computed, input_ids=None):
    num_reqs = len(counts)
    num_tokens = sum(counts)
    buffers = InputBuffers(num_reqs + 1, num_tokens + 2, torch.device("cpu"))
    starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    prefill_lens = np.array(prefill_lens, dtype=np.int32)
    computed = np.array(computed, dtype=np.int32)
    return replace(
        InputBatch.make_dummy(num_reqs, num_tokens, buffers),
        num_reqs_after_padding=num_reqs + 1,
        num_tokens_after_padding=num_tokens + 2,
        num_scheduled_tokens=np.array(counts, dtype=np.int32),
        query_start_loc_np=np.append(starts, num_tokens),
        # Request slots are not necessarily in flattened batch order.
        idx_mapping_np=np.arange(num_reqs)[::-1],
        prefill_len_np=prefill_lens,
        num_computed_prefill_tokens_np=computed,
        is_prefilling_np=computed < prefill_lens,
        has_prefill=bool(np.any(computed < prefill_lens)),
        input_ids=torch.tensor((input_ids or [3] * num_tokens) + [0, 0]),
    )


@pytest.mark.parametrize(
    ("counts", "prefill_lens", "computed", "expected"),
    [
        pytest.param([1, 1], [4, 8], [4, 8], False, id="decode-only"),
        pytest.param([1], [1], [0], True, id="single-token-prefill"),
        pytest.param([1], [8], [7], True, id="last-prompt-token"),
        pytest.param([2], [8], [3], True, id="chunked-prefill"),
        pytest.param([1, 2], [4, 2], [4, 0], [False, True, True], id="decode-before-prefill"),
        pytest.param([2, 1], [2, 4], [0, 4], [True, True, False], id="prefill-before-decode"),
        pytest.param([1, 2, 1], [4, 2, 1], [4, 0, 0], [False, True, True, True], id="different-prefill-lengths"),
        pytest.param([3], [8], [7], [True, False, False], id="prefill-remainder-and-decode"),
        pytest.param([2, 1], [10, 4], [8, 4], [True, True, False], id="resumed-prefill"),
    ],
)
def test_v2_prefill_mask_matches_flattened_tokens(counts, prefill_lens, computed, expected):
    batch = _prefill_mask_batch(counts, prefill_lens, computed)
    mask = OmniModelState._build_prefill_token_mask(batch)
    if isinstance(expected, bool):
        assert mask is expected
    else:
        assert mask.dtype == torch.bool
        assert mask.shape == (batch.num_tokens,)
        assert mask.tolist() == expected


def test_v2_prefill_mask_uses_actual_queries_not_scheduled_upper_bounds():
    batch = _prefill_mask_batch([1, 2], [4, 1], [4, 0])
    batch = replace(batch, num_scheduled_tokens=np.array([1, 4], dtype=np.int32))
    mask = OmniModelState._build_prefill_token_mask(batch)
    assert mask.shape == (batch.num_tokens,)
    assert mask.tolist() == [False, True, False]


def _cosy_embedding_state(mocker):
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model

    model = SimpleNamespace(
        model_stage="cosyvoice3_talker",
        supports_prefill_token_mask=CosyVoice3Model.supports_prefill_token_mask,
        supports_embed_input_ids_query_start_loc=True,
        model=SimpleNamespace(
            speech_embedding=torch.nn.Embedding(10, 4),
            llm=SimpleNamespace(model=SimpleNamespace(embed_tokens=torch.nn.Embedding(32, 4))),
            sos=8,
            task_id=9,
        ),
    )
    model.embed_input_ids = MethodType(CosyVoice3Model.embed_input_ids, model)

    def init_default(state, *_args, **_kwargs):
        state.scheduler_config = SimpleNamespace(max_num_seqs=4)
        state.model = model
        state.device = torch.device("cpu")

    mocker.patch.object(DefaultModelState, "__init__", init_default)
    state = OmniModelState(SimpleNamespace(), model, None, torch.device("cpu"))
    state.supports_mm_inputs = True
    state.mm_pruner = None
    state.prompt_embeds_state = None
    state.encoder_runner = SimpleNamespace(inputs_embeds=torch.zeros(32, 4))
    state.execute_mm_encoder = mocker.Mock()
    state.gather_mm_embeddings = mocker.Mock(return_value=([], None))
    return state


def test_v2_decode_bypasses_cosyvoice3_validation(mocker):
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as cosyvoice3

    state = _cosy_embedding_state(mocker)
    # The capability is cached at initialization, not re-read on every step.
    del state.model.supports_prefill_token_mask
    guard = mocker.patch.object(
        cosyvoice3, "_validate_speech_token_ids", side_effect=AssertionError("decode validated")
    )
    batch = _prefill_mask_batch([1, 1], [4, 8], [4, 8], [3, 5])
    result = state.prepare_inputs_embeds({}, batch, SimpleNamespace())
    guard.assert_not_called()
    torch.testing.assert_close(result[:2], state.model.model.speech_embedding.weight[batch.input_ids[:2]])


def test_v2_single_token_prefill_rejects_out_of_range_id(mocker):
    state = _cosy_embedding_state(mocker)
    batch = _prefill_mask_batch([1], [1], [0], [10])
    with pytest.raises(ValueError, match="out of range"):
        state.prepare_inputs_embeds({}, batch, SimpleNamespace())


def test_v2_mixed_batch_validates_only_prefill_tokens(mocker):
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as cosyvoice3

    state = _cosy_embedding_state(mocker)
    guard = mocker.spy(cosyvoice3, "_validate_speech_token_ids")
    batch = _prefill_mask_batch([1, 3], [4, 8], [4, 7], [3, 4, 5, 6])
    state.prepare_inputs_embeds({}, batch, SimpleNamespace())
    guard.assert_called_once()
    assert guard.call_args.args[0].tolist() == [4]


@pytest.mark.parametrize("decode_first", [True, False], ids=["decode-before-audio", "decode-after-audio"])
def test_v2_mixed_audio_prefill_excludes_decode_from_validation(mocker, decode_first):
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as cosyvoice3

    state = _cosy_embedding_state(mocker)
    counts, prefill_lens, computed = ([1, 4], [8, 4], [8, 0]) if decode_first else ([4, 1], [4, 8], [0, 8])
    ids = [3, 0, 1, 2, 20] if decode_first else [0, 1, 2, 20, 3]
    mm_mask = [False, True, True, True, False] if decode_first else [True, True, True, False, False]
    state.gather_mm_embeddings.return_value = ([torch.ones(1, 4)], torch.tensor(mm_mask))
    batch = _prefill_mask_batch(counts, prefill_lens, computed, ids)
    guard = mocker.spy(cosyvoice3, "_validate_speech_token_ids")
    result = state.prepare_inputs_embeds({}, batch, SimpleNamespace())
    assert result.shape == (batch.num_tokens_after_padding, 4)
    # The multimodal branch may call the helper on an empty slice, which
    # returns before any device reduction. It must never validate decode ids.
    assert all(call.args[0].numel() == 0 for call in guard.call_args_list)


def test_v2_models_without_prefill_mask_keep_embedding_contract(mocker):
    state = _cosy_embedding_state(mocker)
    state._supports_prefill_token_mask = False
    state.model.embed_input_ids = mocker.Mock(return_value=torch.zeros(1, 4))
    builder = mocker.patch.object(state, "_build_prefill_token_mask", side_effect=AssertionError("unexpected mask"))
    state.prepare_inputs_embeds({}, _prefill_mask_batch([1], [4], [4]), SimpleNamespace())
    builder.assert_not_called()
    assert "prefill_token_mask" not in state.model.embed_input_ids.call_args.kwargs


@pytest.mark.cpu
def test_post_sample_rejects_multi_token_sampling():
    state = _make_mtp_state()
    _fill_buffers(state, "decode")
    batch = _EagerBatch([1])
    state._mtp_state.recorded = (batch, FrameBatch([(0, 0, "decode", False)]), torch.tensor([7]))
    with pytest.raises(ValueError, match="one sampled token"):
        state.post_sample(batch, torch.zeros(1, _EAGER_DIM), torch.tensor([[8, 9]]), _eager_outputs(1))
    assert not state.model.mtp_calls


@pytest.mark.cpu
@pytest.mark.parametrize("hook_name", ["mtp", "talker_mtp"])
def test_declared_mtp_defaults_to_current_step_without_first_audio(monkeypatch, hook_name):
    def default_init(self, config, model, encoder_cache, device):
        self.vllm_config = config
        self.scheduler_config = config.scheduler_config
        self.model, self.device, self.dtype = model, device, torch.float32
        self.max_num_tokens, self.supports_mm_inputs = 8, False

    class Model(torch.nn.Module):
        has_preprocess = True
        has_talker_mtp = True
        mtp_disable_graph = True
        mtp_output_key = ("codes", "audio")

        def embed_input_ids(self, ids):
            return torch.zeros(ids.numel(), 4)

    model = Model()

    def hook(ids, embeds, hidden, step):
        return embeds, ids[:, None]

    setattr(model, hook_name, hook)
    monkeypatch.setattr(DefaultModelState, "__init__", default_init)
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2), compilation_config=SimpleNamespace())
    state = OmniModelState(config, model, None, torch.device("cpu"))
    assert state._has_mtp and model.eager_frames_active
    assert state._mtp_state._mtp_runner is hook
    assert state._mtp_state._mtp_hidden.shape == (2, 4)
    assert state._mtp_state.embeds.shape == (2, 4)
    assert state._mtp_state.first_audio_sender is None


@pytest.mark.cpu
def test_generic_preprocess_does_not_dispatch_deferred_mtp():
    state = _make_state(has_preprocess=True)
    _fill_buffers(state, "decode")
    state.model.preprocess = lambda ids, embeds, **info: (
        ids,
        embeds,
        {"mtp_inputs": (torch.zeros(1, 2), torch.zeros(1, 2))},
    )
    with pytest.raises(RuntimeError, match="Deferred MTP is not supported"):
        state.run_preprocess(
            _DummyInputBatch([0], num_computed_tokens_cpu=[1]),
            {"input_ids": torch.tensor([7]), "inputs_embeds": torch.zeros(1, 2)},
            SimpleNamespace(prompt_len=np.array([1])),
        )
    state.model.mtp.assert_not_called()


@pytest.mark.parametrize("payload", [None, {}])
def test_post_sample_skips_empty_mm_without_running_mtp(payload):
    state = _make_mtp_state()
    batch = _EagerBatch([1])
    recorded = (batch, FrameBatch([(0, 0, "decode", False)]), torch.tensor([7]))
    state._mtp_state.recorded = recorded
    state._mtp_state.post_sample = MagicMock()
    state.post_sample(batch, torch.zeros(1, _EAGER_DIM), torch.tensor([[8]]), payload)
    state._mtp_state.post_sample.assert_not_called()
    assert not state.model.mtp_calls
    assert state._mtp_state.recorded is recorded


@pytest.mark.parametrize("async_chunk", [False, True])
@pytest.mark.parametrize("spans", [[1, 1], [3, 1]], ids=["decode", "mixed-prefill-decode"])
def test_qwen3_omni_current_step_output_supports_sync_and_async(async_chunk, spans):
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
    from vllm_omni.worker_v2.omni_ar_model_runner import _slice_pooler_value

    talker = Qwen3OmniMoeForConditionalGeneration.__new__(Qwen3OmniMoeForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.model_stage = "talker"
    talker.talker = SimpleNamespace(num_code_groups=3)
    talker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=async_chunk))
    talker.eager_frames_active = True
    state = _make_mtp_state()
    _fill_buffers(state, "first", "second")
    infos = state.intermediate_buffer.buffers[:2]
    for info, count in zip(infos, [5, 4], strict=True):
        info["codes"] = {"audio": torch.full((count, 3), 99, dtype=torch.long)}
    batch = _EagerBatch(spans)
    total = sum(spans)
    bounds = list(zip(batch.query_start_loc_np, batch.query_start_loc_np[1:] + [total], strict=True))
    hidden = torch.zeros(total, _EAGER_DIM)
    mm = talker.make_omni_output(hidden, model_intermediate_buffer=infos, request_token_spans=bounds).multimodal_outputs
    state._mtp_state.recorded = (
        batch,
        FrameBatch([(i, i, info["req_id"], n > 1) for i, (info, n) in enumerate(zip(infos, spans, strict=True))]),
        torch.full((total,), 7),
    )
    state.post_sample(batch, hidden, torch.tensor([[11], [23]]), mm)
    for i, ((start, end), token) in enumerate(zip(bounds, [11, 23], strict=True)):
        row = _slice_pooler_value(mm, req_index=i, start=start, end=end, total_tokens=total)
        assert row["codes"]["audio"].shape == (spans[i], 3)
        assert row["codes"]["audio"][-1].tolist() == [token, token + 1, token + 2]
        assert not row["codes"]["audio"][:-1].any()
        assert row["meta"]["codec_frame_valid"].tolist() == [0] * (spans[i] - 1) + [1]
    assert all((info["codes"]["audio"] == 99).all() for info in infos)


@pytest.mark.parametrize("async_chunk", [False, True])
@pytest.mark.parametrize("next_spans", [[1, 1], [2, 1]], ids=["decode", "mixed-prefill-decode"])
def test_qwen3_tts_current_step_output_uses_current_spans_and_isolates_requests(async_chunk, next_spans):
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
        Qwen3TTSTalkerForConditionalGeneration,
    )
    from vllm_omni.worker_v2.omni_ar_model_runner import _slice_pooler_value

    talker = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=async_chunk))
    talker.talker_config = SimpleNamespace(num_code_groups=3)
    talker.eager_frames_active = True
    talker.stream_decoder = None
    talker._ref_codes_pending = True
    state = _make_mtp_state()
    _fill_buffers(state, "first", "second")
    infos = state.intermediate_buffer.buffers[:2]
    refs = [torch.full((n, 3), value, dtype=torch.long) for n, value in [(2, 31), (5, 47)]]
    for info, count, ref, streaming in zip(infos, [3, 4], refs, [True, False], strict=True):
        info["codes"] = {"audio": torch.zeros(count, 3, dtype=torch.long), "ref": ref}
        info["meta"] = {"codec_frame_valid": False, "ref_code_len": ref.shape[0], "codec_streaming": streaming}

    def run_frame(spans, sampled):
        batch = _EagerBatch(spans)
        total = sum(spans)
        bounds = list(zip(batch.query_start_loc_np, batch.query_start_loc_np[1:] + [total], strict=True))
        hidden = torch.zeros(total, _EAGER_DIM)
        mm = talker.make_omni_output(
            hidden, model_intermediate_buffer=infos, request_token_spans=bounds
        ).multimodal_outputs
        assert mm["codes"]["audio"].shape == (total, 3)
        assert mm["meta"]["codec_frame_valid"].shape == (total,)
        assert not mm["codes"]["audio"].any()
        assert not mm["meta"]["codec_frame_valid"].any()
        state._mtp_state.recorded = (
            batch,
            FrameBatch(
                [(i, i, info["req_id"], count > 1) for i, (info, count) in enumerate(zip(infos, spans, strict=True))]
            ),
            torch.full((total,), 7),
        )
        state.post_sample(batch, hidden, sampled, mm)
        return mm, bounds

    prefill, _ = run_frame([3, 4], torch.tensor([[7], [19]]))
    prefill_snapshot = prefill["codes"]["audio"].clone()
    assert prefill["meta"]["codec_frame_valid"].tolist() == [0, 0, 1, 0, 0, 0, 1]
    output, bounds = run_frame(next_spans, torch.tensor([[11], [23]]))
    for i, ((start, end), token) in enumerate(zip(bounds, [11, 23], strict=True)):
        row = _slice_pooler_value(output, req_index=i, start=start, end=end, total_tokens=sum(next_spans))
        assert row["codes"]["audio"].shape == (next_spans[i], 3)
        assert row["codes"]["audio"][-1].tolist() == [token, token + 1, token + 2]
        assert row["meta"]["codec_frame_valid"].tolist() == [0] * (next_spans[i] - 1) + [1]
        if not async_chunk:
            assert row["meta"]["ref_code_len"].tolist() == [refs[i].shape[0]] * next_spans[i]
            assert row["meta"]["codec_streaming"].tolist() == [int(i == 0)] * next_spans[i]
            torch.testing.assert_close(row["codes"]["ref"], refs[i])
    assert [info["codes"]["audio"].shape[0] for info in infos] == [3, 4]
    torch.testing.assert_close(prefill["codes"]["audio"], prefill_snapshot)
