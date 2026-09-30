# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OmniModelState: mixed-batch reorder, state isolation/slot reuse, MTP
graph/eager paths with per-request seed independence, async snapshot ownership."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_omni.model_executor.models.output_templates import OmniOutput, OwnedBatchTensor
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState, _make_safe_get_rope

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    is_prefilling_np: np.ndarray

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
    model.preprocess_decode_batch_mrv2 = None
    model.preprocess_decode_batch = None
    model.postprocess_batch_mrv2 = None
    state.model = model
    state.scheduler_config = SimpleNamespace(max_num_seqs=max_num_reqs)
    state.vllm_config = SimpleNamespace(model_config=SimpleNamespace(subtalker_sampling_params={}))
    state.has_preprocess = has_preprocess
    state.has_postprocess = has_postprocess
    state.have_multimodal_outputs = have_multimodal_outputs

    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

    state.intermediate_buffer = OmniIntermediateBuffer(max_num_reqs)
    state._static_inputs_embeds = None
    state._mtp_generators = {}
    state._mtp_runner = None
    for name in ("_mtp_input_ids", "_mtp_input_embeds", "_mtp_hidden", "_mtp_text_step", "_mtp_offsets"):
        setattr(state, name, None)
    return state


def _add(state, req_id="r1", idx=0, **req_kwargs):
    with patch.object(DefaultModelState, "add_request", return_value=None):
        state.add_request(idx, SimpleNamespace(req_id=req_id, mm_features=[], **req_kwargs))


def _fill_buffers(state, *req_ids):
    for idx, req_id in enumerate(req_ids):
        state.intermediate_buffer.buffers[idx] = {"req_id": req_id}


def _seeded(seed):
    return SimpleNamespace(extra_args={"test_seed": seed}, seed=None)


def _init_static(state, bsz, dim=3):
    state._mtp_input_ids = torch.zeros(bsz, dtype=torch.long)
    state._mtp_input_embeds = torch.zeros((bsz, dim))
    state._mtp_hidden = torch.zeros((bsz, dim))
    state._mtp_text_step = torch.zeros((bsz, dim))
    state._mtp_offsets = torch.zeros(bsz, dtype=torch.long)


def _mtp_batches(dim=3):
    return [(0, 0, (torch.ones(dim), torch.ones(dim) * 2)), (1, 1, (torch.ones(dim) * 3, torch.ones(dim) * 4))]


@pytest.fixture(autouse=True)
def _fwd_ctx():
    with patch("vllm.forward_context.set_forward_context", return_value=nullcontext()):
        yield


@pytest.mark.parametrize("req_id,expect_validity", [("_warmup_0_", True), ("r1", False)])
def test_add_request_declared_validity_only_for_warmup(req_id, expect_validity):
    # Real requests must not be masked with a fabricated validity key.
    state = _make_state()
    state.model.mtp_validity_key = ("meta", "codec_frame_valid")
    _add(state, req_id, 0)
    buf = state.intermediate_buffer.buffers[0]
    assert ("meta" in buf) is expect_validity
    if expect_validity:
        assert buf["meta"]["codec_frame_valid"].item() is False


@pytest.mark.parametrize("mode", ["index", "req_id", "unknown"])
def test_remove_request_state_isolation(mode):
    state = _make_state()
    _add(state, "r1", 0)
    state._mtp_generators["r1"] = torch.Generator(device="cpu")
    if mode == "unknown":
        state.remove_request("missing")
        assert state.intermediate_buffer.buffers[0]["req_id"] == "r1"
        assert "r1" in state._mtp_generators
        return
    state.remove_request(0 if mode == "index" else "r1")
    assert state.intermediate_buffer.buffers[0] == {}
    assert "r1" not in state._mtp_generators  # recycled slot inherits no seed stream


def test_output_spans_follow_reordered_mixed_batch():
    # batch=[2, 0]: prefill (3 tokens) reordered ahead of decode (1 token).
    state = _make_state(have_multimodal_outputs=True)
    batch = _DummyInputBatch([2, 0])
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
    seen_prepacked = []

    def capture(batches, *_a, prepacked_mtp_inputs=None):
        seen_prepacked.append(prepacked_mtp_inputs)

    state._run_batched_mtp = capture
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
    assert torch.equal(seen_prepacked[0][0], torch.tensor([[31.0, 32.0], [41.0, 42.0]]))


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
    assert state._get_mtp_generator("r1", SimpleNamespace(extra_args={}, seed=42), cpu) is None
    # Same model-local seed reproduces identical uniforms regardless of batch makeup.
    state._mtp_sample_uniforms = torch.empty((2, 2, 4))
    assert torch.equal(
        state._prepare_mtp_sample_uniforms([torch.Generator().manual_seed(11)], 1).clone(),
        state._prepare_mtp_sample_uniforms([torch.Generator().manual_seed(11)], 1),
    )
    # Per-request generators are resolved once per step and ride sampling kwargs.
    state.model.mtp_sampling_params = {"do_sample": True, "temperature": 0.7}
    state.model.mtp_accepts_per_row_generators = True
    _init_static(state, 2)
    state.intermediate_buffer.buffers[0] = {"req_id": "r0", "sampling_params": _seeded(11)}
    state.intermediate_buffer.buffers[1] = {"req_id": "r1", "sampling_params": _seeded(22)}
    resolver = MagicMock(wraps=state._get_mtp_generator)
    state._get_mtp_generator = resolver

    def mtp(input_ids, input_embeds, last_hidden, text_step, **kwargs):
        assert (kwargs["do_sample"], kwargs["temperature"]) == (True, 0.7)
        assert all(isinstance(g, torch.Generator) for g in kwargs["generators"])
        return input_embeds + 10, torch.tensor([[1, 2, 3], [4, 5, 6]])

    state.model.mtp = mtp
    state._run_batched_mtp(
        _mtp_batches(), torch.tensor([101, 202]), torch.zeros(2, 3), _DummyInputBatch([0, 1]), {("codes", "audio")}
    )
    assert resolver.call_count == 2
    assert torch.equal(state.intermediate_buffer.buffers[0]["codes"]["audio"], torch.tensor([[1, 2, 3]]))
    assert torch.equal(state.intermediate_buffer.buffers[1]["codes"]["audio"], torch.tensor([[4, 5, 6]]))


def test_seeded_mtp_bypasses_outer_graph_runner():
    state = _make_state(max_num_reqs=2)
    raw_calls = []

    def mtp(*args, **kwargs):
        raw_calls.append(kwargs)
        return args[1], args[0].reshape(-1, 1)

    state.model.mtp = mtp
    state._mtp_runner = MagicMock(side_effect=AssertionError("seeded sampling must not replay the graph"))
    generators = [torch.Generator().manual_seed(11), torch.Generator().manual_seed(22)]
    state._call_mtp_runner(
        torch.tensor([101, 202]), torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3), generators=generators
    )
    assert raw_calls == [{"generators": generators}]
    state._mtp_runner.assert_not_called()


def test_run_batched_mtp_uses_dispatched_graph_descriptor():
    state = _make_state(max_num_reqs=4)

    class _FakeGraphRunner:
        def __call__(self, input_ids, input_embeds, last_hidden, text_step, **kwargs):
            assert input_ids.shape[0] == 4  # graph runs the padded descriptor size
            return input_embeds + 2, torch.arange(12, dtype=torch.long).reshape(4, 3)

    state._mtp_runner = _FakeGraphRunner()
    _init_static(state, 4)
    _fill_buffers(state, "r0", "r1")
    graph_desc = SimpleNamespace(cg_mode="FULL", num_tokens=4)
    set_ctx = MagicMock(return_value=nullcontext())
    with (
        patch(
            "vllm_omni.worker_v2.model_states.omni_model_state.current_omni_platform.get_graph_wrapper_cls",
            return_value=_FakeGraphRunner,
        ),
        patch("vllm.forward_context.set_forward_context", set_ctx),
    ):
        state._run_batched_mtp(
            _mtp_batches(),
            torch.tensor([101, 202]),
            torch.zeros(2, 3),
            _DummyInputBatch([0, 1]),
            {("codes", "audio")},
            lambda bsz: graph_desc,
        )
    _, kwargs = set_ctx.call_args
    assert kwargs["batch_descriptor"] is graph_desc
    assert kwargs["num_tokens"] == 4
    assert torch.equal(state.intermediate_buffer.buffers[0]["codes"]["audio"], torch.tensor([[0, 1, 2]]))
    assert torch.equal(state.intermediate_buffer.buffers[1]["codes"]["audio"], torch.tensor([[3, 4, 5]]))


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
    torch.testing.assert_close(positions, torch.arange(4).expand(3, -1))
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
        state._init_mtp_runner(model)


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


def _make_eager_state(max_num_reqs=4):
    state = _make_state(max_num_reqs=max_num_reqs, has_preprocess=True)
    _init_static(state, max_num_reqs, dim=_EAGER_DIM)
    from vllm_omni.worker_v2.model_states.eager_mtp import EagerMTPState

    state._eager_state = EagerMTPState(state)
    state._first_audio_requests = set()
    state._eager_mtp = True
    state._eager_fastpath = False
    state._eager_rows = None
    state._eager_ready = {}
    state._eager_settled = {}
    state._eager_embeds = torch.zeros((max_num_reqs, _EAGER_DIM))
    state._decode_preprocess = None
    state._mtp_sample_uniforms = None
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
    state = _make_eager_state()
    _fill_buffers(state, "prefill", "decode", "ended")
    # prefill: final 3-token chunk; decode: input CB0 42; ended: input CB0 was
    # EOS (async scheduling ran the finished request once more).
    batch = _EagerBatch([3, 1, 1])
    input_ids = torch.tensor([0, 0, 0, 42, _EOS])
    state._eager_rows = (batch, [(0, 0, "prefill", True), (1, 1, "decode", False), (2, 2, "ended", False)], input_ids)
    text_hidden = torch.arange(5 * _EAGER_DIM, dtype=torch.float32).reshape(5, _EAGER_DIM)
    sampled = torch.tensor([[7], [_EOS], [9]])
    outputs = _eager_outputs(5)

    state.run_eager_mtp(batch, text_hidden, sampled, outputs)

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
    assert state._eager_ready == {0: "prefill", 1: "decode", 2: "ended"}
    assert torch.equal(state._eager_embeds[0], torch.full((_EAGER_DIM,), 107.0))
    assert state._eager_rows is None


def test_eager_mtp_ignores_rows_recorded_for_another_batch():
    state = _make_eager_state()
    _fill_buffers(state, "r0")
    state._eager_rows = (_EagerBatch([1]), [(0, 0, "r0", False)], torch.tensor([5]))
    outputs = _eager_outputs(1)
    state.run_eager_mtp(_EagerBatch([1]), torch.zeros(1, _EAGER_DIM), torch.tensor([[3]]), outputs)
    assert state.model.mtp_calls == []
    assert not outputs["codes"]["audio"].any()


def test_eager_mtp_requires_retained_frame_outputs():
    state = _make_eager_state()
    _fill_buffers(state, "r0")
    batch = _EagerBatch([1])
    state._eager_rows = (batch, [(0, 0, "r0", False)], torch.tensor([5]))
    with pytest.raises(RuntimeError, match="requires retained codes.audio"):
        state.run_eager_mtp(batch, torch.zeros(1, _EAGER_DIM), torch.tensor([[3]]), {"codes": {"audio": None}})


def test_eager_decode_input_is_previous_frame_plus_text_step():
    state = _make_eager_state()
    _fill_buffers(state, "r0", "r1")
    state._eager_ready = {0: "r0", 1: "r1"}
    state._eager_embeds[0] = 5.0
    state._eager_embeds[1] = 7.0
    embeds = torch.zeros((2, _EAGER_DIM))
    batches = [
        (0, 0, (torch.zeros(_EAGER_DIM), torch.full((_EAGER_DIM,), 2.0))),
        (1, 1, (None, torch.ones(_EAGER_DIM))),
    ]
    state._eager_state._apply_eager_frames(batches, embeds, _EagerBatch([1, 1]), None)
    assert embeds.tolist() == [[7.0] * _EAGER_DIM, [8.0] * _EAGER_DIM]
    # The deferred MTP must not run for an eager decode row.
    assert state.model.mtp_calls == []


def test_eager_decode_without_a_frame_fails_loudly():
    state = _make_eager_state()
    _fill_buffers(state, "r0")
    batches = [(0, 0, (torch.zeros(_EAGER_DIM), torch.zeros(_EAGER_DIM)))]
    with pytest.raises(RuntimeError, match="frame missing"):
        state._eager_state._apply_eager_frames(batches, torch.zeros((1, _EAGER_DIM)), _EagerBatch([1]), None)


def test_run_preprocess_records_rows_that_keep_a_sample():
    state = _make_eager_state()
    _fill_buffers(state, "chunk", "final", "decode")
    state._eager_ready = {2: "decode"}
    state._eager_embeds[2] = 4.0
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

    recorded_batch, entries, _ids = state._eager_rows
    assert recorded_batch is batch
    # A non-final prefill chunk samples nothing that is kept.
    assert entries == [(1, 1, "final", True), (2, 2, "decode", False)]
    assert model_inputs["inputs_embeds"][6].tolist() == [5.0] * _EAGER_DIM
    assert state.model.mtp_calls == []


@pytest.mark.parametrize("first_was_valid", [False, True])
def test_first_audio_requirement_survives_eos_only_if_audio_was_queued(first_was_valid):
    state = _make_eager_state()
    _fill_buffers(state, "r0")
    state._first_audio_requests.add("r0")
    state._eager_state._first_audio_valid = torch.tensor([first_was_valid, False, False, False])
    batch = _EagerBatch([1])
    state._eager_rows = (batch, [(0, 0, "r0", False)], torch.tensor([5]))
    outputs = _eager_outputs(1)
    outputs["meta"]["first_audio"] = torch.zeros(1, dtype=torch.bool)
    state.run_eager_mtp(batch, torch.zeros(1, _EAGER_DIM), torch.tensor([[_EOS]]), outputs)
    assert outputs["meta"]["first_audio"].tolist() == [first_was_valid]


@pytest.mark.parametrize("accepted", [[], ["r1"], ["r0", "r1"]])
def test_first_audio_marker_requires_accepted_delivery(monkeypatch, accepted):
    from contextlib import nullcontext

    state = _make_eager_state()
    _fill_buffers(state, "r0", "r1")
    state.model.first_frame_decoder = SimpleNamespace(
        sample_rate=24000, decode=lambda codes: torch.ones(codes.shape[0], 2)
    )
    state._first_audio_stream = SimpleNamespace(wait_stream=lambda stream: None)
    state._first_audio_sender = SimpleNamespace(submit=lambda ids, pcm, sr, valid: accepted)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    batch = _EagerBatch([1, 1])
    state._eager_rows = (batch, [(0, 0, "r0", True), (1, 1, "r1", True)], torch.zeros(2, dtype=torch.long))
    outputs = _eager_outputs(2)
    outputs["meta"]["first_audio"] = torch.zeros(2, dtype=torch.bool)

    state.run_eager_mtp(batch, torch.zeros(2, _EAGER_DIM), torch.tensor([[7], [8]]), outputs)

    assert state._first_audio_requests == set(accepted)
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

    # Speculative steps sample several tokens per row: not published.
    assert state.publish_sampled_embeddings(batch, torch.tensor([[11, 1], [12, 1], [13, 1], [14, 1]])) is None


def test_publish_sampled_embeddings_is_opt_in() -> None:
    state = _make_state()
    state.model.publishes_sampled_embeddings = False
    assert state.publish_sampled_embeddings(SimpleNamespace(num_reqs=1), torch.tensor([[1]])) is None


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
