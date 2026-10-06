# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare slot lifecycle with canonical Local hooks, including output ownership."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_omni.model_executor.models.moss_tts.local_model_state import MossLocalModelState
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSLocalTalkerForGeneration
from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.eager_mtp import EagerMTPState
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = pytest.mark.core_model


@pytest.fixture(params=[pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device(request.param)


def _model(device):
    model = MossTTSLocalTalkerForGeneration.__new__(MossTTSLocalTalkerForGeneration)
    torch.nn.Module.__init__(model)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.model = torch.nn.Module()
    weight = torch.arange(32, dtype=dtype, device=device).reshape(8, 4) / 16
    model.model.embed_tokens = torch.nn.Embedding(8, 4, _weight=weight)
    model.hidden_size, model.n_vq = 4, 2
    model.audio_pad_token_id, model.text_vocab_size = 8, 8
    model.audio_assistant_slot_token_id, model.im_end_token_id = 2, 3
    model.talker_mtp_output_key = ("audio_codes", "current")
    model.talker_mtp_graph_safe = False
    model.talker_mtp_accepts_per_row_generators = True
    model.gpu_resident_buffer_keys = {("hidden_states", "last"), ("audio_codes", "current")}
    model.audio_lm_heads = model.audio_embeddings = model.local_text_lm_head = None
    model._audio_embed = lambda codes: codes[:, :1].expand(-1, 4).to(dtype) / 8
    model.force_stop = False

    def frame(hidden, *_args, generator=None, generators=None, temperature=None, top_k=None, top_p=None, **_kwargs):
        assert (temperature, top_k, top_p) == (1.7, 25, 0.8)
        if generators is not None:
            noise = torch.cat([torch.randint(0, 8, (1, 2), device=device, generator=g) for g in generators])
        else:
            noise = torch.randint(0, 8, (hidden.shape[0], 2), device=device, generator=generator)
        codes = (hidden[:, :2].mul(16).long() + noise) % 8
        return torch.full((hidden.shape[0],), not model.force_stop, device=device), codes

    model.local_transformer = SimpleNamespace(generate_frame=frame)
    return model


def _state(cls, device):
    state = object.__new__(cls)
    state.model = _model(device)
    state.device, state.dtype = device, state.model.model.embed_tokens.weight.dtype
    state.has_preprocess = state.has_postprocess = state.have_multimodal_outputs = True
    state.scheduler_config = SimpleNamespace(max_num_seqs=5)
    state.vllm_config = SimpleNamespace()
    state.intermediate_buffer = OmniIntermediateBuffer(5)
    state._static_inputs_embeds = torch.zeros(16, 4, device=device, dtype=state.dtype)
    state._mtp_input_ids = torch.zeros(5, device=device, dtype=torch.long)
    state._mtp_offsets = torch.zeros(5, device=device, dtype=torch.long)
    for name in ("_mtp_input_embeds", "_mtp_hidden", "_mtp_text_step"):
        setattr(state, name, torch.zeros(5, 4, device=device, dtype=state.dtype))
    state._mtp_runner = state._mtp_sample_uniforms = None
    state._mtp_generators = {}
    state._stream_pos = {}
    state._stream_decode_event = None
    state._eager_state = EagerMTPState(state)
    if cls is MossLocalModelState:
        state._init_slot_buffers(5, 4, device, state.dtype)
    return state


def _admit(state, slot, name, seed):
    params = SimpleNamespace(extra_args={"tts_local_seed": seed} if seed is not None else {}, seed=None)
    with patch.object(DefaultModelState, "add_request", return_value=None):
        state.add_request(slot, SimpleNamespace(req_id=name, mm_features=[], sampling_params=params))
    state.intermediate_buffer.buffers[slot]["codes"] = {"ref": torch.tensor([[1, 2], [3, 4], [5, 6], [2, 3]])}


def _batch(device, slots, counts, index_dtype=torch.int32):
    starts = np.array([0, *np.cumsum(counts)], dtype=np.int32)
    return SimpleNamespace(
        idx_mapping_np=np.array(slots),
        idx_mapping=torch.tensor(slots, device=device, dtype=index_dtype),
        num_reqs=len(slots),
        num_tokens=int(starts[-1]),
        num_scheduled_tokens=np.array(counts),
        query_start_loc_np=starts,
        query_start_loc=torch.tensor(starts, device=device),
        # Upstream MRV2 uses int32 token buffers, unlike the MTP graph inputs.
        input_ids=torch.full((int(starts[-1]),), 2, dtype=torch.int32, device=device),
    )


def _step(state, batch, req_states, dispatcher=None):
    inputs = {"input_ids": batch.input_ids, "inputs_embeds": state._static_inputs_embeds[: batch.num_tokens]}
    with (
        patch("vllm.forward_context.set_forward_context", return_value=nullcontext()),
        patch(
            "vllm_omni.model_executor.models.moss_tts.local_model_state.set_forward_context", return_value=nullcontext()
        ),
        torch.inference_mode(),
    ):
        state.run_preprocess(batch, inputs, req_states, dispatcher)
        hidden = inputs["inputs_embeds"] + 0.25
        state.run_postprocess(hidden, batch)
        _, payload = state.postprocess_model_output(hidden, batch, req_states)
        last = batch.query_start_loc[1:] - 1
        tokens = state.model.compute_logits(hidden.index_select(0, last)).argmax(-1)
        result = inputs["inputs_embeds"].clone(), payload, tokens
        hidden.fill_(-100)  # next graph replay can overwrite its output immediately
    return result


@pytest.mark.parametrize("seed", [None, 17])
@pytest.mark.parametrize("batch_prefill", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_slot_matches_canonical_mixed_prefill_reorder_stop_and_reuse(device, seed, batch_prefill, index_dtype):
    reference, candidate = (_state(cls, device) for cls in (OmniModelState, MossLocalModelState))
    candidate._batch_prefill = batch_prefill
    for state in (reference, candidate):
        _admit(state, 3, "long", seed)
        _admit(state, 0, "short", seed)
    computed, prompt = np.zeros(5, dtype=np.int32), np.zeros(5, dtype=np.int32)
    prompt[3], prompt[0] = 4, 2
    req_states = SimpleNamespace(prompt_len=prompt, num_computed_tokens=computed)
    saved = []
    for index, (slots, counts) in enumerate(
        [
            ([3, 0], [2, 2]),
            ([0, 3], [1, 1]),
            ([3, 0], [1, 1]),
            ([3, 0], [1, 1]),
            ([0, 3], [1, 1]),
            ([3, 0], [1, 1]),
        ]
    ):
        if index == 4:
            # Cancellation/removal followed by reuse must not retain hidden,
            # reference offset, control, or the old request's RNG stream.
            for state in (reference, candidate):
                state.remove_request("short")
                assert "short" not in state._mtp_generators
                _admit(state, 0, "replacement", seed)
            computed[0], prompt[0] = 0, 1
        outputs = []
        for state in (reference, candidate):
            state.model.force_stop = index == 3
            torch.manual_seed(91 + index)
            outputs.append(_step(state, _batch(device, slots, counts, index_dtype), req_states))
        a, b = outputs
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
        if index == 3:
            assert b[2].tolist() == [3, 3]
        assert a[1].keys() == b[1].keys()
        for ar, br in zip(a[1].get("codes", {}).get("audio", []), b[1].get("codes", {}).get("audio", [])):
            torch.testing.assert_close(ar, br, rtol=0, atol=0)
            saved.append((br, br.clone()))
        for slot, count in zip(slots, counts):
            computed[slot] += count
            old = reference.intermediate_buffer.buffers[slot]
            new = candidate.intermediate_buffer.buffers[slot]
            assert old["ref_offset"] == new["ref_offset"]
            torch.testing.assert_close(candidate._hidden_pool[slot], old["hidden_states"]["last"], rtol=0, atol=0)
            assert "hidden_states" not in new and "audio_codes" not in new
    candidate._codes_pool.fill_(8)
    candidate._hidden_pool.zero_()
    for current, snapshot in saved:
        torch.testing.assert_close(current, snapshot, rtol=0, atol=0)


def test_slot_honors_external_stopping_control(device):
    states = [_state(cls, device) for cls in (OmniModelState, MossLocalModelState)]
    for state in states:
        _admit(state, 0, "stop", None)
        req = SimpleNamespace(prompt_len=np.ones(5), num_computed_tokens=np.zeros(5))
        _step(state, _batch(device, [0], [1]), req)
        state.intermediate_buffer.buffers[0]["audio_state"]["is_stopping"] = True
        req.num_computed_tokens[:] = 1
        output = _step(state, _batch(device, [0], [1]), req)
        assert output[2].tolist() == [3]
        assert output[1]["codes"]["audio"][0].tolist() == [[8, 8]]


@pytest.mark.cpu
@pytest.mark.parametrize("state_cls", [OmniModelState, MossLocalModelState])
def test_local_seeded_mtp_keeps_per_row_generators_in_one_batch(state_cls):
    state = _state(state_cls, torch.device("cpu"))
    generators = [torch.Generator().manual_seed(seed) for seed in (17, 29)]
    inputs = [torch.zeros(2, dtype=torch.long), *[torch.ones(2, 4) for _ in range(3)]]
    frame = MagicMock(wraps=state.model.local_transformer.generate_frame)
    state.model.local_transformer.generate_frame = frame

    state._call_mtp_with_sampling(*inputs, buffers=[{}, {}], req_ids=["first", "second"], generators=generators)

    frame.assert_called_once()
    assert frame.call_args.args[0].shape[0] == 2
    assert frame.call_args.kwargs["generators"] == generators
    assert frame.call_args.kwargs["generator"] is None


@pytest.mark.cuda
def test_slot_mtp_graph_padding_reorder_and_owned_output():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    states = [_state(cls, torch.device("cuda")) for cls in (OmniModelState, MossLocalModelState)]
    for state in states:

        def frame(hidden, *_args, **_kwargs):
            return hidden[:, 0] < 3, hidden[:, :2].mul(16).long() % 8

        state.model.local_transformer.generate_frame = frame
        for slot in [0, 3, 4]:
            _admit(state, slot, f"req{slot}", None)
    candidate = states[1]
    buffers = [
        candidate._mtp_input_ids[:4],
        candidate._mtp_input_embeds[:4],
        candidate._mtp_hidden[:4],
        candidate._mtp_text_step[:4],
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(3):
            candidate.model.mtp(*buffers)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        graph_output = candidate.model.mtp(*buffers)

    def replay(*args, **kwargs):
        assert all(a.data_ptr() == b.data_ptr() for a, b in zip(args, buffers))
        assert args[0].shape[0] == 4
        graph.replay()
        return graph_output

    candidate._mtp_runner = replay
    candidate._is_mtp_graph_runner = lambda: True
    from vllm.config import CUDAGraphMode

    def dispatcher(bsz):
        return SimpleNamespace(num_tokens=4, cg_mode=CUDAGraphMode.FULL)

    req = SimpleNamespace(prompt_len=np.ones(5, dtype=np.int32), num_computed_tokens=np.zeros(5, dtype=np.int32))
    retained = []
    for slots in [[0, 3, 4], [3, 0], [4, 0, 3], [3, 4]]:
        out = [
            _step(
                state,
                _batch(torch.device("cuda"), slots, [1] * len(slots)),
                req,
                dispatcher if state is candidate else None,
            )
            for state in states
        ]
        torch.testing.assert_close(out[0][0], out[1][0], rtol=0, atol=0)
        torch.testing.assert_close(out[0][2], out[1][2], rtol=0, atol=0)
        for a, b in zip(out[0][1].get("codes", {}).get("audio", []), out[1][1].get("codes", {}).get("audio", [])):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            retained.append((b, b.clone()))
        req.num_computed_tokens[slots] += 1
    with torch.inference_mode():
        for tensor in graph_output:
            tensor.fill_(7)
    for tensor, copy in retained:
        torch.testing.assert_close(tensor, copy, rtol=0, atol=0)


@pytest.mark.cpu
def test_factory_requires_explicit_class_capability_and_valid_state(monkeypatch):
    monkeypatch.setattr(OmniModelState, "__init__", lambda *args: None)
    dynamic = MagicMock(has_preprocess=True, create_omni_model_state=None)
    assert isinstance(init_omni_model_state(None, dynamic, None, torch.device("cpu")), OmniModelState)
    dynamic.create_mrv2_model_state.assert_not_called()

    class Invalid:
        has_preprocess = True

        def create_mrv2_model_state(self, *args):
            return object()

    with pytest.raises(TypeError, match="must return a ModelState"):
        init_omni_model_state(None, Invalid(), None, torch.device("cpu"))


@pytest.mark.cpu
@pytest.mark.parametrize("enabled", [False, True])
def test_local_factory_default_and_option(monkeypatch, enabled):
    monkeypatch.setattr(OmniModelState, "__init__", lambda *args: None)
    monkeypatch.setattr(MossLocalModelState, "__init__", lambda *args: None)
    model = _model(torch.device("cpu"))
    model.config = SimpleNamespace(mrv2_gpu_slot_state=enabled)
    state = init_omni_model_state(None, model, None, torch.device("cpu"))
    assert type(state) is (MossLocalModelState if enabled else OmniModelState)
