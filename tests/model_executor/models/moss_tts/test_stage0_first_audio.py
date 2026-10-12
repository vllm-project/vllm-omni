# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from tests.model_executor.models.moss_tts.test_local_model_state import _batch, _state
from vllm_omni.model_executor.models.moss_tts.first_audio_state import MossEarlyFirstAudioState
from vllm_omni.model_executor.models.moss_tts.local_model_state import MossLocalModelState, _CodeRowsSnapshot
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSLocalTalkerForGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def early_state():
    owner = SimpleNamespace(
        model=SimpleNamespace(audio_pad_token_id=16, audio_assistant_slot_token_id=7),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096, logits_processors=None)),
    )
    state = MossEarlyFirstAudioState(owner, None)
    owner._first_audio_sender = object()
    return state


@pytest.mark.parametrize("batch_prefill", [False, True])
@pytest.mark.parametrize("computed,count,eligible", [(0, 2, False), (2, 2, True), (3, 1, True)])
def test_only_completed_prefill_arms_first_audio_including_prefix_hits(batch_prefill, computed, count, eligible):
    state = _state(MossLocalModelState, torch.device("cpu"))
    state._batch_prefill = batch_prefill
    state._early_first_audio = MossEarlyFirstAudioState(state, None)
    state._first_audio_sender = object()
    state.intermediate_buffer.buffers[0] = {
        "req_id": "a",
        "codes": {"ref": torch.tensor([[1, 2], [3, 4], [5, 6], [2, 3]])},
        "sampling_params": SamplingParams(max_tokens=10),
    }
    batch = _batch(torch.device("cpu"), [0], [count])
    req = SimpleNamespace(prompt_len=np.full(5, 4), num_computed_tokens=np.full(5, computed))
    with torch.inference_mode():
        state.run_preprocess(batch, {"input_ids": batch.input_ids}, req)
    assert ("a" in state._early_first_audio.waiting) == eligible


def test_owned_first_codes_batch_reorder_and_one_time_promise(mocker):
    state = early_state()
    state.record_prefill("b", SamplingParams(max_tokens=10), prompt_len=4)
    publish = mocker.patch.object(state, "_publish", return_value=["b"])
    codes = torch.tensor([[1, 2], [3, 4], [5, 6]])
    state.after_mtp(["a", "b", "c"], codes, torch.tensor([7, 7, 7]))
    codes.fill_(99)
    assert publish.call_args.args[1].tolist() == [[3, 4]]
    flags = state.take_flags(["c", "b", "a"], codes.device)
    assert flags.tolist() == [False, True, False]
    snapshot = _CodeRowsSnapshot(codes.clone(), [0, 1, 2], 3, flags)
    copies = []

    def copy(tensor):
        copies.append(tensor)
        return tensor.clone()

    host = snapshot.copy_to_cpu(copy)
    flags.zero_()
    assert len(copies) == 2
    assert [bool(x) for x in host["meta"]["first_audio"]] == [False, True, False]
    assert state.take_flags(["b"], codes.device) is None
    state.after_mtp(["a", "b", "c"], codes, torch.tensor([7, 7, 7]))
    assert publish.call_count == 1
    state.remove("b")
    assert not state.seen and not state.waiting and not state.delivered and not state.updates


@pytest.mark.parametrize(
    "codes,token,valid",
    [([16, 16], 7, False), ([1, 2], 9, False), ([1, 2], 7, True), ([16, 16], None, False), ([1, 2], None, True)],
)
def test_stop_and_non_audio_tokens_do_not_promise_pcm(codes, token, valid, mocker):
    state = early_state()
    state.record_prefill("a", SamplingParams(max_tokens=10), prompt_len=4)
    publish = mocker.patch.object(state, "_publish", return_value=["a"])
    state.after_mtp(["a"], torch.tensor([codes]), None if token is None else torch.tensor([token]))
    assert publish.call_args.args[2].tolist() == [valid]
    assert state.take_flags(["a"], torch.device("cpu")).tolist() == [valid]


def test_rejected_route_and_cancel_keep_regular_path(mocker):
    state = early_state()
    state.record_prefill("a", SamplingParams(max_tokens=10), prompt_len=4)
    publish = mocker.patch.object(state, "_publish", return_value=[])
    state.after_mtp(["a"], torch.tensor([[1, 2]]), torch.tensor([7]))
    assert state.take_flags(["a"], torch.device("cpu")) is None
    state.record_prefill("b", SamplingParams(max_tokens=10), prompt_len=4)
    state.remove("b")
    state.after_mtp(["b"], torch.tensor([[1, 2]]), torch.tensor([7]))
    assert publish.call_count == 1


def test_cap_one_and_unbound_route_do_not_arm_first_audio():
    state = early_state()
    state.record_prefill("a", SamplingParams(max_tokens=1), prompt_len=4)
    state.owner._first_audio_sender = None
    state.record_prefill("b", SamplingParams(max_tokens=10), prompt_len=4)
    assert not state.waiting


@pytest.mark.parametrize(
    "stop_ids,eos,ignore_eos,eligible",
    [
        ([], 9, False, True),
        ([9], 9, False, True),
        ([7], 9, False, False),
        ([], 7, False, False),
        ([], 7, True, True),
        ([7], 7, True, False),
    ],
)
def test_first_token_stop_matches_effective_sampling_params(stop_ids, eos, ignore_eos, eligible, mocker):
    state = early_state()
    params = SamplingParams(max_tokens=10, stop_token_ids=stop_ids, ignore_eos=ignore_eos)
    params.update_from_generation_config({}, eos_token_id=eos)
    state.record_prefill("a", params, prompt_len=4)
    publish = mocker.patch.object(state, "_publish", side_effect=lambda ids, *_: ids)
    state.after_mtp(["a"], torch.tensor([[1, 2]]))
    assert publish.called == eligible
    flags = state.take_flags(["a"], torch.device("cpu"))
    assert (flags is not None) == eligible


@pytest.mark.parametrize("prompt_len,eligible", [(4094, True), (4095, False)])
def test_context_limit_leaves_first_frame_on_regular_path(prompt_len, eligible):
    state = early_state()
    state.record_prefill("a", SamplingParams(max_tokens=10), prompt_len=prompt_len)
    assert ("a" in state.waiting) == eligible


@pytest.mark.parametrize(
    "overrides",
    [
        {"stop": ["stop"]},
        {"min_tokens": 2},
        {"allowed_token_ids": [9]},
        {"bad_words": ["word"]},
        {"logit_bias": {7: -100}},
        {"structured_outputs": StructuredOutputsParams(choice=["yes", "no"])},
        {"thinking_token_budget": 0},
        {"trace_decode_token_ids": [9]},
    ],
)
def test_constrained_requests_keep_regular_delivery(overrides):
    state = early_state()
    state.record_prefill("a", SamplingParams(max_tokens=10, **overrides), prompt_len=4)
    assert not state.waiting


def test_custom_logits_processors_keep_regular_delivery():
    state = early_state()
    state.owner.vllm_config.model_config.logits_processors = ["custom.Processor"]
    state.record_prefill("a", SamplingParams(max_tokens=10), prompt_len=4)
    assert not state.waiting


def test_mixed_batch_only_publishes_unconstrained_request(mocker):
    state = early_state()
    for request_id, stops in [("stop", [7]), ("normal", [9])]:
        state.record_prefill(request_id, SamplingParams(max_tokens=10, stop_token_ids=stops), prompt_len=4)
    publish = mocker.patch.object(state, "_publish", side_effect=lambda ids, *_: ids)
    state.after_mtp(["stop", "normal"], torch.tensor([[1, 2], [3, 4]]))
    assert publish.call_args.args[0] == ["normal"]
    assert publish.call_args.args[1].tolist() == [[3, 4]]
    assert state.take_flags(["stop", "normal"], torch.device("cpu")).tolist() == [False, True]


@pytest.mark.parametrize(
    "backend,tp,pp,loads_decoder",
    [("uni", 1, 1, True), ("mp", 1, 1, False), ("uni", 2, 1, False), ("uni", 1, 2, False)],
)
def test_weight_loading_skips_unusable_first_decoder(mocker, backend, tp, pp, loads_decoder):
    model = torch.nn.Module()
    model.config = SimpleNamespace(mrv2_gpu_slot_state=True)
    model.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(use_v2_model_runner=True, async_chunk=True, enforce_eager=True),
        parallel_config=SimpleNamespace(
            distributed_executor_backend=backend, tensor_parallel_size=tp, pipeline_parallel_size=pp
        ),
    )
    model.model = SimpleNamespace(load_weights=lambda weights: set())
    model.audio_embeddings = torch.nn.ModuleList([torch.nn.Embedding(2, 2)])
    model.local_transformer = SimpleNamespace(ln_f=torch.nn.LayerNorm(2))
    model.n_vq = 1
    decoder = torch.nn.Module()
    decoder.load = mocker.Mock(return_value={"weight"})
    constructor = mocker.patch(
        "vllm_omni.model_executor.models.moss_tts.first_frame_decoder.MossFirstFrameDecoder", return_value=decoder
    )
    loaded = MossTTSLocalTalkerForGeneration.load_weights(model, [])
    assert constructor.call_count == int(loads_decoder)
    assert decoder.load.call_count == int(loads_decoder)
    assert ("first_frame_decoder.weight" in loaded) == loads_decoder
