# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eager MTP emits the same frames one step earlier with identical backbone inputs."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams

from tests.model_executor.models.moss_tts.test_local_model_state import _admit, _batch, _state, _step
from vllm_omni.model_executor.models.moss_tts.first_audio_state import MossEarlyFirstAudioState
from vllm_omni.model_executor.models.moss_tts.local_model_state import MossLocalModelState

pytestmark = pytest.mark.core_model


@pytest.fixture(params=[pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device(request.param)


def _run(state, device, schedule, stop_step):
    computed, prompt = np.zeros(5, dtype=np.int32), np.zeros(5, dtype=np.int32)
    prompt[3], prompt[0] = 4, 2
    req_states = SimpleNamespace(prompt_len=prompt, num_computed_tokens=computed)
    embeds = []
    frames: dict[int, list[torch.Tensor]] = {3: [], 0: []}
    for index, (slots, counts) in enumerate(schedule):
        state.model.force_stop = index == stop_step
        batch = _batch(device, slots, counts)
        batch.req_ids = [state.intermediate_buffer.buffers[slot]["req_id"] for slot in slots]
        embed, payload, _ = _step(state, batch, req_states)
        embeds.append(embed)
        rows = payload.get("codes", {}).get("audio", [])
        for slot, row in zip(slots, rows):
            if row.numel() and bool(row.ne(state.model.audio_pad_token_id).any()):
                frames[slot].append((index, row.clone()))
        for slot, count in zip(slots, counts):
            computed[slot] += count
    return embeds, frames


@pytest.mark.parametrize("first_only", [False, True])
def test_eager_frames_match_canonical_one_step_earlier(device, mocker, first_only):
    schedule = [([3, 0], [2, 2]), ([3, 0], [2, 1]), ([3, 0], [1, 1]), ([0, 3], [1, 1]), ([3, 0], [1, 1])]
    canonical, eager = (_state(MossLocalModelState, device) for _ in range(2))
    eager._local_eager_mtp = not first_only
    eager._early_first_audio = MossEarlyFirstAudioState(eager, None)
    eager._first_audio_sender = object()
    publish = mocker.patch.object(eager._early_first_audio, "_publish", side_effect=lambda ids, *_: ids)
    for state in (canonical, eager):
        _admit(state, 3, "long", 17)
        _admit(state, 0, "short", 17)
        for slot in (3, 0):
            state.intermediate_buffer.buffers[slot]["sampling_params"] = SamplingParams(
                max_tokens=10, extra_args={"tts_local_seed": 17}
            )
    # Stop both streams at the canonical step that would draw the 4th step's frames.
    c_embeds, c_frames = _run(canonical, device, schedule, stop_step=4)
    e_embeds, e_frames = _run(eager, device, schedule, stop_step=4 if first_only else 3)

    for c, e in zip(c_embeds, e_embeds):
        torch.testing.assert_close(c, e, rtol=0, atol=0)
    for slot in (3, 0):
        assert [f for _, f in c_frames[slot]] and len(c_frames[slot]) == len(e_frames[slot])
        for (ci, cf), (ei, ef) in zip(c_frames[slot], e_frames[slot]):
            assert ei == ci - (0 if first_only else 1)
            torch.testing.assert_close(cf, ef, rtol=0, atol=0)
    assert [call.args[0] for call in publish.call_args_list] == [["short"], ["long"]]
    for call, slot in zip(publish.call_args_list, (0, 3), strict=True):
        torch.testing.assert_close(call.args[1], e_frames[slot][0][1], rtol=0, atol=0)
        assert call.args[2].tolist() == [True]


def test_stopped_stream_does_not_emit_again(device):
    state = _state(MossLocalModelState, device)
    state._local_eager_mtp = True
    _admit(state, 0, "short", 17)
    computed, prompt = np.zeros(5, dtype=np.int32), np.zeros(5, dtype=np.int32)
    prompt[0] = 2
    req_states = SimpleNamespace(prompt_len=prompt, num_computed_tokens=computed)
    emitted = []
    for index, count in enumerate([2, 1, 1, 1]):
        state.model.force_stop = index == 1
        _, payload, _ = _step(state, _batch(device, [0], [count]), req_states)
        row = payload["codes"]["audio"][0]
        emitted.append(bool(row.ne(state.model.audio_pad_token_id).any()))
        computed[0] += count
        state.model.force_stop = False
    # Frame at the prefill step, stop at step 1, then asynchronous overrun steps.
    assert emitted == [True, False, False, False]


def test_unmarked_decode_row_ignores_stale_slot_state(device):
    canonical, eager = (_state(MossLocalModelState, device) for _ in range(2))
    eager._local_eager_mtp = True
    for state in (canonical, eager):
        _admit(state, 0, "short", 17)
    # A decode row whose completing prefill was not observed in this state.
    eager._keep_pool[0] = False
    eager._eager_emb_pool[0].fill_(3.0)
    computed, prompt = np.zeros(5, dtype=np.int32), np.zeros(5, dtype=np.int32)
    prompt[0], computed[0] = 2, 2
    req_states = SimpleNamespace(prompt_len=prompt, num_computed_tokens=computed)
    for state in (canonical, eager):
        state.intermediate_buffer.buffers[0]["audio_state"] = {"is_stopping": False}
    c_embed, c_payload, _ = _step(canonical, _batch(device, [0], [1]), req_states)
    e_embed, e_payload, _ = _step(eager, _batch(device, [0], [1]), req_states)
    torch.testing.assert_close(c_embed, e_embed, rtol=0, atol=0)
    torch.testing.assert_close(c_payload["codes"]["audio"][0], e_payload["codes"]["audio"][0], rtol=0, atol=0)


@pytest.mark.cpu
def test_first_prefill_through_runner_preserves_parent_eager_contract(monkeypatch, mocker):
    """The Local config must not enable the parent's incompatible eager hook."""
    from vllm_omni.worker_v2 import omni_ar_model_runner as runner_module
    from vllm_omni.worker_v2.model_states.eager_mtp import EagerMTPState
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

    device = torch.device("cpu")
    template = _state(MossLocalModelState, device)
    model = template.model
    model.config = SimpleNamespace(mrv2_eager_mtp=True)

    def initialize_parent(state, *args):
        state.__dict__.update(template.__dict__)
        state._eager_mtp, state._eager_rows = False, None
        state._eager_state = EagerMTPState(state)

    monkeypatch.setattr(OmniModelState, "__init__", initialize_parent)
    state = MossLocalModelState(SimpleNamespace(), model, None, device)
    _admit(state, 0, "req", 17)
    batch = _batch(device, [0], [2])
    batch.req_ids = ["req"]
    computed, prompt = np.zeros(5, dtype=np.int32), np.zeros(5, dtype=np.int32)
    prompt[0] = 2
    _step(state, batch, SimpleNamespace(prompt_len=prompt, num_computed_tokens=computed))
    hook = mocker.spy(state, "run_eager_mtp")

    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.execute_model_state = SimpleNamespace(
        input_batch=batch,
        hidden_states=torch.zeros(2, 4),
        finished_req_ids=set(),
        ec_connector_output=None,
    )
    runner._kv_extracted_req_ids = runner._last_aux_output = runner._last_multimodal_outputs = None
    runner.is_last_pp_rank, runner.pp_handler, runner.check_ep_fault = True, None, False
    runner.aux_output_connector = None
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type="text"))
    runner.model_state, runner.model = state, model
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.zeros(5, 2)),
        num_computed_tokens=SimpleNamespace(gpu=torch.from_numpy(computed)),
        prompt_len=SimpleNamespace(np=prompt, gpu=torch.from_numpy(prompt)),
    )
    runner.main_stream = runner.output_copy_stream = mocker.Mock()
    runner.eplb = runner._finalize_native_data_plane_output = runner._reserve_native_data_plane_outputs = mocker.Mock()
    runner.sampler = None
    runner.sample = mocker.Mock(
        return_value=(
            SimpleNamespace(sampled_token_ids=torch.tensor([[2]])),
            torch.ones(1),
            torch.zeros(1),
        )
    )
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=mocker.Mock(return_value={}),
        compute_prompt_token_id_logprobs=mocker.Mock(return_value={}),
    )
    runner.postprocess_sampled = mocker.Mock()
    runner.kv_connector = SimpleNamespace(post_forward=mocker.Mock(return_value=None))
    mock_out = SimpleNamespace(copy_event=None)
    monkeypatch.setattr(runner_module, "OmniAsyncOutput", mocker.Mock(return_value=mock_out))

    assert runner.sample_tokens(None) is mock_out
    hook.assert_called_once()
    assert state._eager_mtp is False and state._eager_rows is None
    assert state._local_eager_mtp is True
