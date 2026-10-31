# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.voxtral_tts.model_state import VoxtralTTSModelState
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_state(monkeypatch):
    def add_request(self, req_index, new_req_data):
        self.intermediate_buffer.buffers[req_index] = {
            "req_id": new_req_data.req_id,
            **(new_req_data.model_intermediate_buffer or {}),
        }

    monkeypatch.setattr(OmniModelState, "add_request", add_request)
    monkeypatch.setattr(OmniModelState, "on_request_preempted", lambda self, req_id, req_index: None)
    monkeypatch.setattr(OmniModelState, "on_requests_finished", lambda self, req_ids: None)

    state = object.__new__(VoxtralTTSModelState)
    state.model = SimpleNamespace(model_stage="audio_generation")
    state.intermediate_buffer = SimpleNamespace(buffers=[{}, {}, {}])
    state._suspended_feedback_history = {}
    return state


def _request(req_id, *, generated=(), model_intermediate_buffer=None):
    prompt = [11, 12]
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=prompt,
        prefill_token_ids=prompt + list(generated),
        model_intermediate_buffer=model_intermediate_buffer,
    )


def test_preempted_history_follows_request_to_reused_slot_and_trims_unaccepted_frame(monkeypatch):
    state = _make_state(monkeypatch)
    frames = [torch.tensor([10, 11, 12, 13]), torch.tensor([20, 21, 22, 23]), torch.tensor([30, 31, 32, 33])]
    state.intermediate_buffer.buffers[0] = {
        "req_id": "request-a",
        "codes": {"audio": frames[-1].reshape(1, 1, -1), "audio_history": frames},
    }

    state.on_request_preempted("request-a", 0)
    saved = state._suspended_feedback_history["request-a"]
    assert saved.device.type == "cpu"
    torch.testing.assert_close(saved, torch.stack(frames))

    # The old slot is cleared/reused before the request is restored elsewhere.
    frames[0].fill_(99)
    state.intermediate_buffer.buffers[0] = {}
    state.add_request(0, _request("request-b"))
    assert "codes" not in state.intermediate_buffer.buffers[0]

    # Scheduler replay contains two accepted generated tokens; the extra saved
    # frame belongs to an output that was not accepted before preemption.
    state.add_request(2, _request("request-a", generated=[101, 102]))
    restored = state.intermediate_buffer.buffers[2]["codes"]
    assert len(restored["audio_history"]) == 2
    torch.testing.assert_close(restored["audio_history"][0], torch.tensor([10, 11, 12, 13]))
    torch.testing.assert_close(restored["audio_history"][1], torch.tensor([20, 21, 22, 23]))
    torch.testing.assert_close(restored["audio"], torch.tensor([20, 21, 22, 23]).reshape(1, 1, -1))
    assert "request-a" not in state._suspended_feedback_history

    # A later preemption after resumed generation must snapshot both the CPU
    # frames restored above and the newly generated frame.
    restored["audio_history"].append(torch.tensor([40, 41, 42, 43]))
    state.on_request_preempted("request-a", 2)
    torch.testing.assert_close(
        state._suspended_feedback_history["request-a"],
        torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23], [40, 41, 42, 43]]),
    )


def test_scheduler_preemption_restores_voxtral_frames_without_cross_request_leakage():
    from torch import nn
    from vllm.sampling_params import SamplingParams
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus

    from tests.core.sched.test_omni_recompute_preemption import _scheduler
    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts import (
        VoxtralTTSForConditionalGeneration,
    )
    from vllm_omni.worker_v2.model_states.eager_mtp import EagerMTPState
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

    class FakeAudioGeneration:
        next_frames: list[torch.Tensor] = []

        def embed_multimodal(self, *, audio_tokens):
            values = audio_tokens[0, :, 0].to(torch.float32)
            return [values.unsqueeze(-1).expand(-1, 4)]

        def compute_mm_logits(self, hidden_states, cfg_alpha):
            del cfg_alpha
            fake_eos = torch.zeros((hidden_states.shape[0], 1), dtype=hidden_states.dtype)
            return fake_eos, {"codes": {"audio": self.next_frames}}

    model = VoxtralTTSForConditionalGeneration.__new__(VoxtralTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "audio_generation"
    model.model = FakeAudioGeneration()
    model._audio_token_id = 42
    model._cudagraph_acoustic_transformer = None

    state = object.__new__(VoxtralTTSModelState)
    state.model = model
    state.rope_state = None
    state.prompt_embeds_state = None
    state.intermediate_buffer = OmniIntermediateBuffer(4)
    state._suspended_feedback_history = {}
    state._mtp_generators = {}
    state._mtp_seeded = set()
    state._eager_state = EagerMTPState(state)

    scheduler = _scheduler("allow", block_size=4, num_blocks=4)
    scheduler.use_v2_model_runner = True
    blocker = Request("blocker", [1] * 8, SamplingParams(max_tokens=8), None)
    assert scheduler.kv_cache_manager.allocate_slots(blocker, 8) is not None

    request_a = Request("request-a", [11, 12, 13, 14], SamplingParams(max_tokens=8), None)
    scheduler.add_request(request_a)
    first_prefill = scheduler.schedule()
    assert first_prefill.num_scheduled_tokens == {"request-a": 4}
    state.add_request(0, first_prefill.scheduled_new_reqs[0])

    def emit_mrv2_frames(req_ids, request_infos, values, scheduled_tokens):
        model.model.next_frames = [torch.full((1, 1, 4), value) for value in values]
        model.make_omni_output_mrv2(
            torch.zeros((len(req_ids), 4)),
            input_batch=SimpleNamespace(
                num_reqs=len(req_ids),
                logits_indices=torch.arange(len(req_ids)),
                num_computed_prefill_tokens_np=[0] * len(req_ids),
                num_scheduled_tokens=scheduled_tokens,
                prefill_len_np=scheduled_tokens,
            ),
            req_states=None,
            model_intermediate_buffer=request_infos,
        )

    emit_mrv2_frames(["request-a"], [state.intermediate_buffer.buffers[0]], [101.0], [4])

    def runner_output(req_ids):
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=[[42] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    first_codes = state.intermediate_buffer.buffers[0]["codes"]
    assert first_codes["audio_history"][0] is first_codes["audio"]

    scheduler.update_from_output(first_prefill, runner_output(["request-a"]))
    preemption = scheduler.schedule()
    assert request_a.status == RequestStatus.PREEMPTED
    assert preemption.preempted_req_ids == {"request-a"}
    assert scheduler.update_from_output(preemption, runner_output([])) == {}

    # Exercise the production model-state hooks and actual runner-slot cleanup.
    state.on_request_preempted("request-a", 0)
    torch.testing.assert_close(
        state._suspended_feedback_history["request-a"],
        torch.full((1, 4), 101.0),
    )
    state.remove_request(0)
    request_b = Request("request-b", [21, 22, 23, 24], SamplingParams(max_tokens=8), None)
    scheduler.add_request(request_b)
    scheduler.kv_cache_manager.free(blocker)

    resumed = scheduler.schedule()
    assert set(resumed.num_scheduled_tokens) == {"request-a", "request-b"}
    new_requests = {request.req_id: request for request in resumed.scheduled_new_reqs}
    # B reuses A's former slot while A resumes in a different slot.
    slot_by_id = {"request-a": 1, "request-b": 0}
    state.add_request(slot_by_id["request-a"], new_requests["request-a"])
    state.add_request(slot_by_id["request-b"], new_requests["request-b"])

    # Replay the real scheduler's prompt-plus-generated-token sequence. The
    # preempted request gets frame 101; the new request has no generated frame.
    request_a_info = state.intermediate_buffer.buffers[slot_by_id["request-a"]]
    request_a_data = new_requests["request-a"]
    _, replayed_a, _ = model.tts_preprocess(
        torch.tensor(request_a_data.prefill_token_ids),
        torch.zeros((len(request_a_data.prefill_token_ids), 4)),
        _omni_prompt_len=len(request_a_data.prompt_token_ids),
        _omni_num_computed_tokens=0,
        **{key: value for key, value in request_a_info.items() if key != "req_id"},
    )
    torch.testing.assert_close(replayed_a[-1], torch.full((4,), 101.0))
    request_b_info = state.intermediate_buffer.buffers[slot_by_id["request-b"]]
    assert "codes" not in request_b_info

    # The next batched model step emits distinct frames; each request's decode
    # must consume its own latest frame after the two scheduler slots coexist.
    request_ids = list(resumed.num_scheduled_tokens)
    request_infos = [state.intermediate_buffer.buffers[slot_by_id[req_id]] for req_id in request_ids]
    emit_mrv2_frames(
        request_ids,
        request_infos,
        [102.0 if req_id == "request-a" else 202.0 for req_id in request_ids],
        [resumed.num_scheduled_tokens[req_id] for req_id in request_ids],
    )
    scheduler.update_from_output(resumed, runner_output(request_ids))

    for req_id, request_data, expected in (
        ("request-a", request_a_data, 102.0),
        ("request-b", new_requests["request-b"], 202.0),
    ):
        info = state.intermediate_buffer.buffers[slot_by_id[req_id]]
        _, decoded, _ = model.tts_preprocess(
            torch.tensor([model._audio_token_id]),
            torch.zeros((1, 4)),
            _omni_prompt_len=len(request_data.prompt_token_ids),
            _omni_num_computed_tokens=resumed.num_scheduled_tokens[req_id],
            **{key: value for key, value in info.items() if key != "req_id"},
        )
        torch.testing.assert_close(decoded[0], torch.full((4,), expected), msg=req_id)


def test_finishing_a_suspended_request_releases_its_feedback(monkeypatch):
    state = _make_state(monkeypatch)
    state.intermediate_buffer.buffers[1] = {
        "req_id": "request-a",
        "codes": {"audio_history": [torch.ones(1, 1, 4)]},
    }
    state.on_request_preempted("request-a", 1)

    state.on_requests_finished({"request-a"})

    assert "request-a" not in state._suspended_feedback_history


def test_resuming_generated_tokens_without_feedback_fails_loudly(monkeypatch):
    state = _make_state(monkeypatch)

    with pytest.raises(RuntimeError, match="no acoustic feedback history"):
        state.add_request(1, _request("request-a", generated=[101]))
