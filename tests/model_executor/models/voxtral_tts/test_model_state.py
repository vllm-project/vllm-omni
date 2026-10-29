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
