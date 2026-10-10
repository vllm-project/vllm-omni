# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Natural speech EOF rebuilds AR KV without retiring waveform/session owners."""

from copy import deepcopy

import pytest
import torch

from tests.model_executor.models.lychee_fd.test_host_audio_delta import _bridge, _commit, _plan, _record_through
from tests.worker_v2.test_lychee_history_recovery import _expanded_logits, _history, _request, _step
from tests.worker_v2.test_lychee_model_state import _session_request, _session_state
from vllm_omni.model_executor.models.lychee_fd.duplex.codec import CodecStreamState
from vllm_omni.model_executor.models.lychee_fd.duplex.plugin import LycheeDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _record_eos(history, tick=8):
    if history.frontier_tick < tick - 1:
        _record_through(history, tick - 1)
    history.record_outputs(
        {
            "lychee_tick": torch.tensor([tick]),
            "lychee_text_token_ids": torch.tensor([101]),
            "lychee_speech_token_ids": torch.tensor([151694]),
            "lychee_control_token_ids": torch.tensor([158357]),
            "lychee_execution_epoch": torch.tensor([0]),
        }
    )


def test_natural_eos_plan_rebuilds_same_ar_owner_preserving_waveform_and_raw_history():
    plugin = LycheeDuplexPlugin(lambda *args: None)
    first = _plan(plugin)
    _commit(plugin, first)
    history = plugin.histories["session"]
    _record_eos(history)
    _record_through(history, 9)
    codec = CodecStreamState(last_tick=9, response_number=3, execution_epoch=0, awaiting_final=True)
    plugin.codec_streams.states["owner"] = codec
    plugin.data_plane._text_number["owner"] = 3
    plugin.data_plane._published_completed["owner"] = 2
    plugin.data_plane._audio_seq[("owner", "response3", 0)] = 7
    channels = deepcopy((history.text, history.speech, history.control, history.raw_speech))
    plan = _plan(plugin, seq=1)
    assert plan.prompt["model_intermediate_buffer"]["meta"]["replace_streaming_prompt"] is True
    bridge = _bridge(plan)
    assert bridge["lychee_kv_rebuild"] == {"reason": "natural_speech_eos", "eos_tick": 8, "frontier_tick": 9}
    assert "lychee_audio_delta" not in bridge
    assert bridge["lychee_history"]["force_listen_at_frontier"] is False
    assert bridge["lychee_history"]["execution_epoch"] == 0
    assert plan.prompt["prompt_token_ids"] == history.text
    assert plan.sampling_params.max_tokens == 10
    assert history.pending_eos_rebuild_tick == 8
    retry = _plan(plugin, seq=1)
    assert len(history.audio_windows) == 2
    assert _bridge(retry)["lychee_kv_rebuild"] == bridge["lychee_kv_rebuild"]
    _commit(plugin, retry)
    assert history.pending_eos_rebuild_tick is None
    assert (history.text, history.speech, history.control, history.raw_speech) == channels
    assert history.bound_request_id == "owner" and history.request_ids == {"owner"}
    assert plugin.codec_streams.states["owner"] is codec and codec.awaiting_final
    assert plugin.data_plane._text_number["owner"] == 3
    assert plugin.data_plane._published_completed["owner"] == 2
    assert plugin.data_plane._audio_seq[("owner", "response3", 0)] == 7
    _record_through(history, 19)
    resident = _plan(plugin, seq=2)
    assert "lychee_audio_delta" in _bridge(resident)
    assert "meta" not in resident.prompt["model_intermediate_buffer"]


def test_uncommitted_eos_tail_keeps_rebuild_budget_and_duplicate_eos_cannot_rearm():
    plugin = LycheeDuplexPlugin(lambda *args: None)
    first = _plan(plugin)
    _commit(plugin, first)
    history = plugin.histories["session"]
    _record_eos(history)
    plan = _plan(plugin, seq=1)
    assert plan.sampling_params.max_tokens == 11
    assert _bridge(plan)["lychee_history"]["logical_ticks"][-1] == 8
    _commit(plugin, plan)
    _record_eos(history)
    assert history.pending_eos_rebuild_tick is None


def _worker_history():
    history = _history()
    n = 20
    history.update(
        text_input_ids=[101, 102] + [2] * n,
        speech_input_ids=[None, None] + [3] * n,
        control_input_ids=[None, None] + [4] * n,
        logical_ticks=[-1, -1] + list(range(n)),
        audio_windows=[dict(seq=i + 1, start_tick=i * 10, payload=i + 1) for i in range(3)],
    )
    history["speech_input_ids"][10] = 13
    history["control_input_ids"][3] = 6
    history["control_input_ids"][4:10] = [9] * 6
    history["control_input_ids"][11] = 7
    for channel in ("text", "speech", "control"):
        history[f"raw_{channel}_output_ids"] = list(history[f"{channel}_input_ids"])
    return history


def _replacement(history):
    request = _request(history)
    request.model_intermediate_buffer["meta"] = {"replace_streaming_prompt": True}
    request.model_intermediate_buffer["duplex"]["lychee_kv_rebuild"] = {
        "reason": "natural_speech_eos",
        "eos_tick": 8,
        "frontier_tick": 19,
    }
    return request


def test_same_id_replacement_replays_from_zero_and_restores_teacher_rng_without_touching_other_owner(monkeypatch):
    state = _session_state(monkeypatch)
    state.add_request(0, _request(_history()))
    state.add_request(1, _session_request(req_id="other"))
    state._last_text_tokens[0] = 999
    state._committed_ticks[0] = 19
    generator = state._speech_generators["req-0"]
    other = {name: getattr(state, name)[1].clone() for name in state._request_row_tensors}
    history = _worker_history()
    state.on_request_rebind("req-0", 0)
    state.remove_request("req-0")
    state.add_request(2, _replacement(deepcopy(history)))
    assert "req-0" not in state._pending_append_text
    assert state._last_text_tokens[2] == 2 and state._committed_ticks[2] == -1
    assert state._execution_epochs[2] == 5
    assert state._speech_generators["req-0"] is generator
    fresh = _session_state(monkeypatch)
    fresh.add_request(2, _request(deepcopy(history)))
    merges = _expanded_logits(state)
    _expanded_logits(fresh)
    for computed, count in [(0, 4), (4, 7), (11, 11)]:
        active = computed + count == 22
        inputs, actual = _step(state, slot=2, computed=computed, count=count, prefill=22, active=active)
        expected_inputs, expected = _step(fresh, slot=2, computed=computed, count=count, prefill=22, active=active)
        assert inputs["input_ids"].tolist() == history["text_input_ids"][computed : computed + count]
        assert inputs["audio_embeddings"].tolist() == expected_inputs["audio_embeddings"].tolist()
        assert {k: v.tolist() for k, v in actual.items()} == {k: v.tolist() for k, v in expected.items()}
        if not active:
            assert merges[-1][-1] == history["text_input_ids"][computed + count]
    assert torch.rand((), generator=generator) == torch.rand((), generator=fresh._speech_generators["req-0"])
    assert len(state._history_audio_cache["req-0"]) <= state._audio_feature_owner_limit
    for name, value in other.items():
        assert torch.equal(getattr(state, name)[1], value), name


@pytest.mark.parametrize(
    "change", ["no_history", "delta", "wrong_reason", "wrong_eos", "wrong_frontier", "force_listen"]
)
def test_replacement_rejects_incomplete_or_cancelled_history(monkeypatch, change):
    state = _session_state(monkeypatch)
    request = _replacement(_worker_history())
    bridge = request.model_intermediate_buffer["duplex"]
    if change == "no_history":
        bridge.pop("lychee_history")
    elif change == "delta":
        bridge["lychee_audio_delta"] = {"version": 1}
    elif change == "wrong_reason":
        bridge["lychee_kv_rebuild"]["reason"] = "cancel"
    elif change == "wrong_eos":
        bridge["lychee_kv_rebuild"]["eos_tick"] = 7
    elif change == "wrong_frontier":
        bridge["lychee_kv_rebuild"]["frontier_tick"] = 18
    else:
        bridge["lychee_history"]["force_listen_at_frontier"] = True
    with pytest.raises(ValueError, match="Lychee"):
        state.add_request(0, request)
