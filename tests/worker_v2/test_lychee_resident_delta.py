# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resident audio deltas retain channel/RNG ownership without copying history."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tests.worker_v2.test_lychee_history_recovery import _expanded_logits, _history, _request, _step
from tests.worker_v2.test_lychee_model_state import _batch, _session_request, _session_state
from vllm_omni.worker_v2.model_states.lychee_history import LycheeResidentHistory

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _bootstrap(state):
    history = _history()
    history["audio_windows"] = history["audio_windows"][:1]
    request = _request(history)
    request.model_intermediate_buffer["duplex"]["epoch"] = 5
    state.add_request(0, request)
    _expanded_logits(state)
    _step(state, computed=0, count=3, prefill=3, active=True)
    for computed in range(3, 11):
        _step(state, computed=computed, count=1, prefill=3, active=True)
    return history


def _delta(seq, *, req_id="req-0", epoch=5):
    request = _session_request(seq=seq, req_id=req_id)
    start = (seq - 1) * 10
    request.model_intermediate_buffer["duplex"].update(
        epoch=epoch,
        payload=dict(
            window_value=seq,
            lychee_audio_ledger=dict(audio_window_seq=seq, consumable_tick_start=start, consumable_tick_end=start + 10),
        ),
        lychee_audio_delta=dict(
            version=1,
            kind="resident_append",
            request_id=req_id,
            session_epoch=epoch,
            execution_epoch=epoch,
            op_seq=seq,
            audio_window_seq=seq,
            previous_audio_window_seq=seq - 1,
            start_tick=start,
            window_ticks=10,
        ),
    )
    return request


def _rebind(state, request, *, old_slot=0, new_slot=2):
    state.on_request_rebind(request.req_id, old_slot)
    state.remove_request(request.req_id)
    state.add_request(new_slot, request)


def _encode(state):
    def encode(payload):
        value = payload["window_value"] if isinstance(payload, dict) else payload
        return torch.arange(20, dtype=torch.float32).reshape(10, 2) + value * 100

    state._encode_audio_steps = encode


def test_resident_delta_matches_full_snapshot_channels_audio_rng_and_bounded_ring(monkeypatch):
    delta_state, full_state = _session_state(monkeypatch), _session_state(monkeypatch)
    for state in (delta_state, full_state):
        # The tiny fixture's original eight codes can exhaust its n-gram
        # support in a long session; use all available synthetic speech logits.
        state.model.config.stoken_token_ids_max = 40
        _encode(state)
    _bootstrap(delta_state)
    full_history = _bootstrap(full_state)
    delta_generator = delta_state._speech_generators["req-0"]
    for seq in range(2, 31):
        _rebind(delta_state, _delta(seq), old_slot=0 if seq == 2 else 2)
        full_history["audio_windows"].append(dict(seq=seq, start_tick=(seq - 1) * 10, payload=seq))
        full_request = _request(deepcopy(full_history))
        full_request.model_intermediate_buffer["duplex"].update(seq=seq, epoch=5)
        _rebind(full_state, full_request, old_slot=0 if seq == 2 else 2)
        computed = 2 + (seq - 1) * 10 - 1
        prefill = computed + 1
        for offset in range(10):
            inputs, actual = _step(
                delta_state, slot=2, computed=computed + offset, count=1, prefill=prefill, active=True
            )
            expected_inputs, expected = _step(
                full_state, slot=2, computed=computed + offset, count=1, prefill=prefill, active=True
            )
            assert inputs["audio_embeddings"].tolist() == expected_inputs["audio_embeddings"].tolist()
            for key in expected:
                assert actual[key].tolist() == expected[key].tolist(), key
        resident = delta_state._resident_histories["req-0"]
        assert isinstance(delta_state._prompt_histories["req-0"][1], LycheeResidentHistory)
        assert set(resident.audio_by_start) == {(seq - 2) * 10, (seq - 1) * 10}
        assert len(delta_state._history_audio_cache["req-0"]) <= delta_state._audio_feature_owner_limit
        assert delta_state._speech_generators["req-0"] is delta_generator
    assert torch.rand((), generator=delta_generator) == torch.rand((), generator=full_state._speech_generators["req-0"])
    delta_state.on_requests_finished({"req-0"})
    assert not delta_state._resident_histories and not delta_state._prompt_histories


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("kind", "unknown"),
        ("request_id", "another-owner"),
        ("session_epoch", 6),
        ("execution_epoch", 6),
        ("op_seq", 1),
        ("op_seq", 3),
        ("audio_window_seq", 3),
        ("previous_audio_window_seq", 0),
        ("start_tick", 9),
        ("window_ticks", 5),
    ],
)
def test_invalid_delta_rejects_before_audio_or_channel_consumption(monkeypatch, field, value):
    state = _session_state(monkeypatch)
    _bootstrap(state)
    request = _delta(2)
    request.model_intermediate_buffer["duplex"]["lychee_audio_delta"][field] = value
    _rebind(state, request)
    before = state._last_text_tokens[2].clone()
    with pytest.raises(ValueError, match="delta.*rebuild"):
        _step(state, slot=2, computed=11, count=1, prefill=12, active=True)
    assert torch.equal(state._last_text_tokens[2], before)
    assert state._resident_histories["req-0"].op_seq == 1


def test_same_packet_getter_is_idempotent_but_new_duplicate_update_is_rejected(monkeypatch):
    state = _session_state(monkeypatch)
    _bootstrap(state)
    request = _delta(2)
    _rebind(state, request)
    first = state._get_prompt_history(2, "req-0")
    assert state._get_prompt_history(2, "req-0") is first
    _step(state, slot=2, computed=11, count=1, prefill=12, active=True)
    _rebind(state, deepcopy(request), old_slot=2)
    with pytest.raises(ValueError, match="sequence/window mismatch"):
        state._get_prompt_history(2, "req-0")
    assert state._resident_histories["req-0"].op_seq == 2


@pytest.mark.parametrize("missing_binding", ["fresh-owner", "finished-owner", "different-owner"])
def test_delta_cannot_create_or_resurrect_an_owner(monkeypatch, missing_binding):
    state = _session_state(monkeypatch)
    if missing_binding != "fresh-owner":
        _bootstrap(state)
        state.remove_request("req-0")
        state.on_requests_finished({"req-0"})
    request = _delta(2, req_id="other" if missing_binding == "different-owner" else "req-0")
    state.add_request(2, request)
    with pytest.raises(ValueError, match="retained bootstrap owner"):
        _step(state, slot=2, req_id=request.req_id, computed=11, count=1, prefill=12, active=True)


def test_pending_append_audio_selection_uses_host_position_without_tensor_item(monkeypatch):
    state = _session_state(monkeypatch)
    _bootstrap(state)
    _encode(state)
    _rebind(state, _delta(2))
    batch = _batch(input_ids=[2], counts=[1], computed=[11], prefill_lens=[12])
    batch.idx_mapping[:] = 2
    batch.idx_mapping_np[:] = 2
    batch.num_tokens_after_padding = 1
    with monkeypatch.context() as patch:
        patch.setattr(
            torch.Tensor, "item", lambda self: (_ for _ in ()).throw(AssertionError("host tick read GPU scalar"))
        )
        inputs = state.prepare_inputs(batch, SimpleNamespace())
    assert inputs["audio_embeddings"].tolist() == [[118, 119]]


def _scalar_policy(state, logits, batch):
    result = logits.clone()
    config = state.model.config
    for row in range(batch.num_reqs):
        index = int(batch.idx_mapping_np[row])
        settings = state._runtime_sampling_config(index)
        forced = None
        if bool(state._control_modes[index] == 0):
            forced = config.text_pad_token_id
        elif bool(state._text_eos_seen[index]):
            forced = config.tts_pad_token_id
        elif int(state._text_generated_steps[index]) + 1 >= settings["text_max_tokens"]:
            forced = config.eos_token_id
        if forced is not None:
            keep = result[row, forced].clone()
            result[row].fill_(float("-inf"))
            result[row, forced] = keep
        else:
            result[row, config.eos_token_id] *= settings["end_speak_token_factor"]
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("eos", [0, 1])
def test_primary_policy_matches_scalar_opmath_with_mixed_slots_and_padded_mappings(monkeypatch, dtype, eos):
    state = _session_state(monkeypatch)
    state.model.config.eos_token_id = eos
    state.model.config.tts_pad_token_id = 5
    state._control_modes[:] = torch.tensor([1, 0, 2])
    state._text_eos_seen[:] = torch.tensor([False, True, True])
    state._text_generated_steps[:] = torch.tensor([2, 0, 4])
    for index, factor in enumerate((1.0007, 0.7, 2.3)):
        state.intermediate_buffer.buffers[index] = {
            "duplex": {"runtime_config": {"text_max_tokens": 7, "end_speak_token_factor": factor}}
        }
    batch = SimpleNamespace(num_reqs=3, idx_mapping_np=[2, 0, 1, 999], idx_mapping=torch.tensor([2, 0, 1, 999]))
    logits = torch.randn(3, 40, generator=torch.Generator().manual_seed(741)).to(dtype)
    expected = _scalar_policy(state, logits, batch)
    actual = state.constrain_primary_logits(logits, batch)
    assert torch.equal(actual, expected)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        state.constrain_primary_logits(logits, batch)
    assert sum(event.count for event in profile.key_averages() if event.key == "aten::item") == 1
    logits[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        state.constrain_primary_logits(logits, batch)


def test_rebuild_backlog_and_cancel_cutoff_survive_global_window_delta_with_reset_opseq(monkeypatch):
    state = _session_state(monkeypatch)
    _encode(state)
    history = _history()
    history["audio_windows"].append(dict(seq=3, start_tick=20, payload=3))
    history["audio_windows"][1]["discard_after_tick"] = 18
    history["force_listen_at_frontier"] = True
    request = _request(history)
    request.model_intermediate_buffer["duplex"]["epoch"] = 5
    state.add_request(0, request)
    _expanded_logits(state)
    _step(state, computed=0, count=3, prefill=3, active=True)
    for computed in range(3, 31):
        inputs, _ = _step(state, computed=computed, count=1, prefill=3, active=True)
        if computed == 21:
            assert inputs["audio_embeddings"].tolist() == [[25, 25]]
    delta = _delta(4)
    bridge = delta.model_intermediate_buffer["duplex"]
    bridge["seq"] = bridge["lychee_audio_delta"]["op_seq"] = 2
    _rebind(state, delta)
    inputs, output = _step(state, slot=2, computed=31, count=1, prefill=32, active=True)
    assert inputs["audio_embeddings"].tolist() == [[318, 319]]
    assert output["lychee_tick"].tolist() == [30]
    assert state._resident_histories["req-0"].op_seq == 2
    assert state._resident_histories["req-0"].audio_window_seq == 4
    assert set(state._resident_histories["req-0"].audio_by_start) == {20, 30}


@pytest.mark.parametrize(
    "damage", ["envelope-op", "envelope-epoch", "ledger-start", "ledger-end", "ledger-seq", "both"]
)
def test_delta_envelope_and_pcm_ledger_must_agree(monkeypatch, damage):
    state = _session_state(monkeypatch)
    _bootstrap(state)
    request = _delta(2)
    bridge = request.model_intermediate_buffer["duplex"]
    if damage == "envelope-op":
        bridge["seq"] = 3
    elif damage == "envelope-epoch":
        bridge["epoch"] = 6
    elif damage == "both":
        bridge["lychee_history"] = _history()
    else:
        key = {
            "ledger-start": "consumable_tick_start",
            "ledger-end": "consumable_tick_end",
            "ledger-seq": "audio_window_seq",
        }[damage]
        bridge["payload"]["lychee_audio_ledger"][key] += 1
    _rebind(state, request)
    with pytest.raises(ValueError):
        state._get_prompt_history(2, "req-0")
    assert state._resident_histories["req-0"].op_seq == 1
