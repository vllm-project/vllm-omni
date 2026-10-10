# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, request-owned whole-window features survive natural EOS AR rebuild."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.worker_v2.test_lychee_history_recovery import _expanded_logits, _history, _request, _step
from tests.worker_v2.test_lychee_model_state import _batch, _session_request, _session_state
from tests.worker_v2.test_lychee_natural_eos_rebuild import _replacement, _worker_history
from vllm_omni.worker_v2.model_states.lychee_history import LycheePromptHistory

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _window_history(count=3, *, epoch=5):
    payload = _history()
    payload["execution_epoch"] = epoch
    payload["audio_windows"] = [dict(seq=i + 1, start_tick=i * 10, payload=i + 1) for i in range(count)]
    return payload


def _bind(state, payload=None, *, req_id="req-0", slot=0):
    state.add_request(slot, _request(_window_history() if payload is None else payload, req_id))
    return state._get_prompt_history(slot, req_id)


def _gather(state, history, ticks, *, req_id="req-0", slot=0):
    return state._history_audio_rows(history, req_id, tuple(ticks), slot)


def _oracle(ticks):
    return [[100 * (tick // 10 + 1) + 2 * (tick % 10) + j for j in range(2)] for tick in ticks]


def _suspend(state, req_id="req-0", slot=0):
    state.on_request_rebind(req_id, slot)
    state.remove_request(req_id)


def _pool_entries(state):
    return sum(len(cache) for cache in state._history_audio_cache.values())


def test_reuses_whole_windows_across_chunk_boundaries_without_preencoding_future_pcm(monkeypatch):
    state = _session_state(monkeypatch)
    history = _bind(state, _window_history(4))
    for ticks in ((0, 1), (8, 9, 10), (11, 19), (0, 3, 10, 12)):
        assert _gather(state, history, ticks).tolist() == _oracle(ticks)
    assert state.encoded_windows == [1, 2]
    assert set(state._history_audio_cache["req-0"]) == {1, 2}
    assert _pool_entries(state) == 2


def test_natural_eos_rebuild_reuses_encoded_matrices_and_preserves_teacher_rng(monkeypatch):
    state = _session_state(monkeypatch)
    payload = _worker_history()
    history = _bind(state, payload)
    expected = _gather(state, history, range(20))
    original = {seq: entry.embeddings for seq, entry in state._history_audio_cache["req-0"].items()}
    generator = state._speech_generators["req-0"]
    _suspend(state)
    state.add_request(2, _replacement(deepcopy(payload)))
    fresh = _session_state(monkeypatch)
    _bind(fresh, deepcopy(payload), slot=2)
    merges, fresh_merges = _expanded_logits(state), _expanded_logits(fresh)
    audio = []
    computed = 0
    for count in (3, 4, 5, 10):
        actual_inputs, actual_outputs = _step(
            state, slot=2, computed=computed, count=count, prefill=22, active=computed + count == 22
        )
        fresh_inputs, fresh_outputs = _step(
            fresh, slot=2, computed=computed, count=count, prefill=22, active=computed + count == 22
        )
        assert actual_inputs["audio_embeddings"].tolist() == fresh_inputs["audio_embeddings"].tolist()
        for key in actual_outputs:
            assert actual_outputs[key].tolist() == fresh_outputs[key].tolist(), key
        audio.extend(actual_inputs["audio_embeddings"].tolist())
        computed += count
    assert audio[2:] == expected.tolist()
    assert merges == fresh_merges
    assert state.encoded_windows == [1, 2]
    assert state._speech_generators["req-0"] is generator
    for seq, matrix in original.items():
        assert state._history_audio_cache["req-0"][seq].embeddings is matrix
    assert torch.rand((), generator=generator) == torch.rand((), generator=fresh._speech_generators["req-0"])


def test_per_owner_lru_eviction_reencodes_original_pcm_without_changing_rows_or_cursors(monkeypatch):
    state = _session_state(monkeypatch, max_model_len=20)
    history = _bind(state)
    before = {name: getattr(state, name)[0].clone() for name in state._request_row_tensors}
    assert state._audio_feature_owner_limit == 2
    for tick in (0, 10, 0, 20):
        assert _gather(state, history, (tick,)).tolist() == _oracle((tick,))
    assert set(state._history_audio_cache["req-0"]) == {1, 3}
    assert state.encoded_windows == [1, 2, 3]
    assert _gather(state, history, (10,)).tolist() == _oracle((10,))
    assert state.encoded_windows == [1, 2, 3, 2]
    assert set(state._history_audio_cache["req-0"]) == {2, 3}
    for name, value in before.items():
        if name != "_audio_window_seqs":
            assert torch.equal(getattr(state, name)[0], value), name
    assert _pool_entries(state) <= state._audio_feature_owner_limit


def test_aggregate_cap_includes_suspended_owners_and_evicts_only_cold_features(monkeypatch):
    state = _session_state(monkeypatch, max_model_len=10, max_num_reqs=1)
    payload = _window_history(1)
    first = _bind(state, payload, req_id="first")
    assert _gather(state, first, (0,), req_id="first").tolist() == _oracle((0,))
    _suspend(state, "first")
    second = _bind(state, payload, req_id="second")
    assert _gather(state, second, (0,), req_id="second").tolist() == _oracle((0,))
    assert state._audio_feature_total_limit == 1 and _pool_entries(state) == 1
    assert "first" not in state._history_audio_cache
    assert "first" in state._rebind_state and "first" in state._speech_generators
    _suspend(state, "second")
    state.add_request(0, _request(deepcopy(payload), "first"))
    first = state._get_prompt_history(0, "first")
    assert _gather(state, first, (0,), req_id="first").tolist() == _oracle((0,))
    assert state.encoded_windows == [1, 1, 1]
    assert _pool_entries(state) == len(state._audio_feature_lru) == 1
    state.on_requests_finished({"first", "second"})
    assert not state._history_audio_cache and not state._audio_feature_lru and not state._audio_feature_owners


def test_cancel_cutoff_masks_gather_without_poisoning_complete_cached_matrix(monkeypatch):
    state = _session_state(monkeypatch)
    history = _bind(state, _window_history(1))
    original = _gather(state, history, range(10))
    matrix = state._history_audio_cache["req-0"][1].embeddings
    payload = _window_history(1)
    payload["audio_windows"][0]["discard_after_tick"] = 4
    cancelled = LycheePromptHistory.from_payload(payload, window_ticks=10)
    gathered = _gather(state, cancelled, range(10))
    assert gathered[:5].tolist() == original[:5].tolist()
    assert gathered[5:].tolist() == [[25, 25]] * 5
    assert state._history_audio_cache["req-0"][1].embeddings is matrix
    assert matrix.tolist() == original.tolist()
    assert _gather(state, history, range(10)).tolist() == original.tolist()
    assert state.encoded_windows == [1]


def test_fully_cancelled_future_window_does_not_encode_or_insert(monkeypatch):
    state = _session_state(monkeypatch)
    payload = _window_history(1)
    payload["audio_windows"][0]["discard_after_tick"] = -1
    history = _bind(state, payload)
    assert _gather(state, history, range(10)).tolist() == [[25, 25]] * 10
    assert not state.encoded_windows and not state._history_audio_cache and not state._audio_feature_lru


def test_new_execution_epoch_clears_same_id_encoded_features(monkeypatch):
    state = _session_state(monkeypatch)
    history = _bind(state, _window_history(1))
    _gather(state, history, (0,))
    old = state._history_audio_cache["req-0"][1].embeddings
    _suspend(state)
    payload = _window_history(1, epoch=6)
    payload["audio_windows"][0]["payload"] = 9
    state.add_request(2, _request(payload))
    new_history = state._get_prompt_history(2, "req-0")
    assert not state._history_audio_cache and not state._audio_feature_lru
    assert state._audio_feature_owners == {"req-0": 6}
    assert _gather(state, new_history, (0,), slot=2).tolist() == [[900, 901]]
    assert state.encoded_windows == [1, 9]
    assert state._history_audio_cache["req-0"][1].embeddings is not old


def test_new_owner_slot_reuse_and_late_old_finish_do_not_share_or_remove_new_features(monkeypatch):
    state = _session_state(monkeypatch)
    old = _bind(state, _window_history(1), req_id="old")
    _gather(state, old, (0,), req_id="old")
    _suspend(state, "old")
    payload = _window_history(1)
    payload["audio_windows"][0]["payload"] = 7
    new = _bind(state, payload, req_id="new")
    assert _gather(state, new, (0,), req_id="new").tolist() == [[700, 701]]
    new_matrix = state._history_audio_cache["new"][1].embeddings
    state.on_requests_finished({"old"})
    assert "old" not in state._history_audio_cache and "old" not in state._audio_feature_owners
    assert state._history_audio_cache["new"][1].embeddings is new_matrix
    state.remove_request("new")
    assert not state._history_audio_cache and not state._audio_feature_owners and not state._audio_feature_lru


@pytest.mark.parametrize("field,value", [("seq", 2), ("start_tick", 10)])
def test_cached_sequence_and_start_must_match_before_reuse(monkeypatch, field, value):
    state = _session_state(monkeypatch)
    history = _bind(state)
    _gather(state, history, (0,))
    cache = state._history_audio_cache["req-0"]
    cache[1] = replace(cache[1], **{field: value})
    with pytest.raises(ValueError, match="sequence/start mismatch"):
        _gather(state, history, (0,))
    assert state.encoded_windows == [1] and _pool_entries(state) == 1


def test_dummy_slot_cannot_read_or_insert_live_owner_features(monkeypatch):
    state = _session_state(monkeypatch)
    history = _bind(state)
    assert _gather(state, history, (0,), req_id="dummy").tolist() == [[0, 0]]
    assert not state.encoded_windows and not state._history_audio_cache and not state._audio_feature_lru


def test_graph_capture_can_read_prepared_features_but_never_touch_lru_or_insert(monkeypatch):
    state = _session_state(monkeypatch)
    history = _bind(state)
    _gather(state, history, (0, 10))
    order = tuple(state._audio_feature_lru)
    monkeypatch.setattr(state, "_audio_feature_cache_mutable", lambda: False)
    assert _gather(state, history, (0, 1)).tolist() == _oracle((0, 1))
    assert tuple(state._audio_feature_lru) == order
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        _gather(state, history, (20,))
    assert state.encoded_windows == [1, 2]
    assert _pool_entries(state) == 2


@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_invalid_encoder_output_cannot_become_retained_feature(monkeypatch, change):
    state = _session_state(monkeypatch)
    history = _bind(state)
    state._encode_audio_steps = lambda payload: (
        torch.zeros(5, 2) if change == "shape" else torch.zeros(10, 2, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="shape/dtype/device"):
        _gather(state, history, (0,))
    assert not state._history_audio_cache and not state._audio_feature_lru


def test_budget_counts_complete_configured_matrices_and_context_ceiling(monkeypatch):
    state = _session_state(monkeypatch, max_model_len=21, max_num_reqs=4)
    assert state._audio_feature_owner_limit == 3
    assert state._audio_feature_total_limit == 12
    assert state._audio_feature_window_bytes == 10 * 2 * 4
    history = _bind(state, _window_history(3))
    _gather(state, history, (0, 10, 20))
    actual_bytes = sum(
        entry.embeddings.numel() * entry.embeddings.element_size()
        for cache in state._history_audio_cache.values()
        for entry in cache.values()
    )
    assert actual_bytes == 3 * state._audio_feature_window_bytes
    assert actual_bytes <= state._audio_feature_total_limit * state._audio_feature_window_bytes


def test_retained_matrix_owns_only_its_budgeted_storage_and_survives_encoder_buffer_reuse(monkeypatch):
    state = _session_state(monkeypatch)
    history = _bind(state)
    backing = torch.arange(200, dtype=torch.float32).reshape(100, 2)

    def encode(payload):
        backing.fill_(payload * 100)
        return backing[::10]

    state._encode_audio_steps = encode
    assert _gather(state, history, (0,)).tolist() == [[100, 100]]
    first = state._history_audio_cache["req-0"][1].embeddings
    assert first.untyped_storage().nbytes() == state._audio_feature_window_bytes
    assert _gather(state, history, (10,)).tolist() == [[200, 200]]
    assert first.tolist() == [[100, 100]] * 10
    assert _gather(state, history, (0,)).tolist() == [[100, 100]]


@pytest.mark.parametrize("bootstrap", [True, False])
@pytest.mark.parametrize("poisoned", [True, False])
def test_actual_dummy_batch_prepare_cannot_register_or_encode_live_slot_owner(monkeypatch, bootstrap, poisoned):
    state = _session_state(monkeypatch)
    if bootstrap:
        _bind(state)
    else:
        state.add_request(0, _session_request())
    if poisoned:
        state._poisoned_rows.add(0)
        state._poisoned[0] = True
    tensors = {name: getattr(state, name)[0].clone() for name in state._request_row_tensors}
    before = {
        name: set(getattr(state, name))
        for name in ("_prompt_histories", "_resident_histories", "_audio_feature_owners")
    }
    batch = _batch(input_ids=[2], counts=[1], computed=[0], prefill_lens=[0])
    batch.req_ids = ["req_0_dummy_uuid"]
    batch.num_tokens_after_padding = 1
    inputs = state.prepare_inputs(batch, SimpleNamespace())
    assert "audio_embeddings" not in inputs
    assert state._get_prompt_history(0, batch.req_ids[0]) is None
    assert not state.encoded_windows and not state._audio_window_cache and not state._history_audio_cache
    assert state._poisoned_rows == ({0} if poisoned else set())
    for name, keys in before.items():
        assert set(getattr(state, name)) == keys, name
    for name, tensor in tensors.items():
        assert torch.equal(getattr(state, name)[0], tensor), name


def test_invalid_new_epoch_bootstrap_does_not_retire_current_feature_owner(monkeypatch):
    state = _session_state(monkeypatch)
    history = _bind(state, _window_history(1))
    _gather(state, history, (0,))
    matrix = state._history_audio_cache["req-0"][1].embeddings
    buffer = state.intermediate_buffer.buffers[0]
    buffer["duplex"]["lychee_history"] = _window_history(1, epoch=6)
    buffer["duplex"]["epoch"] = 5
    with pytest.raises(ValueError, match="invalid owner epoch"):
        state._get_prompt_history(0, "req-0")
    assert state._audio_feature_owners["req-0"] == 5
    assert state._history_audio_cache["req-0"][1].embeddings is matrix
    assert state.encoded_windows == [1]
