# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.worker_v2.test_lychee_model_state import _batch, _req_states, _session_request, _session_state
from vllm_omni.worker_v2.model_states.lychee_history import LycheePromptHistory
from vllm_omni.worker_v2.model_states.lychee_model_state import LycheePoisonedRequestError
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _history():
    return dict(
        version=1,
        text_input_ids=[101, 102, 2],
        speech_input_ids=[None, None, 3],
        control_input_ids=[None, None, 4],
        logical_ticks=[-1, -1, 0],
        execution_epoch=5,
        audio_windows=[dict(seq=1, start_tick=0, payload=1), dict(seq=2, start_tick=10, payload=2)],
    )


def _request(history, req_id="req-0"):
    request = _session_request(req_id=req_id)
    request.model_intermediate_buffer["duplex"]["lychee_history"] = history
    return request


def _step(state, *, slot=0, req_id="req-0", computed, count, prefill, active):
    batch = _batch(input_ids=[999] * count, counts=[count], computed=[computed], prefill_lens=[prefill])
    batch.idx_mapping[:] = slot
    batch.idx_mapping_np[:] = slot
    batch.req_ids[:] = [req_id]
    batch.num_tokens_after_padding = count
    batch.num_draft_tokens = 0
    batch.positions = torch.arange(computed, computed + count)
    batch.logits_indices = torch.tensor([count - 1])
    inputs = state.prepare_inputs(batch, SimpleNamespace())
    sampled = state.constrain_primary_sample(
        sampled_token_ids=torch.tensor([[1000 + computed + count - 1]]),
        num_sampled=torch.tensor([int(active)]),
        input_batch=batch,
    )
    payload = state.continue_after_primary_sample(
        sampled_token_ids=sampled,
        num_sampled=torch.tensor([int(active)]),
        multimodal_outputs=dict(
            lychee_stoken_hidden=torch.zeros(count, 2), lychee_control_hidden=torch.zeros(count, 2)
        ),
        input_batch=batch,
        req_states=_req_states([[999] * 100] * 3),
    )
    return inputs, payload


def _expanded_logits(state):
    seen = []

    def continuation(**kwargs):
        seen.append(kwargs["sampled_text_token_ids"].tolist())
        return None, torch.arange(40, dtype=torch.float32)[None, :].expand(len(kwargs["positions"]), -1) / 40

    state.model.continue_after_primary_sample = continuation
    return seen


def test_initial_system_prefill_aligns_singleton_at_frontier_and_teacher_merge(monkeypatch):
    state = _session_state(monkeypatch)
    history = _history()
    state.add_request(0, _request(history))
    merges = _expanded_logits(state)
    inputs, output = _step(state, computed=0, count=3, prefill=3, active=True)
    assert inputs["input_ids"].tolist() == [101, 102, 2]
    assert inputs["stoken_input_mask"].tolist() == [False, False, True]
    assert inputs["control_input_mask"].tolist() == [False, False, True]
    assert inputs["audio_embeddings"].tolist() == [[0, 0], [0, 0], [100, 101]]
    assert merges == [[102, 2, 2]]
    assert output["lychee_tick"].tolist() == [-1, -1, 1]
    assert output["lychee_model_position"].tolist() == [-1, -1, 2]
    assert state._execution_epochs[0].item() == 5


def test_four_channel_chunked_rebuild_matches_uninterrupted_frontier_and_rng(monkeypatch):
    live = _session_state(monkeypatch)
    history = _history()
    live.add_request(0, _request(history))
    _expanded_logits(live)
    for tick in range(1, 14):
        _, out = _step(live, computed=0 if tick == 1 else tick + 1, count=3 if tick == 1 else 1, prefill=3, active=True)
        history["text_input_ids"].append(int(out["lychee_text_token_ids"][-1]))
        history["speech_input_ids"].append(int(out["lychee_speech_token_ids"][-1]))
        history["control_input_ids"].append(int(out["lychee_control_token_ids"][-1]))
        history["logical_ticks"].append(tick)
    recovered = _session_state(monkeypatch)
    recovery_history = {**history, "execution_epoch": 8}
    recovered.add_request(2, _request(recovery_history, "rebuilt"))
    merges = _expanded_logits(recovered)
    computed = 0
    for count in [2, 3, 4, 7]:
        inputs, actual = _step(
            recovered,
            slot=2,
            req_id="rebuilt",
            computed=computed,
            count=count,
            prefill=16,
            active=computed + count == 16,
        )
        assert inputs["input_ids"].tolist() == history["text_input_ids"][computed : computed + count]
        if computed + count < 16:
            assert actual["lychee_text_token_ids"].tolist() == [-1] * count
            assert merges[-1][-1] == history["text_input_ids"][computed + count]
        computed += count
    _, expected = _step(live, computed=15, count=1, prefill=3, active=True)
    for key in (
        "lychee_text_token_ids",
        "lychee_speech_token_ids",
        "lychee_control_token_ids",
        "lychee_tick",
        "lychee_audio_window_seq",
    ):
        assert actual[key][-1].item() == expected[key][-1].item()
    assert actual["lychee_tick"][-1].item() == 14
    assert actual["lychee_execution_epoch"][-1].item() == 8
    assert recovered._control_modes[2].item() == live._control_modes[0].item()
    assert recovered._speaking_steps[2].item() == live._speaking_steps[0].item()
    assert recovered._speech_history[2].tolist() == live._speech_history[0].tolist()
    # Compare future random draws, never serialized generator bytes.
    assert (
        torch.rand((), generator=recovered._speech_generators["rebuilt"]).item()
        == torch.rand((), generator=live._speech_generators["req-0"]).item()
    )
    assert len(recovered._history_audio_cache["rebuilt"]) <= recovered._audio_feature_owner_limit


@pytest.mark.parametrize(
    "change,message",
    [
        ({"speech_input_ids": [3]}, "align"),
        ({"logical_ticks": [-1, -1, 4]}, "contiguous"),
        ({"audio_windows": []}, "audio windows"),
        ({"execution_epoch": -1}, "epoch"),
        ({"logical_ticks": [-1, 0, 1], "speech_input_ids": [3, None, None]}, "missing committed"),
    ],
)
def test_incomplete_history_fails_before_using_native_kv(change, message):
    with pytest.raises(ValueError, match=message):
        LycheePromptHistory.from_payload(_history() | change, window_ticks=10)


def _runner(state):
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.model_state = state
    runner.main_stream = Mock()
    runner.output_copy_stream = Mock()
    runner.eplb = Mock()
    runner.execute_model_state = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["req-0"]))
    return runner


@pytest.mark.parametrize("phase", ["main", "sample", "postprocess", "merge", "snapshot"])
def test_failed_transaction_aborts_touched_kv_and_allows_fresh_binding(monkeypatch, phase):
    state = _session_state(monkeypatch)
    state.add_request(0, _request(_history()))
    state.add_request(1, _session_request(req_id="unrelated"))
    runner = _runner(state)
    failure = RuntimeError(f"injected {phase} failure")
    if phase == "main":
        monkeypatch.setattr(OmniGPUModelRunner, "execute_model", Mock(side_effect=failure))
        runner._handle_kv_transfer_pre = Mock()
        result = runner.execute_model(SimpleNamespace(num_scheduled_tokens={"req-0": 1}))
    else:
        runner._sample_tokens = Mock(side_effect=failure)
        result = runner.sample_tokens(None)
    assert result.sampled_token_ids == [[]]
    assert "rebuild required" in result.request_errors["req-0"]
    assert state._poisoned_rows == {0}
    assert not state._poisoned[1]
    runner.main_stream.synchronize.assert_called_once()
    runner.output_copy_stream.synchronize.assert_called_once()
    assert runner.execute_model_state is None
    with pytest.raises(LycheePoisonedRequestError):
        _step(state, computed=0, count=3, prefill=3, active=True)
    state.on_requests_finished({"req-0"})
    state.remove_request("req-0")
    state.add_request(0, _request(_history() | {"execution_epoch": 9}, "new-id"))
    _expanded_logits(state)
    _, output = _step(state, req_id="new-id", computed=0, count=3, prefill=3, active=True)
    assert output["lychee_tick"][-1].item() == 1
    assert output["lychee_execution_epoch"][-1].item() == 9
    assert state._poisoned_rows == set()


def test_batched_fault_conservatively_aborts_every_touched_request(monkeypatch):
    state = _session_state(monkeypatch)
    for index, req_id in enumerate(["req-0", "req-1", "parked"]):
        state.add_request(index, _session_request(req_id=req_id))
    runner = _runner(state)
    output = runner._abort_failed_transaction(["req-0", "req-1"], RuntimeError("half merge"), "merge")
    assert set(output.request_errors) == {"req-0", "req-1"}
    assert state._poisoned_rows == {0, 1}
    assert not state._poisoned[2]


def test_device_context_fault_remains_engine_fatal(monkeypatch):
    state = _session_state(monkeypatch)
    state.add_request(0, _session_request())
    runner = _runner(state)
    runner._sample_tokens = Mock(side_effect=RuntimeError("CUDA error: an illegal memory access"))
    with pytest.raises(RuntimeError, match="illegal memory"):
        runner.sample_tokens(None)
    runner.main_stream.synchronize.assert_not_called()


@pytest.mark.parametrize("eos_tick,tail_end,detect", [(1, 9, True), (7, 9, False), (9, 19, True)])
def test_speech_eos_enters_listen_immediately_and_finishes_reference_padding(monkeypatch, eos_tick, tail_end, detect):
    state = _session_state(monkeypatch)
    state.add_request(0, _session_request())
    _expanded_logits(state)
    state._control_modes[0] = 1
    state._speaking_steps[0] = state.model.config.stoken_delay_num + state.model.config.stoken_max_tokens
    state._control_ticks[0] = eos_tick
    state._committed_ticks[0] = eos_tick - 1
    # A deterministic maximum-length EOS is emitted as raw evidence.
    _, eos = _step(state, computed=eos_tick, count=1, prefill=1, active=True)
    assert eos["lychee_speech_token_ids"][-1].item() == state.model.config.tts_end_token_id
    assert eos["lychee_mode_after"][-1].item() == 0
    assert eos["lychee_tail_padding_until"][-1].item() == tail_end
    assert state._speaking_steps[0].item() == -1
    for tick in range(eos_tick + 1, tail_end + 1):
        if tick % 10 == 0:
            state.intermediate_buffer.buffers[0]["duplex"].update(seq=tick // 10 + 1, payload=tick // 10 + 1)
        _, payload = _step(state, computed=tick, count=1, prefill=1, active=True)
        expected_control = state.model.config.sleep_token_id
        if tick == tail_end:
            expected_control = state.model.config.start_listening_token_id
        elif tick == tail_end - 1 and detect:
            expected_control = state.model.config.detect_token_id
        assert payload["lychee_text_token_ids"][-1].item() == state.model.config.text_pad_token_id
        assert payload["lychee_speech_token_ids"][-1].item() == state.model.config.stoken_pad_token_id
        assert payload["lychee_control_token_ids"][-1].item() == expected_control


def test_rebuild_uses_raw_eos_even_when_effective_speech_tail_is_pad(monkeypatch):
    state = _session_state(monkeypatch)
    history = _history()
    history.update(
        text_input_ids=[101, 102, 2, 2],
        speech_input_ids=[None, None, 3, 3],
        control_input_ids=[None, None, 4, 4],
        logical_ticks=[-1, -1, 0, 1],
        raw_text_output_ids=[None, None, None, 2],
        raw_speech_output_ids=[None, None, None, 13],
        raw_control_output_ids=[None, None, None, 4],
    )
    state.add_request(0, _request(history))
    _expanded_logits(state)
    _, output = _step(state, computed=0, count=4, prefill=4, active=True)
    assert output["lychee_tick"][-1].item() == 2
    assert output["lychee_mode_after"][-1].item() == 0
    assert output["lychee_tail_padding_until"][-1].item() == 9
    assert output["lychee_speech_token_ids"][-1].item() == 3


def test_primary_text_eos_then_pad_keeps_speech_channel_alive_and_scoped(monkeypatch):
    state = _session_state(monkeypatch)
    state.model.config.eos_token_id = 23
    state.model.config.tts_pad_token_id = 24
    state.add_request(0, _session_request())
    state.add_request(1, _session_request(req_id="other"))
    state._control_modes[:2] = 1
    state._text_eos_seen[0] = True
    batch = _batch(input_ids=[1, 1], counts=[1, 1], computed=[2, 2], prefill_lens=[1, 1])
    logits = torch.ones(2, 40)
    logits[:, 25] = 20
    constrained = state.constrain_primary_logits(logits, batch)
    assert constrained[0].argmax().item() == 24
    assert constrained[1].argmax().item() == 25
    assert state._control_modes[:2].tolist() == [1, 1]


def test_primary_text_limit_and_eos_factor_are_request_local(monkeypatch):
    state = _session_state(monkeypatch)
    state.model.config.eos_token_id = 23
    state.model.config.tts_pad_token_id = 24
    state.add_request(0, _session_request())
    state.add_request(1, _session_request(req_id="other"))
    state._control_modes[:2] = 1
    state._text_generated_steps[0] = 2
    state.intermediate_buffer.buffers[0]["duplex"]["runtime_config"] = {"text_max_tokens": 3}
    state.intermediate_buffer.buffers[1]["duplex"]["runtime_config"] = {"end_speak_token_factor": 2.0}
    batch = _batch(input_ids=[1, 1], counts=[1, 1], computed=[2, 2], prefill_lens=[1, 1])
    logits = torch.ones(2, 40)
    logits[:, 25] = 1.5
    constrained = state.constrain_primary_logits(logits, batch)
    assert constrained[0].argmax().item() == 23
    assert constrained[1].argmax().item() == 23
    assert logits[0, 23].item() == 1


@pytest.mark.parametrize(
    "invalid", [{"start_listen_token_factor": float("nan")}, {"allowing_backchannel": 1}, {"text_max_tokens": 0}]
)
def test_invalid_scoped_runtime_sampling_params_fail_fast(monkeypatch, invalid):
    state = _session_state(monkeypatch)
    state.add_request(0, _session_request())
    state.intermediate_buffer.buffers[0]["duplex"]["runtime_config"] = invalid
    with pytest.raises(ValueError, match="Lychee"):
        state._runtime_sampling_config(0)


def _voice_history():
    history = _history()
    for tick in range(1, 14):
        history["logical_ticks"].append(tick)
        history["text_input_ids"].append(2 if tick <= 9 else 100 + tick)
        history["speech_input_ids"].append(3 if tick <= 9 else {10: 11, 11: 12, 12: 15, 13: 16}[tick])
        history["control_input_ids"].append(6 if tick == 9 else 5 if tick == 8 else 4)
    return history


def _mixed_step(state, *, slots, req_ids, computed, counts, prefill, sampled, active):
    batch = _batch(input_ids=[999] * sum(counts), counts=counts, computed=computed, prefill_lens=prefill)
    batch.idx_mapping = torch.tensor(slots)
    batch.idx_mapping_np[:] = slots
    batch.req_ids[:] = req_ids
    batch.num_tokens_after_padding = sum(counts)
    batch.num_draft_tokens = 0
    batch.positions = torch.cat([torch.arange(first, first + count) for first, count in zip(computed, counts)])
    batch.logits_indices = torch.tensor(batch.query_start_loc_np[1:] - 1)
    state.prepare_inputs(batch, SimpleNamespace())
    tokens = state.constrain_primary_sample(
        sampled_token_ids=torch.tensor(sampled)[:, None], num_sampled=torch.tensor(active), input_batch=batch
    )
    return state.continue_after_primary_sample(
        sampled_token_ids=tokens,
        num_sampled=torch.tensor(active),
        multimodal_outputs=dict(
            lychee_stoken_hidden=torch.zeros(sum(counts), 2), lychee_control_hidden=torch.zeros(sum(counts), 2)
        ),
        input_batch=batch,
        req_states=_req_states([[999] * 100] * 3),
    )


def test_mixed_chunked_prefill_decode_and_row_reorder_preserve_independent_rng(monkeypatch):
    batch_state = _session_state(monkeypatch)
    isolated_a = _session_state(monkeypatch)
    isolated_b = _session_state(monkeypatch)
    for state, slot, req_id, history in [
        (batch_state, 2, "A", _voice_history()),
        (batch_state, 0, "B", _history()),
        (isolated_a, 0, "A", _voice_history()),
        (isolated_b, 0, "B", _history()),
    ]:
        state.add_request(slot, _request(history, req_id))
        _expanded_logits(state)
    mixed = _mixed_step(
        batch_state,
        slots=[2, 0],
        req_ids=["A", "B"],
        computed=[0, 0],
        counts=[16, 1],
        prefill=[16, 3],
        sampled=[1015, 1000],
        active=[1, 0],
    )
    _, a_expected = _step(isolated_a, req_id="A", computed=0, count=16, prefill=16, active=True)
    for key in ("lychee_text_token_ids", "lychee_speech_token_ids", "lychee_control_token_ids", "lychee_tick"):
        assert mixed[key][15].item() == a_expected[key][-1].item()
        assert mixed[key][16].item() == -1
    # Next batch reorders requests while B finishes prefill and A decodes.
    mixed = _mixed_step(
        batch_state,
        slots=[0, 2],
        req_ids=["B", "A"],
        computed=[1, 16],
        counts=[2, 1],
        prefill=[3, 16],
        sampled=[1002, 1016],
        active=[1, 1],
    )
    _, b_expected = _step(isolated_b, req_id="B", computed=0, count=3, prefill=3, active=True)
    _, a_expected = _step(isolated_a, req_id="A", computed=16, count=1, prefill=16, active=True)
    for key in ("lychee_text_token_ids", "lychee_speech_token_ids", "lychee_control_token_ids", "lychee_tick"):
        assert mixed[key][1].item() == b_expected[key][-1].item()
        assert mixed[key][2].item() == a_expected[key][-1].item()
    for req_id, isolated in [("A", isolated_a), ("B", isolated_b)]:
        assert (
            torch.rand((), generator=batch_state._speech_generators[req_id]).item()
            == torch.rand((), generator=isolated._speech_generators[req_id]).item()
        )


def test_history_out_of_vocabulary_fails_before_unsafe_gpu_embedding_lookup():
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        LycheePromptHistory.from_payload(_history(), window_ticks=10, vocab_size=100)


def test_cancel_frontier_forces_listen_without_rewriting_historical_raw_merge(monkeypatch):
    state = _session_state(monkeypatch)
    history = _voice_history()
    history["force_listen_at_frontier"] = True
    history["raw_text_output_ids"] = [None, None, None] + history["text_input_ids"][3:]
    history["raw_speech_output_ids"] = [None, None, None] + history["speech_input_ids"][3:]
    history["raw_control_output_ids"] = [None, None, None] + history["control_input_ids"][3:]
    history["text_input_ids"][-1] = 2
    state.add_request(0, _request(history))
    merges = _expanded_logits(state)
    inputs, output = _step(state, computed=0, count=16, prefill=16, active=True)
    assert inputs["input_ids"][-1].item() == 2
    assert inputs["stoken_input_ids"][-1].item() == 3
    assert inputs["control_input_ids"][-1].item() == 4
    assert merges[-1][-2] == history["raw_text_output_ids"][-1]
    assert merges[-1][-1] == 2
    assert output["lychee_mode_after"][-1].item() == 0
    assert output["lychee_speech_token_ids"][-1].item() == 3
    assert state._speaking_steps[0].item() == -1
    assert state._text_generated_steps[0].item() == 0
    assert state._forced_listen_bindings == {"req-0"}
    advanced = torch.Generator().manual_seed(17)
    for _ in range(2):
        torch.multinomial(torch.ones(1, 9), 1, generator=advanced)
    assert (
        torch.rand((), generator=advanced).item() == torch.rand((), generator=state._speech_generators["req-0"]).item()
    )


def test_chunked_prefill_teacher_merge_uses_raw_eos_across_chunk_boundary(monkeypatch):
    state = _session_state(monkeypatch)
    history = _history()
    history.update(
        text_input_ids=[101, 102, 2, 2],
        speech_input_ids=[None, None, 3, 3],
        control_input_ids=[None, None, 4, 7],
        logical_ticks=[-1, -1, 0, 1],
        raw_text_output_ids=[None, None, None, 23],
        raw_speech_output_ids=[None, None, None, 13],
        raw_control_output_ids=[None, None, None, 7],
    )
    state.add_request(0, _request(history))
    merges = _expanded_logits(state)
    _, output = _step(state, computed=0, count=3, prefill=4, active=False)
    assert merges[-1][-1] == 23
    assert output["lychee_text_token_ids"].tolist() == [-1] * 3


def test_second_response_ngram_uses_prior_response_and_survives_rebuild(monkeypatch):
    state = _session_state(monkeypatch)
    state.model.config.stoken_do_sample = False
    history = _history()
    history["audio_windows"] = [dict(seq=index + 1, start_tick=index * 10, payload=index + 1) for index in range(4)]
    speech = {10: 11, 11: 12, 12: 14, 13: 15, 14: 16, 15: 17, 16: 13, 30: 11, 31: 12, 32: 14, 33: 15, 34: 16}
    for tick in range(1, 35):
        history["logical_ticks"].append(tick)
        history["text_input_ids"].append(100 + tick if 10 <= tick <= 16 or tick >= 30 else 2)
        history["speech_input_ids"].append(speech.get(tick, 3))
        history["control_input_ids"].append(6 if tick in (9, 29) else 7 if tick == 19 else 5 if tick % 10 == 8 else 4)
    state.add_request(0, _request(history))

    def logits(**kwargs):
        scores = torch.zeros(len(kwargs["positions"]), 40)
        scores[:, 17] = 10
        scores[:, 18] = 9
        return None, scores

    state.model.continue_after_primary_sample = logits
    _, output = _step(state, computed=0, count=37, prefill=37, active=True)
    # The first response already produced 14,15,16,17. The second response's
    # suffix 14,15,16 must forbid 17 even after EOS, listening, and a new SS.
    assert output["lychee_speech_token_ids"][-1].item() == 18
    assert state._speech_history_lengths[0].item() == 36
    assert state._speech_history[0, :35].tolist() == [
        value for value in history["speech_input_ids"] if value is not None
    ]
    assert state._speech_history[0, 35].item() == 18
    assert state._speaking_steps[0].item() == 6


def test_session_ngram_growth_preserves_parked_rebind_and_other_rows(monkeypatch):
    state = _session_state(monkeypatch)
    state.add_request(0, _session_request())
    state._speech_history[0, :5] = torch.tensor([3, 14, 15, 16, 17])
    state._speech_history_lengths[0] = 5
    state.on_request_rebind("req-0", 0)
    state.remove_request("req-0")
    state.add_request(1, _session_request(req_id="other"))
    state._ensure_speech_history_capacity(300)
    state._speech_history[1, 1:4] = torch.tensor([18, 19, 20])
    state._speech_history_lengths[1] = 4
    state.add_request(2, _session_request(seq=2))
    assert state._speech_history_lengths[2].item() == 5
    assert state._speech_history[2, :5].tolist() == [3, 14, 15, 16, 17]
    assert state._speech_history[1, :4].tolist() == [3, 18, 19, 20]
    assert state._speech_history[2, 102:].tolist() == [-1] * 198


def test_input_cancel_rebuild_preserves_committed_audio_and_pads_uncomputed_frontier(monkeypatch):
    original = _session_state(monkeypatch)
    canceled = _session_state(monkeypatch)
    history = _voice_history()
    original.add_request(0, _request(history))
    _expanded_logits(original)
    original_inputs, _ = _step(original, computed=0, count=16, prefill=16, active=True)
    canceled_history = {
        **history,
        "audio_windows": [dict(window) for window in history["audio_windows"]],
        "force_listen_at_frontier": True,
    }
    # Output frontier 13 came from input 12. Only inputs <=12 have committed
    # KV; input 13 is the next frontier and must lose canceled audio.
    canceled_history["audio_windows"][1]["discard_after_tick"] = 12
    canceled.add_request(0, _request(canceled_history))
    _expanded_logits(canceled)
    inputs, output = _step(canceled, computed=0, count=16, prefill=16, active=True)
    torch.testing.assert_close(
        inputs["audio_embeddings"][:-1], original_inputs["audio_embeddings"][:-1], rtol=0, atol=0
    )
    assert original_inputs["audio_embeddings"][-1].tolist() == [206, 207]
    assert inputs["audio_embeddings"][-1].tolist() == [25, 25]
    assert output["lychee_tick"][-1].item() == 14
    assert canceled.encoded_windows == [1, 2]
    next_inputs, _ = _step(canceled, computed=16, count=1, prefill=16, active=True)
    assert next_inputs["audio_embeddings"].tolist() == [[25, 25]]
    assert canceled._audio_window_seqs[0].item() == 2


def test_initial_input_cancel_does_not_encode_discarded_pcm(monkeypatch):
    state = _session_state(monkeypatch)
    history = _history()
    history["audio_windows"][0]["discard_after_tick"] = -1
    history["force_listen_at_frontier"] = True
    state.add_request(0, _request(history))
    _expanded_logits(state)
    inputs, _ = _step(state, computed=0, count=3, prefill=3, active=True)
    assert inputs["audio_embeddings"].tolist() == [[0, 0], [0, 0], [25, 25]]
    assert state.encoded_windows == []
    assert state._audio_window_seqs[0].item() == 1


@pytest.mark.parametrize("cutoff", [True, None, -2, 10])
def test_audio_cancel_cutoff_requires_an_absolute_input_tick_within_its_window(cutoff):
    history = _history()
    history["audio_windows"][0]["discard_after_tick"] = cutoff
    with pytest.raises(ValueError, match="discard_after_tick"):
        LycheePromptHistory.from_payload(history, window_ticks=10)


@pytest.mark.parametrize("host_lag", [0, 1, 9])
def test_actual_same_id_append_replaces_metadata_and_keeps_audio_clock_across_first_nine_then_ten(
    monkeypatch, host_lag
):
    from vllm_omni.engine.duplex.contracts import DuplexFence
    from vllm_omni.model_executor.models.lychee_fd.duplex.plugin import LycheeDuplexPlugin

    plugin = LycheeDuplexPlugin(lambda *args, **kwargs: None)
    fence = DuplexFence("clock-session", epoch=5)
    runtime = dict(
        lychee_system_token_ids=[101, 102],
        duplex_scheduler_token_id=2,
        lychee_speech_pad_token_id=3,
        lychee_sleep_token_id=4,
    )
    state = _session_state(monkeypatch)
    _expanded_logits(state)
    encoded = []

    def encode(payload):
        window = payload["window_value"]
        encoded.append(window)
        rows = state._audio_pad_rows(10)
        rows[0::2] = torch.arange(10).reshape(5, 2) + window * 100
        return rows

    state._encode_audio_steps = encode
    state._prepare_audio_embeddings = Mock(side_effect=AssertionError("native history unexpectedly fell back"))
    history = None
    computed = 0
    expected_inputs = {}
    pending_outputs = []
    scheduler_rows = []
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request, StreamingUpdate

    scheduler_request = Request("req-0", [101, 102, 2], SamplingParams(max_tokens=9), None, resumable=True)
    scheduler_stub = SimpleNamespace(log_stats=False, num_waiting_for_streaming_input=0)
    for seq, budget in ((1, 9), (2, 10), (3, 10)):
        params = SimpleNamespace(seed=17, max_tokens=10, min_tokens=10, ignore_eos=True)
        plan = plugin.plan_append(
            request_id="req-0",
            fence=fence,
            session_config={},
            runtime_config=runtime,
            seq=seq,
            turn_seq=seq,
            payload=dict(window_value=seq, lychee_audio_ledger={}),
            final=False,
            sampling_params=params,
        )
        plugin.commit_append_plan(request_id="req-0", fence=fence, plan=plan)
        bridge = plan.prompt["model_intermediate_buffer"]["duplex"]
        assert ("lychee_history" in bridge) == (seq == 1)
        assert ("lychee_audio_delta" in bridge) == (seq > 1)
        assert plan.sampling_params.max_tokens == budget
        assert plan.prompt["prompt_token_ids"] == ([101, 102, 2] if seq == 1 else [2])
        if seq > 1:
            state.on_request_rebind("req-0", 0)
            state.remove_request("req-0")
        request = _session_request(seq=seq)
        request.model_intermediate_buffer = plan.prompt["model_intermediate_buffer"]
        request.sampling_params = plan.sampling_params
        state.add_request(0, request)
        # This is a whole NEW intermediate buffer, just as native streaming
        # remove/add behaves; previous nested history cannot be inherited.
        assert state.intermediate_buffer.buffers[0]["duplex"]["seq"] == seq
        history = plugin.histories["clock-session"]
        if seq > 1:
            scheduler_request.num_computed_tokens = computed
            update = StreamingUpdate(None, [2], budget, float(seq), SamplingParams(max_tokens=budget))
            Scheduler._update_request_as_session(scheduler_stub, scheduler_request, update)
        prefill = scheduler_request.num_prompt_tokens
        scheduler_rows.append((computed, prefill, len(history.text)))
        # The real fixed-base extension leaves exactly one uncomputed model
        # position, even when the parent history is nine samples behind.
        if seq > 1:
            assert prefill == computed + 1
        for sample in range(budget):
            count = 3 if computed == 0 else 1
            inputs, output = _step(state, computed=computed, count=count, prefill=prefill, active=True)
            input_tick = computed + count - 1 - 2
            expected_inputs[input_tick] = inputs["audio_embeddings"][-1].tolist()
            pending_outputs.append(output)
            if len(pending_outputs) > host_lag:
                history.record_outputs(pending_outputs.pop(0))
            scheduler_request.append_output_token_ids([int(output["lychee_text_token_ids"][-1])])
            computed += count
    assert history is not None
    for output in pending_outputs:
        history.record_outputs(output)
    assert scheduler_rows == [(0, 3, 3), (11, 12, 12 - host_lag), (21, 22, 22 - host_lag)]
    assert history.frontier_tick == 29
    assert expected_inputs[0] == [100, 101]
    assert expected_inputs[9] == [25, 25]
    assert expected_inputs[10] == [200, 201]
    assert expected_inputs[19] == [25, 25]
    assert expected_inputs[20] == [300, 301]
    assert encoded == [1, 2, 3]
    state._prepare_audio_embeddings.assert_not_called()


def test_native_audio_ledger_without_full_history_fails_instead_of_shifting_clock(monkeypatch):
    state = _session_state(monkeypatch)
    request = _session_request(seq=2)
    request.model_intermediate_buffer["duplex"]["payload"] = {
        "lychee_audio_ledger": {"consumable_tick_start": 10, "audio_window_seq": 2}
    }
    state.add_request(0, request)
    batch = _batch(input_ids=[2], counts=[1], computed=[11], prefill_lens=[12])
    batch.num_tokens_after_padding = 1
    with pytest.raises(ValueError, match="complete three-channel/audio history"):
        state.prepare_inputs(batch, SimpleNamespace())


def test_legacy_audio_ledger_uses_absolute_start_and_rejects_outside_window(monkeypatch):
    state = _session_state(monkeypatch)
    state.intermediate_buffer.buffers[0] = dict(
        req_id="req-0",
        duplex=dict(seq=1, payload={"lychee_audio_ledger": {"consumable_tick_start": 10, "audio_window_seq": 2}}),
    )
    state._encode_audio_steps = lambda payload: torch.arange(20).reshape(10, 2).float()
    state._control_ticks[0] = 9
    batch = _batch(input_ids=[2], counts=[1], computed=[0], prefill_lens=[1])
    batch.num_tokens_after_padding = 1
    with pytest.raises(ValueError, match="outside its audio window"):
        state._prepare_audio_embeddings(batch)
    state._control_ticks[0] = 10
    assert state._prepare_audio_embeddings(batch).tolist() == [[0, 1]]
    state._control_ticks[0] = 20
    with pytest.raises(ValueError, match="outside its audio window"):
        state._prepare_audio_embeddings(batch)
