# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.worker_v2.model_states.lychee_model_state import (
    LycheeModelState,
    LycheePoisonedRequestError,
)
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_lychee_structured_output_uses_one_client_channel() -> None:
    assert LycheeModelState.structured_output_via_multimodal_only is True


def _batch(
    *,
    input_ids: list[int],
    counts: list[int],
    computed: list[int],
    prefill_lens: list[int],
) -> SimpleNamespace:
    starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    return SimpleNamespace(
        num_reqs=len(counts),
        query_start_loc_np=starts,
        num_scheduled_tokens=np.asarray(counts, dtype=np.int32),
        num_computed_tokens_np=np.asarray(computed, dtype=np.int32),
        prefill_len_np=np.asarray(prefill_lens, dtype=np.int32),
        idx_mapping_np=np.arange(len(counts), dtype=np.intp),
        idx_mapping=torch.arange(len(counts), dtype=torch.long),
        req_ids=[f"req-{index}" for index in range(len(counts))],
        input_ids=torch.tensor(input_ids, dtype=torch.int32),
    )


def _req_states(rows: list[list[int]]) -> SimpleNamespace:
    return SimpleNamespace(all_token_ids=SimpleNamespace(gpu=torch.tensor(rows, dtype=torch.int32)))


def test_merge_conditioning_uses_shifted_prompt_and_same_step_sample() -> None:
    batch = _batch(
        input_ids=[10, 11, 12],
        counts=[3],
        computed=[0],
        prefill_lens=[3],
    )
    result = LycheeModelState.build_merge_text_token_ids(
        input_batch=batch,
        req_states=_req_states([[10, 11, 12]]),
        sampled_text_token_ids=torch.tensor([[99]], dtype=torch.int32),
        text_pad_token_id=999,
        total_rows=3,
    )
    assert result.tolist() == [11, 12, 99]


def test_chunked_prefill_boundary_uses_next_known_prompt_token() -> None:
    batch = _batch(
        input_ids=[10, 11],
        counts=[2],
        computed=[0],
        prefill_lens=[3],
    )
    result = LycheeModelState.build_merge_text_token_ids(
        input_batch=batch,
        req_states=_req_states([[10, 11, 12]]),
        sampled_text_token_ids=torch.tensor([[99]], dtype=torch.int32),
        text_pad_token_id=999,
        total_rows=2,
    )
    assert result.tolist() == [11, 12]


def test_merge_conditioning_does_not_cross_request_boundaries() -> None:
    batch = _batch(
        input_ids=[10, 11, 20, 21],
        counts=[2, 2],
        computed=[0, 0],
        prefill_lens=[2, 2],
    )
    result = LycheeModelState.build_merge_text_token_ids(
        input_batch=batch,
        req_states=_req_states([[10, 11], [20, 21]]),
        sampled_text_token_ids=torch.tensor([[91], [92]], dtype=torch.int32),
        text_pad_token_id=999,
        total_rows=4,
    )
    assert result.tolist() == [11, 91, 21, 92]


def test_legal_greedy_uses_exclusive_upper_bound() -> None:
    logits = torch.zeros((2, 12))
    logits[0, 5] = 2
    logits[0, 8] = 100
    logits[1, 7] = 3
    sampled = LycheeModelState.sample_legal_greedy(
        logits,
        token_id_min=5,
        token_id_max=8,
    )
    assert sampled.tolist() == [5, 7]


def test_request_tokens_are_committed_only_for_active_rows() -> None:
    batch = _batch(
        input_ids=[10, 11, 20],
        counts=[2, 1],
        computed=[0, 0],
        prefill_lens=[2, 2],
    )
    result = LycheeModelState._request_tokens_on_model_rows(
        torch.tensor([91, 92]),
        input_batch=batch,
        num_sampled=torch.tensor([1, 0]),
        total_rows=3,
    )
    assert result.tolist() == [-1, 91, -1]


def test_prepare_inputs_uses_request_local_previous_side_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        OmniModelState,
        "prepare_inputs",
        lambda self, input_batch, req_states: {},
    )
    state = object.__new__(LycheeModelState)
    state.model = SimpleNamespace(
        config=SimpleNamespace(
            stoken_pad_token_id=158_359,
            sleep_token_id=158_357,
        )
    )
    state._last_stoken_tokens = torch.tensor([155_001, 155_002])
    state._last_control_tokens = torch.tensor([158_354, 158_355])
    state._poisoned_rows = set()
    state.intermediate_buffer = SimpleNamespace(buffers=[{"req_id": "req-0"}, {"req_id": "req-1"}])

    batch = _batch(
        input_ids=[10, 11, 20],
        counts=[2, 1],
        computed=[0, 2],
        prefill_lens=[2, 2],
    )
    batch.num_tokens_after_padding = 3
    prepared = state.prepare_inputs(batch, SimpleNamespace())

    assert prepared["stoken_input_ids"].tolist() == [158_359, 158_359, 155_002]
    assert prepared["control_input_ids"].tolist() == [158_357, 158_357, 158_355]


def test_audio_window_is_encoded_once_and_selected_across_ten_ticks(monkeypatch) -> None:
    monkeypatch.setattr(
        OmniModelState,
        "prepare_inputs",
        lambda self, input_batch, req_states: {},
    )
    monkeypatch.setattr(
        "vllm_omni.worker_v2.model_states.lychee_model_state.log_mel_spectrogram",
        lambda waveform: torch.zeros((128, 42), dtype=torch.float32),
    )

    class FakeModel:
        config = SimpleNamespace(
            stoken_pad_token_id=158_359,
            sleep_token_id=158_357,
            audio_pad_token_id=158_360,
            control_token_chunk_size=10,
            text_config=SimpleNamespace(hidden_size=4),
        )

        def __init__(self) -> None:
            self.encode_calls = 0

        def embed_input_ids(self, ids):
            assert ids.tolist() == [158_360] * len(ids)
            return torch.tensor([41, 42, 43, 44], dtype=torch.float32).expand(len(ids), -1).clone()

        def encode_audio(self, features, feature_lengths):
            self.encode_calls += 1
            assert features.shape == (1, 128, 42)
            assert feature_lengths.tolist() == [40]
            return torch.arange(20, dtype=torch.float32).reshape(1, 5, 4), torch.tensor([5])

    audio = base64.b64encode(struct.pack("<6400f", *([0.05] * 6400))).decode("ascii")
    state = object.__new__(LycheeModelState)
    state.model = FakeModel()
    state.device = torch.device("cpu")
    state.dtype = torch.float32
    state._last_stoken_tokens = torch.tensor([158_359])
    state._last_control_tokens = torch.tensor([158_357])
    state._control_ticks = torch.tensor([0])
    state._audio_window_seqs = torch.tensor([-1])
    state._poisoned_rows = set()
    state._audio_window_cache = {}
    state.intermediate_buffer = SimpleNamespace(
        buffers=[
            {
                "req_id": "req-0",
                "duplex": {
                    "seq": 1,
                    "payload": {
                        "audio": audio,
                        "format": "pcm_f32le",
                        "sample_rate_hz": 16000,
                    },
                },
            }
        ]
    )
    batch = _batch(
        input_ids=[158_358],
        counts=[1],
        computed=[0],
        prefill_lens=[1],
    )
    batch.num_tokens_after_padding = 1

    first = state.prepare_inputs(batch, SimpleNamespace())
    assert first["audio_embeddings"].tolist() == [[0.0, 1.0, 2.0, 3.0]]
    assert state._audio_window_seqs.tolist() == [1]

    state._control_ticks[0] = 1
    padding = state.prepare_inputs(batch, SimpleNamespace())
    assert padding["audio_embeddings"].tolist() == [[41, 42, 43, 44]]

    state._control_ticks[0] = 2
    second = state.prepare_inputs(batch, SimpleNamespace())
    assert second["audio_embeddings"].tolist() == [[4.0, 5.0, 6.0, 7.0]]
    assert state.model.encode_calls == 1


def test_tick_transaction_prepares_then_commits_only_active_rows() -> None:
    state = object.__new__(LycheeModelState)
    state._prepared_ticks = torch.tensor([-1, -1])
    state._committed_ticks = torch.tensor([-1, -1])
    batch = _batch(
        input_ids=[10, 20],
        counts=[1, 1],
        computed=[0, 0],
        prefill_lens=[1, 2],
    )

    state._prepare_tick(input_batch=batch, num_sampled=torch.tensor([1, 0]))
    assert state._prepared_ticks.tolist() == [0, -1]
    assert state._committed_ticks.tolist() == [-1, -1]

    state._commit_tick(
        req_state_indices=batch.idx_mapping,
        active=torch.tensor([True, False]),
    )
    assert state._committed_ticks.tolist() == [0, -1]


def test_failed_continuation_poison_blocks_row_reuse(monkeypatch) -> None:
    monkeypatch.setattr(
        OmniModelState,
        "prepare_inputs",
        lambda self, input_batch, req_states: {},
    )
    state = object.__new__(LycheeModelState)
    state._poisoned = torch.tensor([False])
    state._poisoned_rows = set()
    state._execution_epochs = torch.tensor([3])
    state._next_execution_epoch = 4
    state.intermediate_buffer = SimpleNamespace(buffers=[{"req_id": "req-0"}])
    batch = _batch(
        input_ids=[10],
        counts=[1],
        computed=[0],
        prefill_lens=[1],
    )

    state.mark_primary_continuation_failed(input_batch=batch)

    assert state._poisoned.tolist() == [True]
    assert state._execution_epochs.tolist() == [4]
    assert state._next_execution_epoch == 5
    with pytest.raises(LycheePoisonedRequestError, match="req-0"):
        state.prepare_inputs(batch, SimpleNamespace())


def _session_state(monkeypatch, *, max_model_len=8192, max_num_reqs=3):
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

    def initialize(self, model):
        self.model = model
        self.max_num_reqs = max_num_reqs
        self.model_config = SimpleNamespace(max_model_len=max_model_len)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.intermediate_buffer = OmniIntermediateBuffer(max_num_reqs)

    monkeypatch.setattr(OmniModelState, "__init__", initialize)
    monkeypatch.setattr(
        OmniModelState,
        "add_request",
        lambda self, index, data: self.intermediate_buffer.add_request(index, data),
    )
    monkeypatch.setattr(
        OmniModelState,
        "remove_request",
        lambda self, key: (
            self.intermediate_buffer.remove_request(self._resolve_req_index(key))
            if self._resolve_req_index(key) is not None
            else None
        ),
    )
    monkeypatch.setattr(OmniModelState, "on_requests_finished", lambda self, ids: None)
    monkeypatch.setattr(OmniModelState, "prepare_inputs", lambda self, batch, states: {})

    class Model:
        config = SimpleNamespace(
            text_pad_token_id=2,
            stoken_pad_token_id=3,
            audio_pad_token_id=25,
            sleep_token_id=4,
            detect_token_id=5,
            start_speaking_token_id=6,
            start_listening_token_id=7,
            keep_listening_token_id=8,
            keep_speaking_token_id=9,
            start_bc_token_id=10,
            stoken_delay_token_id=11,
            tts_start_token_id=12,
            tts_end_token_id=13,
            stoken_audio_token_id_min=14,
            stoken_token_ids_max=22,
            stoken_delay_num=1,
            stoken_max_tokens=100,
            stoken_no_repeat_ngram_size=4,
            stoken_do_sample=True,
            stoken_temperature=0.7,
            stoken_top_k=0,
            stoken_top_p=1.0,
            control_token_chunk_size=10,
            text_config=SimpleNamespace(hidden_size=2),
        )

        def embed_input_ids(self, ids):
            return ids.float()[:, None].expand(-1, 2).clone()

        def continue_after_primary_sample(self, **kwargs):
            return None, torch.arange(40, dtype=torch.float32)[None, :] / 40

        def compute_control_logits(self, hidden):
            logits = torch.zeros((hidden.shape[0], 40))
            logits[:, self.config.start_speaking_token_id] = 10
            logits[:, self.config.keep_speaking_token_id] = 10
            return logits

    state = LycheeModelState(Model())
    state.encoded_windows = []

    def encode(payload):
        state.encoded_windows.append(payload)
        return torch.arange(20, dtype=torch.float32).reshape(10, 2) + payload * 100

    state._encode_audio_steps = encode
    return state


def _session_request(seq=1, req_id="req-0"):
    return SimpleNamespace(
        req_id=req_id,
        sampling_params=SimpleNamespace(seed=17),
        mm_features=[],
        model_intermediate_buffer={"duplex": {"data_plane": True, "seq": seq, "payload": seq}},
    )


def _session_tick(state, tick, slot=0):
    batch = _batch(
        input_ids=[2 if tick in (0, 10) else 100 + tick - 1],
        counts=[1],
        computed=[tick],
        prefill_lens=[11 if tick >= 10 else 1],
    )
    batch.idx_mapping_np[:] = slot
    batch.idx_mapping[:] = slot
    batch.num_tokens_after_padding = 1
    batch.num_draft_tokens = 0
    batch.positions = torch.tensor([tick])
    batch.logits_indices = torch.tensor([0])
    prepared = state.prepare_inputs(batch, SimpleNamespace())
    sampled = state.constrain_primary_sample(
        sampled_token_ids=torch.tensor([[100 + tick]], dtype=torch.int32),
        num_sampled=torch.tensor([1]),
        input_batch=batch,
    )
    output = state.continue_after_primary_sample(
        sampled_token_ids=sampled,
        num_sampled=torch.tensor([1]),
        multimodal_outputs={"lychee_stoken_hidden": torch.zeros((1, 2)), "lychee_control_hidden": torch.zeros((1, 2))},
        input_batch=batch,
        req_states=_req_states([[2] * 30] * 3),
    )
    return prepared, {key: tensor.tolist() for key, tensor in output.items()}


def test_streaming_rebind_continues_two_windows_and_moves_slots(monkeypatch):
    state = _session_state(monkeypatch)
    reference = _session_state(monkeypatch)
    state.add_request(0, _session_request())
    reference.add_request(0, _session_request())
    output_ticks = []
    for tick in range(10):
        _, output = _session_tick(state, tick)
        _, expected = _session_tick(reference, tick)
        assert output == expected
        output_ticks += output["lychee_tick"]

    old_generator = state._speech_generators["req-0"]
    old_audio_cache = state._audio_window_cache["req-0"]
    old_epoch = state._execution_epochs[0].item()
    last_text = state._last_text_tokens[0].item()
    last_speech = state._last_stoken_tokens[0].item()
    last_control = state._last_control_tokens[0].item()
    state.on_request_rebind("req-0", 0)
    state.remove_request("req-0")
    # Another session may immediately reuse the released slot.
    state.add_request(0, _session_request(req_id="other"))
    state.add_request(2, _session_request(seq=2))
    reference.intermediate_buffer.update(0, _session_request(seq=2).model_intermediate_buffer)

    assert state._speech_generators["req-0"] is old_generator
    assert state._audio_window_cache["req-0"] is old_audio_cache
    assert state._control_ticks[2].item() == 10
    assert state._execution_epochs[2].item() == old_epoch
    assert state._next_execution_epoch == 2
    assert not state._rebind_state
    assert state._control_modes[2].item() == 1
    for tick in range(10, 20):
        prepared, output = _session_tick(state, tick, slot=2)
        _, expected = _session_tick(reference, tick)
        assert output == expected
        output_ticks += output["lychee_tick"]
        if tick == 10:
            assert prepared["input_ids"].tolist() == [last_text]
            assert prepared["stoken_input_ids"].tolist() == [last_speech]
            assert prepared["control_input_ids"].tolist() == [last_control]
            assert prepared["audio_embeddings"].tolist() == [[200.0, 201.0]]
        else:
            assert "input_ids" not in prepared
    assert output_ticks == list(range(20))
    assert state.encoded_windows == [1, 2]
    assert reference.encoded_windows == [1, 2]
    assert state._control_ticks[2].item() == 20
    assert state._committed_ticks[2].item() == 19
    assert state._speech_history[2].tolist() == reference._speech_history[0].tolist()
    assert state._last_text_tokens[2].item() == reference._last_text_tokens[0].item()
    # Compare the subsequent random draw, not the generator's byte state.
    assert (
        torch.rand((), generator=old_generator).item()
        == torch.rand((), generator=reference._speech_generators["req-0"]).item()
    )
    assert not state._pending_append_text


def test_fresh_binding_initializes_and_finished_rebind_cleans_owned_state(monkeypatch):
    state = _session_state(monkeypatch)
    state.add_request(0, _session_request())
    _session_tick(state, 0)
    state.on_request_rebind("req-0", 0)
    state.remove_request("req-0")
    assert "req-0" in state._rebind_state
    state.on_requests_finished({"req-0"})
    assert not state._rebind_state
    assert not state._speech_generators
    assert not state._audio_window_cache
    assert not state._pending_append_text

    state.add_request(2, _session_request())
    assert state._last_text_tokens[2].item() == 2
    assert state._last_stoken_tokens[2].item() == 3
    assert state._last_control_tokens[2].item() == 4
    assert state._control_ticks[2].item() == 0
    assert state._control_modes[2].item() == 0
    assert state._speaking_steps[2].item() == -1
    assert state._speech_history[2].tolist() == [3] + [-1] * 101
    assert state._prepared_ticks[2].item() == -1
    assert state._committed_ticks[2].item() == -1
    assert state._execution_epochs[2].item() == 1
    state.remove_request(2)
    assert not state._speech_generators
    assert not state._audio_window_cache


def test_rebind_preserves_failure_poison_across_slot_change(monkeypatch):
    state = _session_state(monkeypatch)
    state.add_request(0, _session_request())
    batch = _batch(input_ids=[2], counts=[1], computed=[0], prefill_lens=[1])
    state.mark_primary_continuation_failed(input_batch=batch)
    state.on_request_rebind("req-0", 0)
    state.remove_request("req-0")
    state.add_request(2, _session_request(seq=2))
    batch.idx_mapping_np[:] = 2
    batch.idx_mapping[:] = 2
    assert state._poisoned_rows == {2}
    assert state._execution_epochs[2].item() == 1
    with pytest.raises(LycheePoisonedRequestError, match="req-0"):
        state.prepare_inputs(batch, SimpleNamespace())


def test_runner_announces_same_id_rebind_before_upstream_removal(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace()
    runner.req_states = SimpleNamespace(req_id_to_index={"req-0": 2})
    events: list[tuple[str] | tuple[str, str, int]] = []
    runner.model_state = SimpleNamespace(on_request_rebind=lambda req_id, idx: events.append(("rebind", req_id, idx)))
    monkeypatch.setattr(GPUModelRunner, "add_requests", lambda self, out: events.append(("upstream",)))
    runner.add_requests(SimpleNamespace(scheduled_new_reqs=[_session_request(), _session_request(req_id="fresh")]))
    assert events == [("rebind", "req-0", 2), ("upstream",)]
