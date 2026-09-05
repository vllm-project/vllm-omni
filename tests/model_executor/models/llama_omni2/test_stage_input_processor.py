# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.llama_omni2 import (
    _FULL_PAYLOAD_REPLACE_KEYS,
    LlamaOmni2StreamState,
    LlamaOmni2StreamStateStore,
    talker2code2wav_async_chunk,
    talker2code2wav_full_payload,
    thinker2talker_async_chunk,
    thinker2talker_full_payload,
    thinker2talker_token_only,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_full_payload_cumulative_snapshots_replace_instead_of_append():
    assert _FULL_PAYLOAD_REPLACE_KEYS == frozenset(
        {
            "embed.decode",
            "hidden_states.output",
            "codes.audio",
        }
    )


def test_thinker_waits_for_three_new_tokens_before_scheduling():
    state = LlamaOmni2StreamState()

    assert state.consume_thinker_tokens([11, 12]) == []
    assert state.consume_thinker_tokens([11, 12, 13]) == [[11, 12, 13]]


def test_six_thinker_tokens_schedule_two_independent_bursts():
    state = LlamaOmni2StreamState()

    assert state.consume_thinker_tokens([1, 2, 3, 4, 5, 6]) == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_terminal_separator_is_scheduled_once_and_drain_is_bounded():
    state = LlamaOmni2StreamState(separator_token_id=99, max_drain_units=20)

    assert state.consume_thinker_tokens([1, 2], finished=True) == [[1, 2, 99]]
    assert state.consume_thinker_tokens([1, 2], finished=True) == []
    assert state.separator_scheduled
    assert state.consume_talker_tokens([7] * 10) == [7] * 10
    assert state.should_continue_drain
    assert state.consume_talker_tokens([7] * 10 + [8] * 10) == [8] * 10
    assert not state.should_continue_drain


def test_terminal_separator_joins_exact_boundary_burst():
    state = LlamaOmni2StreamState(separator_token_id=99)

    assert state.consume_thinker_tokens([1, 2, 3], finished=True) == [[1, 2, 3, 99]]


def test_talker_eos_stops_terminal_drain_before_safety_cap():
    state = LlamaOmni2StreamState(separator_token_id=99, talker_eos_token_id=6)

    state.consume_thinker_tokens([1, 2, 3], finished=True)
    assert state.consume_talker_tokens([4, 5, 6]) == [4, 5, 6]
    assert not state.should_continue_drain


def test_request_store_isolates_and_cancels_only_one_request():
    store = LlamaOmni2StreamStateStore()
    first = store.get("request-a")
    second = store.get("request-b")

    assert first.consume_thinker_tokens([1, 2, 3]) == [[1, 2, 3]]
    assert second.consume_thinker_tokens([8, 9]) == []
    store.cancel("request-a")

    assert "request-a" not in store
    assert store.get("request-b") is second
    assert store.get("request-b").consume_thinker_tokens([8, 9, 10]) == [[8, 9, 10]]


def test_codec_delta_is_monotonic_and_terminal_is_emitted_once():
    state = LlamaOmni2StreamState()

    assert state.consume_codec_tokens([100, 101]) == [100, 101]
    assert state.consume_codec_tokens([100, 101, 102], finished=True) == [102]
    assert state.codec_finished
    assert state.consume_codec_tokens([100, 101, 102], finished=True) == []


@pytest.mark.parametrize(
    "tokens",
    (
        [100],
        [100, 999],
    ),
)
def test_codec_duplicate_or_out_of_order_prefix_is_rejected(tokens):
    state = LlamaOmni2StreamState()
    state.consume_codec_tokens([100, 101])

    with pytest.raises(ValueError, match="codec token stream"):
        state.consume_codec_tokens(tokens)


class _TransferManager:
    def __init__(self):
        self.request_payload = {}
        self.code_prompt_token_ids = defaultdict(list)
        self.put_req_chunk = defaultdict(int)


class _Request:
    def __init__(self, request_id="request-a", output_token_ids=None):
        self.request_id = request_id
        self.external_req_id = request_id
        self.prompt_token_ids = [41, 42]
        self.output_token_ids = list(output_token_ids or [])

    def is_finished(self):
        return False


def _thinker_step(token_id: int):
    return {
        "embed": {"decode": torch.tensor([[float(token_id), 0.0]])},
        "hidden_states": {"output": torch.tensor([[0.0, float(token_id)]])},
    }


def test_thinker_async_emits_only_at_three_token_boundary():
    manager = _TransferManager()
    request = _Request()

    for token_id in (11, 12):
        request.output_token_ids.append(token_id)
        assert (
            thinker2talker_async_chunk(
                manager,
                _thinker_step(token_id),
                request,
            )
            is None
        )

    request.output_token_ids.append(13)
    payload = thinker2talker_async_chunk(
        manager,
        _thinker_step(13),
        request,
    )

    assert payload.ids.output == [11, 12, 13]
    assert payload.embed.decode.tolist() == [[11.0, 0.0], [12.0, 0.0], [13.0, 0.0]]
    assert payload.hidden_states.output.tolist() == [[0.0, 11.0], [0.0, 12.0], [0.0, 13.0]]
    assert payload.meta.next_stage_prompt_len == 3
    assert payload.meta.replace_streaming_prompt is True
    assert payload.meta.finished.item() is False


def test_thinker_async_uses_runner_hidden_rows_without_decode_embedding():
    manager = _TransferManager()
    request = _Request()

    for token_id in (11, 12):
        request.output_token_ids.append(token_id)
        assert (
            thinker2talker_async_chunk(
                manager,
                {"hidden": torch.tensor([[0.0, float(token_id)]])},
                request,
            )
            is None
        )

    request.output_token_ids.append(13)
    payload = thinker2talker_async_chunk(
        manager,
        {"hidden": torch.tensor([[0.0, 13.0]])},
        request,
    )

    assert payload.ids.output == [11, 12, 13]
    assert payload.embed.decode is None
    assert payload.hidden_states.output.tolist() == [
        [0.0, 11.0],
        [0.0, 12.0],
        [0.0, 13.0],
    ]
    assert payload.meta.next_stage_prompt_len == 3


def test_thinker_async_uses_runner_hidden_prefill_tail_then_decode_deltas():
    manager = _TransferManager()
    request = _Request()

    request.output_token_ids.append(11)
    assert (
        thinker2talker_async_chunk(
            manager,
            {"hidden": torch.arange(54, dtype=torch.float32).reshape(27, 2)},
            request,
        )
        is None
    )

    request.output_token_ids.append(12)
    assert (
        thinker2talker_async_chunk(
            manager,
            {"hidden": torch.tensor([[0.0, 12.0]])},
            request,
        )
        is None
    )

    request.output_token_ids.append(13)
    payload = thinker2talker_async_chunk(
        manager,
        {"hidden": torch.tensor([[0.0, 13.0]])},
        request,
    )

    assert payload.ids.output == [11, 12, 13]
    assert payload.hidden_states.output.tolist() == [
        [52.0, 53.0],
        [0.0, 12.0],
        [0.0, 13.0],
    ]


def test_thinker_async_isolates_interleaved_request_hidden_rows():
    manager = _TransferManager()
    request_a = _Request(request_id="request-a")
    request_b = _Request(request_id="request-b")

    for token_a, token_b in zip((11, 12), (21, 22), strict=True):
        request_a.output_token_ids.append(token_a)
        assert (
            thinker2talker_async_chunk(
                manager,
                {"hidden": torch.tensor([[1.0, float(token_a)]])},
                request_a,
            )
            is None
        )
        request_b.output_token_ids.append(token_b)
        assert (
            thinker2talker_async_chunk(
                manager,
                {"hidden": torch.tensor([[2.0, float(token_b)]])},
                request_b,
            )
            is None
        )

    request_b.output_token_ids.append(23)
    payload_b = thinker2talker_async_chunk(
        manager,
        {"hidden": torch.tensor([[2.0, 23.0]])},
        request_b,
    )
    request_a.output_token_ids.append(13)
    payload_a = thinker2talker_async_chunk(
        manager,
        {"hidden": torch.tensor([[1.0, 13.0]])},
        request_a,
    )

    assert payload_a.ids.output == [11, 12, 13]
    assert payload_a.hidden_states.output.tolist() == [
        [1.0, 11.0],
        [1.0, 12.0],
        [1.0, 13.0],
    ]
    assert payload_b.ids.output == [21, 22, 23]
    assert payload_b.hidden_states.output.tolist() == [
        [2.0, 21.0],
        [2.0, 22.0],
        [2.0, 23.0],
    ]


def test_thinker_async_accepts_one_request_local_hidden_row_per_batched_step():
    manager = _TransferManager()
    request_a = _Request(request_id="request-a", output_token_ids=[11])
    request_b = _Request(request_id="request-b", output_token_ids=[21])

    assert (
        thinker2talker_async_chunk(
            manager,
            {"hidden": torch.tensor([[1.0, 11.0]])},
            request_a,
        )
        is None
    )
    assert (
        thinker2talker_async_chunk(
            manager,
            {"hidden": torch.tensor([[2.0, 21.0]])},
            request_b,
        )
        is None
    )

    request_a_state = manager.request_payload["request-a"]
    request_b_state = manager.request_payload["request-b"]

    assert request_a_state["_llama_omni2_pending_thinker_rows"]["hidden"][0].tolist() == [[1.0, 11.0]]
    assert request_b_state["_llama_omni2_pending_thinker_rows"]["hidden"][0].tolist() == [[2.0, 21.0]]
    assert "_llama_omni2_stream_state" in request_a_state
    assert "_llama_omni2_stream_state" in request_b_state
    assert not hasattr(manager, "_llama_omni2_stream_states")
    assert not hasattr(manager, "_llama_omni2_pending_thinker_rows")


def test_thinker_async_terminal_tail_appends_separator_once():
    manager = _TransferManager()
    request = _Request(output_token_ids=[11, 12])

    assert thinker2talker_async_chunk(
        manager,
        {
            "embed": {"decode": torch.tensor([[11.0, 0.0], [12.0, 0.0]])},
            "hidden_states": {"output": torch.tensor([[0.0, 11.0], [0.0, 12.0]])},
        },
        request,
        is_finished=True,
    ).ids.output == [11, 12, 151665]

    assert (
        thinker2talker_async_chunk(
            manager,
            None,
            request,
            is_finished=True,
        )
        is None
    )


def test_thinker_full_payload_has_exact_typed_handoff_rows():
    request = _Request(output_token_ids=[11, 12, 13])
    payload = thinker2talker_full_payload(
        _TransferManager(),
        {
            "embed.decode": torch.arange(6, dtype=torch.float32).reshape(3, 2),
            "hidden_states.output": torch.arange(9, dtype=torch.float32).reshape(3, 3),
        },
        request,
    )

    assert set(payload) == {"ids", "embed", "hidden_states", "meta"}
    assert payload["ids"]["output"] == [11, 12, 13, 151665]
    assert payload["embed"]["decode"].shape[0] == 3
    assert payload["hidden_states"]["output"].shape[0] == 3
    assert payload["meta"]["finished"].item() is True
    assert payload["meta"]["next_stage_prompt_len"] == 4


def test_thinker_full_payload_prefers_payload_ids_over_async_placeholders():
    request = _Request(output_token_ids=[11, -1, -1])
    payload = thinker2talker_full_payload(
        _TransferManager(),
        {
            "ids.output": torch.tensor([11, 12, 13], dtype=torch.long),
            "embed.decode": torch.arange(6, dtype=torch.float32).reshape(3, 2),
            "hidden_states.output": torch.arange(9, dtype=torch.float32).reshape(3, 3),
        },
        request,
    )

    assert payload["ids"]["output"] == [11, 12, 13, 151665]


def test_thinker_full_payload_reports_mismatched_row_counts():
    request = _Request(output_token_ids=[11, 12, 13])

    with pytest.raises(
        ValueError,
        match=r"token_ids=3, embed=2, hidden=4",
    ):
        thinker2talker_full_payload(
            _TransferManager(),
            {
                "embed.decode": torch.zeros(2, 2),
                "hidden_states.output": torch.zeros(4, 3),
            },
            request,
        )


def test_thinker_token_only_builds_one_full_payload_placeholder():
    source = SimpleNamespace(
        request_id="request-a",
        prompt_token_ids=[41, 42],
        outputs=[SimpleNamespace(cumulative_token_ids=[11, 12, 13, 14, 15, 16])],
        finished=True,
    )

    [prompt] = thinker2talker_token_only([source])

    assert prompt["prompt_token_ids"] == [0] * 7
    assert prompt["sampling_params_override"] == {"max_tokens": 100}


def test_thinker_token_only_uses_finite_terminal_drain_budget():
    source = SimpleNamespace(
        request_id="request-a",
        prompt_token_ids=[41, 42],
        outputs=[SimpleNamespace(cumulative_token_ids=[11, 12])],
        finished=True,
    )

    [prompt] = thinker2talker_token_only([source])

    assert prompt["prompt_token_ids"] == [0, 0, 0]
    assert prompt["sampling_params_override"] == {"max_tokens": 100}


def test_thinker_token_only_terminal_exact_boundary_includes_separator_slot():
    source = SimpleNamespace(
        request_id="request-a",
        prompt_token_ids=[41, 42],
        outputs=[SimpleNamespace(cumulative_token_ids=[11, 12, 13])],
        finished=True,
    )

    [prompt] = thinker2talker_token_only([source])

    assert prompt["prompt_token_ids"] == [0, 0, 0, 0]
    assert prompt["sampling_params_override"] == {"max_tokens": 100}


def test_talker_async_emits_only_new_codec_delta_and_terminal_once():
    manager = _TransferManager()
    request = _Request()

    first = talker2code2wav_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([151766, 151767])}},
        request,
    )
    second = talker2code2wav_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([151766, 151767, 151768])}},
        request,
        is_finished=True,
    )

    assert first.codes.audio.tolist() == [100, 101]
    assert first.meta.finished.item() is False
    assert first.meta.request_id == "request-a"
    assert first.meta.chunk_seq == 0
    assert second.codes.audio.tolist() == [102]
    assert second.meta.finished.item() is True
    assert second.meta.chunk_seq == 1
    assert (
        talker2code2wav_async_chunk(
            manager,
            {"codes": {"audio": torch.tensor([151766, 151767, 151768])}},
            request,
            is_finished=True,
        )
        is None
    )


def test_talker_async_emits_first_terminal_even_without_new_codec_delta():
    manager = _TransferManager()
    request = _Request()
    talker2code2wav_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([151766])}},
        request,
    )

    terminal = talker2code2wav_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([151766])}},
        request,
        is_finished=True,
    )

    assert terminal.codes.audio.numel() == 0
    assert terminal.meta.finished.item() is True
    assert (
        talker2code2wav_async_chunk(
            manager,
            {"codes": {"audio": torch.tensor([151766])}},
            request,
            is_finished=True,
        )
        is None
    )


def test_talker_async_filters_terminal_eos_without_skipping_chunk_sequence():
    manager = _TransferManager()
    request = _Request()
    first = talker2code2wav_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([151766])}},
        request,
    )

    terminal = talker2code2wav_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([151766, 151643])}},
        request,
        is_finished=True,
    )

    assert first.codes.audio.tolist() == [100]
    assert first.meta.chunk_seq == 0
    assert terminal.codes.audio.numel() == 0
    assert terminal.meta.finished.item() is True
    assert terminal.meta.chunk_seq == 1


def test_talker_async_defers_repeated_eos_terminal_until_finished_flag():
    manager = _TransferManager()
    request = _Request()

    first = talker2code2wav_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([151766, 151643])}},
        request,
        is_finished=False,
    )
    repeated = talker2code2wav_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([151766, 151643, 151643, 151643])}},
        request,
        is_finished=False,
    )
    terminal = talker2code2wav_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([151766, 151643, 151643, 151643])}},
        request,
        is_finished=True,
    )

    assert first.codes.audio.tolist() == [100]
    assert first.meta.finished.item() is False
    assert first.meta.chunk_seq == 0
    assert repeated is None
    assert terminal.codes.audio.numel() == 0
    assert terminal.meta.finished.item() is True
    assert terminal.meta.chunk_seq == 1


def test_talker_async_invalid_codec_token_does_not_mutate_stream_state():
    manager = _TransferManager()
    request = _Request()

    with pytest.raises(ValueError, match="codec token IDs"):
        talker2code2wav_async_chunk(
            manager,
            {"codes": {"audio": torch.tensor([151665])}},
            request,
        )

    first = talker2code2wav_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([151766])}},
        request,
    )

    assert first.codes.audio.tolist() == [100]
    assert first.meta.chunk_seq == 0


def test_talker_async_rejects_out_of_order_codec_chunk():
    manager = _TransferManager()
    request = _Request()
    talker2code2wav_async_chunk(
        manager,
        {"codes": {"audio": torch.tensor([151766, 151767])}},
        request,
    )

    with pytest.raises(ValueError, match="codec token stream"):
        talker2code2wav_async_chunk(
            manager,
            {"codes": {"audio": torch.tensor([151766])}},
            request,
        )


def test_talker_full_payload_propagates_terminal_codec_ids():
    payload = talker2code2wav_full_payload(
        _TransferManager(),
        {"codes.audio": torch.tensor([151766, 151767, 151768])},
        _Request(),
    )

    assert payload["codes"]["audio"].tolist() == [100, 101, 102]
    assert payload["meta"]["finished"].item() is True
    assert payload["meta"]["request_id"] == "request-a"
    assert payload["meta"]["chunk_seq"] == 0


def test_talker_full_payload_filters_terminal_eos():
    payload = talker2code2wav_full_payload(
        _TransferManager(),
        {"codes.audio": torch.tensor([151766, 151643])},
        _Request(),
    )

    assert payload["codes"]["audio"].tolist() == [100]
    assert payload["meta"]["finished"].item() is True
    assert payload["meta"]["chunk_seq"] == 0


def test_talker_full_payload_filters_repeated_terminal_eos():
    payload = talker2code2wav_full_payload(
        _TransferManager(),
        {"codes.audio": torch.tensor([151766, 151643, 151643, 151643])},
        _Request(),
    )

    assert payload["codes"]["audio"].tolist() == [100]
    assert payload["meta"]["finished"].item() is True
    assert payload["meta"]["chunk_seq"] == 0


def test_talker_full_payload_uses_explicit_request_id_when_request_lacks_one():
    payload = talker2code2wav_full_payload(
        _TransferManager(),
        {"codes.audio": torch.tensor([151766])},
        SimpleNamespace(),
        request_id="request-b",
    )

    assert payload["meta"]["request_id"] == "request-b"


@pytest.mark.parametrize("token_id", [151665, 158227])
def test_talker_handoff_rejects_non_codec_token_ids(token_id):
    with pytest.raises(ValueError, match="codec token IDs"):
        talker2code2wav_full_payload(
            _TransferManager(),
            {"codes.audio": torch.tensor([token_id])},
            _Request(),
        )
