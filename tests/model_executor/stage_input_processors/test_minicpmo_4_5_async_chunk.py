# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.request import RequestStatus

from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    _extract_codec_delta,
    tts2code2wav_async_chunk,
    tts2code2wav_full_payload,
    tts2code2wav_token_only,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _manager():
    return SimpleNamespace(
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 25, "codec_left_context_frames": 3}}),
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        put_req_chunk=defaultdict(int),
    )


def _request(external_id: str, internal_id: str | None = None):
    request = SimpleNamespace(
        external_req_id=external_id,
        request_id=internal_id or external_id,
        status=RequestStatus.RUNNING,
    )
    request.is_finished = lambda: RequestStatus.is_finished(request.status)
    return request


def _delta(*codes: int):
    return {
        "codes": {"audio": torch.tensor(codes, dtype=torch.long).reshape(-1, 1)},
        "meta": {"finished": torch.tensor(False)},
    }


def _duplex_delta(
    *codes: int,
    epoch: int = 3,
    turn_id: int = 7,
    text: str = "segment",
    turn_end: bool = False,
):
    text_utf8 = torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)
    return {
        "codes": {"audio": torch.tensor(codes, dtype=torch.long).reshape(-1, 1)},
        "meta": {
            "finished": torch.tensor(False),
            "native_duplex": torch.tensor(True),
            "duplex_epoch": torch.tensor(epoch),
            "duplex_turn_id": torch.tensor(turn_id),
            "llm_output_text_utf8": text_utf8,
            "turn_end": torch.tensor(turn_end),
        },
    }


def _codes(payload) -> list[int]:
    assert payload.codes is not None
    assert isinstance(payload.codes.audio, torch.Tensor)
    assert payload.codes.audio.dtype == torch.long
    assert payload.codes.audio.ndim == 1
    return payload.codes.audio.tolist()


def test_empty_full_payload_releases_consumer_wait_gate() -> None:
    payload = tts2code2wav_full_payload(_manager(), None, _request("req"))

    assert _codes(payload) == []
    assert payload.meta.code_flat_numel == 0
    assert payload.meta.left_context_size == 0
    assert payload.meta.last_chunk is True
    assert payload.meta.finished.item() is True


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("audio_shape", [(0,), (0, 1)])
def test_empty_device_codec_prefill_preserves_next_chunk(flat: bool, audio_shape: tuple[int, ...]) -> None:
    audio = torch.empty(audio_shape, dtype=torch.long)
    valid = torch.empty(0, dtype=torch.bool)
    empty_output = (
        {"codes.audio": audio, "meta.codec_frame_valid": valid}
        if flat
        else {"codes": {"audio": audio}, "meta": {"codec_frame_valid": valid}}
    )
    assert _extract_codec_delta(empty_output, "req") == []
    manager = _manager()
    request = _request("req")
    assert tts2code2wav_async_chunk(manager, empty_output, request, False) is None
    payload = tts2code2wav_async_chunk(manager, _delta(*range(25)), request, False)
    assert payload is not None
    assert _codes(payload) == [4218, 4218, 4218, *range(25)]
    assert payload.meta.chunk_seq == 0


def test_device_codec_validity_filters_terminal_token() -> None:
    output = {
        "codes": {"audio": torch.tensor([[2], [6561], [3]])},
        "meta": {"codec_frame_valid": torch.tensor([True, False, True])},
    }
    assert _extract_codec_delta(output, "req") == [2, 3]


@pytest.mark.parametrize(("count", "emitted"), [(24, False), (25, True), (26, True)])
def test_first_chunk_threshold_is_25_generated_codes(count: int, emitted: bool) -> None:
    manager = _manager()
    payload = tts2code2wav_async_chunk(
        transfer_manager=manager,
        multimodal_output=_delta(*range(count)),
        request=_request("req"),
        is_finished=False,
    )

    assert (payload is not None) is emitted
    if payload is not None:
        assert _codes(payload) == [4218, 4218, 4218, *range(25)]
        assert payload.meta.chunk_seq == 0
        assert payload.meta.code_flat_numel == 28


def test_steady_chunk_has_three_code_overlap_and_25_new_codes() -> None:
    manager = _manager()
    request = _request("req")

    first = tts2code2wav_async_chunk(manager, _delta(*range(25)), request, False)
    manager.put_req_chunk["req"] += 1
    steady = tts2code2wav_async_chunk(manager, _delta(*range(25, 50)), request, False)

    assert first is not None
    assert steady is not None
    assert _codes(steady) == [22, 23, 24, *range(25, 50)]
    assert steady.meta.chunk_seq == 1


def test_exact_boundary_final_flushes_held_lookahead() -> None:
    manager = _manager()
    request = _request("req")

    assert tts2code2wav_async_chunk(manager, _delta(*range(25)), request, False) is not None
    manager.put_req_chunk["req"] += 1
    final = tts2code2wav_async_chunk(manager, None, request, True)

    assert final is not None
    assert _codes(final) == [22, 23, 24]
    assert final.meta.chunk_seq == 1
    assert final.meta.code_flat_numel == 3
    assert final.meta.last_chunk is True
    assert final.meta.finished.item() is True


def test_short_final_flushes_silence_prefix_and_tail() -> None:
    manager = _manager()
    final = tts2code2wav_async_chunk(manager, _delta(*range(7)), _request("req"), True)

    assert final is not None
    assert _codes(final) == [4218, 4218, 4218, *range(7)]
    assert final.meta.last_chunk is True
    assert final.meta.finished.item() is True


def test_duplex_turn_end_waits_for_terminal_codec_flush() -> None:
    manager = _manager()
    request = _request("req-duplex")

    body = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(*range(25), turn_end=True),
        request,
        False,
    )
    final = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(turn_end=True),
        request,
        True,
    )

    assert body is not None
    assert body.meta.last_chunk is False
    assert body.meta.turn_end is False
    assert final is not None
    assert _codes(final) == [22, 23, 24]
    assert final.meta.last_chunk is True
    assert final.meta.turn_end is True


@pytest.mark.parametrize(
    "frame_count,delta_frames,body_chunks",
    [
        # Single-frame deltas across the 25-frame chunk boundary.
        (1, 1, 0),
        (24, 1, 0),
        (25, 1, 1),
        (26, 1, 1),
        (60, 1, 2),
        # Batched deltas: two full chunks, then a short tail held for the final flush.
        (52, 25, 2),
    ],
)
def test_duplex_final_segment_preserves_every_code_and_closes_once(
    frame_count: int, delta_frames: int, body_chunks: int
) -> None:
    manager = _manager()
    request = _request("final-segment")
    emitted = []
    bodies = 0
    for start in range(0, frame_count, delta_frames):
        codes = range(start, min(start + delta_frames, frame_count))
        payload = tts2code2wav_async_chunk(manager, _duplex_delta(*codes, turn_end=True), request, False)
        if payload is not None:
            assert payload.meta.last_chunk is False
            assert payload.meta.turn_end is False
            bodies += 1
            emitted.extend(_codes(payload)[payload.meta.codec_left_context_frames :])
    assert bodies == body_chunks

    final = tts2code2wav_async_chunk(manager, _duplex_delta(turn_end=True), request, True)
    assert final is not None
    assert final.meta.last_chunk is True
    assert final.meta.turn_end is True
    emitted.extend(_codes(final)[final.meta.codec_left_context_frames :])
    assert emitted == list(range(frame_count))

    duplicate = tts2code2wav_async_chunk(manager, _duplex_delta(turn_end=True), request, True)
    assert duplicate is not None
    assert duplicate.codes is None
    assert not duplicate.meta.last_chunk


def test_first_chunk_forwards_reference_voice_and_duplex_identity() -> None:
    manager = _manager()
    request = _request("req")
    request.model_intermediate_buffer = {
        "codes": {"ref": [0.1, -0.1]},
        "meta": {"ref_audio_sr": 16000},
    }
    request.additional_information = {
        "codes": {"ref": [0.9]},
        "meta": {"ref_audio_sr": 8000},
    }

    payload = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(*range(7), text="hello", turn_end=True),
        request,
        True,
    )

    assert payload is not None
    assert payload.codes.ref.tolist() == pytest.approx([0.1, -0.1])
    assert payload.meta.ref_audio_sr == 16000
    torch.testing.assert_close(
        payload.meta.llm_output_text_utf8,
        torch.tensor(list(b"hello"), dtype=torch.uint8),
    )
    assert payload.meta.duplex_epoch == 3
    assert payload.meta.duplex_turn_id == 7
    assert payload.meta.tts_is_last_chunk is True
    assert payload.meta.turn_end is True


def test_first_chunk_falls_back_to_legacy_reference_fields() -> None:
    manager = _manager()
    request = _request("req")
    request.model_intermediate_buffer = {
        "meta": {"ref_audio_sr": 16000},
    }
    request.additional_information = {
        "codes": {"ref": [0.9]},
        "meta": {"ref_audio_sr": 8000},
    }

    payload = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(*range(7), turn_end=True),
        request,
        True,
    )

    assert payload is not None
    assert payload.codes.ref.tolist() == pytest.approx([0.9])
    assert payload.meta.ref_audio_sr == 16000


def test_full_payload_forwards_all_codes_and_request_metadata() -> None:
    manager = _manager()
    request = _request("req")
    request.model_intermediate_buffer = {
        "codes": {"ref": [0.1, -0.1]},
        "meta": {
            "ref_audio_sr": 16000,
            "native_duplex_segment_text": "hello",
            "segment_end": True,
            "turn_end": True,
        },
        "duplex": {"epoch": 3, "model_turn_id": 7},
    }

    payload = tts2code2wav_full_payload(
        transfer_manager=manager,
        pooling_output={
            "codes.audio": torch.arange(7, dtype=torch.long).reshape(-1, 1),
            "meta.finished": torch.tensor(True),
        },
        request=request,
    )

    assert _codes(payload) == [4218, 4218, 4218, *range(7)]
    assert payload.codes.ref.tolist() == pytest.approx([0.1, -0.1])
    assert payload.meta.request_id == "req"
    assert payload.meta.chunk_seq == 0
    assert payload.meta.code_flat_numel == 10
    assert payload.meta.codec_chunk_frames == 7
    assert payload.meta.codec_left_context_frames == 3
    assert payload.meta.left_context_size == 3
    assert payload.meta.last_chunk is True
    assert payload.meta.finished.item() is True
    assert payload.meta.ref_audio_sr == 16000
    assert payload.meta.native_duplex_segment_text == "hello"
    assert payload.meta.duplex_epoch == 3
    assert payload.meta.duplex_turn_id == 7
    assert payload.meta.segment_end is True
    assert payload.meta.turn_end is True


def test_sync_token_only_reserves_codec_and_silence_slots() -> None:
    output = SimpleNamespace(
        finished=True,
        outputs=[
            SimpleNamespace(
                multimodal_output={
                    "codes.audio": torch.arange(7, dtype=torch.long).reshape(-1, 1),
                    "meta.finished": torch.tensor(True),
                }
            )
        ],
    )

    prompts = tts2code2wav_token_only([output])

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [0] * 10
    assert prompts[0]["additional_information"] is None


@pytest.mark.parametrize("with_validity", [False, True])
def test_empty_final_releases_wait_gate_once(with_validity: bool) -> None:
    manager = _manager()
    request = _request("req")

    output = _delta() if with_validity else None
    if output is not None:
        output["meta"]["codec_frame_valid"] = torch.empty(0, dtype=torch.bool)
    final = tts2code2wav_async_chunk(manager, output, request, True)
    duplicate = tts2code2wav_async_chunk(manager, output, request, True)

    assert final is not None
    assert _codes(final) == []
    assert final.meta.chunk_seq == 0
    assert final.meta.request_id == "req"
    assert final.meta.cache_epoch == 0
    assert final.meta.last_chunk is True
    assert duplicate is None


def test_empty_duplex_boundary_uses_zero_length_transport_placeholder() -> None:
    manager = _manager()

    boundary = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(text="boundary"),
        _request("req-duplex"),
        True,
    )

    assert boundary is not None
    assert _codes(boundary) == [0]
    assert boundary.meta.code_flat_numel == 0
    assert boundary.meta.last_chunk is False
    assert boundary.meta.is_segment_finished.item() is False
    torch.testing.assert_close(
        boundary.meta.llm_output_text_utf8,
        torch.tensor(list(b"boundary"), dtype=torch.uint8),
    )


def test_duplex_segments_preserve_stream_state_without_closing_turn() -> None:
    manager = _manager()
    request = _request("req-duplex")
    request.additional_information = {
        "codes": {"ref": [0.1, 0.2, 0.3]},
        "meta": {"ref_audio_sr": 16000},
    }

    first = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(10, 11, text="first"),
        request,
        True,
    )
    second = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(12, 13, text="second"),
        request,
        True,
    )

    assert first is not None
    assert second is not None
    assert first.meta.last_chunk is False
    assert second.meta.last_chunk is False
    assert first.codes is not None
    assert torch.allclose(first.codes.ref, torch.tensor([0.1, 0.2, 0.3]))
    assert first.meta.ref_audio_sr == 16000
    assert second.codes is not None
    assert second.codes.ref is None
    assert second.meta.cache_epoch == first.meta.cache_epoch
    assert second.meta.chunk_seq == first.meta.chunk_seq + 1
    assert second.meta.duplex_epoch == 3
    assert second.meta.duplex_turn_id == 7
    torch.testing.assert_close(
        second.meta.llm_output_text_utf8,
        torch.tensor(list(b"second"), dtype=torch.uint8),
    )
    assert second.meta.tts_is_last_chunk is True
    assert second.meta.turn_end is False
    assert first.meta.is_segment_finished.item() is False
    assert second.meta.is_segment_finished.item() is False


def test_duplex_short_units_wait_for_minimum_stream_body() -> None:
    manager = _manager()
    request = _request("req-duplex")

    first = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(10, 11, 12, text="first"),
        request,
        True,
    )
    second = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(13, 14, text="first"),
        request,
        True,
    )

    assert first is not None
    assert _codes(first) == [0]
    assert first.meta.code_flat_numel == 0
    assert first.meta.last_chunk is False
    assert first.meta.tts_is_last_chunk is True
    assert second is not None
    assert _codes(second) == [4218, 4218, 4218, 10, 11, 12, 13, 14]
    assert second.meta.code_flat_numel == 8
    torch.testing.assert_close(
        second.meta.llm_output_text_utf8,
        torch.tensor(list(b"firstfirst"), dtype=torch.uint8),
    )


def test_duplex_empty_finish_callback_does_not_replay_previous_text() -> None:
    manager = _manager()
    request = _request("req-duplex")

    first = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(*range(25), text="first"),
        request,
        False,
    )
    boundary = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(text="first"),
        request,
        True,
    )
    second = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(*range(25, 50), text="second"),
        request,
        False,
    )

    assert first is not None
    assert boundary is not None
    assert boundary.meta.code_flat_numel == 0
    assert second is not None
    torch.testing.assert_close(
        second.meta.llm_output_text_utf8,
        torch.tensor(list(b"second"), dtype=torch.uint8),
    )


def test_duplex_short_tail_does_not_replay_previous_segment_text() -> None:
    manager = _manager()
    request = _request("req-duplex")

    def chunk(codes, text: str, finished: bool):
        return tts2code2wav_async_chunk(
            manager,
            _duplex_delta(*codes, text=text),
            request,
            finished,
        )

    first = chunk(range(25), "和上海之间", False)
    first_tail = chunk([25, 26], "和上海之间", True)
    assert first is not None
    assert first.meta.llm_output_text_utf8.tolist() == list("和上海之间".encode())
    assert first_tail is not None
    assert first_tail.meta.code_flat_numel == 0
    assert chunk([27, 28, 29], "的距离大约是", False) is None
    next_flush = chunk([], "的距离大约是", True)
    assert next_flush is not None
    assert next_flush.meta.llm_output_text_utf8.tolist() == list("的距离大约是".encode())


def test_duplex_turn_end_closes_epoch_and_next_turn_restarts_sequence() -> None:
    manager = _manager()
    request = _request("req-duplex")

    turn_end = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(14, turn_id=7, turn_end=True),
        request,
        True,
    )
    duplicate_boundary = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(turn_id=7, turn_end=True),
        request,
        True,
    )
    next_turn = tts2code2wav_async_chunk(
        manager,
        _duplex_delta(20, turn_id=8),
        request,
        True,
    )

    assert turn_end is not None
    assert next_turn is not None
    assert turn_end.meta.last_chunk is True
    assert turn_end.meta.turn_end is True
    assert turn_end.meta.is_segment_finished.item() is True
    assert turn_end.meta.replace_runtime_additional_information is True
    assert duplicate_boundary is not None
    assert duplicate_boundary.meta.is_segment_finished.item() is True
    assert duplicate_boundary.meta.replace_runtime_additional_information is True
    assert next_turn.meta.cache_epoch == turn_end.meta.cache_epoch + 1
    assert next_turn.meta.chunk_seq == 0
    assert next_turn.meta.last_chunk is False


def test_staggered_requests_keep_accumulators_isolated() -> None:
    manager = _manager()
    req_a = _request("a")
    req_b = _request("b")

    assert tts2code2wav_async_chunk(manager, _delta(*range(24)), req_a, False) is None
    out_b = tts2code2wav_async_chunk(manager, _delta(*range(100, 125)), req_b, False)
    out_a = tts2code2wav_async_chunk(manager, _delta(24), req_a, False)

    assert out_b is not None
    assert out_a is not None
    assert _codes(out_b) == [4218, 4218, 4218, *range(100, 125)]
    assert _codes(out_a) == [4218, 4218, 4218, *range(25)]
    assert out_a.meta.request_id == "a"
    assert out_b.meta.request_id == "b"


def test_cancel_drops_epoch_state_and_stale_request_cannot_publish() -> None:
    manager = _manager()
    stale = _request("req", "internal-0")

    assert tts2code2wav_async_chunk(manager, _delta(*range(10)), stale, False) is None
    stale.status = RequestStatus.FINISHED_ABORTED
    assert tts2code2wav_async_chunk(manager, None, stale, True) is None
    assert tts2code2wav_async_chunk(manager, _delta(*range(25)), stale, False) is None

    replacement = _request("req", "internal-1")
    payload = tts2code2wav_async_chunk(manager, _delta(*range(25)), replacement, False)

    assert payload is not None
    assert payload.meta.cache_epoch == 1
    assert _codes(payload) == [4218, 4218, 4218, *range(25)]


@pytest.mark.parametrize("aborted", [False, True])
def test_mrv2_abort_terminal_drops_pending_codec_frames(aborted: bool) -> None:
    """The MRv2 snapshot has no scheduler status; its abort mark must behave like V1's FINISHED_ABORTED."""
    from vllm_omni.worker_v2.omni_data_plane import _NativeRequestState

    manager = _manager()
    state = _NativeRequestState(request_id="native", external_req_id="native", prompt_token_ids=[0] * 3)
    pending = state.snapshot(include_token_history=True, sampled_token_ids=[])
    assert tts2code2wav_async_chunk(manager, _duplex_delta(*range(10)), pending) is None
    state.finished = True
    state.aborted = aborted
    terminal = tts2code2wav_async_chunk(manager, None, state.snapshot(include_token_history=True))
    if aborted:
        assert terminal is None
        assert "native" not in manager.code_prompt_token_ids
    else:
        assert terminal is not None
        assert _codes(terminal)[3:] == list(range(10))


@pytest.mark.parametrize("turn_end", [False, True])
def test_mrv2_sampled_codec_eos_flushes_resumable_segment(turn_end: bool) -> None:
    from vllm_omni.worker_v2.omni_data_plane import _NativeRequestState

    state = _NativeRequestState(
        request_id="native",
        external_req_id="native",
        prompt_token_ids=[0] * 3,
        resumable=True,
        sampling_params=SimpleNamespace(stop_token_ids=[6561]),
    )
    state.accept_tokens([6561])
    request = state.snapshot(include_token_history=True, sampled_token_ids=[6561])
    assert not request.is_finished()  # The session remains available for another unit.
    payload = tts2code2wav_async_chunk(_manager(), _duplex_delta(*range(7), turn_end=turn_end), request)
    assert payload is not None
    assert _codes(payload) == [4218] * 3 + list(range(7))
    assert payload.meta.tts_is_last_chunk is True
    assert payload.meta.last_chunk is turn_end


def test_old_sampled_eos_cannot_close_the_next_turn():
    from vllm_omni.worker_v2.omni_data_plane import _NativeRequestState

    manager = _manager()
    state = _NativeRequestState(
        request_id="native",
        external_req_id="native",
        prompt_token_ids=[0, 0],
        resumable=True,
        sampling_params=SimpleNamespace(stop_token_ids=[6561]),
    )
    state.accept_tokens([6561])
    # The next payload has no newly accepted token, even though the ledger
    # still ends in the previous segment's EOS.
    stale = state.snapshot(include_token_history=True)
    first = tts2code2wav_async_chunk(manager, _duplex_delta(turn_id=8, turn_end=True), stale)
    assert first is None
    state.accept_tokens([1])
    body = tts2code2wav_async_chunk(
        manager, _duplex_delta(*range(30), turn_id=8, turn_end=True), state.snapshot(include_token_history=True)
    )
    assert body is not None and body.meta.last_chunk is False
    tail = tts2code2wav_async_chunk(
        manager, _duplex_delta(turn_id=8, turn_end=True), state.snapshot(include_token_history=True), True
    )
    assert tail is not None and tail.meta.last_chunk is True
    assert _codes(body)[3:] + _codes(tail)[3:] == list(range(30))


@pytest.mark.parametrize("lookahead", [1, 2, 4])
def test_mrv2_async_lookahead_cannot_reopen_a_completed_condition(lookahead):
    manager = _manager()
    requests = [_request("a"), _request("b")]
    for request in requests:
        request.sampling_params = SimpleNamespace(stop_token_ids=[6561])

    def output(request, seq, codes, *, eos=False, turn_end=False):
        payload = _duplex_delta(*codes, text=f"condition-{seq}", turn_end=turn_end)
        payload["meta"]["streaming_condition_seq"] = torch.tensor(seq)
        request.sampled_token_ids = [6561] if eos else []
        return tts2code2wav_async_chunk(manager, payload, request, False)

    for request in requests:
        first = output(request, 0, range(25), eos=True)
        assert first is not None and first.meta.tts_is_last_chunk
        assert _codes(first)[3:] == list(range(25))
    for request in reversed(requests):
        for _ in range(lookahead):
            assert output(request, 0, [999], eos=True) is None
        assert output(request, 1, range(25, 35)) is None
        # Even after a new condition starts, an old snapshot must not close it
        # or contaminate its queued codes/text.
        assert output(request, 0, [999], eos=True) is None
        last = output(request, 1, range(35, 50), eos=True, turn_end=True)
        assert last is not None and last.meta.last_chunk
        assert _codes(last)[3:] == list(range(25, 50))
        assert last.meta.chunk_seq == 1
        assert output(request, 1, [999], eos=True, turn_end=True) is None


@pytest.mark.parametrize("request_terminal", [False, True])
def test_mrv2_duplex_turn_end_keeps_live_code2wav_stream_open(request_terminal: bool) -> None:
    # A turn end must not close the Code2Wav stream of a live resumable request,
    # or the next turn's chunks are never received.
    from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import (
        _OmniConnectorPayloadTransportMixin as OmniConnectorPayloadTransport,
    )
    from vllm_omni.worker_v2.omni_data_plane import _NativeRequestState

    state = _NativeRequestState(
        request_id="native",
        external_req_id="native",
        prompt_token_ids=[0] * 3,
        resumable=True,
        sampling_params=SimpleNamespace(stop_token_ids=[6561]),
    )
    state.accept_tokens([6561])
    state.finished = request_terminal
    request = state.snapshot(include_token_history=True, sampled_token_ids=None if request_terminal else [6561])
    payload = tts2code2wav_async_chunk(
        _manager(), _duplex_delta(*range(7), turn_end=True), request, request.is_finished()
    )

    assert payload is not None
    assert payload.meta.last_chunk is True
    assert payload.meta.turn_end is True
    assert payload.meta.is_segment_finished.item() is True
    assert payload.meta.finished.item() is request_terminal
    metadata = OmniConnectorPayloadTransport._extract_scheduling_metadata({"meta": {"finished": payload.meta.finished}})
    assert metadata.get("input_terminal", False) is request_terminal


@pytest.mark.parametrize("last_valid", [False, True])
def test_full_payload_accumulates_codec_validity_per_frame(last_valid):
    """A final invalid/EOS row must not invalidate the whole utterance."""
    from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import (
        _OmniConnectorPayloadTransportMixin,
    )

    transport = _OmniConnectorPayloadTransportMixin()
    transport._pending_full_payload_send = {}
    transport._full_payload_replace_keys_cached = frozenset()
    request = _request("req")
    for code, valid in [(10, True), (0, False), (11, True), (12, last_valid)]:
        transport.accumulate_full_payload_output(
            "req",
            {"codes.audio": torch.tensor([[code]]), "meta.codec_frame_valid": torch.tensor([valid])},
            request,
        )
    full, _ = transport._materialize_full_payload_entry(transport._pending_full_payload_send["req"])
    payload = tts2code2wav_full_payload(_manager(), full, request)
    assert _codes(payload) == [4218, 4218, 4218, 10, 11] + ([12] if last_valid else [])
