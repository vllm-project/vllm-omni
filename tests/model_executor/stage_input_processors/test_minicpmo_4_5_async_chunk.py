# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.request import RequestStatus

from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    tts2code2wav_async_chunk,
    tts2code2wav_full_payload,
    tts2code2wav_token_only,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _manager(
    *,
    early_setup: bool | None = False,
    chunk_frames: int = 25,
    initial_chunk_frames: int | None = 25,
):
    extra = {"codec_chunk_frames": chunk_frames, "codec_left_context_frames": 3}
    if early_setup is not None:
        extra["code2wav_early_setup"] = early_setup
    if initial_chunk_frames is not None:
        extra["initial_codec_chunk_frames"] = initial_chunk_frames
    return SimpleNamespace(
        connector=SimpleNamespace(config={"extra": extra}),
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


@pytest.mark.parametrize("initial_chunk_frames", [8, 12, 16, 25])
def test_first_chunk_uses_configured_threshold_then_returns_to_steady_size(
    initial_chunk_frames: int,
) -> None:
    manager = _manager(initial_chunk_frames=initial_chunk_frames)
    request = _request("req")

    first = tts2code2wav_async_chunk(
        manager,
        _delta(*range(initial_chunk_frames)),
        request,
        False,
    )
    waiting = tts2code2wav_async_chunk(
        manager,
        _delta(*range(initial_chunk_frames, initial_chunk_frames + 24)),
        request,
        False,
    )
    steady = tts2code2wav_async_chunk(
        manager,
        _delta(initial_chunk_frames + 24),
        request,
        False,
    )

    assert first is not None
    assert first.meta.codec_chunk_frames == initial_chunk_frames
    assert first.meta.chunk_seq == 0
    assert waiting is None
    assert steady is not None
    assert steady.meta.codec_chunk_frames == 25
    assert steady.meta.chunk_seq == 1
    assert _codes(steady)[:3] == list(range(initial_chunk_frames - 3, initial_chunk_frames))


def test_initial_chunk_model_default_preserves_25_when_config_is_absent(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES", raising=False)
    manager = _manager(initial_chunk_frames=None)
    request = _request("req")

    assert tts2code2wav_async_chunk(manager, _delta(*range(24)), request, False) is None
    first = tts2code2wav_async_chunk(manager, _delta(24), request, False)

    assert first is not None
    assert first.meta.codec_chunk_frames == 25


def test_initial_chunk_explicit_config_wins_over_environment(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES", "8")
    manager = _manager(initial_chunk_frames=12)

    assert tts2code2wav_async_chunk(manager, _delta(*range(11)), _request("req"), False) is None
    first = tts2code2wav_async_chunk(manager, _delta(11), _request("req"), False)

    assert first is not None
    assert first.meta.codec_chunk_frames == 12


def test_initial_chunk_environment_selects_candidate_without_yaml(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES", "8")
    manager = _manager(initial_chunk_frames=None)

    first = tts2code2wav_async_chunk(
        manager,
        _delta(*range(8)),
        _request("req"),
        False,
    )

    assert first is not None
    assert first.meta.codec_chunk_frames == 8


def test_initial_chunk_environment_zero_restores_steady_threshold(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES", "0")
    manager = _manager(initial_chunk_frames=None)

    assert tts2code2wav_async_chunk(manager, _delta(*range(24)), _request("req"), False) is None


def test_initial_chunk_clamps_to_steady_size() -> None:
    manager = _manager(chunk_frames=25, initial_chunk_frames=40)

    assert tts2code2wav_async_chunk(manager, _delta(*range(24)), _request("req"), False) is None
    first = tts2code2wav_async_chunk(manager, _delta(24), _request("req"), False)

    assert first is not None
    assert first.meta.codec_chunk_frames == 25


@pytest.mark.parametrize("source", ["config", "environment"])
def test_initial_chunk_rejects_invalid_values(monkeypatch, source: str) -> None:
    if source == "config":
        manager = _manager(initial_chunk_frames=0)
    else:
        monkeypatch.setenv("VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES", "invalid")
        manager = _manager(initial_chunk_frames=None)

    with pytest.raises(ValueError, match="initial_codec_chunk_frames|INITIAL_CODEC_CHUNK_FRAMES"):
        tts2code2wav_async_chunk(manager, _delta(1), _request("req"), False)


def test_first_codec_timeline_event_is_emitted_once_per_stream(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(
        "vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni.emit_ultra_timeline_event",
        lambda event, **metadata: events.append((event, metadata)),
    )
    manager = _manager()
    request = _request("req")

    assert tts2code2wav_async_chunk(manager, _delta(1), request, False) is None
    assert tts2code2wav_async_chunk(manager, _delta(2), request, False) is None

    assert [event for event, _ in events] == ["first_codec_token"]
    assert events[0][1]["request_id"] == "req"
    assert events[0][1]["stage"] == 1
    assert events[0][1]["shape"] == (1,)


def test_first_codec_token_sends_init_without_advancing_codec_stream(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_EARLY_CODE2WAV_SETUP", raising=False)
    manager = _manager(early_setup=None)
    request = _request("req")
    request.additional_information = {
        "codes": {"ref": [0.1, -0.1]},
        "meta": {"ref_audio_sr": 16000},
    }

    init = tts2code2wav_async_chunk(manager, _delta(0), request, False)
    waiting = tts2code2wav_async_chunk(manager, _delta(1), request, False)
    first_audio = tts2code2wav_async_chunk(manager, _delta(*range(2, 25)), request, False)

    assert init is not None
    assert init.meta.init_only is True
    assert init.meta.chunk_seq == 0
    assert init.meta.code_flat_numel == 0
    assert init.meta.codec_chunk_frames == 0
    assert init.meta.last_chunk is False
    assert _codes(init) == [4218]
    assert init.codes.ref.tolist() == pytest.approx([0.1, -0.1])
    assert init.meta.ref_audio_sr == 16000
    assert waiting is None
    assert first_audio is not None
    assert first_audio.meta.init_only is False
    assert first_audio.meta.chunk_seq == 0
    assert first_audio.codes.ref is None
    assert _codes(first_audio) == [4218, 4218, 4218, *range(25)]


def test_early_setup_does_not_consume_small_first_chunk_offset(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_MINICPMO45_EARLY_CODE2WAV_SETUP", raising=False)
    manager = _manager(early_setup=None, initial_chunk_frames=8)
    request = _request("req")

    init = tts2code2wav_async_chunk(manager, _delta(0), request, False)
    first_audio = tts2code2wav_async_chunk(manager, _delta(*range(1, 8)), request, False)

    assert init is not None
    assert init.meta.init_only is True
    assert init.meta.chunk_seq == 0
    assert first_audio is not None
    assert first_audio.meta.init_only is False
    assert first_audio.meta.chunk_seq == 0
    assert first_audio.meta.codec_chunk_frames == 8
    assert _codes(first_audio) == [4218, 4218, 4218, *range(8)]


def test_explicit_config_disables_early_setup_even_when_environment_enables(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_EARLY_CODE2WAV_SETUP", "1")

    payload = tts2code2wav_async_chunk(
        _manager(early_setup=False),
        _delta(1),
        _request("req"),
        False,
    )

    assert payload is None


def test_environment_can_disable_default_early_setup(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_MINICPMO45_EARLY_CODE2WAV_SETUP", "0")

    payload = tts2code2wav_async_chunk(
        _manager(early_setup=None),
        _delta(1),
        _request("req"),
        False,
    )

    assert payload is None


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


def test_first_chunk_forwards_reference_voice_and_duplex_identity() -> None:
    manager = _manager()
    request = _request("req")
    request.additional_information = {
        "codes": {"ref": [0.1, -0.1]},
        "meta": {"ref_audio_sr": 16000},
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


def test_full_payload_forwards_all_codes_and_request_metadata() -> None:
    manager = _manager()
    request = _request("req")
    request.additional_information = {
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


def test_empty_final_releases_wait_gate_once() -> None:
    manager = _manager()
    request = _request("req")

    final = tts2code2wav_async_chunk(manager, None, request, True)
    duplicate = tts2code2wav_async_chunk(manager, None, request, True)

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
