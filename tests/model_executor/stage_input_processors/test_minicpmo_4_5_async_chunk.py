# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.request import RequestStatus

from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    tts2code2wav_async_chunk,
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


def test_first_chunk_forwards_reference_voice_and_duplex_identity() -> None:
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

    payload = tts2code2wav_async_chunk(manager, _delta(*range(7)), request, True)

    assert payload is not None
    assert payload.codes.ref.tolist() == pytest.approx([0.1, -0.1])
    assert payload.meta.ref_audio_sr == 16000
    assert payload.meta.native_duplex_segment_text == "hello"
    assert payload.meta.duplex_epoch == 3
    assert payload.meta.duplex_turn_id == 7
    assert payload.meta.segment_end is True
    assert payload.meta.turn_end is True


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
