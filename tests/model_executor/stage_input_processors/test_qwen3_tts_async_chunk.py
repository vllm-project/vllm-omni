# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.qwen3_tts import talker2code2wav_async_chunk

_FRAME = [1, 2, 3, 4]  # 4-codebook frame
_Q = len(_FRAME)  # num quantizers


def _req(rid: str, *, finished: bool, initial_codec_chunk_frames: int | None = None):
    ai = None
    if initial_codec_chunk_frames is not None:
        entry = SimpleNamespace(list_data=[initial_codec_chunk_frames])
        ai = SimpleNamespace(entries={"initial_codec_chunk_frames": entry})
    return SimpleNamespace(
        external_req_id=rid,
        is_finished=lambda: finished,
        additional_information=ai,
    )


def _tm(*, chunk_frames=25, left_context=25, initial_chunk=0):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context,
                    "initial_codec_chunk_frames": initial_chunk,
                }
            }
        ),
    )


def _call(tm, rid, *, n_frames, put_req=0, finished=False, req_ic=None):
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(n_frames)]
    tm.put_req_chunk[rid] = put_req
    return talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"audio_codes": torch.zeros((0,))},
        request=_req(rid, finished=finished, initial_codec_chunk_frames=req_ic),
        is_finished=finished,
    )


def test_does_not_emit_empty_chunk_when_not_finished():
    tm = _tm()
    request = _req("rid-empty", finished=False)

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"audio_codes": torch.zeros((0,))},
        request=request,
    )

    assert payload is None


def test_flushes_tail_when_finished_without_pooler_output():
    tm = _tm()
    rid = "rid-tail"
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(24)]
    request = _req(rid, finished=True)

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=request,
    )

    assert payload is not None
    assert payload["finished"].item() is True
    assert len(payload["code_predictor_codes"]) == _Q * 24


def test_emits_eof_marker_when_finished_with_no_frames():
    tm = _tm()
    request = _req("rid-eof", finished=True)

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=request,
    )

    assert payload == {
        "code_predictor_codes": [],
        "finished": torch.tensor(True, dtype=torch.bool),
    }


_CASES = [
    # Normal path (initial=0): emit at chunk_size boundaries
    ((25, 25, 0), (24, 0, False), None),
    ((25, 25, 0), (25, 0, False), (0, 25)),
    # Initial-chunk phase: hold, first emit, second emit
    ((25, 25, 10), (9, 0, False), None),
    ((25, 25, 10), (10, 0, False), (0, 10)),
    ((25, 25, 10), (20, 1, False), (10, 20)),
    # Non-divisible: holds at chunk boundary
    ((25, 25, 12), (25, 2, False), None),
    # Normal phase: offset by initial_coverage (chunk//initial * initial)
    ((25, 25, 10), (45, 2, False), (20, 45)),
    # Second normal emit (offset must stay stable)
    ((25, 25, 10), (70, 3, False), (25, 50)),
    # initial >= chunk clamps to chunk_size (behaves as normal)
    ((25, 25, 30), (25, 0, False), (0, 25)),
    # finished=True flushes IC tail
    ((25, 25, 10), (5, 0, True), (0, 5)),
    # finished=True flushes non-divisible IC residual
    ((25, 25, 12), (25, 2, True), (24, 25)),
    # finished=True flushes normal phase tail
    ((25, 25, 10), (30, 2, True), (20, 30)),
]


@pytest.mark.parametrize("config, state, expected", _CASES)
def test_streaming_decoding_with_variable_initial(config, state, expected):
    chunk_frames, left_context, initial_chunk = config
    n_frames, put_req, finished = state

    tm = _tm(chunk_frames=chunk_frames, left_context=left_context, initial_chunk=initial_chunk)
    payload = _call(tm, "r", n_frames=n_frames, put_req=put_req, finished=finished)

    if expected is None:
        assert payload is None
    else:
        exp_ctx, exp_window = expected
        assert payload is not None
        assert payload["left_context_size"] == exp_ctx
        assert len(payload["code_predictor_codes"]) == _Q * exp_window


def test_per_request_override_activates_initial_phase():
    tm = _tm(initial_chunk=0)
    payload = _call(tm, "r-override", n_frames=10, req_ic=10)
    assert payload is not None
    assert payload["left_context_size"] == 0
    assert len(payload["code_predictor_codes"]) == _Q * 10


def test_per_request_override_wins_over_stage_config():
    tm = _tm(initial_chunk=5)
    payload = _call(tm, "r-override2", n_frames=10, put_req=0, req_ic=15)
    assert payload is None
