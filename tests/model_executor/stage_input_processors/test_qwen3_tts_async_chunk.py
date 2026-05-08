# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
    max_ic_for_chunk_size,
)
from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
    talker2code2wav,
    talker2code2wav_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_FRAME = [1, 2, 3, 4]
_Q = len(_FRAME)


def _req(rid, *, finished, initial_codec_chunk_frames=None):
    ai = None
    if initial_codec_chunk_frames is not None:
        entry = SimpleNamespace(list_data=[initial_codec_chunk_frames])
        ai = SimpleNamespace(entries={"initial_codec_chunk_frames": entry})
    return SimpleNamespace(
        external_req_id=rid,
        is_finished=lambda: finished,
        additional_information=ai,
    )


def _tm(*, chunk_frames=25, left_context=25, max_num_seqs=1):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        scheduler_max_num_seqs=max_num_seqs,
        put_req_chunk=defaultdict(int),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context,
                }
            }
        ),
    )


def _call(tm, rid, *, n_frames, finished=False, req_ic=None):
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(n_frames)]
    return talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"codes": {"audio": torch.zeros((0,))}},
        request=_req(rid, finished=finished, initial_codec_chunk_frames=req_ic),
        is_finished=finished,
    )


def test_empty_returns_none():
    tm = _tm()
    p = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"codes": {"audio": torch.zeros((0,))}},
        request=_req("r", finished=False),
    )
    assert p is None


def test_eof_marker_when_finished_empty():
    tm = _tm()
    p = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=_req("r", finished=True),
        is_finished=True,
    )
    assert p["codes"] == {"audio": []}
    assert p["meta"]["finished"].item() is True


def test_flush_on_finish():
    tm = _tm()
    tm.code_prompt_token_ids["r"] = [_FRAME[:] for _ in range(24)]
    p = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=_req("r", finished=True),
        is_finished=True,
    )
    assert p is not None
    assert p["meta"]["finished"].item() is True
    assert len(p["codes"]["audio"]) == _Q * 24


_CASES = [
    # ── IC boundary rule ──────────────────────────────────────────────
    # IC phase: length <= chunk_size  (uses <=, consistent with fish_speech)
    # IC emits fill the entire first chunk_size worth of frames, so the
    # normal phase always starts at a clean chunk boundary.
    # initial_coverage = (chunk_size // initial_chunk_size) * initial_chunk_size
    #
    # Dynamic IC=16, cs=25, initial_coverage=16
    # IC does NOT evenly divide cs, so initial_coverage < cs.
    # IC emits at 16; frames 17-25 remain in IC phase but 25%16!=0 -> hold.
    # Normal phase: adjusted = length - 16, emit when adjusted % 25 == 0.
    ((25, 25, 0), 24, False, None),  # IC: 24<=25, 24%16!=0 -> hold
    ((25, 25, 0), 25, False, None),  # IC: 25<=25, 25%16!=0 -> hold
    ((25, 25, 0), 41, False, (16, 41)),  # normal: adjusted=25, 25%25==0 -> emit, lc=16
    #
    # Per-request IC=10, cs=25, initial_coverage=20
    # IC does NOT evenly divide cs; IC emits at 10, 20.
    # Frames 21-25 are still IC phase but 21..25 % 10 != 0 -> hold.
    ((25, 25, 10), 9, False, None),  # IC: 9%10!=0 -> hold
    ((25, 25, 10), 10, False, (0, 10)),  # IC: 10%10==0 -> emit, lc=0
    ((25, 25, 10), 25, False, None),  # IC: 25<=25, 25%10!=0 -> hold
    ((25, 25, 10), 45, False, (20, 45)),  # normal: adjusted=25, 25%25==0 -> emit, lc=20
    ((25, 25, 10), 5, True, (0, 5)),  # finished flushes IC tail
    ((25, 25, 10), 33, True, (20, 33)),  # finished flushes normal tail
    #
    # IC=8, cs=16: IC evenly divides chunk_size (edge case)
    # initial_coverage = (16//8)*8 = 16 == chunk_size.
    # IC fills the entire first chunk: emits at 8 and 16.
    # Normal phase starts at frame 17; first normal emit at 16+16=32.
    ((16, 25, 8), 8, False, (0, 8)),  # IC: 8%8==0 -> emit, lc=0
    ((16, 25, 8), 16, False, (8, 16)),  # IC: 16<=16, 16%8==0 -> emit, lc=8
    ((16, 25, 8), 24, False, None),  # normal: adjusted=8, 8%16!=0 -> hold
    ((16, 25, 8), 32, False, (16, 32)),  # normal: adjusted=16, 16%16==0 -> first emit, lc=16
    #
    # IC=5, cs=25: IC evenly divides chunk_size
    # initial_coverage = (25//5)*5 = 25 == chunk_size.
    # IC fills the entire first chunk: emits at 5, 10, 15, 20, 25.
    # Normal phase starts at frame 26; first normal emit at 25+25=50.
    # Emit intervals: 5,5,5,5,5,25,25,... — smooth transition, no gap.
    ((25, 25, 5), 5, False, (0, 5)),  # IC: 5%5==0 -> emit, lc=0
    ((25, 25, 5), 12, False, None),  # IC: 12%5!=0 -> hold
    ((25, 25, 5), 25, False, (20, 25)),  # IC: 25<=25, 25%5==0 -> emit, lc=20
    ((25, 25, 5), 30, False, None),  # normal: adjusted=5, 5%25!=0 -> hold
    ((25, 25, 5), 50, False, (25, 50)),  # normal: adjusted=25, 25%25==0 -> first emit, lc=25
    #
    # Per-request override: IC=15 at n_frames=10 -> 10%15!=0 -> hold
    ((25, 25, 15), 10, False, None),
]


@pytest.mark.parametrize("config, n_frames, finished, expected", _CASES)
def test_streaming_phases(config, n_frames, finished, expected):
    chunk_frames, left_context, req_ic_val = config
    tm = _tm(chunk_frames=chunk_frames, left_context=left_context)
    req_ic = req_ic_val if req_ic_val > 0 else None
    payload = _call(tm, "r", n_frames=n_frames, finished=finished, req_ic=req_ic)

    if expected is None:
        assert payload is None
    else:
        exp_ctx, exp_window = expected
        assert payload is not None
        assert payload["meta"]["left_context_size"] == exp_ctx
        assert len(payload["codes"]["audio"]) == _Q * exp_window


def test_dynamic_ic_adapts_to_load():
    # chunk_size=25 -> max_ic=16, steps=[2,4,8,16]
    tm = _tm(max_num_seqs=8)

    # Low load (1/8) -> IC=2 -> emit at 2
    p1 = _call(tm, "r", n_frames=2)
    assert p1 is not None
    assert len(p1["codes"]["audio"]) == _Q * 2

    # High load: add 4 others -> active=5/8 -> IC=8 -> emit at 8
    for i in range(4):
        tm.code_prompt_token_ids[f"other-{i}"] = [[0]]
    p2 = _call(tm, "r", n_frames=8)
    assert p2 is not None
    assert len(p2["codes"]["audio"]) == _Q * 8

    # Requests past initial phase still count in load factor
    tm2 = _tm(max_num_seqs=4)
    for i in range(3):
        tm2.code_prompt_token_ids[f"long-{i}"] = [[0]] * 50  # well past cs=25
    # active=4/4=1.0 -> IC=16
    p3 = _call(tm2, "new", n_frames=16)
    assert p3 is not None
    assert len(p3["codes"]["audio"]) == _Q * 16


def test_ic_load_change_mid_request():
    """IC is cached per request; a load spike only affects new requests."""
    tm = _tm(chunk_frames=25, left_context=25, max_num_seqs=8)

    # Low load -> IC=2 (cached for "r"), emit at frame 2
    p1 = _call(tm, "r", n_frames=2)
    assert p1 is not None

    # Spike load: 6 others running
    for i in range(6):
        tm.code_prompt_token_ids[f"other-{i}"] = [[0]] * 10

    # IC for "r" is still cached as 2.
    # initial_coverage = ((25-1)//2)*2 = 24, first normal emit at 24+25=49
    assert _call(tm, "r", n_frames=25) is None
    assert _call(tm, "r", n_frames=27) is None
    p3 = _call(tm, "r", n_frames=49)
    assert p3 is not None
    assert p3["meta"]["left_context_size"] == 24

    # A *new* request under high load gets IC=16 (not IC=2).
    # Frame 2 would emit under IC=2 but must hold under IC=16.
    assert _call(tm, "new_req", n_frames=2) is None
    p4 = _call(tm, "new_req", n_frames=16)
    assert p4 is not None


@pytest.mark.parametrize(
    "active,max_bs,max_ic,expected",
    [
        (0, 4, 32, 2),  # zero load -> min step
        (2, 4, 32, 8),  # mid load
        (4, 4, 32, 32),  # full load
        (10, 4, 16, 16),  # over capacity, capped
        (0, 4, 1, 1),  # max_ic below min step
        (0, 0, 16, 2),  # zero capacity edge case
    ],
)
def test_compute_dynamic_initial_chunk_size(active, max_bs, max_ic, expected):
    assert compute_dynamic_initial_chunk_size(active, max_bs, max_ic) == expected


@pytest.mark.parametrize(
    "chunk_size,expected",
    [
        (25, 16),
        (50, 32),
        (70, 64),
        (8, 4),
        (4, 2),
        (2, 1),
        (1, 1),
    ],
)
def test_max_ic_for_chunk_size(chunk_size, expected):
    assert max_ic_for_chunk_size(chunk_size) == expected


def test_first_streaming_chunk_prepends_ref_code_context():
    tm = _tm()
    rid = "r-ref"
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(10)]
    ref_code = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"codes": {"audio": torch.zeros((0,)), "ref": ref_code}},
        request=_req(rid, finished=False, initial_codec_chunk_frames=10),
        is_finished=False,
    )

    assert payload is not None
    assert payload["meta"]["left_context_size"] == 2
    assert len(payload["codes"]["audio"]) == _Q * 12


def test_ref_code_context_applies_to_all_streaming_chunks():
    """ref_code is prepended as decoder context on every chunk, not just the first."""
    tm = _tm()
    rid = "r-ref2"
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(20)]
    tm.put_req_chunk[rid] = 1
    ref_code = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)
    tm.request_payload[rid] = ref_code

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"codes": {"audio": torch.zeros((0,)), "ref": ref_code}},
        request=_req(rid, finished=False, initial_codec_chunk_frames=10),
        is_finished=False,
    )

    assert payload is not None
    # ref_code (2 frames) prepended as left context on second chunk too
    assert payload["meta"]["left_context_size"] == 10 + 2
    assert len(payload["codes"]["audio"]) == _Q * (20 + 2)


def test_ref_code_context_can_be_buffered_before_first_emit():
    tm = _tm()
    rid = "r-ref-buffered"
    ref_code = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)

    first_payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"codes": {"audio": torch.tensor([[1, 2, 3, 4]]), "ref": ref_code}},
        request=_req(rid, finished=False, initial_codec_chunk_frames=10),
        is_finished=False,
    )
    assert first_payload is None
    assert rid in tm.request_payload

    for _ in range(8):
        talker2code2wav_async_chunk(
            transfer_manager=tm,
            pooling_output={"codes": {"audio": torch.tensor([[1, 2, 3, 4]])}},
            request=_req(rid, finished=False, initial_codec_chunk_frames=10),
            is_finished=False,
        )

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"codes": {"audio": torch.tensor([[1, 2, 3, 4]])}},
        request=_req(rid, finished=False, initial_codec_chunk_frames=10),
        is_finished=False,
    )

    assert payload is not None
    # ref_code (2 frames) is kept (not popped) for subsequent chunks
    assert payload["meta"]["left_context_size"] == 2
    assert len(payload["codes"]["audio"]) == _Q * 12
    assert rid in tm.request_payload


def test_non_async_processor_prepends_ref_code_and_sets_trim_context():
    ref_code = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)
    audio_codes = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    output = SimpleNamespace(
        multimodal_output={"codes": {"audio": audio_codes, "ref": ref_code}},
        token_ids=list(range(3)),
        cumulative_token_ids=list(range(3)),
    )
    stage = SimpleNamespace(
        engine_outputs=[SimpleNamespace(outputs=[output], finished=True)],
    )

    prompts = talker2code2wav(stage.engine_outputs)

    assert len(prompts) == 1
    prompt = prompts[0]
    assert prompt["additional_information"] == {"meta": {"left_context_size": [2]}}
    assert prompt["prompt_token_ids"] == [
        9,
        8,
        1,
        5,
        9,
        8,
        2,
        6,
        9,
        8,
        3,
        7,
        9,
        8,
        4,
        8,
    ]


def test_non_async_processor_filters_out_of_range_codec_values():
    """Frames with values >= codebook_size (e.g. stop_token_id=2150) are filtered."""
    ref_code = torch.tensor([[9, 9, 9, 9]], dtype=torch.long)
    audio_codes = torch.tensor(
        [
            [0, 0, 0, 0],  # zero-padded (filtered)
            [1, 2, 3, 4],  # valid
            [2150, 0, 0, 0],  # stop token (filtered)
            [5, 6, 7, 8],  # valid
        ],
        dtype=torch.long,
    )
    output = SimpleNamespace(
        multimodal_output={"codes": {"audio": audio_codes, "ref": ref_code}},
        token_ids=list(range(4)),
        cumulative_token_ids=list(range(4)),
    )
    stage = SimpleNamespace(
        engine_outputs=[SimpleNamespace(outputs=[output], finished=True)],
    )

    prompts = talker2code2wav(stage.engine_outputs)

    assert len(prompts) == 1
    prompt = prompts[0]
    # Only ref_code (1 frame) + 2 valid frames = 3 frames * 4 quantizers = 12 codes
    assert len(prompt["prompt_token_ids"]) == 4 * 3
    assert prompt["additional_information"] == {"meta": {"left_context_size": [1]}}
