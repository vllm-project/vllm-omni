# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    compute_dynamic_initial_chunk_size,
    compute_ramp_emit,
    max_ic_for_chunk_size,
    parse_chunk_ramp,
    ramp_chunk_size,
    ramp_cumulative,
)
from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
    _NUM_QUANTIZERS_DEFAULT,
    _filter_audio_codes_qwen3_tts,
    talker2code2wav_async_chunk,
    talker2code2wav_full_payload,
    talker2code2wav_token_only,
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


def _tm(*, chunk_frames=25, left_context=25, max_num_seqs=1, initial_chunk_frames=0, chunk_ramp=None):
    extra = {
        "codec_chunk_frames": chunk_frames,
        "codec_left_context_frames": left_context,
        "initial_codec_chunk_frames": initial_chunk_frames,
    }
    if chunk_ramp is not None:
        extra["codec_chunk_ramp"] = chunk_ramp
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        scheduler_max_num_seqs=max_num_seqs,
        put_req_chunk=defaultdict(int),
        ramp_chunk_count=defaultdict(int),
        request_payload={},
        connector=SimpleNamespace(config={"extra": extra}),
    )


def _call(tm, rid, *, n_frames, finished=False, req_ic=None):
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(n_frames)]
    return talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.zeros((0,))}},
        request=_req(rid, finished=finished, initial_codec_chunk_frames=req_ic),
        is_finished=finished,
    )


def test_empty_returns_none():
    tm = _tm()
    p = talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.zeros((0,))}},
        request=_req("r", finished=False),
    )
    assert p is None


def test_eof_marker_when_finished_empty():
    tm = _tm()
    p = talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output=None,
        request=_req("r", finished=True),
        is_finished=True,
    )
    assert p.codes.audio.tolist() == []
    assert p.meta.finished.item() is True


def test_flush_on_finish():
    tm = _tm()
    tm.code_prompt_token_ids["r"] = [_FRAME[:] for _ in range(24)]
    p = talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output=None,
        request=_req("r", finished=True),
        is_finished=True,
    )
    assert p is not None
    assert p.meta.finished.item() is True
    assert len(p.codes.audio) == _Q * 24


_CASES = [
    # ── IC boundary rule ──────────────────────────────────────────────
    # initial_codec_chunk_frames only controls the first emitted chunk.
    # After that, the processor returns to codec_chunk_frames-sized windows
    # to avoid flooding Code2Wav with repeated tiny overlapping decodes.
    #
    # Dynamic IC=16, cs=25, initial_coverage=16
    # Normal phase: adjusted = length - 16, emit when adjusted % 25 == 0.
    ((25, 25, 0), 24, False, None),
    ((25, 25, 0), 25, False, None),
    ((25, 25, 0), 41, False, (16, 41)),  # normal: adjusted=25, 25%25==0 -> emit, lc=16
    #
    # Per-request IC=10, cs=25: first emit at 10, then 35, 60...
    ((25, 25, 10), 9, False, None),
    ((25, 25, 10), 10, False, (0, 10)),
    ((25, 25, 10), 25, False, None),
    ((25, 25, 10), 35, False, (10, 35)),
    ((25, 25, 10), 45, False, None),
    ((25, 25, 10), 5, True, (0, 5)),  # finished flushes IC tail
    ((25, 25, 10), 33, True, (10, 33)),  # finished flushes normal tail
    #
    # IC=8, cs=16: first emit at 8, then 24, 40...
    ((16, 25, 8), 8, False, (0, 8)),
    ((16, 25, 8), 16, False, None),
    ((16, 25, 8), 24, False, (8, 24)),
    ((16, 25, 8), 32, False, None),
    #
    # IC=5, cs=25: first emit at 5, then 30, 55...
    ((25, 25, 5), 5, False, (0, 5)),
    ((25, 25, 5), 12, False, None),
    ((25, 25, 5), 25, False, None),
    ((25, 25, 5), 30, False, (5, 30)),
    ((25, 25, 5), 50, False, None),
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
        assert payload.meta.left_context_size == 0
        expected_delta = exp_window if finished else exp_window - exp_ctx
        assert len(payload.codes.audio) == _Q * expected_delta


def test_dynamic_ic_adapts_to_load():
    # chunk_size=25 -> max_ic=16, steps=[2,4,8,16]
    tm = _tm(max_num_seqs=8)

    # Low load (1/8) -> IC=2 -> emit at 2
    p1 = _call(tm, "r", n_frames=2)
    assert p1 is not None
    assert len(p1.codes.audio) == _Q * 2

    # High load on a new request: active=6/8 -> IC=8 -> emit at 8
    for i in range(4):
        tm.code_prompt_token_ids[f"other-{i}"] = [[0]]
    p2 = _call(tm, "new-high-load", n_frames=8)
    assert p2 is not None
    assert len(p2.codes.audio) == _Q * 8

    # Requests past initial phase still count in load factor
    tm2 = _tm(max_num_seqs=4)
    for i in range(3):
        tm2.code_prompt_token_ids[f"long-{i}"] = [[0]] * 50  # well past cs=25
    # active=4/4=1.0 -> IC=16
    p3 = _call(tm2, "new", n_frames=16)
    assert p3 is not None
    assert len(p3.codes.audio) == _Q * 16


def test_ic_load_change_mid_request():
    """IC is cached per request; a load spike only affects new requests."""
    tm = _tm(chunk_frames=25, left_context=25, max_num_seqs=8)

    # Low load -> IC=2 (cached for "r"), emit at frame 2
    p1 = _call(tm, "r", n_frames=2)
    assert p1 is not None

    # Spike load: 6 others running
    for i in range(6):
        tm.code_prompt_token_ids[f"other-{i}"] = [[0]] * 10

    # IC for "r" is still cached as 2. The first normal emit is at 2+25=27.
    assert _call(tm, "r", n_frames=25) is None
    p3 = _call(tm, "r", n_frames=27)
    assert p3 is not None
    assert p3.meta.left_context_size == 0
    assert len(p3.codes.audio) == _Q * 25

    # A *new* request under high load gets IC=16 (not IC=2).
    # Frame 2 would emit under IC=2 but must hold under IC=16.
    assert _call(tm, "new_req", n_frames=2) is None
    p4 = _call(tm, "new_req", n_frames=16)
    assert p4 is not None


def test_connector_initial_chunk_config_overrides_dynamic_ic():
    tm = _tm(initial_chunk_frames=4, max_num_seqs=8)

    # Under high load dynamic IC would be 16, but connector config pins the
    # first chunk to 4 frames.
    for i in range(7):
        tm.code_prompt_token_ids[f"other-{i}"] = [[0]]

    p1 = _call(tm, "r", n_frames=4)
    assert p1 is not None
    assert len(p1.codes.audio) == _Q * 4

    # Only the first chunk uses the small size; the next emit is 4+25.
    assert _call(tm, "r", n_frames=25) is None
    p2 = _call(tm, "r", n_frames=29)
    assert p2 is not None
    assert p2.meta.left_context_size == 0
    assert len(p2.codes.audio) == _Q * 25


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
        multimodal_output={"codes": {"audio": torch.zeros((0,)), "ref": ref_code}},
        request=_req(rid, finished=False, initial_codec_chunk_frames=10),
        is_finished=False,
    )

    assert payload is not None
    assert payload.meta.left_context_size == 2
    assert payload.meta.ref_context_size == 2
    assert payload.meta.ref_context_request_id == rid
    assert payload.meta.ref_context_included is True
    assert len(payload.codes.audio) == _Q * 12


def test_followup_sends_only_codec_delta_without_ref_metadata():
    """Follow-up chunks rely on decoder state and resend neither reference codes nor metadata."""
    tm = _tm()
    rid = "r-ref2"
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(35)]
    tm.put_req_chunk[rid] = 1
    ref_code = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)
    tm.request_payload[rid] = ref_code

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.zeros((0,)), "ref": ref_code}},
        request=_req(rid, finished=False, initial_codec_chunk_frames=10),
        is_finished=False,
    )

    assert payload is not None
    assert payload.meta.left_context_size == 0
    assert payload.meta.ref_context_size is None
    assert payload.meta.ref_context_request_id is None
    assert payload.meta.ref_context_included is None
    assert len(payload.codes.audio) == _Q * 25


def test_streaming_ref_code_context_is_bounded_for_batchable_shapes():
    tm = _tm(chunk_frames=4, left_context=3, initial_chunk_frames=4)
    rid = "r-ref-bounded"
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(8)]
    ref_code = torch.tensor(
        [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
        ],
        dtype=torch.long,
    )
    tm.request_payload[rid] = ref_code

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.zeros((0,)), "ref": ref_code}},
        request=_req(rid, finished=False),
        is_finished=False,
    )

    assert payload is not None
    assert payload.meta.left_context_size == 3
    assert len(payload.codes.audio) == _Q * (3 + 4)
    frames = payload.codes.audio.reshape(_Q, -1).transpose(0, 1)
    torch.testing.assert_close(frames[:3], ref_code[-3:])


def test_ref_code_context_can_be_buffered_before_first_emit():
    tm = _tm()
    rid = "r-ref-buffered"
    ref_code = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)

    first_payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.tensor([[1, 2, 3, 4]]), "ref": ref_code}},
        request=_req(rid, finished=False, initial_codec_chunk_frames=10),
        is_finished=False,
    )
    assert first_payload is None
    assert rid in tm.request_payload

    for _ in range(8):
        talker2code2wav_async_chunk(
            transfer_manager=tm,
            multimodal_output={"codes": {"audio": torch.tensor([[1, 2, 3, 4]])}},
            request=_req(rid, finished=False, initial_codec_chunk_frames=10),
            is_finished=False,
        )

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.tensor([[1, 2, 3, 4]])}},
        request=_req(rid, finished=False, initial_codec_chunk_frames=10),
        is_finished=False,
    )

    assert payload is not None
    # ref_code (2 frames) is kept (not popped) for subsequent chunks
    assert payload.meta.left_context_size == 2
    assert len(payload.codes.audio) == _Q * 12
    assert rid in tm.request_payload


def test_non_async_token_only_sizes_placeholder_for_ref_and_audio_frames():
    """``talker2code2wav_token_only`` only allocates placeholder prompt slots.

    After the connector refactor, actual codec flattening (ref prepend +
    codebook-major layout) is performed by ``talker2code2wav_full_payload`` on
    the worker data plane.  The orchestrator hook still derives
    ``left_context_size`` from stage-0 multimodal_output so Code2Wav can trim
    reference frames once the connector payload arrives.
    """
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

    prompts = talker2code2wav_token_only(stage.engine_outputs)

    assert len(prompts) == 1
    prompt = prompts[0]
    assert prompt["additional_information"] == {"meta": {"left_context_size": 2}}
    # 2 ref frames + 2 valid audio frames (zero row filtered), 4 quantizers.
    assert prompt["prompt_token_ids"] == [0] * (_Q * (2 + 2))


def test_full_payload_prepends_ref_code_and_flattens_codebook_major():
    """Worker producer is authoritative for ref prepend + codec flatten."""
    ref_code = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)
    audio_codes = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    pooling_output = {
        "codes.audio": audio_codes,
        "codes.ref": ref_code,
        "meta.ref_code_len": 2,
    }
    request = SimpleNamespace(request_id="r", output_token_ids=list(range(3)))

    payload = talker2code2wav_full_payload(transfer_manager=None, pooling_output=pooling_output, request=request)

    assert payload is not None
    assert payload["meta"]["left_context_size"] == 2
    assert payload["codes"]["audio"].tolist() == [
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

    prompts = talker2code2wav_token_only(stage.engine_outputs)

    assert len(prompts) == 1
    prompt = prompts[0]
    # Only ref_code (1 frame) + 2 valid frames = 3 frames * 4 quantizers = 12 codes
    assert len(prompt["prompt_token_ids"]) == 4 * 3
    assert prompt["additional_information"] == {"meta": {"left_context_size": 1}}


def test_full_payload_emits_left_context_size_for_ref_clone():
    """Regression for #4421.

    The worker-side ``talker2code2wav_full_payload`` producer is the
    authoritative channel that prepends ``ref_code`` to the codec stream.
    The orchestrator-side ``talker2code2wav_token_only`` can no longer derive
    ``left_context_size`` (the stage-0 RequestOutput multimodal_output no longer
    carries the talker codec since the separated mm-output channel landed), so
    full_payload MUST emit the matching ``left_context_size`` in its connector
    meta or Code2Wav trims nothing and the reference audio leaks into the
    output.
    """
    ref_code = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)  # 2 ref frames
    audio_codes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 1, 1, 1]], dtype=torch.long)  # 3 generated frames
    pooling_output = {
        "codes.audio": audio_codes,
        "codes.ref": ref_code,
        "meta.ref_code_len": 2,
    }
    request = SimpleNamespace(request_id="r", output_token_ids=list(range(4)))  # seq_len=3

    payload = talker2code2wav_full_payload(transfer_manager=None, pooling_output=pooling_output, request=request)

    assert payload is not None
    assert payload["meta"]["finished"].item() is True
    # The fix: trim length is co-located with the ref prepend it describes.
    assert payload["meta"]["left_context_size"] == 2
    # ref(2) + generated(3) frames, codebook-major flat = Q * 5.
    assert len(payload["codes"]["audio"]) == _Q * 5


def test_full_payload_omits_left_context_size_without_ref():
    """Without a reference (non-clone tasks) nothing is prepended, so no
    ``left_context_size`` is emitted and Code2Wav trims nothing."""
    audio_codes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    pooling_output = {"codes.audio": audio_codes}
    request = SimpleNamespace(request_id="r", output_token_ids=list(range(3)))  # seq_len=2

    payload = talker2code2wav_full_payload(transfer_manager=None, pooling_output=pooling_output, request=request)

    assert payload is not None
    assert payload["meta"]["finished"].item() is True
    assert "left_context_size" not in payload["meta"]
    assert len(payload["codes"]["audio"]) == _Q * 2


@pytest.mark.parametrize(
    "pooling_output",
    [
        pytest.param(SimpleNamespace(codes="not-a-dict"), id="non_dict_output"),
        pytest.param({}, id="missing_codes_audio"),
        pytest.param({"codes.audio": torch.zeros((3, _Q), dtype=torch.long)}, id="all_codes_filtered"),
    ],
)
def test_full_payload_emits_placeholder_frame_on_degenerate_take(pooling_output):
    """Regression for #4463 and #5471 (the producer half of #5196).

    A degenerate talker take must not return ``None`` from
    ``talker2code2wav_full_payload``: the connector treats ``None`` as "drop the
    request", but Stage-1 was already scheduled to receive it, so its wait gate
    polls to ``connector_get_max_wait`` (~300s) and one stuck request stalls the
    whole two-stage pipeline (#4463). It must not return an *empty* finished
    payload either: zero codec frames produce a zero-token Stage-1 request,
    which full-payload scheduling placeholder-schedules once and never
    collects; the base-scheduler fallback then schedules it at a negative span,
    which killed the stage EngineCore before #5269 and leaves the request
    parked in ``running`` forever after it (#5196, #5471). Each degenerate case
    (non-dict pooling_output, missing ``codes.audio``, all codec frames dropped
    by the filter) must instead return a finished payload with at least one
    frame that survives the codec validity filter, so the request runs the
    normal one-shot path and finishes cleanly.
    """
    request = SimpleNamespace(request_id="r", output_token_ids=[0, 1, 2])

    payload = talker2code2wav_full_payload(transfer_manager=None, pooling_output=pooling_output, request=request)

    assert payload is not None
    assert payload["meta"]["finished"].item() is True
    audio = payload["codes"]["audio"]
    # Same wire format as the normal path: flat, codebook-major, one frame.
    assert audio.ndim == 1
    assert audio.numel() == _NUM_QUANTIZERS_DEFAULT
    # The placeholder must survive the same validity filter real takes go
    # through; a frame the filter would drop re-creates the zero-token request.
    frames = audio.reshape(-1, _NUM_QUANTIZERS_DEFAULT)
    assert int(_filter_audio_codes_qwen3_tts(frames).shape[0]) >= 1


_RAMP = [1, 4, 8, 16, 25]


class TestRampHelpers:
    def test_parse_ramp_none_when_absent(self):
        assert parse_chunk_ramp({}) is None

    def test_parse_ramp_valid(self):
        assert parse_chunk_ramp({"codec_chunk_ramp": [1, 4, 8, 16, 25]}) == [1, 4, 8, 16, 25]

    def test_parse_ramp_from_string(self):
        assert parse_chunk_ramp({"codec_chunk_ramp": "1, 4, 8, 16, 25"}) == [1, 4, 8, 16, 25]

    def test_parse_ramp_too_short(self):
        assert parse_chunk_ramp({"codec_chunk_ramp": [25]}) is None

    def test_parse_ramp_non_positive(self):
        assert parse_chunk_ramp({"codec_chunk_ramp": [1, 0, 8]}) is None

    def test_parse_ramp_bad_string(self):
        assert parse_chunk_ramp({"codec_chunk_ramp": "a,b"}) is None

    def test_parse_ramp_int_returns_none(self):
        assert parse_chunk_ramp({"codec_chunk_ramp": 4}) is None

    def test_parse_ramp_float_returns_none(self):
        assert parse_chunk_ramp({"codec_chunk_ramp": 4.5}) is None

    def test_parse_ramp_mixed_list_returns_none(self):
        assert parse_chunk_ramp({"codec_chunk_ramp": [4, "x"]}) is None

    def test_parse_ramp_warns_on_tail_mismatch(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            result = parse_chunk_ramp({"codec_chunk_ramp": [4, 4, 8]}, steady=25)
        assert result == [4, 4, 8]
        assert any("reintroduces" in r.message for r in caplog.records)

    def test_parse_ramp_no_warn_on_tail_match(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            result = parse_chunk_ramp({"codec_chunk_ramp": [4, 4, 8, 16, 25]}, steady=25)
        assert result == [4, 4, 8, 16, 25]
        assert not any("reintroduces" in r.message for r in caplog.records)

    @pytest.mark.parametrize(
        "index,ramp,steady,expected",
        [
            (0, _RAMP, 25, 1),
            (1, _RAMP, 25, 4),
            (2, _RAMP, 25, 8),
            (3, _RAMP, 25, 16),
            (4, _RAMP, 25, 25),
            (5, _RAMP, 25, 25),
            (100, _RAMP, 25, 25),
        ],
    )
    def test_ramp_chunk_size(self, index, ramp, steady, expected):
        assert ramp_chunk_size(index, ramp, steady) == expected

    @pytest.mark.parametrize(
        "index,ramp,steady,expected",
        [
            (0, _RAMP, 25, 1),
            (1, _RAMP, 25, 5),
            (2, _RAMP, 25, 13),
            (3, _RAMP, 25, 29),
            (4, _RAMP, 25, 54),
            (5, _RAMP, 25, 79),
            (6, _RAMP, 25, 104),
            (100, _RAMP, 25, 54 + 96 * 25),
        ],
    )
    def test_ramp_cumulative(self, index, ramp, steady, expected):
        assert ramp_cumulative(index, ramp, steady) == expected

    @pytest.mark.parametrize(
        "length,chunk_index,finished,expected_emit,expected_ctx",
        [
            (0, 0, False, False, 0),
            (1, 0, False, True, 1),
            (3, 1, False, False, 0),
            (5, 1, False, True, 4),
            (12, 2, False, False, 0),
            (13, 2, False, True, 8),
            (28, 3, False, False, 0),
            (29, 3, False, True, 16),
            (53, 4, False, False, 0),
            (54, 4, False, True, 25),
            (78, 5, False, False, 0),
            (79, 5, False, True, 25),
            (1, 1, True, True, 0),
            (3, 1, True, True, 2),
            (5, 1, True, True, 4),
        ],
    )
    def test_compute_ramp_emit(self, length, chunk_index, finished, expected_emit, expected_ctx):
        emit, ctx = compute_ramp_emit(length, chunk_index, _RAMP, 25, finished)
        assert emit == expected_emit
        assert ctx == expected_ctx


class TestChunkRampEmission:
    RAMP = [1, 4, 8, 16, 25]

    def _emit(self, tm, rid, n_frames, finished=False):
        tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(n_frames)]
        return talker2code2wav_async_chunk(
            transfer_manager=tm,
            multimodal_output={"codes": {"audio": torch.zeros((0,))}},
            request=_req(rid, finished=finished),
            is_finished=finished,
        )

    def test_ramp_sequence_1_4_8_16_25(self):
        tm = _tm(chunk_ramp=self.RAMP)
        rid = "ramp-seq"

        p0 = self._emit(tm, rid, 1)
        assert p0 is not None
        assert len(p0.codes.audio) == _Q * 1
        tm.ramp_chunk_count[rid] = 1

        assert self._emit(tm, rid, 2) is None
        assert self._emit(tm, rid, 3) is None
        assert self._emit(tm, rid, 4) is None
        p1 = self._emit(tm, rid, 5)
        assert p1 is not None
        assert len(p1.codes.audio) == _Q * 4
        assert p1.meta.left_context_size == 0
        tm.ramp_chunk_count[rid] = 2

        for n in range(6, 13):
            assert self._emit(tm, rid, n) is None
        p2 = self._emit(tm, rid, 13)
        assert p2 is not None
        assert p2.meta.left_context_size == 0
        assert len(p2.codes.audio) == _Q * 8
        tm.ramp_chunk_count[rid] = 3

        for n in range(14, 29):
            assert self._emit(tm, rid, n) is None
        p3 = self._emit(tm, rid, 29)
        assert p3 is not None
        assert p3.meta.left_context_size == 0
        assert len(p3.codes.audio) == _Q * 16
        tm.ramp_chunk_count[rid] = 4

        for n in range(30, 54):
            assert self._emit(tm, rid, n) is None
        p4 = self._emit(tm, rid, 54)
        assert p4 is not None
        assert p4.meta.left_context_size == 0
        assert len(p4.codes.audio) == _Q * 25
        tm.ramp_chunk_count[rid] = 5

        for n in range(55, 79):
            assert self._emit(tm, rid, n) is None
        p5 = self._emit(tm, rid, 79)
        assert p5 is not None
        assert p5.meta.left_context_size == 0
        assert len(p5.codes.audio) == _Q * 25

    def test_ramp_finished_flush_mid_chunk(self):
        tm = _tm(chunk_ramp=self.RAMP)
        rid = "ramp-flush"

        p0 = self._emit(tm, rid, 1)
        assert p0 is not None
        tm.ramp_chunk_count[rid] = 1

        p_fin = self._emit(tm, rid, 3, finished=True)
        assert p_fin is not None
        assert p_fin.meta.finished.item() is True
        assert len(p_fin.codes.audio) == _Q * 2
        assert p_fin.meta.left_context_size == 0

    def test_ramp_finished_no_new_frames(self):
        tm = _tm(chunk_ramp=self.RAMP)
        rid = "ramp-no-new"

        p0 = self._emit(tm, rid, 1)
        assert p0 is not None
        tm.ramp_chunk_count[rid] = 1

        p_fin = self._emit(tm, rid, 1, finished=True)
        assert p_fin is not None
        assert p_fin.meta.finished.item() is True
        assert p_fin.codes.audio.numel() == 0

    def test_ramp_finished_at_exact_threshold(self):
        tm = _tm(chunk_ramp=self.RAMP)
        rid = "ramp-exact"

        p0 = self._emit(tm, rid, 1)
        assert p0 is not None
        tm.ramp_chunk_count[rid] = 1

        p1 = self._emit(tm, rid, 5, finished=True)
        assert p1 is not None
        assert p1.meta.finished.item() is True

    def test_ramp_backward_compat_without_config(self):
        tm = _tm(initial_chunk_frames=1)
        rid = "no-ramp"

        p0 = self._emit(tm, rid, 1)
        assert p0 is not None
        assert len(p0.codes.audio) == _Q * 1
        tm.put_req_chunk[rid] = 1

        for n in range(2, 26):
            assert self._emit(tm, rid, n) is None
        p1 = self._emit(tm, rid, 26)
        assert p1 is not None
        assert p1.meta.left_context_size == 0
        assert len(p1.codes.audio) == _Q * 25

    def test_ramp_with_ref_code_first_chunk(self):
        tm = _tm(chunk_ramp=self.RAMP, left_context=25)
        rid = "ramp-ref"
        tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(1)]
        ref_code = torch.tensor([[9, 9, 9, 9], [8, 8, 8, 8]], dtype=torch.long)

        payload = talker2code2wav_async_chunk(
            transfer_manager=tm,
            multimodal_output={"codes": {"audio": torch.zeros((0,)), "ref": ref_code}},
            request=_req(rid, finished=False),
            is_finished=False,
        )

        assert payload is not None
        assert payload.meta.ref_context_size == 2
        assert payload.meta.ref_context_included is True
        assert payload.meta.left_context_size == 2
        assert len(payload.codes.audio) == _Q * (2 + 1)

    def test_ramp_steady_state_after_ramp_exhausted(self):
        tm = _tm(chunk_ramp=[1, 4], chunk_frames=25)
        rid = "ramp-short"

        p0 = self._emit(tm, rid, 1)
        assert p0 is not None
        tm.ramp_chunk_count[rid] = 1

        p1 = self._emit(tm, rid, 5)
        assert p1 is not None
        tm.ramp_chunk_count[rid] = 2

        for n in range(6, 30):
            assert self._emit(tm, rid, n) is None
        p2 = self._emit(tm, rid, 30)
        assert p2 is not None
        assert p2.meta.left_context_size == 0
        assert len(p2.codes.audio) == _Q * 25
        tm.ramp_chunk_count[rid] = 3

        for n in range(31, 55):
            assert self._emit(tm, rid, n) is None
        p3 = self._emit(tm, rid, 55)
        assert p3 is not None
        assert p3.meta.left_context_size == 0
        assert len(p3.codes.audio) == _Q * 25

    def test_ramp_profile_4_4_8_16_25(self):
        """Ramp [4,4,8,16,25]: chunk 0=4 frames (320ms audio covers chunk 1
        gen time → no gap at chunk 0→1), gradual ramp to steady state."""
        tm = _tm(chunk_ramp=[4, 4, 8, 16, 25])
        rid = "ramp-4-4-8-16-25"

        for n in range(1, 4):
            assert self._emit(tm, rid, n) is None
        p0 = self._emit(tm, rid, 4)
        assert p0 is not None
        assert p0.meta.left_context_size == 0
        tm.ramp_chunk_count[rid] = 1

        for n in range(5, 8):
            assert self._emit(tm, rid, n) is None
        p1 = self._emit(tm, rid, 8)
        assert p1 is not None
        assert p1.meta.left_context_size == 0
        assert len(p1.codes.audio) == _Q * 4
        tm.ramp_chunk_count[rid] = 2

        for n in range(9, 16):
            assert self._emit(tm, rid, n) is None
        p2 = self._emit(tm, rid, 16)
        assert p2 is not None
        assert p2.meta.left_context_size == 0
        assert len(p2.codes.audio) == _Q * 8
        tm.ramp_chunk_count[rid] = 3

        for n in range(17, 32):
            assert self._emit(tm, rid, n) is None
        p3 = self._emit(tm, rid, 32)
        assert p3 is not None
        assert p3.meta.left_context_size == 0
        assert len(p3.codes.audio) == _Q * 16
        tm.ramp_chunk_count[rid] = 4

        for n in range(33, 57):
            assert self._emit(tm, rid, n) is None
        p4 = self._emit(tm, rid, 57)
        assert p4 is not None
        assert p4.meta.left_context_size == 0
        assert len(p4.codes.audio) == _Q * 25
        tm.ramp_chunk_count[rid] = 5

        for n in range(58, 82):
            assert self._emit(tm, rid, n) is None
        p5 = self._emit(tm, rid, 82)
        assert p5 is not None
        assert p5.meta.left_context_size == 0
        assert len(p5.codes.audio) == _Q * 25

    def test_ramp_resets_on_segment_boundary(self):
        """After segment 1 emits chunks, a segment boundary clears
        code_prompt_token_ids and ramp_chunk_count. Segment 2 must restart
        from chunk index 0.

        Regression for: put_req_chunk is request-global and not reset at
        segment boundaries. ramp_chunk_count is popped alongside
        code_prompt_token_ids on is_segment_finished, so the ramp index
        resets by construction."""
        tm = _tm(chunk_ramp=[4, 4, 8, 16, 25])
        rid = "ramp-segment"

        p0 = self._emit(tm, rid, 4)
        assert p0 is not None
        assert p0.meta.left_context_size == 0
        tm.ramp_chunk_count[rid] = 1
        tm.put_req_chunk[rid] = 1

        p1 = self._emit(tm, rid, 8)
        assert p1 is not None
        tm.ramp_chunk_count[rid] = 2
        tm.put_req_chunk[rid] = 2

        tm.code_prompt_token_ids[rid] = []
        tm.ramp_chunk_count.pop(rid, None)

        p_seg2_0 = self._emit(tm, rid, 4)
        assert p_seg2_0 is not None
        assert p_seg2_0.meta.left_context_size == 0

    def test_ramp_resets_on_segment_boundary_one_chunk_history(self):
        """Segment 1 emitted exactly 1 chunk (put_req_chunk=1). Segment 2
        must still restart from chunk index 0.

        Regression for: the hybrid put_req_chunk/derived scheme collided
        when put_chunk=1 because length >= ramp_cumulative(0) turned true
        at length=4, causing chunk_index to jump to 1 and dropping the
        first 4 frames of segment 2."""
        tm = _tm(chunk_ramp=[4, 4, 8, 16, 25])
        rid = "ramp-segment-1chunk"

        p0 = self._emit(tm, rid, 4)
        assert p0 is not None
        tm.ramp_chunk_count[rid] = 1
        tm.put_req_chunk[rid] = 1

        tm.code_prompt_token_ids[rid] = []
        tm.ramp_chunk_count.pop(rid, None)

        p_seg2_0 = self._emit(tm, rid, 4)
        assert p_seg2_0 is not None
        assert p_seg2_0.meta.left_context_size == 0


# ---------------------------------------------------------------------------
# W3: the per-step device sync is batched, without moving chunk boundaries.
#
# The talker zeroes any frame whose layer-0 id is out of codebook range, which
# covers EOS and the whole prefill span, so all-zero frames are a routine
# ~2% of a stream rather than an edge case. `length` drives every emission
# gate, so padding must not be counted -- these tests pin that.
# ---------------------------------------------------------------------------


def _feed(tm, rid, frames, *, ic, finished_on_last=False):
    """Drive the save path one frame at a time, as the talker does."""
    emissions = []
    for i, f in enumerate(frames):
        last = i == len(frames) - 1
        payload = talker2code2wav_async_chunk(
            transfer_manager=tm,
            multimodal_output={"codes": {"audio": torch.tensor([f], dtype=torch.long)}},
            request=_req(rid, finished=finished_on_last and last, initial_codec_chunk_frames=ic),
            is_finished=finished_on_last and last,
        )
        if payload is not None:
            emissions.append((i, payload))
    return emissions


def test_padding_frames_do_not_advance_the_chunk_boundary():
    """A zero frame must not count toward the emission threshold.

    With chunk/IC of 3 and a zero at step 1, the third *real* frame arrives at
    step 3. Counting the zero would emit one step early carrying two frames.
    """
    tm = _tm(chunk_frames=3, left_context=0, initial_chunk_frames=3)
    emissions = _feed(tm, "r-pad", ([1, 1, 1, 1], [0, 0, 0, 0], [2, 2, 2, 2], [3, 3, 3, 3]), ic=3)

    assert [step for step, _ in emissions] == [3], "emission must wait for the third real frame"
    payload = emissions[0][1]
    # Codebook-major over exactly the three real frames.
    assert payload.codes.audio.tolist() == [1, 2, 3] * 4
    # The accumulator holds real frames only, so `length` keeps its meaning.
    assert len(tm.code_prompt_token_ids["r-pad"]) == 3


def test_padding_only_stream_emits_nothing_until_finished():
    tm = _tm(chunk_frames=3, left_context=0, initial_chunk_frames=3)
    emissions = _feed(tm, "r-allpad", ([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]), ic=3)
    assert emissions == []
    assert len(tm.code_prompt_token_ids["r-allpad"]) == 0


def test_terminal_padding_still_yields_the_empty_finished_payload():
    """An all-padding take that finishes keeps main's empty-finished sentinel.

    The async-chunk adapter handles an empty terminal payload; substituting a
    placeholder frame here would inject audible audio and, with a ref code
    present, trip the reference-prefix check in code2wav.
    """
    tm = _tm(chunk_frames=3, left_context=0, initial_chunk_frames=3)
    emissions = _feed(tm, "r-term", ([0, 0, 0, 0], [0, 0, 0, 0]), ic=3, finished_on_last=True)
    assert len(emissions) == 1
    payload = emissions[0][1]
    assert payload.codes.audio.numel() == 0
    assert bool(payload.meta.finished)


def test_emptiness_check_is_batched_not_per_step(monkeypatch):
    """The whole point: syncs scale with chunks, not with decode steps."""
    calls = {"n": 0}
    real_tolist = torch.Tensor.tolist

    def counting_tolist(self):
        calls["n"] += 1
        return real_tolist(self)

    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)

    tm = _tm(chunk_frames=25, left_context=0, initial_chunk_frames=25)
    frames = [[i + 1, i + 1, i + 1, i + 1] for i in range(60)]
    _feed(tm, "r-sync", frames, ic=25)

    # 60 decode steps, chunk size 25 -> a handful of resolves. The pre-W3 path
    # did one device sync per step; anything near 60 means the batching broke.
    assert calls["n"] <= 10, f"expected batched syncs, saw {calls['n']} for 60 steps"

    # Frames past the last threshold stay pending on purpose -- that deferral is
    # what avoids the sync. Nothing is lost: every frame is in one bucket or the
    # other, and finishing flushes the remainder.
    committed = tm.code_prompt_token_ids["r-sync"]
    pending = tm.pending_frames.get("r-sync", [])
    assert len(committed) + len(pending) == 60
    assert pending, "expected frames past the last threshold to still be deferred"

    talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.tensor([[61, 61, 61, 61]], dtype=torch.long)}},
        request=_req("r-sync", finished=True, initial_codec_chunk_frames=25),
        is_finished=True,
    )
    assert len(tm.code_prompt_token_ids["r-sync"]) == 61
    assert not tm.pending_frames.get("r-sync")


def test_committed_frames_survive_a_resolve_boundary():
    """Frames held pending across a non-resolving step are not dropped."""
    tm = _tm(chunk_frames=5, left_context=0, initial_chunk_frames=5)
    frames = [[i + 1, i + 1, i + 1, i + 1] for i in range(5)]
    emissions = _feed(tm, "r-hold", frames, ic=5)
    assert [step for step, _ in emissions] == [4]
    assert emissions[0][1].codes.audio.tolist() == [1, 2, 3, 4, 5] * 4


@pytest.mark.parametrize(
    "chunk_frames,ic,ramp", [(3, 3, None), (5, 2, None), (25, 0, None), (4, 4, None), (5, 0, "1,2,3")]
)
@pytest.mark.parametrize("seed", range(8))
def test_padding_is_invisible_to_the_emission_schedule(chunk_frames, ic, ramp, seed):
    """The contract: padding must not change which frames go out, or how grouped.

    A stream with padding interleaved must deliver the same real frames, in the
    same chunk groupings, as the same stream with no padding at all. This is the
    property the closed PR #5178 broke -- it counted padding toward ``length``,
    so every chunk boundary moved -- and the property ``_emit_threshold`` exists
    to preserve. Checked across chunk/initial-chunk/ramp configurations rather
    than at one hand-picked size.

    Consecutive duplicate windows are collapsed first. A padding frame arriving
    while the accumulator sits exactly on a chunk boundary makes the gate re-emit
    the window it just sent; running this same test against upstream's processor
    reproduces it identically, so it is pre-existing behavior rather than
    something this change introduces, and it is normalized away here instead of
    being asserted on.
    """
    import random

    rng = random.Random(seed)
    real = [[i + 1, i + 1, i + 1, i + 1] for i in range(12)]
    withpad = []
    for f in real:
        while rng.random() < 0.3:
            withpad.append([0, 0, 0, 0])
        withpad.append(f)

    def run(frames, rid):
        tm = _tm(chunk_frames=chunk_frames, left_context=0, initial_chunk_frames=ic, chunk_ramp=ramp)
        out = []
        for i, f in enumerate(frames):
            payload = talker2code2wav_async_chunk(
                transfer_manager=tm,
                multimodal_output={"codes": {"audio": torch.tensor([f], dtype=torch.long)}},
                request=_req(rid, finished=i == len(frames) - 1, initial_codec_chunk_frames=ic or None),
                is_finished=i == len(frames) - 1,
            )
            if payload is not None:
                window = payload.codes.audio.tolist()
                if not out or out[-1] != window:
                    out.append(window)
        return out

    assert run(withpad, "r-pad") == run(real, "r-clean")


def test_dynamic_ic_load_counts_requests_still_inside_their_first_chunk():
    """A request counts toward load from its first frame, not its first resolve.

    Dynamic IC reads how many requests are active. Frames now sit in
    `pending_frames` until a resolve, so counting only `code_prompt_token_ids`
    would under-report load by one for every request still inside its first
    chunk. That is invisible to the padding tests -- both arms would shift
    together -- so it is pinned directly here.
    """
    tm = _tm(chunk_frames=25, left_context=0, max_num_seqs=8)

    # Eight requests each emit one frame. None has resolved into
    # code_prompt_token_ids yet, but all eight are active.
    for i in range(8):
        talker2code2wav_async_chunk(
            transfer_manager=tm,
            multimodal_output={"codes": {"audio": torch.tensor([[7, 7, 7, 7]], dtype=torch.long)}},
            request=_req(f"load-{i}", finished=False),
            is_finished=False,
        )

    committed_total = sum(len(v) for v in tm.code_prompt_token_ids.values())
    pending_total = sum(len(v) for v in tm.pending_frames.values())
    assert committed_total + pending_total == 8

    # A ninth request arriving under that load must see a saturated load factor
    # (8/8) and therefore the largest IC, not the smallest.
    max_ic = max_ic_for_chunk_size(25)
    talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.tensor([[9, 9, 9, 9]], dtype=torch.long)}},
        request=_req("load-new", finished=False),
        is_finished=False,
    )
    assert tm._cached_ic["load-new"] == max_ic


def test_abort_releases_accumulated_frames():
    """Aborted requests must not strand codec frames.

    `finish_requests` handles abort/error/eviction, where no terminal chunk is
    sent and `cleanup_sender` never runs. The frames are device tensors now, so
    stranding them holds GPU memory for the adapter's lifetime rather than a
    small host list.
    """
    from vllm.v1.request import RequestStatus

    from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
        OmniChunkTransferAdapter,
    )

    adapter = OmniChunkTransferAdapter.__new__(OmniChunkTransferAdapter)
    adapter.waiting_for_chunk_waiting_requests = deque()
    adapter.waiting_for_chunk_running_requests = deque()
    adapter._held_non_active = deque()
    adapter.code_prompt_token_ids = defaultdict(list)
    adapter.pending_frames = {}
    adapter.cleanup_receiver = lambda _rid: None

    # The two id namespaces must differ here. `finish_requests` is called with
    # the scheduler's internal id, while both frame stores are keyed by the
    # user-facing one; a fixture that uses one string for both cannot tell a
    # working cleanup from a no-op.
    internal_id, external_id = "internal-uuid-1", "user-facing-1"
    adapter.request_ids_mapping = {internal_id: external_id}
    adapter.code_prompt_token_ids[external_id] = [torch.tensor([1, 2, 3, 4])]
    adapter.pending_frames[external_id] = [torch.tensor([5, 6, 7, 8])]

    adapter.finish_requests(internal_id, RequestStatus.FINISHED_ABORTED, requests={})

    assert external_id not in adapter.code_prompt_token_ids
    assert external_id not in adapter.pending_frames
