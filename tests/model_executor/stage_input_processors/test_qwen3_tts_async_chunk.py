# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import (
    AdaptiveChunkController,
    compute_adaptive_emit,
    compute_dynamic_initial_chunk_size,
    compute_ramp_emit,
    max_ic_for_chunk_size,
    parse_adaptive_config,
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


def _tm(
    *,
    chunk_frames=25,
    left_context=25,
    max_num_seqs=1,
    initial_chunk_frames=0,
    chunk_ramp=None,
    adaptive=False,
    adaptive_min=2,
    adaptive_margin=200.0,
):
    extra = {
        "codec_chunk_frames": chunk_frames,
        "codec_left_context_frames": left_context,
        "initial_codec_chunk_frames": initial_chunk_frames,
    }
    if chunk_ramp is not None:
        extra["codec_chunk_ramp"] = chunk_ramp
    if adaptive:
        extra["codec_chunk_adaptive"] = True
        extra["codec_chunk_min_frames"] = adaptive_min
        extra["codec_chunk_safety_margin_ms"] = adaptive_margin
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        scheduler_max_num_seqs=max_num_seqs,
        put_req_chunk=defaultdict(int),
        ramp_chunk_count=defaultdict(int),
        _adaptive_states={},
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

        target_logger = logging.getLogger("vllm_omni.model_executor.stage_input_processors.chunk_size_utils")
        target_logger.addHandler(caplog.handler)
        prev_level = target_logger.level
        target_logger.setLevel(logging.WARNING)
        try:
            result = parse_chunk_ramp({"codec_chunk_ramp": [4, 4, 8]}, steady=25)
        finally:
            target_logger.removeHandler(caplog.handler)
            target_logger.setLevel(prev_level)
        assert result == [4, 4, 8]
        assert "reintroduces" in caplog.text

    def test_parse_ramp_no_warn_on_tail_match(self, caplog):
        import logging

        target_logger = logging.getLogger("vllm_omni.model_executor.stage_input_processors.chunk_size_utils")
        target_logger.addHandler(caplog.handler)
        prev_level = target_logger.level
        target_logger.setLevel(logging.WARNING)
        try:
            result = parse_chunk_ramp({"codec_chunk_ramp": [4, 4, 8, 16, 25]}, steady=25)
        finally:
            target_logger.removeHandler(caplog.handler)
            target_logger.setLevel(prev_level)
        assert result == [4, 4, 8, 16, 25]
        assert "reintroduces" not in caplog.text

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


class TestAdaptiveController:
    def test_compute_next_chunk_size_basic(self):
        ctrl = AdaptiveChunkController(
            frame_time_ewma_ms=20.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=10,
        )
        now = 0.5
        size = ctrl.compute_next_chunk_size(now, min_frames=2, max_frames=25, safety_margin_ms=100.0)
        assert 2 <= size <= 25

    def test_compute_next_chunk_size_clamp_min(self):
        """min_frames floor holds when ramp_floor (last_target + delta) is
        below min_frames. Uses a small EWMA so delta=2 and last_target=0,
        i.e. ramp_floor=2 < min_frames=4 -> greedy=min_frames wins."""
        ctrl = AdaptiveChunkController(
            frame_time_ewma_ms=10.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=1,
        )
        # buffer = 80 - 70 = 10ms (positive but < margin=200)
        # greedy: available=0, max_n=0, greedy=min_frames=4
        # ramp_floor: last_target=0 + delta=max(2,int(10/15))=2 -> 2 < 4
        now = 0.07
        size = ctrl.compute_next_chunk_size(now, min_frames=4, max_frames=25, safety_margin_ms=200.0)
        assert size == 4  # min_frames floor dominates ramp_floor=2

    def test_compute_next_chunk_size_clamp_max(self):
        ctrl = AdaptiveChunkController(
            frame_time_ewma_ms=10.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=100,
        )
        now = 1.0
        size = ctrl.compute_next_chunk_size(now, min_frames=2, max_frames=25, safety_margin_ms=50.0)
        assert size == 25

    def test_hysteresis_locks_steady(self):
        ctrl = AdaptiveChunkController(
            frame_time_ewma_ms=10.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=50,
        )
        now = 0.1
        size1 = ctrl.compute_next_chunk_size(now, min_frames=2, max_frames=25, safety_margin_ms=50.0)
        assert ctrl.steady_state_reached
        assert size1 == 25
        size2 = ctrl.compute_next_chunk_size(now + 10.0, min_frames=2, max_frames=25, safety_margin_ms=50.0)
        assert size2 == 25

    def test_ramp_floor_overrides_greedy_when_buffer_low(self):
        """When buffer is negative (overloaded), ramp_floor still forces
        gradual growth to avoid stagnating at min_frames (which would cause
        too many Code2Wav decode passes and RTF explosion)."""
        ctrl = AdaptiveChunkController(
            frame_time_ewma_ms=50.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=2,
            last_target=4,
        )
        # buffer = 160 - 200 = -40ms (negative)
        # greedy: available=0, max_n=0, greedy=2
        # ramp_floor: last_target=4 + delta=max(2,int(50/15))=3 -> 7
        now = 0.2
        size = ctrl.compute_next_chunk_size(now, min_frames=2, max_frames=25, safety_margin_ms=50.0)
        assert not ctrl.steady_state_reached
        assert size == 7  # ramp_floor (unconditional) dominates greedy=2

    def test_ramp_floor_dominates_when_last_target_set(self):
        """Once last_target is set (post-emit), the unconditional ramp_floor
        (last_target + ewma-scaled delta) dominates greedy whenever greedy is
        smaller. This is intended: chunk size grows monotonically toward max
        to minimize Code2Wav decode passes (pre-W5; see W5 coupling)."""
        ctrl = AdaptiveChunkController(
            frame_time_ewma_ms=50.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=4,
            last_target=4,
        )
        # buffer = 320 - 100 = 220ms (positive)
        # greedy: available=170, max_n=int(170/50)=3, greedy=3
        # ramp_floor: 4 + delta=max(2,int(50/15))=3 -> 7
        now = 0.1
        size = ctrl.compute_next_chunk_size(now, min_frames=2, max_frames=25, safety_margin_ms=50.0)
        assert size == 7  # ramp_floor (7) dominates greedy (3)

    def test_greedy_dominates_when_ramp_floor_low(self):
        """When last_target=0 (fresh, no prior emit) and buffer is positive
        but below the hysteresis threshold, greedy selects an intermediate
        size larger than ramp_floor and greedy dominates. This is the only
        regime where greedy (not ramp_floor or hysteresis) sets the size."""
        ctrl = AdaptiveChunkController(
            frame_time_ewma_ms=50.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=8,
            last_target=0,
        )
        # buffer = 640 - 100 = 540ms (positive, < hysteresis 1300ms)
        # greedy: available=490, max_n=int(490/50)=9, greedy=9
        # ramp_floor: 0 + delta=max(2,int(50/15))=3 -> 3
        now = 0.1
        size = ctrl.compute_next_chunk_size(now, min_frames=2, max_frames=25, safety_margin_ms=50.0)
        assert not ctrl.steady_state_reached
        assert size == 9  # greedy (9) dominates ramp_floor (3)

    def test_record_emit_ewma_update(self):
        ctrl = AdaptiveChunkController(
            frame_time_ewma_ms=20.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=4,
        )
        ctrl.record_emit(now=0.5, n_frames=8, accumulated=12, target_size=8)
        assert ctrl.frames_emitted_this_segment == 12
        assert ctrl.accumulated_at_last_emit == 12
        assert ctrl.last_emit_time == 0.5
        assert ctrl.last_target == 8  # D6: stores target, not actual
        measured = (0.5 - 0.0) * 1000.0 / 8
        expected = 0.3 * measured + 0.7 * 20.0
        assert abs(ctrl.frame_time_ewma_ms - expected) < 0.01

    def test_record_emit_first_then_second(self):
        ctrl = AdaptiveChunkController(
            frame_time_ewma_ms=0.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=0,
        )
        ctrl.record_emit(now=0.4, n_frames=4, accumulated=4, target_size=4)
        assert ctrl.frame_time_ewma_ms == 0.0
        assert ctrl.frames_emitted_this_segment == 4
        assert ctrl.last_target == 4

        ctrl.record_emit(now=0.8, n_frames=4, accumulated=8, target_size=4)
        expected = (0.8 - 0.4) * 1000.0 / 4
        assert abs(ctrl.frame_time_ewma_ms - expected) < 0.01

    def test_record_emit_collects_underrun(self):
        """record_emit mirrors stream_client.py's playback-cursor underrun
        model: per_chunk_gap_ms records each chunk's lateness, total_gap_s
        accumulates gaps > 5ms. Deterministic given explicit `now`."""
        ctrl = AdaptiveChunkController(
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=0,
        )
        # chunk0: now=0.0 (= first_emit_time), n=2 -> 0.16s audio
        ctrl.record_emit(now=0.0, n_frames=2, accumulated=2, target_size=2)
        assert ctrl.chunk_sizes == [2]
        assert ctrl.per_chunk_gap_ms == [0.0]
        assert ctrl.total_gap_s == 0.0
        assert abs(ctrl.playback_cursor_s - 0.16) < 1e-9
        # chunk1: now=0.2 (200ms wall) -> arrival 0.2, late by 0.2-0.16=0.04s
        ctrl.record_emit(now=0.2, n_frames=2, accumulated=4, target_size=2)
        assert ctrl.chunk_sizes == [2, 2]
        assert abs(ctrl.per_chunk_gap_ms[1] - 40.0) < 1e-6
        assert abs(ctrl.total_gap_s - 0.04) < 1e-9
        assert abs(ctrl.playback_cursor_s - 0.36) < 1e-9

    def test_ewma_zero_skips_hysteresis(self):
        """When EWMA=0, hysteresis must not trigger (steady_gen_ms=0
        would make any positive buffer satisfy the condition)."""
        ctrl = AdaptiveChunkController(
            frame_time_ewma_ms=0.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=10,
        )
        now = 0.001
        size = ctrl.compute_next_chunk_size(now, min_frames=2, max_frames=25, safety_margin_ms=50.0)
        assert not ctrl.steady_state_reached
        assert size == 2  # min_frames (EWMA=0 → max_n=min_frames)

    def test_ramp_divisor_configurable(self):
        """ramp_divisor/ramp_delta_min are controller fields wired from config.
        Defaults (15/2) preserve the original formula; changing them changes
        delta. Lets a deploy tune the ramp via yaml without code changes, while
        the unit tests (which use defaults) stay green."""
        # default divisor=15: ewma=50 -> delta=max(2,int(50/15))=3 -> floor=4+3=7
        c1 = AdaptiveChunkController(
            frame_time_ewma_ms=50.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=2,
            last_target=4,
        )
        assert c1.compute_next_chunk_size(0.2, 2, 25, 50.0) == 7
        # divisor=10: delta=max(2,int(50/10))=5 -> floor=4+5=9
        c2 = AdaptiveChunkController(
            frame_time_ewma_ms=50.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=2,
            last_target=4,
            ramp_divisor=10,
        )
        assert c2.compute_next_chunk_size(0.2, 2, 25, 50.0) == 9
        # delta_min=4 + ewma=10 (int(10/15)=0): delta=max(4,0)=4 -> floor=4+4=8
        c3 = AdaptiveChunkController(
            frame_time_ewma_ms=10.0,
            first_emit_time=0.0,
            last_emit_time=0.0,
            frames_emitted_this_segment=2,
            last_target=4,
            ramp_delta_min=4,
        )
        assert c3.compute_next_chunk_size(0.2, 2, 25, 50.0) == 8


class TestAdaptiveEmitHelper:
    @pytest.mark.parametrize(
        "length,accumulated,target,finished,expected_emit,expected_ctx",
        [
            (4, 0, 4, False, True, 4),
            (3, 0, 4, False, False, 0),
            (12, 4, 8, False, True, 8),
            (10, 4, 8, False, False, 0),
            # D3: finished flushes ALL remaining frames, not just target_size
            (6, 4, 8, True, True, 2),
            (4, 4, 8, True, True, 0),
            (20, 4, 8, True, True, 16),  # new_frames=16 > target=8 → flush 16
            (30, 0, 25, True, True, 30),  # finished, new_frames=30 > target=25 → flush 30
            # B1: non-finished path also ships ALL accumulated frames (new_frames),
            # not just target_size. When new_frames > target_size (e.g. scheduler
            # stall dropped target below queued frames), returning target_size
            # would cause the [-context_length:] slice to skip surplus frames.
            (10, 0, 4, False, True, 10),  # new_frames=10 > target=4 → ship 10
            (8, 2, 4, False, True, 6),  # new_frames=6 > target=4 → ship 6
            (100, 0, 25, False, True, 100),  # new_frames=100 > target=25 → ship 100
        ],
    )
    def test_compute_adaptive_emit(self, length, accumulated, target, finished, expected_emit, expected_ctx):
        emit, ctx = compute_adaptive_emit(length, accumulated, target, finished)
        assert emit == expected_emit
        assert ctx == expected_ctx


class TestAdaptiveConfigParsing:
    def test_disabled_by_default(self):
        enabled, min_f, margin, divisor, delta_min = parse_adaptive_config({})
        assert enabled is False
        assert min_f == 2
        assert margin == 50.0
        assert divisor == 15
        assert delta_min == 2

    def test_enabled(self):
        cfg = {"codec_chunk_adaptive": True}
        enabled, *_ = parse_adaptive_config(cfg)
        assert enabled is True

    def test_custom_values(self):
        cfg = {
            "codec_chunk_adaptive": True,
            "codec_chunk_min_frames": 4,
            "codec_chunk_safety_margin_ms": 300.0,
            "codec_chunk_ramp_divisor": 10,
            "codec_chunk_ramp_delta_min": 3,
        }
        enabled, min_f, margin, divisor, delta_min = parse_adaptive_config(cfg)
        assert enabled is True
        assert min_f == 4
        assert margin == 300.0
        assert divisor == 10
        assert delta_min == 3

    def test_min_frames_clamped(self):
        cfg = {"codec_chunk_adaptive": True, "codec_chunk_min_frames": 0}
        _, min_f, _, _, _ = parse_adaptive_config(cfg)
        assert min_f == 1

    def test_ramp_divisor_clamped(self):
        cfg = {"codec_chunk_adaptive": True, "codec_chunk_ramp_divisor": 0}
        _, _, _, divisor, _ = parse_adaptive_config(cfg)
        assert divisor == 1

    def test_ramp_delta_min_clamped(self):
        cfg = {"codec_chunk_adaptive": True, "codec_chunk_ramp_delta_min": 0}
        _, _, _, _, delta_min = parse_adaptive_config(cfg)
        assert delta_min == 1


class TestAdaptiveEmission:
    def _emit(self, tm, rid, n_frames, finished=False):
        tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(n_frames)]
        return talker2code2wav_async_chunk(
            transfer_manager=tm,
            multimodal_output={"codes": {"audio": torch.zeros((0,))}},
            request=_req(rid, finished=finished),
            is_finished=finished,
        )

    def test_adaptive_chunk0_uses_dynamic_ic(self):
        tm = _tm(adaptive=True, adaptive_min=2, max_num_seqs=8)
        rid = "adaptive-ic"

        p0 = self._emit(tm, rid, 2)
        assert p0 is not None
        assert len(p0.codes.audio) == _Q * 2

    def test_adaptive_chunk0_holds_until_ic(self):
        tm = _tm(adaptive=True, adaptive_min=2, max_num_seqs=8)
        rid = "adaptive-hold"
        p = self._emit(tm, rid, 1)
        assert p is None

        p0 = self._emit(tm, rid, 2)
        assert p0 is not None
        assert len(p0.codes.audio) == _Q * 2

    def test_adaptive_finished_flush(self):
        tm = _tm(adaptive=True, adaptive_min=2, max_num_seqs=8)
        rid = "adaptive-flush"

        p0 = self._emit(tm, rid, 2)
        assert p0 is not None
        tm.ramp_chunk_count[rid] = 1

        p_fin = self._emit(tm, rid, 5, finished=True)
        assert p_fin is not None
        assert p_fin.meta.finished.item() is True

    def test_adaptive_finished_no_new_frames(self):
        tm = _tm(adaptive=True, adaptive_min=2, max_num_seqs=8)
        rid = "adaptive-no-new"

        p0 = self._emit(tm, rid, 2)
        assert p0 is not None
        tm.ramp_chunk_count[rid] = 1

        p_fin = self._emit(tm, rid, 2, finished=True)
        assert p_fin is not None
        assert p_fin.meta.finished.item() is True
        assert p_fin.codes.audio.numel() == 0

    def test_adaptive_precedence_over_fixed_ramp(self):
        tm = _tm(chunk_ramp=[4, 4, 8, 16, 25], adaptive=True, adaptive_min=2, max_num_seqs=8)
        rid = "adaptive-precedence"

        p = self._emit(tm, rid, 1)
        assert p is None

        p0 = self._emit(tm, rid, 2)
        assert p0 is not None
        assert len(p0.codes.audio) == _Q * 2

    def test_adaptive_backward_compat(self):
        tm = _tm(initial_chunk_frames=1)
        rid = "no-adaptive"

        p0 = self._emit(tm, rid, 1)
        assert p0 is not None
        assert len(p0.codes.audio) == _Q * 1

    def test_adaptive_segment_reset(self):
        tm = _tm(adaptive=True, adaptive_min=2, max_num_seqs=8)
        rid = "adaptive-segment"

        p0 = self._emit(tm, rid, 2)
        assert p0 is not None
        tm.ramp_chunk_count[rid] = 1
        tm.put_req_chunk[rid] = 1

        tm.code_prompt_token_ids[rid] = []
        tm.ramp_chunk_count.pop(rid, None)
        tm._adaptive_states.pop(rid, None)

        p_seg2 = self._emit(tm, rid, 2)
        assert p_seg2 is not None
        assert p_seg2.meta.left_context_size == 0

    def test_adaptive_with_ref_code_first_chunk(self):
        tm = _tm(adaptive=True, adaptive_min=2, left_context=25, max_num_seqs=8)
        rid = "adaptive-ref"
        tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(2)]
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


class TestAdaptiveAccumulationPath:
    """Regression test for the frame-skipping bug (PR #6001 review B1).

    Appends one frame per call with a controlled clock, asserting that
    the concatenation of every emitted chunk equals the accumulated
    frame sequence, in order, with no gaps. Each frame has a distinct
    value so the assertion can detect a skipped frame rather than just
    a short total.
    """

    def test_no_frames_skipped_when_target_drops(self, monkeypatch):
        """When greedy target_size is high (large buffer) and then drops
        (clock jumps), new_frames > target_size at the emit point.
        compute_adaptive_emit now returns new_frames (not target_size),
        so all accumulated frames are shipped and the [-context_length:]
        slice stays contiguous with the previous emit. last_target is set
        to target_size (not n_frames) to prevent the ramp floor from
        ratcheting up on transient overshoot.

        Clock schedule (max_num_seqs=4 gives IC=2 via dynamic IC):
          t=0.0   chunk 0 (IC=2), emit 2. ewma=0.
          t=0.05  emit target=4 (ramp_floor). ewma=12.5 (direct assign).
          t=0.05  greedy dominates (target=18), 17 frames accumulate (holds).
          t=100.0 clock jumps -> greedy collapses, target drops to
                  ramp_floor=6. new_frames=18 > 6 -> emit 18 (all new frames).
                  last_target=6 (not 18, preventing floor ratchet).
        """
        tm = _tm(
            adaptive=True,
            adaptive_min=2,
            max_num_seqs=4,
        )
        rid = "adaptive-accum"
        tm.code_prompt_token_ids[rid] = []

        clock = [0.0]
        monkeypatch.setattr(
            "vllm_omni.model_executor.stage_input_processors.qwen3_tts.time.monotonic",
            lambda: clock[0],
        )

        emitted_indices: list[int] = []
        total_frames = 40

        def append_and_emit(frame_idx, finished=False):
            tm.code_prompt_token_ids[rid].append([frame_idx] * _Q)
            p = talker2code2wav_async_chunk(
                transfer_manager=tm,
                multimodal_output={"codes": {"audio": torch.zeros((0,))}},
                request=_req(rid, finished=finished),
                is_finished=finished,
            )
            if p is not None:
                if p.codes.audio.numel() > 0:
                    n = p.codes.audio.numel() // _Q
                    emitted_indices.extend(p.codes.audio[:n].tolist())
                tm.ramp_chunk_count[rid] += 1
            return p

        # Phase 1 (t=0.0): chunk 0, IC=2
        for i in range(2):
            append_and_emit(i)

        # Phase 2 (t=0.05): build EWMA, emit target=4 at length=6
        clock[0] = 0.05
        for i in range(2, 6):
            append_and_emit(i)

        # Phase 3 (t=0.05): greedy target=18 (large buffer), accumulate
        # 17 frames while holding (new_frames 1..17 < 18)
        for i in range(6, 23):
            append_and_emit(i)

        # Phase 4 (t=100.0): clock jumps, greedy collapses, target drops
        # to ramp_floor=6. new_frames=18 > 6 -> bug trigger point.
        clock[0] = 100.0
        for i in range(23, total_frames):
            append_and_emit(i, finished=(i == total_frames - 1))

        # Assert: all frames 0..39 emitted, in order, no gaps.
        expected = list(range(total_frames))
        assert emitted_indices == expected, (
            f"Frame loss detected! "
            f"Missing: {sorted(set(expected) - set(emitted_indices))}, "
            f"Extra: {sorted(set(emitted_indices) - set(expected))}"
        )
