"""Unit tests for the FunAudioChat async_chunk streaming producer.

The producer (``funaudiochat2code2wav_async_chunk``) runs once per stage-0 decode
step, accumulates the per-step CRQ codec increment (``multimodal_output["audio_token_ids"]``)
and emits ``OmniPayloadStruct`` chunks once ``hop + pre_lookahead`` codec ids are
available (or the remaining tail on finish). The hop is two-tier: a 10-frame
``initial`` hop fires the first chunk early (fast time-to-first-audio), then the
50-frame steady-state hop takes over. The producer also rolls the Flow prefix
on a fixed 200-frame boundary. All values are yaml-tunable via the connector
extra. These tests stub the ``transfer_manager`` / ``request`` exactly as the
connector data plane shapes them.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.funaudiochat import (
    _FAC_FLOW_SEGMENT_LEN,
    _FAC_INITIAL_TOKEN_HOP_LEN,
    _FAC_PRE_LOOKAHEAD_LEN,
    _FAC_TOKEN_HOP_LEN,
    funaudiochat2code2wav_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_INITIAL_HOP = _FAC_INITIAL_TOKEN_HOP_LEN  # 10 — fast first-chunk hop
_HOP = _FAC_TOKEN_HOP_LEN  # 50 — steady-state hop after the first chunk
_PRE = _FAC_PRE_LOOKAHEAD_LEN  # 3
_SEGMENT = _FAC_FLOW_SEGMENT_LEN  # 200 — bounded Flow segment


def _transfer_manager() -> SimpleNamespace:
    tm = SimpleNamespace()
    tm.request_payload = defaultdict(dict)
    tm.code_prompt_token_ids = defaultdict(list)
    return tm


def _request(req_id: str, *, finished: bool = False) -> SimpleNamespace:
    req = SimpleNamespace(external_req_id=req_id, request_id=req_id)
    req.is_finished = lambda: finished
    return req


def _mm(audio_token_ids) -> dict:
    if audio_token_ids is None:
        return {}
    return {"audio_token_ids": audio_token_ids}


def _transfer_manager_with_connector(
    steady,
    initial,
    pre,
    segment=_SEGMENT,
) -> SimpleNamespace:
    """Stub a transfer manager whose connector config sets stream hop sizes."""
    tm = _transfer_manager()
    tm.connector = SimpleNamespace(config={
        "codec_chunk_frames": steady,
        "initial_codec_chunk_frames": initial,
        "codec_pre_lookahead_frames": pre,
        "codec_flow_segment_frames": segment,
    })
    return tm


def test_emits_nothing_before_initial_hop_plus_lookahead_accumulated():
    tm = _transfer_manager()
    req = _request("r1")
    # First chunk uses the small initial hop; nothing emits until initial_hop + pre (13).
    payload = torch.full((1, 5), -1, dtype=torch.long)
    codes = [5] * (_INITIAL_HOP + _PRE - 1)  # 12 codes — just shy of the threshold
    i = 0
    while i < len(codes):
        chunk = codes[i : i + 5]
        out = funaudiochat2code2wav_async_chunk(tm, _mm(torch.tensor(chunk, dtype=torch.long)), req)
        assert out is None
        i += len(chunk)
    # An all-placeholder step must not pollute the accumulator or trip the threshold.
    assert funaudiochat2code2wav_async_chunk(tm, _mm(payload), req) is None
    assert len(tm.code_prompt_token_ids["r1"]) == _INITIAL_HOP + _PRE - 1


def test_first_chunk_fires_fast_with_zero_offset_and_switches_to_steady_hop():
    tm = _transfer_manager()
    req = _request("r1")
    out = funaudiochat2code2wav_async_chunk(
        tm, _mm(torch.tensor([5] * (_INITIAL_HOP + _PRE), dtype=torch.long)), req
    )
    assert out is not None
    # The initial chunk ships initial_hop + lookahead codes (13 at defaults).
    assert out.codes.audio.tolist() == [5] * (_INITIAL_HOP + _PRE)
    assert int(out.meta.left_context_size) == 0
    assert bool(out.meta.stream_finished.item()) is False
    state = tm.request_payload["r1"]["_fac_async_state"]
    # emitted advances by the initial hop only; the steady-state hop is now activated.
    assert state["emitted_token_len"] == _INITIAL_HOP
    assert state["token_hop_len"] == _HOP
    assert state["is_first_chunk"] is False


def test_second_chunk_uses_steady_state_hop_and_offset_equals_initial_hop():
    tm = _transfer_manager()
    req = _request("r1")
    funaudiochat2code2wav_async_chunk(
        tm, _mm(torch.tensor([5] * (_INITIAL_HOP + _PRE), dtype=torch.long)), req
    )
    # After the fast first chunk 13 codes accumulated, emitted=10, steady hop (50) active.
    # The second chunk fires once total frames reach emitted + steady_hop + pre = 63.
    # Feed one short of that (49 more codes -> 62 total); no emit yet.
    assert (
        funaudiochat2code2wav_async_chunk(
            tm, _mm(torch.tensor([5] * (_HOP - 1), dtype=torch.long)), req
        )
        is None
    )
    out = funaudiochat2code2wav_async_chunk(tm, _mm(torch.tensor([5], dtype=torch.long)), req)
    assert out is not None
    assert int(out.meta.left_context_size) == _INITIAL_HOP  # offset = 25
    assert len(out.codes.audio) == _INITIAL_HOP + _HOP + _PRE  # 63 — prefix = 63


def test_finish_flushes_remaining_codes_including_lookahead_without_loss():
    tm = _transfer_manager()
    req = _request("r1")
    # Emit the fast first chunk (13 codes; emitted -> 10, 3 lookahead held).
    funaudiochat2code2wav_async_chunk(
        tm, _mm(torch.tensor([5] * (_INITIAL_HOP + _PRE), dtype=torch.long)), req
    )
    # Finish with 10 extra codes: everything accumulated (23) must flush as a final chunk.
    out = funaudiochat2code2wav_async_chunk(
        tm, _mm(torch.tensor([5] * 10, dtype=torch.long)), req, is_finished=True
    )
    assert out is not None
    assert bool(out.meta.stream_finished.item()) is True
    assert int(out.meta.left_context_size) == _INITIAL_HOP  # 10
    assert len(out.codes.audio) == _INITIAL_HOP + _PRE + 10  # 23 — no code lost
    # After a final flush, subsequent calls return None (no duplicate terminal).
    assert funaudiochat2code2wav_async_chunk(tm, _mm(None), req, is_finished=True) is None


def test_text_only_finish_emits_empty_terminal_sentinel():
    tm = _transfer_manager()
    req = _request("r1")
    # No codec ids ever produced (text-only / ASR) -> an empty terminal sentinel once.
    out = funaudiochat2code2wav_async_chunk(tm, _mm(None), req, is_finished=True)
    assert out is not None
    assert out.codes.audio.numel() == 0
    assert bool(out.meta.stream_finished.item()) is True
    assert funaudiochat2code2wav_async_chunk(tm, _mm(None), req, is_finished=True) is None


def test_placeholder_steps_are_dropped_and_do_not_block_threshold():
    tm = _transfer_manager()
    req = _request("r1")
    # Interleave -1 placeholder groups; only the in-range codes count toward the hop.
    out = funaudiochat2code2wav_async_chunk(
        tm,
        _mm(
            torch.tensor(
                [[-1, -1, -1, -1, -1], [5] * (_INITIAL_HOP + _PRE)], dtype=torch.long
            )
        ),
        req,
    )
    assert out is not None
    assert out.codes.audio.tolist() == [5] * (_INITIAL_HOP + _PRE)


def test_filters_out_of_range_codec_ids():
    tm = _transfer_manager()
    req = _request("r1")
    # 6561+ ids are out of the CosyVoice3 codec range and must be filtered out.
    # Provide enough in-range ids to reach the initial hop + lookahead (13) so a chunk emits.
    ids = [5] * _INITIAL_HOP + [6561, 7000, 5, 6, 7]
    out = funaudiochat2code2wav_async_chunk(tm, _mm(torch.tensor(ids, dtype=torch.long)), req)
    assert out is not None
    # 10 leading + trailing in-range [5,6,7] (6561/7000 dropped) = 13.
    assert out.codes.audio.tolist() == [5] * _INITIAL_HOP + [5, 6, 7]


def test_request_finished_signal_takes_precedence_over_is_finished_kwarg():
    tm = _transfer_manager()
    req = _request("r1", finished=True)
    out = funaudiochat2code2wav_async_chunk(tm, _mm(torch.tensor([5] * 4, dtype=torch.long)), req)
    assert out is not None
    assert bool(out.meta.finished.item()) is True
    assert bool(out.meta.stream_finished.item()) is True


# --- yaml connector-extra override tests ------------------------------------------


def test_connector_extra_overrides_all_three_hops():
    # steady=50, initial=10, pre=4 -> first chunk fires at 14, then steady 50.
    tm = _transfer_manager_with_connector(steady=50, initial=10, pre=4)
    req = _request("r1")
    # 13 codes: one short of initial+pre (14) -> no emit.
    assert funaudiochat2code2wav_async_chunk(tm, _mm(torch.tensor([5] * 13)), req) is None
    out = funaudiochat2code2wav_async_chunk(tm, _mm(torch.tensor([5] * 1)), req)
    assert out is not None
    # First chunk carries initial(10) + pre(4) = 14 codes, offset 0.
    assert len(out.codes.audio) == 14
    assert int(out.meta.left_context_size) == 0
    state = tm.request_payload["r1"]["_fac_async_state"]
    assert state["emitted_token_len"] == 10  # advances by the initial hop
    assert state["token_hop_len"] == 50  # flipped to steady (not the 10 constant)
    assert state["pre_lookahead_len"] == 4
    assert state["flow_segment_len"] == _SEGMENT
    assert state["is_first_chunk"] is False


def test_config_safe_switch_when_initial_differs_from_steady():
    # The legacy `token_hop_len == _FAC_INITIAL_TOKEN_HOP_LEN` guard would NOT flip to
    # steady here (10 != 25), so second chunk would be sized wrong. Verify the flag-based
    # switch handles a configured initial != steady correctly.
    tm = _transfer_manager_with_connector(steady=50, initial=10, pre=4)
    req = _request("r1")
    # First chunk at 14 (initial 10 + pre 4).
    funaudiochat2code2wav_async_chunk(tm, _mm(torch.tensor([5] * 14)), req)
    # emitted=10; second chunk needs emitted+steady+pre = 10+50+4 = 64 total.
    # Feed until 63 total -> no emit; one more -> emit at 64.
    assert funaudiochat2code2wav_async_chunk(tm, _mm(torch.tensor([5] * 49)), req) is None  # 63 tot
    out = funaudiochat2code2wav_async_chunk(tm, _mm(torch.tensor([5] * 1)), req)  # 64 tot
    assert out is not None
    assert len(out.codes.audio) == 64
    assert int(out.meta.left_context_size) == 10  # offset == initial hop emitted
    assert int(out.meta.stream_finished.item()) is False


def test_no_connector_uses_defaults():
    # transfer_manager without .connector (unit-test stub / IPC fallback) falls back to the
    # _FAC_* defaults, so chunking is identical to the no-config path.
    tm = _transfer_manager()
    req = _request("r1")
    out = funaudiochat2code2wav_async_chunk(
        tm, _mm(torch.tensor([5] * (_INITIAL_HOP + _PRE))), req
    )
    assert out is not None
    assert len(out.codes.audio) == _INITIAL_HOP + _PRE
    state = tm.request_payload["r1"]["_fac_async_state"]
    assert state["token_hop_len"] == _HOP
    assert state["steady_hop_len"] == _HOP


def test_malformed_connector_config_falls_back_to_defaults():
    # A non-dict / junk connector.config must not crash the producer; it falls back.
    tm = _transfer_manager()
    tm.connector = SimpleNamespace(config="not-a-dict")
    req = _request("r1")
    out = funaudiochat2code2wav_async_chunk(
        tm, _mm(torch.tensor([5] * (_INITIAL_HOP + _PRE))), req
    )
    assert out is not None
    assert len(out.codes.audio) == _INITIAL_HOP + _PRE


def test_bounded_flow_segment_rolls_whole_segment_not_every_hop():
    tm = _transfer_manager_with_connector(
        steady=25,
        initial=10,
        pre=3,
        segment=50,
    )
    req = _request("r-segment")
    funaudiochat2code2wav_async_chunk(
        tm,
        _mm(torch.arange(120, dtype=torch.long) % 6000),
        req,
    )

    out = None
    while tm.request_payload["r-segment"]["_fac_async_state"]["emitted_token_len"] <= 60:
        out = funaudiochat2code2wav_async_chunk(tm, _mm(None), req)
        assert out is not None

    # emitted progresses 0 -> 10 -> 35 -> 60. At 60, the fixed segment starts
    # at token 50, so only ten already-emitted tokens are Flow context.
    assert int(out.meta.num_processed_tokens) == 50
    assert int(out.meta.left_context_size) == 10
    assert out.codes.audio.tolist() == list(range(50, 88))
    state = tm.request_payload["r-segment"]["_fac_async_state"]
    assert state["buffer_start_token"] == 50
    assert len(tm.code_prompt_token_ids["r-segment"]) == 70

    next_out = funaudiochat2code2wav_async_chunk(tm, _mm(None), req)
    assert next_out is not None
    # The segment start stays at 50 within the block; it does not slide by 25.
    assert int(next_out.meta.num_processed_tokens) == 50
    assert int(next_out.meta.left_context_size) == 35
    assert next_out.codes.audio.tolist() == list(range(50, 113))
