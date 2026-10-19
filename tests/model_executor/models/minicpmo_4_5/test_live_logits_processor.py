# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Live duplex decoding: forced listens, unit cut-offs, turn state per session, one-hot rows."""

from collections import OrderedDict

import pytest
import torch
from vllm import SamplingParams

from vllm_omni.model_executor.models.minicpmo_4_5.live import logits_processor as lp
from vllm_omni.model_executor.models.minicpmo_4_5.live.logits_processor import (
    FORCE_LISTEN_COUNT,
    UNIT_TEXT_LIMIT,
    MiniCPMOLiveLogitsProcessor,
    _RowState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LISTEN, TTS_BOS, CHUNK_EOS, CHUNK_TTS_EOS, TURN_EOS, TTS_PAD = 1, 2, 3, 4, 5, 6
TEXT_A, TEXT_B, TEXT_C = 10, 11, 12
VOCAB = 20
TEXT_CHARS = 10


class _Tokenizer:
    """Every token from TEXT_A up decodes to TEXT_CHARS characters; lower ids are special."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join("x" * TEXT_CHARS for token in ids if token >= TEXT_A or not skip_special_tokens)


def _processor() -> MiniCPMOLiveLogitsProcessor:
    proc = object.__new__(MiniCPMOLiveLogitsProcessor)
    proc.device = torch.device("cpu")
    proc._rows = {}
    proc._sessions = OrderedDict()
    proc._tokenizer = _Tokenizer()
    proc.listen, proc.tts_bos, proc.chunk_eos = LISTEN, TTS_BOS, CHUNK_EOS
    proc.chunk_tts_eos, proc.turn_eos = CHUNK_TTS_EOS, TURN_EOS
    proc.terminators = {LISTEN, CHUNK_EOS, CHUNK_TTS_EOS}
    proc._forbidden = torch.tensor([CHUNK_EOS, TTS_PAD], dtype=torch.long)
    return proc


def _row(proc, *, session="s", prefix=False, unit_index=FORCE_LISTEN_COUNT, max_unit_tokens=16, step=0):
    return _RowState(
        session=proc._session(session),
        prefix=prefix,
        unit_index=unit_index,
        max_unit_tokens=max_unit_tokens,
        step=step,
    )


def _logits(**scores) -> torch.Tensor:
    logits = torch.zeros(VOCAB)
    for token, score in scores.items():
        logits[globals()[token]] = score
    return logits


def test_prefix_rows_always_listen():
    proc = _processor()
    row = _row(proc, prefix=True, step=3)
    assert proc._decode(row, _logits(TEXT_A=9.0)) == LISTEN


@pytest.mark.parametrize("unit_index", range(FORCE_LISTEN_COUNT))
def test_first_units_are_forced_to_listen(unit_index):
    proc = _processor()
    assert proc._decode(_row(proc, unit_index=unit_index), _logits(TEXT_A=9.0)) == LISTEN


def test_forced_listen_only_applies_to_the_first_token():
    proc = _processor()
    assert proc._decode(_row(proc, unit_index=0, step=1), _logits(TEXT_A=9.0)) == TEXT_A


def test_units_after_the_forced_ones_decode():
    proc = _processor()
    assert proc._decode(_row(proc), _logits(TEXT_A=9.0)) == TEXT_A


def test_last_token_of_a_unit_is_chunk_eos():
    proc = _processor()
    row = _row(proc, max_unit_tokens=3, step=2)
    assert proc._decode(row, _logits(TEXT_A=9.0)) == CHUNK_EOS
    assert row.session.history == []


def test_raw_argmax_chunk_eos_is_taken_without_joining_history():
    proc = _processor()
    row = _row(proc)
    assert proc._decode(row, _logits(CHUNK_EOS=9.0, TEXT_A=8.0)) == CHUNK_EOS
    assert row.session.history == []


def test_forbidden_tokens_are_masked():
    proc = _processor()
    assert proc._decode(_row(proc), _logits(TTS_PAD=9.0, TEXT_B=1.0)) == TEXT_B


def test_repetition_penalty_demotes_recent_tokens():
    proc = _processor()
    row = _row(proc)
    assert proc._decode(row, _logits(TEXT_A=5.0, TEXT_B=4.9)) == TEXT_A
    row.step = 1
    assert proc._decode(row, _logits(TEXT_A=5.0, TEXT_B=4.9)) == TEXT_B
    assert row.session.history == [TEXT_A, TEXT_B]


def test_listen_before_turn_end_becomes_tts_bos():
    proc = _processor()
    row = _row(proc)
    row.session.turn_ended = False
    assert proc._decode(row, _logits(LISTEN=9.0)) == TTS_BOS
    assert row.session.turn_ended is False


def test_listen_after_turn_end_stays_listen():
    proc = _processor()
    row = _row(proc)
    assert row.session.turn_ended is True
    assert proc._decode(row, _logits(LISTEN=9.0)) == LISTEN
    assert row.session.turn_ended is True


def test_unit_text_limit_cuts_the_unit():
    proc = _processor()
    row = _row(proc)
    tokens = []
    for _ in range(8):
        token = proc._decode(row, _logits(TEXT_A=9.0, TEXT_B=8.0, TEXT_C=7.0))
        row.step += 1
        tokens.append(token)
        if token in proc.terminators:
            break
    # The first token is not counted; two more reach 20 characters, a third would reach 30.
    assert 2 * TEXT_CHARS < UNIT_TEXT_LIMIT <= 3 * TEXT_CHARS
    assert tokens == [TEXT_A, TEXT_A, TEXT_A, CHUNK_EOS]
    assert row.unit_ids == [TEXT_A, TEXT_A]


def test_turn_state_spans_units_of_one_session():
    proc = _processor()
    speaking = _row(proc, session="a")
    assert proc._decode(speaking, _logits(TEXT_A=9.0)) == TEXT_A
    assert speaking.session.turn_ended is False

    next_unit = _row(proc, session="a")
    assert next_unit.session is speaking.session
    assert proc._decode(next_unit, _logits(LISTEN=9.0)) == TTS_BOS

    other = _row(proc, session="b")
    assert other.session.turn_ended is True
    assert proc._decode(other, _logits(LISTEN=9.0)) == LISTEN


def test_turn_eos_ends_the_turn():
    proc = _processor()
    row = _row(proc, step=1)
    row.session.turn_ended = False
    assert proc._decode(row, _logits(TURN_EOS=9.0)) == TURN_EOS
    assert row.session.turn_ended is True
    later = _row(proc)
    assert proc._decode(later, _logits(LISTEN=9.0)) == LISTEN


def test_apply_writes_one_hot_rows_and_marks_terminated_rows_done():
    proc = _processor()
    proc._rows[0] = _row(proc, session="a")
    proc._rows[2] = _row(proc, session="b", unit_index=0)
    logits = torch.stack([_logits(TEXT_A=9.0), _logits(TEXT_B=9.0), _logits(TEXT_C=9.0)])
    untouched = logits[1].clone()

    out = proc.apply(logits)

    assert torch.argmax(out[0]) == TEXT_A
    assert torch.isinf(out[0]).sum() == VOCAB - 1
    assert out[0, TEXT_A] == 9.0
    assert torch.argmax(out[2]) == LISTEN
    assert torch.isinf(out[2]).sum() == VOCAB - 1
    assert torch.equal(out[1], untouched)
    assert proc._rows[0].step == 1 and not proc._rows[0].done
    assert proc._rows[2].done


def test_apply_gives_a_masked_choice_a_finite_logit():
    proc = _processor()
    proc._rows[0] = _row(proc, max_unit_tokens=1)
    logits = _logits(TEXT_A=9.0).unsqueeze(0)
    logits[0, CHUNK_EOS] = float("-inf")
    out = proc.apply(logits)
    assert out[0, CHUNK_EOS] == 0.0
    assert torch.isinf(out[0]).sum() == VOCAB - 1


def test_apply_skips_done_rows_and_rows_beyond_the_batch():
    proc = _processor()
    done = _row(proc)
    done.done = True
    proc._rows[0] = done
    proc._rows[5] = _row(proc)
    logits = _logits(TEXT_A=9.0).unsqueeze(0)
    out = proc.apply(logits.clone())
    assert torch.equal(out, logits)
    assert done.step == 0 and proc._rows[5].step == 0


def test_requests_without_the_live_spec_get_no_row():
    proc = _processor()
    proc._tokenizer = None
    assert proc._new_row(SamplingParams(), None, []) is None
    assert proc._new_row(SamplingParams(extra_args={"other": 1}), None, []) is None
    assert proc._new_row(None, None, []) is None
    assert proc._tokenizer is None


def test_new_row_reads_the_live_spec():
    proc = _processor()
    spec = {"session": "s1", "unit_index": 4, "max_unit_tokens": 7}
    row = proc._new_row(SamplingParams(extra_args={"minicpmo_live": spec}), None, [])
    assert (row.prefix, row.unit_index, row.max_unit_tokens, row.step) == (False, 4, 7, 0)
    prefix = proc._new_row(SamplingParams(extra_args={"minicpmo_live": {"session": "s1", "prefix": True}}), None, [])
    assert prefix.prefix and prefix.session is row.session


def test_oldest_session_is_evicted(monkeypatch):
    monkeypatch.setattr(lp, "MAX_SESSIONS", 2)
    proc = _processor()
    first = proc._session("a")
    proc._session("b")
    assert proc._session("a") is first
    proc._session("c")
    assert list(proc._sessions) == ["a", "c"]


def test_is_not_argmax_invariant():
    assert _processor().is_argmax_invariant() is False


@pytest.mark.parametrize(
    "extra_args",
    [
        None,
        {"other": 1},
        {"minicpmo_live": {"session": "s", "prefix": True}},
        {"minicpmo_live": {"session": "s", "max_unit_tokens": 1}},
    ],
)
def test_validate_params_accepts(extra_args):
    MiniCPMOLiveLogitsProcessor.validate_params(SamplingParams(extra_args=extra_args))


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"max_unit_tokens": 4}, "session id"),
        ({"session": "", "max_unit_tokens": 4}, "session id"),
        ("s", "session id"),
        ({"session": "s"}, "max_unit_tokens"),
        ({"session": "s", "max_unit_tokens": 0}, "max_unit_tokens"),
    ],
)
def test_validate_params_rejects(spec, message):
    with pytest.raises(ValueError, match=message):
        MiniCPMOLiveLogitsProcessor.validate_params(SamplingParams(extra_args={"minicpmo_live": spec}))
