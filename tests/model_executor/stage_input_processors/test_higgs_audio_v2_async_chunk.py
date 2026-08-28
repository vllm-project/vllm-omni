# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async-chunk save path for higgs-audio v2.

The path accumulates one codec frame per AR step and flushes when
``length % chunk_size`` hits a boundary. Frames carrying stream specials are
dropped and must not be counted, or every chunk boundary moves -- these tests
pin that, plus the batching that keeps the drop test off the per-step path.
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.higgs_audio_v2 import (
    _NUM_CODEBOOKS,
    _NUM_REAL_CODES,
    talker2code2wav_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _req(rid: str, *, finished: bool):
    return SimpleNamespace(external_req_id=rid, is_finished=lambda: finished)


def _tm(*, chunk_frames=4, left_context=0):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context,
                }
            }
        ),
    )


def _frame(value: int) -> torch.Tensor:
    return torch.full((1, _NUM_CODEBOOKS), value, dtype=torch.long)


_SPECIAL = _NUM_REAL_CODES  # 1024: a stream special, must never be counted


def _feed(tm, rid, values, *, finished_on_last=False):
    """Drive the save path one AR step at a time, as the talker does."""
    emitted = []
    for i, v in enumerate(values):
        last = i == len(values) - 1
        payload = talker2code2wav_async_chunk(
            transfer_manager=tm,
            multimodal_output={"codes": {"audio": _frame(v)}},
            request=_req(rid, finished=finished_on_last and last),
            is_finished=finished_on_last and last,
        )
        if payload is not None:
            emitted.append(payload.codes.audio.tolist())
    return emitted


def test_stream_specials_do_not_advance_the_chunk_boundary():
    """A special-carrying frame must not count toward the emission threshold.

    With chunk_frames=4 and a special at step 1, the fourth *real* frame lands
    at step 4. Counting the special would emit one step early carrying three
    frames instead of four.
    """
    tm = _tm(chunk_frames=4)
    emitted = _feed(tm, "r-special", (1, _SPECIAL, 2, 3, 4))

    assert len(emitted) == 1, "exactly one chunk should flush across these five steps"
    # Codebook-major over the four real frames only.
    assert emitted[0] == [1, 2, 3, 4] * _NUM_CODEBOOKS
    assert len(tm.code_prompt_token_ids["r-special"]) == 4


def test_specials_only_stream_emits_nothing_until_finished():
    tm = _tm(chunk_frames=4)
    assert _feed(tm, "r-all", (_SPECIAL, _SPECIAL, _SPECIAL)) == []
    assert len(tm.code_prompt_token_ids["r-all"]) == 0


def test_terminal_specials_yield_the_empty_finished_payload():
    tm = _tm(chunk_frames=4)
    emitted = _feed(tm, "r-term", (_SPECIAL, _SPECIAL), finished_on_last=True)
    assert emitted == [[]]


def test_negative_pads_are_dropped_like_the_scalar_test_did():
    """The old per-frame test rejected on ``frame.min() < 0`` too."""
    tm = _tm(chunk_frames=2)
    emitted = _feed(tm, "r-neg", (5, -1, 6))
    assert emitted == [[5, 6] * _NUM_CODEBOOKS]
    assert len(tm.code_prompt_token_ids["r-neg"]) == 2


def test_specials_are_invisible_to_the_emission_schedule():
    """The contract: a stream with specials emits what the clean stream emits."""
    real = list(range(1, 13))

    def run(values, rid):
        return _feed(_tm(chunk_frames=4), rid, values)

    withspecials = []
    for i, v in enumerate(real):
        if i % 3 == 0:
            withspecials.append(_SPECIAL)
        withspecials.append(v)

    assert run(withspecials, "r-mixed") == run(real, "r-clean")


def test_drop_test_is_batched_not_per_step(monkeypatch):
    """Syncs scale with chunks, not AR steps.

    The pre-change path did two ``.item()`` calls per step to decide whether to
    keep the frame; the batched resolve does one ``.tolist()`` per resolve.
    """
    calls = {"n": 0}
    real_tolist = torch.Tensor.tolist

    def counting(self):
        calls["n"] += 1
        return real_tolist(self)

    monkeypatch.setattr(torch.Tensor, "tolist", counting)

    tm = _tm(chunk_frames=25)
    _feed(tm, "r-sync", range(1, 61))

    assert calls["n"] <= 10, f"expected batched syncs, saw {calls['n']} across 60 steps"
    committed = tm.code_prompt_token_ids["r-sync"]
    pending = tm.pending_frames.get("r-sync", [])
    assert len(committed) + len(pending) == 60, "every frame is in exactly one bucket"


def test_emitted_layout_is_codebook_major():
    """Element order must match the Python double loop this replaced.

    Old form was ``[window[f][q] for q in range(Q) for f in range(F)]``, i.e.
    all frames of codebook 0, then codebook 1, and so on.
    """
    tm = _tm(chunk_frames=3)
    rid = "r-layout"
    for step, row in enumerate(([1, 2, 3, 4, 5, 6, 7, 8], [11, 12, 13, 14, 15, 16, 17, 18])):
        talker2code2wav_async_chunk(
            transfer_manager=tm,
            multimodal_output={"codes": {"audio": torch.tensor([row], dtype=torch.long)}},
            request=_req(rid, finished=False),
            is_finished=False,
        )
    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.tensor([[21, 22, 23, 24, 25, 26, 27, 28]], dtype=torch.long)}},
        request=_req(rid, finished=False),
        is_finished=False,
    )
    assert payload is not None
    window = [[1, 2, 3, 4, 5, 6, 7, 8], [11, 12, 13, 14, 15, 16, 17, 18], [21, 22, 23, 24, 25, 26, 27, 28]]
    expected = [window[f][q] for q in range(_NUM_CODEBOOKS) for f in range(len(window))]
    assert payload.codes.audio.tolist() == expected
