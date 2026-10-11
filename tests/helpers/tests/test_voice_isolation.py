# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for tests.helpers.voice_isolation request planning and first-window gating. No server or model."""

import pytest

from tests.helpers.speaker_similarity import LABEL_CORRECT, LABEL_DISAGREE, LABEL_WRONG_BOTH_AGREE, OutputScore
from tests.helpers.voice_isolation import VoiceIsolationSuite, gate_first_window, server_mode

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _suite() -> VoiceIsolationSuite:
    return VoiceIsolationSuite(
        model="m",
        voices={"a": ("a.wav", "ref a"), "b": ("b.wav", "ref b"), "c": ("c.wav", "ref c")},
        texts=["t0", "t1", "t2", "t3"],
        n_requests=12,
        base_seed=2,
    )


def _score(idx: int, label: str) -> OutputScore:
    wrong = "b" if label == LABEL_WRONG_BOTH_AGREE else None
    return OutputScore(
        idx=idx,
        voice="a",
        duration_s=3.0,
        scored_s=2.0,
        window_short=False,
        voices=["a", "b"],
        sims={},
        argmax={},
        margin={},
        label=label,
        wrong_voice=wrong,
    )


def test_plan_requests_round_robin_with_decorrelated_texts() -> None:
    plan = _suite().plan_requests()
    assert [r["voice"] for r in plan] == ["a", "b", "c"] * 4
    assert [r["text_idx"] for r in plan] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert [r["seed"] for r in plan] == [2000 + i for i in range(12)]
    assert all(r["text"] == f"t{r['text_idx']}" for r in plan)


def test_request_kwargs_per_source() -> None:
    suite = _suite()
    req = {"voice": "b", "seed": 7}
    urls = {"a": "data:a", "b": "data:b", "c": "data:c"}
    assert suite.request_kwargs(req, urls, "registered") == ("voice_isolation_b", {"seed": 7})
    assert suite.request_kwargs(req, urls, "inline") == (None, {"ref_audio": "data:b", "ref_text": "ref b", "seed": 7})
    assert suite.request_kwargs(req, urls, "inline_xvec") == (
        None,
        {"ref_audio": "data:b", "x_vector_only_mode": True, "seed": 7},
    )
    with pytest.raises(ValueError, match="unknown voice source"):
        suite.request_kwargs(req, urls, "speaker_embedding")


def test_server_mode() -> None:
    assert server_mode(["--trust-remote-code"]) == "async_chunk"
    assert server_mode(None) == "async_chunk"
    assert server_mode(["--trust-remote-code", "--no-async-chunk"]) == "no_async_chunk"


def test_gate_first_window_counts_only_openings_right_at_c1() -> None:
    mismatched = [_score(0, LABEL_WRONG_BOTH_AGREE), _score(2, LABEL_WRONG_BOTH_AGREE)]
    baseline = [_score(0, LABEL_CORRECT), _score(1, LABEL_CORRECT), _score(2, LABEL_DISAGREE)]
    leaks, excused = gate_first_window(mismatched, baseline)
    assert [r.idx for r in leaks] == [0]
    assert [(r.idx, base.label) for r, base in excused] == [(2, LABEL_DISAGREE)]
