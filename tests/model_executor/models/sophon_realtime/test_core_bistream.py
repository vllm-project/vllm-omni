# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bistream schedule (framework-independent core)."""

import ast
import pathlib

import pytest

from vllm_omni.model_executor.models.sophon_realtime.core.bistream import (
    REPEAT,
    BistreamSchedule,
    BistreamTokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

T = BistreamTokens(speech_vocab=20)  # sos 20, eos 21, task 22, fill 23


def _decode(st: BistreamSchedule, sampled: int) -> int:
    """One step: take the input, apply the plan, return the emitted token."""
    st.next_input()
    plan = st.plan()
    token = plan.forced if plan.forced is not None else sampled
    if not plan.discard:
        st.accept(token)
    return token


def test_fill_every_15_speech_tokens_then_5_text_tokens():
    st = BistreamSchedule(T)
    st.update_text(list(range(100, 120)), finished=False)
    assert [_decode(st, k) for k in range(15)] == list(range(15))
    assert _decode(st, 3) == T.fill
    assert st.queue == [("text", i) for i in range(100, 105)] and st.pos == 5
    # 4 intermediate positions (discarded), then the last text position is real
    assert [_decode(st, 7) for _ in range(4)] == [T.fill] * 4
    assert _decode(st, 7) == 7
    assert st.out[-2:] == [T.fill, 7]


def test_final_phase_repeats_input_feeds_rest_and_task():
    st = BistreamSchedule(T)
    st.update_text([100, 101], finished=True)
    for k in range(15):
        _decode(st, k)
    assert _decode(st, 0) == T.fill
    assert st.phase == "final"
    assert st.queue == [REPEAT, ("text", 100), ("text", 101), ("speech", T.task)]
    for _ in range(3):
        st.next_input()
        assert st.plan().discard
    st.next_input()
    assert st.plan().final and not st.plan().discard
    st.accept(T.eos)
    assert st.done


def test_starved_talker_waits_for_text():
    st = BistreamSchedule(T)
    st.update_text([100, 101], finished=False)
    for k in range(15):
        _decode(st, k)
    _decode(st, 0)
    assert st.starved == 1 and st.queue == [("text", 100), ("text", 101)]
    st.queue.clear()
    st.out.append(T.fill)  # a fill with nothing left to feed
    st._feed_text()
    assert st.queue == [REPEAT] and st.starved == 2


def test_text_never_shrinks():
    st = BistreamSchedule(T)
    st.update_text([1, 2, 3], finished=False)
    st.update_text([1, 2], finished=False)
    assert st.text == [1, 2, 3]
    st.update_text(None, finished=True)
    assert st.finished


def test_core_is_framework_independent():
    core = pathlib.Path(__file__).parents[4] / "vllm_omni/model_executor/models/sophon_realtime/core"
    for path in core.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in {"vllm", "vllm_omni", "torch"}, f"{path.name} imports {name}"
