# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CrossTurnPenalty."""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.stage_input_processors.aura_cross_turn_penalty import CrossTurnPenalty

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTokenizer:
    all_special_ids = [0, 1]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(c) for c in text if c.isalnum() or ord(c) > 127]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(tid) for tid in token_ids)


def test_cross_turn_penalty_builds_logit_bias_for_repeated_tokens():
    penalty = CrossTurnPenalty(_FakeTokenizer(), window=2, logit_penalty=2.0)
    penalty.record("hello world")
    penalty.record("hello again")

    kwargs = penalty.build_sampling_kwargs()

    assert "logit_bias" in kwargs
    assert any(bias < 0 for bias in kwargs["logit_bias"].values())


def test_cross_turn_penalty_records_silent_turn_as_none():
    penalty = CrossTurnPenalty(_FakeTokenizer(), window=2, logit_penalty=2.0)
    penalty.record("spoken")
    penalty.record(None)

    assert penalty._spoken_history() == ["spoken"]
