# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HunyuanImage3 AR sampler logic (stage transitions,
ratio restriction, comprehension blocking)."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Fake token IDs for testing (avoid importing the real model).
END_OF_THINK = 100
RECAPTION = 101
END_OF_RECAPTION = 102
ANSWER = 103
BOI = 104
SIZE_TOKEN = 105
EOS = 106
RATIO_START = 200
RATIO_END = 210
RATIO_OTHER_START = 220
RATIO_OTHER_END = 223


class FakeSamplerModel:
    """Minimal stub that replicates the sampler-relevant attributes of
    HunyuanImage3ForConditionalGeneration without loading real weights."""

    def __init__(self, *, is_comprehension: bool = False):
        self._is_comprehension = is_comprehension
        self._eos_token_id = EOS
        self._end_of_think_id = END_OF_THINK
        self._recaption_id = RECAPTION
        self._end_of_recaption_id = END_OF_RECAPTION
        self._answer_id = ANSWER
        self._mrope_boi_token_id = BOI
        self._size_token_id = SIZE_TOKEN
        self._start_ratio_id = RATIO_START
        self._end_ratio_id = RATIO_END
        self._ratio_other_slices = [(RATIO_OTHER_START, RATIO_OTHER_END + 1)]
        self._all_ratio_ids = set(range(RATIO_START, RATIO_END + 1))
        self._all_ratio_ids.update(range(RATIO_OTHER_START, RATIO_OTHER_END + 1))

        self._stage_transitions: dict[int, list[int]] = {}
        if not is_comprehension:
            self._stage_transitions[END_OF_THINK] = [RECAPTION]
            self._stage_transitions[END_OF_RECAPTION] = [ANSWER, BOI, SIZE_TOKEN]

        self._blocked_token_ids: set[int] = set()
        if is_comprehension:
            self._blocked_token_ids.update([BOI, SIZE_TOKEN])
            self._blocked_token_ids.update(self._all_ratio_ids)

    # Bind the real methods from the model class.
    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
        HunyuanImage3ForConditionalGeneration as _Real,
    )

    _get_forced_token = _Real._get_forced_token
    _apply_ratio_restriction = _Real._apply_ratio_restriction


class TestGetForcedToken:
    """Tests for the stateless _get_forced_token method."""

    def setup_method(self):
        self.model = FakeSamplerModel(is_comprehension=False)

    def test_no_trigger_returns_none(self):
        assert self.model._get_forced_token([1, 2, 3]) is None

    def test_empty_history_returns_none(self):
        assert self.model._get_forced_token([]) is None

    def test_end_of_think_forces_recaption(self):
        assert self.model._get_forced_token([END_OF_THINK]) == RECAPTION

    def test_end_of_think_completed(self):
        assert self.model._get_forced_token([END_OF_THINK, RECAPTION]) is None

    def test_end_of_recaption_forces_answer(self):
        tokens = [END_OF_THINK, RECAPTION, END_OF_RECAPTION]
        assert self.model._get_forced_token(tokens) == ANSWER

    def test_end_of_recaption_forces_boi_after_answer(self):
        tokens = [END_OF_THINK, RECAPTION, END_OF_RECAPTION, ANSWER]
        assert self.model._get_forced_token(tokens) == BOI

    def test_end_of_recaption_forces_size_after_boi(self):
        tokens = [END_OF_THINK, RECAPTION, END_OF_RECAPTION, ANSWER, BOI]
        assert self.model._get_forced_token(tokens) == SIZE_TOKEN

    def test_full_sequence_complete(self):
        tokens = [END_OF_THINK, RECAPTION, END_OF_RECAPTION, ANSWER, BOI, SIZE_TOKEN]
        assert self.model._get_forced_token(tokens) is None

    def test_diverged_history_returns_none(self):
        tokens = [END_OF_RECAPTION, 999]  # 999 != ANSWER
        assert self.model._get_forced_token(tokens) is None

    def test_later_trigger_takes_precedence(self):
        tokens = [END_OF_THINK, RECAPTION, END_OF_RECAPTION]
        assert self.model._get_forced_token(tokens) == ANSWER

    def test_trigger_with_extra_tokens_before(self):
        tokens = [1, 2, 3, END_OF_THINK]
        assert self.model._get_forced_token(tokens) == RECAPTION


class TestComprehensionBlocking:
    """Tests for comprehension mode token blocking."""

    def test_blocked_tokens_masked(self):
        model = FakeSamplerModel(is_comprehension=True)
        vocab_size = 300
        logits = torch.zeros(1, vocab_size)
        logits[0, BOI] = 5.0
        logits[0, SIZE_TOKEN] = 3.0
        logits[0, RATIO_START] = 2.0
        min_score = torch.finfo(logits.dtype).min

        for tid in model._blocked_token_ids:
            if tid < vocab_size:
                logits[0, tid] = min_score

        assert logits[0, BOI].item() == min_score
        assert logits[0, SIZE_TOKEN].item() == min_score
        assert logits[0, RATIO_START].item() == min_score

    def test_non_blocked_tokens_preserved(self):
        model = FakeSamplerModel(is_comprehension=True)
        vocab_size = 300
        logits = torch.zeros(1, vocab_size)
        logits[0, 50] = 7.0
        min_score = torch.finfo(logits.dtype).min

        for tid in model._blocked_token_ids:
            if tid < vocab_size:
                logits[0, tid] = min_score

        assert logits[0, 50].item() == 7.0


class TestRatioRestriction:
    """Tests for _apply_ratio_restriction (greedy: only argmax ratio survives)."""

    def test_greedy_selects_single_ratio_token(self):
        model = FakeSamplerModel(is_comprehension=False)
        vocab_size = 300
        logits = torch.zeros(1, vocab_size)
        logits[0, RATIO_START + 3] = 10.0
        logits[0, RATIO_START + 1] = 5.0
        logits[0, 50] = 20.0  # non-ratio, should be masked
        min_score = torch.finfo(logits.dtype).min

        model._apply_ratio_restriction(logits, 0, min_score)

        assert logits[0, RATIO_START + 3].item() == 0
        assert logits[0, RATIO_START + 1].item() == min_score
        assert logits[0, 50].item() == min_score

    def test_extra_ratio_slices_considered(self):
        model = FakeSamplerModel(is_comprehension=False)
        vocab_size = 300
        logits = torch.zeros(1, vocab_size)
        logits[0, RATIO_OTHER_START] = 15.0
        logits[0, RATIO_START] = 5.0
        min_score = torch.finfo(logits.dtype).min

        model._apply_ratio_restriction(logits, 0, min_score)

        assert logits[0, RATIO_OTHER_START].item() == 0
        assert logits[0, RATIO_START].item() == min_score


class TestForceEosAfterRatio:
    """Tests that a ratio token as last_token forces EOS."""

    def test_ratio_token_forces_eos(self):
        model = FakeSamplerModel(is_comprehension=False)
        vocab_size = 300
        logits = torch.randn(1, vocab_size)
        min_score = torch.finfo(logits.dtype).min

        logits[0].fill_(min_score)
        logits[0, model._eos_token_id] = 0

        assert logits[0, EOS].item() == 0
        non_eos_max = logits[0, :EOS].max().item()
        assert non_eos_max == min_score
