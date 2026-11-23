# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import random
import unittest
from types import SimpleNamespace as S

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext, LogitsProcRequestState

from vllm_omni.model_executor.logits_processor import NoRepeatNGramLogitsProcessor, banned_tokens

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class NoRepeatTests(unittest.TestCase):
    def _processor(self, history):
        tokens = torch.tensor(history)
        state = LogitsProcRequestState(
            torch.device("cpu"),
            len(history),
            10,
            S(gpu=tokens),
            S(np=np.zeros(len(history))),
            S(np=np.zeros(len(history))),
            S(gpu=torch.zeros(len(history))),
        )
        return NoRepeatNGramLogitsProcessor(S(speculative_config=None), state)

    def _context(self, slots, sequences):
        mapping = torch.tensor(slots)
        lengths = np.array([len(s) for s in sequences])
        return LogitsContext(
            mapping,
            mapping,
            np.array(slots),
            torch.zeros(len(slots)),
            torch.tensor([s[-1] for s in sequences]),
            torch.tensor(lengths - 1),
            lengths,
        )

    def test_disabled_requests_do_not_read_history_and_slot_reuse_clears_state(self):
        p = self._processor([[1, 2, 3], [1, 2, 3]])
        assert not p.add_request(1, S(extra_args={}))
        p.req_states = None
        logits = torch.zeros(1, 10)
        assert p.apply(logits, self._context([1], [[1, 2, 3]])) is logits

    def test_bounded_native_history_preserves_bans_across_prompt_output_boundary(self):
        rng = random.Random(42)
        for n in (1, 3, 10):
            for window in (1, 8, 30):
                for output_length in (0, 2, 50):
                    sequence = [rng.randrange(8) for _ in range(1000 + output_length)]
                    p = self._processor([sequence, sequence])
                    extras = dict(
                        no_repeat_ngram_size=n, no_repeat_ngram_window=window, no_repeat_ngram_whitelist=[0, -1, 999]
                    )
                    assert p.add_request(1, S(extra_args=extras))
                    logits = p.apply(torch.zeros(1, 10), self._context([1], [sequence]))
                    actual = set(torch.where(torch.isneginf(logits[0]))[0].tolist())
                    self.assertEqual(actual, banned_tokens(sequence, n, window, [0]))

    def test_prompt_history_and_whitelist(self):
        self.assertEqual(banned_tokens([1, 2, 3, 9, 1, 2], 3, 8192), {3})
        self.assertEqual(banned_tokens([1, 2, 3, 9, 1, 2], 3, 8192, [3]), set())
        self.assertEqual(banned_tokens([1, 2, 3, 9, 1, 2], 3, 2), set())
        self.assertEqual(banned_tokens([1, 2], 0, 8192), set())

    def test_native_history_slot_reorder_current_input_and_reuse(self):
        # The latest input token is deliberately not committed yet.
        p = self._processor([[0, 0, 0, 0, 0, 0], [1, 2, 3, 8, 1, 0]])
        params = S(extra_args={"no_repeat_ngram_size": 3})
        assert p.add_request(1, params)
        context = self._context([1, 0], [[1, 2, 3, 8, 1, 2], [0]])
        logits = p.apply(torch.zeros(2, 10), context)
        self.assertTrue(torch.isneginf(logits[0, 3]))
        self.assertTrue(torch.isfinite(logits[1]).all())
        assert not p.add_request(1, S(extra_args={}))
        self.assertTrue(torch.isfinite(p.apply(torch.zeros(2, 10), context)).all())

    def test_ten_token_repetition(self):
        seq = [7] * 10
        self.assertEqual(banned_tokens(seq, 10, 8192), {7})
        self.assertEqual(banned_tokens(seq[:9], 10, 8192), set())


if __name__ == "__main__":
    unittest.main()


def test_no_repeat_ngram_rejects_speculative_activation():
    proc = NoRepeatNGramLogitsProcessor(S(speculative_config=object()), None)
    assert not proc.add_request(0, S(extra_args={}))
    with pytest.raises(ValueError, match="non-speculative"):
        proc.add_request(0, S(extra_args={"no_repeat_ngram_size": 3}))
