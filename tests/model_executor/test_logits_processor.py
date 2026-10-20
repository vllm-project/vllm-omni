# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import random
import unittest
from types import SimpleNamespace as S

import pytest
import torch

from vllm_omni.model_executor.logits_processor import MoveDirectionality, NoRepeatNGramLogitsProcessor, banned_tokens

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class NoRepeatTests(unittest.TestCase):
    def test_disabled_requests_are_not_read_and_moves_allow_absent_entries(self):
        class Unreadable(list):
            def __iter__(self):
                raise AssertionError("disabled history was copied")

        p = NoRepeatNGramLogitsProcessor(None, None, False)
        p.update_state(S(removed=[], moved=[], added=[(0, S(extra_args={}), Unreadable([1]), Unreadable([2]))]))
        p.apply(torch.zeros(2, 8))
        self.assertEqual(p.requests, {})
        p.update_state(S(removed=[], added=[], moved=[(0, 1, MoveDirectionality.SWAP)]))
        p.update_state(S(removed=[], added=[], moved=[(1, 0, MoveDirectionality.UNIDIRECTIONAL)]))
        self.assertEqual(p.requests, {})

    def test_bounded_history_preserves_bans_across_prompt_output_boundary(self):
        rng = random.Random(42)

        class BoundedHistory(list):
            def __iter__(self):
                raise AssertionError("entire history was copied")

            def __getitem__(self, key):
                if isinstance(key, slice):
                    assert key.start is not None and key.start >= -40
                return super().__getitem__(key)

        for n in (1, 3, 10):
            for window in (1, 8, 30):
                for output_length in (0, 2, 50):
                    prompt = [rng.randrange(8) for _ in range(1000)]
                    output = [rng.randrange(8) for _ in range(output_length)]
                    p = NoRepeatNGramLogitsProcessor(None, None, False)
                    extras = dict(no_repeat_ngram_size=n, no_repeat_ngram_window=window, no_repeat_ngram_whitelist=[0])
                    p.update_state(
                        S(
                            removed=[],
                            moved=[],
                            added=[(0, S(extra_args=extras), BoundedHistory(prompt), BoundedHistory(output))],
                        )
                    )
                    logits = p.apply(torch.zeros(1, 8))
                    actual = set(torch.where(torch.isneginf(logits[0]))[0].tolist())
                    self.assertEqual(actual, banned_tokens(prompt + output, n, window, [0]))

    def test_prompt_history_and_whitelist(self):
        self.assertEqual(banned_tokens([1, 2, 3, 9, 1, 2], 3, 8192), {3})
        self.assertEqual(banned_tokens([1, 2, 3, 9, 1, 2], 3, 8192, [3]), set())
        self.assertEqual(banned_tokens([1, 2, 3, 9, 1, 2], 3, 2), set())
        self.assertEqual(banned_tokens([1, 2], 0, 8192), set())

    def test_live_output_and_batch_moves(self):
        p = NoRepeatNGramLogitsProcessor(None, None, False)
        output: list[int] = []
        params = S(extra_args={"no_repeat_ngram_size": 3, "no_repeat_ngram_window": 8192})
        p.update_state(
            S(removed=[], added=[(0, params, [1, 2, 3, 8], output), (1, S(extra_args={}), [], [])], moved=[])
        )
        output.extend([1, 2])
        x = torch.zeros(2, 10)
        p.apply(x)
        self.assertTrue(torch.isneginf(x[0, 3]))
        self.assertTrue(torch.isfinite(x[1]).all())
        p.update_state(S(removed=[], added=[], moved=[(0, 1, MoveDirectionality.SWAP)]))
        x = torch.zeros(2, 10)
        p.apply(x)
        self.assertTrue(torch.isneginf(x[1, 3]))
        self.assertTrue(torch.isfinite(x[0]).all())
        p.update_state(S(removed=[0], added=[], moved=[(1, 0, MoveDirectionality.UNIDIRECTIONAL)]))
        self.assertIs(p.requests[0][1], output)
        p.update_state(S(removed=[0], added=[], moved=[]))
        self.assertEqual(p.requests, {})

    def test_ten_token_repetition(self):
        seq = [7] * 10
        self.assertEqual(banned_tokens(seq, 10, 8192), {7})
        self.assertEqual(banned_tokens(seq[:9], 10, 8192), set())


if __name__ == "__main__":
    unittest.main()
