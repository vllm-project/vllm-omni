# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Added-token control flags apply to fixed and generated lexical routes."""

from dataclasses import dataclass

import pytest

from vllm_omni.model_executor.models.drvexo_speech_8b.lexical import GeneratedTokenRouter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class AddedToken:
    special: bool


class Tokenizer:
    all_special_ids = [3]
    eos_token_id = 3
    added_tokens_decoder = {2: AddedToken(special=True), 1: AddedToken(special=False)}

    def get_vocab(self):
        return {"hello": 1, "<|im_start|>": 2, "<|im_end|>": 3}

    def encode(self, text, **kwargs):
        if text.startswith("<|im_start|>assistant"):
            return list(range(9))
        return [self.get_vocab()[text]]

    def decode(self, ids, **kwargs):
        inverse = {value: key for key, value in self.get_vocab().items()}
        return "".join(inverse[index] for index in ids)


def test_generated_and_fixed_routes_preserve_words_and_reject_added_controls():
    router = GeneratedTokenRouter(Tokenizer(), Tokenizer())
    expected = list(range(3)) + [1] + list(range(4, 9))
    assert router.prepare_fixed("hello") == expected
    assert router.prepare([1, 3], "hello") == expected
    assert 2 not in router.mapping
    with pytest.raises(ValueError, match="control tokens"):
        router.prepare_fixed("<|im_start|>")
    with pytest.raises(ValueError, match="unmapped"):
        router.prepare([2], "<|im_start|>")
