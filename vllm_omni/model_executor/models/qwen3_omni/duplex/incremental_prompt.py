# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned tokenization of append-only Qwen chat prompts.

Keep the multimodal inputs on the ordinary processor path: its sender cache
must observe the same media accesses as the EngineCore receiver cache.
"""

from vllm.tokenizers import TokenizerLike


class QwenIncrementalPrompt:
    """Reuse tokenization only across a verified special-token boundary."""

    _boundary = "<|im_start|>"

    def __init__(self, tokenizer: TokenizerLike):
        self.tokenizer = tokenizer
        self._prefix = ""
        self._prefix_ids: list[int] = []

    def tokenize(self, text: str) -> list[int]:
        tokenizer = self.tokenizer
        # Qwen's chat template embeds its special tokens explicitly. Refuse
        # the optimization if a tokenizer adds implicit BOS/EOS tokens.
        if self._boundary not in tokenizer.all_special_tokens:
            return tokenizer.encode(text, add_special_tokens=True)
        token = tokenizer.convert_tokens_to_ids(self._boundary)
        added = getattr(tokenizer, "added_tokens_decoder", {}).get(token)
        if added is None or added.lstrip or added.rstrip or added.single_word:
            # A token whose match depends on adjacent text is not a stable split.
            return tokenizer.encode(text, add_special_tokens=True)
        if self._prefix and text.startswith(self._prefix):
            ids = self._prefix_ids + tokenizer.encode(text[len(self._prefix) :], add_special_tokens=False)
        else:
            ids = tokenizer.encode(text, add_special_tokens=True)
            if ids != tokenizer.encode(text, add_special_tokens=False):
                self._prefix = ""
                self._prefix_ids = []
                return ids

        # End at an atomic special token, never an arbitrary text boundary:
        # BPE merges cannot cross it. Keep the generation header in the suffix
        # so assistant text from the previous turn is encoded canonically.
        end = text.rfind(self._boundary)
        if end >= 0 and token in ids:
            self._prefix = text[: end + len(self._boundary)]
            last = len(ids) - 1 - ids[::-1].index(token)
            self._prefix_ids = ids[: last + 1]
        else:
            self._prefix = ""
            self._prefix_ids = []
        return ids
