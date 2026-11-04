# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned input ledger for Qwen turn deltas and snapshot fallback.

Keep the multimodal inputs on the ordinary processor path: its sender cache
must observe the same media accesses as the EngineCore receiver cache.
"""

from vllm.tokenizers import TokenizerLike

from vllm_omni.engine import OmniEngineCoreRequest

DELTA_KEY = "_qwen_turn_delta"


class QwenIncrementalPrompt:
    """Append to accepted model tokens, or tokenize a changed history snapshot."""

    _boundary = "<|im_start|>"

    def __init__(self, tokenizer: TokenizerLike):
        self.tokenizer = tokenizer
        self._prefix = ""
        self._prefix_ids: list[int] = []
        self._prompt_ids: list[int] = []
        self._output_ids: list[int] = []
        self._finished = False
        self._append_safe = False

    def record_output(self, token_ids: list[int], finished: bool) -> None:
        self._output_ids.extend(token_ids)
        self._finished = self._finished or finished

    def can_append_turn(self, assistant_text: str) -> bool:
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        return bool(
            self._append_safe
            and self._finished
            and self._output_ids
            and self._output_ids[-1] == end_id
            and "<|im_end|>" in self.tokenizer.all_special_tokens
            and self.tokenizer.encode("<|im_end|>", add_special_tokens=False) == [end_id]
            and self.tokenizer.decode(self._output_ids[:-1], skip_special_tokens=False) == assistant_text
        )

    def prepare_turn(self, prompt: dict) -> tuple[dict, list[int] | None]:
        """Use the session-authorized delta, or process a replacement snapshot."""
        if prompt.get(DELTA_KEY):
            if not self._finished or not self._output_ids or not self._append_safe:
                raise ValueError("Turn delta has no completed resident token ledger")
            return prompt, self._prompt_ids + self._output_ids[:-1]
        return prompt, None

    def complete_turn(self, prompt: dict, request: OmniEngineCoreRequest, prefix: list[int] | None) -> None:
        if prefix is not None:
            request.append_prefix_length = len(prefix)
            request.prompt_token_ids = prefix + (request.prompt_token_ids or [])
        self._prompt_ids = list(request.prompt_token_ids or ())
        self._output_ids = []
        self._finished = False

    def _encode_full(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        # Tokenizers may omit chat delimiters from their special-token catalog
        # or omit AddedToken metadata. That disables text-prefix splitting,
        # but not continuation of the actual sampled token sequence.
        self._append_safe = ids == self.tokenizer.encode(text, add_special_tokens=False)
        return ids

    def tokenize(self, text: str) -> list[int]:
        tokenizer = self.tokenizer
        self._append_safe = False
        # Qwen's chat template embeds its special tokens explicitly. Refuse
        # the optimization if a tokenizer adds implicit BOS/EOS tokens.
        if self._boundary not in tokenizer.all_special_tokens:
            return self._encode_full(text)
        token = tokenizer.convert_tokens_to_ids(self._boundary)
        added = getattr(tokenizer, "added_tokens_decoder", {}).get(token)
        if added is None or added.lstrip or added.rstrip or added.single_word:
            # A token whose match depends on adjacent text is not a stable split.
            return self._encode_full(text)
        if self._prefix and text.startswith(self._prefix):
            ids = self._prefix_ids + tokenizer.encode(text[len(self._prefix) :], add_special_tokens=False)
        else:
            ids = self._encode_full(text)
            if not self._append_safe:
                self._prefix = ""
                self._prefix_ids = []
                return ids
        self._append_safe = True

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
