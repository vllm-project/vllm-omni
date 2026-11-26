# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Scheduler-owned text input for the interleaved Qwen3-TTS Talker."""


class Qwen3TTSTextInput:
    """Scheduler-owned text and the stock interleaved Talker's readiness rule."""

    def __init__(self, token_ids, capacity, vocab_size, *, final=False):
        self.token_ids = []
        self.capacity = capacity
        self.vocab_size = vocab_size
        self.final = False
        self.wait_started_at = None
        self.delivered_tokens = 0
        self.delivered_final = False
        self.append(token_ids, final=final)

    def append(self, tokens: list[int], *, final: bool = False) -> None:
        if type(final) is not bool:
            raise ValueError("Text input requires boolean final")
        if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Text input requires nonnegative token IDs")
        if any(token >= self.vocab_size for token in tokens):
            raise ValueError("Input token ID exceeds the model's text vocabulary")
        if self.final and (tokens or not final):
            raise ValueError("Cannot append after input EOF")
        if not tokens and not final:
            raise ValueError("An empty text update must close the input")
        if len(self.token_ids) + len(tokens) > self.capacity:
            raise ValueError("Text input exceeds its token capacity")
        self.token_ids.extend(tokens)
        self.final = final

    def is_ready(self, request) -> bool:
        if self.final:
            return True
        if request.num_computed_tokens < request.num_prompt_tokens:
            return not request.num_preemptions or request.num_output_tokens < len(self.token_ids)
        next_token = request.num_computed_tokens - request.num_prompt_tokens + 1
        return next_token < len(self.token_ids)

    def take_update(self, *, reset: bool) -> dict[str, object] | None:
        offset = 0 if reset else self.delivered_tokens
        if not reset and offset == len(self.token_ids) and self.delivered_final == self.final:
            return None
        update = {"token_ids": self.token_ids[offset:], "final": self.final}
        self.delivered_tokens = len(self.token_ids)
        self.delivered_final = self.final
        return update
