# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request-controlled logits processors."""

from vllm.v1.sample.logits_processor.interface import (
    LogitsProcessor,
    MoveDirectionality,
)


def banned_tokens(sequence, n, window, whitelist=()):
    if n <= 0 or window <= 0 or len(sequence) < n:
        return set()
    prefix = sequence[-(n - 1) :] if n > 1 else []
    banned = {
        sequence[i + n - 1]
        for i in range(max(0, len(sequence) - window), len(sequence) - n + 1)
        if n == 1 or sequence[i : i + n - 1] == prefix
    }
    return banned.difference(whitelist)


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Prevent repeated n-grams across prompt and generated tokens.

    Configure via SamplingParams.extra_args: no_repeat_ngram_size (default 0,
    disabled), no_repeat_ngram_window (default 8192), and
    no_repeat_ngram_whitelist (token IDs exempt from the restriction).
    """

    def __init__(self, vllm_config, device, is_pin_memory):
        self.requests = {}

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.requests.pop(index, None)
        for index, params, prompt_ids, output_ids in batch_update.added:
            extras = params.extra_args or {}
            n = int(extras.get("no_repeat_ngram_size", 0))
            window = int(extras.get("no_repeat_ngram_window", 8192))
            self.requests.pop(index, None)
            if n <= 0 or window <= 0:
                continue
            self.requests[index] = (
                prompt_ids or [],
                output_ids,
                n,
                window,
                extras.get("no_repeat_ngram_whitelist", []),
            )
        for source, target, direction in batch_update.moved:
            source_request = self.requests.pop(source, None)
            target_request = self.requests.pop(target, None)
            if direction == MoveDirectionality.SWAP:
                if target_request is not None:
                    self.requests[source] = target_request
            if source_request is not None:
                self.requests[target] = source_request

    def apply(self, logits):
        for index, (prompt, output, n, window, whitelist) in self.requests.items():
            # Only the scan window and current n-gram prefix can affect bans.
            tail_length = window + n
            sequence = list(output[-tail_length:])
            if len(sequence) < tail_length:
                sequence = list(prompt[-(tail_length - len(sequence)) :]) + sequence
            banned = banned_tokens(sequence, n, window, whitelist)
            if banned:
                logits[index, list(banned)] = -float("inf")
        return logits
