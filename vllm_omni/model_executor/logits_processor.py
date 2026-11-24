# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request-controlled logits processors."""

import torch
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext, LogitsProcessor, LogitsProcRequestState


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

    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState) -> None:
        self.req_states = req_states
        self.requests: dict[int, tuple[int, int, list[int]]] = {}
        self._speculative = vllm_config.speculative_config is not None

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.requests.pop(req_idx, None)
        extras = sampling_params.extra_args or {}
        n = int(extras.get("no_repeat_ngram_size", 0))
        window = int(extras.get("no_repeat_ngram_window", 8192))
        if n <= 0 or window <= 0:
            return False
        if self._speculative:
            raise ValueError("No-repeat n-grams require non-speculative decoding")
        self.requests[req_idx] = (n, window, extras.get("no_repeat_ngram_whitelist", []))
        return True

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        for row, slot in enumerate(ctx.idx_mapping_np):
            config = self.requests.get(int(slot))
            if config is None:
                continue
            n, window, whitelist = config
            # Non-speculative context gives the exact visible length. Keep the
            # bounded native history on device and append the current input,
            # which need not yet be present in committed history.
            length = int(ctx.seq_lens_upper_bound_np[row])
            start = max(0, length - window - n)
            sequence = torch.cat(
                (self.req_states.all_token_ids.gpu[slot, start : length - 1], ctx.input_ids[row : row + 1])
            )
            if len(sequence) < n:
                continue
            candidates = sequence.unfold(0, n, 1)[max(0, len(sequence) - window) :]
            matches = torch.ones(len(candidates), dtype=torch.bool, device=logits.device)
            if n > 1:
                matches = (candidates[:, :-1] == sequence[-(n - 1) :]).all(dim=-1)
            banned = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
            banned.scatter_reduce_(0, candidates[:, -1].long(), matches, reduce="amax")
            if whitelist:
                allowed = [token for token in whitelist if 0 <= token < logits.shape[-1]]
                banned[allowed] = False
            logits[row].masked_fill_(banned, -float("inf"))
        return logits
