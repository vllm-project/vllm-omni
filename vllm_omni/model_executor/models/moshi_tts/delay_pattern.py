"""Delay pattern logic for Moshi's multi-codebook generation.

The current HF/Transformers Moshi implementation delays codebooks 1-7 by one
step relative to codebook 0.

During prefill, the delay is applied by shifting the code arrays:
  - codebook 0: no shift
  - codebooks 1-7: prepend one BOS token and shift right by 1
"""

from __future__ import annotations

import torch


class DelayPattern:
    """Manages the delay pattern for multi-codebook generation.

    Args:
        num_codebooks: Number of audio codebooks (default 8).
        delays: Per-codebook delay values. Default: [0, 1, 1, 1, 1, 1, 1, 1].
        bos_token_id: BOS token for padding delayed positions (= audio_vocab_size).
        pad_token_id: PAD token for positions past sequence end (= audio_vocab_size).
    """

    def __init__(
        self,
        num_codebooks: int = 8,
        delays: list[int] | None = None,
        bos_token_id: int = 2048,
        pad_token_id: int = 2048,
    ):
        self.num_codebooks = num_codebooks
        self.delays = delays if delays is not None else [0] + [1] * (num_codebooks - 1)
        self.max_delay = max(self.delays)
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id

    def apply_to_prefill(
        self,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        """Apply delay pattern to a prefill code array.

        Args:
            codes: [T, num_codebooks] audio codes.

        Returns:
            delayed_codes: [T + max_delay, num_codebooks] with delays applied.
            Codebook i is shifted right by delays[i], with BOS tokens prepended
            and PAD tokens appended to maintain alignment.
        """
        T, K = codes.shape
        out_len = T + self.max_delay
        delayed = codes.new_full((out_len, K), self.pad_token_id)

        for cb_idx in range(K):
            d = self.delays[cb_idx]
            # Fill BOS tokens for the delay region
            if d > 0:
                delayed[:d, cb_idx] = self.bos_token_id
            # Copy the actual codes, shifted by delay
            end = min(d + T, out_len)
            delayed[d:end, cb_idx] = codes[: end - d, cb_idx]

        return delayed
