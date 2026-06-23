# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-request AR decode state for SongGeneration v2 LeLM Stage 0.

Manages the delayed codebook pattern [0, 250, 250] state machine.
Stream 0 is driven by vLLM; streams 1+2 are driven internally by transformer2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .pattern import DelayedPatternProvider, Pattern


@dataclass
class ArState:
    """Mutable per-request AR state for SongGenerationV2LeLM Stage 0."""

    # --- Pattern ---
    pattern: Pattern
    delays: list[int]
    code_depth: int
    code_size: int
    special_token_id: int
    eos_token_id: int
    max_gen_len: int

    # --- Generation state ---
    # Dense code storage: [K, T] where K=code_depth, T grows per step
    gen_codes: torch.Tensor  # [K, max_gen_len], init -1 (unknown)
    step: int = 0  # current AR step (= stream-0 frame index)
    is_end: torch.Tensor = field(default_factory=lambda: torch.zeros(0))  # [K] bool per stream
    finished: bool = False

    # --- Streams 1+2 internal tokens (filled by transformer2) ---
    # For now these are populated with special_token_id until transformer2
    stream12_pending: bool = False

    @classmethod
    def create(
        cls,
        *,
        code_depth: int,
        code_size: int,
        max_gen_len: int,
        delays: list[int],
        device: torch.device = torch.device("cpu"),
    ) -> ArState:
        """Initialize state for a new generation request."""
        provider = DelayedPatternProvider(
            code_depth=code_depth,
            delays=delays,
        )
        pattern = provider.get_pattern(max_gen_len)

        gen_codes = torch.full(
            (code_depth, max_gen_len),
            -1,
            dtype=torch.long,
            device=device,
        )
        is_end = torch.zeros(code_depth, dtype=torch.bool, device=device)

        special_token_id = code_size  # = config.code_size + 1
        eos_token_id = code_size - 1  # = config.code_size

        return cls(
            pattern=pattern,
            delays=delays,
            code_depth=code_depth,
            code_size=code_size,
            special_token_id=special_token_id,
            eos_token_id=eos_token_id,
            max_gen_len=max_gen_len,
            gen_codes=gen_codes,
            is_end=is_end,
        )

    @property
    def active_streams(self) -> list[int]:
        """Which streams are active at the current step."""
        active = []
        for q, delay in enumerate(self.delays):
            if self.step >= delay:
                active.append(q)
        return active

    @property
    def stream0_frame(self) -> int:
        """Current frame index for stream 0."""
        return self.step

    def stream_frame(self, q: int) -> int:
        """Current frame index for stream q (accounting for delay)."""
        return self.step - self.delays[q]

    def record_stream0_token(self, token_id: int) -> None:
        """Record the stream-0 token sampled by vLLM at this step."""
        if self.step < self.max_gen_len:
            self.gen_codes[0, self.step] = token_id

        if token_id == self.eos_token_id:
            self.is_end[0] = True

    def record_stream12_tokens(self, tokens: torch.Tensor) -> None:
        """Record streams 1+2 tokens computed by transformer2.

        Args:
            tokens: [K-1] tensor of token IDs for streams 1..K-1
        """
        for i, q in enumerate(range(1, self.code_depth)):
            frame = self.stream_frame(q)
            if frame >= 0 and frame < self.max_gen_len:
                self.gen_codes[q, frame] = tokens[i].item()
                if tokens[i].item() == self.eos_token_id:
                    self.is_end[q] = True

    def fill_inactive_streams(self) -> None:
        """Fill inactive stream positions with special_token_id.

        Called at each step for streams that haven't started yet.
        """
        for q in range(1, self.code_depth):
            frame = self.stream_frame(q)
            if frame < 0:
                # Stream not yet active; nothing to fill at a "negative" frame
                pass
            elif self.is_end[q]:
                # Stream already ended
                if frame < self.max_gen_len:
                    self.gen_codes[q, frame] = self.special_token_id

    def advance(self) -> None:
        """Move to the next step after all streams are processed."""
        self.step += 1
        self._check_finished()

    def _check_finished(self) -> None:
        """Check if all active streams have hit EOS or max length."""
        if self.step >= self.max_gen_len:
            self.finished = True
            return
        if torch.all(self.is_end):
            self.finished = True

    def get_output_codes(self) -> torch.Tensor:
        """Get the final output codes after delay pattern reversion.

        Storage layout:
          - Stream 0 (delay=0): gen_codes[0, step] = token at absolute step
          - Stream q (delay=D_q): gen_codes[q, frame] where frame = step - D_q
            So gen_codes[q, f] holds the token generated at step f + D_q.

        Temporal alignment: output frame f maps to:
          - Stream 0: gen_codes[0, f] (generated at step f)
          - Stream q: gen_codes[q, f] (generated at step f + D_q)
        All streams at output frame f correspond to the same musical timestep.

        No additional offset is needed because streams 1+2 are already stored
        at their "natural frame" index (step - delay).

        Returns frame-major [T_out, K] for Stage 1 compatibility,
        where T_out = min(step, max_gen_len) - max_delay.
        """
        t_actual = min(self.step, self.max_gen_len)
        max_delay = max(self.delays)

        if t_actual <= max_delay:
            return torch.empty(0, self.code_depth, dtype=torch.long, device=self.gen_codes.device)

        t_out = t_actual - max_delay
        out = self.gen_codes[:, :t_out].clone()

        out[out == -1] = self.special_token_id
        return out.permute(1, 0)

    def should_compute_stream12(self) -> bool:
        """Whether streams 1+2 need computation at this step."""
        if self.code_depth <= 1:
            return False
        for q in range(1, self.code_depth):
            if not self.is_end[q] and self.stream_frame(q) >= 0:
                return True
        return False

    def get_stream0_input_token(self) -> int:
        """Get the previous stream-0 token to feed as input for this step.

        On the first step (step=0), returns special_token_id (BOS-like).
        On subsequent steps, returns the last generated stream-0 token.
        """
        if self.step == 0:
            return self.special_token_id
        prev = self.gen_codes[0, self.step - 1].item()
        return int(prev) if prev >= 0 else self.special_token_id


__all__ = ["ArState"]
