# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-only delayed RVQ stop state, following reference TTSReq.check_eos.

The default ``any`` EOA policy starts an eleven-step drain, counting that
same sampling step. CPU stage processing never controls this state machine.
"""

from dataclasses import dataclass

import torch

CONTINUE = 1
EOS = 2
NUM_CODEBOOKS = 10


@dataclass
class Zonos3SamplingStep:
    codes: torch.Tensor
    logical_token_ids: torch.Tensor
    eoa_frame: torch.Tensor
    codebook_valid: torch.Tensor
    active: torch.Tensor
    finished: torch.Tensor


class Zonos3ARSamplingState:
    """Persistent state indexed by AR slots, independent of batch row order.

    Call reset when allocating or reusing slots. Each batch must contain
    unique slots, including separate reserved dummy slots for graph padding.
    No request decision reads a device tensor on the host.
    """

    def __init__(self, capacity: int, eoa_id: int, pad_id: int, device: torch.device | str) -> None:
        self.eoa_id = eoa_id
        self.pad_id = pad_id
        self.position = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.eoa_frame = torch.full((capacity,), -1, dtype=torch.int64, device=device)
        self.drain_remaining = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.has_audio_prefix = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.ignore_eos = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.eoa_codebook0 = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.next_codes = torch.full((capacity, NUM_CODEBOOKS), pad_id, dtype=torch.int64, device=device)
        self.codebook_valid = torch.zeros((capacity, NUM_CODEBOOKS), dtype=torch.bool, device=device)
        self.columns = torch.arange(NUM_CODEBOOKS, dtype=torch.int64, device=device)

    def reset(
        self,
        slots: torch.Tensor,
        has_audio_prefix: torch.Tensor,
        ignore_eos: torch.Tensor | None = None,
        eoa_codebook0: torch.Tensor | None = None,
    ) -> None:
        self.position.index_fill_(0, slots, 0)
        self.eoa_frame.index_fill_(0, slots, -1)
        self.drain_remaining.index_fill_(0, slots, 0)
        self.next_codes.index_fill_(0, slots, self.pad_id)
        self.codebook_valid.index_fill_(0, slots, False)
        self.has_audio_prefix.index_copy_(0, slots, has_audio_prefix)
        for target, value in ((self.ignore_eos, ignore_eos), (self.eoa_codebook0, eoa_codebook0)):
            if value is None:
                target.index_fill_(0, slots, False)
            else:
                target.index_copy_(0, slots, value)

    def accept(
        self, slots: torch.Tensor, sampled_codes: torch.Tensor, active: torch.Tensor | None = None
    ) -> Zonos3SamplingStep:
        position = self.position.index_select(0, slots)
        previous_eoa = self.eoa_frame.index_select(0, slots)
        remaining = self.drain_remaining.index_select(0, slots)
        # Chunked-prefill rows whose logical sample count is zero must not
        # advance RVQ/RNG feedback. Completed slots also remain absorbing while
        # an async scheduler has a final in-flight step awaiting CPU delivery.
        if active is None:
            active = torch.ones_like(previous_eoa, dtype=torch.bool)
        active = active & ~((previous_eoa >= 0) & (remaining == 0))
        has_prefix = self.has_audio_prefix.index_select(0, slots)
        # The leading unsupervised triangle must not introduce a false EOA.
        started = has_prefix[:, None] | (position[:, None] >= self.columns)
        codes = torch.where(started & active[:, None], sampled_codes, self.pad_id)
        is_eoa = codes == self.eoa_id
        is_eoa = is_eoa & ~self.ignore_eos.index_select(0, slots)[:, None]
        is_eoa = is_eoa & (~self.eoa_codebook0.index_select(0, slots)[:, None] | (self.columns == 0))
        last_eoa_column = torch.where(is_eoa, self.columns, -1).amax(dim=-1)
        first_eoa = active & (previous_eoa < 0) & (last_eoa_column >= 0)
        aligned_eoa = (position - last_eoa_column).clamp(min=0)
        eoa_frame = torch.where(first_eoa, aligned_eoa, previous_eoa)
        remaining = torch.where(first_eoa, NUM_CODEBOOKS + 1, remaining)
        # Reference _check_eos decrements on the first detection step too.
        remaining = torch.where(active & (eoa_frame >= 0), (remaining - 1).clamp(min=0), remaining)
        finished = (eoa_frame >= 0) & (remaining == 0)
        aligned_frames = position[:, None] - self.columns
        valid = (
            active[:, None]
            & started
            & (codes < self.eoa_id)
            & ((eoa_frame[:, None] < 0) | (aligned_frames < eoa_frame[:, None]))
        )
        self.position.index_copy_(0, slots, position + active.long())
        self.eoa_frame.index_copy_(0, slots, eoa_frame)
        self.drain_remaining.index_copy_(0, slots, remaining)
        previous_codes = self.next_codes.index_select(0, slots)
        self.next_codes.index_copy_(0, slots, torch.where(active[:, None], codes, previous_codes))
        previous_valid = self.codebook_valid.index_select(0, slots)
        self.codebook_valid.index_copy_(0, slots, torch.where(active[:, None], valid, previous_valid))
        logical = torch.where(finished, EOS, CONTINUE).to(torch.int32).unsqueeze(-1)
        return Zonos3SamplingStep(codes, logical, eoa_frame, valid, active, finished)
