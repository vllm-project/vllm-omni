# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded-window decoding of the SAME predictor sequence as offline.

Eight kernel-3 ConvNeXt blocks have radius 8 at the interpolated frame rate;
center ISTFT adds at most 2 frames (1024 / 512). Interpolation has four frames
per hidden interval. Holding back three intervals, and retaining four on the
left, covers that radius of 10 frames. Thus CONTEXT=ceil(10/4)+1=4, also the
frozen reference's RECEPTIVE_FIELD. Tests measure the resulting boundaries;
the reference constant alone is not evidence of equivalence.
"""

from dataclasses import dataclass

import torch
from torch import nn

from vllm_omni.model_executor.models.soprano.contracts import HIDDEN_SIZE, SAMPLES_PER_INTERVAL

CONTEXT = 4


@dataclass
class DecoderWindow:
    """One synthesis unit; the owner must discard this object on abort/finish."""

    chunk_size: int
    hidden: torch.Tensor | None = None
    total: int = 0
    emitted: int = 0
    closed: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.chunk_size <= 64:
            raise ValueError("Soprano hidden chunk_size must be in 1..64")

    @torch.inference_mode()
    def push(self, decoder: nn.Module, hidden: torch.Tensor, *, finished: bool) -> torch.Tensor:
        """Return only new samples, retaining at most 2*CONTEXT+chunk_size rows.

        Hidden is an owned, FP32 [T,512] delta, not a cumulative trajectory.
        Empty terminal chunks flush exactly to (total-1)*2048; a unit with
        fewer than two vectors is an explicit synthesis error, never padded.
        """
        if self.closed:
            raise ValueError("Soprano decoder window is already finished")
        if hidden.ndim != 2 or hidden.shape[1] != HIDDEN_SIZE or hidden.dtype != torch.float32:
            raise ValueError("Soprano stream needs float32 [T,512] hidden deltas")
        if hidden.shape[0] > self.chunk_size:
            raise ValueError("Hidden delta exceeds configured chunk_size")
        chunks = []
        offset = 0
        while offset < hidden.shape[0]:
            # Match the reference cadence even when a transport delta crosses
            # the initial CONTEXT+chunk_size threshold. Append in segments,
            # not per-token copies, and decode only at an emission boundary.
            target = self.emitted // SAMPLES_PER_INTERVAL + CONTEXT + self.chunk_size
            count = min(target - self.total, hidden.shape[0] - offset)
            part = hidden[offset : offset + count]
            pieces = [part] if self.hidden is None else [self.hidden, part]
            self.hidden = torch.cat(pieces)[-(2 * CONTEXT + self.chunk_size) :]
            self.total += count
            offset += count
            if self.total == target:
                chunks.append(self._decode_to(decoder, (self.total - CONTEXT) * SAMPLES_PER_INTERVAL))
        if finished:
            self.closed = True
            if self.total < 2:
                raise ValueError("Soprano synthesis needs at least two predictor states")
            end = (self.total - 1) * SAMPLES_PER_INTERVAL
            if end > self.emitted:
                chunks.append(self._decode_to(decoder, end))
            self.hidden = None
        if not chunks:
            return hidden.new_empty(0)
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def _decode_to(self, decoder: nn.Module, end: int) -> torch.Tensor:
        origin = (self.total - self.hidden.shape[0]) * SAMPLES_PER_INTERVAL
        wave = decoder(self.hidden.T.unsqueeze(0))[0]
        delta = wave[self.emitted - origin : end - origin]
        if delta.numel() != end - self.emitted:
            raise ValueError("Soprano streaming window failed to cover the new sample interval")
        self.emitted = end
        return delta
