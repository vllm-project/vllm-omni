# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Select the existing MUSA FP32 stream-contraction policy for MAGI-2."""

from __future__ import annotations

import torch

from ..native.mhc import MHCHandler as NativeMHCHandler


class MHCHandler(NativeMHCHandler):
    """Retain the original MUSA policy without a hardware branch in common math."""

    def __init__(
        self,
        num_streams: int,
        hidden_size: int,
        *,
        sinkhorn_iterations: int = 20,
        sinkhorn_epsilon: float = 1e-12,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            num_streams,
            hidden_size,
            sinkhorn_iterations=sinkhorn_iterations,
            sinkhorn_epsilon=sinkhorn_epsilon,
            dtype=dtype,
        )
        self.fp32_stream_contractions = True


__all__ = ["MHCHandler"]
