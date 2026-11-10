# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastH3 Omni's Grid32 integration and head-major output projections.

The shift fixes 0.999, unlike the original H3/DMD2 shift which fixes 1.0.
Each forward uses the integration-weighted mean of four consecutive heads.
The ordinary H3 Euler update then applies the total interval increment.
"""

from __future__ import annotations

import torch
from torch import nn

PDD_STEPS = 32
PDD_INDICES = tuple(range(0, 33, 4))
PDD_MAX_T = 0.999


def pdd_sigmas(shift: float) -> list[float]:
    base = torch.linspace(PDD_MAX_T, 0.0, PDD_STEPS + 1, dtype=torch.float64)[list(PDD_INDICES)]
    # FastVideo feeds float32 node sigmas into MiniMaxH3Scheduler.
    return (shift * base * PDD_MAX_T / (base * (shift - 1.0) + PDD_MAX_T)).float().tolist()


def _validate_pdd_step(step: int) -> None:
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < len(PDD_INDICES) - 1:
        raise ValueError(f"FastH3 Omni pdd_step must be an integer in [0, 7], got {step!r}")


def pdd_head_weights(shift: float, step: int) -> torch.Tensor:
    _validate_pdd_step(step)
    grid = torch.linspace(PDD_MAX_T, 0.0, PDD_STEPS + 1, dtype=torch.float64)
    start, end = PDD_INDICES[step : step + 2]
    a, b = grid[start:end], grid[start + 1 : end + 1]
    # Analytic difference avoids cancellation between neighbouring sigmas.
    delta = shift * PDD_MAX_T**2 * (b - a) / ((PDD_MAX_T + (shift - 1) * a) * (PDD_MAX_T + (shift - 1) * b))
    return (delta / delta.sum()).float()


class MiniMaxH3PDDOutput(nn.Module):
    """Replicated FP32 output heads; the block stack retains TP/SP support.

    A head must stay intact while fusing its parameters. These small output
    projections are replicated on TP ranks, like the reference implementation.
    The step is an explicit input so interleaved requests cannot share mutable
    fusion state. No widened per-token output tensor is materialized.
    """

    def __init__(self, input_size: int, output_size: int, *, shift: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(PDD_STEPS * output_size, input_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(PDD_STEPS * output_size, dtype=torch.float32))
        self.output_size = output_size
        self.register_buffer(
            "coefficients",
            torch.stack([pdd_head_weights(shift, step) for step in range(len(PDD_INDICES) - 1)]),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        _validate_pdd_step(step)
        coefficients = self.coefficients[step]
        start, end = PDD_INDICES[step : step + 2]

        def fuse(parameter: torch.Tensor) -> torch.Tensor:
            heads = parameter.unflatten(0, (PDD_STEPS, self.output_size))[start:end]
            return torch.einsum("n,n...->...", coefficients, heads.float()).to(parameter.dtype)

        return nn.functional.linear(x, fuse(self.weight), fuse(self.bias))
