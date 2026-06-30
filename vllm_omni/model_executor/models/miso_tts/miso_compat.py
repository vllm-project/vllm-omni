# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Moshi quantize patches for unquantized Mimi weights (legacy moshi API)."""

from __future__ import annotations

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False


def patch_bitsandbytes_import_for_unquantized_layers() -> None:
    """Patch legacy ``moshi.utils.quantize.linear`` for FP Mimi checkpoints.

    Moshi >=0.2 uses ``QLinear`` modules instead of ``quantize.linear`` /
    ``multi_linear``. In that case this is a no-op; ``get_mimi`` loads FP weights
    without the old hook.
    """
    global _PATCHED
    if _PATCHED:
        return

    from moshi.utils import quantize

    if not hasattr(quantize, "linear") or not hasattr(quantize, "multi_linear"):
        _PATCHED = True
        return

    original_linear = quantize.linear
    original_multi_linear = quantize.multi_linear

    def linear(module: nn.Module, x: torch.Tensor, name: str = "weight") -> torch.Tensor:
        if quantize.is_quantized(module, name):
            return original_linear(module, x, name)
        return nn.functional.linear(x, getattr(module, name))

    def multi_linear(
        num_steps: int,
        schedule: list[int] | None,
        module: nn.Module,
        x: torch.Tensor,
        offset: int,
        name: str = "weight",
    ) -> torch.Tensor:
        if quantize.is_quantized(module, name):
            return original_multi_linear(num_steps, schedule, module, x, offset, name)

        weight = getattr(module, name)
        num_linear = num_steps if schedule is None else max(schedule) + 1
        weight = weight.view(num_linear, -1, weight.shape[-1])

        outputs = []
        for t in range(x.shape[1]):
            linear_index = t + offset
            if schedule is not None:
                linear_index = schedule[linear_index]
            outputs.append(nn.functional.linear(x[:, t], weight[linear_index]))
        return torch.stack(outputs, 1)

    quantize.linear = linear
    quantize.multi_linear = multi_linear
    _PATCHED = True
