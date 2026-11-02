# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Optional NPU broadcast residual fusion with eager BF16 rounding."""

from collections.abc import Callable
from functools import partial
from importlib import import_module

import torch
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)
ResidualGateOp = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def resolve_residual_gate_add() -> ResidualGateOp | None:
    """Resolve once at model construction, outside compiled forward regions."""
    if not current_omni_platform.is_npu():
        return None
    try:
        module = import_module("mindiesd.layers.residual_gate_add")
    except ImportError:
        logger.warning("MindIE-SD residual fusion is unavailable; keeping the eager residual path.")
        return None
    op = getattr(module, "residual_gate_add", None)
    if not callable(op) or not getattr(module, "_TRITON_ON_ASCEND", False):
        return None
    schema = torch.ops.mindiesd.residual_gate_add.default._schema
    if not any(arg.name == "preserve_mul_rounding" for arg in schema.arguments):
        logger.warning("MindIE-SD lacks residual product rounding support; keeping the eager residual path.")
        return None
    return partial(op, preserve_mul_rounding=True)


def try_residual_gate_add(
    op: ResidualGateOp | None,
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor | None:
    """Return None for unsupported inputs so the caller preserves its fallback."""
    if op is None or torch.is_grad_enabled() or torch.compiler.is_compiling():
        return None
    if (
        hidden.device.type != "npu"
        or hidden.dtype != torch.bfloat16
        or hidden.ndim != 3
        or hidden.shape[0] != 1
        or hidden.numel() == 0
        or update.shape != hidden.shape
        or gate.shape != (hidden.shape[0], 1, hidden.shape[-1])
        or any(
            t.device != hidden.device or t.dtype != hidden.dtype or not t.is_contiguous()
            for t in (hidden, update, gate)
        )
    ):
        return None
    return op(hidden, update, gate)
