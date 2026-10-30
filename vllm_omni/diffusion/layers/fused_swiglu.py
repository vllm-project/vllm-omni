# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Optional MindIE-SD adapter for a complete SwiGLU feed-forward operation."""

from collections.abc import Callable
from functools import cache

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.platforms import current_omni_platform


@cache
def _resolve_eagle_ffn() -> Callable[..., torch.Tensor] | None:
    # Resolve outside compiled execution; installation changes require restart.
    try:
        from mindiesd import layers
    except ImportError:
        return None
    op = getattr(layers, "eagle_ffn_linear", None)
    return op if callable(op) else None


def fused_swiglu_available() -> bool:
    if not current_omni_platform.is_npu():
        return False
    from vllm_omni.platforms.npu import is_a5

    return is_a5() and _resolve_eagle_ffn() is not None


class FusedSwiGLU(CustomOp):
    """Consume prepacked [gate, up] and down-projection weights owned by the caller."""

    def forward_native(self, x: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
        gate, up = F.linear(x, gate_up_weight).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, down_weight)

    def forward_npu(self, x: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
        op = _resolve_eagle_ffn()
        if op is None:
            return self.forward_native(x, gate_up_weight, down_weight)
        # Once selected, operator errors propagate rather than silently falling back.
        return op(x, gate_up_weight, down_weight, activation="swiglu")

    def forward_cuda(self, x: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x, gate_up_weight, down_weight)
