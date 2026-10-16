# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default operation adapter used by Flux3 transformer blocks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


QKNormRopeFn = Callable[["Tensor", "Tensor", "Tensor", Any, "Tensor"], tuple["Tensor", "Tensor"]]


@dataclass(frozen=True)
class Flux3BlockKernelPlan:
    """Constant dispatch decisions for one Flux3 transformer block."""

    qkv_linear: bool = False
    qkv_qknorm_rope: bool = False
    attention_output: bool = False
    mlp_swiglu: bool = False
    mlp_output: bool = False

    @property
    def enabled(self) -> bool:
        return any(
            (
                self.qkv_linear,
                self.qkv_qknorm_rope,
                self.attention_output,
                self.mlp_swiglu,
                self.mlp_output,
            )
        )


class Flux3KernelAdapter:
    """Default block operations behind the Flux3 kernel dispatch boundary.

    The adapter is deliberately stateless and is shared by all transformer
    blocks. Specialized implementations override these coarse operations so
    weight preparation and fused execution do not leak into block forwards.
    """

    backend = "vllm"

    def __init__(self, qk_norm_rope: QKNormRopeFn) -> None:
        self._qk_norm_rope = qk_norm_rope

    def uses_fused_qkv(self, prefix: str, head_dim: int) -> bool:
        """Whether this adapter needs a fused QKV projection for a block."""
        del prefix, head_dim
        return False

    def configure_block(self, block: Any, prefix: str) -> Flux3BlockKernelPlan:
        # Default setup is intentionally a no-op. Specialized adapters use
        # these arguments to inspect the block and install a fixed kernel plan.
        del block, prefix
        return Flux3BlockKernelPlan()

    def prepare_input(self, block: Any, hidden_states: Tensor) -> Any:
        # Default preparation is intentionally an identity operation.
        # Specialized adapters may derive reusable inputs from the block plan.
        del block
        return hidden_states

    def qkv_qknorm_rope(self, block: Any, prepared_input: Any, rope: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden_states = prepared_input
        qkv, _ = block.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.unflatten(-1, (block.num_heads_local, block.head_dim))
        k = k.unflatten(-1, (block.num_kv_heads_local, block.head_dim))
        v = v.unflatten(-1, (block.num_kv_heads_local, block.head_dim))
        q, k = self._qk_norm_rope(q, k, v, block.norm, rope)
        return q, k, v

    def attention_output(self, block: Any, hidden_states: Tensor) -> Tensor:
        return block.attn_out(hidden_states)

    def mlp(self, block: Any, prepared_input: Any) -> Tensor:
        hidden_states = prepared_input
        return block.mlp_out(block.mlp_act(block.mlp_in(hidden_states)))

    def finalize(self) -> None:
        """Validate adapter state after every transformer block is configured."""


__all__ = ["Flux3BlockKernelPlan", "Flux3KernelAdapter"]
