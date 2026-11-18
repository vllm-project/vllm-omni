# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.diffusion.layers.rope import apply_rotary_emb_torch_npu


class Cosmos3RotaryEmbedding(CustomOp):
    """Apply half-rotation RoPE using Cosmos3's full-width cos/sin."""

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the original Cosmos3 arithmetic.

        Args:
            q: Query tensor [B, S, H_q, D].
            k: Key tensor [B, S, H_kv, D].
            cos: Cosine coefficients [B or 1, S, 1, D].
            sin: Sine coefficients [B or 1, S, 1, D].
        """
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(q, k, cos, sin)

    def forward_npu(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Tables already span D; do not expand them or change half-rotation pairing.
        return (
            apply_rotary_emb_torch_npu(q, cos, sin, interleaved=False, half_head_dim=False),
            apply_rotary_emb_torch_npu(k, cos, sin, interleaved=False, half_head_dim=False),
        )
