# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Q/K affine RMS normalization and interleaved RoPE through MindIE-SD."""

import torch

from vllm_omni.diffusion.layers.custom_op import CustomOp

RopeTables = tuple[torch.Tensor, torch.Tensor]


def prepare_rope_tables(freqs: torch.Tensor, dtype: torch.dtype) -> RopeTables:
    """Expand Qwen's complex three-axis frequencies into adjacent channel pairs."""
    sin = freqs.imag.repeat_interleave(2, dim=-1).to(dtype).contiguous()
    cos = freqs.real.repeat_interleave(2, dim=-1).to(dtype).contiguous()
    return sin, cos


class FusedNormRope(CustomOp):
    """Stateless adapter; the model's existing norm_q/norm_k own their weights.

    NPU uses MindIE-SD. Other backends compose PyTorch normalization and RoPE
    operations so the layer retains a working unfused implementation.
    """

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        tables: RopeTables,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from mindiesd import norm_rope_concat

        sin, cos = tables
        outputs = norm_rope_concat(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            norm_query_weight=q_weight.to(query.dtype).contiguous(),
            norm_key_weight=k_weight.to(key.dtype).contiguous(),
            rope_sin=sin,
            rope_cos=cos,
            norm_type=4,
            rope_type=1,
            eps=eps,
        )
        # MindIE-SD emits BNSD; Attention and prefix cache use BSND.
        return tuple(output.transpose(1, 2) for output in outputs[:3])

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        tables: RopeTables,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sin, cos = tables

        def norm_rope(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
            normalized = (normalized * weight.float()).to(x.dtype)
            paired = torch.stack((-normalized[..., 1::2], normalized[..., 0::2]), dim=-1).flatten(-2)
            return (normalized.float() * cos[None, :, None].float() + paired.float() * sin[None, :, None].float()).to(
                x.dtype
            )

        return norm_rope(query, q_weight), norm_rope(key, k_weight), value

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        tables: RopeTables,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_native(query, key, value, q_weight, k_weight, eps, tables)

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        tables: RopeTables,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_native(query, key, value, q_weight, k_weight, eps, tables)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        tables: RopeTables,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_native(query, key, value, q_weight, k_weight, eps, tables)

    def forward_musa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        tables: RopeTables,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_native(query, key, value, q_weight, k_weight, eps, tables)
