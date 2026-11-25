# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native FP8 sparse branch with the existing MindIE-SD SLA projection.

The sparse branch uses block-scaled E4M3 FP8. Linear projection
C7 and attention quantization are separate numerical contracts.
"""

import inspect
from typing import Protocol, cast

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)


class _SLAProvider(Protocol):
    proj_l: nn.Linear
    feature_map_q: nn.Module
    feature_map_k: nn.Module
    topk: float
    BLKQ: int
    BLKK: int


class MindIESDFP8SLA(nn.Module):
    """Retain SLA block selection and linear correction; quantize sparse QKV."""

    def __init__(self, provider: nn.Module) -> None:
        super().__init__()
        from mindiesd.layers._custom_ops import block_sparse_attention
        from mindiesd.layers.flash_attn.sparse_linear_attn_triton import get_block_map

        if "q_dequant_scale" not in inspect.signature(block_sparse_attention).parameters:
            raise RuntimeError("FP8 SLA requires MindIE-SD BlockSparseAttentionV3 with FP8/scales support")
        sla_provider = cast(_SLAProvider, provider)
        if sla_provider.proj_l.in_features not in (64, 128):
            raise ValueError("FP8 SLA requires head dimension 64 or 128")
        if sla_provider.BLKQ not in (64, 128) or sla_provider.BLKK not in (64, 128):
            raise ValueError("FP8 SLA requires query/KV blocks of 64 or 128")
        self.proj_l = sla_provider.proj_l
        self.feature_map_q = sla_provider.feature_map_q
        self.feature_map_k = sla_provider.feature_map_k
        self.topk = sla_provider.topk
        self.BLKQ, self.BLKK = sla_provider.BLKQ, sla_provider.BLKK
        self._block_map = get_block_map
        self._sparse_attention = block_sparse_attention

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        import torch_npu

        if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[0] != 1:
            raise ValueError("FP8 SLA requires one square BNSD request")
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype or q.device.type != "npu":
            raise ValueError("FP8 SLA requires BF16 Q/K/V on NPU")
        length = q.shape[2]
        mask, _, _ = self._block_map(q, k, self.topk, self.BLKQ, self.BLKK)

        def quant(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
            batch, heads, sequence, dimension = tensor.shape
            padding = (-sequence) % block_size
            padded = F.pad(tensor, (0, 0, 0, padding)) if padding else tensor
            grouped = padded.reshape(batch, heads, padded.shape[2] // block_size, -1)
            values, scales = torch_npu.npu_dynamic_quant(grouped, dst_type=torch_npu.float8_e4m3fn)
            values = values.reshape(batch, heads, -1, dimension)[:, :, :sequence].contiguous()
            return values, scales.unsqueeze(-1).float().contiguous()

        qp, qs = quant(q, self.BLKQ)
        kp, ks = quant(k, self.BLKK)
        vp, vs = quant(v, self.BLKK)
        output, _ = self._sparse_attention(
            query=qp,
            key=kp,
            value=vp,
            block_sparse_mask=mask,
            block_shape=[self.BLKQ, self.BLKK],
            q_input_layout="BNSD",
            kv_input_layout="BNSD",
            num_key_value_heads=q.shape[1],
            scale_value=q.shape[-1] ** -0.5,
            actual_seq_lengths=[length],
            actual_seq_lengths_kv=[length],
            inner_precise=4,
            softmax_lse_flag=0,
            q_dequant_scale=qs,
            k_dequant_scale=ks,
            v_dequant_scale=vs,
            quant_mode=1,
            q_dtype=torch_npu.float8_e4m3fn,
            k_dtype=torch_npu.float8_e4m3fn,
            v_dtype=torch_npu.float8_e4m3fn,
        )
        q_feature, k_feature = self.feature_map_q(q), self.feature_map_k(k)
        kv = k_feature.transpose(-1, -2) @ v
        denominator = 1e-5 + (q_feature * k_feature.sum(-2, keepdim=True)).sum(-1, keepdim=True)
        linear = (q_feature @ kv) / denominator
        with torch.amp.autocast("npu", dtype=torch.bfloat16):
            linear = self.proj_l(linear)
        logger.info_once("Executed MindIE-SD FP8 SparseLinearAttention sparse branch; linear correction is BF16")
        return output[:, :, :length] + linear
