# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MindIE-SD SLA dispatch; the H3 model owns the learned projection module."""

from __future__ import annotations

import math

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from vllm_omni.diffusion.attention.backends.sla_config import SLASpec

logger = init_logger(__name__)


def _mindiesd_sla_cls() -> type[nn.Module]:
    # Importing mindiesd also registers its Ascend custom-op paths. Keep this
    # optional dependency out of CPU/CUDA/default-attention imports.
    try:
        from mindiesd.layers import SparseLinearAttention
    except ImportError as exc:
        raise ImportError(
            "SLA_ATTN requires MindIE-SD exporting mindiesd.layers.SparseLinearAttention "
            "and its compatible torch-npu/triton-ascend dependencies."
        ) from exc
    return SparseLinearAttention


class SLAAttentionBackend(AttentionBackend):
    supported_platforms = ("npu",)

    @classmethod
    def validate_available(cls) -> None:
        _mindiesd_sla_cls()

    @staticmethod
    def get_name() -> str:
        return "SLA_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SLAAttentionImpl]:
        return SLAAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        return None

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        return True


class SLAAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        role: str | None = None,
        **extra_impl_args,
    ) -> None:
        # This backend is enabled only on H3 DiT blocks. In particular, the
        # text token refiner must have a separate dense per-role selection.
        if role != "self" or not prefix.startswith("blocks."):
            raise ValueError("SLA_ATTN supports MiniMax-H3 DiT blocks only; keep minimax_h3.token_refiner dense.")
        if causal or qkv_layout != "BSND" or num_kv_heads not in (None, num_heads):
            raise ValueError("SLA_ATTN requires noncausal BSND self-attention with equal Q/K/V heads.")
        if head_size not in SLAAttentionBackend.get_supported_head_sizes():
            raise ValueError(f"SLA_ATTN does not support head_size={head_size}.")
        if not math.isclose(softmax_scale, head_size**-0.5):
            raise ValueError("MindIE-SD SLA requires the standard head_size**-0.5 scale.")
        self.config = SLASpec(**(backend_kwargs or {}).get("sla", {}))
        # AttentionImpl is not an nn.Module. The model must register this
        # exact object as attn.sla_module before checkpoint loading/offload.
        self.sla_module = _mindiesd_sla_cls()(
            head_dim=head_size,
            topk=self.config.topk,
            BLKQ=self.config.blkq,
            BLKK=self.config.blkk,
            feature_map=self.config.feature_map,
            use_bf16=True,
        )

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if query.ndim != 4 or query.shape != key.shape or key.shape != value.shape or query.shape[0] != 1:
            raise ValueError("SLA_ATTN requires one square packed request in BSND layout.")
        if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("MiniMax-H3 SLA_ATTN requires BF16 Q/K/V.")
        used = query.shape[1]
        if attn_metadata is not None:
            if (
                attn_metadata.attn_mask is not None
                or attn_metadata.full_attn_spans is not None
                or attn_metadata.query_ranges is not None
                or attn_metadata.joint_query is not None
                or attn_metadata.joint_key is not None
                or attn_metadata.joint_value is not None
            ):
                raise ValueError("SLA_ATTN does not support masks, piecewise spans or sharded query ranges.")
            padding = attn_metadata.packed_padding
            if padding is None and "cu_seqlens_q" in attn_metadata.extra:
                raise ValueError("SLA_ATTN packed input requires validated [real, pad] metadata.")
            if padding is not None:
                if padding.q_length != padding.kv_length:
                    raise ValueError("SLA_ATTN does not support compact reference KV or unequal Q/KV lengths.")
                used = padding.q_length
        if not 0 < used <= query.shape[1]:
            raise ValueError("SLA_ATTN valid sequence length is outside the packed tensor.")
        kv_blocks = math.ceil(used / self.config.blkk)
        if int(self.config.topk * kv_blocks) < 1:
            raise ValueError(
                f"MindIE-SD SLA topk={self.config.topk} selects zero KV blocks for {used} tokens; "
                "increase the sequence length or sla.topk."
            )
        # Exclude padding from BOTH sparse block selection and linear sums.
        # Pure Ulysses has already gathered sequence and scattered heads here.
        q, k, v = (x[:, :used].transpose(1, 2).contiguous() for x in (query, key, value))
        result = self.sla_module(q, k, v).transpose(1, 2)
        logger.info_once(
            "Executed MindIE-SD SparseLinearAttention: topk=%s BLKQ=%d BLKK=%d",
            self.config.topk,
            self.config.blkq,
            self.config.blkk,
        )
        out = torch.zeros_like(query)
        out[:, :used] = result
        return out
