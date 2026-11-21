# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl, _maybe_reshape_attn_mask
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload_strategy


def _npu_fusion_attention(query, key, value, attn_metadata, softmax_scale, inner_precise: int):
    """BSND FlashAttention via torch_npu. True in the source mask means keep."""
    import torch_npu

    if query.shape[2] != key.shape[2]:
        raise RuntimeError(f"fusion attention expects equal heads, got {query.shape[2]} and {key.shape[2]}")
    atten_mask = None
    sparse_mode = 0
    source = attn_metadata.attn_mask if attn_metadata is not None else None
    if source is not None:
        keep = _maybe_reshape_attn_mask(query, key, source, mask_mode="full_qk")
        if keep is None:
            keep = source
        atten_mask = (~keep.to(device=query.device, dtype=torch.bool)).contiguous()
        sparse_mode = 1
    return torch_npu.npu_fusion_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        head_num=query.shape[2],
        input_layout="BSND",
        atten_mask=atten_mask,
        scale=float(softmax_scale),
        keep_prob=1.0,
        inner_precise=inner_precise,
        sparse_mode=sparse_mode,
    )[0]


class QwenImage21SDPAImpl(SDPAImpl):
    """Model-local NPU defaults, inside shared Attention parallel dispatch."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args: object,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            softmax_scale,
            causal,
            num_kv_heads,
            prefix,
            backend_kwargs,
            **extra_impl_args,
        )
        config = get_current_diffusion_config_or_none()
        self.auto_npu_fusion = (
            config is not None
            and config.enforce_eager
            and config.parallel_config.tensor_parallel_size == 1
            and config.parallel_config.sequence_parallel_size == 1
            and resolve_offload_strategy(config) == OffloadStrategy.NONE
        )

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if (
            self.auto_npu_fusion
            and query.dtype == key.dtype == value.dtype == torch.bfloat16
            and query.shape[0] == 1
            and query.shape[2] == key.shape[2] == value.shape[2]
            and attn_metadata is None
            and not self.causal
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        ):
            return _npu_fusion_attention(
                query,
                key,
                value,
                None,
                self.softmax_scale,
                int(os.environ.get("QWEN21_FUSION_PRECISE", "0")),
            )
        return super().forward_npu(query, key, value, attn_metadata)
