# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

try:
    from sageattn3 import sageattn3_blackwell  # noqa: F401
except ImportError:
    logger.warning(
        "SageAttention3Backend is not available. Install `sageattn3` from "
        "https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell"
    )
    raise ImportError


# Wrapping sageattn3_blackwell as a torch.library custom op keeps it opaque to
# torch.compile. Otherwise Dynamo graph-breaks on the raw pybind11 kernel and
# Inductor fails scheduling with KeyError: 'op5'. The hasattr guard keeps this
# idempotent across test re-imports that pop the module from sys.modules.
if not hasattr(torch.ops.vllm_omni, "sageattn3_blackwell"):

    @torch.library.custom_op("vllm_omni::sageattn3_blackwell", mutates_args=())
    def _sageattn3_blackwell_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        from sageattn3 import sageattn3_blackwell as _kernel

        return _kernel(query, key, value, is_causal=is_causal)

    @_sageattn3_blackwell_op.register_fake
    def _(query, key, value, is_causal):
        return torch.empty_like(query)


_sageattn3_blackwell_op = torch.ops.vllm_omni.sageattn3_blackwell


class SageAttention3Backend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN_3"

    @staticmethod
    def get_impl_cls() -> type["SageAttention3Impl"]:
        return SageAttention3Impl


class SageAttention3Impl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        expected_scale = head_size**-0.5
        if not math.isclose(softmax_scale, expected_scale, rel_tol=1e-6):
            raise ValueError(
                "SAGE_ATTN_3 does not expose a custom softmax scale; "
                f"expected {expected_scale} for head_size={head_size}, got {softmax_scale}."
            )

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            raise ValueError("SAGE_ATTN_3 does not support attn_mask. Select a mask-capable backend.")
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        if key.shape[1] != query.shape[1]:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError(
                    "GQA/MQA requires query heads to be a multiple of KV heads, "
                    f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}"
                )
            raise NotImplementedError(
                "SAGE_ATTN_3 was explicitly selected but does not support GQA/MQA "
                f"(q_heads={query.shape[1]}, kv_heads={key.shape[1]}). Select a GQA-capable backend."
            )

        output = _sageattn3_blackwell_op(query, key, value, self.causal)

        return output.transpose(1, 2).contiguous()
