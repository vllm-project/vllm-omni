# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention-4 (CuTe DSL) diffusion attention backend.

Install with ``pip install --pre flash-attn-4``. The wheel publishes only the
``flash_attn.cute`` namespace, so this backend probes it directly instead of
going through the FA2/FA3 detection chain in ``utils/fa.py``.
"""

import os

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import (
    piecewise_attn,
)

logger = init_logger(__name__)

try:
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func  # noqa: F401
except Exception as e:
    logger.warning(
        "FlashAttention4Backend is not available (%s). Install it with `pip install --pre flash-attn-4`.",
        e,
    )
    raise ImportError from e


def _unwrap_fa4_output(out: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    # FA4 returns (out, lse); lse is None unless return_lse=True.
    return out[0] if isinstance(out, tuple) else out


# sm_103a codegen is up to ~14% faster than the sm_100 default at DiT shapes.
if torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3):
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_103a")


# Custom ops keep the CuTe-DSL launchers opaque to torch.compile; the hasattr
# guard keeps registration idempotent across re-imports.
if not hasattr(torch.ops.vllm_omni, "flash_attn4"):

    @torch.library.custom_op("vllm_omni::flash_attn4", mutates_args=())
    def _flash_attn4_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
    ) -> torch.Tensor:
        from flash_attn.cute import flash_attn_func as _kernel

        return _unwrap_fa4_output(_kernel(query, key, value, softmax_scale=softmax_scale, causal=causal))

    @_flash_attn4_op.register_fake
    def _(query, key, value, softmax_scale, causal):
        return query.new_empty(*query.shape[:-1], value.shape[-1])


_flash_attn4_op = torch.ops.vllm_omni.flash_attn4


class FlashAttention4Backend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # sm_100/sm_103: head dims up to 128 plus a dedicated (256, 256) kernel.
        return [64, 96, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_4"

    @staticmethod
    def get_impl_cls() -> type["FlashAttention4Impl"]:
        return FlashAttention4Impl


class FlashAttention4Impl(AttentionImpl):
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
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale
        if backend_kwargs:
            logger.warning("FlashAttention4Impl ignoring backend_kwargs: %s", list(backend_kwargs.keys()))

    @staticmethod
    def _fa4_dense(q, k, v, *, softmax_scale, causal):
        return _flash_attn4_op(q, k, v, softmax_scale, causal)

    def _forward_varlen_masked(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.utils.fa import (
            _pad_input,
            _unpad_input,
            _upad_input,
        )

        assert attention_mask.ndim == 2, "attention_mask must be 2D, (batch_size, seq_len)"
        query_length = query.size(1)
        q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
            query, key, value, attention_mask, query_length, _unpad_input
        )

        out_unpad = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        out_unpad = _unwrap_fa4_output(out_unpad)
        return _pad_input(out_unpad, indices_q, query.size(0), query_length)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None
        full_attn_spans = attn_metadata.full_attn_spans if attn_metadata is not None else None

        if full_attn_spans is not None:
            logger.debug("Using piecewise FlashAttention-4 for mixed causal/full mask")
            return piecewise_attn(
                query,
                key,
                value,
                full_attn_spans,
                self.softmax_scale,
                FlashAttention4Impl._fa4_dense,
            )

        if attention_mask is not None and torch.any(~attention_mask):
            return self._forward_varlen_masked(
                query,
                key,
                value,
                attention_mask,
            )

        return _flash_attn4_op(query, key, value, self.softmax_scale, self.causal)
