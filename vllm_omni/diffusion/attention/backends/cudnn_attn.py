# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.sdpa import _maybe_reshape_attn_mask


class CuDNNAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supports_prefix_kv_slicing: bool = True

    # cuDNN 9.5+ FMHA on Blackwell: head_dim divisible by 8 and at most 256
    # for BF16/FP16. Used by automatic platform selection; explicit CUDNN_ATTN
    # raises when the configured head size is outside this set.
    _MAX_HEAD_SIZE = 256
    _HEAD_SIZE_MULTIPLE = 8

    @classmethod
    def supports_attention_mask(cls, attention_spec: object | None = None) -> bool:
        del attention_spec
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return list(
            range(
                CuDNNAttentionBackend._HEAD_SIZE_MULTIPLE,
                CuDNNAttentionBackend._MAX_HEAD_SIZE + 1,
                CuDNNAttentionBackend._HEAD_SIZE_MULTIPLE,
            )
        )

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return 0 < head_size <= cls._MAX_HEAD_SIZE and head_size % cls._HEAD_SIZE_MULTIPLE == 0

    @staticmethod
    def get_name() -> str:
        return "CUDNN_ATTN"

    @staticmethod
    def get_impl_cls() -> type["CuDNNAttentionImpl"]:
        return CuDNNAttentionImpl


class CuDNNAttentionImpl(AttentionImpl):
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
        # Set when AttentionConfig (or --diffusion-attention-backend) selected
        # CUDNN_ATTN. The kv_seq_len=1 MATH path is automatic-only.
        self.backend_explicit = bool(extra_impl_args.get("backend_explicit", False))

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        attention_mask = None
        if attn_metadata:
            valid_kv_length = attn_metadata.extra.get("valid_kv_length")
            if attn_metadata.attn_mask is None and isinstance(valid_kv_length, int):
                if not 0 < valid_kv_length <= key.shape[1]:
                    raise ValueError(
                        "valid_kv_length must be within the K/V sequence, "
                        f"got {valid_kv_length} for length {key.shape[1]}"
                    )
                # A contiguous valid prefix is mathematically equivalent to a
                # broadcast key-padding mask. Slicing K/V keeps Q/output in the
                # checkpoint's aligned layout while letting cuDNN select its
                # much faster mask-free FMHA plan.
                key = key[:, :valid_kv_length]
                value = value[:, :valid_kv_length]
            else:
                attention_mask = _maybe_reshape_attn_mask(
                    query,
                    key,
                    attn_metadata.attn_mask,
                    mask_mode="broadcast_k",
                )

        enable_gqa = query.shape[2] != key.shape[2]
        kv_seq_len = key.shape[1]
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))

        # Pin one backend only. A priority list like [CUDNN, FLASH, MATH] hits a
        # PyTorch SDPA dispatch quirk: when FLASH rejects a non-None attn_mask,
        # cuDNN gets runtime-disabled in the same call and the dispatcher falls
        # through to MATH even though cuDNN alone handles the mask fine
        # (~11 ms vs ~215 ms for MATH on sm_120 HV-1.5 shapes).
        # Explicit CUDNN_ATTN must not silently substitute MATH/EFFICIENT.
        #
        # Automatic-only: cuDNN FMHA rejects KV sequence length 1
        # ("cudnn SDPA does not support key/value sequence length 1"). Dummy
        # warmup and some I2V layers (e.g. LTX-2) hit this when CUDNN_ATTN is
        # the platform default. MATH is the only kernel that can run that shape.
        # An explicit CUDNN_ATTN request raises instead of silently switching.
        if kv_seq_len <= 1:
            if self.backend_explicit:
                raise ValueError(
                    "CUDNN_ATTN was explicitly selected but cuDNN FMHA does not "
                    f"support key/value sequence length {kv_seq_len}. "
                    "Select TORCH_SDPA or another backend for this shape."
                )
            backends = [SDPBackend.MATH]
        else:
            backends = [SDPBackend.CUDNN_ATTENTION]
        with sdpa_kernel(backends):
            output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=self.causal,
                scale=self.softmax_scale,
                enable_gqa=enable_gqa,
            )
        return output.permute(0, 2, 1, 3)
