#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Prebuild the BF16 AITER MHA modules used by Cosmos3 on ROCm."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from vllm_omni.diffusion.attention.backends.utils import fa
from vllm_omni.platforms import current_omni_platform

ATTENTION_VARIANTS = (
    (True, "mha_fwd_bf16_nbias_mask_nlse_ndropout_nqscale"),
    (False, "mha_fwd_bf16_nbias_nmask_nlse_ndropout_nqscale"),
)


def main() -> None:
    if not current_omni_platform.is_rocm():
        raise RuntimeError("The Cosmos3 AITER prewarm must run on ROCm")
    if fa.flash_attn_func is None:
        raise RuntimeError("ROCm FlashAttention/AITER is unavailable")

    cache_dir_value = os.environ.get("AITER_JIT_DIR")
    if not cache_dir_value:
        raise RuntimeError("AITER_JIT_DIR must identify the job-local cache")
    cache_dir = Path(cache_dir_value)

    device = torch.device(current_omni_platform.device_type)
    dtype = torch.bfloat16
    batch_size = 1
    sequence_length = 64
    num_heads = 8
    head_dim = 64

    torch.manual_seed(42)
    query = torch.randn(
        batch_size,
        sequence_length,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    for causal, expected_module in ATTENTION_VARIANTS:
        output = fa.flash_attn_func(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=head_dim**-0.5,
            causal=causal,
        )
        if isinstance(output, tuple):
            output = output[0]
        current_omni_platform.synchronize()

        if output.shape != query.shape:
            raise RuntimeError(f"Unexpected AITER output shape: {output.shape}")
        if not torch.isfinite(output).all():
            raise RuntimeError("AITER prewarm produced non-finite output")

        module_path = cache_dir / f"{expected_module}.so"
        if not module_path.is_file():
            raise RuntimeError(f"Expected AITER module was not built: {module_path}")
        print(f"Prebuilt Cosmos3 AITER module: {module_path}", flush=True)


if __name__ == "__main__":
    main()
