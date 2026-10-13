#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Prebuild one AITER module used by READY attention coverage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from aiter import flash_attn_func, flash_attn_varlen_func

CORE_MODULE = "module_aiter_core.so"
DENSE_MODULE = "mha_fwd_bf16_nbias_nmask_nlse_ndropout_nqscale.so"
VARLEN_MODULE = "mha_varlen_fwd_bf16_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale.so"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("core", "dense", "varlen"))
    return parser.parse_args()


def _unwrap_output(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _run_dense(device: torch.device) -> torch.Tensor:
    dtype = torch.bfloat16
    query = torch.randn(1, 64, 8, 64, device=device, dtype=dtype)
    output = flash_attn_func(
        query,
        query.clone(),
        query.clone(),
        dropout_p=0.0,
        softmax_scale=64**-0.5,
        causal=False,
    )
    return _unwrap_output(output)


def _run_varlen(device: torch.device) -> torch.Tensor:
    dtype = torch.bfloat16
    query = torch.randn(64, 8, 64, device=device, dtype=dtype)
    cu_seqlens = torch.tensor((0, 64), dtype=torch.int32, device=device)
    output = flash_attn_varlen_func(
        q=query,
        k=query.clone(),
        v=query.clone(),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=64,
        max_seqlen_k=64,
        causal=False,
        softmax_scale=64**-0.5,
    )
    return _unwrap_output(output)


def main() -> None:
    args = _parse_args()
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("The FlashAttention/AITER prewarm must run on ROCm")

    cache_dir_value = os.environ.get("AITER_JIT_DIR")
    if not cache_dir_value:
        raise RuntimeError("AITER_JIT_DIR must identify the job-local cache")
    cache_dir = Path(cache_dir_value)

    if args.variant == "core":
        expected_module = CORE_MODULE
    else:
        device = torch.device("cuda")
        torch.manual_seed(42)
        output = _run_dense(device) if args.variant == "dense" else _run_varlen(device)
        torch.accelerator.synchronize()
        expected_shape = (1, 64, 8, 64) if args.variant == "dense" else (64, 8, 64)
        if output.shape != expected_shape:
            raise RuntimeError(f"Unexpected AITER output shape: {output.shape}")
        if not torch.isfinite(output).all():
            raise RuntimeError("AITER prewarm produced non-finite output")
        expected_module = DENSE_MODULE if args.variant == "dense" else VARLEN_MODULE

    module_path = cache_dir / expected_module
    if not module_path.is_file():
        raise RuntimeError(f"Expected AITER module was not built: {module_path}")
    print(f"Prepared {args.variant} AITER module: {module_path}", flush=True)


if __name__ == "__main__":
    main()
