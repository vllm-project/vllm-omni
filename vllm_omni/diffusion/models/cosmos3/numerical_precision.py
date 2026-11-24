# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit accuracy profile for comparing CUDA and Ascend Cosmos3 inference."""

import os

import torch


def configure_numerical_precision(device: torch.device, dtype: torch.dtype) -> bool:
    """Configure every model worker, including workers started with spawn.

    FP32 alone does not disable TF32/HF32. Keep this diagnostic profile opt-in:
    BF16 deployment and other models retain their existing execution policy.
    """
    enabled = os.environ.get("COSMOS3_STRICT_PRECISION", "0")
    if enabled not in {"0", "1"}:
        raise ValueError("COSMOS3_STRICT_PRECISION must be 0 or 1")
    if enabled == "0":
        return False
    if dtype != torch.float32:
        raise ValueError("COSMOS3_STRICT_PRECISION=1 requires --dtype float32")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif device.type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    else:
        raise ValueError(f"Strict Cosmos3 device comparison requires CUDA or NPU, got {device}")
    return True
