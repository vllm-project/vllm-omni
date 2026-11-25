# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tensor operations that preserve eager rounding in compiled diffusion models."""

import torch


def add_stored_gemm_outputs(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Add GEMM outputs after each result has been stored.

    Inductor rewrites ``gemm + gemm`` as ``addmm`` and adds in the accumulator,
    skipping the bf16 store eager inserts between the GEMM and the add.
    Promoting the compiled add restores that store-then-add rounding.
    """
    if torch.compiler.is_compiling():
        return (left.float() + right.float()).to(dtype=left.dtype)
    return left + right
