# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native manifold-constrained hyper-connection operations for MAGI-2."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

MHCTensorTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def sinkhorn_knopp(matrix_logits: torch.Tensor, iterations: int, epsilon: float) -> torch.Tensor:
    matrix = torch.exp(matrix_logits - matrix_logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iterations):
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + epsilon)
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + epsilon)
    return matrix


class MHCHandler:
    """Exact four-stream manifold-constrained hyper-connection math."""

    def __init__(
        self,
        num_streams: int,
        hidden_size: int,
        *,
        sinkhorn_iterations: int = 20,
        sinkhorn_epsilon: float = 1e-12,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_streams = num_streams
        self.hidden_size = hidden_size
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.dtype = dtype
        self.matmul_scale = 1.0 / math.sqrt(float(num_streams * hidden_size))

    def flatten(self, tensor: torch.Tensor) -> torch.Tensor:
        self._check_multi(tensor)
        return tensor.view(tensor.shape[0], -1)

    def compute_logits(
        self,
        flattened: torch.Tensor,
        norm: Callable[[torch.Tensor], torch.Tensor],
        phi_fused: torch.Tensor,
    ) -> MHCTensorTuple:
        if flattened.ndim != 2 or flattened.shape[-1] != self.num_streams * self.hidden_size:
            raise ValueError("invalid flattened mHC shape")
        fused = norm(flattened).to(self.dtype) @ phi_fused
        pre, post, residual = torch.split(
            fused,
            (self.num_streams, self.num_streams, self.num_streams**2),
            dim=-1,
        )
        return pre, post, residual.view(-1, self.num_streams, self.num_streams)

    def apply_pre(
        self,
        streams: torch.Tensor,
        alpha_bias_logits: MHCTensorTuple,
        *,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        self._check_multi(streams)
        alpha, bias, logits = alpha_bias_logits
        coefficients = torch.sigmoid(alpha * self.matmul_scale * logits + bias.unsqueeze(0))
        return torch.einsum("tn,tnc->tc", coefficients.to(out_dtype or streams.dtype), streams)

    def compute_post_residual(
        self,
        post: MHCTensorTuple,
        residual: MHCTensorTuple,
        *,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_post, bias_post, post_logits = post
        alpha_residual, bias_residual, residual_logits = residual
        post_coefficients = 2.0 * torch.sigmoid(alpha_post * self.matmul_scale * post_logits + bias_post.unsqueeze(0))
        residual_matrix = sinkhorn_knopp(
            alpha_residual * self.matmul_scale * residual_logits.float() + bias_residual.unsqueeze(0).float(),
            self.sinkhorn_iterations,
            self.sinkhorn_epsilon,
        )
        return post_coefficients.to(out_dtype), residual_matrix.to(out_dtype)

    def hyper_connect(
        self,
        residual_streams: torch.Tensor,
        branch_output: torch.Tensor,
        post_coefficients: torch.Tensor,
        residual_matrix: torch.Tensor,
    ) -> torch.Tensor:
        self._check_multi(residual_streams)
        if branch_output.ndim != 2 or branch_output.shape[-1] != self.hidden_size:
            raise ValueError("invalid mHC branch-output shape")
        branch = torch.einsum("tn,tc->tnc", post_coefficients, branch_output)
        mixed = torch.einsum("tij,tjc->tic", residual_matrix, residual_streams)
        return mixed + branch

    def _check_multi(self, tensor: torch.Tensor) -> None:
        expected = (self.num_streams, self.hidden_size)
        if tensor.ndim != 3 or tensor.shape[1:] != expected:
            raise ValueError(f"expected mHC tensor [tokens,{expected[0]},{expected[1]}], got {tuple(tensor.shape)}")


__all__ = ["MHCHandler", "MHCTensorTuple", "sinkhorn_knopp"]
