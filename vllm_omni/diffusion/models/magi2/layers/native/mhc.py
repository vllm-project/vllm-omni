# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native MAGI-2 manifold-constrained hyper-connections."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from vllm_omni.diffusion.layers.mhc import MHCMix, MHCPostResidual

from ...common_layers import MHCTensorTuple

_mhc_post_residual = MHCPostResidual()
_mhc_mix = MHCMix()


def _mhc_mix_fp32(
    streams: torch.Tensor,
    branch_output: torch.Tensor,
    post_coefficients: torch.Tensor,
    residual_matrix: torch.Tensor,
) -> torch.Tensor:
    """``MHCMix.forward_native`` with the stream mix as an elementwise contraction in at least FP32."""
    branch = torch.einsum("tn,tc->tnc", post_coefficients, branch_output)
    accumulate = torch.promote_types(streams.dtype, torch.float32)
    mixed = (residual_matrix.to(accumulate).unsqueeze(-1) * streams.to(accumulate).unsqueeze(1)).sum(2)
    return mixed.to(streams.dtype) + branch


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
        # Hardware leaves may opt into FP32 stream contractions.
        self.fp32_stream_contractions = False

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
        if self.fp32_stream_contractions:
            fused = torch.bmm(
                self._stream_major_normed(flattened, norm),
                phi_fused.reshape(self.num_streams, self.hidden_size, -1),
            ).sum(0)
        else:
            fused = norm(flattened).to(self.dtype) @ phi_fused
        pre, post, residual = torch.split(
            fused,
            (self.num_streams, self.num_streams, self.num_streams**2),
            dim=-1,
        )
        return pre, post, residual.view(-1, self.num_streams, self.num_streams)

    def _stream_major_normed(
        self,
        flattened: torch.Tensor,
        norm: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Normalize the streams into a dense [streams, tokens, hidden] bmm operand.

        MUSA bmm copies a batch-strided operand before its GEMM. Compiled, the norm receives the
        [tokens, streams, hidden] view and its kernel stores this layout directly; eager copies once.
        """
        tokens = flattened.shape[0]
        if not torch.compiler.is_compiling():
            normed = norm(flattened).to(self.dtype)
            return normed.reshape(tokens, self.num_streams, self.hidden_size).transpose(0, 1).contiguous()
        # Reducing over (streams, hidden) reads the streams with the same index form as the
        # stream-major store, so Inductor keeps a producing hyper-connection mix in the norm kernel.
        normed = norm(flattened.reshape(tokens, self.num_streams, self.hidden_size)).to(self.dtype)
        stream_major = normed.new_empty_strided(normed.shape, (self.hidden_size, tokens * self.hidden_size, 1))
        stream_major.copy_(normed)
        # Inductor keeps the strides of an as_strided input, so the norm kernel stores stream-major.
        return stream_major.as_strided(stream_major.shape, stream_major.stride()).transpose(0, 1)

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
        coefficients = coefficients.to(out_dtype or streams.dtype)
        if (
            self.fp32_stream_contractions
            and coefficients.shape == streams.shape[:2]
            and coefficients.dtype == streams.dtype
            and streams.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and not torch.is_autocast_enabled(streams.device.type)
        ):
            return (coefficients.float().unsqueeze(-1) * streams.float()).sum(1).to(streams.dtype)
        return torch.einsum("tn,tnc->tc", coefficients, streams)

    def compute_post_residual(
        self,
        post: MHCTensorTuple,
        residual: MHCTensorTuple,
        *,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_post, bias_post, post_logits = post
        alpha_residual, bias_residual, residual_logits = residual
        prepare = _mhc_post_residual if not torch.compiler.is_compiling() else _mhc_post_residual.forward_native
        return prepare(
            post_logits,
            residual_logits,
            alpha_post,
            bias_post,
            alpha_residual,
            bias_residual,
            scale=self.matmul_scale,
            iterations=self.sinkhorn_iterations,
            epsilon=self.sinkhorn_epsilon,
            out_dtype=out_dtype,
        )

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
        if not torch.compiler.is_compiling():
            mix = _mhc_mix
        else:
            mix = _mhc_mix_fp32 if self.fp32_stream_contractions else _mhc_mix.forward_native
        return mix(residual_streams, branch_output, post_coefficients, residual_matrix)

    def _check_multi(self, tensor: torch.Tensor) -> None:
        expected = (self.num_streams, self.hidden_size)
        if tensor.ndim != 3 or tensor.shape[1:] != expected:
            raise ValueError(f"expected mHC tensor [tokens,{expected[0]},{expected[1]}], got {tuple(tensor.shape)}")
