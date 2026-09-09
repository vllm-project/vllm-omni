# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Checkpoint-compatible native layers for MAGI-2 Preview.

The grouped-linear, normalization, Fourier, and mHC formulas are adapted from
SandAI's Apache-2.0 preview implementation.  They have been modified to remove
MagiCompiler and external Triton runtime dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .parallel import Magi2ParallelGroup, get_magi2_tp_group


def swiglu7(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Released GPT-OSS-style clamped SwiGLU activation."""

    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    gate, linear = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(max=limit)
    linear = linear.clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (linear + 1.0)).to(out_dtype)


class ModalityDispatcher:
    """Precompute stable modality grouping and inverse permutation metadata."""

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int) -> None:
        if modality_mapping.ndim != 1:
            raise ValueError("modality mapping must be a one-dimensional tensor")
        self.modality_mapping = modality_mapping
        self.num_modalities = num_modalities
        self.permute_mapping = torch.argsort(modality_mapping, stable=True)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)
        self.permuted_modality_mapping = modality_mapping.index_select(0, self.permute_mapping)
        self.group_size = torch.bincount(self.permuted_modality_mapping.long(), minlength=num_modalities).to(
            torch.int32
        )
        self.group_size_cpu = [int(value) for value in self.group_size.cpu().tolist()]
        self.cu_group_sizes = F.pad(torch.cumsum(self.group_size, dim=0), (1, 0))

    def dispatch(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(tensor, self.group_size_cpu, dim=0))

    @staticmethod
    def undispatch(*groups: torch.Tensor) -> torch.Tensor:
        return torch.cat(groups, dim=0)

    def permute(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(0, self.permute_mapping)

    def inverse_permute(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(0, self.inv_permute_mapping)


class Magi2GroupedLinear(nn.Module):
    """Modality-grouped linear with the released flattened weight layout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_experts: int = 1,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        parallel_mode: str | None = None,
        qkv_splits: tuple[int, int, int] | None = None,
        tp_group: Magi2ParallelGroup | None = None,
    ) -> None:
        super().__init__()
        if parallel_mode not in (None, "column", "row"):
            raise ValueError(f"unknown MAGI-2 linear parallel mode {parallel_mode!r}")
        if qkv_splits is not None and parallel_mode != "column":
            raise ValueError("segmented QKV slicing requires column parallelism")
        if qkv_splits is not None and sum(qkv_splits) != out_features:
            raise ValueError("QKV splits must sum to out_features")
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.parallel_mode = parallel_mode
        self.qkv_splits = qkv_splits
        self.tp_group = tp_group or get_magi2_tp_group()
        self.local_in_features = in_features
        self.local_out_features = out_features
        if parallel_mode == "column" and self.tp_group.world_size > 1:
            split_dims = qkv_splits or (out_features,)
            if any(size % self.tp_group.world_size for size in split_dims):
                raise ValueError(
                    f"column-parallel output dimensions {split_dims} must divide TP={self.tp_group.world_size}"
                )
            self.local_out_features = sum(size // self.tp_group.world_size for size in split_dims)
        elif parallel_mode == "row" and self.tp_group.world_size > 1:
            if in_features % self.tp_group.world_size:
                raise ValueError(f"row-parallel input {in_features} must divide TP={self.tp_group.world_size}")
            self.local_in_features = in_features // self.tp_group.world_size

        self.weight = nn.Parameter(
            torch.empty(
                num_experts * self.local_out_features,
                self.local_in_features,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            bias_features = self.local_out_features if parallel_mode == "column" else out_features
            self.bias = nn.Parameter(torch.empty(num_experts * bias_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)
        if self.tp_group.world_size > 1 and parallel_mode is not None:
            self.weight.checkpoint_weight_transform = self.shard_checkpoint_weight
            if self.bias is not None and parallel_mode == "column":
                self.bias.checkpoint_weight_transform = self.shard_checkpoint_bias

    def _column_slices(self) -> tuple[tuple[int, int], ...]:
        splits = self.qkv_splits or (self.out_features,)
        offsets: list[tuple[int, int]] = []
        base = 0
        for size in splits:
            local = size // self.tp_group.world_size
            start = base + self.tp_group.rank * local
            offsets.append((start, start + local))
            base += size
        return tuple(offsets)

    def shard_checkpoint_weight(self, checkpoint_tensor: torch.Tensor) -> torch.Tensor:
        """Convert the released grouped weight into this TP rank's shard."""

        if tuple(checkpoint_tensor.shape) == tuple(self.weight.shape):
            return checkpoint_tensor
        expected = (self.num_experts * self.out_features, self.in_features)
        if tuple(checkpoint_tensor.shape) != expected:
            raise ValueError(
                f"grouped linear checkpoint has shape {tuple(checkpoint_tensor.shape)}, expected {expected}"
            )
        grouped = checkpoint_tensor.view(self.num_experts, self.out_features, self.in_features)
        if self.parallel_mode == "column":
            shards = [grouped[:, start:end] for start, end in self._column_slices()]
            return torch.cat(shards, dim=1).reshape(self.num_experts * self.local_out_features, self.in_features)
        if self.parallel_mode == "row":
            start = self.tp_group.rank * self.local_in_features
            return grouped[:, :, start : start + self.local_in_features].reshape(
                self.num_experts * self.out_features,
                self.local_in_features,
            )
        return checkpoint_tensor

    def shard_checkpoint_bias(self, checkpoint_tensor: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            raise ValueError("cannot load bias into a bias-free grouped linear")
        if tuple(checkpoint_tensor.shape) == tuple(self.bias.shape):
            return checkpoint_tensor
        expected = (self.num_experts * self.out_features,)
        if tuple(checkpoint_tensor.shape) != expected:
            raise ValueError(f"grouped linear bias has shape {tuple(checkpoint_tensor.shape)}, expected {expected}")
        grouped = checkpoint_tensor.view(self.num_experts, self.out_features)
        shards = [grouped[:, start:end] for start, end in self._column_slices()]
        return torch.cat(shards, dim=1).reshape(-1)

    def forward(
        self,
        tensor: torch.Tensor,
        modality_dispatcher: ModalityDispatcher | None = None,
    ) -> torch.Tensor:
        weight = self.weight.view(self.num_experts, self.local_out_features, self.local_in_features)
        bias_features = self.local_out_features if self.parallel_mode == "column" else self.out_features
        bias = self.bias.view(self.num_experts, bias_features) if self.bias is not None else None
        linear_bias = None if self.parallel_mode == "row" else bias
        if self.num_experts == 1:
            output = F.linear(tensor, weight[0], None if linear_bias is None else linear_bias[0])
        else:
            if modality_dispatcher is None:
                raise ValueError("modality_dispatcher is required for grouped linear")
            if modality_dispatcher.num_modalities != self.num_experts:
                raise ValueError("grouped linear expert count does not match modality dispatcher")
            inputs = modality_dispatcher.dispatch(tensor)
            outputs = [
                F.linear(part, weight[index], None if linear_bias is None else linear_bias[index])
                for index, part in enumerate(inputs)
            ]
            output = torch.cat(outputs, dim=0)

        if self.parallel_mode == "row" and self.tp_group.world_size > 1:
            dist.all_reduce(output, group=self.tp_group.group)
        if self.parallel_mode == "row" and bias is not None:
            if self.num_experts == 1:
                output = output + bias[0]
            else:
                assert modality_dispatcher is not None
                outputs = [part + bias[index] for index, part in enumerate(modality_dispatcher.dispatch(output))]
                output = torch.cat(outputs, dim=0)
        return output


def make_grouped_linear(
    in_features: int,
    out_features: int,
    *,
    num_experts: int = 1,
    bias: bool = False,
    dtype: torch.dtype | None = None,
    parallel_mode: str | None = None,
    qkv_splits: tuple[int, int, int] | None = None,
) -> Magi2GroupedLinear:
    return Magi2GroupedLinear(
        in_features,
        out_features,
        num_experts=num_experts,
        bias=bias,
        dtype=dtype,
        parallel_mode=parallel_mode,
        qkv_splits=qkv_splits,
    )


class MultiModalityRMSNorm(nn.Module):
    """RMSNorm with independent modality and mHC-stream scales."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-6,
        num_modality: int = 1,
        num_patterns: int = 1,
        out_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality
        self.num_patterns = num_patterns
        self.out_dtype = out_dtype
        self.weight = nn.Parameter(torch.zeros(num_patterns * dim * num_modality, dtype=torch.float32, device=device))

    def forward(
        self,
        tensor: torch.Tensor,
        modality_dispatcher: ModalityDispatcher | None = None,
    ) -> torch.Tensor:
        original_dtype = tensor.dtype
        normalized = tensor.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + self.eps)
        if self.num_modality == 1:
            weight = self.weight.view(self.num_patterns, self.dim) + 1.0
            result = normalized * weight
        else:
            if modality_dispatcher is None:
                raise ValueError("modality_dispatcher is required for multimodal RMSNorm")
            inputs = modality_dispatcher.dispatch(normalized)
            weights = self.weight.view(self.num_modality, self.num_patterns, self.dim)
            result = modality_dispatcher.undispatch(
                *(part * (weights[index] + 1.0) for index, part in enumerate(inputs))
            )
        return result.to(self.out_dtype or original_dtype)


def _frequency_bands(
    num_bands: int,
    *,
    temperature: float,
    device: torch.device | str | None,
) -> torch.Tensor:
    exponent = torch.arange(num_bands, dtype=torch.float32, device=device) / num_bands
    return 1.0 / temperature**exponent


class ElementWiseFourierEmbed(nn.Module):
    """Nine-coordinate Fourier embedding used as MAGI's element-wise RoPE."""

    def __init__(
        self,
        dim: int,
        *,
        temperature: float = 10000.0,
        learnable: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        bands = _frequency_bands(dim // 8, temperature=temperature, device=device).to(dtype)
        if learnable:
            self.bands = nn.Parameter(bands)
        else:
            # ``bands`` is part of the released checkpoint.  Keep it in the
            # parameter namespace so the generic DLO mmap loader can bind it
            # without first materializing a full CPU transformer.  It remains
            # immutable during inference.
            self.bands = nn.Parameter(bands, requires_grad=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[-1] != 9:
            raise ValueError("MAGI coordinates must have shape [tokens,9]")
        xyz = coords[:, :3]
        sizes = coords[:, 3:6]
        references = coords[:, 6:9]
        scales = (references - 1) / (sizes - 1)
        scales = torch.where((references == 1) & (sizes == 1), torch.ones_like(scales), scales)
        if not torch.isfinite(scales).all():
            raise ValueError("invalid MAGI coordinate scale")
        centers = (sizes - 1) / 2
        centers = centers.clone()
        centers[:, 0] = 0
        projection = (xyz - centers).unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
        return torch.cat((projection.sin(), projection.cos()), dim=1).flatten(1)


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


__all__ = [
    "ElementWiseFourierEmbed",
    "MHCHandler",
    "Magi2GroupedLinear",
    "ModalityDispatcher",
    "MultiModalityRMSNorm",
    "make_grouped_linear",
    "sinkhorn_knopp",
    "swiglu7",
]
