# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native MAGI-2 modality and mHC-stream RMS normalization."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...common_layers import ModalityDispatcher


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
        """Normalize the last dimension, or the last two when they split ``dim`` features."""
        original_dtype = tensor.dtype
        split = tensor.shape[-1] != self.dim
        if split and (tensor.ndim < 2 or tensor.shape[-2] * tensor.shape[-1] != self.dim or self.num_patterns != 1):
            raise ValueError(f"cannot normalize a {tuple(tensor.shape)} tensor over {self.dim} features")
        feature_shape = tuple(tensor.shape[-2:]) if split else (self.num_patterns, self.dim)
        normalized = tensor.float()
        reduce_dims = (-2, -1) if split else -1
        normalized = normalized * torch.rsqrt(normalized.square().mean(dim=reduce_dims, keepdim=True) + self.eps)
        if self.num_modality == 1:
            weight = self.weight.view(feature_shape) + 1.0
            result = normalized * weight
        else:
            if modality_dispatcher is None:
                raise ValueError("modality_dispatcher is required for multimodal RMSNorm")
            inputs = modality_dispatcher.dispatch(normalized)
            weights = self.weight.view(self.num_modality, *feature_shape)
            result = modality_dispatcher.undispatch(
                *(part * (weights[index] + 1.0) for index, part in enumerate(inputs))
            )
        return result.to(self.out_dtype or original_dtype)
