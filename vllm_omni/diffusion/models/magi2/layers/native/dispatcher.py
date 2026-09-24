# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native modality dispatch helpers for MAGI-2."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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


__all__ = ["ModalityDispatcher"]
