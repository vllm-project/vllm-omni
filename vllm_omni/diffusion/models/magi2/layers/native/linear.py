# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Checkpoint-compatible grouped linear layers for MAGI-2."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ...parallel import Magi2ParallelGroup
from .dispatcher import ModalityDispatcher


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
        if tp_group is None:
            # Keep the patch point exposed by the former ``magi2.layers``
            # module while avoiding an import cycle during package setup.
            from .. import get_magi2_tp_group

            tp_group = get_magi2_tp_group()
        self.tp_group = tp_group
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


__all__ = ["Magi2GroupedLinear", "make_grouped_linear"]
