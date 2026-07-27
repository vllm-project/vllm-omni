# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel layers bound to the text encoder process group."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.distributed.parallel_state import get_text_encoder_tp_group


def _copy(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


class TextEncoderColumnParallelLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True, return_bias: bool = True, **_: Any):
        super().__init__()
        group = get_text_encoder_tp_group()
        self.tp_rank, self.tp_size = group.rank_in_group, group.world_size
        if output_size % self.tp_size:
            raise ValueError(f"output_size {output_size} is not divisible by text encoder TP {self.tp_size}")
        self.output_size_per_partition = output_size // self.tp_size
        self.return_bias = return_bias
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size))
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        shard_size = param.shape[0]
        _copy(param, loaded_weight.narrow(0, self.tp_rank * shard_size, shard_size))

    def forward(self, input_: torch.Tensor):
        output = F.linear(input_, self.weight, self.bias)
        return (output, None) if self.return_bias else output


class TextEncoderRowParallelLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True, return_bias: bool = True, **_: Any):
        super().__init__()
        self.group = get_text_encoder_tp_group()
        self.tp_rank, self.tp_size = self.group.rank_in_group, self.group.world_size
        if input_size % self.tp_size:
            raise ValueError(f"input_size {input_size} is not divisible by text encoder TP {self.tp_size}")
        self.return_bias = return_bias
        self.weight = nn.Parameter(torch.empty(output_size, input_size // self.tp_size))
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        self.weight.weight_loader = self.weight_loader
        if self.bias is not None:
            self.bias.weight_loader = _copy

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        shard_size = param.shape[1]
        _copy(param, loaded_weight.narrow(1, self.tp_rank * shard_size, shard_size))

    def forward(self, input_: torch.Tensor):
        bias = self.bias if self.tp_rank == 0 else None
        output = self.group.all_reduce(F.linear(input_, self.weight, bias))
        return (output, None) if self.return_bias else output


class TextEncoderMergedColumnParallelLinear(TextEncoderColumnParallelLinear):
    def __init__(self, input_size: int, output_sizes: list[int], **kwargs: Any):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), **kwargs)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int | None = None) -> None:
        if shard_id is None:
            offset = 0
            for index, size in enumerate(self.output_sizes):
                self.weight_loader(param, loaded_weight.narrow(0, offset, size), index)
                offset += size
            return
        local_size = self.output_sizes[shard_id] // self.tp_size
        target_offset = sum(size // self.tp_size for size in self.output_sizes[:shard_id])
        source = loaded_weight.narrow(0, self.tp_rank * local_size, local_size)
        param.data.narrow(0, target_offset, local_size).copy_(source)


class TextEncoderQKVParallelLinear(TextEncoderMergedColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        **kwargs: Any,
    ):
        group = get_text_encoder_tp_group()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        if total_num_heads % group.world_size or total_num_kv_heads % group.world_size:
            raise ValueError("T5 attention heads must be divisible by text encoder TP")
        self.num_heads = total_num_heads // group.world_size
        self.num_kv_heads = total_num_kv_heads // group.world_size
        super().__init__(
            hidden_size,
            [total_num_heads * head_size, total_num_kv_heads * head_size, total_num_kv_heads * head_size],
            **kwargs,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str | int | None = None,
    ) -> None:
        if isinstance(shard_id, str):
            shard_id = {"q": 0, "k": 1, "v": 2}[shard_id]
        super().weight_loader(param, loaded_weight, shard_id)


class TextEncoderVocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.group = get_text_encoder_tp_group()
        self.tp_rank, self.tp_size = self.group.rank_in_group, self.group.world_size
        self.start = (num_embeddings * self.tp_rank) // self.tp_size
        self.end = (num_embeddings * (self.tp_rank + 1)) // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.end - self.start, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        _copy(param, loaded_weight.narrow(0, self.start, self.end - self.start))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        mask = (input_ < self.start) | (input_ >= self.end)
        local_ids = (input_ - self.start).masked_fill(mask, 0)
        output = F.embedding(local_ids, self.weight)
        output.masked_fill_(mask.unsqueeze(-1), 0)
        return self.group.all_reduce(output)
