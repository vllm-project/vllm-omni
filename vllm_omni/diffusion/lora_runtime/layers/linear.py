# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.utils import split_tensor_along_last_dim
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from ..bindings import ResolvedLoRABinding
from ..types import DiffusionLoRAComposition, LoadedDiffusionLoRA, LowRankUpdate


def _validate_update(update: LowRankUpdate, *, expected_input: int, expected_output: int) -> None:
    if update.lora_a.ndim != 2 or update.lora_b.ndim != 2:
        raise ValueError(
            f"LoRA matrices for {update.component}.{update.logical_target} must be two-dimensional, "
            f"got A={tuple(update.lora_a.shape)}, B={tuple(update.lora_b.shape)}"
        )
    if update.rank <= 0 or update.lora_b.shape[1] != update.rank:
        raise ValueError(
            f"LoRA rank mismatch for {update.component}.{update.logical_target}: "
            f"A={tuple(update.lora_a.shape)}, B={tuple(update.lora_b.shape)}"
        )
    if update.lora_a.shape[1] != expected_input or update.lora_b.shape[0] != expected_output:
        raise ValueError(
            f"LoRA shape mismatch for {update.component}.{update.logical_target}: "
            f"expected A=({update.rank}, {expected_input}), B=({expected_output}, {update.rank}); "
            f"got A={tuple(update.lora_a.shape)}, B={tuple(update.lora_b.shape)}"
        )


def _shard_column_b(module: ColumnParallelLinear, tensor: torch.Tensor) -> torch.Tensor:
    shard_size = tensor.shape[0] // module.tp_size
    if shard_size * module.tp_size != tensor.shape[0]:
        raise ValueError(f"LoRA output dimension {tensor.shape[0]} is not divisible by TP size {module.tp_size}")
    return tensor.narrow(0, module.tp_rank * shard_size, shard_size)


def _shard_merged_b(module: MergedColumnParallelLinear, tensor: torch.Tensor) -> torch.Tensor:
    pieces = tensor.split(module.output_sizes, dim=0)
    return torch.cat([_shard_column_b(module, piece) for piece in pieces], dim=0)


def _qkv_global_sizes(module: QKVParallelLinear) -> tuple[int, int, int]:
    return (
        module.total_num_heads * module.head_size,
        module.total_num_kv_heads * module.head_size,
        module.total_num_kv_heads * module.v_head_size,
    )


def _shard_qkv_piece(module: QKVParallelLinear, tensor: torch.Tensor, slice_index: int) -> torch.Tensor:
    shard_ids = ("q", "k", "v")
    shard_id = shard_ids[slice_index]
    local_size = module._get_shard_size_mapping(shard_id)
    if local_size is None:
        raise ValueError(f"Unable to resolve QKV LoRA slice {shard_id!r}")
    shard_rank = module.tp_rank if shard_id == "q" else module.tp_rank // module.num_kv_head_replicas
    return tensor.narrow(0, shard_rank * local_size, local_size)


def _localize_update(
    module: nn.Module,
    update: LowRankUpdate,
    *,
    slice_index: int,
    slice_count: int,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return rank-local A/B and whether shrink results require TP reduction."""

    if isinstance(module, QKVParallelLinear):
        global_sizes = _qkv_global_sizes(module)
        expected_output = global_sizes[slice_index] if slice_count == 3 else sum(global_sizes)
        _validate_update(update, expected_input=module.input_size, expected_output=expected_output)
        if slice_count == 3:
            local_b = _shard_qkv_piece(module, update.lora_b, slice_index)
        else:
            local_b = torch.cat(
                [
                    _shard_qkv_piece(module, piece, index)
                    for index, piece in enumerate(update.lora_b.split(global_sizes, dim=0))
                ],
                dim=0,
            )
        return update.lora_a, local_b, False

    if isinstance(module, MergedColumnParallelLinear):
        if slice_count == len(module.output_sizes):
            expected_output = module.output_sizes[slice_index]
            _validate_update(update, expected_input=module.input_size, expected_output=expected_output)
            local_b = _shard_column_b(module, update.lora_b)
        else:
            expected_output = sum(module.output_sizes)
            _validate_update(update, expected_input=module.input_size, expected_output=expected_output)
            local_b = _shard_merged_b(module, update.lora_b)
        if module.gather_output and module.tp_size > 1:
            raise NotImplementedError("Dynamic diffusion LoRA does not support gathered ColumnParallel output")
        return update.lora_a, local_b, False

    if isinstance(module, ColumnParallelLinear):
        _validate_update(update, expected_input=module.input_size, expected_output=module.output_size)
        if module.gather_output and module.tp_size > 1:
            raise NotImplementedError("Dynamic diffusion LoRA does not support gathered ColumnParallel output")
        return update.lora_a, _shard_column_b(module, update.lora_b), False

    if isinstance(module, RowParallelLinear):
        _validate_update(update, expected_input=module.input_size, expected_output=module.output_size)
        if not module.reduce_results and module.tp_size > 1:
            raise NotImplementedError("Dynamic diffusion LoRA requires reduced RowParallel output")
        shard_size = module.input_size_per_partition
        local_a = update.lora_a.narrow(1, module.tp_rank * shard_size, shard_size)
        return local_a, update.lora_b, module.tp_size > 1

    if isinstance(module, ReplicatedLinear):
        _validate_update(update, expected_input=module.input_size, expected_output=module.output_size)
        return update.lora_a, update.lora_b, False

    if isinstance(module, nn.Linear):
        _validate_update(update, expected_input=module.in_features, expected_output=module.out_features)
        return update.lora_a, update.lora_b, False

    raise TypeError(
        f"Default diffusion LoRA executor cannot wrap {type(module).__name__}; the model must provide a custom executor"
    )


class _LowRankBank(nn.Module):
    """Fixed-shape concatenated A/B bank with request-mutable rank scales."""

    def __init__(
        self,
        entries: Sequence[tuple[str, LowRankUpdate, torch.Tensor, torch.Tensor]],
        *,
        device: torch.device,
        dtype: torch.dtype,
        reduce_rank: bool,
    ) -> None:
        super().__init__()
        self.reduce_rank = reduce_rank
        self._slots: dict[str, tuple[int, int, float]] = {}
        offset = 0
        lora_as: list[torch.Tensor] = []
        lora_bs: list[torch.Tensor] = []
        for name, update, local_a, local_b in entries:
            rank = update.rank
            self._slots[name] = (offset, rank, float(update.intrinsic_scale))
            offset += rank
            lora_as.append(local_a.to(device=device, dtype=dtype))
            lora_bs.append(local_b.to(device=device, dtype=dtype))
        self.register_buffer("lora_a", torch.cat(lora_as, dim=0), persistent=False)
        self.register_buffer("lora_b", torch.cat(lora_bs, dim=1), persistent=False)
        self.register_buffer("active_rank_scales", torch.zeros(offset, device=device, dtype=dtype), persistent=False)

    @property
    def output_size(self) -> int:
        return int(self.lora_b.shape[0])

    def activate(self, composition: DiffusionLoRAComposition) -> None:
        self.active_rank_scales.zero_()
        for selection in composition:
            slot = self._slots.get(selection.name)
            if slot is None:
                continue
            offset, rank, intrinsic_scale = slot
            self.active_rank_scales.narrow(0, offset, rank).fill_(selection.scale * intrinsic_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(x, self.lora_a)
        if self.reduce_rank:
            hidden = tensor_model_parallel_all_reduce(hidden)
        hidden = hidden * self.active_rank_scales
        return F.linear(hidden, self.lora_b)


class DynamicLoRALinear(nn.Module):
    """Signature-transparent dynamic LoRA wrapper around one linear layer."""

    def __init__(
        self,
        binding: ResolvedLoRABinding,
        loras: Mapping[str, LoadedDiffusionLoRA],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base_layer = binding.module
        self._bank_indices: list[int | None] = []
        self._slice_output_sizes = self._resolve_slice_output_sizes(
            binding.module,
            len(binding.logical_targets),
        )
        self.banks = nn.ModuleList()
        slice_count = len(binding.logical_targets)
        for slice_index in range(slice_count):
            entries: list[tuple[str, LowRankUpdate, torch.Tensor, torch.Tensor]] = []
            reduce_rank: bool | None = None
            for lora_name in sorted(loras):
                update = binding.updates[lora_name][slice_index]
                if update is None:
                    continue
                local_a, local_b, update_reduce_rank = _localize_update(
                    binding.module,
                    update,
                    slice_index=slice_index,
                    slice_count=slice_count,
                )
                if reduce_rank is not None and reduce_rank != update_reduce_rank:
                    raise ValueError(f"LoRA TP behavior differs within binding {binding.full_module_name}")
                reduce_rank = update_reduce_rank
                entries.append((lora_name, update, local_a, local_b))
            if not entries:
                self._bank_indices.append(None)
                continue
            self._bank_indices.append(len(self.banks))
            self.banks.append(
                _LowRankBank(
                    entries,
                    device=device,
                    dtype=dtype,
                    reduce_rank=bool(reduce_rank),
                )
            )

    @staticmethod
    def _resolve_slice_output_sizes(module: nn.Module, slice_count: int) -> tuple[int, ...]:
        if isinstance(module, QKVParallelLinear) and slice_count == 3:
            sizes = tuple(module._get_shard_size_mapping(name) for name in ("q", "k", "v"))
            if any(size is None for size in sizes):
                raise ValueError("Unable to resolve QKV LoRA output slices")
            return tuple(int(size) for size in sizes if size is not None)
        if isinstance(module, MergedColumnParallelLinear) and slice_count == len(module.output_sizes):
            return tuple(int(size // module.tp_size) for size in module.output_sizes)
        if slice_count != 1:
            raise ValueError(f"Unsupported packed LoRA binding for {type(module).__name__}")
        if isinstance(module, ColumnParallelLinear):
            return (int(module.output_size_per_partition),)
        if isinstance(module, (RowParallelLinear, ReplicatedLinear)):
            return (int(module.output_size),)
        if isinstance(module, nn.Linear):
            return (int(module.out_features),)
        raise TypeError(f"Default diffusion LoRA executor cannot wrap {type(module).__name__}")

    def activate(self, composition: DiffusionLoRAComposition) -> None:
        for bank in self.banks:
            bank.activate(composition)

    def _lora_input(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.base_layer, RowParallelLinear) and not self.base_layer.input_is_parallel:
            return split_tensor_along_last_dim(x, self.base_layer.tp_size)[self.base_layer.tp_rank].contiguous()
        return x

    def forward(self, x: torch.Tensor, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        output = result[0] if isinstance(result, tuple) else result
        lora_input = self._lora_input(x)
        deltas: list[torch.Tensor] = []
        for bank_index, output_size in zip(self._bank_indices, self._slice_output_sizes, strict=True):
            if bank_index is None:
                deltas.append(output.new_zeros((*output.shape[:-1], output_size)))
            else:
                deltas.append(self.banks[bank_index](lora_input))
        delta = deltas[0] if len(deltas) == 1 else torch.cat(deltas, dim=-1)
        output = output + delta
        if not isinstance(result, tuple):
            return output
        return (output, *result[1:])

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            base_layer = object.__getattribute__(self, "_modules").get("base_layer")
            if base_layer is None:
                raise exc
            try:
                return getattr(base_layer, name)
            except AttributeError:
                raise exc
