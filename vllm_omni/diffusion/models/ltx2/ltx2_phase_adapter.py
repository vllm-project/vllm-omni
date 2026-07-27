# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Runtime for a fixed LTX refinement-phase adapter slot.

This module intentionally knows nothing about official LTX safetensors names.
It receives an :class:`AdapterManifest` and makes the existing LTX transformer
linears adapter-capable while retaining their original quantization and tensor
parallel implementations.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear

from .ltx2_adapter_parser import AdapterManifest, AdapterTarget, iter_adapter_tensors

logger = init_logger(__name__)


@dataclass(frozen=True)
class _AdapterLayout:
    """Rank-local storage and execution layout for one LoRA pair."""

    kind: str
    input_size: int
    output_size: int
    a_input_start: int
    a_input_size: int
    b_output_start: int
    b_output_size: int
    output_start: int
    reduce_results: bool = False


def _module_device(module: nn.Module) -> torch.device:
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.device.type != "meta":
            return tensor.device
    raise RuntimeError(f"Cannot materialize an LTX phase adapter for meta module {module._get_name()}.")


def _build_layout(layer: nn.Module, target: AdapterTarget) -> _AdapterLayout:
    if isinstance(layer, QKVParallelLinear):
        if target.slice not in {"q", "k", "v"}:
            raise ValueError(
                f"LTX packed QKV target {target.module!r} requires a q/k/v adapter slice, got {target.slice!r}."
            )
        q_local = layer.num_heads * layer.head_size
        k_local = layer.num_kv_heads * layer.head_size
        v_local = layer.num_kv_heads * layer.v_head_size
        if target.slice == "q":
            output_size = layer.total_num_heads * layer.head_size
            b_output_start = layer.tp_rank * q_local
            b_output_size = q_local
            output_start = 0
        elif target.slice == "k":
            output_size = layer.total_num_kv_heads * layer.head_size
            shard_rank = layer.tp_rank // layer.num_kv_head_replicas
            b_output_start = shard_rank * k_local
            b_output_size = k_local
            output_start = q_local
        else:
            output_size = layer.total_num_kv_heads * layer.v_head_size
            shard_rank = layer.tp_rank // layer.num_kv_head_replicas
            b_output_start = shard_rank * v_local
            b_output_size = v_local
            output_start = q_local + k_local
        return _AdapterLayout(
            kind="qkv",
            input_size=layer.input_size,
            output_size=output_size,
            a_input_start=0,
            a_input_size=layer.input_size,
            b_output_start=b_output_start,
            b_output_size=b_output_size,
            output_start=output_start,
        )

    if isinstance(layer, ColumnParallelLinear):
        if layer.gather_output:
            raise ValueError(
                f"LTX phase adapter does not support gathered ColumnParallelLinear target {target.module!r}."
            )
        return _AdapterLayout(
            kind="column",
            input_size=layer.input_size,
            output_size=layer.output_size,
            a_input_start=0,
            a_input_size=layer.input_size,
            b_output_start=layer.tp_rank * layer.output_size_per_partition,
            b_output_size=layer.output_size_per_partition,
            output_start=0,
        )

    if isinstance(layer, RowParallelLinear):
        return _AdapterLayout(
            kind="row",
            input_size=layer.input_size,
            output_size=layer.output_size,
            a_input_start=layer.tp_rank * layer.input_size_per_partition,
            a_input_size=layer.input_size_per_partition,
            b_output_start=0,
            b_output_size=layer.output_size,
            output_start=0,
            reduce_results=layer.reduce_results,
        )

    if isinstance(layer, nn.Linear):
        return _AdapterLayout(
            kind="replicated",
            input_size=layer.in_features,
            output_size=layer.out_features,
            a_input_start=0,
            a_input_size=layer.in_features,
            b_output_start=0,
            b_output_size=layer.out_features,
            output_start=0,
        )

    raise TypeError(f"LTX phase adapter target {target.module!r} has unsupported layer type {type(layer).__name__}.")


class _AdapterPiece(nn.Module):
    """One rank-local A/B pair stored as non-persistent buffers."""

    def __init__(self, target: AdapterTarget, layout: _AdapterLayout) -> None:
        super().__init__()
        self.target = target
        self.layout = layout
        # The fixed graph is installed before base checkpoint loading.  The
        # empty buffers are replaced only after the base layer's quant/TP layout
        # has settled and before offload/compile snapshots are taken.
        self.register_buffer("lora_a", torch.empty(0), persistent=False)
        self.register_buffer("lora_b", torch.empty(0), persistent=False)

    def load(self, lora_a: torch.Tensor, lora_b: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> None:
        if tuple(lora_a.shape) != self.target.a_source.shape or tuple(lora_b.shape) != self.target.b_source.shape:
            raise ValueError(
                f"LTX adapter source shape changed for {self.target.source_module!r}: "
                f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}."
            )
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(f"LTX adapter target {self.target.source_module!r} must use rank-2 A/B tensors.")
        if lora_a.shape[0] != self.target.rank or lora_b.shape[1] != self.target.rank:
            raise ValueError(
                f"LTX adapter rank mismatch for {self.target.source_module!r}: "
                f"expected {self.target.rank}, got A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}."
            )
        if lora_a.shape[1] != self.layout.input_size or lora_b.shape[0] != self.layout.output_size:
            raise ValueError(
                f"LTX adapter shape does not match {self.target.module!r}: "
                f"expected A=({self.target.rank}, {self.layout.input_size}), "
                f"B=({self.layout.output_size}, {self.target.rank}); "
                f"got A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}."
            )

        local_a = lora_a.narrow(1, self.layout.a_input_start, self.layout.a_input_size)
        local_b = lora_b.narrow(0, self.layout.b_output_start, self.layout.b_output_size)
        self.lora_a = local_a.to(device=device, dtype=dtype)
        self.lora_b = local_b.to(device=device, dtype=dtype)

    def delta(self, input_: torch.Tensor) -> torch.Tensor:
        if self.lora_a.numel() == 0 or self.lora_b.numel() == 0:
            raise RuntimeError(f"LTX phase adapter target {self.target.module!r} was not materialized.")
        if self.layout.kind == "row" and input_.shape[-1] != self.layout.a_input_size:
            input_ = input_.narrow(-1, self.layout.a_input_start, self.layout.a_input_size)
        input_ = input_.to(dtype=self.lora_a.dtype)
        delta = F.linear(F.linear(input_, self.lora_a), self.lora_b)
        if self.layout.kind == "row" and self.layout.reduce_results:
            delta = tensor_model_parallel_all_reduce(delta)
        return delta


class _PhaseAdapterLinear(nn.Module):
    """Delegate the base linear unchanged and add an enabled adapter slot."""

    def __init__(
        self,
        base_layer: nn.Module,
        targets: Iterable[AdapterTarget],
        *,
        adapter_name: str,
        adapter_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.adapter_dtype = adapter_dtype
        pieces = [_AdapterPiece(target, _build_layout(base_layer, target)) for target in targets]
        self.adapters = nn.ModuleDict({adapter_name: nn.ModuleList(pieces)})
        self._pieces_by_target = {piece.target: piece for piece in pieces}
        self._active_adapter: str | None = None

    def load_target(self, target: AdapterTarget, lora_a: torch.Tensor, lora_b: torch.Tensor) -> None:
        try:
            piece = self._pieces_by_target[target]
        except KeyError as exc:
            raise ValueError(f"Adapter target {target.module!r} is not installed on this layer.") from exc
        piece.load(lora_a, lora_b, device=_module_device(self.base_layer), dtype=self.adapter_dtype)

    def set_active(self, adapter_name: str | None) -> None:
        if adapter_name is not None and adapter_name not in self.adapters:
            raise ValueError(f"Unknown LTX phase adapter slot {adapter_name!r}.")
        self._active_adapter = adapter_name

    def _add_adapter_delta(self, input_: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        adapter_name = self._active_adapter
        if adapter_name is None:
            return output
        if output.requires_grad:
            output = output.clone()
        for piece in self.adapters[adapter_name]:
            delta = piece.delta(input_).to(dtype=output.dtype)
            start = piece.layout.output_start
            output[..., start : start + piece.layout.b_output_size].add_(delta)
        return output

    def forward(self, input_: torch.Tensor):
        output = self.base_layer(input_)
        if self._active_adapter is None:
            return output
        if isinstance(output, tuple):
            return (self._add_adapter_delta(input_, output[0]), *output[1:])
        return self._add_adapter_delta(input_, output)


class LTXPhaseAdapterRuntime:
    """Install and operate one fixed phase adapter on an existing transformer."""

    def __init__(self, transformer: nn.Module, manifest: AdapterManifest, *, dtype: torch.dtype) -> None:
        self.transformer = transformer
        self.manifest = manifest
        self.dtype = dtype
        self._wrappers: dict[str, _PhaseAdapterLinear] = {}
        self._target_names: tuple[str, ...] = ()
        self._installed = False
        self._materialized = False

    def install_structure(self) -> None:
        """Replace target linears before base checkpoint loading begins."""
        if self._installed:
            raise RuntimeError("LTX phase adapter structure is already installed.")
        grouped: dict[str, list[AdapterTarget]] = defaultdict(list)
        for target in self.manifest.targets:
            grouped[target.module].append(target)

        original_layers = {name: self.transformer.get_submodule(name) for name in grouped}
        for name, targets in grouped.items():
            wrapper = _PhaseAdapterLinear(
                original_layers[name],
                targets,
                adapter_name=self.manifest.name,
                adapter_dtype=self.dtype,
            )
            parent_name, _, child_name = name.rpartition(".")
            parent = self.transformer.get_submodule(parent_name) if parent_name else self.transformer
            if child_name not in parent._modules:
                raise ValueError(f"LTX phase adapter target {name!r} is no longer a direct module child.")
            parent._modules[child_name] = wrapper
            self._wrappers[name] = wrapper

        self._target_names = tuple(sorted(self._wrappers, key=len, reverse=True))
        # LTX's custom loader receives the original checkpoint names.  Keep the
        # adaptation local to that loader rather than changing AutoWeightsLoader.
        self.transformer._phase_adapter_parameter_name = self.base_parameter_name
        self._installed = True
        logger.info("Installed %d LTX phase-adapter linears for slot %s", len(self._wrappers), self.manifest.name)

    def base_parameter_name(self, name: str) -> str:
        for target_name in self._target_names:
            prefix = f"{target_name}."
            if name.startswith(prefix):
                return f"{target_name}.base_layer.{name[len(prefix) :]}"
        return name

    def prepare_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Keep base weights on the transformer's standard loading path."""
        return weights

    def finalize(self) -> None:
        """Stream rank-local A/B buffers after base weight processing completes."""
        if not self._installed:
            raise RuntimeError("LTX phase adapter structure must be installed before materialization.")
        if self._materialized:
            return
        loaded: set[AdapterTarget] = set()
        for target, lora_a, lora_b in iter_adapter_tensors(self.manifest):
            self._wrappers[target.module].load_target(target, lora_a, lora_b)
            loaded.add(target)
        missing = set(self.manifest.targets) - loaded
        if missing:
            missing_modules = sorted(target.module for target in missing)
            raise RuntimeError(f"LTX phase adapter did not load targets: {missing_modules}.")
        self._materialized = True
        logger.info("Materialized %d LTX phase-adapter tensor pairs for slot %s", len(loaded), self.manifest.name)

    def activate(self, adapter_slot: str | None) -> None:
        if adapter_slot is not None:
            if adapter_slot != self.manifest.name:
                raise ValueError(f"Unknown LTX phase adapter slot {adapter_slot!r}.")
            if not self._materialized:
                raise RuntimeError("LTX phase adapter data must be materialized before activation.")
        for wrapper in self._wrappers.values():
            wrapper.set_active(adapter_slot)
