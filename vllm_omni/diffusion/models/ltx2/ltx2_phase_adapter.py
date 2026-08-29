# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Construction and execution of a fixed LTX refinement-phase adapter.

Checkpoint key parsing remains in :mod:`ltx2_adapter_parser`; this module
resolves the declared artifact and makes the existing Transformer linears
adapter-capable while retaining their quantization and tensor-parallel paths.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm.envs as envs
from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.platforms import current_platform

from .ltx2_adapter_parser import AdapterManifest, AdapterTarget, iter_adapter_tensors, parse_ltx_adapter
from .ltx2_components import resolve_ltx_artifact
from .ltx2_recipes import LTX_DISTILLED_ADAPTER_SLOT

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
        if tuple(lora_a.shape) != self.target.a_shape or tuple(lora_b.shape) != self.target.b_shape:
            raise ValueError(
                f"LTX adapter source shape changed for {self.target.source_module!r}: "
                f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}."
            )
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(f"LTX adapter target {self.target.source_module!r} must use rank-2 A/B tensors.")
        rank = self.target.a_shape[0]
        if lora_a.shape[1] != self.layout.input_size or lora_b.shape[0] != self.layout.output_size:
            raise ValueError(
                f"LTX adapter shape does not match {self.target.module!r}: "
                f"expected A=({rank}, {self.layout.input_size}), "
                f"B=({self.layout.output_size}, {rank}); "
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

    def weight_delta(self) -> torch.Tensor:
        """Materialize the rank-local BF16 weight delta used by official fusion."""
        if self.lora_a.numel() == 0 or self.lora_b.numel() == 0:
            raise RuntimeError(f"LTX phase adapter target {self.target.module!r} was not materialized.")
        return torch.matmul(self.lora_b, self.lora_a)


def _apply_unquantized_gemm_with_weight(
    layer: nn.Module,
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # GEMM dispatch is selected from the host platform, not the tensor device.
    # In CPU tests on a ROCm host that selects rocm_unquantized_gemm, which has
    # no CPU kernel. Keep that combination on PyTorch's portable implementation.
    if current_platform.is_rocm() and input_.device.type == "cpu":
        return F.linear(input_, weight, bias)
    if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
        return linear_batch_invariant(input_, weight, bias)
    return dispatch_unquantized_gemm()(layer, input_, weight, bias)


def _forward_with_weight(
    layer: nn.Module,
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Run an unquantized base layer with an ephemeral fused weight."""
    if isinstance(layer, nn.Linear):
        return F.linear(input_, weight, layer.bias)

    quant_method = getattr(layer, "quant_method", None)
    if not isinstance(quant_method, UnquantizedLinearMethod):
        raise ValueError("LTX layer-fused phase LoRA requires unquantized Transformer linears.")

    if isinstance(layer, RowParallelLinear):
        if layer.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = split_tensor_along_last_dim(input_, num_partitions=layer.tp_size)[
                layer.tp_rank
            ].contiguous()
        bias = None if layer.tp_rank > 0 or layer.skip_bias_add else layer.bias
        output_parallel = _apply_unquantized_gemm_with_weight(layer, input_parallel, weight, bias)
        if layer.reduce_results and layer.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel
    elif isinstance(layer, ColumnParallelLinear):
        bias = layer.bias if not layer.skip_bias_add else None
        output = _apply_unquantized_gemm_with_weight(layer, input_, weight, bias)
    else:
        raise TypeError(f"Unsupported LTX layer-fused linear type {type(layer).__name__}.")

    if not layer.return_bias:
        return output
    return output, layer.bias if layer.skip_bias_add else None


class _PhaseAdapterLinear(nn.Module):
    """Delegate the base linear unchanged and add an enabled adapter slot."""

    def __init__(
        self,
        base_layer: nn.Module,
        targets: Iterable[AdapterTarget],
        *,
        adapter_dtype: torch.dtype,
        layer_fused: bool = False,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.adapter_dtype = adapter_dtype
        if layer_fused and not (
            isinstance(base_layer, nn.Linear)
            or isinstance(getattr(base_layer, "quant_method", None), UnquantizedLinearMethod)
        ):
            raise ValueError("LTX layer-fused phase LoRA requires unquantized Transformer linears.")
        self.layer_fused = layer_fused
        pieces = [_AdapterPiece(target, _build_layout(base_layer, target)) for target in targets]
        self.pieces = nn.ModuleList(pieces)
        self._pieces_by_target = {piece.target: piece for piece in pieces}
        self._enabled = False

    def load_target(self, target: AdapterTarget, lora_a: torch.Tensor, lora_b: torch.Tensor) -> None:
        try:
            piece = self._pieces_by_target[target]
        except KeyError as exc:
            raise ValueError(f"Adapter target {target.module!r} is not installed on this layer.") from exc
        piece.load(lora_a, lora_b, device=_module_device(self.base_layer), dtype=self.adapter_dtype)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def _add_adapter_delta(self, input_: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return output
        if output.requires_grad:
            output = output.clone()
        for piece in self.pieces:
            delta = piece.delta(input_).to(dtype=output.dtype)
            start = piece.layout.output_start
            output[..., start : start + piece.layout.b_output_size].add_(delta)
        return output

    @torch.no_grad()
    def _fused_weight(self) -> torch.Tensor:
        base_weight = self.base_layer.weight
        fused_weight = base_weight.clone()
        for piece in self.pieces:
            start = piece.layout.output_start
            base_slice = base_weight.narrow(0, start, piece.layout.b_output_size)
            delta = piece.weight_delta()
            if delta.shape != base_slice.shape:
                raise ValueError(
                    f"LTX layer-fused delta shape {tuple(delta.shape)} does not match base weight slice "
                    f"{tuple(base_slice.shape)} for {piece.target.module!r}."
                )
            # Match official fusion exactly: materialize B@A in the adapter
            # dtype, then add the base tensor in-place before the main GEMM.
            delta.add_(base_slice)
            fused_weight.narrow(0, start, piece.layout.b_output_size).copy_(delta)
        return fused_weight

    def forward(self, input_: torch.Tensor):
        if not self._enabled:
            return self.base_layer(input_)
        if self.layer_fused:
            return _forward_with_weight(self.base_layer, input_, self._fused_weight())
        output = self.base_layer(input_)
        if isinstance(output, tuple):
            return (self._add_adapter_delta(input_, output[0]), *output[1:])
        return self._add_adapter_delta(input_, output)


class LTXPhaseAdapterRuntime:
    """Install and operate one fixed phase adapter on an existing transformer."""

    def __init__(
        self,
        transformer: nn.Module,
        manifest: AdapterManifest,
        *,
        dtype: torch.dtype,
        layer_fused: bool = False,
    ) -> None:
        self.transformer = transformer
        self.manifest = manifest
        self._wrappers: dict[str, _PhaseAdapterLinear] = {}
        self._target_names: tuple[str, ...] = ()
        self._materialized = False
        grouped: dict[str, list[AdapterTarget]] = defaultdict(list)
        for target in manifest.targets:
            grouped[target.module].append(target)

        original_layers = {name: self.transformer.get_submodule(name) for name in grouped}
        for name, targets in grouped.items():
            wrapper = _PhaseAdapterLinear(
                original_layers[name],
                targets,
                adapter_dtype=dtype,
                layer_fused=layer_fused,
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
        logger.info(
            "Installed %d LTX %s phase-adapter linears",
            len(self._wrappers),
            "layer-fused" if layer_fused else "dynamic",
        )

    def base_parameter_name(self, name: str) -> str:
        for target_name in self._target_names:
            prefix = f"{target_name}."
            if name.startswith(prefix):
                return f"{target_name}.base_layer.{name[len(prefix) :]}"
        return name

    def finalize(self) -> None:
        """Stream rank-local A/B buffers after base weight processing completes."""
        if self._materialized:
            return
        for target, lora_a, lora_b in iter_adapter_tensors(self.manifest):
            self._wrappers[target.module].load_target(target, lora_a, lora_b)
        self._materialized = True
        logger.info("Materialized %d LTX phase-adapter tensor pairs", len(self.manifest.targets))

    def activate(self, adapter_slot: str | None) -> None:
        if adapter_slot is not None:
            if adapter_slot != LTX_DISTILLED_ADAPTER_SLOT:
                raise ValueError(f"Unknown LTX phase adapter slot {adapter_slot!r}.")
            if not self._materialized:
                raise RuntimeError("LTX phase adapter data must be materialized before activation.")
        for wrapper in self._wrappers.values():
            wrapper.set_enabled(adapter_slot is not None)


def build_ltx_phase_adapter(pipeline: Any) -> LTXPhaseAdapterRuntime | None:
    """Build the fixed adapter required by a multi-phase LTX recipe."""
    adapter_slots = {phase.adapter_slot for phase in pipeline.pipeline_recipe.phases if phase.adapter_slot is not None}
    if not adapter_slots:
        return None
    if adapter_slots != {LTX_DISTILLED_ADAPTER_SLOT}:
        raise ValueError(f"Unsupported LTX phase adapter slots: {sorted(adapter_slots)}.")

    profile = pipeline.component_profile
    if profile.distilled_lora_filename is None or profile.artifact_repo_id is None:
        raise ValueError(f"{profile.name} does not declare the required distilled adapter artifact.")
    if getattr(pipeline.od_config, "lora_path", None) is not None:
        raise ValueError(
            f"{pipeline.__class__.__name__} reserves LoRA execution for its phase adapter; "
            "request or static LoRA composition is not supported yet."
        )

    quantization_config = getattr(pipeline.od_config, "quantization_config", None)
    layer_fused = quantization_config is None
    adapter_path = resolve_ltx_artifact(
        pipeline.od_config.model,
        profile.artifact_repo_id,
        profile.distilled_lora_filename,
        model_revision=getattr(pipeline.od_config, "revision", None),
        artifact_revision=profile.artifact_revision,
    )

    if layer_fused and pipeline.od_config.dtype is not torch.bfloat16:
        raise ValueError("LTX layer-fused phase LoRA requires a bfloat16 pipeline dtype.")

    manifest = parse_ltx_adapter(pipeline.transformer, adapter_path)
    runtime = LTXPhaseAdapterRuntime(
        pipeline.transformer,
        manifest,
        dtype=pipeline.od_config.dtype,
        layer_fused=layer_fused,
    )
    return runtime
