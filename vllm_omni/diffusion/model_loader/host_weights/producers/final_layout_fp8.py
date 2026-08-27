# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded checkpoint-stream producer for final-layout FP8 artifacts."""

from __future__ import annotations

import json
import math
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from safetensors import safe_open
from torch import nn
from vllm import _custom_ops as ops

from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan
from vllm_omni.host_weight_runtime import (
    ArtifactWriter,
    CoordinationScope,
    LookupPhase,
    ProductionMetadata,
    ProductionSourceMode,
    TensorFileWriter,
    TensorWriteSpec,
    WeightProductionSpec,
)

from ..contracts import (
    FINAL_LAYOUT_TENSOR_RESTORER_SCHEMA,
    FinalLayoutContractError,
    final_layout_producer_error,
)
from ..fp8_layout import (
    FINAL_LAYOUT_FP8_MANIFEST_SCHEMA,
    FINAL_LAYOUT_FP8_POLICY,
    FINAL_LAYOUT_FP8_PRODUCER_ID,
)
from ..identity_adapter import FinalLayoutIdentityContext, validate_final_layout_identity
from ..tensor_layout import RuntimeTensorTarget, collect_final_layout_targets, split_tensor_targets_by_bytes

DEFAULT_FP8_SHARD_SIZE_BYTES = 5 * 1024**3
DEFAULT_FP8_QUANT_CHUNK_BYTES = 256 * 1024**2


def _tensor_ranges(path: str) -> dict[str, tuple[int, int]]:
    with open(path, "rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    payload_offset = 8 + header_size
    return {
        name: (payload_offset + entry["data_offsets"][0], entry["data_offsets"][1] - entry["data_offsets"][0])
        for name, entry in header.items()
        if name != "__metadata__"
    }


@dataclass
class _QuantSlot:
    stream: torch.cuda.Stream
    ready: torch.cuda.Event
    host_input: torch.Tensor
    device_input: torch.Tensor
    host_output: torch.Tensor
    device_output: torch.Tensor
    host_amax: torch.Tensor
    pending_start: int | None = None
    pending_rows: int = 0


class FinalLayoutFP8Producer:
    """Build exact runtime FP8 bytes with two bounded CUDA staging slots."""

    def __init__(
        self,
        context: FinalLayoutIdentityContext,
        pipeline: nn.Module,
        dit_modules: Sequence[tuple[str, nn.Module]],
        checkpoint_plan: HostWeightPlan,
        *,
        device: torch.device,
        max_shard_bytes: int = DEFAULT_FP8_SHARD_SIZE_BYTES,
        quant_chunk_bytes: int = DEFAULT_FP8_QUANT_CHUNK_BYTES,
    ) -> None:
        self._context = context
        self._pipeline = pipeline
        self._dit_modules = tuple(dit_modules)
        self._bindings = checkpoint_plan.bindings
        self._device = device
        self._max_shard_bytes = max_shard_bytes
        self._quant_chunk_bytes = quant_chunk_bytes
        self._source_ranges: dict[str, dict[str, tuple[int, int]]] = {}
        self._spec = WeightProductionSpec(
            producer_id=FINAL_LAYOUT_FP8_PRODUCER_ID,
            outputs=(context.identity,),
            source_mode=ProductionSourceMode.CHECKPOINT_STREAM,
            coordination_scope=CoordinationScope.SINGLE_PROCESS,
            lookup_phase=LookupPhase.PRE_LOAD_SAFE,
        )

    @property
    def spec(self) -> WeightProductionSpec:
        return self._spec

    def produce(self, writer: ArtifactWriter) -> ProductionMetadata:
        try:
            return self._produce(writer)
        except FinalLayoutContractError as exc:
            raise final_layout_producer_error(exc) from exc

    def _produce(self, writer: ArtifactWriter) -> ProductionMetadata:
        records = collect_final_layout_targets(
            self._pipeline,
            self._dit_modules,
            policy=FINAL_LAYOUT_FP8_POLICY,
            require_materialized=False,
        )
        contract_digest = validate_final_layout_identity(self._context, records)
        records = tuple(sorted(records, key=self._production_order))
        generated_scales: dict[str, torch.Tensor] = {}

        shards = split_tensor_targets_by_bytes(records, self._max_shard_bytes)
        for index, shard in enumerate(shards, start=1):
            specs = tuple(
                TensorWriteSpec(record.name, tuple(record.tensor.shape), record.tensor.dtype, record.kind, record.role)
                for record in shard
            )
            file_name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
            with writer.open_tensor_file(file_name, specs) as output:
                for record in shard:
                    scale = generated_scales.pop(record.name, None)
                    if scale is not None:
                        output.write_tensor(record.name, scale)
                    elif record.role == "fp8_weight":
                        scale_name = f"{record.name.removesuffix('.weight')}.weight_scale"
                        generated_scales[scale_name] = self._write_fp8_weight(output, record)
                    else:
                        self._write_checkpoint_tensor(output, record)

        if generated_scales:
            raise ValueError(f"unwritten FP8 scales: {sorted(generated_scales)[:5]}")
        return ProductionMetadata(
            producer_schema=FINAL_LAYOUT_FP8_MANIFEST_SCHEMA,
            restorer_schema=FINAL_LAYOUT_TENSOR_RESTORER_SCHEMA,
            format_metadata=FINAL_LAYOUT_FP8_POLICY.build_format_metadata(
                component_names=self._context.dit_names,
                tensor_contract_digest=contract_digest,
                tensor_count=len(records),
            ),
        )

    @staticmethod
    def _production_order(record: RuntimeTensorTarget) -> tuple[str, int]:
        if record.role == "fp8_scale":
            return record.name.removesuffix("_scale"), 1
        return record.name, 0

    def _source(self, name: str) -> torch.Tensor:
        binding = self._bindings[name]
        with safe_open(binding.file_path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(binding.checkpoint_key)
        if binding.transform is not None:
            tensor = binding.transform(tensor)
        return tensor

    def _release_source_pages(self, name: str) -> None:
        binding = self._bindings[name]
        ranges = self._source_ranges.get(binding.file_path)
        if ranges is None:
            ranges = _tensor_ranges(binding.file_path)
            self._source_ranges[binding.file_path] = ranges
        offset, nbytes = ranges[binding.checkpoint_key]
        if not nbytes:
            return
        with open(binding.file_path, "rb") as handle:
            os.posix_fadvise(handle.fileno(), offset, nbytes, os.POSIX_FADV_DONTNEED)

    def _write_checkpoint_tensor(self, output: TensorFileWriter, record: RuntimeTensorTarget) -> None:
        source = self._source(record.name)
        output.write_tensor(record.name, source)
        del source
        self._release_source_pages(record.name)

    def _write_fp8_weight(self, output: TensorFileWriter, record: RuntimeTensorTarget) -> torch.Tensor:
        source = self._source(record.name)
        rows_per_chunk = max(
            1,
            self._quant_chunk_bytes // (source.shape[1] * source.element_size()),
        )
        rows_per_chunk = min(rows_per_chunk, source.shape[0])
        slots = self._make_slots(source, record.tensor.dtype, rows_per_chunk)
        scale = self._find_scale(source, slots, rows_per_chunk)
        self._quantize_rows(output, record.name, source, slots, rows_per_chunk, scale)
        host_scale = slots[0].host_amax
        host_scale.copy_(scale, non_blocking=True)
        del source
        self._release_source_pages(record.name)
        torch.cuda.current_stream(self._device).synchronize()
        return host_scale

    def _make_slots(
        self,
        source: torch.Tensor,
        output_dtype: torch.dtype,
        rows: int,
    ) -> tuple[_QuantSlot, _QuantSlot]:
        shape = (rows, source.shape[1])

        def make_slot() -> _QuantSlot:
            return _QuantSlot(
                stream=torch.cuda.Stream(device=self._device),
                ready=torch.cuda.Event(),
                host_input=torch.empty(shape, dtype=source.dtype, pin_memory=True),
                device_input=torch.empty(shape, dtype=source.dtype, device=self._device),
                host_output=torch.empty(shape, dtype=output_dtype, pin_memory=True),
                device_output=torch.empty(shape, dtype=output_dtype, device=self._device),
                host_amax=torch.empty((1,), dtype=torch.float32, pin_memory=True),
            )

        return make_slot(), make_slot()

    def _find_scale(
        self,
        source: torch.Tensor,
        slots: tuple[_QuantSlot, _QuantSlot],
        rows_per_chunk: int,
    ) -> torch.Tensor:
        amax = torch.zeros((1,), dtype=torch.float32)
        for index, start in enumerate(range(0, source.shape[0], rows_per_chunk)):
            slot = slots[index % 2]
            amax = self._collect_amax(slot, amax)
            rows = min(rows_per_chunk, source.shape[0] - start)
            slot.host_input[:rows].copy_(source[start : start + rows])
            with torch.cuda.stream(slot.stream):
                slot.device_input[:rows].copy_(slot.host_input[:rows], non_blocking=True)
                low, high = slot.device_input[:rows].aminmax()
                slot.host_amax.copy_(torch.maximum(low.abs(), high.abs()).float().reshape(1), non_blocking=True)
                slot.ready.record(slot.stream)
            slot.pending_rows = rows
        for slot in slots:
            amax = self._collect_amax(slot, amax)
        amax_value = amax.item()
        if not math.isfinite(amax_value):
            raise ValueError("cannot quantize a tensor containing non-finite values")
        if amax_value == 0:
            scale = torch.ones((1,), dtype=torch.float32, device=self._device)
        else:
            probe = amax.to(device=self._device, dtype=source.dtype).reshape(1, 1)
            scale = ops.scaled_fp8_quant(probe)[1]
        scale_ready = torch.cuda.Event()
        scale_ready.record(torch.cuda.current_stream(self._device))
        for slot in slots:
            slot.stream.wait_event(scale_ready)
        return scale

    @staticmethod
    def _collect_amax(slot: _QuantSlot, current: torch.Tensor) -> torch.Tensor:
        if not slot.pending_rows:
            return current
        slot.ready.synchronize()
        slot.pending_rows = 0
        torch.maximum(current, slot.host_amax, out=current)
        return current

    def _quantize_rows(
        self,
        output: TensorFileWriter,
        name: str,
        source: torch.Tensor,
        slots: tuple[_QuantSlot, _QuantSlot],
        rows_per_chunk: int,
        scale: torch.Tensor,
    ) -> None:
        for index, start in enumerate(range(0, source.shape[0], rows_per_chunk)):
            slot = slots[index % 2]
            self._flush(output, name, slot)
            rows = min(rows_per_chunk, source.shape[0] - start)
            slot.host_input[:rows].copy_(source[start : start + rows])
            with torch.cuda.stream(slot.stream):
                slot.device_input[:rows].copy_(slot.host_input[:rows], non_blocking=True)
                ops.scaled_fp8_quant(
                    slot.device_input[:rows],
                    scale=scale,
                    output=slot.device_output[:rows],
                )
                slot.host_output[:rows].copy_(slot.device_output[:rows], non_blocking=True)
                slot.ready.record(slot.stream)
            slot.pending_start = start
            slot.pending_rows = rows
        for slot in sorted(slots, key=lambda item: item.pending_start if item.pending_start is not None else -1):
            self._flush(output, name, slot)

    @staticmethod
    def _flush(output: TensorFileWriter, name: str, slot: _QuantSlot) -> None:
        if slot.pending_start is None:
            return
        slot.ready.synchronize()
        output.write_rows(name, slot.pending_start, slot.host_output[: slot.pending_rows])
        slot.pending_start = None
        slot.pending_rows = 0


__all__ = [
    "DEFAULT_FP8_QUANT_CHUNK_BYTES",
    "DEFAULT_FP8_SHARD_SIZE_BYTES",
    "FinalLayoutFP8Producer",
]
