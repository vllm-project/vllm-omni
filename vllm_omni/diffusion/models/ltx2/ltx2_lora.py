# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resident stage-2 LoRA weights for official LTX two-stage pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

from .ltx2_adapter_parser import LTXAdapterParser, iter_adapter_tensors
from .ltx2_components import create_transformer_from_config

logger = init_logger(__name__)

@dataclass(frozen=True)
class _LTXLoRAEntry:
    module_name: str
    target_name: str
    shard_id: str | int | None
    lora_a: torch.Tensor
    lora_b: torch.Tensor


def _transformer_config_to_dict(config: Any) -> dict[str, Any]:
    """Copy a Diffusers-style mapping or namespace Transformer config."""
    if isinstance(config, Mapping):
        return dict(config)
    attributes = getattr(config, "__dict__", None)
    if isinstance(attributes, dict):
        return dict(attributes)
    raise TypeError(f"Unsupported LTX Transformer config type: {type(config).__name__}.")


def _uses_serialized_quantization(quant_config: Any) -> bool:
    """Whether Transformer weights are already quantized in the checkpoint."""
    if quant_config is None:
        return False
    if hasattr(quant_config, "resolve"):
        quant_config = quant_config.resolve("transformer")
    if quant_config is None:
        return False
    serialized_flags = (
        "is_checkpoint_quantized",
        "is_checkpoint_fp8_serialized",
        "is_checkpoint_nvfp4_serialized",
        "is_checkpoint_mxfp8_serialized",
        "is_checkpoint_torchao_serialized",
    )
    return getattr(quant_config, "data_type", None) == "mx_fp" or any(
        bool(getattr(quant_config, name, False)) for name in serialized_flags
    )


def _load_resident_lora_entries(transformer: nn.Module, path: str, dtype: torch.dtype) -> list[_LTXLoRAEntry]:
    """Materialize parser output only for the resident pre-merge path."""
    manifest = LTXAdapterParser(transformer).parse(path)
    return [
        _LTXLoRAEntry(
            module_name=target.source_module,
            target_name=target.module,
            shard_id=target.slice,
            lora_a=lora_a.to(dtype=dtype),
            lora_b=lora_b.to(dtype=dtype),
        )
        for target, lora_a, lora_b in iter_adapter_tensors(manifest)
    ]


class LTXResidentLoRAController:
    """Keep base and stage-2 LoRA weights as two offloadable DiT modules."""

    def __init__(self, pipeline: Any, adapter_path: str) -> None:
        self.pipeline = pipeline
        self.adapter_path = adapter_path
        self.active_transformer_name = "transformer"
        self._merged = False

        quant_config = getattr(pipeline.od_config, "quantization_config", None)
        if _uses_serialized_quantization(quant_config):
            raise ValueError(
                "LTX resident LoRA mode requires an unquantized base checkpoint so the stage-2 adapter can be "
                "merged before quantization; serialized quantized checkpoints are not supported."
            )

        transformer_config = _transformer_config_to_dict(
            getattr(pipeline, "_transformer_init_config", pipeline.transformer.config)
        )
        pipeline.transformer_2 = create_transformer_from_config(transformer_config, quant_config=quant_config)
        source = next(
            (source for source in pipeline.weights_sources if source.prefix == "transformer."),
            None,
        )
        if source is None:
            raise RuntimeError("LTX resident LoRA mode requires a transformer weight source.")
        pipeline.weights_sources.append(replace(source, prefix="transformer_2."))
        pipeline._dit_modules = ["transformer", "transformer_2"]
        logger.info("Using resident LTX stage-2 LoRA weights")

    @property
    def transformer(self) -> nn.Module:
        return getattr(self.pipeline, self.active_transformer_name)

    @torch.no_grad()
    def merge_stage2_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Merge LoRA into full stage-2 tensors before the standard loader."""
        if self._merged:
            raise RuntimeError("LTX resident LoRA weights have already been merged.")

        entries = _load_resident_lora_entries(
            self.pipeline.transformer_2,
            self.adapter_path,
            self.pipeline.od_config.dtype,
        )
        entries_by_module: dict[str, _LTXLoRAEntry] = {}
        for entry in entries:
            if entry.module_name in entries_by_module:
                raise ValueError(f"Official LTX LoRA contains duplicate module {entry.module_name!r}.")
            entries_by_module[entry.module_name] = entry

        fusion_device = torch.device(self.pipeline.device)
        if fusion_device.type == "cuda" and not torch.cuda.is_available():
            fusion_device = torch.device("cpu")
        merged_modules: set[str] = set()
        stage2_prefix = "transformer_2."
        weight_suffix = ".weight"

        for name, weight in weights:
            module_name = None
            if name.startswith(stage2_prefix) and name.endswith(weight_suffix):
                module_name = name[len(stage2_prefix) : -len(weight_suffix)]
            entry = entries_by_module.get(module_name) if module_name is not None else None
            if entry is not None:
                delta = self._compute_delta(entry, fusion_device, weight.dtype)
                if delta.shape != weight.shape:
                    raise ValueError(
                        f"Cannot match LoRA delta {tuple(delta.shape)} to checkpoint tensor "
                        f"{name} {tuple(weight.shape)}."
                    )
                weight = weight.to(device=fusion_device) + delta
                merged_modules.add(entry.module_name)
            yield name, weight

        missing = set(entries_by_module) - merged_modules
        if missing:
            raise ValueError(f"LTX stage-2 checkpoint is missing LoRA target weights: {sorted(missing)}.")
        self._merged = True

    @staticmethod
    def _compute_delta(entry: _LTXLoRAEntry, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        lora_a = entry.lora_a.to(device=device, dtype=dtype)
        lora_b = entry.lora_b.to(device=device, dtype=dtype)
        return torch.matmul(lora_b, lora_a)

    def enter(self, transformer_phase: str) -> None:
        if not self._merged:
            raise RuntimeError("LTX resident LoRA weights have not been finalized.")
        if transformer_phase == "base":
            self.active_transformer_name = "transformer"
        elif transformer_phase == "distilled_lora":
            self.active_transformer_name = "transformer_2"
        else:
            raise ValueError(f"Unsupported LTX Transformer phase: {transformer_phase!r}.")

    def set_active(self, adapter_name: str | None) -> None:
        """Match the fixed-slot API used by the dynamic phase adapter."""
        if adapter_name is None:
            self.enter("base")
        elif adapter_name == "ltx_distilled":
            self.enter("distilled_lora")
        else:
            raise ValueError(f"Unknown LTX phase adapter slot {adapter_name!r}.")
