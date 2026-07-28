# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Phase-specific Transformer weights for LTX pipeline recipes."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch
import torch.nn as nn
from vllm.logger import init_logger

from .ltx2_adapter_parser import LTXAdapterParser, iter_adapter_tensors
from .ltx2_components import create_transformer_from_config, resolve_ltx_artifact
from .ltx2_phase_adapter import LTXPhaseAdapterRuntime

logger = init_logger(__name__)

_LTX_TWO_STAGE_LORA_MODE_ENV = "VLLM_OMNI_LTX_TWO_STAGE_LORA_MODE"


class LTXPhaseWeights(Protocol):
    """Common lifecycle for resident and dynamic phase-specific weights."""

    @property
    def transformer(self) -> nn.Module: ...

    def prepare_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]: ...

    def finalize(self) -> None: ...

    def activate(self, adapter_slot: str | None) -> None: ...


@dataclass(frozen=True)
class _LTXLoRAEntry:
    source_module: str
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


def _resolve_two_stage_lora_mode() -> str:
    mode = os.getenv(_LTX_TWO_STAGE_LORA_MODE_ENV, "dynamic").strip().lower()
    if mode not in {"resident", "dynamic"}:
        raise ValueError(f"{_LTX_TWO_STAGE_LORA_MODE_ENV} must be either 'resident' or 'dynamic', got {mode!r}.")
    return mode


def _load_resident_lora_entries(transformer: nn.Module, path: str, dtype: torch.dtype) -> list[_LTXLoRAEntry]:
    """Materialize parser output only for the resident pre-merge path."""
    manifest = LTXAdapterParser(transformer).parse(path)
    return [
        _LTXLoRAEntry(
            source_module=target.source_module,
            lora_a=lora_a.to(dtype=dtype),
            lora_b=lora_b.to(dtype=dtype),
        )
        for target, lora_a, lora_b in iter_adapter_tensors(manifest)
    ]


class LTXResidentLoRAController:
    """Keep base and refinement weights as two offloadable DiT modules."""

    def __init__(self, pipeline: Any, adapter_path: str) -> None:
        self.pipeline = pipeline
        self.adapter_path = adapter_path
        self.active_transformer_name = "transformer"
        self._merged = False

        quant_config = getattr(pipeline.od_config, "quantization_config", None)
        if _uses_serialized_quantization(quant_config):
            raise ValueError(
                "LTX resident LoRA mode requires an unquantized base checkpoint so the refinement adapter can be "
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
        logger.info("Using resident LTX refinement-phase LoRA weights")

    @property
    def transformer(self) -> nn.Module:
        return getattr(self.pipeline, self.active_transformer_name)

    @torch.no_grad()
    def prepare_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Merge LoRA into full refinement tensors before the standard loader."""
        if self._merged:
            raise RuntimeError("LTX resident LoRA weights have already been merged.")

        entries = _load_resident_lora_entries(
            self.pipeline.transformer_2,
            self.adapter_path,
            self.pipeline.od_config.dtype,
        )
        entries_by_module: dict[str, _LTXLoRAEntry] = {}
        for entry in entries:
            if entry.source_module in entries_by_module:
                raise ValueError(f"Official LTX LoRA contains duplicate module {entry.source_module!r}.")
            entries_by_module[entry.source_module] = entry

        fusion_device = torch.device(self.pipeline.device)
        if fusion_device.type == "cuda" and not torch.cuda.is_available():
            fusion_device = torch.device("cpu")
        merged_modules: set[str] = set()
        refinement_prefix = "transformer_2."
        weight_suffix = ".weight"

        for name, weight in weights:
            module_name = None
            if name.startswith(refinement_prefix) and name.endswith(weight_suffix):
                module_name = name[len(refinement_prefix) : -len(weight_suffix)]
            entry = entries_by_module.get(module_name) if module_name is not None else None
            if entry is not None:
                delta = self._compute_delta(entry, fusion_device, weight.dtype)
                if delta.shape != weight.shape:
                    raise ValueError(
                        f"Cannot match LoRA delta {tuple(delta.shape)} to checkpoint tensor "
                        f"{name} {tuple(weight.shape)}."
                    )
                weight = weight.to(device=fusion_device) + delta
                merged_modules.add(entry.source_module)
            yield name, weight

        missing = set(entries_by_module) - merged_modules
        if missing:
            raise ValueError(f"LTX refinement checkpoint is missing LoRA target weights: {sorted(missing)}.")
        self._merged = True

    @staticmethod
    def _compute_delta(entry: _LTXLoRAEntry, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        lora_a = entry.lora_a.to(device=device, dtype=dtype)
        lora_b = entry.lora_b.to(device=device, dtype=dtype)
        return torch.matmul(lora_b, lora_a)

    def finalize(self) -> None:
        if not self._merged:
            raise RuntimeError("LTX resident LoRA weights have not been finalized.")

    def activate(self, adapter_slot: str | None) -> None:
        self.finalize()
        if adapter_slot is None:
            self.active_transformer_name = "transformer"
        elif adapter_slot == "ltx_distilled":
            self.active_transformer_name = "transformer_2"
        else:
            raise ValueError(f"Unknown LTX phase adapter slot {adapter_slot!r}.")


def build_ltx_phase_weights(pipeline: Any) -> LTXPhaseWeights | None:
    """Build the fixed phase-weight strategy required by a pipeline recipe."""
    adapter_slots = {phase.adapter_slot for phase in pipeline.pipeline_recipe.phases if phase.adapter_slot is not None}
    if not adapter_slots:
        return None
    if adapter_slots != {"ltx_distilled"}:
        raise ValueError(f"Unsupported LTX phase adapter slots: {sorted(adapter_slots)}.")

    profile = pipeline.component_profile
    if profile.distilled_lora_filename is None or profile.artifact_repo_id is None:
        raise ValueError(f"{profile.name} does not declare the required distilled adapter artifact.")
    if getattr(pipeline.od_config, "lora_path", None) is not None:
        raise ValueError(
            f"{pipeline.__class__.__name__} reserves LoRA execution for its phase adapter; "
            "request or static LoRA composition is not supported yet."
        )

    mode = _resolve_two_stage_lora_mode()
    model_paths = getattr(pipeline.od_config, "model_paths", {}) or {}
    adapter_path = resolve_ltx_artifact(
        pipeline.od_config.model,
        profile.artifact_repo_id,
        profile.distilled_lora_filename,
        explicit_path=model_paths.get("distilled_lora"),
    )

    if mode == "resident":
        return LTXResidentLoRAController(pipeline, adapter_path)

    manifest = LTXAdapterParser(pipeline.transformer).parse(adapter_path, name="ltx_distilled")
    runtime = LTXPhaseAdapterRuntime(pipeline.transformer, manifest, dtype=pipeline.od_config.dtype)
    runtime.install_structure()
    return runtime
