# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Resident stage-2 LoRA weights for official LTX two-stage pipelines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open
from vllm.logger import init_logger

from .ltx2_components import create_transformer_from_config

logger = init_logger(__name__)

_IGNORED_OFFICIAL_LORA_MODULES = {"text_embedding_projection.aggregate_embed"}
_OFFICIAL_TO_DIFFUSERS_PREFIXES = (
    ("audio_prompt_adaln_single.", "audio_prompt_adaln."),
    ("prompt_adaln_single.", "prompt_adaln."),
    ("audio_adaln_single.", "audio_time_embed."),
    ("adaln_single.", "time_embed."),
    ("audio_patchify_proj", "audio_proj_in"),
    ("patchify_proj", "proj_in"),
    ("av_ca_video_scale_shift_adaln_single.", "av_cross_attn_video_scale_shift."),
    ("av_ca_audio_scale_shift_adaln_single.", "av_cross_attn_audio_scale_shift."),
    ("av_ca_a2v_gate_adaln_single.", "av_cross_attn_video_a2v_gate."),
    ("av_ca_v2a_gate_adaln_single.", "av_cross_attn_audio_v2a_gate."),
)


@dataclass(frozen=True)
class _LTXLoRAEntry:
    module_name: str
    target_name: str
    shard_id: str | int | None
    lora_a: torch.Tensor
    lora_b: torch.Tensor


def _to_diffusers_module_name(name: str) -> str:
    for official_prefix, diffusers_prefix in _OFFICIAL_TO_DIFFUSERS_PREFIXES:
        if name.startswith(official_prefix):
            return diffusers_prefix + name[len(official_prefix) :]
    return name


def _resolve_lora_target(transformer: nn.Module, module_name: str) -> tuple[str, str | int | None] | None:
    modules = dict(transformer.named_modules())
    if module_name in modules:
        return module_name, None
    for packed_suffix, sub_suffix, shard_id in getattr(transformer, "stacked_params_mapping", ()):
        if sub_suffix in module_name:
            packed_name = module_name.replace(sub_suffix, packed_suffix)
            if packed_name in modules:
                return packed_name, shard_id
    return None


def _load_lora_entries(transformer: nn.Module, path: str, dtype: torch.dtype) -> list[_LTXLoRAEntry]:
    entries: list[_LTXLoRAEntry] = []
    unresolved: list[str] = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor_names = set(handle.keys())
        module_names = sorted(
            name.removeprefix("diffusion_model.").rsplit(".lora_", 1)[0]
            for name in tensor_names
            if name.endswith(".lora_A.weight")
        )
        for official_name in module_names:
            if official_name in _IGNORED_OFFICIAL_LORA_MODULES:
                continue
            key_prefix = f"diffusion_model.{official_name}"
            key_a = f"{key_prefix}.lora_A.weight"
            key_b = f"{key_prefix}.lora_B.weight"
            if key_b not in tensor_names:
                raise ValueError(f"Missing paired LoRA tensor {key_b!r} in {path}.")

            module_name = _to_diffusers_module_name(official_name)
            target = _resolve_lora_target(transformer, module_name)
            if target is None:
                unresolved.append(official_name)
                continue

            lora_a = handle.get_tensor(key_a).to(dtype=dtype)
            lora_b = handle.get_tensor(key_b).to(dtype=dtype)
            if lora_b.shape[1] != lora_a.shape[0]:
                raise ValueError(
                    f"Invalid LoRA pair for {official_name}: A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}."
                )
            entries.append(
                _LTXLoRAEntry(
                    module_name=module_name,
                    target_name=target[0],
                    shard_id=target[1],
                    lora_a=lora_a,
                    lora_b=lora_b,
                )
            )

    if unresolved:
        raise ValueError(f"Official LTX LoRA contains unmapped modules: {unresolved}.")
    if not entries:
        raise ValueError(f"No Transformer LoRA tensors were loaded from {path}.")
    return entries


class LTXResidentLoRAController:
    """Keep base and stage-2 LoRA weights as two offloadable DiT modules."""

    def __init__(self, pipeline: Any, adapter_path: str) -> None:
        self.pipeline = pipeline
        self.adapter_path = adapter_path
        self.active_transformer_name = "transformer"
        self._merged = False

        if getattr(pipeline.od_config, "quantization_config", None) is not None:
            raise ValueError("LTX resident LoRA mode does not support quantized Transformer weights yet.")

        transformer_config = dict(pipeline.transformer.config)
        pipeline.transformer_2 = create_transformer_from_config(transformer_config)
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

    def after_weights_loaded(self) -> None:
        if self._merged:
            return
        entries = _load_lora_entries(
            self.pipeline.transformer_2,
            self.adapter_path,
            self.pipeline.od_config.dtype,
        )
        self._merge_entries(entries)
        self._merged = True

    @torch.no_grad()
    def _merge_entries(self, entries: list[_LTXLoRAEntry]) -> None:
        grouped: dict[str, list[_LTXLoRAEntry]] = {}
        for entry in entries:
            grouped.setdefault(entry.target_name, []).append(entry)

        modules = dict(self.pipeline.transformer_2.named_modules())
        fusion_device = torch.device(self.pipeline.device)
        if fusion_device.type == "cuda" and not torch.cuda.is_available():
            fusion_device = torch.device("cpu")

        for target_name, target_entries in grouped.items():
            module = modules.get(target_name)
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor):
                raise TypeError(f"Expected a weight tensor at transformer_2.{target_name}.")
            if not weight.is_floating_point():
                raise TypeError(f"Cannot merge LTX LoRA into non-floating weight transformer_2.{target_name}.")

            weight_loader = getattr(weight, "weight_loader", None)
            if weight_loader is None:
                if len(target_entries) != 1 or target_entries[0].shard_id is not None:
                    raise ValueError(f"Cannot merge packed LoRA shards into transformer_2.{target_name}.")
                delta = self._compute_delta(target_entries[0], fusion_device, weight.dtype)
                weight.add_(delta.to(device=weight.device, dtype=weight.dtype))
                continue

            base_weight = weight.detach().clone()
            weight.zero_()
            try:
                for entry in target_entries:
                    delta = self._compute_delta(entry, fusion_device, weight.dtype)
                    if entry.shard_id is None:
                        weight_loader(weight, delta)
                    else:
                        weight_loader(weight, delta, entry.shard_id)
                weight.add_(base_weight)
            except Exception:
                weight.copy_(base_weight)
                raise

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
