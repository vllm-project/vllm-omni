# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Typed capability contract and registry audit for engine-side merged LoRA."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import torch
import torch.nn as nn
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.model_executor.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)

DEFAULT_DIFFUSION_LORA_COMPONENTS: tuple[str, ...] = (
    "transformer",
    "transformer_2",
    "unet",
)


class MergedLoRACapabilityError(ValueError):
    """Named error raised when a model lacks required merged-LoRA declarations."""


class MergedLoRACapabilityStatus(str, Enum):
    """Declared merged-LoRA readiness status for a registered model."""

    SUPPORTED = "supported"
    RUNTIME_ONLY = "runtime_only"
    NOT_CAPABLE = "not_capable"


@dataclass(frozen=True)
class MergedLoRARegistryEntry:
    """Registry declaration for one diffusion pipeline or omni AR stage model."""

    status: MergedLoRACapabilityStatus
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, MergedLoRACapabilityStatus):
            raise TypeError(f"Expected MergedLoRACapabilityStatus, got {type(self.status).__name__}")
        if self.status is not MergedLoRACapabilityStatus.SUPPORTED and not self.reason.strip():
            raise ValueError(f"Registry status {self.status.value!r} requires a non-empty reason")


def _normalize_packed_mapping(
    packed_modules_mapping: Mapping[str, Sequence[str]],
) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(packed_modules_mapping, Mapping):
        raise MergedLoRACapabilityError(
            f"MergedLoRASpec.packed_modules_mapping must be a mapping, got {type(packed_modules_mapping).__name__}"
        )
    normalized: dict[str, tuple[str, ...]] = {}
    for packed_name, sub_names in packed_modules_mapping.items():
        if not isinstance(packed_name, str) or not packed_name:
            raise MergedLoRACapabilityError(
                f"MergedLoRASpec packed module key must be a non-empty str, got {packed_name!r}"
            )
        if isinstance(sub_names, str) or not isinstance(sub_names, Sequence) or not sub_names:
            raise MergedLoRACapabilityError(
                f"MergedLoRASpec packed module {packed_name!r} must map to a non-empty sequence of str, "
                f"got {type(sub_names).__name__}"
            )
        cleaned: list[str] = []
        for sub in sub_names:
            if not isinstance(sub, str) or not sub:
                raise MergedLoRACapabilityError(
                    f"MergedLoRASpec packed sublayer under {packed_name!r} must be a non-empty str, got {sub!r}"
                )
            if sub in cleaned:
                raise MergedLoRACapabilityError(
                    f"MergedLoRASpec packed module {packed_name!r} has duplicate sublayer {sub!r}"
                )
            cleaned.append(sub)
        normalized[packed_name] = tuple(cleaned)
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class MergedLoRASpec:
    """Immutable, validated merged-LoRA capability specification.

    Every field corresponds to an active model requirement:
    - ``components``: pipeline attribute names scanned for target projections
      (``("",)`` for root AR stage models, e.g. ``("transformer",)`` for DiTs).
    - ``packed_modules_mapping``: fused projection leaf name to ordered logical
      sublayer names (derived from ``stacked_params_mapping`` or upstream
      ``SupportsLoRA.packed_modules_mapping``).
    - ``name_aliases``: optional engine-module-name to PEFT-adapter-name alias
      function (e.g. Qwen-Image ``to_out`` -> ``to_out.0``, HunyuanImage-3
      ``transformer.*`` -> ``model.*``).
    - ``deinterleave_fused_qkv_lora_b``: optional fused-QKV LoRA-B deinterleave
      hook for checkpoints using GQA-interleaved QKV rows.
    - ``validate_binding``: optional post-bind validation hook.
    - ``weights_mapper``: optional upstream ``WeightsMapper`` applied at tensor
      ingest (e.g. AR stage prefix rewrites).
    - ``unquantized_linear_only``: confirms merged targets are unquantized vLLM
      linear projections.
    - ``allow_custom_loader``: whether custom pipeline loader hooks are
      permitted under merged mode (default ``False``).
    """

    components: tuple[str, ...]
    packed_modules_mapping: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    name_aliases: Callable[[str], Sequence[str]] | None = None
    deinterleave_fused_qkv_lora_b: Callable[[torch.Tensor], torch.Tensor | None] | None = None
    validate_binding: Callable[..., None] | None = None
    weights_mapper: Any | None = None
    unquantized_linear_only: bool = True
    allow_custom_loader: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.components, str) or not isinstance(self.components, Sequence) or not self.components:
            raise MergedLoRACapabilityError("MergedLoRASpec.components must be a non-empty sequence of str")
        components = tuple(self.components)
        for comp in components:
            if not isinstance(comp, str):
                raise MergedLoRACapabilityError(
                    f"MergedLoRASpec.components entries must be str, got {type(comp).__name__}"
                )
        if len(set(components)) != len(components):
            raise MergedLoRACapabilityError(f"MergedLoRASpec.components must be unique, got {components}")
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self,
            "packed_modules_mapping",
            _normalize_packed_mapping(self.packed_modules_mapping),
        )
        for hook_name in ("name_aliases", "deinterleave_fused_qkv_lora_b", "validate_binding"):
            hook = getattr(self, hook_name)
            if hook is not None and not callable(hook):
                raise MergedLoRACapabilityError(
                    f"MergedLoRASpec.{hook_name} must be callable or None, got {type(hook).__name__}"
                )
        if self.unquantized_linear_only is not True:
            raise MergedLoRACapabilityError("Merged LoRA requires unquantized_linear_only=True")


@runtime_checkable
class SupportsMergedLoRA(Protocol):
    """Protocol implemented by diffusion pipelines and AR models supporting merged LoRA."""

    def merged_lora_spec(self) -> MergedLoRASpec:
        """Return an immutable, validated merged-LoRA specification for this model."""
        ...


def derive_packed_modules_mapping_from_stacked(stacked: object) -> dict[str, tuple[str, ...]]:
    """Convert a ``stacked_params_mapping`` sequence into leaf packed mappings."""
    if not isinstance(stacked, (list, tuple)):
        return {}
    derived: dict[str, list[str]] = {}
    for item in stacked:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        packed_suffix, sub_suffix = item[0], item[1]
        if not isinstance(packed_suffix, str) or not packed_suffix:
            continue
        if not isinstance(sub_suffix, str) or not sub_suffix:
            continue
        packed_name = packed_suffix.strip(".").split(".")[-1]
        sub_name = sub_suffix.strip(".").split(".")[-1]
        existing = derived.get(packed_name)
        if existing is None:
            derived[packed_name] = [sub_name]
        elif sub_name not in existing:
            existing.append(sub_name)
    return {k: tuple(v) for k, v in derived.items()}


def derive_packed_modules_mapping(module: nn.Module) -> dict[str, tuple[str, ...]]:
    """Collect packed-projection mappings from ``stacked_params_mapping`` attributes."""
    mapping: dict[str, tuple[str, ...]] = {}
    for sub in module.modules():
        derived = derive_packed_modules_mapping_from_stacked(getattr(sub, "stacked_params_mapping", None))
        for packed_name, sub_names in derived.items():
            if packed_name not in mapping:
                mapping[packed_name] = sub_names
    return mapping


def merged_lora_spec_from_ar_model(model: nn.Module) -> MergedLoRASpec:
    """Build a :class:`MergedLoRASpec` from upstream ``SupportsLoRA`` attributes.

    Replaces bare ``AttributeError`` crashes (e.g. models that do not re-export
    ``packed_modules_mapping`` onto their outer stage class) with a named
    :class:`MergedLoRACapabilityError`.
    """
    raw_mapping = getattr(model, "packed_modules_mapping", None)
    if raw_mapping is None:
        raise MergedLoRACapabilityError(
            f"{type(model).__name__} does not support merged LoRA: missing 'packed_modules_mapping' declaration "
            "(implement SupportsMergedLoRA or expose upstream SupportsLoRA packed_modules_mapping)"
        )
    if not isinstance(raw_mapping, Mapping):
        raise MergedLoRACapabilityError(
            f"{type(model).__name__}.packed_modules_mapping must be a mapping, got {type(raw_mapping).__name__}"
        )
    packed = {
        name.rsplit(".", 1)[-1]: tuple(part.rsplit(".", 1)[-1] for part in parts) for name, parts in raw_mapping.items()
    }
    return MergedLoRASpec(
        components=("",),
        packed_modules_mapping=packed,
        weights_mapper=getattr(model, "hf_to_vllm_mapper", None),
        unquantized_linear_only=True,
    )


def _iter_spec_components(model: nn.Module, spec: MergedLoRASpec) -> list[tuple[str, nn.Module]]:
    resolved: list[tuple[str, nn.Module]] = []
    for comp_name in spec.components:
        if comp_name == "":
            resolved.append(("", model))
            continue
        if not hasattr(model, comp_name):
            raise MergedLoRACapabilityError(
                f"{type(model).__name__} merged_lora_spec declares missing component {comp_name!r}"
            )
        comp = getattr(model, comp_name)
        if not isinstance(comp, nn.Module):
            raise MergedLoRACapabilityError(
                f"{type(model).__name__} merged_lora_spec component {comp_name!r} is "
                f"{type(comp).__name__}, expected torch.nn.Module"
            )
        resolved.append((comp_name, comp))
    return resolved


def validate_merged_lora_model(model: nn.Module, spec: MergedLoRASpec) -> None:
    """Validate runtime-only conflicts and projection structure against ``spec``."""
    model_name = type(model).__name__
    if getattr(model, "diffusion_lora_fixed_adapter", False) is True:
        raise MergedLoRACapabilityError(
            f"{model_name} uses diffusion_lora_fixed_adapter (runtime-only specialization) "
            "and does not support generic merged LoRA"
        )
    if getattr(model, "diffusion_lora_scale_in_epilogue", False) is True:
        raise MergedLoRACapabilityError(
            f"{model_name} uses diffusion_lora_scale_in_epilogue (runtime-only specialization) "
            "and does not support generic merged LoRA"
        )
    if not spec.allow_custom_loader and callable(getattr(model, "_load_diffusion_lora_adapter", None)):
        raise MergedLoRACapabilityError(
            f"{model_name} uses a custom _load_diffusion_lora_adapter hook (runtime-only unless "
            "explicitly declared in MergedLoRASpec)"
        )

    components = _iter_spec_components(model, spec)
    vllm_linear_count = 0
    plain_nn_linear_count = 0
    unmapped_packed: list[str] = []

    for comp_name, comp in components:
        for mod_name, mod in comp.named_modules():
            if isinstance(mod, (LinearBase, BaseLayerWithLoRA)):
                vllm_linear_count += 1
                base_mod = mod.base_layer if isinstance(mod, BaseLayerWithLoRA) else mod
                if isinstance(base_mod, (QKVParallelLinear, MergedColumnParallelLinear)):
                    leaf = mod_name.rsplit(".", 1)[-1] if mod_name else comp_name
                    if leaf not in spec.packed_modules_mapping and spec.deinterleave_fused_qkv_lora_b is None:
                        full = f"{comp_name}.{mod_name}" if comp_name and mod_name else (mod_name or comp_name)
                        unmapped_packed.append(full)
            elif type(mod) is nn.Linear:
                plain_nn_linear_count += 1

    if vllm_linear_count == 0 and plain_nn_linear_count > 0:
        raise MergedLoRACapabilityError(
            f"{model_name} cannot bind merged LoRA: declared components {spec.components} build plain "
            f"torch.nn.Linear projections ({plain_nn_linear_count} found) instead of vLLM LinearBase layers"
        )
    if vllm_linear_count == 0:
        raise MergedLoRACapabilityError(
            f"{model_name} cannot bind merged LoRA: no vLLM LinearBase projections found under "
            f"components {spec.components}"
        )
    if unmapped_packed:
        raise MergedLoRACapabilityError(
            f"{model_name} has packed projections {unmapped_packed[:4]} missing from "
            "packed_modules_mapping/stacked_params_mapping (hoist stacked_params_mapping or declare "
            "all packed projections in MergedLoRASpec.packed_modules_mapping)"
        )


def _legacy_diffusion_components(pipeline: nn.Module) -> tuple[str, ...]:
    declared = tuple(getattr(pipeline, "_dit_modules", ()) or ())
    extra = tuple(getattr(pipeline, "_lora_components", ()) or ())
    present = [
        name
        for name in dict.fromkeys((*DEFAULT_DIFFUSION_LORA_COMPONENTS, *declared, *extra))
        if hasattr(pipeline, name) and isinstance(getattr(pipeline, name), nn.Module)
    ]
    return tuple(present)


def lookup_merged_lora_registry_entry(
    model_name: str,
    *,
    stage: Literal["diffusion", "ar"] = "diffusion",
) -> MergedLoRARegistryEntry | None:
    """Look up a registered diffusion pipeline or AR stage class/arch name."""
    if stage == "diffusion":
        if model_name in DIFFUSION_MERGED_LORA_REGISTRY:
            return DIFFUSION_MERGED_LORA_REGISTRY[model_name]
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

        for arch, (_, _, cls_name) in _DIFFUSION_MODELS.items():
            if cls_name == model_name and arch in DIFFUSION_MERGED_LORA_REGISTRY:
                return DIFFUSION_MERGED_LORA_REGISTRY[arch]
        return None

    if model_name in AR_MERGED_LORA_REGISTRY:
        return AR_MERGED_LORA_REGISTRY[model_name]
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    for arch, (_, _, cls_name) in _OMNI_MODELS.items():
        if cls_name == model_name and arch in AR_MERGED_LORA_REGISTRY:
            return AR_MERGED_LORA_REGISTRY[arch]
    return None


def resolve_merged_lora_spec(
    model: nn.Module,
    *,
    stage: Literal["diffusion", "ar"] = "diffusion",
    require_protocol: bool = False,
) -> MergedLoRASpec:
    """Resolve and validate the merged-LoRA capability spec for ``model``.

    Args:
        model: Diffusion pipeline or AR stage model instance.
        stage: ``"diffusion"`` or ``"ar"``.
        require_protocol: If True on diffusion models, require an explicit
            :class:`SupportsMergedLoRA` implementation rather than synthesizing
            a legacy spec for synthetic ``nn.Module`` fixtures.

    Raises:
        MergedLoRACapabilityError: The model lacks a required declaration or
            violates merged-LoRA structural invariants.
    """
    model_name = type(model).__name__
    entry = lookup_merged_lora_registry_entry(model_name, stage=stage)
    if entry is not None and entry.status is not MergedLoRACapabilityStatus.SUPPORTED:
        raise MergedLoRACapabilityError(
            f"{model_name} is registered as {entry.status.value} for merged LoRA: {entry.reason}"
        )

    if isinstance(model, SupportsMergedLoRA):
        spec = model.merged_lora_spec()
        if not isinstance(spec, MergedLoRASpec):
            raise MergedLoRACapabilityError(
                f"{model_name}.merged_lora_spec() must return MergedLoRASpec, got {type(spec).__name__}"
            )
        validate_merged_lora_model(model, spec)
        return spec

    if stage == "ar":
        spec = merged_lora_spec_from_ar_model(model)
        validate_merged_lora_model(model, spec)
        return spec

    if require_protocol:
        raise MergedLoRACapabilityError(
            f"{model_name} does not implement SupportsMergedLoRA (missing merged_lora_spec() declaration)"
        )

    components = _legacy_diffusion_components(model)
    if not components:
        raise MergedLoRACapabilityError(
            f"{model_name} declares no LoRA components (expected SupportsMergedLoRA or "
            "_dit_modules/_lora_components/transformer/unet)"
        )
    spec = MergedLoRASpec(
        components=components,
        packed_modules_mapping=derive_packed_modules_mapping(model),
        name_aliases=getattr(model, "_get_diffusion_lora_name_aliases", None),
        deinterleave_fused_qkv_lora_b=getattr(model, "_deinterleave_fused_qkv_lora_b", None),
        validate_binding=getattr(model, "_validate_diffusion_lora_binding", None),
        weights_mapper=getattr(model, "hf_to_vllm_mapper", None),
        unquantized_linear_only=True,
    )
    validate_merged_lora_model(model, spec)
    return spec


# ---------------------------------------------------------------------------
# Explicit capability registries for CI coverage (operationalizes DIFF-MODEL-INV-001)
# ---------------------------------------------------------------------------

_SUPPORTED = MergedLoRARegistryEntry(MergedLoRACapabilityStatus.SUPPORTED)


def _runtime_only(reason: str) -> MergedLoRARegistryEntry:
    return MergedLoRARegistryEntry(MergedLoRACapabilityStatus.RUNTIME_ONLY, reason)


def _not_capable(reason: str) -> MergedLoRARegistryEntry:
    return MergedLoRARegistryEntry(MergedLoRACapabilityStatus.NOT_CAPABLE, reason)


_PENDING_DIFFUSION_MIGRATION = "pending supports_weight_reload audit and SupportsMergedLoRA migration (RFC #10444 §4)"
_PENDING_AR_MAPPING = "missing packed_modules_mapping / SupportsMergedLoRA declaration (RFC #10444 §4)"
_RUNTIME_SUPPORTS_LORA = (
    "implements upstream SupportsLoRA for runtime adapters; "
    "pending merged-LoRA validation and SupportsMergedLoRA migration"
)
_NON_LLM_STAGE = "non-LLM audio/codec/encoder stage without merged-LoRA linear adapter surface"

DIFFUSION_MERGED_LORA_REGISTRY: dict[str, MergedLoRARegistryEntry] = {
    # Migrated to SupportsMergedLoRA
    "QwenImagePipeline": _SUPPORTED,
    "QwenImageDMD2Pipeline": _SUPPORTED,
    "HunyuanImage3ForCausalMM": _SUPPORTED,
    # Runtime-only LoRA specializations
    "TaoMateH3Pipeline": _runtime_only(
        "uses diffusion_lora_fixed_adapter, diffusion_lora_scale_in_epilogue, and custom loader"
    ),
    "Flux3Pipeline": _runtime_only("uses custom _load_diffusion_lora_adapter and binding validator"),
    "MiniMaxH3Pipeline": _runtime_only("uses custom _load_diffusion_lora_adapter and MoE routing"),
    "MiniMaxH3ModularPipeline": _runtime_only("uses custom _load_diffusion_lora_adapter and MoE routing"),
    "BagelPipeline": _runtime_only("uses layer-level _create_omni_lora_layer factory; pending spec migration"),
    # Structural / declaration defects named in RFC #10444 §2.3 & §4
    "MammothModa2DiTPipeline": _not_capable(
        "declares stacked_params_mapping on plain torch.nn.Linear projections instead of vLLM LinearBase"
    ),
    "NextStep11Pipeline": _not_capable(
        "keeps stacked_params_mapping as a load_weights local; hoist to a class attribute before migrating"
    ),
    "BooguImagePipeline": _not_capable(
        "pending supports_weight_reload, SupportsMergedLoRA, and engine-side name aliases"
    ),
    "BooguImageTurboPipeline": _not_capable(
        "pending supports_weight_reload, SupportsMergedLoRA, and engine-side name aliases"
    ),
    "StableDiffusion3Pipeline": _not_capable(
        "pending supports_weight_reload loader audit (CLIP/T5/VAE/pos_embed) and SupportsMergedLoRA migration"
    ),
    "QwenImageEditPipeline": _not_capable(
        "shares Qwen-Image transformer; pending supports_weight_reload validation and SupportsMergedLoRA migration"
    ),
    "QwenImageEditPlusPipeline": _not_capable(
        "shares Qwen-Image transformer; pending supports_weight_reload validation and SupportsMergedLoRA migration"
    ),
    "QwenImageLayeredPipeline": _not_capable(
        "shares Qwen-Image transformer; pending supports_weight_reload validation and SupportsMergedLoRA migration"
    ),
    "QwenImage21Pipeline": _not_capable("pending supports_weight_reload validation and SupportsMergedLoRA migration"),
    # Remaining registered diffusion pipelines (explicitly tracked)
    "ABotWorldCausalPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "Alpamayo2SuperPipeline": _not_capable("action-expert Euler pipeline; no merged-LoRA contract"),
    "AnimaPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "AuKPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "CausVidWanPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "Cosmos3MultiviewPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "Cosmos3NanoSimBimanualPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "Cosmos3OmniDiffusersPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "Cosmos3OmniPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "DiffusersAdapterPipeline": _not_capable("generic diffusers fallback wrapper; no native vLLM linear contract"),
    "DreamZeroPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "ErnieImagePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "Flux2KleinPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "Flux2Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "FluxDMD2Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "FluxKontextPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "FluxPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "GlmImagePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "Gr00tN1d7Pipeline": _not_capable("VLA action pipeline; no merged-LoRA contract"),
    "HeliosPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "HeliosPyramidPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "HiDreamO1ImagePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "HunyuanVideo15ImageToVideoPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "HunyuanVideo15Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "HyNomasModularPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "HyNomasPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "InternVLAA1Pipeline": _not_capable("VLA action pipeline; no merged-LoRA contract"),
    "Kandinsky6TI2VAPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "Krea2Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LTX2DistilledOneStagePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LTX2DistilledPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LTX2DistilledTwoStagePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LTX2I2VDMD2Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LTX2Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LTX2T2VDMD2Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LTX2TextToAudioPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LTX2TwoStagePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LancePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LingBotVideoPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LingBotWorldCausalDMDPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LongCatImageEditPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LongCatImagePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "LongCatVideoAvatarPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "MOSSImagePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "Magi2Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "MingImageDiffusionPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "MingImageLayeredDiffusionPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "MingImagePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "MiniMaxH3DecoderPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "MiniMaxH3ImagePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "OmniGen2Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "OmniVoice": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "OmniVoicePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "OvisImagePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "PAN2ModularPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "PhysisH3Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "PhysisLTXPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "PhysisWanPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "Pi05Pipeline": _not_capable("VLA flow-matching pipeline; no merged-LoRA contract"),
    "Pi0Pipeline": _not_capable("VLA flow-matching pipeline; no merged-LoRA contract"),
    "Praxis1Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "Sam3Model": _not_capable("segmentation detector pipeline; no merged-LoRA contract"),
    "Sam3VideoModel": _not_capable("segmentation detector pipeline; no merged-LoRA contract"),
    "SanaImageToVideoPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "SanaVideoPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "SanaWmPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "SeedVR2Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "SenseNovaU1Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "StableAudio3Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "StableAudioPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "StableDiffusionXLPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "VideoDepthAnythingPipeline": _not_capable("depth estimator pipeline; no merged-LoRA contract"),
    "WanDMDPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "WanI2VDMD2Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "WanImageToVideoPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "WanPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "WanS2VPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "WanT2VDMD2Pipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "WanVACEPipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
    "ZImagePipeline": _not_capable(_PENDING_DIFFUSION_MIGRATION),
}

AR_MERGED_LORA_REGISTRY: dict[str, MergedLoRARegistryEntry] = {
    # Migrated to SupportsMergedLoRA
    "Qwen3OmniMoeForConditionalGeneration": _SUPPORTED,
    "Qwen3OmniMoeThinkerForConditionalGeneration": _SUPPORTED,
    # Upstream SupportsLoRA models (pending merged-LoRA validation / migration)
    "HunyuanImage3ForCausalMM": _runtime_only(_RUNTIME_SUPPORTS_LORA),
    "NemotronDenseForCausalLM": _runtime_only(_RUNTIME_SUPPORTS_LORA),
    "Qwen2ForCausalLM_old": _runtime_only(_RUNTIME_SUPPORTS_LORA),
    "Qwen2_5OmniThinkerModel": _runtime_only(_RUNTIME_SUPPORTS_LORA),
    "ResponseJudgeQwen3ForCausalLM": _runtime_only(_RUNTIME_SUPPORTS_LORA),
    # MiniCPM-o 4.5 family (named in RFC #10444 §1.2 & §4)
    "MiniCPMO45OmniForConditionalGeneration": _not_capable(
        "missing packed_modules_mapping re-export from inner Qwen stack onto outer stage class"
    ),
    "MiniCPMO45OmniLLMForConditionalGeneration": _not_capable(
        "missing packed_modules_mapping re-export from inner Qwen stack"
    ),
    "MiniCPMO45OmniTTSForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "MiniCPMO45Code2Wav": _not_capable(_NON_LLM_STAGE),
    # Qwen3-Omni / Qwen2.5-Omni non-thinker stages
    "Qwen3OmniMoeTalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "Qwen3OmniMoeCode2Wav": _not_capable(_NON_LLM_STAGE),
    "Qwen2_5OmniForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "Qwen2_5OmniTalkerModel": _not_capable(_PENDING_AR_MAPPING),
    "Qwen2_5OmniToken2WavDiTModel": _not_capable(_NON_LLM_STAGE),
    "Qwen2_5OmniToken2WavModel": _not_capable(_NON_LLM_STAGE),
    # Remaining registered omni AR stage models
    "Alpamayo2SuperPolicy": _not_capable(_PENDING_AR_MAPPING),
    "AuKForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "AudexCode2Wav": _not_capable(_NON_LLM_STAGE),
    "AudexXCodec1": _not_capable(_NON_LLM_STAGE),
    "Audio8TTSCodecDecoder": _not_capable(_NON_LLM_STAGE),
    "Audio8TTSSlowARForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "AuraQwen3VLForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "BailingMM2NativeForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "BailingMoeV3AudioForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "BailingTalker2": _not_capable(_PENDING_AR_MAPPING),
    "BreezeCode2Wav": _not_capable(_NON_LLM_STAGE),
    "BreezeForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "ChatterboxForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "CosyVoice3Model": _not_capable(_PENDING_AR_MAPPING),
    "CovoAudioCode2WavModel": _not_capable(_NON_LLM_STAGE),
    "CovoAudioForCausalLM": _not_capable(_PENDING_AR_MAPPING),
    "CovoAudioForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "CovoAudioLLMModel": _not_capable(_PENDING_AR_MAPPING),
    "CovoAudioModel": _not_capable(_PENDING_AR_MAPPING),
    "FishOmniForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "FishSpeechDACDecoder": _not_capable(_NON_LLM_STAGE),
    "FishSpeechSlowARForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "GLMTTSForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "GepardTalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "GlmImageForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "HaruForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "HiggsAudioV2Code2WavForConditionalGeneration": _not_capable(_NON_LLM_STAGE),
    "HiggsAudioV2ForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "HiggsAudioV2TalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "HiggsAudioV3Code2WavForConditionalGeneration": _not_capable(_NON_LLM_STAGE),
    "HiggsAudioV3TalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "HiggsMultimodalQwen3ForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "IndexTTS25S2MelDecoder": _not_capable(_NON_LLM_STAGE),
    "IndexTTS25TalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "IndexTTS2S2MelDecoder": _not_capable(_NON_LLM_STAGE),
    "IndexTTS2TalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "LayaDecisionModel": _not_capable(_NON_LLM_STAGE),
    "Lfm3AudioForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "Lfm3OmniCode2Wav": _not_capable(_NON_LLM_STAGE),
    "Lfm3OmniForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "LingV3TTSForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "MammothModa2ARForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "MammothModa2ForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "MammothModa2Qwen2ForCausalLM": _not_capable(_PENDING_AR_MAPPING),
    "MammothModa2Qwen3ARForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "MammothModa2Qwen3ForCausalLM": _not_capable(_PENDING_AR_MAPPING),
    "Mammothmoda2Model": _not_capable(_PENDING_AR_MAPPING),
    "MiMoAudioLLMModel": _not_capable(_PENDING_AR_MAPPING),
    "MiMoAudioModel": _not_capable(_PENDING_AR_MAPPING),
    "MiMoAudioToken2WavModel": _not_capable(_NON_LLM_STAGE),
    "MiMoV2ASRForCausalLM": _not_capable(_PENDING_AR_MAPPING),
    "Ming3TTSAudioVAEModel": _not_capable(_NON_LLM_STAGE),
    "Ming3TTSForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "Ming3TTSLLMModel": _not_capable(_PENDING_AR_MAPPING),
    "MingAudioVAEModel": _not_capable(_NON_LLM_STAGE),
    "MingFlashOmniForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "MingFlashOmniTalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "MingFlashOmniThinkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "MingImageForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "MingLLMModel": _not_capable(_PENDING_AR_MAPPING),
    "MingTTSForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "MiniMaxH3Encoder": _not_capable(_NON_LLM_STAGE),
    "MiniMaxMusic3AcousticForConditionalGeneration": _not_capable(_NON_LLM_STAGE),
    "MiniMaxMusic3TalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "MossTTSCodecDecoder": _not_capable(_NON_LLM_STAGE),
    "MossTTSDelayModel": _not_capable(_PENDING_AR_MAPPING),
    "MossTTSLocalModel": _not_capable(_PENDING_AR_MAPPING),
    "MossTTSNanoForCausalLM": _not_capable(_PENDING_AR_MAPPING),
    "MossTTSRealtime": _not_capable(_PENDING_AR_MAPPING),
    "NemotronDenseAudexForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "NemotronHAudexForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "NemotronVoiceChatCode2Wav": _not_capable(_NON_LLM_STAGE),
    "NemotronVoiceChatTalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "NemotronVoiceChatThinkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "OmniBagelForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "OmniVoiceModel": _not_capable(_PENDING_AR_MAPPING),
    "PersonaPlexCode2Wav": _not_capable(_NON_LLM_STAGE),
    "PersonaPlexTalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "PhononTalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "Qwen3TTSCode2Wav": _not_capable(_NON_LLM_STAGE),
    "Qwen3TTSForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "Qwen3TTSTalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "StepAudio2ForCausalLM": _not_capable(_PENDING_AR_MAPPING),
    "StepAudio2ForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "StepAudio2ThinkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "StepAudio2Token2WavForConditionalGeneration": _not_capable(_NON_LLM_STAGE),
    "StepAudio2Token2WavModel": _not_capable(_NON_LLM_STAGE),
    "VibeVoiceForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "VoxCPM2TalkerForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "VoxtralTTSAudioGeneration": _not_capable(_PENDING_AR_MAPPING),
    "VoxtralTTSAudioTokenizer": _not_capable(_NON_LLM_STAGE),
    "VoxtralTTSForConditionalGeneration": _not_capable(_PENDING_AR_MAPPING),
    "Yue2ForCausalLM": _not_capable(_PENDING_AR_MAPPING),
}


def audit_merged_lora_registries(
    *,
    diffusion_models: Mapping[str, tuple[str, str, str]] | None = None,
    ar_models: Mapping[str, tuple[str, str, str]] | None = None,
    diffusion_registry: Mapping[str, MergedLoRARegistryEntry] | None = None,
    ar_registry: Mapping[str, MergedLoRARegistryEntry] | None = None,
    verify_supported_classes: bool = True,
) -> list[str]:
    """Audit registered diffusion pipelines and omni AR stage models.

    Every registered model must either implement :class:`SupportsMergedLoRA`
    (status ``SUPPORTED``) or carry an explicit ``RUNTIME_ONLY`` /
    ``NOT_CAPABLE`` entry with a non-empty reason. Returns a list of human-
    readable violation messages (empty when all declarations are valid).
    """
    from vllm_omni.diffusion.registry import _DIFFUSION_MODELS, DiffusionModelRegistry
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS, OmniModelRegistry

    diff_models = _DIFFUSION_MODELS if diffusion_models is None else diffusion_models
    omni_models = _OMNI_MODELS if ar_models is None else ar_models
    diff_reg = DIFFUSION_MERGED_LORA_REGISTRY if diffusion_registry is None else diffusion_registry
    ar_reg = AR_MERGED_LORA_REGISTRY if ar_registry is None else ar_registry

    violations: list[str] = []

    for label, reg_name, models, reg, loader in (
        ("diffusion", "DIFFUSION_MERGED_LORA_REGISTRY", diff_models, diff_reg, DiffusionModelRegistry),
        ("ar", "AR_MERGED_LORA_REGISTRY", omni_models, ar_reg, OmniModelRegistry),
    ):
        missing = sorted(set(models) - set(reg))
        for arch in missing:
            violations.append(
                f"{label} model {arch!r} is missing a merged-LoRA capability declaration "
                f"(add {arch!r} to {reg_name} in vllm_omni/lora/capability.py as "
                "_SUPPORTED after implementing SupportsMergedLoRA, or _runtime_only(...) / _not_capable(...) "
                "with a reason)"
            )
        stale = sorted(set(reg) - set(models))
        for arch in stale:
            violations.append(
                f"{label} merged-LoRA registry ({reg_name} in vllm_omni/lora/capability.py) "
                f"contains unregistered arch {arch!r}"
            )

        for arch in sorted(set(models) & set(reg)):
            entry = reg[arch]
            if entry.status is not MergedLoRACapabilityStatus.SUPPORTED:
                if not entry.reason.strip():
                    violations.append(f"{label} model {arch!r} ({entry.status.value}) has an empty reason")
                continue
            if verify_supported_classes:
                model_cls = loader._try_load_model_cls(arch)
                if model_cls is None:
                    violations.append(f"{label} model {arch!r} marked SUPPORTED could not be imported")
                elif not issubclass(model_cls, SupportsMergedLoRA):
                    violations.append(
                        f"{label} model {arch!r} ({model_cls.__name__}) is marked SUPPORTED "
                        "but does not implement SupportsMergedLoRA"
                    )

    return violations
