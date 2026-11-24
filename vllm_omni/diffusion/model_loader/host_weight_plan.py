# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Loader-owned host-weight plans shared by direct-checkpoint consumers."""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    get_direct_mmap_adapter,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_omni.host_weight_runtime import HostWeightLeaseCarrier

TensorTransform = Callable[[torch.Tensor], torch.Tensor]
FuseTransform = Callable[[Sequence[torch.Tensor]], torch.Tensor]
SourceRemap = Callable[[str], str | None]


@dataclass(frozen=True)
class TensorBinding:
    """One runtime tensor backed by one safetensors entry."""

    checkpoint_key: str
    file_path: str
    transform: TensorTransform | None = None


@dataclass(frozen=True)
class FusedTensorBinding:
    """One runtime tensor fused from several checkpoint slices during staging.

    ``checkpoint_keys`` and ``file_paths`` are aligned and ordered the way
    ``fuse`` consumes them. A fused runtime tensor has no single checkpoint
    entry, so its slices are read from disk and combined while packing a
    bounded staging slot instead of being realized as one mmap view.
    """

    checkpoint_keys: tuple[str, ...]
    file_paths: tuple[str, ...]
    fuse: FuseTransform | None = None


@dataclass(frozen=True)
class HostWeightPlan:
    """Complete, prevalidated host backing consumed by an offload backend."""

    backing_kind: str
    bindings: dict[str, TensorBinding | FusedTensorBinding]
    planned_source_prefixes: frozenset[str] = frozenset()
    lease_carrier: HostWeightLeaseCarrier | None = None


@dataclass(frozen=True)
class HostWeightPlanResult:
    """A complete plan or a fail-closed reason to use the ordinary loader."""

    plan: HostWeightPlan | None
    fallback_reason: str | None = None


class _PlanIncompatibleError(RuntimeError):
    pass


_SAFETENSORS_DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
for _safe_name, _torch_name in (
    ("F8_E4M3", "float8_e4m3fn"),
    ("F8_E5M2", "float8_e5m2"),
):
    if (_dtype := getattr(torch, _torch_name, None)) is not None:
        _SAFETENSORS_DTYPES[_safe_name] = _dtype


def has_online_quantization(model: nn.Module) -> bool:
    """Whether any module defers online-quantized weights on ``meta``."""
    return any(getattr(getattr(module, "quant_method", None), "uses_meta_device", False) for module in model.modules())


def _resolve_source_root(source: object) -> str:
    source_root = os.fspath(getattr(source, "model_or_path"))
    subfolder = getattr(source, "subfolder", None)
    if not os.path.isdir(source_root):
        from vllm.model_executor.model_loader.weight_utils import (
            download_weights_from_hf,
        )

        source_root = download_weights_from_hf(
            model_name_or_path=source_root,
            cache_dir=None,
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
            revision=getattr(source, "revision", None),
            subfolder=subfolder,
        )
    return os.path.join(source_root, subfolder) if subfolder is not None else source_root


def _record_binding(
    model_to_ckpt: dict[str, list[tuple[str, str]]],
    model_name: str,
    checkpoint_key: str,
    file_path: str,
) -> None:
    """Accumulate every checkpoint entry that maps to one runtime tensor.

    Multi-entry runtime names are legitimate for a component that fuses several
    checkpoint slices at load time; the caller decides from the tensor's policy
    whether that is a fused binding or an incompatible layout.
    """
    entries = model_to_ckpt.setdefault(model_name, [])
    candidate = (checkpoint_key, file_path)
    if candidate in entries:
        return
    entries.append(candidate)


def _add_safetensors_source(
    checkpoint_dir: str,
    prefix: str,
    remap_fn: Callable[[str], str | None] | None,
    model_to_ckpt: dict[str, list[tuple[str, str]]],
) -> int:
    indexed_keys = 0
    index_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors.index.json")))
    if index_files:
        for index_file in index_files:
            with open(index_file, encoding="utf-8") as handle:
                index = json.load(handle)
            for checkpoint_key, filename in index.get("weight_map", {}).items():
                source_name = prefix + checkpoint_key
                model_name = remap_fn(source_name) if remap_fn is not None else source_name
                if model_name is not None:
                    _record_binding(
                        model_to_ckpt,
                        model_name,
                        checkpoint_key,
                        os.path.join(checkpoint_dir, filename),
                    )
                indexed_keys += 1
        return indexed_keys

    for filename in sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))):
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for checkpoint_key in handle.keys():
                source_name = prefix + checkpoint_key
                model_name = remap_fn(source_name) if remap_fn is not None else source_name
                if model_name is not None:
                    _record_binding(model_to_ckpt, model_name, checkpoint_key, filename)
                indexed_keys += 1
    return indexed_keys


def _remap_for_prefix(
    prefix: str,
    remap_fn: Callable[[str], str | None] | None,
    component_remaps: dict[str, SourceRemap] | None,
) -> Callable[[str], str | None] | None:
    """Select the remapper that owns one component's weight source."""
    if component_remaps and prefix.endswith(".") and prefix[:-1] in component_remaps:
        return component_remaps[prefix[:-1]]
    return remap_fn


def _build_source_map(
    *,
    sources: Sequence[object],
    model_path: str | None,
    remap_fn: Callable[[str], str | None] | None,
    component_remaps: dict[str, SourceRemap] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    model_to_ckpt: dict[str, list[tuple[str, str]]] = {}
    indexed_keys = 0

    for source in sources:
        prefix = getattr(source, "prefix", "")
        indexed_keys += _add_safetensors_source(
            _resolve_source_root(source),
            prefix,
            _remap_for_prefix(prefix, remap_fn, component_remaps),
            model_to_ckpt,
        )

    if not sources:
        if remap_fn is None:
            raise _PlanIncompatibleError(
                "the loader has no component weight sources and the pipeline has no checkpoint-key remapper"
            )
        if not model_path:
            raise _PlanIncompatibleError("the loader has no model path for legacy checkpoint remapping")

        resolved_model_path = model_path
        if not os.path.isdir(resolved_model_path):
            from vllm.model_executor.model_loader.weight_utils import (
                download_weights_from_hf,
            )

            resolved_model_path = download_weights_from_hf(
                model_name_or_path=resolved_model_path,
                cache_dir=None,
                allow_patterns=["*.safetensors", "*.safetensors.index.json"],
            )
        checkpoint_dirs = {
            os.path.dirname(path)
            for pattern in ("*.safetensors.index.json", "*.safetensors")
            for path in glob.glob(os.path.join(resolved_model_path, "**", pattern), recursive=True)
        }
        for checkpoint_dir in sorted(checkpoint_dirs):
            indexed_keys += _add_safetensors_source(
                checkpoint_dir,
                "",
                _remap_for_prefix("", remap_fn, component_remaps),
                model_to_ckpt,
            )

    if not model_to_ckpt:
        raise _PlanIncompatibleError("no compatible safetensors entries were found in the loader's model sources")
    logger.info(
        "Indexed %d runtime tensor names from %d checkpoint keys for checkpoint preflight",
        len(model_to_ckpt),
        indexed_keys,
    )
    return model_to_ckpt


def _is_persistent_buffer(module: nn.Module, local_name: str) -> bool:
    parts = local_name.split(".")
    owner = module.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else module
    return parts[-1] not in owner._non_persistent_buffers_set


def _collect_required_component_tensors(
    component_modules: Sequence[tuple[str, nn.Module]],
) -> dict[str, torch.Tensor]:
    required: dict[str, torch.Tensor] = {}
    for component_name, component_module in component_modules:
        for local_name, tensor in component_module.named_parameters():
            required[f"{component_name}.{local_name}"] = tensor
        for local_name, tensor in component_module.named_buffers():
            if _is_persistent_buffer(component_module, local_name):
                required[f"{component_name}.{local_name}"] = tensor
    return required


def _planned_source_prefixes(
    sources: Sequence[object],
    component_modules: Sequence[tuple[str, nn.Module]],
) -> frozenset[str]:
    """Identify component sources that the checkpoint-mmap plan owns completely."""
    if not sources:
        return frozenset()

    expected = frozenset(f"{component_name}." for component_name, _ in component_modules)
    available = {getattr(source, "prefix", "") for source in sources}
    missing = sorted(expected - available)
    if missing:
        raise _PlanIncompatibleError(
            f"direct mmap requires a dedicated component weight source for each component; missing prefixes: {missing}"
        )
    return expected


def _validate_source_metadata(
    required: dict[str, torch.Tensor],
    bindings: dict[str, TensorBinding | FusedTensorBinding],
) -> None:
    fused: dict[str, tuple[torch.Tensor, FusedTensorBinding]] = {}
    by_file: dict[str, list[tuple[str, torch.Tensor, TensorBinding]]] = {}
    for runtime_name, target in required.items():
        binding = bindings[runtime_name]
        if isinstance(binding, FusedTensorBinding):
            fused[runtime_name] = (target, binding)
            continue
        by_file.setdefault(binding.file_path, []).append((runtime_name, target, binding))

    for file_path, entries in by_file.items():
        if not os.path.isfile(file_path):
            raise _PlanIncompatibleError(f"checkpoint file does not exist: {file_path}")
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for runtime_name, target, binding in entries:
                if binding.checkpoint_key not in available:
                    raise _PlanIncompatibleError(
                        f"checkpoint key {binding.checkpoint_key!r} for {runtime_name!r} is missing from {file_path}"
                    )
                source = handle.get_slice(binding.checkpoint_key)
                source_shape = tuple(source.get_shape())
                source_dtype = _SAFETENSORS_DTYPES.get(source.get_dtype())
                if source_dtype is None:
                    raise _PlanIncompatibleError(
                        f"dtype mismatch for {runtime_name!r}: checkpoint={source.get_dtype()}, runtime={target.dtype}"
                    )

                runtime_shape = source_shape
                runtime_dtype = source_dtype
                if binding.transform is not None:
                    # Adapter transforms can map full checkpoint tensors to a
                    # rank-local runtime shape. Validate that contract without
                    # materializing the source tensor during preflight.
                    try:
                        transformed = binding.transform(
                            torch.empty(
                                source_shape,
                                dtype=source_dtype,
                                device="meta",
                            )
                        )
                    except (NotImplementedError, RuntimeError, TypeError, ValueError) as exc:
                        raise _PlanIncompatibleError(
                            f"checkpoint transform for {runtime_name!r} cannot be validated on tensor metadata: {exc}"
                        ) from exc
                    if not isinstance(transformed, torch.Tensor):
                        raise _PlanIncompatibleError(
                            f"checkpoint transform for {runtime_name!r} returned {type(transformed).__name__}, "
                            "expected torch.Tensor"
                        )
                    runtime_shape = tuple(transformed.shape)
                    runtime_dtype = transformed.dtype

                if runtime_shape != tuple(target.shape):
                    raise _PlanIncompatibleError(
                        f"shape mismatch for {runtime_name!r}: checkpoint={source_shape}, "
                        f"transformed={runtime_shape}, runtime={tuple(target.shape)}"
                    )
                if runtime_dtype != target.dtype:
                    raise _PlanIncompatibleError(
                        f"dtype mismatch for {runtime_name!r}: checkpoint={source.get_dtype()}, "
                        f"transformed={runtime_dtype}, runtime={target.dtype}"
                    )

    for runtime_name, (target, binding) in fused.items():
        slices: list[torch.Tensor] = []
        for checkpoint_key, file_path in zip(binding.checkpoint_keys, binding.file_paths, strict=True):
            if not os.path.isfile(file_path):
                raise _PlanIncompatibleError(f"checkpoint file does not exist: {file_path}")
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                if checkpoint_key not in set(handle.keys()):
                    raise _PlanIncompatibleError(
                        f"checkpoint key {checkpoint_key!r} for fused tensor "
                        f"{runtime_name!r} is missing from {file_path}"
                    )
                source = handle.get_slice(checkpoint_key)
                source_dtype = _SAFETENSORS_DTYPES.get(source.get_dtype())
                if source_dtype is None:
                    raise _PlanIncompatibleError(
                        f"dtype mismatch for fused tensor {runtime_name!r}: "
                        f"checkpoint={source.get_dtype()}, runtime={target.dtype}"
                    )
                slices.append(torch.empty(tuple(source.get_shape()), dtype=source_dtype, device="meta"))
        if binding.fuse is None:
            raise _PlanIncompatibleError(
                f"fused tensor {runtime_name!r} has no fusion callable; a multi-entry binding cannot be staged"
            )
        try:
            fused_meta = binding.fuse(slices)
        except (NotImplementedError, RuntimeError, TypeError, ValueError) as exc:
            raise _PlanIncompatibleError(
                f"checkpoint fusion for {runtime_name!r} cannot be validated on tensor metadata: {exc}"
            ) from exc
        if not isinstance(fused_meta, torch.Tensor):
            raise _PlanIncompatibleError(
                f"checkpoint fusion for {runtime_name!r} returned {type(fused_meta).__name__}, expected torch.Tensor"
            )
        if tuple(fused_meta.shape) != tuple(target.shape):
            raise _PlanIncompatibleError(
                f"shape mismatch for fused tensor {runtime_name!r}: fused={tuple(fused_meta.shape)}, "
                f"runtime={tuple(target.shape)}"
            )
        if fused_meta.dtype != target.dtype:
            raise _PlanIncompatibleError(
                f"dtype mismatch for fused tensor {runtime_name!r}: fused={fused_meta.dtype}, runtime={target.dtype}"
            )


def build_checkpoint_binding_plan(
    pipeline: nn.Module,
    *,
    dit_modules: Sequence[tuple[str, nn.Module]],
    sources: Sequence[object],
    model_path: str | None,
    tensor_parallel_size: int,
    online_quantization: bool,
    use_pipeline_key_remap: bool = True,
    has_distilled_lora: bool = False,
    encoder_modules: Sequence[tuple[str, nn.Module]] = (),
) -> HostWeightPlanResult:
    """Build a complete binding plan for the pre-sharded HSDP consumer."""
    if tensor_parallel_size != 1:
        return HostWeightPlanResult(None, f"TP={tensor_parallel_size} requires the ordinary loader")
    if online_quantization:
        return HostWeightPlanResult(None, "online quantization requires the ordinary loader")
    if has_distilled_lora:
        return HostWeightPlanResult(None, "distilled LoRA requires the ordinary loader")

    remap_fn = getattr(type(pipeline), "_remap_ckpt_key", None)
    if not callable(remap_fn) and use_pipeline_key_remap:
        remap_fn = getattr(pipeline, "remap_checkpoint_key", None)
    if not callable(remap_fn):
        remap_fn = None
    adapter = get_direct_mmap_adapter(pipeline)
    component_modules = tuple(dit_modules) + tuple(encoder_modules)

    try:
        component_remaps = _component_remaps(encoder_modules)
        planned_source_prefixes = _planned_source_prefixes(sources, component_modules)
        planned_sources = tuple(
            source for source in sources if getattr(source, "prefix", "") in planned_source_prefixes
        )
        model_to_ckpt = _build_source_map(
            sources=planned_sources,
            model_path=model_path,
            remap_fn=remap_fn,
            component_remaps=component_remaps,
        )
        required = _collect_required_component_tensors(component_modules)
        if not required:
            raise _PlanIncompatibleError("no component parameters or persistent buffers were discovered")

        missing = sorted(set(required) - set(model_to_ckpt))
        if missing:
            raise _PlanIncompatibleError(
                f"{len(missing)} required component tensors have no checkpoint binding (first 5: {missing[:5]})"
            )

        bindings: dict[str, TensorBinding | FusedTensorBinding] = {}
        for runtime_name, target in required.items():
            policy = adapter.policy_for(runtime_name, target) if adapter is not None else None
            has_custom_loader = callable(getattr(target, "weight_loader", None))
            if has_custom_loader and not (policy and policy.allow_custom_loader):
                raise _PlanIncompatibleError(
                    f"{runtime_name!r} requires a custom weight_loader with no direct-mmap adapter"
                )
            entries = model_to_ckpt[runtime_name]
            if len(entries) == 1:
                checkpoint_key, file_path = entries[0]
                bindings[runtime_name] = TensorBinding(
                    checkpoint_key=checkpoint_key,
                    file_path=file_path,
                    transform=policy.transform if policy is not None else None,
                )
                continue
            fused_sources = policy.fused_sources if policy is not None else None
            if fused_sources is None:
                raise _PlanIncompatibleError(
                    f"multiple checkpoint entries map to runtime tensor {runtime_name!r}: "
                    f"{entries[0]!r} and {entries[1]!r}"
                )
            ordered = _order_fused_entries(runtime_name, entries, fused_sources)
            bindings[runtime_name] = FusedTensorBinding(
                checkpoint_keys=tuple(key for key, _ in ordered),
                file_paths=tuple(path for _, path in ordered),
                fuse=policy.fuse,
            )

        _validate_source_metadata(required, bindings)
    except (OSError, ValueError, _PlanIncompatibleError) as exc:
        if not (encoder_modules and dit_modules):
            return HostWeightPlanResult(None, str(exc))
        # An encoder is planned only through its own source and name mapping,
        # so a failure that disappears without it is encoder-specific, for
        # example a TP-sharded runtime shape. The DiT must not fall back to the
        # ordinary loader and materialize in full for that; plan it alone. The
        # encoder then keeps its ordinary weight source like any unplanned one.
        logger.info(
            "Text encoder(s) %s cannot join the checkpoint-mmap plan (%s); planning the DiT alone",
            [component_path for component_path, _module in encoder_modules],
            exc,
        )
        dit_only = build_checkpoint_binding_plan(
            pipeline,
            dit_modules=dit_modules,
            sources=sources,
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            online_quantization=online_quantization,
            use_pipeline_key_remap=use_pipeline_key_remap,
            has_distilled_lora=has_distilled_lora,
        )
        return dit_only

    return HostWeightPlanResult(
        HostWeightPlan(
            backing_kind="checkpoint_mmap",
            bindings=bindings,
            planned_source_prefixes=planned_source_prefixes,
        )
    )


def _component_remaps(
    encoder_modules: Sequence[tuple[str, nn.Module]],
) -> dict[str, SourceRemap]:
    """Map each encoder component's checkpoint names onto its runtime names."""
    remaps: dict[str, SourceRemap] = {}
    for component_path, module in encoder_modules:
        mapper = getattr(module, "map_checkpoint_weight_name", None)
        if not callable(mapper):
            raise _PlanIncompatibleError(
                f"text encoder {component_path!r} cannot declare disk backing without a checkpoint-name mapping"
            )
        remaps[component_path] = _prefixed_remap(component_path, mapper)
    return remaps


def _prefixed_remap(component_path: str, mapper: Callable[[str], tuple[str, str | int | None] | None]) -> SourceRemap:
    prefix_length = len(component_path) + 1

    def remap(source_name: str) -> str | None:
        mapped = mapper(source_name[prefix_length:])
        if mapped is None:
            return None
        runtime_name, _shard_id = mapped
        return f"{component_path}.{runtime_name}"

    return remap


def _order_fused_entries(
    runtime_name: str,
    entries: list[tuple[str, str]],
    fused_sources: tuple[str, ...],
) -> list[tuple[str, str]]:
    """Order a runtime tensor's checkpoint entries the way its loader fuses them."""
    ordered: list[tuple[str, str]] = []
    for source in fused_sources:
        matches = [entry for entry in entries if f".{source}." in entry[0]]
        if len(matches) != 1:
            raise _PlanIncompatibleError(
                f"fused runtime tensor {runtime_name!r} needs exactly one checkpoint entry for {source!r}, "
                f"found {len(matches)}"
            )
        ordered.append(matches[0])
    if len(ordered) != len(entries):
        raise _PlanIncompatibleError(
            f"fused runtime tensor {runtime_name!r} maps {len(entries)} checkpoint entries, "
            f"but its fusion policy declares {len(fused_sources)}"
        )
    return ordered


def build_checkpoint_mmap_plan(
    pipeline: nn.Module,
    *,
    dit_modules: Sequence[tuple[str, nn.Module]],
    sources: Sequence[object],
    model_path: str | None,
    tensor_parallel_size: int,
    use_hsdp: bool,
    online_quantization: bool,
    has_distilled_lora: bool = False,
    encoder_modules: Sequence[tuple[str, nn.Module]] = (),
) -> HostWeightPlanResult:
    """Build the DLO mmap plan without changing its HSDP eligibility."""
    if use_hsdp:
        return HostWeightPlanResult(None, "HSDP requires the ordinary loader")
    return build_checkpoint_binding_plan(
        pipeline,
        dit_modules=dit_modules,
        sources=sources,
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        online_quantization=online_quantization,
        use_pipeline_key_remap=False,
        has_distilled_lora=has_distilled_lora,
        encoder_modules=encoder_modules,
    )


__all__ = [
    "FusedTensorBinding",
    "HostWeightPlan",
    "HostWeightPlanResult",
    "TensorBinding",
    "build_checkpoint_binding_plan",
    "build_checkpoint_mmap_plan",
    "has_online_quantization",
]
