# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loader-owned host-weight plans shared with diffusion offload backends."""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from safetensors import safe_open
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    get_direct_mmap_adapter,
)

logger = init_logger(__name__)

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TensorBinding:
    """One runtime tensor backed by one safetensors entry."""

    checkpoint_key: str
    file_path: str
    transform: TensorTransform | None = None


@dataclass(frozen=True)
class HostWeightPlan:
    """Complete, prevalidated host backing consumed by an offload backend."""

    backing_kind: str
    bindings: dict[str, TensorBinding]
    planned_source_prefixes: frozenset[str] = frozenset()


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
    model_to_ckpt: dict[str, tuple[str, str]],
    model_name: str,
    checkpoint_key: str,
    file_path: str,
) -> None:
    existing = model_to_ckpt.get(model_name)
    candidate = (checkpoint_key, file_path)
    if existing is not None and existing != candidate:
        raise _PlanIncompatibleError(
            f"multiple checkpoint entries map to runtime tensor {model_name!r}: {existing!r} and {candidate!r}"
        )
    model_to_ckpt[model_name] = candidate


def _add_safetensors_source(
    checkpoint_dir: str,
    prefix: str,
    remap_fn: Callable[[str], str | None] | None,
    model_to_ckpt: dict[str, tuple[str, str]],
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


def _build_source_map(
    *,
    sources: Sequence[object],
    model_path: str | None,
    remap_fn: Callable[[str], str | None] | None,
) -> dict[str, tuple[str, str]]:
    model_to_ckpt: dict[str, tuple[str, str]] = {}
    indexed_keys = 0

    for source in sources:
        indexed_keys += _add_safetensors_source(
            _resolve_source_root(source),
            getattr(source, "prefix", ""),
            remap_fn,
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
                remap_fn,
                model_to_ckpt,
            )

    if not model_to_ckpt:
        raise _PlanIncompatibleError("no compatible safetensors entries were found in the loader's model sources")
    logger.info(
        "Indexed %d runtime tensor names from %d checkpoint keys for DLO preflight",
        len(model_to_ckpt),
        indexed_keys,
    )
    return model_to_ckpt


def _is_persistent_buffer(module: nn.Module, local_name: str) -> bool:
    parts = local_name.split(".")
    owner = module.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else module
    return parts[-1] not in owner._non_persistent_buffers_set


def _collect_required_dit_tensors(
    dit_modules: Sequence[tuple[str, nn.Module]],
) -> dict[str, torch.Tensor]:
    required: dict[str, torch.Tensor] = {}
    for dit_name, dit_module in dit_modules:
        for local_name, tensor in dit_module.named_parameters():
            required[f"{dit_name}.{local_name}"] = tensor
        for local_name, tensor in dit_module.named_buffers():
            if _is_persistent_buffer(dit_module, local_name):
                required[f"{dit_name}.{local_name}"] = tensor
    return required


def _planned_source_prefixes(
    sources: Sequence[object],
    dit_modules: Sequence[tuple[str, nn.Module]],
) -> frozenset[str]:
    """Identify component sources that the DiT-only plan owns completely."""
    if not sources:
        return frozenset()

    expected = frozenset(f"{dit_name}." for dit_name, _ in dit_modules)
    available = {getattr(source, "prefix", "") for source in sources}
    missing = sorted(expected - available)
    if missing:
        raise _PlanIncompatibleError(
            f"direct mmap requires a dedicated component weight source for each DiT; missing prefixes: {missing}"
        )
    return expected


def _validate_source_metadata(
    required: dict[str, torch.Tensor],
    bindings: dict[str, TensorBinding],
) -> None:
    by_file: dict[str, list[tuple[str, torch.Tensor, TensorBinding]]] = {}
    for runtime_name, target in required.items():
        by_file.setdefault(bindings[runtime_name].file_path, []).append((runtime_name, target, bindings[runtime_name]))

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
                if source_shape != tuple(target.shape):
                    raise _PlanIncompatibleError(
                        f"shape mismatch for {runtime_name!r}: checkpoint={source_shape}, runtime={tuple(target.shape)}"
                    )
                if source_dtype is None or source_dtype != target.dtype:
                    raise _PlanIncompatibleError(
                        f"dtype mismatch for {runtime_name!r}: checkpoint={source.get_dtype()}, runtime={target.dtype}"
                    )


def build_checkpoint_mmap_plan(
    pipeline: nn.Module,
    *,
    dit_modules: Sequence[tuple[str, nn.Module]],
    sources: Sequence[object],
    model_path: str | None,
    tensor_parallel_size: int,
    use_hsdp: bool,
    online_quantization: bool,
) -> HostWeightPlanResult:
    """Build a complete direct-checkpoint plan or return a fallback reason."""
    if tensor_parallel_size != 1:
        return HostWeightPlanResult(None, f"TP={tensor_parallel_size} requires the ordinary loader")
    if use_hsdp:
        return HostWeightPlanResult(None, "HSDP requires the ordinary loader")
    if online_quantization:
        return HostWeightPlanResult(None, "online quantization requires the ordinary loader")

    remap_fn = getattr(type(pipeline), "_remap_ckpt_key", None)
    if not callable(remap_fn):
        remap_fn = None
    adapter = get_direct_mmap_adapter(pipeline)

    try:
        planned_source_prefixes = _planned_source_prefixes(sources, dit_modules)
        planned_sources = tuple(
            source for source in sources if getattr(source, "prefix", "") in planned_source_prefixes
        )
        model_to_ckpt = _build_source_map(
            sources=planned_sources,
            model_path=model_path,
            remap_fn=remap_fn,
        )
        required = _collect_required_dit_tensors(dit_modules)
        if not required:
            raise _PlanIncompatibleError("no DiT parameters or persistent buffers were discovered")

        missing = sorted(set(required) - set(model_to_ckpt))
        if missing:
            raise _PlanIncompatibleError(
                f"{len(missing)} required DiT tensors have no checkpoint binding (first 5: {missing[:5]})"
            )

        bindings: dict[str, TensorBinding] = {}
        for runtime_name, target in required.items():
            policy = adapter.policy_for(runtime_name, target) if adapter is not None else None
            has_custom_loader = callable(getattr(target, "weight_loader", None))
            if has_custom_loader and not (policy and policy.allow_custom_loader):
                raise _PlanIncompatibleError(
                    f"{runtime_name!r} requires a custom weight_loader with no direct-mmap adapter"
                )
            checkpoint_key, file_path = model_to_ckpt[runtime_name]
            bindings[runtime_name] = TensorBinding(
                checkpoint_key=checkpoint_key,
                file_path=file_path,
                transform=policy.transform if policy is not None else None,
            )

        _validate_source_metadata(required, bindings)
    except (OSError, ValueError, _PlanIncompatibleError) as exc:
        return HostWeightPlanResult(None, str(exc))

    return HostWeightPlanResult(
        HostWeightPlan(
            backing_kind="checkpoint_mmap",
            bindings=bindings,
            planned_source_prefixes=planned_source_prefixes,
        )
    )


__all__ = [
    "HostWeightPlan",
    "HostWeightPlanResult",
    "TensorBinding",
    "build_checkpoint_mmap_plan",
    "has_online_quantization",
]
