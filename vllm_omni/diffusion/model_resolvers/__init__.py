"""Helpers for resolving model-specific diffusion pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vllm_omni.diffusion.model_resolvers.dreamzero import (
    is_dreamzero_model_family,
    resolve_dreamzero_model_class_name,
)
from vllm_omni.diffusion.model_resolvers.lance import resolve_lance_model_class_name
from vllm_omni.diffusion.model_resolvers.standard import resolve_standard_model_class_name


@dataclass(frozen=True)
class ModelClassResolution:
    """Resolution result for model-class detection.

    `handled=True` means a resolver claimed the model family, even if it could
    not resolve a concrete pipeline class. This lets callers preserve
    family-specific failure behavior without leaking model-specific checks back
    into shared config code.
    """

    handled: bool
    model_class_name: str | None = None


def resolve_model_class_resolution(
    model: str | None,
    cfg: Mapping[str, Any] | None,
) -> ModelClassResolution:
    lance_model_class = resolve_lance_model_class_name(model, cfg)
    if lance_model_class is not None:
        return ModelClassResolution(handled=True, model_class_name=lance_model_class)

    dreamzero_model_class = resolve_dreamzero_model_class_name(model, cfg)
    if dreamzero_model_class is not None:
        return ModelClassResolution(handled=True, model_class_name=dreamzero_model_class)
    if is_dreamzero_model_family(cfg):
        return ModelClassResolution(handled=True)

    standard_model_class = resolve_standard_model_class_name(model, cfg)
    if standard_model_class is not None:
        return ModelClassResolution(handled=True, model_class_name=standard_model_class)

    return ModelClassResolution(handled=False)


__all__ = [
    "ModelClassResolution",
    "resolve_model_class_resolution",
]
