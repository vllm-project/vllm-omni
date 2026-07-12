"""Helpers for resolving model-specific diffusion pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.diffusion.model_resolvers.dreamzero import (
    is_dreamzero_model_family,
    resolve_dreamzero_model_class_name,
)
from vllm_omni.diffusion.model_resolvers.lance import resolve_lance_model_class_name
from vllm_omni.diffusion.model_resolvers.standard import resolve_standard_model_class_name
from vllm_omni.diffusion.model_resolvers.types import ModelConfigLike


@dataclass(frozen=True)
class ModelClassResolution:
    """Resolution result for model-class detection.

    Attributes:
        handled: True when a resolver claimed the model family. Callers should
            stop generic fallback when this is set, even if
            ``model_class_name`` is ``None``.
        model_class_name: Resolved pipeline class name for the claimed family,
            or ``None`` when the family was recognized but could not be mapped
            to a concrete diffusion pipeline.
    """

    handled: bool
    model_class_name: str | None = None


def resolve_model_class_resolution(
    model: str | None,
    cfg: ModelConfigLike | None,
) -> ModelClassResolution:
    """Resolve diffusion model family in precedence order.

    Resolver precedence is:
    1. Lance-specific rules, including subfolder detection.
    2. DreamZero-specific rules for ``model_type == "vla"``.
    3. Standard model-family rules keyed by ``model_type`` / ``architectures``.

    Args:
        model: User-provided model path or HF repo id.
        cfg: Read-only model config view, typically from
            ``get_hf_file_to_dict("config.json", model)``. Resolvers currently
            rely on fields such as ``model_type``, ``architectures``, and
            ``model_name`` when present.

    Returns:
        A ``ModelClassResolution`` describing whether a resolver claimed the
        family and, if so, which concrete pipeline class should be used.
    """
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
    "ModelConfigLike",
    "ModelClassResolution",
    "resolve_model_class_resolution",
]
