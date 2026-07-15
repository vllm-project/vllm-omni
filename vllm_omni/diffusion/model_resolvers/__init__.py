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
        preserve_explicit: When True, callers should keep an already-specified
            explicit ``model_class_name`` instead of overwriting it with the
            inferred class for this family.
    """

    handled: bool
    model_class_name: str | None = None
    preserve_explicit: bool = False


def resolve_model_class(
    model: str | None,
    cfg: ModelConfigLike | None,
    *,
    for_enrich: bool = False,
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
        for_enrich: When True, enable enrich-only families that historically
            existed only in ``OmniDiffusionConfig.enrich_config()`` and were
            not part of ``resolve_model_class_name()``.

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
        return ModelClassResolution(
            handled=True,
            model_class_name=standard_model_class,
            preserve_explicit=standard_model_class in {"NextStep11Pipeline", "WanS2VPipeline"},
        )

    # Gr00tN1d7 is enrich-only by design. Keep it out of
    # resolve_model_class_name() to preserve historical client-side behavior.
    if for_enrich and cfg is not None:
        model_type = cfg.get("model_type")
        architectures = cfg.get("architectures") or []
        if model_type == "Gr00tN1d7" or "Gr00tN1d7" in architectures:
            return ModelClassResolution(
                handled=True,
                model_class_name="Gr00tN1d7Pipeline",
            )

    return ModelClassResolution(handled=False)


__all__ = [
    "ModelConfigLike",
    "ModelClassResolution",
    "resolve_model_class",
]
