"""Lance-specific model resolution helpers.

These helpers intentionally live outside ``models/lance`` to avoid importing
that package's heavy pipeline modules from shared config code.
"""

from __future__ import annotations

import os

from vllm_omni.diffusion.model_resolvers.types import ModelConfigLike


def looks_like_lance_subfolder(model: str | None) -> bool:
    """Return True when ``model`` points at a Lance per-component subfolder."""
    if not model:
        return False
    base = os.path.basename(str(model).rstrip("/"))
    return base in {"Lance_3B", "Lance_3B_Video"}


def resolve_lance_model_class_name(
    model: str | None,
    cfg: ModelConfigLike | None,
) -> str | None:
    """Resolve Lance to its pipeline class when config/path markers match.

    ``cfg`` is expected to be a model config view containing optional fields
    such as ``model_type``, ``architectures``, or ``model_name``.
    """
    if cfg is None:
        return "LancePipeline" if looks_like_lance_subfolder(model) else None

    model_type = cfg.get("model_type")
    architectures = cfg.get("architectures") or []
    if (
        model_type == "lance"
        or "LancePipeline" in architectures
        or cfg.get("model_name") == "Lance"
        or looks_like_lance_subfolder(model)
    ):
        return "LancePipeline"
    return None
