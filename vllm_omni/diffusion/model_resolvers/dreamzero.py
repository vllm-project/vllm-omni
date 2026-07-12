"""DreamZero-specific model resolution helpers."""

from __future__ import annotations

from vllm_omni.diffusion.model_resolvers.types import ModelConfigLike


def is_dreamzero_model_family(cfg: ModelConfigLike | None) -> bool:
    if cfg is None:
        return False
    return cfg.get("model_type") == "vla"


def resolve_dreamzero_model_class_name(
    model: str | None,
    cfg: ModelConfigLike | None,
) -> str | None:
    """Resolve DreamZero model class from VLA-family configs.

    ``cfg`` is expected to include ``model_type`` when available.
    """
    if not is_dreamzero_model_family(cfg):
        return None

    from vllm_omni.diffusion.utils.hf_utils import _looks_like_dreamzero

    return "DreamZeroPipeline" if _looks_like_dreamzero(model) else None
