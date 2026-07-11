"""DreamZero-specific model resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def is_dreamzero_model_family(cfg: Mapping[str, Any] | None) -> bool:
    if cfg is None:
        return False
    return cfg.get("model_type") == "vla"


def resolve_dreamzero_model_class_name(
    model: str | None,
    cfg: Mapping[str, Any] | None,
) -> str | None:
    if not is_dreamzero_model_family(cfg):
        return None

    from vllm_omni.diffusion.utils.hf_utils import _looks_like_dreamzero

    return "DreamZeroPipeline" if _looks_like_dreamzero(model) else None
