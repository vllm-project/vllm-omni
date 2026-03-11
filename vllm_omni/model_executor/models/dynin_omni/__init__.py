"""Public exports for model components used by training/inference scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .dynin_omni import DyninOmniForConditionalGeneration
from .dynin_omni_common import get_dynin_magvit_attr, get_dynin_modeling_attr, get_dynin_sampling_attr

if TYPE_CHECKING:
    from .dynin_omni_token2audio import DyninOmniToken2Audio
    from .dynin_omni_token2image import DyninOmniToken2Image
    from .dynin_omni_token2text import DyninOmniToken2Text


def __getattr__(name: str) -> Any:
    if name == "DyninOmniToken2Audio":
        from .dynin_omni_token2audio import DyninOmniToken2Audio

        return DyninOmniToken2Audio
    if name == "DyninOmniToken2Image":
        from .dynin_omni_token2image import DyninOmniToken2Image

        return DyninOmniToken2Image
    if name == "DyninOmniToken2Text":
        from .dynin_omni_token2text import DyninOmniToken2Text

        return DyninOmniToken2Text
    if name in ("DyninOmniConfig", "DyninOmniModelLM", "VideoTokenMerger"):
        return get_dynin_modeling_attr(name)
    if name in ("VQGANEncoder", "VQGANDecoder", "LFQuantizer", "MAGVITv2"):
        return get_dynin_magvit_attr(name)
    if name == "get_mask_schedule":
        return get_dynin_sampling_attr("get_mask_schedule")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DyninOmniForConditionalGeneration",
    "DyninOmniToken2Audio",
    "DyninOmniToken2Image",
    "DyninOmniToken2Text",
    "DyninOmniConfig",
    "DyninOmniModelLM",
    "VideoTokenMerger",
    "VQGANEncoder",
    "VQGANDecoder",
    "LFQuantizer",
    "MAGVITv2",
    "get_mask_schedule",
]
