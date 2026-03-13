from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .dynin_omni import DyninOmniForConditionalGeneration
from .dynin_omni_common import (
    get_dynin_magvit_attr,
    get_dynin_modeling_attr,
    get_dynin_sampling_attr,
)

if TYPE_CHECKING:
    from .dynin_omni_token2audio import DyninOmniToken2Audio
    from .dynin_omni_token2image import DyninOmniToken2Image
    from .dynin_omni_token2text import DyninOmniToken2Text


_STAGE_EXPORTS = {
    "DyninOmniToken2Audio": (".dynin_omni_token2audio", "DyninOmniToken2Audio"),
    "DyninOmniToken2Image": (".dynin_omni_token2image", "DyninOmniToken2Image"),
    "DyninOmniToken2Text": (".dynin_omni_token2text", "DyninOmniToken2Text"),
}

_MODELING_EXPORTS = {"DyninOmniConfig", "DyninOmniModelLM", "VideoTokenMerger"}
_MAGVIT_EXPORTS = {"VQGANEncoder", "VQGANDecoder", "LFQuantizer", "MAGVITv2"}


def __getattr__(name: str) -> Any:
    if name in _STAGE_EXPORTS:
        module_name, attr_name = _STAGE_EXPORTS[name]
        module = __import__(module_name, globals(), locals(), [attr_name], 1)
        return getattr(module, attr_name)

    if name in _MODELING_EXPORTS:
        return get_dynin_modeling_attr(name)

    if name in _MAGVIT_EXPORTS:
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
