"""Public exports for model components used by training/inference scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .dynin_omni import DyninOmniForConditionalGeneration

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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DyninOmniForConditionalGeneration",
    "DyninOmniToken2Audio",
    "DyninOmniToken2Image",
    "DyninOmniToken2Text",
]
