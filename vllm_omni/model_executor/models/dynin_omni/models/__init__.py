"""Public exports for model components used by Dynin-Omni."""

from __future__ import annotations

from .image.modeling_magvitv2 import LFQuantizer, MAGVITv2, VQGANDecoder, VQGANEncoder
from .backbone.modeling_dynin_omni import DyninOmniConfig, DyninOmniModelLM, VideoTokenMerger
from .backbone.sampling import get_mask_schedule

# Backward compatibility for legacy class names used in prior MMaDA scripts.
MMadaModelLM = DyninOmniModelLM
MMadaConfig = DyninOmniConfig

__all__ = [
    "VQGANEncoder",
    "VQGANDecoder",
    "LFQuantizer",
    "MAGVITv2",
    "DyninOmniModelLM",
    "DyninOmniConfig",
    "MMadaModelLM",
    "MMadaConfig",
    "VideoTokenMerger",
    "get_mask_schedule",
]
