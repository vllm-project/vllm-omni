"""Vendored TADA codec/diffusion components (from MIT-licensed hume-tada).

Re-exported so the TADA integration imports from here instead of taking a runtime
dependency on the external ``hume-tada`` package. Only the codec *weights* are
still fetched from ``HumeAI/tada-codec``.
"""

from .decoder import Decoder, DecoderConfig
from .vibevoice import VibeVoiceDiffusionHead, VibeVoiceDiffusionHeadConfig

__all__ = [
    "Decoder",
    "DecoderConfig",
    "VibeVoiceDiffusionHead",
    "VibeVoiceDiffusionHeadConfig",
]
